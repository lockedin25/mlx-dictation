import AppKit
import AVFoundation

@available(macOS 15.0, *)
@MainActor
final class MeetingController: NSObject {
    private struct GoogleMeetSettings: Decodable {
        var enabled: Bool?
        var start_debounce_seconds: Double?
        var end_grace_seconds: Double?
    }

    private struct DiscordSettings: Decodable {
        var enabled: Bool?
    }

    private struct Settings: Decodable {
        var guild_id: String?
        var user_id: String?
        var peer_id: String?
        var channel_id: String?
        var peer_name: String?
        var vault_raw_dir: String?
        var summary_model: String?
        var auto_record: Bool?
        var google_meet: GoogleMeetSettings?
        var discord: DiscordSettings?
    }

    private let root: URL
    private let files = FileManager.default
    private let data = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/Fluid Dictation/meetings")
    private let config = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".config/fluid-dictation")
    private let state = NSMenuItem(title: "Meetings", action: nil, keyEquivalent: "")
    private let record = NSMenuItem(title: "Record meeting…", action: nil, keyEquivalent: "")
    private var settings = Settings()
    private var listener: Process?
    private var meetDetector: ChromeMeetDetector?
    private var pipeline: [URL: Process] = [:]
    private var capture: MeetingCapture?
    private var session: URL?
    private var activeSource: MeetingCaptureSource?
    private var activeMeetIdentity: String?
    private var transitioning = false
    private var targetPresent = false
    private var manuallyStoppedDiscord = false
    private var manuallyStoppedMeetIdentity: String?
    private var autoSession = false
    private var restart: Task<Void, Never>?
    private var quitting = false
    var recordingChanged: ((Bool) -> Void)?
    var standaloneMicInUse: (() -> Bool)?
    var beforeCaptureStops: (() -> Void)?

    func beginDictationClip() -> (() -> Data)? { capture?.beginDictationClip() }

    func retryJoinAfterDictation() {
        guard targetPresent, !manuallyStoppedDiscord, !active else { return }
        Task { await start(source: .discord, automatic: true) }
    }

    var active: Bool { session != nil || transitioning }

    init(root: URL, menu: NSMenu) {
        self.root = root
        super.init()
        menu.addItem(.separator())
        menu.addItem(state)
        record.target = self
        record.action = #selector(toggle)
        menu.addItem(record)
        menu.addItem(item("Open meeting notes", #selector(openNotes)))
        menu.addItem(item("Retry unfinished meeting notes", #selector(retry)))
        menu.addItem(item("Reload meeting settings", #selector(reload)))
        reload()
        recover(interrupted: true)
    }

    private func item(_ title: String, _ action: Selector) -> NSMenuItem {
        let result = NSMenuItem(title: title, action: action, keyEquivalent: "")
        result.target = self
        return result
    }

    private var notesIndex: URL {
        files.homeDirectoryForCurrentUser.appendingPathComponent("Documents/Meeting Notes/index.md")
    }

    @objc private func openNotes() {
        NSWorkspace.shared.open(files.fileExists(atPath: notesIndex.path) ? notesIndex : data)
    }

    @objc private func reload() {
        restart?.cancel()
        listener?.terminate()
        listener = nil
        meetDetector?.stop()
        meetDetector = nil
        targetPresent = false
        manuallyStoppedDiscord = false
        manuallyStoppedMeetIdentity = nil
        if autoSession { Task { await finish() } }
        let path = config.appendingPathComponent("meetings.json")
        do {
            let content = try Data(contentsOf: path)
            settings = try JSONDecoder().decode(Settings.self, from: content)
        } catch {
            settings = Settings()
            state.title = files.fileExists(atPath: path.path) ? "Meeting settings are invalid" : "Meetings · manual"
            return
        }
        startDetectors()
    }

    private func startDetectors() {
        let meetEnabled = (settings.auto_record == true) && (settings.google_meet?.enabled != false)
        let discordEnabled = (settings.auto_record == true) && (settings.discord?.enabled != false)
            && settings.guild_id != nil && settings.user_id != nil && settings.channel_id != nil

        if meetEnabled {
            startMeetDetector()
        }
        if discordEnabled {
            startDiscordListener()
        }

        if meetEnabled && discordEnabled {
            state.title = "Meetings · auto (Meet & Discord)"
        } else if meetEnabled {
            state.title = "Meetings · auto (Google Meet)"
        } else if discordEnabled {
            state.title = "Meetings · auto (Discord)"
        } else {
            state.title = "Meetings · manual"
        }
    }

    private func startMeetDetector() {
        let debounce = settings.google_meet?.start_debounce_seconds ?? 5.0
        let grace = settings.google_meet?.end_grace_seconds ?? 12.0
        let detector = ChromeMeetDetector(
            pollInterval: 2.0,
            startDebounceSeconds: debounce,
            endGraceSeconds: grace
        ) { [weak self] event in
            self?.chromeMeetEvent(event)
        }
        meetDetector = detector
        detector.start()
    }

    private func chromeMeetEvent(_ event: ChromeMeetEvent) {
        guard !quitting else { return }
        switch event {
        case let .joined(match):
            if manuallyStoppedMeetIdentity == match.identityKey {
                return
            }
            if !active {
                Task { await start(source: .chromeMeet, automatic: true, meetMatch: match) }
            }
        case .left:
            manuallyStoppedMeetIdentity = nil
            if autoSession && activeSource == .chromeMeet {
                Task { await finish() }
            }
        }
    }

    private func startDiscordListener() {
        guard let guild = settings.guild_id, let user = settings.user_id,
              let channel = settings.channel_id else {
            return
        }
        var env = ProcessInfo.processInfo.environment
        let tokenFile = config.appendingPathComponent("discord-bot-token")
        if let token = try? String(contentsOf: tokenFile, encoding: .utf8) {
            env["DISCORD_TOKEN"] = token.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard let token = env["DISCORD_TOKEN"], !token.isEmpty else {
            state.title = "Discord bot token required"
            return
        }
        env["DISCORD_GUILD_ID"] = guild
        env["DISCORD_USER_ID"] = user
        env["DISCORD_PEER_ID"] = settings.peer_id
        env["DISCORD_CHANNEL_ID"] = channel
        env["DISCORD_VOICE_LEAVE_GRACE_SECONDS"] = "0"
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = [root.appendingPathComponent("hotkey/discord_voice.py").path]
        process.environment = env
        let pipe = Pipe()
        process.standardOutput = pipe
        process.terminationHandler = { [weak self] terminated in
            Task { @MainActor in
                guard let self, self.listener === terminated else { return }
                self.listener = nil
                self.targetPresent = false
                if self.autoSession && self.activeSource == .discord { await self.finish() }
                self.restart = Task {
                    try? await Task.sleep(nanoseconds: 60_000_000_000)
                    if !Task.isCancelled { self.startDiscordListener() }
                }
            }
        }
        do {
            try process.run()
            listener = process
            DispatchQueue.global(qos: .utility).async { [weak self] in
                let handle = pipe.fileHandleForReading
                var pending = Data()
                while true {
                    let bytes = handle.availableData
                    if bytes.isEmpty { break }
                    pending.append(bytes)
                    while let newline = pending.firstIndex(of: 10) {
                        let line = Data(pending[..<newline])
                        pending.removeSubrange(...newline)
                        guard let object = (try? JSONSerialization.jsonObject(with: line)) as? [String: Any],
                              let event = object["event"] as? String else { continue }
                        DispatchQueue.main.async { [weak self] in
                            self?.discordVoiceEvent(event, source: process)
                        }
                    }
                }
            }
        } catch {
            state.title = "Could not start Discord call detector"
        }
    }

    private func discordVoiceEvent(_ event: String, source: Process) {
        guard listener === source, !quitting else { return }
        if event == "join" {
            targetPresent = true
            if !manuallyStoppedDiscord && !active {
                Task { await start(source: .discord, automatic: true) }
            }
        } else if event == "leave" {
            targetPresent = false
            manuallyStoppedDiscord = false
            if autoSession && activeSource == .discord {
                Task { await finish() }
            }
        }
    }

    @objc func toggle() {
        guard !transitioning else { return }
        if session != nil {
            if activeSource == .discord {
                manuallyStoppedDiscord = targetPresent
            } else if activeSource == .chromeMeet {
                manuallyStoppedMeetIdentity = activeMeetIdentity
            }
            Task { await finish() }
            return
        }
        // Manual trigger: prefer Google Chrome if running with audio, otherwise Discord.
        let chromeRunning = NSWorkspace.shared.runningApplications.contains {
            $0.bundleIdentifier == "com.google.Chrome"
        }
        let discordRunning = NSWorkspace.shared.runningApplications.contains {
            ($0.bundleIdentifier ?? "").hasPrefix("com.hnc.Discord")
        }
        let chosen: MeetingCaptureSource = (chromeRunning && !discordRunning) ? .chromeMeet : (discordRunning ? .discord : .chromeMeet)
        Task { await start(source: chosen, automatic: false) }
    }

    private func start(
        source: MeetingCaptureSource,
        automatic: Bool,
        meetMatch: MeetTabMatch? = nil
    ) async {
        guard !active, !quitting else { return }
        if automatic && source == .discord && !targetPresent { return }
        if standaloneMicInUse?() == true {
            state.title = "Finish Fn dictation to start meeting recording"
            return
        }
        transitioning = true
        state.title = "Starting \(source.displayName) recording…"
        autoSession = automatic
        activeSource = source
        activeMeetIdentity = meetMatch?.identityKey
        defer { transitioning = false }
        let directory = data.appendingPathComponent(UUID().uuidString.lowercased())
        do {
            guard await AVCaptureDevice.requestAccess(for: .audio) else {
                reportStartFailure("Allow Fluid Dictation under Privacy & Security → Microphone.", automatic: automatic)
                resetActiveMetadata()
                return
            }
            if automatic && source == .discord && !targetPresent {
                resetActiveMetadata()
                return
            }
            if standaloneMicInUse?() == true {
                state.title = "Finish Fn dictation to start meeting recording"
                resetActiveMetadata()
                return
            }
            try files.createDirectory(at: directory, withIntermediateDirectories: true,
                                      attributes: [.posixPermissions: 0o700])
            var manifest: [String: Any] = [
                "session_id": directory.lastPathComponent,
                "started_at": ISO8601DateFormatter().string(from: Date()),
                "source": source == .chromeMeet ? "google-meet" : "discord",
                "trigger": automatic ? (source == .chromeMeet ? "chrome-meet-auto" : "discord-voice-state") : "manual",
                "mic_speaker": "Shubham Korde",
                "participants": ["Shubham Korde"],
            ]
            if let match = meetMatch {
                manifest["topic"] = match.title
                if let code = match.meetCode {
                    manifest["meet_code"] = code
                }
            } else if source == .discord && automatic, let peerName = settings.peer_name {
                manifest["participants"] = ["Shubham Korde", peerName]
            }
            try JSONSerialization.data(withJSONObject: manifest).write(
                to: directory.appendingPathComponent("manifest.json"), options: .atomic)
            var captureSettings = MeetingCapture.Configuration()
            captureSettings.chunkMilliseconds = 15_000
            let engine = MeetingCapture(configuration: captureSettings) { [weak self] error in
                Task { @MainActor in
                    guard let self, self.session == directory else { return }
                    self.state.title = "Recording issue: \(error.localizedDescription)"
                    var metadata = (try? JSONSerialization.jsonObject(
                        with: Data(contentsOf: directory.appendingPathComponent("manifest.json")))) as? [String: Any] ?? [:]
                    var warnings = metadata["capture_warnings"] as? [String] ?? []
                    warnings.append(error.localizedDescription)
                    metadata["capture_warnings"] = warnings
                    try? JSONSerialization.data(withJSONObject: metadata).write(
                        to: directory.appendingPathComponent("manifest.json"), options: .atomic)
                }
            }
            try await engine.start(session: directory, source: source)
            capture = engine
            session = directory
            if automatic && source == .discord && !targetPresent {
                await finish()
                return
            }
            state.title = source == .chromeMeet ? "Recording Google Meet" : "Recording Discord meeting"
            record.title = "Stop meeting recording"
            recordingChanged?(true)
        } catch {
            resetActiveMetadata()
            try? files.removeItem(at: directory)
            reportStartFailure(error.localizedDescription, automatic: automatic)
        }
    }

    private func resetActiveMetadata() {
        autoSession = false
        activeSource = nil
        activeMeetIdentity = nil
    }

    private func reportStartFailure(_ reason: String, automatic: Bool) {
        state.title = "Recording failed: \(reason)"
        guard !automatic else { return }
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = "Meeting recording didn’t start"
        alert.informativeText = reason
        alert.runModal()
    }

    private func finish() async {
        guard let directory = session, let engine = capture else { return }
        transitioning = true
        beforeCaptureStops?()
        capture = nil
        await engine.stop()
        session = nil
        resetActiveMetadata()
        transitioning = false
        record.title = "Record meeting…"
        recordingChanged?(false)
        do {
            try Data().write(to: directory.appendingPathComponent("capture.done"), options: .atomic)
            state.title = "Processing meeting (diarize & transcribe)…"
            startPipeline(directory)
        } catch {
            state.title = "Could not finalize the recording"
        }
        if targetPresent && !manuallyStoppedDiscord && !quitting {
            await start(source: .discord, automatic: true)
        }
    }

    func shutdown() async {
        quitting = true
        restart?.cancel()
        listener?.terminate()
        listener = nil
        meetDetector?.stop()
        meetDetector = nil
        while transitioning {
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        await finish()
    }

    @objc private func retry() { recover(interrupted: false) }

    private func recover(interrupted: Bool) {
        guard let directories = try? files.contentsOfDirectory(at: data, includingPropertiesForKeys: nil,
                                                                 options: [.skipsHiddenFiles]) else { return }
        for directory in directories where UUID(uuidString: directory.lastPathComponent) != nil {
            guard directory != session,
                  !files.fileExists(atPath: directory.appendingPathComponent("pipeline.done").path) else { continue }
            let done = directory.appendingPathComponent("capture.done")
            if interrupted && !files.fileExists(atPath: done.path) {
                try? Data().write(to: done, options: .atomic)
            }
            if files.fileExists(atPath: done.path) { startPipeline(directory) }
        }
    }

    private func startPipeline(_ directory: URL) {
        guard pipeline[directory] == nil else { return }
        let process = Process()
        process.executableURL = root.appendingPathComponent(".venv-rumble/bin/python")
        var args = [root.appendingPathComponent("hotkey/meeting_pipeline.py").path, directory.path,
                    "--notes-index", notesIndex.path]
        if let raw = settings.vault_raw_dir {
            args += ["--vault-raw-dir", (raw as NSString).expandingTildeInPath]
        }
        if let model = settings.summary_model { args += ["--model", model] }
        process.arguments = args
        process.currentDirectoryURL = root
        var env = ProcessInfo.processInfo.environment
        let homeBin = files.homeDirectoryForCurrentUser.appendingPathComponent(".local/bin").path
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:\(homeBin):/usr/bin:/bin:/usr/sbin:/sbin"
        env["PYTHONUNBUFFERED"] = "1"
        process.environment = env

        let logFile = files.homeDirectoryForCurrentUser.appendingPathComponent("Library/Logs/FluidDictation-meeting-pipeline.log")
        if !files.fileExists(atPath: logFile.path) { files.createFile(atPath: logFile.path, contents: nil) }
        if let handle = try? FileHandle(forWritingTo: logFile) {
            handle.seekToEndOfFile()
            process.standardError = handle
        }

        process.terminationHandler = { [weak self] ended in
            Task { @MainActor in
                guard let self else { return }
                self.pipeline[directory] = nil
                if ended.terminationStatus == 0 {
                    try? Data().write(to: directory.appendingPathComponent("pipeline.done"), options: .atomic)
                }
                if !self.active {
                    if ended.terminationStatus != 0 {
                        self.state.title = "Meeting notes pending · retry from menu"
                    } else if self.files.fileExists(atPath: directory.appendingPathComponent("notes.md").path) {
                        self.state.title = "Meeting notes ready"
                    } else {
                        self.state.title = "No speech detected in the recording"
                    }
                }
            }
        }
        do {
            try process.run()
            pipeline[directory] = process
        } catch {
            if !active { state.title = "Could not start meeting pipeline" }
        }
    }
}
