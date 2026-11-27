import AppKit
import AVFoundation
import CoreAudio
import Foundation
import os

// Records two time-aligned tracks into a session folder:
//   <session>/mic/<start-ms>.wav     microphone (AVAudioEngine input)
//   <session>/remote/<start-ms>.wav  target call app output (Core Audio process tap)
// Both tracks are 16 kHz mono 16-bit PCM. Every sample is placed on one session
// timeline derived from mach_absolute_time, which Core Audio host times and
// AVAudioTime host times share. Chunk boundaries are fixed multiples of the chunk
// length on that timeline, so a chunk name means the same instant in both folders.
// Chunks are written to a temporary name and renamed into place (atomic publish).

private let meetingCaptureLog = Logger(subsystem: "local.shubham.fluid-dictation", category: "MeetingCapture")

enum MeetingCaptureSource: String, Sendable {
    case discord
    case chromeMeet

    var displayName: String {
        switch self {
        case .discord: return "Discord"
        case .chromeMeet: return "Google Chrome"
        }
    }

    var bundleIDPrefixes: [String] {
        switch self {
        case .discord: return ["com.hnc.Discord"]
        case .chromeMeet: return ["com.google.Chrome"]
        }
    }

    var remoteDirectoryName: String { "remote" }
}

enum MeetingCaptureError: Error, LocalizedError, Sendable {
    case alreadyRunning
    case appNotRunning(String)
    case appAudioSessionNotFound(String)
    case microphonePermissionDenied
    case microphoneUnavailable(String)
    /// Process tap creation failed. The usual cause is that System Audio Recording
    /// permission was denied for this app, which macOS does not report separately.
    case systemAudioTapFailed(OSStatus, String)
    case coreAudio(step: String, status: OSStatus)
    case sessionDirectoryUnavailable(path: String, reason: String)
    case writeFailed(path: String, reason: String)
    case conversionFailed(String)
    case microphoneStalled
    case systemAudioStalled(String)

    var errorDescription: String? {
        switch self {
        case .alreadyRunning:
            return "Meeting capture is already running."
        case let .appNotRunning(name):
            return "\(name) is not running. Start \(name) and join a call before recording."
        case let .appAudioSessionNotFound(name):
            return "\(name) is running but has no audio session yet. Join a call or play audio, then try again."
        case .microphonePermissionDenied:
            return "Microphone access is not granted for Fluid Dictation."
        case let .microphoneUnavailable(reason):
            return "Microphone unavailable: \(reason)"
        case let .systemAudioTapFailed(status, name):
            return "Could not tap \(name) audio (\(meetingCaptureFourCC(status))). Check System Audio Recording permission for Fluid Dictation."
        case let .coreAudio(step, status):
            return "Core Audio failed at \(step) (\(meetingCaptureFourCC(status)))."
        case let .sessionDirectoryUnavailable(path, reason):
            return "Cannot create session folder \(path): \(reason)"
        case let .writeFailed(path, reason):
            return "Failed to write \(path): \(reason). The chunk was dropped."
        case let .conversionFailed(reason):
            return "Audio conversion failed: \(reason)"
        case .microphoneStalled:
            return "Microphone stopped delivering audio. Resume speaking only after the microphone recovers."
        case let .systemAudioStalled(name):
            return "\(name) is playing audio but the recorder stopped receiving it. Reconnecting the audio track."
        }
    }
}

/// Records microphone and process-scoped remote audio into a session folder.
/// Runs on the main actor. Audio processing happens on a private queue.
/// `onFailure` is delivered on the main queue. Failures during recording do not
/// stop the other track; the failed track is padded with silence so timing holds.
@available(macOS 14.4, *)
@MainActor
final class MeetingCapture {
    struct Configuration: Sendable {
        /// Chunk length in milliseconds, clamped to 15_000...30_000.
        var chunkMilliseconds: Int = 30_000
    }

    private(set) var isRunning = false
    /// Most recent runtime failure reported during a recording, if any.
    private(set) var lastFailure: MeetingCaptureError?

    private let configuration: Configuration
    private let onFailure: (@Sendable (MeetingCaptureError) -> Void)?
    private var active: MeetingCaptureSession?
    private var starting = false

    init(configuration: Configuration = Configuration(), onFailure: (@Sendable (MeetingCaptureError) -> Void)? = nil) {
        self.configuration = configuration
        self.onFailure = onFailure
    }

    /// Starts recording into `session`, creating `mic/` and `remote/` inside it.
    func start(session: URL, source: MeetingCaptureSource = .discord) async throws {
        guard active == nil, !starting else { throw MeetingCaptureError.alreadyRunning }
        starting = true
        defer { starting = false }

        // No side effects before this point: fail fast if target app is absent.
        let processes = try MeetingCaptureProcessAudio.processObjects(
            prefixes: source.bundleIDPrefixes,
            displayName: source.displayName
        )
        let (micDirectory, remoteDirectory) = try prepareDirectories(session, remoteName: source.remoteDirectoryName)
        try await requestMicrophoneAccess()

        let chunkMilliseconds = min(max(configuration.chunkMilliseconds, 15_000), 30_000)
        let capture = try MeetingCaptureSession(
            source: source,
            micDirectory: micDirectory,
            remoteDirectory: remoteDirectory,
            chunkMilliseconds: chunkMilliseconds,
            report: { [weak self] error in
                Task { @MainActor in self?.deliverFailure(error) }
            }
        )
        try capture.begin(processes: processes)
        active = capture
        isRunning = true
    }

    /// Copies microphone samples for Fn dictation while the meeting owns the device.
    func beginDictationClip() -> (() -> Data)? {
        guard let capture = active else { return nil }
        capture.beginDictationClip()
        return { capture.endDictationClip() }
    }

    /// Stops recording. Returns after every closed chunk and the final partial chunk are on disk.
    func stop() async {
        guard let capture = active else { return }
        active = nil
        isRunning = false
        await capture.finish()
    }

    private func deliverFailure(_ error: MeetingCaptureError) {
        lastFailure = error
        onFailure?(error)
    }

    private func prepareDirectories(_ session: URL, remoteName: String) throws -> (URL, URL) {
        let mic = session.appendingPathComponent("mic", isDirectory: true)
        let remote = session.appendingPathComponent(remoteName, isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: mic, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: remote, withIntermediateDirectories: true)
        } catch {
            throw MeetingCaptureError.sessionDirectoryUnavailable(path: session.path, reason: error.localizedDescription)
        }
        return (mic, remote)
    }

    private func requestMicrophoneAccess() async throws {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            return
        case .notDetermined:
            guard await AVCaptureDevice.requestAccess(for: .audio) else {
                throw MeetingCaptureError.microphonePermissionDenied
            }
        default:
            throw MeetingCaptureError.microphonePermissionDenied
        }
    }
}

// MARK: - Session

/// Owns all capture resources for one recording. Mutable track state is touched only on `queue`.
@available(macOS 14.4, *)
private final class MeetingCaptureSession: @unchecked Sendable {
    private let queue = DispatchQueue(label: "local.shubham.fluid-dictation.meeting-capture", qos: .userInitiated)
    private let sampleRate = 16_000
    private let chunkSamples: Int
    /// Chunks close this long after their boundary so late callbacks still land in the right place.
    private let lagSamples: Int64 = 16_000
    private let stallNanoseconds: UInt64 = 5_000_000_000
    private let source: MeetingCaptureSource
    private let report: @Sendable (MeetingCaptureError) -> Void

    // Written on the main actor before sources start; read on `queue` afterwards.
    private var originHost: UInt64 = 0

    // Queue-confined state.
    private var mic: MeetingCaptureTrack!
    private var remote: MeetingCaptureTrack!
    private var accepting = true
    private var timer: DispatchSourceTimer?
    private var nextMicRecovery: UInt64 = 0
    private var micRecoveryAttempts = 0
    private var micStallReported = false
    private var remoteStallReported = false
    private var silenceNoted = false
    private var nextRemoteCheck: UInt64 = 0
    private var dictationSamples: [Int16]?

    // Main-actor-confined sources.
    private var tap: MeetingCaptureSystemTap?
    private var tappedProcesses: Set<AudioObjectID> = []
    private var engine: AVAudioEngine?
    private var configObserver: NSObjectProtocol?
    private var microphoneFormat: AVAudioFormat?
    private var sourcesRunning = false

    init(
        source: MeetingCaptureSource,
        micDirectory: URL,
        remoteDirectory: URL,
        chunkMilliseconds: Int,
        report: @escaping @Sendable (MeetingCaptureError) -> Void
    ) throws {
        self.source = source
        self.report = report
        self.chunkSamples = sampleRate * chunkMilliseconds / 1000
        guard let target = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: Double(sampleRate), channels: 1, interleaved: true) else {
            throw MeetingCaptureError.conversionFailed("cannot create 16 kHz mono target format")
        }
        let reporter: (MeetingCaptureError) -> Void = { [weak self] error in self?.report(error) }
        mic = MeetingCaptureTrack(directory: micDirectory, sampleRate: sampleRate, chunkSamples: chunkSamples, target: target, report: reporter)
        remote = MeetingCaptureTrack(directory: remoteDirectory, sampleRate: sampleRate, chunkSamples: chunkSamples, target: target, report: reporter)
    }

    /// Starts remote process tap, microphone engine and the roll timer. Cleans up on failure.
    @MainActor func begin(processes: [AudioObjectID]) throws {
        originHost = MeetingCaptureClock.now
        mic.lastArrival = originHost
        remote.lastArrival = originHost
        do {
            try startRemoteTap(processes: processes)
            try startMicrophone()
            sourcesRunning = true

            let source = DispatchSource.makeTimerSource(queue: queue)
            source.schedule(deadline: .now() + 1, repeating: 1, leeway: .milliseconds(50))
            source.setEventHandler { [self] in self.tick() }
            queue.sync { self.timer = source }
            source.resume()
        } catch {
            teardownSources()
            queue.async { self.accepting = false }
            throw error
        }
    }

    @MainActor private func startRemoteTap(processes: [AudioObjectID]) throws {
        let systemTap = MeetingCaptureSystemTap(sourceName: source.displayName)
        do {
            try systemTap.start(processes: processes) { [weak self] block in
                guard let self else { return }
                self.queue.async { self.ingest(block, into: self.remote) }
            }
        } catch {
            systemTap.stop()
            throw error
        }
        tap = systemTap
        tappedProcesses = Set(processes)
    }

    @MainActor private func recoverRemoteTapIfNeeded() {
        guard sourcesRunning,
              let processes = try? MeetingCaptureProcessAudio.processObjects(
                  prefixes: source.bundleIDPrefixes,
                  displayName: source.displayName
              ) else { return }

        let currentSet = Set(processes)
        let playing = processes.contains(where: MeetingCaptureProcessAudio.isPlaying)
        let newProcessPlaying = processes.contains {
            !tappedProcesses.contains($0) && MeetingCaptureProcessAudio.isPlaying($0)
        }
        let stalled = playing && queue.sync {
            MeetingCaptureClock.elapsedNanoseconds(from: remote.lastArrival, to: MeetingCaptureClock.now) > stallNanoseconds
        }

        guard newProcessPlaying || (stalled && (currentSet != tappedProcesses || stalled)) else { return }

        if stalled {
            queue.sync {
                if !remoteStallReported {
                    remoteStallReported = true
                    report(.systemAudioStalled(source.displayName))
                }
            }
        }
        tap?.stop()
        tap = nil
        do {
            try startRemoteTap(processes: processes)
            meetingCaptureLog.info("Restarted \(self.source.displayName, privacy: .public) process tap (newProcess=\(newProcessPlaying), stalled=\(stalled))")
        } catch {
            meetingCaptureLog.error("Process tap restart failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    @MainActor private func startMicrophone() throws {
        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw MeetingCaptureError.microphoneUnavailable("no usable input format")
        }
        let tapBlock: AVAudioNodeTapBlock = { [weak self] buffer, when in
            guard let self else { return }
            let host = when.isHostTimeValid ? when.hostTime : MeetingCaptureClock.now
            let block = MeetingCaptureBlock(format: buffer.format, frames: Int(buffer.frameLength), host: host,
                                            list: buffer.audioBufferList)
            self.queue.async { self.ingest(block, into: self.mic) }
        }
        input.installTap(onBus: 0, bufferSize: 4096, format: format, block: tapBlock)
        do {
            engine.prepare()
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            engine.stop()
            throw error
        }
        self.engine = engine
        microphoneFormat = format
        configObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: nil
        ) { [weak self, weak engine] _ in
            Task { @MainActor in
                guard let self, let engine, self.engine === engine, self.sourcesRunning else { return }
                if !engine.isRunning || engine.inputNode.outputFormat(forBus: 0) != self.microphoneFormat {
                    self.recoverMicrophone()
                }
            }
        }
    }

    @MainActor private func recoverMicrophone() {
        guard sourcesRunning else { return }
        let now = MeetingCaptureClock.nanoseconds(MeetingCaptureClock.now)
        let shouldAttempt = queue.sync {
            guard accepting, now >= nextMicRecovery else { return false }
            micRecoveryAttempts = min(micRecoveryAttempts + 1, 5)
            let delay = min(UInt64(1 << micRecoveryAttempts), 30)
            nextMicRecovery = now + delay * 1_000_000_000
            return true
        }
        guard shouldAttempt else { return }
        stopMicrophone()
        do {
            try startMicrophone()
            meetingCaptureLog.info("Microphone engine restarted after input change or stalled callbacks")
        } catch {
            meetingCaptureLog.error("Microphone restart failed: \(error.localizedDescription, privacy: .public)")
            report(.microphoneUnavailable(error.localizedDescription))
        }
    }

    /// Stops sources, then flushes every track so all audio up to now is on disk.
    @MainActor func finish() async {
        teardownSources()
        await withCheckedContinuation { continuation in
            queue.async {
                self.accepting = false
                self.timer?.cancel()
                self.timer = nil
                let end = self.sampleIndex(MeetingCaptureClock.now)
                self.mic.pad(to: end)
                self.remote.pad(to: end)
                self.mic.flush()
                self.remote.flush()
                continuation.resume()
            }
        }
    }

    @MainActor private func teardownSources() {
        sourcesRunning = false
        stopMicrophone()
        tap?.stop()
        tap = nil
        tappedProcesses.removeAll()
    }

    @MainActor private func stopMicrophone() {
        if let observer = configObserver {
            NotificationCenter.default.removeObserver(observer)
            configObserver = nil
        }
        if let engine {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
            self.engine = nil
        }
        microphoneFormat = nil
    }

    // MARK: Queue-confined work

    func beginDictationClip() {
        queue.sync { dictationSamples = [] }
    }

    func endDictationClip() -> Data {
        queue.sync {
            let samples = dictationSamples ?? []
            dictationSamples = nil
            return MeetingCaptureWAV.encode(samples, sampleRate: sampleRate)
        }
    }

    private func sampleIndex(_ host: UInt64) -> Int64 {
        let delta = Int64(bitPattern: MeetingCaptureClock.nanoseconds(host)) - Int64(bitPattern: MeetingCaptureClock.nanoseconds(originHost))
        return delta * Int64(sampleRate) / 1_000_000_000
    }

    private func ingest(_ block: MeetingCaptureBlock, into track: MeetingCaptureTrack) {
        guard accepting else { return }
        guard let samples = track.convert(block), !samples.isEmpty else { return }
        track.lastArrival = MeetingCaptureClock.now
        if track === mic {
            dictationSamples?.append(contentsOf: samples)
            if micStallReported { meetingCaptureLog.info("Microphone audio resumed") }
            micStallReported = false
            micRecoveryAttempts = 0
            nextMicRecovery = 0
        } else if remoteStallReported {
            remoteStallReported = false
            meetingCaptureLog.info("\(self.source.displayName, privacy: .public) audio resumed")
        }
        if !track.sawSignal, samples.contains(where: { $0 != 0 }) {
            track.sawSignal = true
        }
        track.append(samples, at: sampleIndex(block.host))
    }

    private func tick() {
        guard accepting else { return }
        let now = MeetingCaptureClock.now
        let target = sampleIndex(now) - lagSamples
        mic.pad(to: target)
        remote.pad(to: target)

        if MeetingCaptureClock.elapsedNanoseconds(from: mic.lastArrival, to: now) > stallNanoseconds {
            if !micStallReported {
                micStallReported = true
                meetingCaptureLog.error("Microphone callbacks stopped; attempting recovery")
                report(.microphoneStalled)
            }
            Task { @MainActor in self.recoverMicrophone() }
        }
        let nowNanoseconds = MeetingCaptureClock.nanoseconds(now)
        if nowNanoseconds >= nextRemoteCheck {
            nextRemoteCheck = nowNanoseconds + 5_000_000_000
            Task { @MainActor in self.recoverRemoteTapIfNeeded() }
        }
        if !silenceNoted, !remote.sawSignal, MeetingCaptureClock.elapsedNanoseconds(from: originHost, to: now) > 60_000_000_000 {
            silenceNoted = true
            meetingCaptureLog.info("No non-silent \(self.source.displayName, privacy: .public) audio after 60 s.")
        }
    }
}

// MARK: - Tracks

/// Places converted samples on the session timeline and rolls fixed-length chunks.
/// Queue-confined.
private final class MeetingCaptureTrack {
    let directory: URL
    let sampleRate: Int
    let chunkSamples: Int
    var lastArrival: UInt64 = 0
    var sawSignal = false

    private let target: AVAudioFormat
    private let report: (MeetingCaptureError) -> Void
    private var chunkStart: Int64 = 0
    private var pending: [Int16] = []
    private var converter: AVAudioConverter?
    private var converterSource: AVAudioFormat?
    private var conversionReported = false

    init(directory: URL, sampleRate: Int, chunkSamples: Int, target: AVAudioFormat, report: @escaping (MeetingCaptureError) -> Void) {
        self.directory = directory
        self.sampleRate = sampleRate
        self.chunkSamples = chunkSamples
        self.target = target
        self.report = report
    }

    private var cursor: Int64 { chunkStart + Int64(pending.count) }

    /// Writes `samples` beginning at session sample `start`. Gaps become silence;
    /// overlap with already-written time is trimmed.
    func append(_ samples: [Int16], at start: Int64) {
        var offset = 0
        if start < cursor {
            offset = Int(min(cursor - start, Int64(samples.count)))
        } else if start > cursor {
            pad(to: start)
        }
        while offset < samples.count {
            let take = min(chunkSamples - pending.count, samples.count - offset)
            pending.append(contentsOf: samples[offset..<(offset + take)])
            offset += take
            if pending.count == chunkSamples { closeChunk() }
        }
    }

    /// Extends the track with silence up to session sample `target`.
    func pad(to target: Int64) {
        while cursor < target {
            let take = Int(min(Int64(chunkSamples - pending.count), target - cursor))
            pending.append(contentsOf: repeatElement(0, count: take))
            if pending.count == chunkSamples { closeChunk() }
        }
    }

    /// Writes the partial chunk, if any.
    func flush() {
        if !pending.isEmpty { closeChunk() }
    }

    private func closeChunk() {
        let startMilliseconds = chunkStart * 1000 / Int64(sampleRate)
        let url = directory.appendingPathComponent(MeetingCaptureNaming.fileName(startMilliseconds: startMilliseconds))
        do {
            try MeetingCaptureWAV.encode(pending, sampleRate: sampleRate).write(to: url, options: .atomic)
        } catch {
            report(.writeFailed(path: url.path, reason: error.localizedDescription))
        }
        chunkStart += Int64(pending.count)
        pending.removeAll(keepingCapacity: true)
    }

    /// Resamples and downmixes one captured block to 16 kHz mono Int16.
    func convert(_ block: MeetingCaptureBlock) -> [Int16]? {
        guard block.frames > 0, let input = block.makeBuffer() else { return nil }
        if converter == nil || converterSource != block.format {
            converter = AVAudioConverter(from: block.format, to: target)
            converterSource = block.format
        }
        guard let converter else {
            if !conversionReported {
                conversionReported = true
                report(.conversionFailed("no converter from \(block.format)"))
            }
            return nil
        }
        let capacity = AVAudioFrameCount(Double(block.frames) * target.sampleRate / block.format.sampleRate) + 512
        guard let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return nil }

        var supplied = false
        var conversionError: NSError?
        let status = converter.convert(to: output, error: &conversionError) { _, inputStatus in
            if supplied {
                inputStatus.pointee = .noDataNow
                return nil
            }
            supplied = true
            inputStatus.pointee = .haveData
            return input
        }
        guard status != .error, let channel = output.int16ChannelData else {
            if !conversionReported {
                conversionReported = true
                report(.conversionFailed(conversionError?.localizedDescription ?? "unknown"))
            }
            return nil
        }
        return Array(UnsafeBufferPointer(start: channel[0], count: Int(output.frameLength)))
    }
}

// MARK: - Captured audio

private struct MeetingCaptureBlock: @unchecked Sendable {
    let format: AVAudioFormat
    let frames: Int
    let host: UInt64
    private let planes: [Data]

    init(format: AVAudioFormat, frames: Int, host: UInt64, list: UnsafePointer<AudioBufferList>) {
        self.format = format
        self.frames = frames
        self.host = host
        let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: list))
        planes = buffers.map { buffer in
            guard let data = buffer.mData else { return Data() }
            return Data(bytes: data, count: Int(buffer.mDataByteSize))
        }
    }

    func makeBuffer() -> AVAudioPCMBuffer? {
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames)) else { return nil }
        buffer.frameLength = AVAudioFrameCount(frames)
        let destination = UnsafeMutableAudioBufferListPointer(buffer.mutableAudioBufferList)
        guard destination.count == planes.count else { return nil }
        let bytesPerFrame = Int(format.streamDescription.pointee.mBytesPerFrame)
        for index in 0..<destination.count {
            guard let out = destination[index].mData else { return nil }
            let count = min(planes[index].count, frames * bytesPerFrame)
            planes[index].withUnsafeBytes { source in
                if let base = source.baseAddress, count > 0 {
                    memcpy(out, base, count)
                }
            }
        }
        return buffer
    }
}

// MARK: - Process-scoped system audio via Core Audio process tap

@available(macOS 14.4, *)
private final class MeetingCaptureSystemTap {
    private let sourceName: String
    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var aggregateID = AudioObjectID(kAudioObjectUnknown)
    private var procID: AudioDeviceIOProcID?
    private var started = false

    init(sourceName: String) {
        self.sourceName = sourceName
    }

    /// Taps only the given process objects, mixed to stereo. Output is not muted,
    /// so the call app still plays through the speakers/headphones.
    func start(processes: [AudioObjectID], deliver: @escaping @Sendable (MeetingCaptureBlock) -> Void) throws {
        let description = CATapDescription(stereoMixdownOfProcesses: processes)
        description.uuid = UUID()
        description.muteBehavior = CATapMuteBehavior.unmuted
        description.isPrivate = true

        var newTap = AudioObjectID(kAudioObjectUnknown)
        var status = AudioHardwareCreateProcessTap(description, &newTap)
        guard status == noErr else { throw MeetingCaptureError.systemAudioTapFailed(status, sourceName) }
        tapID = newTap

        var asbd: AudioStreamBasicDescription = try meetingCaptureProperty(tapID, kAudioTapPropertyFormat, default: AudioStreamBasicDescription())
        guard let format = AVAudioFormat(streamDescription: &asbd) else {
            throw MeetingCaptureError.coreAudio(step: "tap format", status: kAudioHardwareUnspecifiedError)
        }
        let bytesPerFrame = Int(asbd.mBytesPerFrame)
        guard bytesPerFrame > 0 else {
            throw MeetingCaptureError.coreAudio(step: "tap format", status: kAudioHardwareUnspecifiedError)
        }

        let outputDevice: AudioObjectID = try meetingCaptureProperty(
            AudioObjectID(kAudioObjectSystemObject),
            kAudioHardwarePropertyDefaultOutputDevice,
            default: AudioObjectID(kAudioObjectUnknown)
        )
        let outputUID = try meetingCaptureStringProperty(outputDevice, kAudioDevicePropertyDeviceUID)

        let aggregateDescription: [String: Any] = [
            kAudioAggregateDeviceNameKey: "FluidDictation Remote Tap",
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceMainSubDeviceKey: outputUID,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: outputUID]],
            kAudioAggregateDeviceTapListKey: [[
                kAudioSubTapDriftCompensationKey: true,
                kAudioSubTapUIDKey: description.uuid.uuidString,
            ]],
        ]
        var newAggregate = AudioObjectID(kAudioObjectUnknown)
        status = AudioHardwareCreateAggregateDevice(aggregateDescription as CFDictionary, &newAggregate)
        guard status == noErr else {
            throw MeetingCaptureError.coreAudio(step: "create aggregate device", status: status)
        }
        aggregateID = newAggregate

        var newProc: AudioDeviceIOProcID?
        status = AudioDeviceCreateIOProcIDWithBlock(&newProc, aggregateID, nil) { _, inputData, inputTime, _, _ in
            let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inputData))
            guard let first = buffers.first else { return }
            let frames = Int(first.mDataByteSize) / bytesPerFrame
            let hostTime = inputTime.pointee.mHostTime
            let block = MeetingCaptureBlock(
                format: format,
                frames: frames,
                host: hostTime != 0 ? hostTime : MeetingCaptureClock.now,
                list: inputData
            )
            deliver(block)
        }
        guard status == noErr, let newProc else {
            stop()
            throw MeetingCaptureError.coreAudio(step: "create IO proc", status: status)
        }
        procID = newProc

        status = AudioDeviceStart(aggregateID, newProc)
        guard status == noErr else {
            stop()
            throw MeetingCaptureError.coreAudio(step: "start aggregate device", status: status)
        }
        started = true
    }

    func stop() {
        if let procID {
            if started { AudioDeviceStop(aggregateID, procID) }
            AudioDeviceDestroyIOProcID(aggregateID, procID)
            self.procID = nil
        }
        started = false
        if aggregateID != AudioObjectID(kAudioObjectUnknown) {
            AudioHardwareDestroyAggregateDevice(aggregateID)
            aggregateID = AudioObjectID(kAudioObjectUnknown)
        }
        if tapID != AudioObjectID(kAudioObjectUnknown) {
            AudioHardwareDestroyProcessTap(tapID)
            tapID = AudioObjectID(kAudioObjectUnknown)
        }
    }
}

@available(macOS 14.4, *)
private enum MeetingCaptureProcessAudio {
    static func isPlaying(_ object: AudioObjectID) -> Bool {
        (try? meetingCaptureProperty(object, kAudioProcessPropertyIsRunningOutput, default: UInt32(0))) == 1
    }

    static func processObjects(prefixes: [String], displayName: String) throws -> [AudioObjectID] {
        let running = NSWorkspace.shared.runningApplications.contains { app in
            guard let bid = app.bundleIdentifier else { return false }
            return prefixes.contains { bid.hasPrefix($0) }
        }
        guard running else { throw MeetingCaptureError.appNotRunning(displayName) }

        let all = try meetingCaptureObjectList(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyProcessObjectList)
        let matches = all.filter { object in
            guard let bundleID = try? meetingCaptureStringProperty(object, kAudioProcessPropertyBundleID) else {
                return false
            }
            return prefixes.contains { bundleID.hasPrefix($0) }
        }
        guard !matches.isEmpty else { throw MeetingCaptureError.appAudioSessionNotFound(displayName) }
        return matches
    }
}

// MARK: - Core Audio helpers

private func meetingCaptureProperty<T>(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector, default value: T) throws -> T {
    var address = AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var result = value
    var size = UInt32(MemoryLayout<T>.size)
    let status = AudioObjectGetPropertyData(object, &address, 0, nil, &size, &result)
    guard status == noErr else {
        throw MeetingCaptureError.coreAudio(step: "property \(selector)", status: status)
    }
    return result
}

private func meetingCaptureStringProperty(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) throws -> String {
    var address = AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var rawRef: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    let status = AudioObjectGetPropertyData(object, &address, 0, nil, &size, &rawRef)
    guard status == noErr, let ref = rawRef else {
        throw MeetingCaptureError.coreAudio(step: "string property \(selector)", status: status)
    }
    return ref.takeRetainedValue() as String
}

private func meetingCaptureObjectList(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) throws -> [AudioObjectID] {
    var address = AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var size: UInt32 = 0
    var status = AudioObjectGetPropertyDataSize(object, &address, 0, nil, &size)
    guard status == noErr else {
        throw MeetingCaptureError.coreAudio(step: "list size \(selector)", status: status)
    }
    let count = Int(size) / MemoryLayout<AudioObjectID>.stride
    guard count > 0 else { return [] }
    var list = [AudioObjectID](repeating: AudioObjectID(kAudioObjectUnknown), count: count)
    status = list.withUnsafeMutableBytes { buffer in
        AudioObjectGetPropertyData(object, &address, 0, nil, &size, buffer.baseAddress!)
    }
    guard status == noErr else {
        throw MeetingCaptureError.coreAudio(step: "list \(selector)", status: status)
    }
    return list
}

private func meetingCaptureFourCC(_ status: OSStatus) -> String {
    let value = UInt32(bitPattern: status)
    let bytes = [24, 16, 8, 0].map { UInt8((value >> $0) & 0xFF) }
    if bytes.allSatisfy({ $0 >= 32 && $0 < 127 }) {
        return "'" + String(decoding: bytes, as: UTF8.self) + "'"
    }
    return "\(status)"
}

// MARK: - Clock, naming, WAV

private enum MeetingCaptureClock {
    private static let timebase: mach_timebase_info_data_t = {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        return info
    }()

    static var now: UInt64 { mach_absolute_time() }

    static func nanoseconds(_ ticks: UInt64) -> UInt64 {
        let numer = UInt64(timebase.numer)
        let denom = UInt64(timebase.denom)
        return (ticks / denom) * numer + ((ticks % denom) * numer) / denom
    }

    static func elapsedNanoseconds(from earlier: UInt64, to later: UInt64) -> UInt64 {
        let a = nanoseconds(earlier)
        let b = nanoseconds(later)
        return b > a ? b - a : 0
    }
}

private enum MeetingCaptureNaming {
    /// Zero-padded 12-digit start offset in milliseconds, for example 000000030000.wav.
    static func fileName(startMilliseconds: Int64) -> String {
        let digits = String(startMilliseconds)
        return String(repeating: "0", count: max(0, 12 - digits.count)) + digits + ".wav"
    }
}

private enum MeetingCaptureWAV {
    static func encode(_ samples: [Int16], sampleRate: Int) -> Data {
        let dataBytes = samples.count * MemoryLayout<Int16>.size
        var data = Data(capacity: 44 + dataBytes)

        func put<T: FixedWidthInteger>(_ value: T) {
            var little = value.littleEndian
            withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
        }

        data.append(contentsOf: Array("RIFF".utf8))
        put(UInt32(36 + dataBytes))
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8))
        put(UInt32(16))
        put(UInt16(1))                          // PCM
        put(UInt16(1))                          // mono
        put(UInt32(sampleRate))
        put(UInt32(sampleRate * 2))             // byte rate
        put(UInt16(2))                          // block align
        put(UInt16(16))                         // bits per sample
        data.append(contentsOf: Array("data".utf8))
        put(UInt32(dataBytes))
        let littleSamples = samples.map { $0.littleEndian }
        littleSamples.withUnsafeBytes { data.append(contentsOf: $0) }
        return data
    }
}
