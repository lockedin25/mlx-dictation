import AppKit
import ApplicationServices
import CoreAudio
import Foundation
import os

struct MeetTabMatch: Equatable, Sendable {
    let title: String
    let meetCode: String?
    let hasMediaBadge: Bool

    var identityKey: String {
        if let meetCode, !meetCode.isEmpty {
            return meetCode.lowercased()
        }
        return title.lowercased()
    }
}

enum ChromeMeetEvent: Equatable, Sendable {
    case joined(MeetTabMatch)
    case left
}

/// Parses Chrome window and tab titles for active Google Meet calls.
/// Rejects landing pages like "Google Meet" or unrelated browser tabs.
func parseMeetTabTitle(_ rawTitle: String) -> MeetTabMatch? {
    var working = rawTitle.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !working.isEmpty else { return nil }

    let mediaRegex = try? NSRegularExpression(
        pattern: #"\s+[-–—]\s+(?:Camera and microphone recording|Microphone recording|Camera recording|Audio playing|Audio muted|Tab is using your camera or microphone|Tab is using your microphone|Tab is using your camera|Sharing this tab|Sharing your screen)$"#
    )
    let chromeProfileRegex = try? NSRegularExpression(
        pattern: #"\s+[-–—]\s+Google Chrome(?:\s+(?:Canary|Beta))?(?:\s+[-–—]\s+.+)?$"#
    )

    var hasMediaBadge = false
    var changed = true
    while changed {
        changed = false
        let range = NSRange(working.startIndex..., in: working)
        if let mediaRegex, let m = mediaRegex.firstMatch(in: working, range: range),
           let r = Range(m.range, in: working) {
            working = String(working[..<r.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            hasMediaBadge = true
            changed = true
            continue
        }
        if let chromeProfileRegex, let m = chromeProfileRegex.firstMatch(in: working, range: range),
           let r = Range(m.range, in: working) {
            working = String(working[..<r.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            changed = true
        }
    }

    guard let regex = try? NSRegularExpression(pattern: #"^Meet\s*[-–—]\s*(.+)$"#),
          let match = regex.firstMatch(in: working, range: NSRange(working.startIndex..., in: working)),
          let titleRange = Range(match.range(at: 1), in: working) else {
        return nil
    }

    let detail = String(working[titleRange]).trimmingCharacters(in: .whitespacesAndNewlines)
    guard !detail.isEmpty, detail != "Google Chrome" else { return nil }

    var meetCode: String?
    if let codeRegex = try? NSRegularExpression(pattern: #"\b([a-z]{3}-[a-z]{4}-[a-z]{3})\b"#),
       let codeMatch = codeRegex.firstMatch(in: detail, range: NSRange(detail.startIndex..., in: detail)),
       let codeRange = Range(codeMatch.range(at: 1), in: detail) {
        meetCode = String(detail[codeRange])
    }

    return MeetTabMatch(title: detail, meetCode: meetCode, hasMediaBadge: hasMediaBadge)
}

/// Pure hysteresis state machine for Google Meet detection.
struct ChromeMeetStateMachine: Sendable {
    let startDebounceSeconds: TimeInterval
    let endGraceSeconds: TimeInterval

    private(set) var currentSession: MeetTabMatch?
    private var pendingStartMatch: MeetTabMatch?
    private var pendingStartSince: Date?
    private var pendingSwitchMatch: MeetTabMatch?
    private var pendingSwitchSince: Date?
    private var inactiveSince: Date?

    init(startDebounceSeconds: TimeInterval = 5.0, endGraceSeconds: TimeInterval = 12.0) {
        self.startDebounceSeconds = max(0, startDebounceSeconds)
        self.endGraceSeconds = max(0, endGraceSeconds)
    }

    mutating func update(match: MeetTabMatch?, isCallActive: Bool, now: Date) -> [ChromeMeetEvent] {
        let activeMatch: MeetTabMatch? = isCallActive ? match : nil

        guard let current = currentSession else {
            if let activeMatch {
                if pendingStartMatch?.identityKey == activeMatch.identityKey {
                    pendingStartMatch = activeMatch
                    if let since = pendingStartSince, now.timeIntervalSince(since) >= startDebounceSeconds {
                        currentSession = activeMatch
                        pendingStartMatch = nil
                        pendingStartSince = nil
                        inactiveSince = nil
                        return [.joined(activeMatch)]
                    }
                } else {
                    pendingStartMatch = activeMatch
                    pendingStartSince = now
                }
            } else {
                pendingStartMatch = nil
                pendingStartSince = nil
            }
            return []
        }

        if let activeMatch {
            inactiveSince = nil
            if activeMatch.identityKey == current.identityKey {
                pendingSwitchMatch = nil
                pendingSwitchSince = nil
                currentSession = activeMatch
                return []
            }
            if pendingSwitchMatch?.identityKey == activeMatch.identityKey {
                pendingSwitchMatch = activeMatch
                if let since = pendingSwitchSince, now.timeIntervalSince(since) >= startDebounceSeconds {
                    currentSession = activeMatch
                    pendingSwitchMatch = nil
                    pendingSwitchSince = nil
                    return [.left, .joined(activeMatch)]
                }
            } else {
                pendingSwitchMatch = activeMatch
                pendingSwitchSince = now
            }
            return []
        }

        pendingSwitchMatch = nil
        pendingSwitchSince = nil
        if inactiveSince == nil {
            inactiveSince = now
        }
        if let since = inactiveSince, now.timeIntervalSince(since) >= endGraceSeconds {
            currentSession = nil
            inactiveSince = nil
            return [.left]
        }
        return []
    }
}

/// Polls Google Chrome Core Audio process state and Accessibility tab titles on a background queue.
/// Zero AppleScript, zero Automation permissions, zero UI prompts.
@available(macOS 15.0, *)
final class ChromeMeetDetector: @unchecked Sendable {
    private static let chromeBundlePrefix = "com.google.Chrome"
    private static let chromeMainBundleID = "com.google.Chrome"

    private let log = Logger(subsystem: "local.shubham.fluid-dictation", category: "ChromeMeetDetector")
    private let queue = DispatchQueue(label: "local.shubham.fluid-dictation.chrome-meet-detector", qos: .utility)
    private let pollInterval: TimeInterval
    private let onEvent: @MainActor @Sendable (ChromeMeetEvent) -> Void

    private var timer: DispatchSourceTimer?
    private var stateMachine: ChromeMeetStateMachine
    private var running = false

    init(
        pollInterval: TimeInterval = 2.0,
        startDebounceSeconds: TimeInterval = 5.0,
        endGraceSeconds: TimeInterval = 12.0,
        onEvent: @escaping @MainActor @Sendable (ChromeMeetEvent) -> Void
    ) {
        self.pollInterval = max(1.0, pollInterval)
        self.stateMachine = ChromeMeetStateMachine(
            startDebounceSeconds: startDebounceSeconds,
            endGraceSeconds: endGraceSeconds
        )
        self.onEvent = onEvent
    }

    func start() {
        queue.async { [weak self] in
            guard let self, !self.running else { return }
            self.running = true
            let source = DispatchSource.makeTimerSource(queue: self.queue)
            source.schedule(deadline: .now() + 0.5, repeating: self.pollInterval, leeway: .milliseconds(250))
            source.setEventHandler { [weak self] in
                self?.pollOnce()
            }
            self.timer = source
            source.resume()
            self.log.info("Started Chrome Google Meet detector")
        }
    }

    func stop() {
        queue.async { [weak self] in
            guard let self else { return }
            self.running = false
            self.timer?.cancel()
            self.timer = nil
            self.stateMachine = ChromeMeetStateMachine(
                startDebounceSeconds: self.stateMachine.startDebounceSeconds,
                endGraceSeconds: self.stateMachine.endGraceSeconds
            )
            self.log.info("Stopped Chrome Google Meet detector")
        }
    }

    private func pollOnce() {
        guard running else { return }
        let now = Date()
        let (match, active) = inspectChrome()
        let events = stateMachine.update(match: match, isCallActive: active, now: now)
        guard !events.isEmpty else { return }
        let handler = onEvent
        DispatchQueue.main.async {
            for event in events {
                handler(event)
            }
        }
    }

    private func inspectChrome() -> (MeetTabMatch?, Bool) {
        let chromeApps = NSWorkspace.shared.runningApplications.filter {
            $0.bundleIdentifier == Self.chromeMainBundleID
        }
        guard let chromeApp = chromeApps.first else {
            return (nil, false)
        }

        let audioState = chromeAudioActivity()
        guard audioState.micActive || audioState.audioPlaying else {
            return (nil, false)
        }
        guard AXIsProcessTrusted() else {
            return (nil, false)
        }

        let match = findMeetTab(chromePID: chromeApp.processIdentifier)
        guard let match else {
            return (nil, false)
        }
        let isActive = audioState.micActive || (audioState.audioPlaying && match.hasMediaBadge)
        return (match, isActive)
    }

    private func chromeAudioActivity() -> (micActive: Bool, audioPlaying: Bool) {
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyProcessObjectList,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size) == noErr else {
            return (false, false)
        }
        let count = Int(size) / MemoryLayout<AudioObjectID>.stride
        guard count > 0 else { return (false, false) }
        var objects = [AudioObjectID](repeating: AudioObjectID(kAudioObjectUnknown), count: count)
        let status = objects.withUnsafeMutableBytes { buf in
            AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, buf.baseAddress!)
        }
        guard status == noErr else { return (false, false) }

        var micActive = false
        var audioPlaying = false
        for obj in objects {
            guard let bundleID = audioProcessBundleID(obj),
                  bundleID.hasPrefix(Self.chromeBundlePrefix) else { continue }
            if audioProcessFlag(obj, selector: kAudioProcessPropertyIsRunningInput) {
                micActive = true
            }
            if audioProcessFlag(obj, selector: kAudioProcessPropertyIsRunningOutput) {
                audioPlaying = true
            }
            if micActive && audioPlaying { break }
        }
        return (micActive, audioPlaying)
    }

    private func audioProcessBundleID(_ object: AudioObjectID) -> String? {
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioProcessPropertyBundleID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var rawRef: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        let status = AudioObjectGetPropertyData(object, &addr, 0, nil, &size, &rawRef)
        guard status == noErr, let ref = rawRef else { return nil }
        return ref.takeRetainedValue() as String
    }

    private func audioProcessFlag(_ object: AudioObjectID, selector: AudioObjectPropertySelector) -> Bool {
        var addr = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var value: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        guard AudioObjectGetPropertyData(object, &addr, 0, nil, &size, &value) == noErr else {
            return false
        }
        return value == 1
    }

    private func findMeetTab(chromePID: pid_t) -> MeetTabMatch? {
        let axApp = AXUIElementCreateApplication(chromePID)
        AXUIElementSetMessagingTimeout(axApp, 0.5)

        var windowsValue: CFTypeRef?
        guard AXUIElementCopyAttributeValue(axApp, kAXWindowsAttribute as CFString, &windowsValue) == .success,
              let windows = windowsValue as? [AXUIElement] else {
            return nil
        }

        var candidates: [MeetTabMatch] = []
        for window in windows {
            if let winTitle = axStringAttribute(window, kAXTitleAttribute as CFString),
               let parsed = parseMeetTabTitle(winTitle) {
                candidates.append(parsed)
            }
            var visited = 0
            collectTabMatches(in: window, depth: 0, visited: &visited, matches: &candidates)
        }

        // Prefer matches with active media indicator or explicit meet code
        return candidates.sorted { lhs, rhs in
            let lhsScore = (lhs.hasMediaBadge ? 2 : 0) + (lhs.meetCode != nil ? 1 : 0)
            let rhsScore = (rhs.hasMediaBadge ? 2 : 0) + (rhs.meetCode != nil ? 1 : 0)
            return lhsScore > rhsScore
        }.first
    }

    private func collectTabMatches(
        in element: AXUIElement,
        depth: Int,
        visited: inout Int,
        matches: inout [MeetTabMatch]
    ) {
        if depth > 10 || visited >= 400 { return }
        visited += 1

        let role = axStringAttribute(element, kAXRoleAttribute as CFString) ?? ""
        if role == "AXWebArea" || role == "AXScrollArea" {
            return
        }

        let subrole = axStringAttribute(element, kAXSubroleAttribute as CFString) ?? ""
        if subrole == "AXTabButton" || role == "AXRadioButton" {
            if let title = axStringAttribute(element, kAXTitleAttribute as CFString),
               let parsed = parseMeetTabTitle(title) {
                matches.append(parsed)
            }
            if subrole == "AXTabButton" { return }
        }

        var childrenValue: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, kAXChildrenAttribute as CFString, &childrenValue) == .success,
              let children = childrenValue as? [AXUIElement] else {
            return
        }
        for child in children {
            collectTabMatches(in: child, depth: depth + 1, visited: &visited, matches: &matches)
            if visited >= 400 { break }
        }
    }

    private func axStringAttribute(_ element: AXUIElement, _ attribute: CFString) -> String? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, attribute, &value) == .success else {
            return nil
        }
        return value as? String
    }
}
