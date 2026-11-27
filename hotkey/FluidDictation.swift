import AppKit
import AVFoundation
import Carbon
import ApplicationServices
import QuartzCore

enum BezelMode: Equatable {
    case hidden
    case recording
    case transcribing
    case feedback(symbol: String, tint: NSColor, message: String)
}

final class BezelView: NSView {
    private let effectView = NSVisualEffectView()
    private let overlayView = NSView()
    private let tintLayer = CALayer()
    private let borderLayer = CALayer()
    private var barLayers: [CALayer] = []
    private let iconView = NSImageView()
    private let messageLabel = NSTextField(labelWithString: "")
    private let barCount = 7
    private let envelopes: [CGFloat] = [0.42, 0.70, 0.92, 1.0, 0.92, 0.70, 0.42]
    private var phase: Double = 0
    private var smoothedLevel: Double = 0
    private var mode: BezelMode = .hidden
    private var lastMaskSize: NSSize = .zero

    static let compactSize = NSSize(width: 76, height: 26)
    private static let messageFont = NSFont.systemFont(ofSize: 12, weight: .medium)

    static func preferredSize(for mode: BezelMode) -> NSSize {
        switch mode {
        case .hidden, .recording, .transcribing:
            return compactSize
        case let .feedback(_, _, message):
            let textWidth = (message as NSString).size(withAttributes: [.font: messageFont]).width
            let width = min(380.0, max(120.0, ceil(textWidth) + 44.0))
            return NSSize(width: width, height: 28)
        }
    }

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        layer?.backgroundColor = NSColor.clear.cgColor

        effectView.autoresizingMask = [.width, .height]
        effectView.material = .hudWindow
        effectView.blendingMode = .behindWindow
        effectView.state = .active
        effectView.appearance = NSAppearance(named: .darkAqua)
        effectView.wantsLayer = true
        addSubview(effectView)

        overlayView.autoresizingMask = [.width, .height]
        overlayView.wantsLayer = true
        overlayView.layer?.backgroundColor = NSColor.clear.cgColor
        addSubview(overlayView)

        tintLayer.backgroundColor = NSColor(calibratedRed: 0.06, green: 0.06, blue: 0.08, alpha: 0.78).cgColor
        overlayView.layer?.addSublayer(tintLayer)

        borderLayer.borderWidth = 1.0
        borderLayer.borderColor = NSColor.white.withAlphaComponent(0.18).cgColor
        overlayView.layer?.addSublayer(borderLayer)

        let barWidth: CGFloat = 2.5
        for _ in 0..<barCount {
            let bar = CALayer()
            bar.cornerRadius = barWidth / 2.0
            bar.backgroundColor = NSColor.white.withAlphaComponent(0.95).cgColor
            overlayView.layer?.addSublayer(bar)
            barLayers.append(bar)
        }

        iconView.imageScaling = .scaleProportionallyUpOrDown
        iconView.isHidden = true
        overlayView.addSubview(iconView)

        messageLabel.font = Self.messageFont
        messageLabel.textColor = NSColor.white.withAlphaComponent(0.92)
        messageLabel.alignment = .left
        messageLabel.lineBreakMode = .byTruncatingTail
        messageLabel.maximumNumberOfLines = 1
        messageLabel.isHidden = true
        overlayView.addSubview(messageLabel)

        updateGeometry()
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layout() {
        super.layout()
        updateGeometry()
    }

    func updateGeometry() {
        let radius = bounds.height / 2.0
        CATransaction.begin()
        CATransaction.setDisableActions(true)

        layer?.cornerRadius = radius
        layer?.masksToBounds = true

        effectView.frame = bounds
        effectView.layer?.cornerRadius = radius
        effectView.layer?.masksToBounds = true

        if lastMaskSize != bounds.size, bounds.width > 0, bounds.height > 0 {
            lastMaskSize = bounds.size
            let mask = NSImage(size: bounds.size, flipped: false) { rect in
                NSColor.black.setFill()
                NSBezierPath(roundedRect: rect, xRadius: rect.height / 2.0, yRadius: rect.height / 2.0).fill()
                return true
            }
            effectView.maskImage = mask
        }

        overlayView.frame = bounds
        overlayView.layer?.cornerRadius = radius
        overlayView.layer?.masksToBounds = true

        tintLayer.frame = bounds
        tintLayer.cornerRadius = radius
        borderLayer.frame = bounds
        borderLayer.cornerRadius = radius

        let barWidth: CGFloat = 2.5
        let spacing: CGFloat = 4.0
        let totalWidth = CGFloat(barCount) * barWidth + CGFloat(barCount - 1) * spacing
        let startX = (bounds.width - totalWidth) / 2.0
        for i in 0..<barCount {
            var f = barLayers[i].frame
            f.origin.x = startX + CGFloat(i) * (barWidth + spacing)
            f.size.width = barWidth
            if f.size.height <= 0 {
                f.size.height = 3.0
                f.origin.y = (bounds.height - 3.0) / 2.0
            }
            barLayers[i].frame = f
        }

        iconView.frame = NSRect(x: 11, y: (bounds.height - 13) / 2.0, width: 13, height: 13)
        messageLabel.frame = NSRect(x: 29, y: (bounds.height - 16) / 2.0, width: max(0, bounds.width - 41), height: 16)

        CATransaction.commit()
    }

    func setMode(_ newMode: BezelMode) {
        mode = newMode
        updateGeometry()
        switch newMode {
        case .hidden:
            break
        case .recording:
            iconView.isHidden = true
            messageLabel.isHidden = true
            CATransaction.begin()
            CATransaction.setDisableActions(true)
            for bar in barLayers { bar.isHidden = false }
            CATransaction.commit()
            tick(audioLevel: 0)
        case .transcribing:
            iconView.isHidden = true
            messageLabel.isHidden = true
            CATransaction.begin()
            CATransaction.setDisableActions(true)
            for bar in barLayers { bar.isHidden = false }
            CATransaction.commit()
            tick(audioLevel: 0)
        case let .feedback(symbol, tint, message):
            CATransaction.begin()
            CATransaction.setDisableActions(true)
            for bar in barLayers { bar.isHidden = true }
            CATransaction.commit()
            if let img = NSImage(systemSymbolName: symbol, accessibilityDescription: nil) {
                let config = NSImage.SymbolConfiguration(pointSize: 11.5, weight: .semibold)
                iconView.image = img.withSymbolConfiguration(config)
                iconView.contentTintColor = tint
            }
            iconView.isHidden = false
            messageLabel.stringValue = message
            messageLabel.isHidden = false
        }
    }

    func tick(audioLevel: Double) {
        if audioLevel > smoothedLevel {
            smoothedLevel = smoothedLevel * 0.42 + audioLevel * 0.58
        } else {
            smoothedLevel = smoothedLevel * 0.76 + audioLevel * 0.24
        }

        CATransaction.begin()
        CATransaction.setDisableActions(true)

        switch mode {
        case .recording:
            phase += 0.24
            let midY = bounds.height / 2.0
            for i in 0..<barCount {
                let bar = barLayers[i]
                let ripple = 0.78 + 0.22 * sin(phase + Double(i) * 0.95)
                let idle = 2.8 + 0.35 * sin(phase * 0.35 + Double(i) * 0.7)
                let dynamic = CGFloat(smoothedLevel) * 13.5 * envelopes[i] * CGFloat(ripple)
                let h = max(2.8, min(16.5, CGFloat(idle) + dynamic))
                var f = bar.frame
                f.size.height = h
                f.origin.y = midY - h / 2.0
                bar.frame = f
                bar.opacity = Float(0.78 + 0.22 * smoothedLevel)
            }
        case .transcribing:
            phase += 0.22
            let midY = bounds.height / 2.0
            for i in 0..<barCount {
                let bar = barLayers[i]
                let wave = (sin(phase - Double(i) * 0.65) + 1.0) * 0.5
                let h = CGFloat(3.0 + wave * 6.0)
                var f = bar.frame
                f.size.height = h
                f.origin.y = midY - h / 2.0
                bar.frame = f
                bar.opacity = Float(0.38 + 0.58 * wave)
            }
        case .hidden, .feedback:
            break
        }

        CATransaction.commit()
    }
}

final class DictationApp: NSObject, NSApplicationDelegate {
    static func resolveRoot() -> URL {
        let fm = FileManager.default
        if let envRoot = ProcessInfo.processInfo.environment["FLUID_ROOT"], !envRoot.isEmpty {
            let url = URL(fileURLWithPath: envRoot)
            if fm.fileExists(atPath: url.appendingPathComponent("hotkey/worker.py").path) { return url }
        }
        if let res = Bundle.main.resourceURL,
           fm.fileExists(atPath: res.appendingPathComponent("hotkey/worker.py").path) {
            return res
        }
        let appSupport = fm.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/FluidDictation")
        if fm.fileExists(atPath: appSupport.appendingPathComponent("hotkey/worker.py").path) {
            return appSupport
        }
        let devRepo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        if fm.fileExists(atPath: devRepo.appendingPathComponent("hotkey/worker.py").path) {
            return devRepo
        }
        return appSupport
    }

    let root = DictationApp.resolveRoot()
    var statusItem: NSStatusItem!
    var info: NSMenuItem!
    var toggleItem: NSMenuItem!
    var fnItem: NSMenuItem!
    var worker: Process?
    var input: Pipe?
    var outputBuffer = Data()
    var ready = false
    var busy = false
    var recorder: AVAudioRecorder?
    private var sharedDictation: (() -> Data)?
    private var isDictating: Bool { recorder != nil || sharedDictation != nil }
    var recordingURL: URL?
    var recordingStartedAt = Date.distantPast
    var targetPID: pid_t?
    var targetAppName: String = ""
    var targetAppBundleID: String = ""
    var lastModelName: String = ""
    var limit: Timer?
    var hotkey: EventHotKeyRef?
    var eventTap: CFMachPort?
    var eventTapRunLoop: CFRunLoop?
    var eventTapThread: Thread?
    var tapRetryTimer: Timer?
    var globalMonitor: Any?
    var localMonitor: Any?
    var fnDown = false
    var fnUsed = false
    var fnStart = Date()
    var lastFnEventAt = Date.distantPast
    private var meetings: AnyObject?
    private var waitingForMeeting = false
    private var meetingRecording = false
    var hotkeyRegistered = false
    var watchdog: Timer?
    var hud: NSPanel?
    var bezelView: BezelView?
    var hudTimer: Timer?
    var meterTimer: Timer?
    var hudGeneration: Int = 0
    var lastAudioSeconds: Double = 0
    var lastAudioRMS: Double = 0
    var fnHoldTimer: Timer?
    var isHoldRecording = false
    var polishRequested = false
    var polishing = false
    var lastStopAt = Date.distantPast
    var useFn: Bool {
        if UserDefaults.standard.object(forKey: "useFn") == nil { return true }
        return UserDefaults.standard.bool(forKey: "useFn")
    }

    func applicationDidFinishLaunching(_ note: Notification) {
        NSApp.setActivationPolicy(.accessory)
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        setMenuBarIcon(symbol: "waveform", fallback: "◌")
        statusItem.button?.toolTip = "Fluid local dictation"
        let menu = NSMenu()
        info = NSMenuItem(title: "Loading local model…", action: nil, keyEquivalent: "")
        menu.addItem(info)
        toggleItem = NSMenuItem(title: "Start / stop recording (Fn)", action: #selector(toggle), keyEquivalent: "")
        toggleItem.target = self; menu.addItem(toggleItem)
        fnItem = NSMenuItem(title: "Fn / Globe hotkey", action: #selector(toggleFn), keyEquivalent: "")
        fnItem.target = self; menu.addItem(fnItem)
        menu.addItem(.separator())
        let perms = NSMenuItem(title: "Enable microphone and paste permissions…", action: #selector(permissions), keyEquivalent: "")
        perms.target = self; menu.addItem(perms)
        let quit = NSMenuItem(title: "Quit Fluid Dictation", action: #selector(quitApp), keyEquivalent: "")
        quit.target = self; menu.addItem(quit)
        if #available(macOS 15.0, *) {
            let controller = MeetingController(root: root, menu: menu)
            controller.recordingChanged = { [weak self] recording in
                self?.meetingRecording = recording
                self?.statusItem.button?.attributedTitle = recording
                    ? NSAttributedString(string: " REC", attributes: [.foregroundColor: NSColor.systemRed])
                    : NSAttributedString(string: "")
            }
            controller.standaloneMicInUse = { [weak self] in self?.recorder != nil }
            controller.beforeCaptureStops = { [weak self] in
                if self?.sharedDictation != nil { self?.stopRecording() }
            }
            meetings = controller
            if CommandLine.arguments.contains("--record-meeting") {
                DispatchQueue.main.async { controller.toggle() }
            }
        }
        statusItem.menu = menu
        registerHotkey()
        installFnMonitor()
        startWorker()
        if CommandLine.arguments.contains("--setup") || !AXIsProcessTrusted() {
            promptAccessibility()
        }
    }

    func promptAccessibility() {
        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
        _ = AXIsProcessTrustedWithOptions(options)
    }

    func setMenuBarIcon(symbol: String, fallback: String) {
        if let img = NSImage(systemSymbolName: symbol, accessibilityDescription: "Fluid Dictation") {
            let config = NSImage.SymbolConfiguration(pointSize: 13, weight: .medium)
            let configured = img.withSymbolConfiguration(config) ?? img
            configured.isTemplate = true
            statusItem.button?.image = configured
            if meetingRecording {
                statusItem.button?.attributedTitle = NSAttributedString(
                    string: " REC", attributes: [.foregroundColor: NSColor.systemRed])
            } else {
                statusItem.button?.title = ""
            }
        } else {
            statusItem.button?.image = nil
            statusItem.button?.title = fallback
        }
    }

    func update(_ text: String, symbol: String = "waveform", fallback: String = "◌", bezel: BezelMode = .hidden, autoHideAfter: TimeInterval? = nil) {
        info.title = text
        setMenuBarIcon(symbol: symbol, fallback: fallback)
        statusItem.button?.toolTip = "Fluid Dictation: \(text)"
        updateBezel(bezel, autoHideAfter: autoHideAfter)
        let status: [String: Any] = ["message": text, "modelReady": ready,
            "recording": isDictating, "transcribing": busy,
            "hotkeyRegistered": hotkeyRegistered, "eventTapActive": eventTap != nil,
            "accessibilityGranted": AXIsProcessTrusted(),
            "microphoneGranted": AVCaptureDevice.authorizationStatus(for: .audio) == .authorized,
            "lastAudioSeconds": lastAudioSeconds, "lastAudioRMS": lastAudioRMS,
            "updatedAt": ISO8601DateFormatter().string(from: Date())]
        if let data = try? JSONSerialization.data(withJSONObject: status, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: root.appendingPathComponent("hotkey/status.json"), options: .atomic)
        }
    }

    func ensureHUD() {
        guard hud == nil else { return }
        let size = NSSize(width: 76, height: 26)
        let panel = NSPanel(contentRect: NSRect(origin: .zero, size: size), styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
        panel.level = NSWindow.Level(rawValue: Int(CGWindowLevelForKey(.statusWindow)) + 1)
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.ignoresMouseEvents = true
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .ignoresCycle]
        panel.alphaValue = 0

        let view = BezelView(frame: NSRect(origin: .zero, size: size))
        panel.contentView = view
        hud = panel
        bezelView = view
    }

    func activeScreen() -> NSScreen? {
        let mouse = NSEvent.mouseLocation
        return NSScreen.screens.first(where: { NSMouseInRect(mouse, $0.frame, false) }) ?? NSScreen.main ?? NSScreen.screens.first
    }

    func updateBezel(_ mode: BezelMode, autoHideAfter: TimeInterval? = nil) {
        hudTimer?.invalidate()
        hudTimer = nil

        if mode == .hidden {
            stopMeterTimer()
            hideHUD()
            return
        }

        ensureHUD()
        guard let panel = hud else { return }
        hudGeneration += 1

        let size = BezelView.preferredSize(for: mode)
        if let screen = activeScreen() {
            let x = screen.visibleFrame.midX - size.width / 2.0
            let y = screen.visibleFrame.minY + 14
            panel.setFrame(NSRect(origin: NSPoint(x: x, y: y), size: size), display: true)
        } else {
            panel.setContentSize(size)
        }
        bezelView?.frame = NSRect(origin: .zero, size: size)
        bezelView?.setMode(mode)

        if !panel.isVisible || panel.alphaValue < 0.99 {
            panel.orderFrontRegardless()
            NSAnimationContext.runAnimationGroup { ctx in
                ctx.duration = 0.14
                ctx.timingFunction = CAMediaTimingFunction(name: .easeOut)
                panel.animator().alphaValue = 1.0
            }
        }

        switch mode {
        case .recording, .transcribing:
            startMeterTimer()
        case .feedback, .hidden:
            stopMeterTimer()
        }

        if let delay = autoHideAfter {
            hudTimer = Timer.scheduledTimer(withTimeInterval: delay, repeats: false) { [weak self] _ in
                self?.hideHUD()
            }
        }
    }

    func hideHUD() {
        stopMeterTimer()
        guard let panel = hud, panel.isVisible else { return }
        hudGeneration += 1
        let gen = hudGeneration
        NSAnimationContext.runAnimationGroup({ ctx in
            ctx.duration = 0.18
            ctx.timingFunction = CAMediaTimingFunction(name: .easeIn)
            panel.animator().alphaValue = 0.0
        }, completionHandler: { [weak self] in
            DispatchQueue.main.async {
                guard let self = self, self.hudGeneration == gen else { return }
                self.hud?.orderOut(nil)
                self.bezelView?.setMode(.hidden)
            }
        })
    }

    func startMeterTimer() {
        guard meterTimer == nil else { return }
        let timer = Timer(timeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            var level = 0.0
            if let rec = self.recorder, rec.isRecording {
                rec.updateMeters()
                let db = Double(rec.averagePower(forChannel: 0))
                let raw = max(0.0, min(1.0, (db + 52.0) / 40.0))
                level = pow(raw, 1.15)
            }
            self.bezelView?.tick(audioLevel: level)
        }
        RunLoop.main.add(timer, forMode: .common)
        meterTimer = timer
    }

    func stopMeterTimer() {
        meterTimer?.invalidate()
        meterTimer = nil
    }

    func registerHotkey() {
        var spec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        InstallEventHandler(GetApplicationEventTarget(), { _, _, context in
            guard let context = context else { return noErr }
            let app = Unmanaged<DictationApp>.fromOpaque(context).takeUnretainedValue()
            DispatchQueue.main.async { app.toggle() }
            return noErr
        }, 1, &spec, Unmanaged.passUnretained(self).toOpaque(), nil)
        let code = RegisterEventHotKey(UInt32(kVK_Space), UInt32(controlKey | optionKey), EventHotKeyID(signature: 0x464C5544, id: 1), GetApplicationEventTarget(), 0, &hotkey)
        hotkeyRegistered = code == noErr
    }

    func disableSystemFnEmojiPicker() {
        let domain = "com.apple.HIToolbox" as CFString
        let key = "AppleFnUsageType" as CFString
        let current = CFPreferencesCopyValue(key, domain, kCFPreferencesCurrentUser, kCFPreferencesAnyHost) as? NSNumber
        if current?.intValue != 0 {
            CFPreferencesSetValue(key, 0 as CFNumber, domain, kCFPreferencesCurrentUser, kCFPreferencesAnyHost)
            CFPreferencesAppSynchronize(domain)
            let activate = "/System/Library/PrivateFrameworks/SystemAdministration.framework/Resources/activateSettings"
            if FileManager.default.isExecutableFile(atPath: activate) {
                let p = Process()
                p.executableURL = URL(fileURLWithPath: activate)
                p.arguments = ["-u"]
                try? p.run()
            }
        }
    }

    func installFnMonitor() {
        removeFnMonitors()
        fnItem.state = useFn ? .on : .off
        guard useFn else { return }

        disableSystemFnEmojiPicker()

        if startEventTap() {
            return
        }

        // Fallback if Accessibility isn't granted yet; upgrade automatically once granted
        globalMonitor = NSEvent.addGlobalMonitorForEvents(matching: [.flagsChanged, .keyDown]) { [weak self] event in self?.handleLegacyFn(event) }
        localMonitor = NSEvent.addLocalMonitorForEvents(matching: [.flagsChanged, .keyDown]) { [weak self] event in self?.handleLegacyFn(event); return event }
        tapRetryTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            guard let self = self, self.useFn else { return }
            if AXIsProcessTrusted() {
                self.installFnMonitor()
            }
        }
    }

    func removeFnMonitors() {
        tapRetryTimer?.invalidate()
        tapRetryTimer = nil
        if let monitor = globalMonitor { NSEvent.removeMonitor(monitor); globalMonitor = nil }
        if let monitor = localMonitor { NSEvent.removeMonitor(monitor); localMonitor = nil }
        if let tap = eventTap {
            CGEvent.tapEnable(tap: tap, enable: false)
            CFMachPortInvalidate(tap)
            eventTap = nil
        }
        if let rl = eventTapRunLoop {
            CFRunLoopStop(rl)
            eventTapRunLoop = nil
        }
        eventTapThread = nil
    }

    func startEventTap() -> Bool {
        let mask: CGEventMask =
            (1 << CGEventType.flagsChanged.rawValue) |
            (1 << CGEventType.keyDown.rawValue) |
            (1 << CGEventType.keyUp.rawValue)

        let refcon = Unmanaged.passUnretained(self).toOpaque()
        let callback: CGEventTapCallBack = { _, type, event, userInfo in
            guard let userInfo = userInfo else { return Unmanaged.passUnretained(event) }
            let app = Unmanaged<DictationApp>.fromOpaque(userInfo).takeUnretainedValue()
            return app.handleCGEvent(type: type, event: event)
        }

        let createdTap =
            CGEvent.tapCreate(tap: .cghidEventTap, place: .headInsertEventTap, options: .defaultTap, eventsOfInterest: mask, callback: callback, userInfo: refcon) ??
            CGEvent.tapCreate(tap: .cgSessionEventTap, place: .headInsertEventTap, options: .defaultTap, eventsOfInterest: mask, callback: callback, userInfo: refcon)

        guard let tap = createdTap else { return false }
        eventTap = tap

        let thread = Thread { [weak self] in
            guard let self = self, let tap = self.eventTap else { return }
            let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
            let rl = CFRunLoopGetCurrent()
            self.eventTapRunLoop = rl
            CFRunLoopAddSource(rl, source, .commonModes)
            CGEvent.tapEnable(tap: tap, enable: true)
            CFRunLoopRun()
        }
        thread.qualityOfService = .userInteractive
        eventTapThread = thread
        thread.start()
        return true
    }

    func handleCGEvent(type: CGEventType, event: CGEvent) -> Unmanaged<CGEvent>? {
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            if let tap = eventTap {
                CGEvent.tapEnable(tap: tap, enable: true)
            }
            return Unmanaged.passUnretained(event)
        }

        guard useFn else { return Unmanaged.passUnretained(event) }
        let keyCode = event.getIntegerValueField(.keyboardEventKeycode)

        // Globe key (keyCode 179 on macOS 26+ / Apple Silicon / external keyboards) -> always swallow so Emoji picker never opens
        if keyCode == 179 && (type == .keyDown || type == .keyUp) {
            let isDown = (type == .keyDown)
            let hasOtherMods = !event.flags.intersection([.maskCommand, .maskControl, .maskAlternate, .maskShift]).isEmpty
            DispatchQueue.main.async { [weak self] in
                self?.processFnTransition(down: isDown, hasOtherModifiers: hasOtherMods)
            }
            return nil
        }

        // Built-in Fn key (keyCode 63) or Globe flagsChanged (keyCode 179)
        if type == .flagsChanged && (keyCode == 63 || keyCode == 179) {
            let isDown = event.flags.contains(.maskSecondaryFn)
            let hasOtherMods = !event.flags.intersection([.maskCommand, .maskControl, .maskAlternate, .maskShift]).isEmpty
            DispatchQueue.main.async { [weak self] in
                self?.processFnTransition(down: isDown, hasOtherModifiers: hasOtherMods)
            }
            if !hasOtherMods {
                return nil
            }
            return Unmanaged.passUnretained(event)
        }

        // Any other key pressed while Fn is down marks it as a chord (e.g., Fn+Left, Fn+F1)
        if type == .keyDown {
            DispatchQueue.main.async { [weak self] in
                self?.markFnUsed()
            }
            return Unmanaged.passUnretained(event)
        }

        if type == .flagsChanged {
            let hasOtherMods = !event.flags.intersection([.maskCommand, .maskControl, .maskAlternate, .maskShift]).isEmpty
            if hasOtherMods {
                DispatchQueue.main.async { [weak self] in
                    self?.markFnUsed()
                }
            }
        }

        return Unmanaged.passUnretained(event)
    }

    func handleLegacyFn(_ event: NSEvent) {
        guard useFn else { return }
        if event.type == .keyDown {
            markFnUsed()
            return
        }
        let down = event.modifierFlags.contains(.function)
        let other = !event.modifierFlags.intersection([.command, .control, .option, .shift]).isEmpty
        processFnTransition(down: down, hasOtherModifiers: other)
    }

    func markFnUsed() {
        if fnDown {
            fnUsed = true
            fnHoldTimer?.invalidate()
            fnHoldTimer = nil
        }
    }

    func processFnTransition(down: Bool, hasOtherModifiers other: Bool) {
        let now = Date()
        if down && !fnDown {
            if now.timeIntervalSince(lastFnEventAt) < 0.04 { return }
            lastFnEventAt = now
            fnDown = true
            fnUsed = other
            fnStart = now
            isHoldRecording = false
            fnHoldTimer?.invalidate()
            if !other && !isDictating && !busy && ready {
                fnHoldTimer = Timer.scheduledTimer(withTimeInterval: 0.4, repeats: false) { [weak self] _ in
                    guard let self = self, self.fnDown, !self.fnUsed, !self.isDictating else { return }
                    self.isHoldRecording = true
                    self.beginRecording()
                }
            }
        } else if down && other {
            markFnUsed()
        } else if !down && fnDown {
            lastFnEventAt = now
            fnDown = false
            fnHoldTimer?.invalidate()
            fnHoldTimer = nil
            if isHoldRecording {
                isHoldRecording = false
                stopRecording()
            } else if !fnUsed && (now.timeIntervalSince(fnStart) < 0.4 || isDictating) {
                toggle()
            }
        }
    }

    @objc func toggleFn() {
        UserDefaults.standard.set(!useFn, forKey: "useFn")
        installFnMonitor()
        if useFn {
            permissions()
        }
    }

    @objc func permissions() {
        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
        _ = AXIsProcessTrustedWithOptions(options)
        AVCaptureDevice.requestAccess(for: .audio) { [weak self] _ in
            DispatchQueue.main.async { self?.installFnMonitor() }
        }
    }

    func startWorker() {
        guard worker == nil else { return }
        ready = false
        let process = Process(); let stdinPipe = Pipe(); let stdoutPipe = Pipe()
        process.executableURL = root.appendingPathComponent(".venv/bin/python")
        process.arguments = ["-u", root.appendingPathComponent("hotkey/worker.py").path]
        var env = ProcessInfo.processInfo.environment
        let homeBin = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".local/bin").path
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:\(homeBin):/usr/bin:/bin:/usr/sbin:/sbin"
        env["HF_HUB_OFFLINE"] = "1"; env["PYTHONUNBUFFERED"] = "1"
        process.environment = env
        process.standardInput = stdinPipe; process.standardOutput = stdoutPipe
        // Diagnostics only: no audio or transcript logging.
        let log = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Logs/FluidDictation-worker.log")
        if !FileManager.default.fileExists(atPath: log.path) { FileManager.default.createFile(atPath: log.path, contents: nil) }
        if let handle = try? FileHandle(forWritingTo: log) { handle.seekToEndOfFile(); process.standardError = handle }
        stdoutPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty { handle.readabilityHandler = nil; return }
            DispatchQueue.main.async { self?.receive(data) }
        }
        process.terminationHandler = { [weak self] _ in
            DispatchQueue.main.async {
                self?.watchdog?.invalidate()
                self?.ready = false; self?.busy = false; self?.worker = nil
                self?.recorder?.stop(); self?.recorder = nil
                self?.removeRecording()
                self?.update("Model stopped. Press shortcut to reload.", symbol: "exclamationmark.triangle", fallback: "!", bezel: .feedback(symbol: "exclamationmark.triangle.fill", tint: .systemYellow, message: "Model stopped. Press shortcut to reload."), autoHideAfter: 2.8)
            }
        }
        do { try process.run(); worker = process; input = stdinPipe }
        catch {
            let msg = "Could not start model"
            update(msg, symbol: "exclamationmark.triangle", fallback: "!", bezel: .feedback(symbol: "exclamationmark.triangle.fill", tint: .systemYellow, message: msg), autoHideAfter: 3.0)
            alert("Model failed", error.localizedDescription)
        }
    }

    func receive(_ data: Data) {
        outputBuffer.append(data)
        while let newline = outputBuffer.firstIndex(of: 10) {
            let line = outputBuffer.prefix(upTo: newline)
            outputBuffer.removeSubrange(...newline)
            guard let value = try? JSONSerialization.jsonObject(with: line) as? [String: Any], let status = value["status"] as? String else { continue }
            if status == "ready" {
                ready = true
                update("Ready · Press Fn to dictate", symbol: "waveform", fallback: "◌", bezel: .hidden)
            } else if status == "result" {
                watchdog?.invalidate()
                busy = false
                lastAudioSeconds = value["audio_seconds"] as? Double ?? 0
                lastAudioRMS = value["rms"] as? Double ?? 0
                lastModelName = value["model"] as? String ?? ""
                let text = (value["text"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                removeRecording()
                guard !text.isEmpty else {
                    update("Ready · Press Fn to dictate", symbol: "waveform", fallback: "◌", bezel: .hidden)
                    continue
                }
                guard polishRequested else {
                    deliver(text)
                    logDictation(rawText: text, polishedText: nil)
                    continue
                }
                polishRequested = false
                busy = true
                polishing = true
                update("Polishing…", symbol: "sparkles", fallback: "…", bezel: .transcribing)
                polish(text) { [weak self] polished in
                    self?.busy = false
                    self?.polishing = false
                    let pasted = self?.deliver(polished ?? text) ?? false
                    self?.logDictation(rawText: text, polishedText: polished)
                    // Only report the fallback when it was actually pasted; otherwise keep deliver's clipboard/focus message.
                    if polished == nil && pasted {
                        let msg = "Polish failed · pasted raw text"
                        self?.update(msg, symbol: "exclamationmark.triangle", fallback: "!", bezel: .feedback(symbol: "exclamationmark.triangle.fill", tint: .systemYellow, message: msg), autoHideAfter: 2.5)
                    }
                }
            } else if status == "error" || status == "fatal" {
                watchdog?.invalidate()
                busy = false; removeRecording()
                let msg = value["message"] as? String ?? "Transcription failed"
                update(msg, symbol: "exclamationmark.triangle", fallback: "!", bezel: .feedback(symbol: "exclamationmark.triangle.fill", tint: .systemYellow, message: msg), autoHideAfter: 3.0)
            }
        }
    }

    @objc func toggle() {
        if isDictating { stopRecording(); return }
        // Fn (or the shortcut) again while transcribing: polish this dictation. A quick double-tap to stop does this.
        guard !busy else { if !requestPolish() { NSSound.beep() }; return }
        // A polish double-tap that lands just after a very fast transcription must not start a new recording.
        guard Date().timeIntervalSince(lastStopAt) > 0.5 else { return }
        guard ready else { if worker == nil { update("Loading local model…", symbol: "waveform", fallback: "◌", bezel: .transcribing); startWorker() }; NSSound.beep(); return }
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: beginRecording()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
                DispatchQueue.main.async {
                    if granted { self?.beginRecording() }
                    else {
                        let msg = "Microphone permission needed"
                        self?.update(msg, symbol: "mic.slash", fallback: "!", bezel: .feedback(symbol: "mic.slash.fill", tint: .systemYellow, message: msg), autoHideAfter: 3.0)
                    }
                }
            }
        default:
            let msg = "Microphone permission needed"
            update(msg, symbol: "mic.slash", fallback: "!", bezel: .feedback(symbol: "mic.slash.fill", tint: .systemYellow, message: msg), autoHideAfter: 3.0)
            NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")!)
        }
    }

    func beginRecording() {
        let frontApp = NSWorkspace.shared.frontmostApplication
        targetPID = frontApp?.processIdentifier
        targetAppName = frontApp?.localizedName ?? ""
        targetAppBundleID = frontApp?.bundleIdentifier ?? ""
        polishRequested = false
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("fluid-dictation-\(UUID().uuidString).wav")
        if #available(macOS 15.0, *) {
            let clip = MainActor.assumeIsolated { (meetings as? MeetingController)?.beginDictationClip() }
            if let clip {
                sharedDictation = clip
                markDictationStarted(at: url)
                return
            }
        }
        do {
            let mic = try AVAudioRecorder(url: url, settings: [AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 16000, AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false])
            mic.isMeteringEnabled = true
            guard mic.record() else { throw NSError(domain: "Fluid", code: 1, userInfo: [NSLocalizedDescriptionKey: "Could not open the default microphone"]) }
            recorder = mic
            markDictationStarted(at: url)
        } catch {
            try? FileManager.default.removeItem(at: url)
            let msg = "Microphone failed: \(error.localizedDescription)"
            update(msg, symbol: "mic.slash", fallback: "!", bezel: .feedback(symbol: "mic.slash.fill", tint: .systemYellow, message: msg), autoHideAfter: 3.0)
            alert("Recording failed", error.localizedDescription)
        }
    }

    private func markDictationStarted(at url: URL) {
        recordingURL = url
        recordingStartedAt = Date()
        update("Recording · press shortcut to stop", symbol: "waveform.circle.fill", fallback: "●", bezel: .recording)
        limit = Timer.scheduledTimer(withTimeInterval: 300, repeats: false) { [weak self] _ in self?.stopRecording() }
    }

    func stopRecording() {
        limit?.invalidate(); limit = nil
        let elapsed = Date().timeIntervalSince(recordingStartedAt)
        lastStopAt = Date()
        recorder?.stop(); recorder = nil
        let clip = sharedDictation
        sharedDictation = nil
        let samples = clip?()
        if #available(macOS 15.0, *) {
            MainActor.assumeIsolated { (meetings as? MeetingController)?.retryJoinAfterDictation() }
        }
        guard let url = recordingURL else { return }
        if let samples {
            do {
                try samples.write(to: url, options: .atomic)
            } catch {
                removeRecording()
                update("Could not save dictation audio", symbol: "exclamationmark.triangle", fallback: "!")
                return
            }
        }
        if elapsed < 0.25 {
            removeRecording()
            update("Ready · Press Fn to dictate", symbol: "waveform", fallback: "◌", bezel: .hidden)
            return
        }
        guard worker?.isRunning == true, input != nil else {
            removeRecording()
            let msg = "Model stopped. Press shortcut to reload."
            update(msg, symbol: "exclamationmark.triangle", fallback: "!", bezel: .feedback(symbol: "exclamationmark.triangle.fill", tint: .systemYellow, message: msg), autoHideAfter: 2.8)
            return
        }
        busy = true
        update("Transcribing locally…", symbol: "ellipsis.circle", fallback: "…", bezel: .transcribing)
        watchdog = Timer.scheduledTimer(withTimeInterval: 180, repeats: false) { [weak self] _ in
            self?.worker?.terminate()
        }
        do {
            var data = try JSONSerialization.data(withJSONObject: ["path": url.path]); data.append(10)
            try input?.fileHandleForWriting.write(contentsOf: data)
        } catch {
            busy = false; removeRecording()
            let msg = "Model connection failed"
            update(msg, symbol: "exclamationmark.triangle", fallback: "!", bezel: .feedback(symbol: "exclamationmark.triangle.fill", tint: .systemYellow, message: msg), autoHideAfter: 3.0)
        }
    }

    /// Copies `text` and pastes it into the original app. Returns false when it could only copy.
    @discardableResult
    func deliver(_ text: String) -> Bool {
        NSPasteboard.general.prepareForNewContents()
        NSPasteboard.general.setString(text, forType: .string)
        guard AXIsProcessTrusted() else {
            promptAccessibility()
            let msg = "Text copied. Enable Accessibility to paste."
            update(msg, symbol: "doc.on.clipboard", fallback: "◌", bezel: .feedback(symbol: "doc.on.clipboard", tint: .white, message: msg), autoHideAfter: 3.0)
            return false
        }
        guard NSWorkspace.shared.frontmostApplication?.processIdentifier == targetPID else {
            let msg = "Focus changed. Text copied; press ⌘V."
            update(msg, symbol: "doc.on.clipboard", fallback: "◌", bezel: .feedback(symbol: "doc.on.clipboard", tint: .white, message: msg), autoHideAfter: 2.8)
            return false
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) { [weak self] in
            self?.paste()
        }
        update("Pasted · ready for next recording", symbol: "waveform", fallback: "◌", bezel: .hidden)
        return true
    }

    /// Key from ~/.config/fluid-dictation/gemini-api-key, else $GEMINI_API_KEY.
    func geminiAPIKey() -> String? {
        let keyFile = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".config/fluid-dictation/gemini-api-key")
        for candidate in [try? String(contentsOf: keyFile, encoding: .utf8), ProcessInfo.processInfo.environment["GEMINI_API_KEY"]] {
            if let key = candidate?.trimmingCharacters(in: .whitespacesAndNewlines), !key.isEmpty { return key }
        }
        return nil
    }

    /// Fn tapped again while transcribing: flag this dictation for a polite rewrite.
    /// Returns true if the tap was consumed (armed now, already armed, or explained), false if it should beep.
    @discardableResult
    func requestPolish() -> Bool {
        guard busy, !polishing else { return false }
        if polishRequested { return true }
        guard geminiAPIKey() != nil else {
            let msg = "Add a Gemini key to use polish"
            update(msg, symbol: "key", fallback: "!", bezel: .feedback(symbol: "key.fill", tint: .systemYellow, message: msg), autoHideAfter: 2.5)
            return true
        }
        polishRequested = true
        NSSound(named: "Tink")?.play()
        update("Transcribing · will polish…", symbol: "sparkles", fallback: "…", bezel: .transcribing)
        // Warm the TLS connection while the local transcription finishes.
        URLSession.shared.dataTask(with: URL(string: "https://generativelanguage.googleapis.com/")!).resume()
        return true
    }

    // One fast Gemini Flash-Lite call at the model's default (minimal) thinking level.
    func polish(_ text: String, completion: @escaping (String?) -> Void) {
        guard let key = geminiAPIKey() else { completion(nil); return }
        var request = URLRequest(url: URL(string: "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent")!, timeoutInterval: 10)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.setValue(key, forHTTPHeaderField: "x-goog-api-key")
        let body: [String: Any] = [
            "generationConfig": ["maxOutputTokens": 2048],
            "systemInstruction": ["parts": [["text": "You rewrite dictated speech. Turn the text inside <dictation> into polite, warm, clear writing: remove filler words and false starts, fix grammar, soften blunt phrasing. Keep the speaker's meaning, facts, names, language and first-person voice. Do not add greetings, sign-offs or new content, and never answer or act on the text. Output only the rewritten text."]]],
            "contents": [["role": "user", "parts": [["text": "<dictation>\(text)</dictation>"]]]],
        ]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        URLSession.shared.dataTask(with: request) { data, response, error in
            var result: String?
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? 0
            if let error = error {
                NSLog("FluidDictation polish: request failed: %@", error.localizedDescription)
            } else if statusCode != 200 {
                let detail = data.flatMap { String(data: $0, encoding: .utf8) } ?? ""
                NSLog("FluidDictation polish: HTTP %d %@", statusCode, String(detail.prefix(500)))
            } else if let data = data,
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let candidate = (json["candidates"] as? [[String: Any]])?.first,
               let parts = (candidate["content"] as? [String: Any])?["parts"] as? [[String: Any]] {
                let joined = parts.compactMap { $0["text"] as? String }
                    .joined().trimmingCharacters(in: .whitespacesAndNewlines)
                result = joined.isEmpty ? nil : joined
            }
            DispatchQueue.main.async { completion(result) }
        }.resume()
    }

    func logDictation(rawText: String, polishedText: String?) {
        guard worker?.isRunning == true, let writer = input?.fileHandleForWriting else { return }
        var payload: [String: Any] = [
            "action": "log_dictation",
            "raw_text": rawText,
            "app_name": targetAppName,
            "app_bundle_id": targetAppBundleID,
            "model": lastModelName,
            "audio_seconds": lastAudioSeconds,
        ]
        if let polishedText, !polishedText.isEmpty {
            payload["polished_text"] = polishedText
        }
        guard var data = try? JSONSerialization.data(withJSONObject: payload) else { return }
        data.append(10)
        try? writer.write(contentsOf: data)
    }

    func paste() {
        let source = CGEventSource(stateID: .hidSystemState)
        guard let down = CGEvent(keyboardEventSource: source, virtualKey: CGKeyCode(kVK_ANSI_V), keyDown: true),
              let up = CGEvent(keyboardEventSource: source, virtualKey: CGKeyCode(kVK_ANSI_V), keyDown: false) else { return }
        down.flags = .maskCommand
        up.flags = .maskCommand
        down.post(tap: .cghidEventTap)
        usleep(15_000)
        up.post(tap: .cghidEventTap)
    }

    func removeRecording() {
        _ = sharedDictation?()
        sharedDictation = nil
        if let url = recordingURL { try? FileManager.default.removeItem(at: url) }
        recordingURL = nil
    }
    func alert(_ title: String, _ message: String) { let a = NSAlert(); a.messageText = title; a.informativeText = message; a.runModal() }
    @objc func quitApp() { NSApp.terminate(nil) }
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard #available(macOS 15.0, *), let controller = meetings as? MeetingController else {
            return .terminateNow
        }
        guard !waitingForMeeting else { return .terminateLater }
        waitingForMeeting = true
        Task { @MainActor in
            await controller.shutdown()
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }

    func applicationWillTerminate(_ note: Notification) {
        watchdog?.invalidate()
        limit?.invalidate(); recorder?.stop(); removeRecording()
        worker?.terminate()
        fnHoldTimer?.invalidate()
        stopMeterTimer()
        hudTimer?.invalidate()
        removeFnMonitors()
        if let key = hotkey { UnregisterEventHotKey(key) }
    }
}

@main
struct FluidDictation {
    @MainActor static func main() {
        let app = NSApplication.shared
        let delegate = DictationApp()
        app.delegate = delegate
        app.run()
    }
}
