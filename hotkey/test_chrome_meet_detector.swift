import Foundation

func expect(_ condition: Bool, _ message: String) {
    if !condition {
        fputs("FAIL: \(message)\n", stderr)
        exit(1)
    }
}

@main
struct ChromeMeetDetectorTests {
    static func main() {
        // 1. Tab title parsing
        if let m1 = parseMeetTabTitle("Meet - abc-defg-hij - Google Chrome - Audio playing") {
            expect(m1.title == "abc-defg-hij", "m1 title: \(m1.title)")
            expect(m1.meetCode == "abc-defg-hij", "m1 code: \(String(describing: m1.meetCode))")
            expect(m1.hasMediaBadge, "m1 hasMediaBadge")
        } else {
            expect(false, "m1 should parse")
        }

        if let m2 = parseMeetTabTitle("Meet – Weekly Design Sync - Tab is using your camera or microphone") {
            expect(m2.title == "Weekly Design Sync", "m2 title: \(m2.title)")
            expect(m2.meetCode == nil, "m2 meetCode should be nil")
            expect(m2.hasMediaBadge, "m2 hasMediaBadge")
        } else {
            expect(false, "m2 should parse")
        }

        if let m3 = parseMeetTabTitle("Meet — xyz-abcd-efg") {
            expect(m3.meetCode == "xyz-abcd-efg", "m3 code")
            expect(!m3.hasMediaBadge, "m3 no media badge")
        } else {
            expect(false, "m3 should parse")
        }

        if let m4 = parseMeetTabTitle("Meet – epf-ypvk-oax – Camera and microphone recording - Google Chrome – Shubham (joinef.com)") {
            expect(m4.title == "epf-ypvk-oax", "m4 title: \(m4.title)")
            expect(m4.meetCode == "epf-ypvk-oax", "m4 code")
            expect(m4.hasMediaBadge, "m4 hasMediaBadge")
        } else {
            expect(false, "m4 should parse")
        }

        if let m5 = parseMeetTabTitle("Meet - Weekly Sync – Microphone recording - Google Chrome – Work") {
            expect(m5.title == "Weekly Sync", "m5 title: \(m5.title)")
            expect(m5.meetCode == nil, "m5 meetCode nil")
            expect(m5.hasMediaBadge, "m5 hasMediaBadge")
        } else {
            expect(false, "m5 should parse")
        }

        expect(parseMeetTabTitle("Google Meet - Google Chrome – Shubham") == nil, "landing page with profile must not match")
        expect(parseMeetTabTitle("Meet - Google Chrome – Shubham") == nil, "empty Meet with profile must not match")

        expect(parseMeetTabTitle("Google Meet - Google Chrome") == nil, "landing page must not match")
        expect(parseMeetTabTitle("Google Meet") == nil, "plain Google Meet must not match")
        expect(parseMeetTabTitle("Meet the Team - Company Blog - Google Chrome") == nil, "blog title must not match")
        expect(parseMeetTabTitle("Inbox (3) - Gmail - Google Chrome") == nil, "gmail must not match")
        expect(parseMeetTabTitle("Meet - Google Chrome") == nil, "empty Meet must not match")

        // 2. Hysteresis state machine
        var sm = ChromeMeetStateMachine(startDebounceSeconds: 5.0, endGraceSeconds: 12.0)
        let t0 = Date(timeIntervalSince1970: 1000)
        let callA = MeetTabMatch(title: "aaa-bbbb-ccc", meetCode: "aaa-bbbb-ccc", hasMediaBadge: true)
        let callB = MeetTabMatch(title: "ddd-eeee-fff", meetCode: "ddd-eeee-fff", hasMediaBadge: true)

        // Brief 3s pre-join flicker should NOT start
        expect(sm.update(match: callA, isCallActive: true, now: t0).isEmpty, "t0 no start")
        expect(sm.update(match: callA, isCallActive: true, now: t0.addingTimeInterval(3)).isEmpty, "t+3 no start")
        expect(sm.update(match: callA, isCallActive: false, now: t0.addingTimeInterval(4)).isEmpty, "t+4 reset")

        // Sustained >= 5s DOES start
        expect(sm.update(match: callA, isCallActive: true, now: t0.addingTimeInterval(10)).isEmpty, "t+10 start pending")
        let joinEvents = sm.update(match: callA, isCallActive: true, now: t0.addingTimeInterval(15))
        expect(joinEvents == [.joined(callA)], "t+15 should join: \(joinEvents)")

        // Brief 8s interruption stays in call (< 12s grace)
        expect(sm.update(match: nil, isCallActive: false, now: t0.addingTimeInterval(20)).isEmpty, "t+20 grace starts")
        expect(sm.update(match: nil, isCallActive: false, now: t0.addingTimeInterval(28)).isEmpty, "t+28 still in grace")
        expect(sm.update(match: callA, isCallActive: true, now: t0.addingTimeInterval(29)).isEmpty, "t+29 recovered")

        // Switch to different call B sustained for 5s emits left then joined(B)
        expect(sm.update(match: callB, isCallActive: true, now: t0.addingTimeInterval(40)).isEmpty, "t+40 switch pending")
        let switchEvents = sm.update(match: callB, isCallActive: true, now: t0.addingTimeInterval(45))
        expect(switchEvents == [.left, .joined(callB)], "t+45 should switch: \(switchEvents)")

        // Leaving for >= 12s emits left
        expect(sm.update(match: nil, isCallActive: false, now: t0.addingTimeInterval(50)).isEmpty, "t+50 grace begins")
        expect(sm.update(match: nil, isCallActive: false, now: t0.addingTimeInterval(61)).isEmpty, "t+61 < 12s")
        let leaveEvents = sm.update(match: nil, isCallActive: false, now: t0.addingTimeInterval(62))
        expect(leaveEvents == [.left], "t+62 should leave: \(leaveEvents)")

        print("PASS: ChromeMeetDetector parser and state machine tests")
    }
}
