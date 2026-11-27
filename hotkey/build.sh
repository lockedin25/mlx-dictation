#!/bin/zsh
set -eu
ROOT="${0:A:h:h}"
APP="${1:-$HOME/Applications/Fluid Dictation.app}"
mkdir -p "$APP/Contents/MacOS"
/usr/bin/swiftc -parse-as-library -O \
  "$ROOT/hotkey/FluidDictation.swift" \
  "$ROOT/hotkey/MeetingController.swift" \
  "$ROOT/hotkey/MeetingCapture.swift" \
  "$ROOT/hotkey/ChromeMeetDetector.swift" \
  -o "$APP/Contents/MacOS/FluidDictation" \
  -framework AppKit -framework AVFoundation -framework Carbon -framework ApplicationServices -framework CoreAudio -framework ScreenCaptureKit
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>local.shubham.fluid-dictation</string>
<key>CFBundleName</key><string>Fluid Dictation</string>
<key>CFBundleExecutable</key><string>FluidDictation</string>
<key>CFBundleVersion</key><string>1</string>
<key>CFBundleShortVersionString</key><string>1.0</string>
<key>LSUIElement</key><true/>
<key>NSMicrophoneUsageDescription</key><string>Record dictation and Google Meet or Discord calls for local transcription.</string>
<key>NSAudioCaptureUsageDescription</key><string>Capture Google Meet and Discord call audio for local meeting transcription.</string>
</dict></plist>
PLIST

SIGN_ID="${FLUID_CODESIGN_IDENTITY:-}"
if [[ -z "$SIGN_ID" ]] && security find-identity -v -p codesigning 2>/dev/null | grep -q "Fluid Dictation Local"; then
  SIGN_ID="Fluid Dictation Local"
fi
if [[ -n "$SIGN_ID" ]]; then
  /usr/bin/codesign --force --sign "$SIGN_ID" --identifier local.shubham.fluid-dictation "$APP"
else
  /usr/bin/codesign --force --sign - --identifier local.shubham.fluid-dictation "$APP"
fi
