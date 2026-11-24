#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
DIST_DIR="$ROOT/dist"
DMG_PATH="$DIST_DIR/Fluid-Dictation.dmg"
STAGE="$(mktemp -d -t fluid-dmg-stage)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$DIST_DIR" "$STAGE/payload/hotkey"

echo "Building portable Fluid Dictation.app..."
"$ROOT/hotkey/build.sh" "$STAGE/Fluid Dictation.app"

echo "Staging payload..."
cp "$ROOT/setup.sh" "$STAGE/setup.sh"
chmod +x "$STAGE/setup.sh"

cat > "$STAGE/Install Fluid Dictation.command" <<'CMD'
#!/bin/zsh
set -euo pipefail
cd "${0:A:h}"
./setup.sh
echo ""
read -k 1 "?Press any key to close..."
CMD
chmod +x "$STAGE/Install Fluid Dictation.command"

cp "$ROOT/fluid_speech_mlx.py" "$STAGE/payload/"
cp "$ROOT/transcribe.py" "$STAGE/payload/"
cp "$ROOT/transcribe" "$STAGE/payload/"
chmod +x "$STAGE/payload/transcribe"
cp "$ROOT/requirements.lock.txt" "$STAGE/payload/"
cp "$ROOT/requirements-rumble.lock.txt" "$STAGE/payload/"
cp "$ROOT/SHA256SUMS" "$STAGE/payload/"
cp "$ROOT/MODEL_CARD.md" "$STAGE/payload/"
cp "$ROOT/README.md" "$STAGE/payload/"
cp "$ROOT/LICENSE" "$STAGE/payload/"
cp "$ROOT/LICENSE" "$STAGE/LICENSE"

cp "$ROOT/hotkey/config.json" "$STAGE/payload/hotkey/"
cp "$ROOT/hotkey/worker.py" "$STAGE/payload/hotkey/"
cp "$ROOT/hotkey/inference.py" "$STAGE/payload/hotkey/"
cp "$ROOT/hotkey/vocabulary_boost.py" "$STAGE/payload/hotkey/"
cp "$ROOT/hotkey/spoken_punctuation.py" "$STAGE/payload/hotkey/"
cp "$ROOT/hotkey/FluidDictation.swift" "$STAGE/payload/hotkey/"
cp "$ROOT/hotkey/build.sh" "$STAGE/payload/hotkey/"
chmod +x "$STAGE/payload/hotkey/build.sh"

cat > "$STAGE/README.txt" <<'TXT'
Fluid Dictation (Local MLX Speech-to-Text for Apple Silicon)

One-step install:
  Double-click "Install Fluid Dictation.command"
  (or run ./setup.sh in Terminal)

What setup.sh does automatically:
  1. Installs uv & ffmpeg if missing
  2. Creates isolated Python 3.12 environments in ~/Library/Application Support/FluidDictation
  3. Downloads local 8-bit MLX speech models (Rumble-2-Flash & Fluid Speech Pro)
  4. Installs Fluid Dictation.app to ~/Applications/Fluid Dictation.app
  5. Disables the macOS Fn Emoji picker and enables launch at login
  6. Launches the app and prompts for Microphone + Accessibility permissions

Controls:
  - Tap Fn once to start/stop, or hold Fn for push-to-talk
  - Backup shortcut: Control + Option + Space
TXT

xattr -cr "$STAGE" 2>/dev/null || true

echo "Creating $DMG_PATH..."
rm -f "$DMG_PATH"
/usr/bin/hdiutil create -volname "Fluid Dictation" -srcfolder "$STAGE" -ov -format UDZO "$DMG_PATH"

echo "Built: $DMG_PATH"
ls -lh "$DMG_PATH"
