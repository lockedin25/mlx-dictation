# Fluid Dictation hotkey

Installed app: `/Users/shubhamkorde/Applications/Fluid Dictation.app`.

Click in an editable text field, press **Control + Option + Space**, speak, then press the same shortcut again. A floating banner shows recording and transcription status. The menu-bar symbol changes to a filled circle while recording and an ellipsis while transcribing. The transcript is copied to the clipboard and pasted into the focused app. It never presses Return or submits a message.

The app starts with the model unloaded. Recording begins immediately when you press the shortcut; after you stop recording, the worker loads the model if needed, transcribes and pastes. It keeps the model for **10 seconds after the last completed transcription**, then exits the model process to release its weights and GPU caches. A transcription within the idle window reuses the loaded model. “Ready” means the helper is ready to accept audio; it does not mean model weights are resident. The worker uses the downloaded 8-bit model with a conservative 186-term vocabulary boost (beam size 2), preserving fillers rather than applying the clean-mode adapter. Recordings stop automatically after five minutes. Microphone audio is written to a temporary 16 kHz mono 16-bit WAV on disk, about 32 KB per second or 9.6 MB for five minutes. It is removed after success or failure and on normal app quit. A hard crash or force quit can leave a temporary file behind; the app does not maintain an audio archive. Transcripts stay in the system clipboard and destination app; this helper does not save them to a transcript log.

If the foreground app changes during transcription, text is copied but not pasted. Press Command + V yourself in the intended field. Protected password fields and apps that block simulated paste may require manual paste.

## Idle timeout and memory

Change `idle_unload_seconds` in `hotkey/config.json` to any number from 0 to 3600. The coordinator rereads this file while idle, so no restart or rebuild is required. Default: 10 seconds. Zero unloads immediately after transcription. Invalid settings fall back to 10 seconds.

The timeout counts inactivity between transcription requests, not ordinary keyboard typing. It does not discard microphone audio. If the model unloads while you are recording, it reloads after you stop and before transcribing.

Measured idle footprint is about 23 MB for the native helper plus lightweight Python coordinator. Before vocabulary boosting, a five-minute repeated speech sample took 4.33 seconds including model process startup and transcription, with an observed peak model footprint of 1610 MB using 30-second chunks, 5-second overlap and a 128 MB MLX cache cap. A pair of shorter samples took 0.66 seconds for the first request and 0.18 seconds warm; reload took 0.49 seconds. These runs benefit from existing filesystem/Metal caches; latency after reboot and real audio memory needs may differ. There is no hard five-second latency guarantee or fixed total RAM cap.

The original 120-second chunks peaked at 5128 MB on the same five-minute test, so the hotkey worker now uses the smaller chunks. The separate file-transcription command retains its original configuration.

## Permissions

Allow Microphone access and Accessibility for Fluid Dictation. On this Mac's macOS version, the Accessibility permission appears under **System Settings → Privacy & Security → Device Control and Data Access**. This system permission is broad; the helper uses it to post Command + V. The default Carbon hotkey does not require Input Monitoring. Fn is optional and disabled because Wispr Flow already uses it.

The menu item “Enable microphone and paste permissions…” requests these permissions again. After a permission change, quit and reopen Fluid Dictation if needed. If the switch is on but `status.json` still reports `accessibilityGranted: false`, remove Fluid Dictation with the minus button, add `~/Applications/Fluid Dictation.app` again with the plus button, and reopen it. This resolved a stale permission entry after the app was rebuilt during setup.

## Startup and controls

The app starts at login using `~/Library/LaunchAgents/local.shubham.fluid-dictation.plist`. Quit it from its menu to release model memory. Reopen the app from `~/Applications` to restart.

To disable launch at login:

```bash
launchctl bootout "gui/$(id -u)/local.shubham.fluid-dictation"
mv ~/Library/LaunchAgents/local.shubham.fluid-dictation.plist ~/Library/LaunchAgents/local.shubham.fluid-dictation.plist.disabled
```

The optional “Also enable Fn / Globe tap” menu item uses tap once to start and tap again to stop. Disable the Fn shortcut in Wispr Flow and set Keyboard → Press Globe key to → Do Nothing before using that option. Fn combinations with other keys are ignored. The configured default remains Control + Option + Space.

## Development and diagnostics

- `FluidDictation.swift`: native AppKit menu-bar app, microphone recording, global shortcut and paste.
- `worker.py`: lightweight coordinator with configurable idle timeout, using private stdin/stdout pipes. No network listener.
- `inference.py`: MLX model child, started when a transcription arrives and terminated on idle.
- `config.json`: live idle timeout.
- `model-state.json`: actual model residence/PID and current timeout; contains no transcript.
- `test_idle_lifecycle.py` and `test_long_audio.py`: lifecycle and long-audio integration checks.
- `build.sh`: compile, install and ad-hoc sign the app. The project path is embedded in the Swift source; update it before moving the project.
- `status.json`: current readiness, shortcut registration and permission flags. No audio or transcript content. It updates when app state changes, so permission flags may be stale until the next recording or restart.
- `~/Library/Logs/FluidDictation-worker.log`: worker diagnostics.

```bash
cd /Users/shubhamkorde/Desktop/Projects/fluid-speech-pro
./hotkey/build.sh
open "$HOME/Applications/Fluid Dictation.app"
cat hotkey/status.json
```

Quit the running app before rebuilding. Rebuilds change the ad-hoc signature and may require reauthorizing the app in Privacy & Security. It uses the installed project virtual environment and ffmpeg at `/opt/homebrew/bin`, so keep those files in place.

## Verification (2026-11-15)

The installed app registered Control + Option + Space successfully. Microphone and Accessibility/app-control permissions were confirmed from inside the running app. A real 3.89-second microphone recording completed transcription and paste; Shubham confirmed that the text appeared correctly in the focused field. The persistent worker also passed two sequential sample transcriptions, rejected a silent recording and a missing file, and successfully transcribed another sample after those errors. App signature and both app/startup plist files passed validation.

Idle-update checks passed reuse, configurable timeout, model-process exit, reload, recovery after file errors and shutdown cleanup. `idle-verification.json` and `long-audio-verification.json` contain measurements. The signed native app was not rebuilt for this update.


## Vocabulary boosting

Enabled on 2026-11-15 using 186 terms from the wiki/Wispr starter pack. Fourteen common words/phrases such as Clean, Circle, Form and Launch are omitted to reduce unnecessary bias. Names receive score bonuses of 1.0, most longer terms 0.8, and short acronyms 0.35. Matching ignores letter case. No text-replacement rules or pronunciation guesses are applied.

Settings are in `config.json`, under `vocabulary_boost`; weights are in `../vocabulary/active-boost.json`. Set `enabled` to false to restore the original greedy decoder. Settings and weights are reread on each transcription, even while the model remains loaded. No app rebuild or permission refresh is required. Invalid configuration falls back to greedy decoding and records a diagnostic warning.

Controlled tests fixed “Rattle” to “Rahul” and “Rahul Samad” to “Rahul Samat”. The recorded speech control and a synthetic common-word control stayed unchanged. Several names/terms, including Qwen and LoRA in the synthetic vocabulary clip, were still misheard; modest boosting does not force a match. See `../results/vocabulary-boost-verification.json` for complete outputs and timings. The native shortcut/paste app, 10-second idle timeout and audio cleanup behavior are unchanged.

With vocabulary boosting enabled, the same five-minute sample took 7.19 seconds including model startup, with an observed peak model footprint of 1657 MB. The lifecycle checks passed with boosting active. This is a controlled sample measurement, not a latency guarantee for arbitrary audio.

## Discord meeting notes

On macOS 15 or later, use **Record Discord meeting…** from the Fluid Dictation menu for a private Discord call. Stop it from the same menu. Grant microphone and audio capture access on the first test call. The red REC indicator stays visible while capture runs. The recorder reads the Mac microphone directly, so use the Fluid Dictation stop control to end local microphone capture even if Discord is muted. Confirm the meeting participants agree to recording before using either mode.

To enable automatic recording in a server voice channel, save the following as `~/.config/fluid-dictation/meetings.json`. Copy the verified IDs from Discord Developer Mode. Add `peer_id` only when recording should require a specific second participant. Leave `auto_record` false until the IDs and the target call have been tested.

```json
{
  "auto_record": false,
  "guild_id": "REPLACE_WITH_SERVER_ID",
  "user_id": "REPLACE_WITH_YOUR_DISCORD_ID",
  "channel_id": "REPLACE_WITH_THE_VOICE_CHANNEL_ID",
  "vault_raw_dir": "~/Projects/knowledge-and-wikis/Edge Inference Thesis/Edge Inference Vault/Meetings/Raw",
  "summary_model": "openai/gpt-4.1-mini"
}
```

The detector needs a bot with permission to connect to the selected channel. Supply its token in the app environment as `DISCORD_TOKEN` or in the private file `~/.config/fluid-dictation/discord-bot-token` with owner-only permissions. Select **Reload meeting settings** after editing. Omitting `vault_raw_dir` keeps the notes in the local meeting folder only.

Rumble runs locally as audio chunks close. OpenRouter receives the original mixed-language transcript for classification and notes using `~/.config/openrouter/credentials.env`; no translated transcript is generated. Audio remains in `~/Library/Application Support/Fluid Dictation/meetings/`. **Open meeting notes** opens `~/Documents/Meeting Notes/index.md`. To rerun a failed summary or vault write, select **Retry unfinished meeting notes**. The shared thesis's `/ingest` command handles the later review, source registration, commit and push.
