# Fluid Dictation (Local MLX Speech-to-Text on Apple Silicon)

A lightweight, model-agnostic macOS menu-bar dictation app running local 8-bit MLX speech models on Apple Silicon. Press `Fn` anywhere on your Mac, speak, and the transcript gets pasted into whatever field has focus. Everything runs locally on-device, and the model unloads from RAM 10 seconds after you finish speaking.

Supports switching between **[Rumble-2-Flash (8-bit MLX)](https://huggingface.co/TypeWhisper/Rumble-2-Flash-8bit)** (default, multilingual including English and Hindi/Hinglish) and **[Fluid Speech Pro MLX](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro-mlx)** (Parakeet-based, with beam-search vocabulary boosting and `--clean` mode) via a single `"model"` setting.

![RAM Footprint Comparison](assets/memory-savings.svg)

## Switching models

Change `"model"` in [`hotkey/config.json`](hotkey/config.json) (or pass `--model` to `./transcribe`). The worker detects the change on the next dictation and reloads automatically—no rebuild or restart required:

```json
{
  "model": "rumble-2-flash",
  "idle_unload_seconds": 10
}
```

| `"model"` value | Upstream checkpoint | Highlights | License |
|---|---|---|---|
| `"rumble-2-flash"` *(default)* | [`TypeWhisper/Rumble-2-Flash-8bit`](https://huggingface.co/TypeWhisper/Rumble-2-Flash-8bit) (`Reson8/Rumble-2-Flash`) | 0.5B Qwen3-ASR (~725 MB), 28 languages (including English & Hindi/Hinglish), `<digits><nofiller><verbatim>` prompt control | Apache 2.0 |
| `"fluid-speech"` | [`FluidVoice-Labs/fluid-speech-pro-mlx`](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro-mlx) | 0.6B Parakeet (~751 MB), English + 24 European languages, `parakeet-mlx` beam-search vocabulary boost, `--clean` adapter | CC BY 4.0 |

## Why I built this

### 1. Almost zero RAM when idle
Most dictation apps keep large runtimes sitting in memory all day even when you are not speaking. On an Apple M5 (16 GB) measured with macOS `footprint`:

| App / State | Memory Footprint | Notes |
|---|---:|---|
| Fluid Dictation (idle, model unloaded) | **31 MB** | 21 MB Swift app + 10 MB Python coordinator, 0 MB weights |
| Wispr Flow (idle, not recording) | **1,081 MB** | 13 processes (585 MB Electron helpers + 496 MB renderers) |
| Fluid Dictation (10s dictation peak, Fluid Speech Pro) | **1,286 MB** | 1,255 MB 8-bit MLX peak (0.92s cold, 0.78s warm), then drops back to 31 MB |
| Fluid Dictation (5-minute stress peak, Fluid Speech Pro) | **1,688 MB** | 30s chunks with a 128 MB MLX cache cap |

After 10 idle seconds, the worker kills the model child process and hands all model memory back to macOS.

### 2. Fast, accurate daily transcription
With 8-bit MLX quantization (`~700–750 MB` on disk), both Rumble-2-Flash and Fluid Speech Pro run locally on Apple Silicon in a fraction of a second for typical dictations.

### 3. No extra LLM rewrite layer when talking to LLMs
If your main use case is dictating prompts into Claude Code, Cursor, Codex, or ChatGPT, an LLM is already about to read what you said. Running a cloud post-processing LLM inside your dictation app first adds a second network hop, slows things down, and sometimes rewrites technical terms before your actual coding agent sees them. Fluid Dictation drops the raw transcript straight into the text box (with optional opt-in polish via double-tap `Fn` when you actually want it).

![Built for LLM Workflows and Custom Vocabulary](assets/architecture-and-control.svg)

### 4. Full control over your vocabulary
With `"model": "fluid-speech"`, you can bias recognition through a plain JSON file (`vocabulary/active-boost.json`) backed by `parakeet-mlx` beam search:

```json
{
  "weights": {
    "Claude Code": 0.9,
    "Parakeet": 0.8,
    "MLX": 0.6,
    "API": 0.35
  }
}
```

You can pick your own weights (`0.1` to `2.0`), tune `beam_size` (`1` to `5`) in `hotkey/config.json`, or disable boosting completely. The coordinator reloads these files on every dictation, so edits apply right away without rebuilding or restarting the app. (For `rumble-2-flash`, optional hotwords can also be toggled via `rumble_2_flash.use_hotwords` in `hotkey/config.json`.)

## Honest drawbacks

This setup fits a specific workflow and has real limits:

- **Raw speech by default.** Because there is no LLM cleanup pass by default, self-corrections and rough phrasing land verbatim unless you double-tap `Fn` to opt into Gemini Flash-Lite polish.
- **Model trade-offs.** `rumble-2-flash` handles 28 languages including Hindi/Hinglish well, while `fluid-speech` is tuned for English + 24 European languages and supports `parakeet-mlx` beam-search vocabulary boosting.
- **Can struggle with certain accents or rare proper nouns.** Vocabulary boosting helps on names you add, though it is a gentle bias rather than a hard override.

## Quick install (macOS, Apple Silicon)

### Option A: Download the DMG
1. Download `dist/Fluid-Dictation.dmg` from this repository and open it.
2. Double-click **Install Fluid Dictation.command**.

### Option B: Clone and run one script
```bash
git clone https://github.com/lockedin25/mlx-dictation.git
cd mlx-dictation
./setup.sh
```

Pass `--model rumble-2-flash` or `--model fluid-speech` to `./setup.sh` if you only want to download one of the two models.

### What `setup.sh` sets up automatically
1. Installs `uv` and `ffmpeg` (via Homebrew) if they are missing.
2. Creates isolated Python 3.12 environments in `~/Library/Application Support/FluidDictation` (`.venv` and `.venv-rumble`).
3. Downloads the 8-bit MLX model weights (`TypeWhisper/Rumble-2-Flash-8bit` and `FluidVoice-Labs/fluid-speech-pro-mlx`, verified against `SHA256SUMS`).
4. Installs `Fluid Dictation.app` into `~/Applications/`.
5. Sets macOS `AppleFnUsageType` to `0` so pressing `Fn` does not pop up the Emoji picker, and adds a LaunchAgent so it starts at login.
6. Opens the app and prompts for Microphone and Accessibility permissions.

## Controls

- **Tap `Fn`** once to start recording, tap `Fn` again to transcribe and paste.
- **Hold `Fn`** to record while held and paste when released.
- **Backup shortcut:** `Control + Option + Space` works at any time.
- **Double-tap `Fn` to stop** (or tap `Fn` again while it says "Transcribing") to paste a polite, cleaned-up rewrite instead of the raw transcript. Opt-in per dictation; it makes one Gemini Flash-Lite call (about 0.5 to 0.8 s) and needs a key in `~/.config/fluid-dictation/gemini-api-key` or `$GEMINI_API_KEY`. If the call fails, the raw text is pasted.

To enable vocabulary boosting after installing, create `vocabulary/active-boost.json` inside `~/Library/Application Support/FluidDictation/` (or your cloned folder) and set `"enabled": true` in `hotkey/config.json`.

## Spoken punctuation and symbols

Say a symbol's name and the symbol is pasted. This runs locally on every transcript, for both models.

| Say | Get |
|---|---|
| `src slash app slash main dot py` | `src/app/main.py` |
| `run the slash goal command` | `run the /goal command` |
| `state hyphen of hyphen the hyphen art` | `state-of-the-art` |
| `run npm dash dash version` | `run npm --version` |
| `snake underscore case` | `snake_case` |
| `hello comma how are you question mark` | `hello, how are you?` |
| `first line new line second line` | two lines |

Also: `backslash`, `forward slash`, `at sign`, `tilde`, `period` / `full stop`, `colon`, `semicolon`, `exclamation mark`, `ellipsis`, `open paren` / `close paren`, `open bracket` / `close bracket`, `open brace` / `close brace`, `hashtag`, `dollar sign`, `percent sign`, `ampersand`, `asterisk`, `plus sign`, `equals sign`, `new paragraph`. The full list is in `hotkey/spoken_punctuation.py`.

Say `literal` first to keep the word (`literal slash` gives "slash"). To switch off a phrase for good, list it in `hotkey/config.json`, for example `"spoken_punctuation": {"enabled": true, "ignore": ["period", "dash"]}`. Set `"enabled": false` to turn the feature off. Edits apply on the next dictation.

## CLI transcription

You can also transcribe audio files directly from the terminal (`HF_HUB_OFFLINE=1` after setup):

```bash
./transcribe /absolute/path/to/recording.m4a
./transcribe /absolute/path/to/recording.m4a --model rumble-2-flash
./transcribe /absolute/path/to/recording.m4a --model fluid-speech --clean
./transcribe /absolute/path/to/recording.m4a --json results/my-recording.json
```

If `--model` is omitted, `./transcribe` uses `"model"` from `hotkey/config.json`.

## Credits

- **[Reson8](https://huggingface.co/Reson8/Rumble-2-Flash) & [TypeWhisper](https://huggingface.co/TypeWhisper/Rumble-2-Flash-8bit)** for **Rumble-2-Flash** and its 8-bit MLX conversion (`mlx-audio`).
- **[FluidVoice](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro-mlx)** for packaging and open-sourcing Fluid Speech Pro MLX along with the custom loader and clean-mode adapter.

Required upstream attribution for Fluid Speech Pro:
> "Fluid Speech Pro by FluidVoice. Built on NVIDIA Parakeet TDT 0.6B v3 and Moondream Parakeet Ultra."

- [TypeWhisper/Rumble-2-Flash-8bit](https://huggingface.co/TypeWhisper/Rumble-2-Flash-8bit) / [Reson8/Rumble-2-Flash](https://huggingface.co/Reson8/Rumble-2-Flash) (Apache 2.0)
- [FluidVoice-Labs/fluid-speech-pro-mlx](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro-mlx) (CC BY 4.0)
- [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (CC BY 4.0)
- [moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra) (CC BY 4.0)

## License

- **App, worker, and setup scripts** (`hotkey/`, `setup.sh`, `transcribe`, `transcribe.py`): [MIT License](LICENSE).
- **Rumble-2-Flash 8-bit MLX weights** (downloaded at setup): **Apache License 2.0**, copyright Reson8 / TypeWhisper.
- **Fluid Speech Pro MLX model weights and `fluid_speech_mlx.py` loader**: Creative Commons Attribution 4.0 International (**CC BY 4.0**), copyright FluidVoice.
