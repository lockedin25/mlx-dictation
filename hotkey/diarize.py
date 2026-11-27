"""Isolated NVIDIA Nemotron-3-Diarization worker.

Runs as a short-lived subprocess in `.venv-rumble` so model weights and Metal
caches are completely released back to macOS on exit before ASR starts.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time
import wave

HOTKEY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = HOTKEY_DIR.parent
DEFAULT_MODEL_PATH = "models/nemotron-3-diarization"
DEFAULT_HF_REPO = "mlx-community/Nemotron-3-Diarization"


def load_diarization_config() -> dict:
    cfg_path = Path(os.environ.get("FLUID_CONFIG_PATH", HOTKEY_DIR / "config.json"))
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        section = data.get("diarization", {})
        return section if isinstance(section, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def resolve_model_path(explicit: str | None = None) -> str:
    cfg = load_diarization_config()
    raw = (
        explicit
        or os.environ.get("FLUID_DIARIZATION_MODEL")
        or cfg.get("model_path")
        or DEFAULT_MODEL_PATH
    )
    candidate = Path(str(raw)).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if candidate.is_dir() and (candidate / "config.json").is_file():
        return str(candidate)
    if Path(str(raw)).is_dir():
        return str(raw)
    if str(raw) == DEFAULT_MODEL_PATH:
        return DEFAULT_HF_REPO
    return str(raw)


def inspect_wav(audio_path: Path) -> tuple[float, float]:
    """Return (duration_seconds, rms) for a WAV file."""
    import numpy as np

    with contextlib.closing(wave.open(str(audio_path), "rb")) as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.getnframes()
        if rate <= 0 or frames <= 0:
            return 0.0, 0.0
        duration = frames / float(rate)
        if width != 2:
            return duration, 1.0
        raw = handle.readframes(frames)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1 and samples.size:
        samples = samples.reshape(-1, channels).mean(axis=1)
    rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
    return duration, rms


def diarize_file(
    audio_path: Path,
    *,
    model_path: str,
    threshold: float = 0.5,
    min_duration: float = 0.3,
    merge_gap: float = 0.5,
    streaming_preset: str = "offline",
    cache_limit_mb: int = 128,
) -> dict:
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    duration, rms = inspect_wav(audio_path)
    if duration < 0.3 or rms < 0.0005:
        return {
            "status": "ok",
            "model": DEFAULT_HF_REPO,
            "model_path": model_path,
            "audio_seconds": round(duration, 3),
            "num_speakers": 0,
            "elapsed_seconds": 0.0,
            "segments": [],
        }

    with contextlib.redirect_stdout(sys.stderr):
        import mlx.core as mx
        from mlx_audio.vad import load as load_vad

        model = load_vad(model_path)
        if streaming_preset and hasattr(model, "set_streaming_config"):
            model.set_streaming_config(streaming_preset)
        mx.eval(model.parameters())
        mx.set_cache_limit(int(cache_limit_mb) * 1024 * 1024)

        started = time.perf_counter()
        result = model.generate(
            str(audio_path),
            sample_rate=16000,
            threshold=threshold,
            min_duration=min_duration,
            merge_gap=merge_gap,
        )
        elapsed = time.perf_counter() - started

    segments = []
    for seg in getattr(result, "segments", []) or []:
        start_ms = max(0, int(round(float(seg.start) * 1000)))
        end_ms = max(start_ms, int(round(float(seg.end) * 1000)))
        spk_idx = int(seg.speaker)
        segments.append({
            "start_ms": start_ms,
            "end_ms": end_ms,
            "speaker": f"REMOTE_{spk_idx}",
            "speaker_index": spk_idx,
        })
    segments.sort(key=lambda item: (item["start_ms"], item["end_ms"], item["speaker_index"]))
    speakers = {item["speaker"] for item in segments}
    return {
        "status": "ok",
        "model": DEFAULT_HF_REPO,
        "model_path": model_path,
        "audio_seconds": round(duration, 3),
        "num_speakers": len(speakers),
        "elapsed_seconds": round(elapsed, 3),
        "segments": segments,
    }


def main(argv: list[str] | None = None) -> int:
    cfg = load_diarization_config()
    parser = argparse.ArgumentParser(description="Diarize a WAV file using Nemotron-3-Diarization.")
    parser.add_argument("audio_wav", type=Path)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--threshold", type=float, default=float(cfg.get("threshold", 0.5)))
    parser.add_argument("--min-duration", type=float, default=float(cfg.get("min_duration", 0.3)))
    parser.add_argument("--merge-gap", type=float, default=float(cfg.get("merge_gap", 0.5)))
    parser.add_argument("--preset", default=str(cfg.get("streaming_preset", "offline")))
    parser.add_argument("--cache-limit-mb", type=int, default=int(cfg.get("cache_limit_mb", 128)))
    args = parser.parse_args(argv)

    try:
        resolved = resolve_model_path(args.model_path)
        payload = diarize_file(
            args.audio_wav.expanduser().resolve(),
            model_path=resolved,
            threshold=args.threshold,
            min_duration=args.min_duration,
            merge_gap=args.merge_gap,
            streaming_preset=args.preset,
            cache_limit_mb=args.cache_limit_mb,
        )
        print(json.dumps(payload), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "error", "message": str(exc)}), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
