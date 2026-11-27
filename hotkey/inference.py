"""MLX inference child; parent terminates it to release idle model memory."""
import contextlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'hotkey'))
CONFIG_PATH = Path(os.environ.get('FLUID_CONFIG_PATH', ROOT / 'hotkey/config.json'))

FLUID_ALIASES = {'fluid-speech', 'fluid_speech', 'fluid', 'fluid-speech-pro', 'fluid_speech_pro', 'parakeet'}
RUMBLE_ALIASES = {'rumble-2-flash', 'rumble_2_flash', 'rumble'}


def emit(**message):
    print(json.dumps(message), flush=True)


def read_config():
    try:
        data = json.loads(CONFIG_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def resolve_model():
    cfg = read_config()
    raw = (
        os.environ.get('FLUID_MODEL')
        or os.environ.get('FLUID_BACKEND')
        or cfg.get('model')
        or cfg.get('backend')
        or 'rumble-2-flash'
    )
    key = str(raw).strip().lower()
    if key in FLUID_ALIASES:
        return 'fluid-speech'
    if key in RUMBLE_ALIASES:
        return 'rumble-2-flash'
    raise ValueError(f'Unknown model {raw!r}. Expected "rumble-2-flash" or "fluid-speech".')


def rumble_hotwords(config):
    rumble_cfg = config.get('rumble_2_flash', {})
    if not isinstance(rumble_cfg, dict) or rumble_cfg.get('use_hotwords', False) is not True:
        return None
    boost_cfg = config.get('vocabulary_boost', {})
    if not isinstance(boost_cfg, dict) or boost_cfg.get('enabled', False) is not True:
        return None
    terms_file = ROOT / str(boost_cfg.get('terms_file', 'vocabulary/active-boost.json'))
    try:
        weights = json.loads(terms_file.read_text()).get('weights', {})
        if isinstance(weights, dict):
            return [str(k) for k in weights.keys() if isinstance(k, str) and k.strip()] or None
    except (OSError, ValueError, TypeError):
        pass
    return None


try:
    active_model = resolve_model()
    with contextlib.redirect_stdout(sys.stderr):
        import mlx.core as mx
        import numpy as np

        if active_model == 'fluid-speech':
            import soundfile as sf
            from fluid_speech_mlx import load
            from vocabulary_boost import decoding_config

            model = load(str(ROOT / 'models/fluid-speech-pro-mlx'), bits=8)
        else:
            from mlx_audio.audio_io import read as audio_read
            from mlx_audio.stt import load as load_stt

            cfg = read_config().get('rumble_2_flash', {})
            rel_model = cfg.get('model_path', 'models/rumble-2-flash-8bit') if isinstance(cfg, dict) else 'models/rumble-2-flash-8bit'
            model_path = Path(rel_model)
            if not model_path.is_absolute():
                model_path = ROOT / model_path
            model = load_stt(str(model_path))

        mx.eval(model.parameters())
        mx.set_cache_limit(128 * 1024 * 1024)

    emit(status='ready', model=active_model)

    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            audio = Path(request['path'])
            if not audio.is_file():
                raise FileNotFoundError(f'Audio file not found: {audio}')

            if active_model == 'fluid-speech':
                samples, rate = sf.read(audio)
                duration = len(samples) / rate if rate else 0.0
                rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0
            else:
                samples, rate = audio_read(str(audio), dtype='float32', sample_rate=16000, nchannels=1)
                duration = len(samples) / rate if rate else 0.0
                rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0

            if duration < 0.3 or (duration < 0.5 and rms < 0.001):
                emit(status='result', model=active_model, text='', seconds=0.0, audio_seconds=duration, rms=rms)
                continue
            if rms < 0.001:
                raise ValueError('No audible speech. Check the selected microphone.')

            start = time.perf_counter()
            if active_model == 'fluid-speech':
                with contextlib.redirect_stdout(sys.stderr):
                    config, boost_info = decoding_config(model)
                    result = model.transcribe(str(audio), decoding_config=config, chunk_duration=30.0, overlap_duration=5.0)
                emit(
                    status='result',
                    model=active_model,
                    text=result.text,
                    seconds=time.perf_counter() - start,
                    audio_seconds=duration,
                    rms=rms,
                    **boost_info,
                )
            else:
                cfg = read_config()
                rcfg = cfg.get('rumble_2_flash', {}) if isinstance(cfg.get('rumble_2_flash'), dict) else {}
                req_lang = request.get('language')
                language = (str(req_lang).strip() or None) if req_lang is not None else (rcfg.get('language') or None)
                system_prompt = rcfg.get('system_prompt', '<digits><nofiller><verbatim>')
                chunk_duration = float(rcfg.get('chunk_duration', 30.0))
                hotwords = rumble_hotwords(cfg)
                with contextlib.redirect_stdout(sys.stderr):
                    result = model.generate(
                        samples,
                        language=language,
                        system_prompt=system_prompt,
                        chunk_duration=chunk_duration,
                        hotwords=hotwords,
                    )
                emit(
                    status='result',
                    model=active_model,
                    text=(result.text or '').strip(),
                    seconds=time.perf_counter() - start,
                    audio_seconds=duration,
                    rms=rms,
                    language=getattr(result, 'language', None),
                    boost_enabled=bool(hotwords),
                    boost_term_count=len(hotwords) if hotwords else 0,
                )
        except Exception as exc:
            emit(status='error', model=active_model, message=str(exc))
except Exception as exc:
    emit(status='fatal', message=str(exc))
    sys.exit(1)
