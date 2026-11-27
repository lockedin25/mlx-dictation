"""Integration check using real weights: reuse, idle release, reload, recovery."""
import json
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def response(process):
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        assert selector.select(45), 'Coordinator response timed out'
    return json.loads(process.stdout.readline())


def request(process, filename):
    process.stdin.write(json.dumps({'path': str(ROOT / filename)}) + '\n')
    process.stdin.flush()
    return response(process)


with tempfile.TemporaryDirectory(prefix='fluid-idle-test-') as tmp:
    config, state = Path(tmp)/'config.json', Path(tmp)/'state.json'
    base = json.loads((ROOT/'hotkey/config.json').read_text())
    config.write_text(json.dumps(dict(base, idle_unload_seconds=2)))
    env = dict(os.environ, HF_HUB_OFFLINE='1', FLUID_CONFIG_PATH=str(config), FLUID_MODEL_STATE_PATH=str(state))
    process = subprocess.Popen([str(ROOT/'.venv/bin/python'), '-u', str(ROOT/'hotkey/worker.py')], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
    try:
        assert response(process)['status'] == 'ready'
        assert json.loads(state.read_text())['model_loaded'] is False
        first = request(process, 'samples/recorded-speech.flac')
        assert first['status'] == 'result' and 'stew' in first['text'], first
        if first.get('model') == 'fluid-speech':
            assert first.get('boost_enabled', False) == base.get('vocabulary_boost', {}).get('enabled', False), first
        second = request(process, 'samples/smoke-test.wav')
        assert second['status'] == 'result' and 'meeting' in second['text'], second
        assert first['model_pid'] == second['model_pid'], 'Model was not reused'
        print('PASS: repeated dictation reuses loaded model', flush=True)
        # Change the timeout live; the coordinator polls settings while idle.
        config.write_text(json.dumps(dict(base, idle_unload_seconds=0)))
        deadline = time.monotonic() + 5
        while json.loads(state.read_text())['model_loaded']:
            assert time.monotonic() < deadline, 'Model did not unload'
            time.sleep(0.1)
        try:
            os.kill(second['model_pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError('Model child still exists after unload')
        print('PASS: live timeout change unloads and exits model process', flush=True)
        config.write_text(json.dumps(dict(base, idle_unload_seconds=2)))
        third = request(process, 'samples/recorded-speech.flac')
        assert third['status'] == 'result' and third['model_pid'] != first['model_pid']
        print('PASS: subsequent dictation reloads model', flush=True)
        error = request(process, 'samples/nonexistent-test.wav')
        assert error['status'] == 'error', error
        recovered = request(process, 'samples/smoke-test.wav')
        assert recovered['status'] == 'result' and 'meeting' in recovered['text']
        print('PASS: file error does not prevent next dictation', flush=True)
        child_pid = recovered['model_pid']
        results = {'first_request_seconds': first['request_seconds'], 'warm_request_seconds': second['request_seconds'], 'reloaded_request_seconds': third['request_seconds'], 'checks_passed': ['model absent at startup', 'model reuse', 'live configurable timeout', 'child exit releases model', 'reload after idle', 'recovery after error']}
    finally:
        process.terminate();process.wait(timeout=8)
    try:
        os.kill(child_pid, 0)
    except ProcessLookupError:
        print('PASS: quitting coordinator also stops model child', flush=True)
    else:
        raise AssertionError('Model child orphaned on shutdown')
    results['checks_passed'].append('no model child after quit')
    (ROOT/'hotkey/idle-verification.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2), flush=True)
