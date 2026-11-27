"""Offline tests for llm_provider.py. No network, no local weights."""
import io
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm_provider as lp  # noqa: E402


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="llm-prov-")
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_openrouter_zdr_and_retry(self):
        sent = []
        replies = [
            urllib.error.HTTPError("u", 429, "rate", None, io.BytesIO(b"")),
            json.dumps({"choices": [{"message": {"content": "ok-zdr"}}]}).encode(),
        ]

        def transport(url, body, headers, timeout):
            sent.append((url, json.loads(body), headers))
            val = replies.pop(0)
            if isinstance(val, Exception):
                raise val
            return val

        creds = self.root / "or.env"
        creds.write_text('OPENROUTER_API_KEY="sk-or-secret"\n')
        prov = lp.create_llm_provider({
            "provider": "openrouter",
            "model": "openai/gpt-4.1-mini",
            "zdr": True,
            "credentials_file": str(creds),
            "api_key_env": "OPENROUTER_API_KEY",
        }, transport=transport)
        # Override sleep to avoid delay on 429 retry
        prov.sleep = lambda _: None
        out = prov.chat("sys", "usr")
        self.assertEqual(out, "ok-zdr")
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0][0], lp.OPENROUTER_URL)
        self.assertEqual(sent[0][1]["provider"], {"zdr": True})
        self.assertEqual(sent[0][2]["Authorization"], "Bearer sk-or-secret")

    def test_openai_compatible_localhost_needs_no_key_and_normalizes_v1(self):
        sent = []

        def transport(url, body, headers, timeout):
            sent.append((url, json.loads(body), headers))
            return json.dumps({"choices": [{"message": {"content": "local-reply"}}]}).encode()

        prov = lp.create_llm_provider({
            "provider": "openai_compatible",
            "base_url": "http://localhost:11434/v1",
            "model": "qwen3:8b",
        }, transport=transport)
        self.assertEqual(prov.chat("sys", "hi"), "local-reply")
        self.assertEqual(sent[0][0], "http://localhost:11434/v1/chat/completions")
        self.assertNotIn("Authorization", sent[0][2])
        self.assertNotIn("provider", sent[0][1])

    def test_anthropic_provider_shape_and_error_redaction(self):
        sent = []

        def transport(url, body, headers, timeout):
            sent.append((url, json.loads(body), headers))
            return json.dumps({"content": [{"type": "text", "text": "claude-ok"}]}).encode()

        prov = lp.create_llm_provider({
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "api_key": "sk-ant-secret",
        }, transport=transport)
        self.assertEqual(prov.chat("sys", "usr"), "claude-ok")
        self.assertEqual(sent[0][0], lp.ANTHROPIC_URL)
        self.assertEqual(sent[0][2]["x-api-key"], "sk-ant-secret")
        self.assertEqual(sent[0][1]["system"], "sys")

        def bad_transport(url, body, headers, timeout):
            raise urllib.error.HTTPError(url, 401, "auth", None, None)

        bad = lp.AnthropicProvider("sk-ant-secret", "claude-sonnet-4-6", transport=bad_transport, sleep=lambda _: None)
        with self.assertRaises(lp.LLMError) as ctx:
            bad.chat("s", "u")
        self.assertNotIn("sk-ant-secret", str(ctx.exception))
        self.assertIn("401", str(ctx.exception))

    def test_gemini_provider_reads_raw_key_file(self):
        sent = []

        def transport(url, body, headers, timeout):
            sent.append((url, json.loads(body), headers))
            return json.dumps({"candidates": [{"content": {"parts": [{"text": "gemini-ok"}]}}]}).encode()

        key_file = self.root / "gemini-key"
        key_file.write_text("AIza-test-secret\n")
        prov = lp.create_llm_provider({
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "credentials_file": str(key_file),
            "api_key_env": "UNSET_GEMINI_ENV_FOR_TEST",
        }, transport=transport)
        self.assertEqual(prov.chat("sys", "usr"), "gemini-ok")
        self.assertTrue(sent[0][0].endswith("/models/gemini-2.5-flash:generateContent"))
        self.assertEqual(sent[0][2]["x-goog-api-key"], "AIza-test-secret")

    def test_mlx_local_provider_uses_runner(self):
        seen = []

        def fake_runner(cmd, payload, timeout):
            seen.append((cmd, json.loads(payload)))
            return "mlx-ok"

        prov = lp.create_llm_provider({
            "provider": "mlx_local",
            "model": "models/qwen35-cleanup-2b",
        }, runner=fake_runner)
        self.assertEqual(prov.chat("sys", "usr"), "mlx-ok")
        self.assertEqual(seen[0][1]["model"], "models/qwen35-cleanup-2b")


if __name__ == "__main__":
    unittest.main()
