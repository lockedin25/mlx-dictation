"""Functional tests for meeting_pipeline.py. No network, no model weights: fake ASR, fake LLM, fake transport.

Run: python3 -m unittest hotkey/test_meeting_pipeline.py  (from the project root)
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))

import archive_db  # noqa: E402
import meeting_pipeline as mp  # noqa: E402

SETTINGS = mp.load_settings({"poll_seconds": 0.01})


def publish(path: Path, data: bytes) -> None:
    """Mimic the recorder: write a hidden temp file, then rename it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_wav(path: Path, seconds: float, channels: int = 2, rate: int = 44100) -> None:
    frames = int(seconds * rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x10" * frames * channels)


class FakeAsr:
    """Stands in for mp.RumbleWorker. `texts` maps file name or (file name, language) -> (text, language)."""

    def __init__(self, texts):
        self.texts = texts
        self.calls = []
        self.starts = 0
        self.closes = 0

    def start(self):
        self.starts += 1

    def close(self):
        self.closes += 1

    def transcribe(self, path: Path, *, language=None):
        self.calls.append((path.name, language))
        if language is not None and (path.name, language) in self.texts:
            text, out_lang = self.texts[(path.name, language)]
        else:
            text, out_lang = self.texts.get(path.name, ("", "en"))
        if text == "SILENT":
            return {"status": "error", "message": "No audible speech. Check the selected microphone."}
        if text == "BROKEN":
            raise RuntimeError("model crashed")
        return {"status": "result", "text": text, "language": out_lang, "audio_seconds": 30.0}


class FakeLlm:
    def __init__(self, fail_summary=False):
        self.fail_summary = fail_summary
        self.calls = []

    def chat(self, system, user):
        self.calls.append((system, user))
        if self.fail_summary:
            raise mp.LLMError("OpenRouter request failed with HTTP 503")
        return "## Where we landed\n- Ship it [00:00:30]\n\n## Action items\n| Owner | Item | Status |\n|---|---|---|\n| unassigned | Write doc | open |"


class MeetingPipelineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="meeting-test-")
        self.root = Path(self._tmp.name)
        self.session = self.root / "session-abc123"
        self.index = self.root / "index.md"
        self.raw = self.root / "vault" / "Meetings" / "Raw"

    def tearDown(self):
        self._tmp.cleanup()

    def no_sleep(self, _):
        time.sleep(0.005)

    def test_chunk_names_and_order(self):
        publish(self.session / "discord" / "000000030000.wav", b"x")
        publish(self.session / "mic" / "000000060000.wav", b"x")
        publish(self.session / "mic" / "000000000000.wav", b"x")
        publish(self.session / "mic" / ".000000090000.wav.part", b"x")  # in-flight, ignored
        publish(self.session / "mic" / "notes.txt", b"x")
        chunks = mp.discover_chunks(self.session)
        self.assertEqual([(c.track, c.start_ms) for c in chunks],
                         [("mic", 0), ("discord", 30000), ("mic", 60000)])

    def test_live_capture_waits_for_marker_then_finalises_without_translation(self):
        asr = FakeAsr({"000000000000.wav": ("hello team", "English"),
                       "000000030000.wav": ("नमस्ते सब", "Hindi")})
        llm = FakeLlm()
        write_wav(self.session / "mic" / "000000000000.wav", 30)

        def recorder():
            time.sleep(0.15)
            write_wav(self.session / "mic" / "000000030000.wav", 30)
            time.sleep(0.15)
            write_wav(self.session / "discord" / "000000030000.wav", 30)
            (self.session / "capture.done").write_text("")

        thread = threading.Thread(target=recorder)
        thread.start()
        code = mp.run_session(self.session, SETTINGS, asr, llm=llm, index=self.index,
                              raw_dir=self.raw, sleep=self.no_sleep)
        thread.join()
        self.assertEqual(code, 0)
        self.assertEqual(asr.starts, 1)
        self.assertEqual(asr.closes, 1)
        self.assertEqual(sorted(name for name, _ in asr.calls),
                         ["000000000000.wav", "000000030000.wav", "000000030000.wav"])

        transcript = (self.session / "transcript.md").read_text()
        self.assertIn("[00:00:00] Mic: hello team", transcript)
        self.assertIn("[00:00:30] Mic: नमस्ते सब", transcript)
        self.assertFalse((self.session / "translation.md").exists())
        self.assertEqual(len(llm.calls), 1)  # summary only, no translation call
        self.assertIn("नमस्ते सब", llm.calls[0][1])
        self.assertEqual(json.loads((self.session / "transcript.json").read_text())["segments"][0]["end_ms"], 30000)

        notes = (self.session / "notes.md").read_text()
        self.assertTrue(notes.startswith("---\nsession_id: "))
        self.assertIn("## Where we landed", notes)
        self.assertIn("[Transcript](transcript.md)", notes)
        self.assertNotIn("translation.md", notes)

        index_text = self.index.read_text()
        self.assertEqual(index_text.count("<!-- session: session-abc123 -->"), 1)
        self.assertNotIn("translation.md", index_text)

        drops = list(self.raw.glob("*.md"))
        self.assertEqual(len(drops), 1)
        drop = drops[0].read_text()
        self.assertIn('session_id: "session-abc123"', drop)
        self.assertNotIn(str(self.root), drop)
        self.assertIn("## Transcript", drop)
        self.assertNotIn("English translation", drop)
        self.assertEqual(list(self.raw.glob(".*.tmp")), [])

        status = json.loads((self.session / "pipeline-status.json").read_text())
        self.assertEqual(status["summary"], "done")
        self.assertEqual(status["chunks"]["ok"], 3)

    def test_recovery_skips_done_chunks_and_retries_errors(self):
        write_wav(self.session / "mic" / "000000000000.wav", 30)
        write_wav(self.session / "mic" / "000000030000.wav", 30)
        write_wav(self.session / "mic" / "000000060000.wav", 30)
        (self.session / "capture.done").write_text("")
        done_path = mp.artifact_path(self.session, "mic", 0)
        mp.write_json_atomic(done_path, {"track": "mic", "start_ms": 0, "status": "ok", "text": "kept",
                                         "language": "English", "end_ms": 30000})
        mp.write_json_atomic(mp.artifact_path(self.session, "mic", 30000),
                             {"track": "mic", "start_ms": 30000, "status": "error", "text": "", "end_ms": 30000})
        asr = FakeAsr({"000000030000.wav": ("retried", "English"), "000000060000.wav": ("third", "English")})
        code = mp.run_session(self.session, SETTINGS, asr, llm=FakeLlm(), index=self.index,
                              raw_dir=self.raw, sleep=self.no_sleep)
        self.assertEqual(code, 0)
        self.assertEqual(sorted(name for name, _ in asr.calls), ["000000030000.wav", "000000060000.wav"])
        self.assertIn("kept", (self.session / "transcript.md").read_text())

    def test_stereo_44k_chunk_gets_true_duration(self):
        path = self.session / "discord" / "000000010000.wav"
        write_wav(path, 2.5, channels=2, rate=44100)
        self.assertEqual(mp.wav_duration_ms(path), 2500)
        asr = FakeAsr({})
        artifact = mp.transcribe_chunk(asr, mp.Chunk(10000, "discord", path))
        self.assertEqual(artifact["status"], "silent")
        self.assertEqual(artifact["end_ms"], 12500)

    def test_silent_capture_reports_no_speech_without_calling_provider(self):
        write_wav(self.session / "mic" / "000000000000.wav", 1)
        (self.session / "capture.done").touch()
        asr = FakeAsr({"000000000000.wav": ("SILENT", "English")})
        code = mp.run_session(self.session, SETTINGS, asr, index=self.index, raw_dir=self.raw,
                              llm_factory=lambda: self.fail("the provider must not be used for silence"))
        self.assertEqual(code, 0)
        status = json.loads((self.session / "pipeline-status.json").read_text())
        self.assertEqual(status["summary"], "skipped (no speech)")
        self.assertFalse(self.index.exists())
        self.assertFalse((self.session / "notes.md").exists())

    def test_silent_and_failed_chunks_are_recorded_not_fatal(self):
        write_wav(self.session / "mic" / "000000000000.wav", 1)
        write_wav(self.session / "mic" / "000000001000.wav", 1)
        (self.session / "capture.done").write_text("")
        asr = FakeAsr({"000000000000.wav": ("SILENT", "English"), "000000001000.wav": ("BROKEN", "English")})
        mp.run_session(self.session, SETTINGS, asr, llm=FakeLlm(), sleep=self.no_sleep)
        statuses = {a["start_ms"]: a["status"] for _, a in mp.load_artifacts(self.session)}
        self.assertEqual(statuses, {0: "silent", 1000: "error"})
        self.assertIn("Failed chunks", (self.session / "transcript.md").read_text())

    def test_summary_failure_keeps_transcript_then_one_drop_on_retry(self):
        write_wav(self.session / "mic" / "000000000000.wav", 1)
        (self.session / "capture.done").write_text("")
        asr = FakeAsr({"000000000000.wav": ("hello", "English")})
        code = mp.run_session(self.session, SETTINGS, asr, llm=FakeLlm(fail_summary=True),
                              index=self.index, raw_dir=self.raw, sleep=self.no_sleep)
        self.assertEqual(code, 2)
        self.assertTrue((self.session / "transcript.md").exists())
        self.assertFalse((self.session / "translation.md").exists())
        self.assertFalse((self.session / "notes.md").exists())
        self.assertFalse(self.index.exists())
        self.assertEqual(list(self.raw.glob("*.md")) if self.raw.exists() else [], [])

        code = mp.run_session(self.session, SETTINGS, asr, llm=FakeLlm(), index=self.index,
                              raw_dir=self.raw, sleep=self.no_sleep)
        self.assertEqual(code, 0)
        code = mp.run_session(self.session, SETTINGS, asr, llm=FakeLlm(), index=self.index,
                              raw_dir=self.raw, sleep=self.no_sleep)
        self.assertEqual(code, 0)
        self.assertEqual(len(list(self.raw.glob("*.md"))), 1)
        self.assertEqual(self.index.read_text().count("<!-- session:"), 1)

    def test_no_llm_writes_transcript_only(self):
        write_wav(self.session / "mic" / "000000000000.wav", 1)
        (self.session / "capture.done").write_text("")
        code = mp.run_session(self.session, SETTINGS, FakeAsr({"000000000000.wav": ("hi", "English")}),
                              llm=None, use_llm=False, index=self.index, raw_dir=self.raw, sleep=self.no_sleep)
        self.assertEqual(code, 0)
        self.assertTrue((self.session / "transcript.md").exists())
        self.assertFalse((self.session / "notes.md").exists())
        self.assertFalse(self.index.exists())

    def test_session_lock_refuses_second_pipeline(self):
        self.session.mkdir(parents=True)
        with mp.session_lock(self.session):
            with self.assertRaises(mp.PipelineError):
                with mp.session_lock(self.session):
                    pass


class LanguageAllowlistAndScriptTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="lang-guard-")
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_allowed_language_and_script_helpers(self):
        allowed = ["English", "Hindi"]
        self.assertTrue(mp.is_allowed_language("English", allowed))
        self.assertTrue(mp.is_allowed_language(["Hindi"], allowed))
        self.assertTrue(mp.is_allowed_language("English, Hindi", allowed))
        self.assertTrue(mp.is_allowed_language(["English, Hindi"], allowed))
        self.assertTrue(mp.is_allowed_language(None, allowed))
        self.assertFalse(mp.is_allowed_language("Russian", allowed))
        self.assertFalse(mp.is_allowed_language("Spanish", allowed))
        self.assertFalse(mp.is_allowed_language(["Hindi", "Bengali"], allowed))

        self.assertFalse(mp.has_disallowed_script("Hello Shubham, मेरा thesis ये है कि ₹100 ok!"))
        self.assertTrue(mp.has_disallowed_script("Где?"))
        self.assertTrue(mp.has_disallowed_script("티게 됩니다."))
        self.assertTrue(mp.has_disallowed_script("মুছে আপনি আবার।"))

    def test_short_clip_retry_and_drop_rules(self):
        # 1. Disallowed Bengali short clip recovers when Hindi retry succeeds
        c_bengali = self.root / "remote_clips/00000.wav"
        write_wav(c_bengali, 0.84, channels=1, rate=16000)
        asr = FakeAsr({
            "00000.wav": ("মুছে আপনি আবার।", "Bengali"),
            ("00000.wav", "Hindi"): ("मुझे अपनी आवाज़।", "Hindi"),
        })
        art1 = mp.transcribe_chunk(
            asr, mp.Chunk(37400, "remote", c_bengali),
            speaker="REMOTE_0", end_ms_override=38240,
            allowed_languages=["English", "Hindi"], short_clip_max_ms=1500,
        )
        self.assertEqual(art1["status"], "ok")
        self.assertEqual(art1["text"], "मुझे अपनी आवाज़।")
        self.assertEqual(art1["retried_with_language"], "Hindi")

        # 2. Short clip whose Hindi retry is empty or still bad gets dropped
        c_spanish = self.root / "remote_clips/00001.wav"
        write_wav(c_spanish, 0.43, channels=1, rate=16000)
        asr2 = FakeAsr({
            "00001.wav": ("Chega.", "Portuguese"),
            ("00001.wav", "Hindi"): ("SILENT", "Hindi"),
        })
        art2 = mp.transcribe_chunk(
            asr2, mp.Chunk(77740, "remote", c_spanish),
            speaker="REMOTE_0", end_ms_override=78170,
            allowed_languages=["English", "Hindi"], short_clip_max_ms=1500,
        )
        self.assertEqual(art2["status"], "silent")
        self.assertEqual(art2["text"], "")

        # 3. Long mic chunk (>= 1500ms) is NEVER dropped even if retry stays weird
        c_long_mic = self.root / "mic/000000000000.wav"
        write_wav(c_long_mic, 15.0, channels=1, rate=16000)
        asr3 = FakeAsr({
            "000000000000.wav": ("Some mixed audio words", "Spanish"),
            ("000000000000.wav", "Hindi"): ("कुछ शब्द और English", "Spanish"),
        })
        art3 = mp.transcribe_chunk(
            asr3, mp.Chunk(0, "mic", c_long_mic),
            allowed_languages=["English", "Hindi"], short_clip_max_ms=1500,
        )
        self.assertEqual(art3["status"], "ok")
        self.assertEqual(art3["language_warning"], "kept_long_mic")

        # 4. Combined "English, Hindi" passes without retry
        c_combo = self.root / "mic/000000015000.wav"
        write_wav(c_combo, 5.0, channels=1, rate=16000)
        asr4 = FakeAsr({"000000015000.wav": ("मेरा thesis ये है", "English, Hindi")})
        art4 = mp.transcribe_chunk(
            asr4, mp.Chunk(15000, "mic", c_combo),
            allowed_languages=["English", "Hindi"], short_clip_max_ms=1500,
        )
        self.assertEqual(art4["status"], "ok")
        self.assertEqual(len(asr4.calls), 1)

        # A long remote turn must not be discarded on a failed language retry.
        long_remote = self.root / "remote_clips/long.wav"
        write_wav(long_remote, 3, channels=1, rate=16000)
        asr5 = FakeAsr({"long.wav": ("Original", "Spanish"), ("long.wav", "Hindi"): ("", "Hindi")})
        art5 = mp.transcribe_chunk(asr5, mp.Chunk(0, "remote", long_remote), allowed_languages=["English", "Hindi"])
        self.assertEqual((art5["status"], art5["text"]), ("ok", "Original"))

        short_mic = self.root / "mic/short.wav"
        write_wav(short_mic, 0.5, channels=1, rate=16000)
        asr6 = FakeAsr({"short.wav": ("Chega", "Portuguese"), ("short.wav", "Hindi"): ("", "Hindi")})
        art6 = mp.transcribe_chunk(asr6, mp.Chunk(0, "mic", short_mic), allowed_languages=["English", "Hindi"])
        self.assertEqual(art6["status"], "silent")


class OpenRouterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="meeting-or-")
        self.creds = Path(self._tmp.name) / "credentials.env"

    def tearDown(self):
        self._tmp.cleanup()

    def test_reads_key_from_env_style_file(self):
        self.creds.write_text("# comment\nOTHER=1\nexport OPENROUTER_API_KEY=\"sk-test-123\"\n")
        self.assertEqual(mp.read_api_key(str(self.creds)), "sk-test-123")

    def test_missing_key_error_does_not_leak_values(self):
        self.creds.write_text("OTHER=secret-value\n")
        with self.assertRaises(mp.LLMError) as ctx:
            mp.read_api_key(str(self.creds))
        self.assertNotIn("secret-value", str(ctx.exception))

    def test_request_shape_zdr_and_retry_on_429(self):
        sent = []
        responses = [
            urllib.error.HTTPError("u", 429, "rate", None, None),
            json.dumps({"choices": [{"message": {"content": "pong"}}]}).encode(),
        ]

        def transport(url, body, headers, timeout):
            sent.append((url, json.loads(body), headers))
            reply = responses.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

        client = mp.OpenRouter("sk-test-xyz", "model/x", zdr=True, transport=transport,
                               backoff=0, sleep=lambda _: None)
        self.assertEqual(client.chat("sys", "usr"), "pong")
        self.assertEqual(len(sent), 2)
        url, body, headers = sent[0]
        self.assertEqual(url, mp.OPENROUTER_URL)
        self.assertEqual(body["model"], "model/x")
        self.assertEqual(body["provider"], {"zdr": True})
        self.assertEqual(headers["Authorization"], "Bearer sk-test-xyz")

    def test_http_error_message_has_no_key(self):
        def transport(url, body, headers, timeout):
            raise urllib.error.HTTPError("u", 401, "bad", None, None)

        client = mp.OpenRouter("sk-test-xyz", "m", zdr=False, transport=transport, sleep=lambda _: None)
        with self.assertRaises(mp.LLMError) as ctx:
            client.chat("s", "u")
        self.assertNotIn("sk-test-xyz", str(ctx.exception))
        self.assertIn("401", str(ctx.exception))

    def test_parse_json_object_handles_code_fences(self):
        self.assertEqual(mp.parse_json_object('```json\n{"1-mic": "hi"}\n```'), {"1-mic": "hi"})
        with self.assertRaises(mp.LLMError):
            mp.parse_json_object("no json here")

    def test_split_sections_for_long_meetings(self):
        lines = [f"[00:00:{i:02d}] Mic: {'x' * 50}" for i in range(10)]
        sections = mp.split_sections(lines, 200)
        self.assertGreater(len(sections), 1)
        self.assertEqual("\n".join(sections).count("Mic:"), 10)


class FakeDiarizer:
    def __init__(self, segments):
        self.segments = segments
        self.calls = []
        self.model_name = "mlx-community/Nemotron-3-Diarization"

    def diarize(self, wav_path: Path):
        self.calls.append(wav_path.name)
        speakers = {s["speaker"] for s in self.segments}
        return {
            "status": "ok",
            "model": self.model_name,
            "num_speakers": len(speakers),
            "segments": list(self.segments),
        }


class SmartFakeLlm:
    def __init__(self, classify_reply=None, fail_classify_or_summary=False):
        self.classify_reply = classify_reply or {}
        self.fail_summary = fail_classify_or_summary
        self.prompts = []
        self.model = "fake/test-model"

    def chat(self, system, user):
        self.prompts.append((system, user))
        if self.fail_summary:
            raise mp.LLMError("provider down")
        if system == mp.CLASSIFY_SYSTEM:
            return json.dumps(self.classify_reply)
        if system == mp.SUMMARY_EF_SYSTEM:
            return "## Updates & Progress\n- Checked Invariant AI [00:00:01]\n\n## Product & Market Clarity\n- Clear wedge [00:00:01]\n\n## Team Dynamics\n- Aligned.\n\n## Key Risks & Open Questions\n- None.\n\n## Next Steps\n| Owner | Item | Status |\n|---|---|---|\n| Pranav Goswami | Ship demo | open |"
        if system == mp.SUMMARY_DISCOVERY_SYSTEM:
            return "## Customer Context\n- Enterprise ML team [00:00:01]\n\n## Pain Points & Current Workflow\n- Slow evals [00:00:01]\n\n## Key Insights & Quotes\n- \"Need local router\" [00:00:01]\n\n## Objections & Risks\n- Security.\n\n## Action items\n| Owner | Item | Status |\n|---|---|---|\n| Shubham Korde | Send benchmark | open |"
        return "## Where we landed\n- Ship it [00:00:01]\n\n## What we discussed\n- Details [00:00:01]\n\n## Disagreements and open questions\n- None.\n\n## Action items\n| Owner | Item | Status |\n|---|---|---|\n| unassigned | Follow up | open |"


class DiarizeClassifyAndRoutingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="meeting-v2-")
        self.root = Path(self._tmp.name)
        self.session = self.root / "session-v2"
        self.db = self.root / "archive.sqlite"
        self.ef_dir = self.root / "Personal_Wiki/Overall/Clippings"
        self.startup_dir = self.root / "EdgeInference/Meetings/Raw"
        self.registry_path = self.root / "team_registry.json"
        self.registry_path.write_text(json.dumps({
            "teams": {
                "pranav-parthiv": {
                    "founders": ["Pranav Goswami", "Parthiv C"],
                    "team_name": "Invariant AI",
                    "granola_search_terms": ["Pranav Goswami", "Invariant AI"],
                }
            }
        }))
        self.settings = mp.load_settings({
            "poll_seconds": 0.01,
            "team_registry_path": str(self.registry_path),
            "vault_confidence_threshold": 0.80,
        })

    def tearDown(self):
        self._tmp.cleanup()

    def test_build_speaker_clips_merges_and_splits_overlapping_speakers(self):
        segs = [
            {"start_ms": 1000, "end_ms": 5000, "speaker": "REMOTE_0"},
            {"start_ms": 5400, "end_ms": 9000, "speaker": "REMOTE_0"},  # gap 400ms -> merges
            {"start_ms": 4000, "end_ms": 6000, "speaker": "REMOTE_1"},  # overlaps REMOTE_0
            {"start_ms": 20000, "end_ms": 80000, "speaker": "REMOTE_0"},  # 60s -> splits into <=28s
        ]
        clips = mp.build_speaker_clips(segs, total_ms=85000, max_clip_ms=28000, merge_gap_ms=1500, pad_ms=200)
        self.assertEqual(clips[0]["speaker"], "REMOTE_0")
        self.assertEqual((clips[0]["start_ms"], clips[0]["end_ms"]), (1000, 9000))
        self.assertEqual((clips[0]["slice_start_ms"], clips[0]["slice_end_ms"]), (800, 9200))
        self.assertEqual(clips[1]["speaker"], "REMOTE_1")
        self.assertEqual((clips[1]["start_ms"], clips[1]["end_ms"]), (4000, 6000))
        long_clips = [c for c in clips if c["start_ms"] >= 20000]
        self.assertEqual(len(long_clips), 3)
        self.assertTrue(all(c["end_ms"] - c["start_ms"] <= 28000 for c in long_clips))

    def test_placeholder_speaker_guard_keeps_remote_ids_distinct(self):
        llm = SmartFakeLlm({
            "category": "other",
            "confidence": 0.90,
            "title": "Test",
            "team_key": None,
            "speaker_map": {
                "REMOTE_0": "Unknown",
                "REMOTE_1": "Speaker 1",
                "REMOTE_2": "N/A",
                "REMOTE_3": "Guest",
                "REMOTE_4": "Pranav Goswami",
            },
            "participants": ["Shubham Korde", "Unknown", "Participant", "Pranav Goswami"],
            "rationale": "Mixed",
        })
        res = mp.classify_meeting(llm, {}, ["[00:00:00] REMOTE_0: hi"], {}, {})
        self.assertEqual(res["speaker_map"], {"REMOTE_4": "Pranav Goswami"})
        self.assertCountEqual(res["participants"], ["Shubham Korde", "Pranav Goswami"])

    def test_clean_topic_title_strips_chrome_profile_suffix(self):
        cleaned = mp.clean_topic_title(
            "epf-ypvk-oax – Camera and microphone recording - Google Chrome – Shubham (joinef.com)"
        )
        self.assertEqual(cleaned, "epf-ypvk-oax")

    def test_diarize_first_speaker_map_pre_llm_persistence_and_reprocess(self):
        write_wav(self.session / "mic" / "000000000000.wav", 10, channels=1, rate=16000)
        write_wav(self.session / "remote" / "000000000000.wav", 10, channels=1, rate=16000)
        (self.session / "manifest.json").write_text(json.dumps({
            "session_id": "session-v2",
            "started_at": "2026-11-28T09:00:00Z",
            "source": "google-meet",
            "trigger": "chrome-meet-auto",
            "topic": "Invariant AI Checkin – Camera and microphone recording - Google Chrome – Shubham (joinef.com)",
            "meet_code": "abc-defg-hij",
            "mic_speaker": "Shubham Korde",
            "participants": ["Shubham Korde"],
        }))
        (self.session / "capture.done").write_text("")

        diarizer = FakeDiarizer([
            {"start_ms": 1000, "end_ms": 4000, "speaker": "REMOTE_0"},
            {"start_ms": 5000, "end_ms": 8000, "speaker": "REMOTE_1"},
        ])
        asr = FakeAsr({
            "000000000000.wav": ("Hey Pranav and Parthiv, मेरा thesis clear है", "Hindi"),
            "00000_000000001000_REMOTE_0.wav": ("Hi Shubham, Invariant AI demo is ready", "English"),
            "00001_000000005000_REMOTE_1.wav": ("और latency fix कर दी", "Hindi"),
        })

        # 1. Fail LLM on first run: raw transcript + archive.sqlite must STILL be persisted!
        code = mp.run_session(
            self.session, self.settings, asr,
            llm=SmartFakeLlm(fail_classify_or_summary=True),
            diarizer=diarizer,
            classify=True,
            archive_db_path=self.db,
            ef_clippings_dir=self.ef_dir,
            startup_raw_dir=self.startup_dir,
            sleep=lambda _: None,
        )
        self.assertEqual(code, 2)
        self.assertEqual(len(diarizer.calls), 1)
        self.assertTrue((self.session / "transcript.md").is_file())
        self.assertFalse((self.session / "translation.md").exists())
        saved_before = archive_db.get_meeting(self.db, "session-v2", include_segments=True)
        self.assertEqual(saved_before["pipeline_stage"], "transcribed")
        self.assertEqual(len(saved_before["segments"]), 3)
        self.assertEqual(saved_before["segments"][1]["raw_speaker"], "REMOTE_0")

        # 2. Reprocess with working LLM: ASR & diarizer must NOT rerun; mixed Devanagari+English feeds directly
        asr_reprocess = FakeAsr({})
        llm_ok = SmartFakeLlm({
            "category": "ef_checkin",
            "confidence": 0.94,
            "title": "Invariant AI Weekly Checkin",
            "team_key": "pranav-parthiv",
            "speaker_map": {"REMOTE_0": "Pranav Goswami", "REMOTE_1": "Parthiv C"},
            "participants": ["Shubham Korde", "Pranav Goswami", "Parthiv C"],
            "rationale": "Matched Invariant AI founders",
        })
        code2 = mp.run_session(
            self.session, self.settings, asr_reprocess,
            llm=llm_ok,
            classify=True,
            archive_db_path=self.db,
            ef_clippings_dir=self.ef_dir,
            startup_raw_dir=self.startup_dir,
            reprocess=True,
            sleep=lambda _: None,
        )
        self.assertEqual(code2, 0)
        self.assertEqual(asr_reprocess.calls, [])
        self.assertFalse((self.session / "translation.md").exists())
        transcript = (self.session / "transcript.md").read_text()
        self.assertIn("Shubham Korde: Hey Pranav and Parthiv, मेरा thesis clear है", transcript)
        self.assertIn("Pranav Goswami: Hi Shubham", transcript)
        self.assertIn("Parthiv C: और latency fix कर दी", transcript)
        self.assertEqual(len(list(self.ef_dir.glob("*.md"))), 1)
        self.assertFalse(self.startup_dir.exists())
        saved_after = archive_db.get_meeting(self.db, "session-v2", include_segments=True)
        self.assertEqual(saved_after["routing_status"], "routed_personal_wiki")
        self.assertEqual(saved_after["segments"][1]["raw_speaker"], "REMOTE_0")
        self.assertEqual(saved_after["segments"][1]["speaker"], "Pranav Goswami")

    def test_vault_isolation_confidence_gate_and_discovery_routing(self):
        # A. Low confidence (<0.80) writes to NEITHER vault and sets needs_review
        s_low = self.root / "session-low"
        write_wav(s_low / "mic" / "000000000000.wav", 1, channels=1, rate=16000)
        (s_low / "capture.done").touch()
        code_low = mp.run_session(
            s_low, self.settings, FakeAsr({"000000000000.wav": ("hello", "English")}),
            llm=SmartFakeLlm({
                "category": "ef_checkin",
                "confidence": 0.62,
                "title": "Uncertain Call",
                "team_key": None,
                "speaker_map": {},
                "participants": ["Shubham Korde"],
                "rationale": "Unclear",
            }),
            classify=True,
            archive_db_path=self.db,
            ef_clippings_dir=self.ef_dir,
            startup_raw_dir=self.startup_dir,
            sleep=lambda _: None,
        )
        self.assertEqual(code_low, 0)
        self.assertFalse(self.ef_dir.exists())
        self.assertFalse(self.startup_dir.exists())
        self.assertEqual(archive_db.get_meeting(self.db, "session-low")["routing_status"], "needs_review")

        # B. High confidence startup_customer_discovery writes ONLY to startup_raw_dir
        s_disc = self.root / "session-disc"
        write_wav(s_disc / "mic" / "000000000000.wav", 1, channels=1, rate=16000)
        (s_disc / "capture.done").touch()
        hook_marker = self.root / "hook-ran.txt"
        hook_settings = mp._deep_merge(self.settings, {
            "post_ingest_hooks": {
                "enabled": True,
                "startup_command": f"echo \"$FLUID_SESSION_ID:$FLUID_CATEGORY\" > {hook_marker}",
            }
        })
        code_disc = mp.run_session(
            s_disc, hook_settings, FakeAsr({"000000000000.wav": ("tell me about your inference bottleneck", "English")}),
            llm=SmartFakeLlm({
                "category": "startup_customer_discovery",
                "confidence": 0.91,
                "title": "ML Inference Buyer Discovery",
                "team_key": None,
                "speaker_map": {},
                "participants": ["Shubham Korde"],
                "rationale": "Customer discovery interview",
            }),
            classify=True,
            archive_db_path=self.db,
            ef_clippings_dir=self.ef_dir,
            startup_raw_dir=self.startup_dir,
            sleep=lambda _: None,
        )
        self.assertEqual(code_disc, 0)
        self.assertFalse(self.ef_dir.exists())
        self.assertEqual(len(list(self.startup_dir.glob("*.md"))), 1)
        self.assertEqual(hook_marker.read_text().strip(), "session-disc:startup_customer_discovery")


if __name__ == "__main__":
    unittest.main()
