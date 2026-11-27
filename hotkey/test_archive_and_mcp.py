"""Offline tests for archive_db.py, meetings_cli.py backfill, and meeting_mcp.py."""
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import archive_db  # noqa: E402
import meeting_mcp as mcp  # noqa: E402
import meetings_cli  # noqa: E402


class ArchiveAndMcpTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="fluid-archive-")
        self.root = Path(self._tmp.name)
        self.db = self.root / "archive.sqlite"
        self.meetings_dir = self.root / "meetings"
        self.ef_dir = self.root / "Personal_Wiki/Overall/Clippings"
        self.startup_dir = self.root / "EdgeInference/Meetings/Raw"
        self.settings = {
            "ef_clippings_dir": str(self.ef_dir),
            "startup_raw_dir": str(self.startup_dir),
            "vault_confidence_threshold": 0.80,
            "speaker_labels": {"mic": "Shubham Korde", "remote": "Remote", "discord": "Discord"},
            "default_mic_speaker": "Shubham Korde",
        }

    def tearDown(self):
        self._tmp.cleanup()

    def test_upsert_search_and_dictations(self):
        archive_db.upsert_meeting(
            self.db,
            {
                "session_id": "sess-1111",
                "source": "google-meet",
                "trigger": "chrome-meet-auto",
                "started_at": "2026-11-28T10:00:00Z",
                "date": "2026-11-28",
                "duration_ms": 60000,
                "title": "Invariant AI Weekly Checkin",
                "meet_code": "abc-defg-hij",
                "category": "ef_checkin",
                "confidence": 0.95,
                "team_key": "pranav-parthiv",
                "participants": ["Shubham Korde", "Pranav Goswami"],
                "speaker_map": {"REMOTE_0": "Pranav Goswami"},
                "routing_status": "routed_personal_wiki",
                "summary_md": "## Updates\n- Shipped telemetry [00:00:10]",
                "transcript_md": "[00:00:10] Pranav Goswami: We shipped telemetry",
                "pipeline_stage": "done",
            },
            segments=[
                {
                    "start_ms": 10000,
                    "end_ms": 15000,
                    "track": "remote",
                    "raw_speaker": "REMOTE_0",
                    "speaker": "Pranav Goswami",
                    "language": "English",
                    "text": "We shipped telemetry",
                }
            ],
        )

        rec = archive_db.get_meeting(self.db, "sess-1111", include_segments=True)
        self.assertIsNotNone(rec)
        self.assertNotIn("translation_md", rec)
        self.assertEqual(rec["team_key"], "pranav-parthiv")
        self.assertEqual(rec["segments"][0]["raw_speaker"], "REMOTE_0")

        hits = archive_db.search_meetings(self.db, "telemetry")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["session_id"], "sess-1111")

        punct_hits = archive_db.search_meetings(self.db, '"telemetry": (Invariant)?')
        self.assertEqual(len(punct_hits), 1)

        archive_db.log_dictation(
            self.db,
            raw_text="uh build local mlx diarization",
            polished_text="Build local MLX diarization.",
            app_name="T3 Code",
            app_bundle_id="com.t3.code",
            model="rumble-2-flash",
            audio_seconds=2.4,
        )
        dicts = archive_db.search_dictations(self.db, "diarization")
        self.assertEqual(len(dicts), 1)
        self.assertEqual(dicts[0]["app_name"], "T3 Code")

    def test_legacy_db_with_translation_columns_remains_compatible(self):
        legacy_db = self.root / "legacy.sqlite"
        conn = sqlite3.connect(str(legacy_db))
        conn.executescript("""
            CREATE TABLE meetings (
                session_id TEXT PRIMARY KEY, source TEXT, trigger TEXT, started_at TEXT, date TEXT,
                duration_ms INTEGER, title TEXT, meet_code TEXT, category TEXT, confidence REAL,
                team_key TEXT, participants_json TEXT, speaker_map_json TEXT, routing_status TEXT,
                vault_path TEXT, session_dir TEXT, audio_paths_json TEXT, asr_model TEXT,
                diarization_model TEXT, summary_model TEXT, pipeline_stage TEXT, summary_md TEXT,
                transcript_md TEXT, translation_md TEXT, created_at TEXT, updated_at TEXT
            );
            CREATE TABLE segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, start_ms INTEGER, end_ms INTEGER,
                track TEXT, raw_speaker TEXT, speaker TEXT, language TEXT, text TEXT, english TEXT
            );
            CREATE VIRTUAL TABLE meetings_fts USING fts5(
                session_id UNINDEXED, title, participants, category, team_key,
                summary_md, transcript_md, translation_md
            );
        """)
        conn.close()
        archive_db.upsert_meeting(
            legacy_db,
            {
                "session_id": "legacy-1",
                "title": "Legacy Call",
                "summary_md": "Done",
                "transcript_md": "[00:00:00] Mic: hello",
            },
            segments=[{"start_ms": 0, "end_ms": 1000, "track": "mic", "text": "hello"}],
        )
        got = archive_db.get_meeting(legacy_db, "legacy-1", include_segments=True)
        self.assertEqual(got["title"], "Legacy Call")
        self.assertEqual(len(archive_db.search_meetings(legacy_db, "hello")), 1)

    def test_backfill_idempotent(self):
        sdir = self.meetings_dir / "2da727f4-3858-4a57-9340-229e7c8ea5ab"
        sdir.mkdir(parents=True)
        (sdir / "manifest.json").write_text(json.dumps({
            "session_id": "2da727f4-3858-4a57-9340-229e7c8ea5ab",
            "started_at": "2026-11-19T17:58:00Z",
            "trigger": "discord-voice-state",
            "participants": ["Shubham Korde"],
        }))
        (sdir / "transcript.json").write_text(json.dumps({
            "session_id": "2da727f4-3858-4a57-9340-229e7c8ea5ab",
            "asr_model": "rumble-2-flash",
            "segments": [
                {"track": "mic", "label": "Mic", "start_ms": 0, "end_ms": 15000, "language": ["English"], "text": "Testing perplexity score"}
            ],
        }))
        (sdir / "transcript.md").write_text("[00:00:00] Mic: Testing perplexity score\n")
        (sdir / "notes.md").write_text("---\ndate: \"2026-11-19\"\nsummary_model: \"openai/gpt-4.1-mini\"\n---\n\n# Meeting notes: Meeting (2026-11-19)\n\n## Where we landed\n- Checked perplexity.\n")
        (sdir / "pipeline-status.json").write_text(json.dumps({"summary": "done", "raw_drop": "written: x.md"}))

        res1 = meetings_cli.backfill_meetings(self.meetings_dir, self.db)
        res2 = meetings_cli.backfill_meetings(self.meetings_dir, self.db)
        self.assertEqual(res1["indexed"], 1)
        self.assertEqual(res2["indexed"], 1)
        rows = archive_db.list_meetings(self.db)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(archive_db.search_meetings(self.db, "perplexity")), 1)

    def test_mcp_stdio_protocol_and_reclassify_routing(self):
        archive_db.upsert_meeting(
            self.db,
            {
                "session_id": "abcd1234-0000-0000-0000-000000000001",
                "source": "google-meet",
                "trigger": "chrome-meet-auto",
                "started_at": "2026-11-28T12:00:00Z",
                "date": "2026-11-28",
                "duration_ms": 45000,
                "title": "Ambiguous Founder Call",
                "category": "other",
                "confidence": 0.55,
                "routing_status": "needs_review",
                "summary_md": "## Updates\n- Discussed fine-tuning pilot.",
                "transcript_md": "[00:00:05] Remote: Discussed fine-tuning pilot.",
                "pipeline_stage": "done",
            },
        )

        req_lines = "\n".join([
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
            json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
                "name": "list_meetings", "arguments": {"routing_status": "needs_review"}
            }}),
        ]) + "\n"
        out = io.StringIO()
        mcp.serve_stdio(io.StringIO(req_lines), out, db_path=self.db)
        responses = [json.loads(ln) for ln in out.getvalue().splitlines() if ln.strip()]
        self.assertEqual(len(responses), 3)
        self.assertEqual(responses[0]["result"]["serverInfo"]["name"], "fluid-meetings")
        tool_names = [t["name"] for t in responses[1]["result"]["tools"]]
        self.assertIn("reclassify_meeting", tool_names)
        listed = json.loads(responses[2]["result"]["content"][0]["text"])
        self.assertEqual(len(listed["meetings"]), 1)

        reclassified = mcp.call_tool(
            "reclassify_meeting",
            {
                "session_id": "abcd1234-0000-0000-0000-000000000001",
                "category": "ef_checkin",
                "title": "Invariant AI Checkin",
                "team_key": "pranav-parthiv",
                "route": True,
            },
            db_path=self.db,
            settings=self.settings,
        )
        self.assertEqual(reclassified["routing_status"], "routed_personal_wiki")
        self.assertEqual(len(list(self.ef_dir.glob("*.md"))), 1)
        self.assertFalse(self.startup_dir.exists())


if __name__ == "__main__":
    unittest.main()
