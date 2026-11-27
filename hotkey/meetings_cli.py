#!/usr/bin/env python3
"""Management CLI for Fluid Dictation meetings and dictation archive.

Usage:
    meetings_cli.py list [--category CAT] [--needs-review] [--limit 20]
    meetings_cli.py search QUERY [--category CAT] [--limit 10]
    meetings_cli.py show SESSION_ID [--transcript]
    meetings_cli.py reprocess SESSION_ID [--reasr] [--provider ...] [--model ...] [--base-url ...] [--reclassify] [--resummarize]
    meetings_cli.py reclassify SESSION_ID --category CAT [--title ...] [--team-key ...] [--route]
    meetings_cli.py backfill [--meetings-dir DIR] [--archive-db FILE]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HOTKEY_DIR = Path(__file__).resolve().parent
if str(HOTKEY_DIR) not in sys.path:
    sys.path.insert(0, str(HOTKEY_DIR))

import archive_db  # noqa: E402
import meeting_pipeline as mp  # noqa: E402

DEFAULT_MEETINGS_DIR = Path.home() / "Library/Application Support/Fluid Dictation/meetings"


def _resolve_session_dir(session_arg: str, meetings_dir: Path, db_path: Path) -> Path:
    candidate = Path(session_arg).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    direct = meetings_dir / session_arg
    if direct.is_dir():
        return direct.resolve()
    rec = archive_db.get_meeting(db_path, session_arg)
    if rec and rec.get("session_dir") and Path(rec["session_dir"]).is_dir():
        return Path(rec["session_dir"]).resolve()
    if meetings_dir.is_dir() and len(session_arg) >= 4:
        matches = [
            p for p in meetings_dir.iterdir()
            if p.is_dir() and (p.name.startswith(session_arg) or p.name.endswith(session_arg))
        ]
        if len(matches) == 1:
            return matches[0].resolve()
    raise SystemExit(f"Session not found: {session_arg}")


def _extract_summary_body(notes_md: str) -> str:
    body = notes_md
    if body.startswith("---\n"):
        parts = body.split("\n---\n", 1)
        if len(parts) == 2:
            body = parts[1]
    body = re.sub(r"^\s*# [^\n]+\n+", "", body)
    body = re.split(r"\n## Links\s*\n", body, maxsplit=1)[0]
    return body.strip()


def _load_existing_session_record(session: Path) -> tuple[dict, list[dict]]:
    meta = mp.session_meta(session)
    session_id = str(meta.get("session_id") or session.name)
    started = meta.get("started_at")
    date = str(started)[:10] if started else ""

    t_json: dict = {}
    if (session / "transcript.json").is_file():
        try:
            t_json = json.loads((session / "transcript.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            t_json = {}

    classification: dict = {}
    if (session / "classification.json").is_file():
        try:
            classification = json.loads((session / "classification.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            classification = {}

    status_json: dict = {}
    if (session / "pipeline-status.json").is_file():
        try:
            status_json = json.loads((session / "pipeline-status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            status_json = {}

    transcript_md = (session / "transcript.md").read_text(encoding="utf-8") if (session / "transcript.md").is_file() else ""
    notes_md = (session / "notes.md").read_text(encoding="utf-8") if (session / "notes.md").is_file() else ""
    summary_md = _extract_summary_body(notes_md) if notes_md else ""

    if not date and notes_md.startswith("---\n"):
        m = re.search(r'^date:\s*"?(\d{4}-\d{2}-\d{2})"?', notes_md, re.M)
        if m:
            date = m.group(1)

    segments = list(t_json.get("segments") or [])
    duration_ms = max((int(s.get("end_ms") or 0) for s in segments), default=0)

    audio_paths = {}
    for key in ("mic", "remote"):
        opus = session / "audio" / f"{key}.opus"
        if opus.is_file():
            audio_paths[key] = f"audio/{key}.opus"
    if not audio_paths:
        if (session / "mic").is_dir():
            audio_paths["mic"] = "mic/"
        if (session / "remote").is_dir():
            audio_paths["remote"] = "remote/"
        elif (session / "discord").is_dir():
            audio_paths["discord"] = "discord/"

    routing_status = status_json.get("routing_status")
    if not routing_status:
        if status_json.get("raw_drop"):
            routing_status = "routed_edge_inference"
        elif notes_md:
            routing_status = "archive_only"
        else:
            routing_status = "pending"

    pipeline_stage = "done" if notes_md or status_json.get("summary") == "done" else (
        "transcribed" if transcript_md else "captured"
    )
    source = meta.get("source") or ("discord" if meta.get("trigger") == "discord-voice-state" else "local")
    summary_model = None
    if notes_md.startswith("---\n"):
        m = re.search(r'^summary_model:\s*"?([^"\n]+)"?', notes_md, re.M)
        if m:
            summary_model = m.group(1)

    record = {
        "session_id": session_id,
        "source": source,
        "trigger": meta.get("trigger") or "unknown",
        "started_at": started,
        "date": date or None,
        "duration_ms": duration_ms,
        "title": classification.get("title") or mp.clean_topic_title(str(meta.get("topic") or meta.get("kind") or "Meeting")),
        "meet_code": meta.get("meet_code"),
        "category": classification.get("category") or meta.get("kind") or "unknown",
        "confidence": classification.get("confidence"),
        "team_key": classification.get("team_key"),
        "participants": classification.get("participants") or mp.participant_names(meta),
        "speaker_map": classification.get("speaker_map") or t_json.get("speaker_map") or {},
        "routing_status": routing_status,
        "vault_path": None,
        "session_dir": str(session),
        "audio_paths": audio_paths,
        "asr_model": t_json.get("asr_model") or mp.ASR_MODEL,
        "diarization_model": t_json.get("diarization_model"),
        "summary_model": summary_model,
        "pipeline_stage": pipeline_stage,
        "summary_md": summary_md,
        "transcript_md": transcript_md,
    }
    return record, segments


def backfill_meetings(
    meetings_dir: Path | None = None,
    db_path: Path | None = None,
) -> dict:
    root = (meetings_dir or DEFAULT_MEETINGS_DIR).expanduser()
    resolved_db = archive_db.resolve_db_path(db_path)
    archive_db.init_db(resolved_db)
    indexed = 0
    skipped = 0
    if not root.is_dir():
        return {"indexed": 0, "skipped": 0, "db_path": str(resolved_db)}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        has_content = any(
            (entry / name).exists()
            for name in ("manifest.json", "transcript.json", "transcript.md", "notes.md", "chunks")
        )
        if not has_content:
            skipped += 1
            continue
        record, segments = _load_existing_session_record(entry)
        archive_db.upsert_meeting(resolved_db, record, segments=segments)
        indexed += 1
    return {"indexed": indexed, "skipped": skipped, "db_path": str(resolved_db)}


def reclassify_and_route(
    *,
    session_id: str,
    category: str,
    title: str | None = None,
    team_key: str | None = None,
    route: bool = False,
    confidence: float = 1.0,
    db_path: Path | None = None,
    settings: dict | None = None,
) -> dict:
    if category not in mp.VALID_CATEGORIES:
        raise ValueError(f"Invalid category {category!r}; expected one of {mp.VALID_CATEGORIES}")
    resolved_db = archive_db.resolve_db_path(db_path)
    meeting = archive_db.get_meeting(resolved_db, session_id, include_segments=True)
    if not meeting:
        raise ValueError(f"Meeting not found in archive: {session_id}")

    settings = settings or mp.load_settings()
    full_id = meeting["session_id"]
    session_dir = Path(meeting["session_dir"]) if meeting.get("session_dir") else None
    meta = mp.session_meta(session_dir) if (session_dir and session_dir.is_dir()) else {
        "session_id": full_id,
        "started_at": meeting.get("started_at"),
        "source": meeting.get("source"),
        "trigger": meeting.get("trigger"),
        "participants": meeting.get("participants") or [],
    }
    new_title = (title or meeting.get("title") or "Meeting").strip()
    new_team = team_key if team_key is not None else meeting.get("team_key")

    classification = {
        "category": category,
        "confidence": float(confidence),
        "title": new_title,
        "team_key": new_team,
        "speaker_map": meeting.get("speaker_map") or {},
        "participants": meeting.get("participants") or [],
        "rationale": "manual reclassification",
    }
    if session_dir and session_dir.is_dir():
        cls_path = session_dir / "classification.json"
        if cls_path.is_file():
            try:
                old = json.loads(cls_path.read_text(encoding="utf-8"))
                if isinstance(old, dict):
                    old.update(classification)
                    classification = old
            except (OSError, ValueError):
                pass
        mp.write_json_atomic(cls_path, classification)

    routing_status = meeting.get("routing_status") or "archive_only"
    vault_path = Path(meeting["vault_path"]) if meeting.get("vault_path") else None

    if route:
        ef_dir = Path(str(settings["ef_clippings_dir"])).expanduser() if settings.get("ef_clippings_dir") else None
        startup_dir = Path(str(settings["startup_raw_dir"])).expanduser() if settings.get("startup_raw_dir") else None
        raw_dir = Path(str(settings["vault_raw_dir"])).expanduser() if settings.get("vault_raw_dir") else None
        date = meeting.get("date") or (str(meeting.get("started_at") or "")[:10]) or "2026-01-01"
        summary = meeting.get("summary_md") or "_No summary available._"
        transcript_lines = [
            ln for ln in (meeting.get("transcript_md") or "").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        head = mp.frontmatter(
            full_id, meta, date, int(meeting.get("duration_ms") or 0),
            meeting.get("summary_model") or "manual",
            meeting.get("participants") or [],
            classification=classification,
            diarization_model=meeting.get("diarization_model"),
        )
        shared_text = (
            f"{head}\n# Meeting notes: {new_title} ({date})\n\n{summary}\n\n"
            "## Transcript\n\n" + "\n".join(transcript_lines or ["_No transcript._"]) + "\n"
        )
        routing_status, vault_path = mp.route_meeting(
            session_id=full_id,
            date=date,
            meta=meta,
            shared_text=shared_text,
            summary=summary,
            transcript_lines=transcript_lines,
            classification=classification,
            settings=settings,
            raw_dir=raw_dir,
            ef_clippings_dir=ef_dir,
            startup_raw_dir=startup_dir,
            classify_enabled=True,
        )

    meeting.update({
        "title": new_title,
        "category": category,
        "confidence": float(confidence),
        "team_key": new_team,
        "routing_status": routing_status,
        "vault_path": str(vault_path) if vault_path else None,
    })
    archive_db.upsert_meeting(resolved_db, meeting)
    return {
        "session_id": full_id,
        "category": category,
        "confidence": float(confidence),
        "title": new_title,
        "team_key": new_team,
        "routing_status": routing_status,
        "vault_path": str(vault_path) if vault_path else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fluid Dictation meeting archive CLI.")
    parser.add_argument("--archive-db", type=Path, default=None)
    parser.add_argument("--meetings-dir", type=Path, default=DEFAULT_MEETINGS_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List meetings from archive")
    p_list.add_argument("--category", default=None)
    p_list.add_argument("--needs-review", action="store_true")
    p_list.add_argument("--limit", type=int, default=20)
    p_list.add_argument("--json", action="store_true")

    p_search = sub.add_parser("search", help="Full-text search across meetings")
    p_search.add_argument("query")
    p_search.add_argument("--category", default=None)
    p_search.add_argument("--limit", type=int, default=10)
    p_search.add_argument("--json", action="store_true")

    p_show = sub.add_parser("show", help="Show meeting notes and metadata")
    p_show.add_argument("session_id")
    p_show.add_argument("--transcript", action="store_true")

    p_reproc = sub.add_parser("reprocess", help="Reprocess an existing session with current/custom ASR or LLM")
    p_reproc.add_argument("session_id")
    p_reproc.add_argument("--provider", default=None)
    p_reproc.add_argument("--model", default=None)
    p_reproc.add_argument("--base-url", default=None)
    p_reproc.add_argument("--reasr", action="store_true", help="Re-slice from audio/*.opus and rerun ASR")
    p_reproc.add_argument("--reclassify", action="store_true")
    p_reproc.add_argument("--resummarize", action="store_true")

    p_recls = sub.add_parser("reclassify", help="Manually reclassify and optionally route a meeting")
    p_recls.add_argument("session_id")
    p_recls.add_argument("--category", required=True, choices=mp.VALID_CATEGORIES)
    p_recls.add_argument("--title", default=None)
    p_recls.add_argument("--team-key", default=None)
    p_recls.add_argument("--route", action="store_true")

    sub.add_parser("backfill", help="Index all existing meeting folders into archive.sqlite")

    args = parser.parse_args(argv)
    settings = mp.load_settings()
    db_path = archive_db.resolve_db_path(
        args.archive_db or settings.get("archive_db") or settings.get("configured_archive_db")
    )

    if args.command == "backfill":
        result = backfill_meetings(args.meetings_dir, db_path)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "list":
        rows = archive_db.list_meetings(
            db_path,
            category=args.category,
            routing_status="needs_review" if args.needs_review else None,
            limit=args.limit,
        )
        if args.json:
            print(json.dumps(rows, indent=2, ensure_ascii=False))
        else:
            for r in rows:
                print(
                    f"{r.get('date') or '?'} | {r['session_id'][:8]} | {r.get('category')} "
                    f"| {r.get('routing_status')} | {r.get('title')}"
                )
        return 0

    if args.command == "search":
        rows = archive_db.search_meetings(db_path, args.query, category=args.category, limit=args.limit)
        if args.json:
            print(json.dumps(rows, indent=2, ensure_ascii=False))
        else:
            for r in rows:
                print(f"{r.get('date') or '?'} | {r['session_id'][:8]} | {r.get('category')} | {r.get('title')}")
                if r.get("match_snippet"):
                    print(f"  ... {r['match_snippet']}")
        return 0

    if args.command == "show":
        rec = archive_db.get_meeting(
            db_path,
            args.session_id,
            include_transcript=args.transcript,
        )
        if not rec:
            print(f"Meeting not found: {args.session_id}", file=sys.stderr)
            return 1
        print(f"# {rec.get('title')} ({rec.get('date')})")
        print(f"Session: {rec['session_id']} | Category: {rec.get('category')} | Routing: {rec.get('routing_status')}")
        if rec.get("participants"):
            print(f"Participants: {', '.join(rec['participants'])}")
        print("\n" + (rec.get("summary_md") or "_No summary._"))
        if args.transcript and rec.get("transcript_md"):
            print("\n" + rec["transcript_md"])
        return 0

    if args.command == "reclassify":
        res = reclassify_and_route(
            session_id=args.session_id,
            category=args.category,
            title=args.title,
            team_key=args.team_key,
            route=args.route,
            db_path=db_path,
            settings=settings,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    if args.command == "reprocess":
        session_dir = _resolve_session_dir(args.session_id, args.meetings_dir, db_path)
        cmd_args = [str(session_dir), "--reprocess", "--archive-db", str(db_path)]
        if args.provider:
            cmd_args += ["--provider", args.provider]
        if args.model:
            cmd_args += ["--model", args.model]
        if args.base_url:
            cmd_args += ["--base-url", args.base_url]
        if args.reasr:
            cmd_args.append("--reasr")
        if args.reclassify:
            cmd_args.append("--reclassify")
        if args.resummarize:
            cmd_args.append("--resummarize")
        return mp.main(cmd_args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
