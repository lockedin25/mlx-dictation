"""Unified SQLite + FTS5 archive for meetings and local dictations.

Pure Python stdlib (sqlite3). Stores meeting metadata, raw transcripts,
summaries, segment timestamps, classification results, vault routing state, and
dictation history in WAL mode for safe concurrent reads/writes.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import re
import sqlite3
from pathlib import Path

DEFAULT_ARCHIVE_DB = Path.home() / "Library/Application Support/Fluid Dictation/archive.sqlite"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meetings (
    session_id TEXT PRIMARY KEY,
    source TEXT,
    trigger TEXT,
    started_at TEXT,
    date TEXT,
    duration_ms INTEGER DEFAULT 0,
    title TEXT,
    meet_code TEXT,
    category TEXT,
    confidence REAL,
    team_key TEXT,
    participants_json TEXT,
    speaker_map_json TEXT,
    routing_status TEXT,
    vault_path TEXT,
    session_dir TEXT,
    audio_paths_json TEXT,
    asr_model TEXT,
    diarization_model TEXT,
    summary_model TEXT,
    pipeline_stage TEXT,
    summary_md TEXT,
    transcript_md TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES meetings(session_id) ON DELETE CASCADE,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    track TEXT,
    raw_speaker TEXT,
    speaker TEXT,
    language TEXT,
    text TEXT
);
CREATE INDEX IF NOT EXISTS idx_segments_session_start ON segments(session_id, start_ms);

CREATE TABLE IF NOT EXISTS dictations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    app_name TEXT,
    app_bundle_id TEXT,
    model TEXT,
    audio_seconds REAL DEFAULT 0.0,
    raw_text TEXT NOT NULL,
    polished_text TEXT
);
CREATE INDEX IF NOT EXISTS idx_dictations_created_at ON dictations(created_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS meetings_fts USING fts5(
    session_id UNINDEXED,
    title,
    participants,
    category,
    team_key,
    summary_md,
    transcript_md
);

CREATE VIRTUAL TABLE IF NOT EXISTS dictations_fts USING fts5(
    dictation_id UNINDEXED,
    app_name,
    raw_text,
    polished_text
);
"""


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def resolve_db_path(db_path: str | Path | None = None) -> Path:
    if db_path is not None:
        return Path(db_path).expanduser()
    env = os.environ.get("FLUID_ARCHIVE_DB", "").strip()
    if env:
        return Path(env).expanduser()
    return DEFAULT_ARCHIVE_DB


@contextlib.contextmanager
def connect(db_path: str | Path | None = None):
    path = resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA foreign_keys=ON;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | Path | None = None) -> Path:
    path = resolve_db_path(db_path)
    with connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
    return path


def _json_text(value, default) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(default if value is None else value, ensure_ascii=False)


def _decode_json(value: str | None, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return default


def _scalar_str(value, *, allow_none: bool = False) -> str | None:
    if value is None:
        return None if allow_none else ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if v is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _row_to_meeting(row: sqlite3.Row) -> dict:
    item = dict(row)
    item["participants"] = _decode_json(item.pop("participants_json", None), [])
    item["speaker_map"] = _decode_json(item.pop("speaker_map_json", None), {})
    item["audio_paths"] = _decode_json(item.pop("audio_paths_json", None), {})
    return item


def _fts_query(raw: str) -> str:
    tokens = [tok.strip('"') for tok in re.findall(r"[A-Za-z0-9_@.+-]+", raw or "")]
    tokens = [tok for tok in tokens if tok]
    if not tokens:
        return ""
    return " ".join('"' + tok.replace('"', '""') + '"' for tok in tokens)


def upsert_meeting(
    db_path: str | Path | None,
    record: dict,
    segments: list[dict] | None = None,
) -> str:
    session_id = str(record["session_id"])
    now = utc_now_iso()
    init_db(db_path)
    with connect(db_path) as conn:
        existing = conn.execute(
            "SELECT created_at FROM meetings WHERE session_id = ?", (session_id,)
        ).fetchone()
        created_at = record.get("created_at") or (existing["created_at"] if existing else now)
        updated_at = record.get("updated_at") or now
        participants = record.get("participants")
        participants_json = _json_text(
            participants if participants is not None else record.get("participants_json"), []
        )
        speaker_map = record.get("speaker_map")
        speaker_map_json = _json_text(
            speaker_map if speaker_map is not None else record.get("speaker_map_json"), {}
        )
        audio_paths = record.get("audio_paths")
        audio_paths_json = _json_text(
            audio_paths if audio_paths is not None else record.get("audio_paths_json"), {}
        )

        values = {
            "session_id": session_id,
            "source": record.get("source"),
            "trigger": record.get("trigger"),
            "started_at": record.get("started_at"),
            "date": record.get("date"),
            "duration_ms": int(record.get("duration_ms") or 0),
            "title": record.get("title") or "Meeting",
            "meet_code": record.get("meet_code"),
            "category": record.get("category") or "unknown",
            "confidence": record.get("confidence"),
            "team_key": record.get("team_key"),
            "participants_json": participants_json,
            "speaker_map_json": speaker_map_json,
            "routing_status": record.get("routing_status") or "pending",
            "vault_path": record.get("vault_path"),
            "session_dir": record.get("session_dir"),
            "audio_paths_json": audio_paths_json,
            "asr_model": record.get("asr_model"),
            "diarization_model": record.get("diarization_model"),
            "summary_model": record.get("summary_model"),
            "pipeline_stage": record.get("pipeline_stage") or "unknown",
            "summary_md": record.get("summary_md") or "",
            "transcript_md": record.get("transcript_md") or "",
            "created_at": created_at,
            "updated_at": updated_at,
        }

        conn.execute(
            """
            INSERT INTO meetings (
                session_id, source, trigger, started_at, date, duration_ms, title, meet_code,
                category, confidence, team_key, participants_json, speaker_map_json,
                routing_status, vault_path, session_dir, audio_paths_json,
                asr_model, diarization_model, summary_model, pipeline_stage,
                summary_md, transcript_md, created_at, updated_at
            ) VALUES (
                :session_id, :source, :trigger, :started_at, :date, :duration_ms, :title, :meet_code,
                :category, :confidence, :team_key, :participants_json, :speaker_map_json,
                :routing_status, :vault_path, :session_dir, :audio_paths_json,
                :asr_model, :diarization_model, :summary_model, :pipeline_stage,
                :summary_md, :transcript_md, :created_at, :updated_at
            )
            ON CONFLICT(session_id) DO UPDATE SET
                source = excluded.source,
                trigger = excluded.trigger,
                started_at = excluded.started_at,
                date = excluded.date,
                duration_ms = excluded.duration_ms,
                title = excluded.title,
                meet_code = excluded.meet_code,
                category = excluded.category,
                confidence = excluded.confidence,
                team_key = excluded.team_key,
                participants_json = excluded.participants_json,
                speaker_map_json = excluded.speaker_map_json,
                routing_status = excluded.routing_status,
                vault_path = excluded.vault_path,
                session_dir = excluded.session_dir,
                audio_paths_json = excluded.audio_paths_json,
                asr_model = excluded.asr_model,
                diarization_model = excluded.diarization_model,
                summary_model = excluded.summary_model,
                pipeline_stage = excluded.pipeline_stage,
                summary_md = excluded.summary_md,
                transcript_md = excluded.transcript_md,
                updated_at = excluded.updated_at
            """,
            values,
        )

        if segments is not None:
            conn.execute("DELETE FROM segments WHERE session_id = ?", (session_id,))
            conn.executemany(
                """
                INSERT INTO segments (
                    session_id, start_ms, end_ms, track, raw_speaker, speaker, language, text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        session_id,
                        int(seg.get("start_ms") or 0),
                        int(seg.get("end_ms") or seg.get("start_ms") or 0),
                        _scalar_str(seg.get("track"), allow_none=True),
                        _scalar_str(seg.get("raw_speaker") or seg.get("label") or seg.get("speaker"), allow_none=True),
                        _scalar_str(seg.get("speaker") or seg.get("label") or seg.get("track"), allow_none=True),
                        _scalar_str(seg.get("language"), allow_none=True),
                        _scalar_str(seg.get("text")),
                    )
                    for seg in segments
                ],
            )

        part_names = _decode_json(participants_json, [])
        participants_text = ", ".join(str(p) for p in part_names if p)
        conn.execute("DELETE FROM meetings_fts WHERE session_id = ?", (session_id,))
        conn.execute(
            """
            INSERT INTO meetings_fts (
                session_id, title, participants, category, team_key, summary_md, transcript_md
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                values["title"],
                participants_text,
                values["category"] or "",
                values["team_key"] or "",
                values["summary_md"],
                values["transcript_md"],
            ),
        )
    return session_id


def list_meetings(
    db_path: str | Path | None = None,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    category: str | None = None,
    team_key: str | None = None,
    routing_status: str | None = None,
    query: str | None = None,
    limit: int = 20,
) -> list[dict]:
    if query and query.strip():
        return search_meetings(
            db_path,
            query=query,
            category=category,
            start_date=start_date,
            end_date=end_date,
            routing_status=routing_status,
            limit=limit,
        )
    init_db(db_path)
    clauses = []
    params: list = []
    if start_date:
        clauses.append("date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("date <= ?")
        params.append(end_date)
    if category:
        clauses.append("category = ?")
        params.append(category)
    if team_key:
        clauses.append("team_key = ?")
        params.append(team_key)
    if routing_status:
        clauses.append("routing_status = ?")
        params.append(routing_status)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT session_id, source, trigger, started_at, date, duration_ms, title, meet_code,
               category, confidence, team_key, participants_json, speaker_map_json,
               routing_status, vault_path, session_dir, audio_paths_json,
               asr_model, diarization_model, summary_model, pipeline_stage,
               created_at, updated_at
        FROM meetings
        {where}
        ORDER BY COALESCE(started_at, date, created_at) DESC
        LIMIT ?
    """
    params.append(max(1, int(limit)))
    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_meeting(row) for row in rows]


def get_meeting(
    db_path: str | Path | None,
    session_id: str,
    *,
    include_segments: bool = False,
    include_transcript: bool = True,
) -> dict | None:
    init_db(db_path)
    cols = (
        "session_id, source, trigger, started_at, date, duration_ms, title, meet_code, "
        "category, confidence, team_key, participants_json, speaker_map_json, "
        "routing_status, vault_path, session_dir, audio_paths_json, "
        "asr_model, diarization_model, summary_model, pipeline_stage, "
        "summary_md, transcript_md, created_at, updated_at"
    )
    with connect(db_path) as conn:
        row = conn.execute(
            f"SELECT {cols} FROM meetings WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None and len(session_id) >= 4:
            candidates = conn.execute(
                f"SELECT {cols} FROM meetings WHERE session_id LIKE ? OR session_id LIKE ?",
                (f"{session_id}%", f"%{session_id}"),
            ).fetchall()
            if len(candidates) == 1:
                row = candidates[0]
        if row is None:
            return None
        item = _row_to_meeting(row)
        if not include_transcript:
            item.pop("transcript_md", None)
        if include_segments:
            segs = conn.execute(
                """
                SELECT start_ms, end_ms, track, raw_speaker, speaker, language, text
                FROM segments
                WHERE session_id = ?
                ORDER BY start_ms ASC, id ASC
                """,
                (item["session_id"],),
            ).fetchall()
            item["segments"] = [dict(s) for s in segs]
        return item


def search_meetings(
    db_path: str | Path | None,
    query: str,
    *,
    category: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    routing_status: str | None = None,
    limit: int = 10,
) -> list[dict]:
    init_db(db_path)
    fts_q = _fts_query(query)
    if not fts_q:
        return list_meetings(
            db_path,
            start_date=start_date,
            end_date=end_date,
            category=category,
            routing_status=routing_status,
            limit=limit,
        )
    clauses = ["meetings_fts MATCH ?"]
    params: list = [fts_q]
    if category:
        clauses.append("m.category = ?")
        params.append(category)
    if start_date:
        clauses.append("m.date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("m.date <= ?")
        params.append(end_date)
    if routing_status:
        clauses.append("m.routing_status = ?")
        params.append(routing_status)

    sql = f"""
        SELECT m.session_id, m.source, m.trigger, m.started_at, m.date, m.duration_ms,
               m.title, m.meet_code, m.category, m.confidence, m.team_key,
               m.participants_json, m.speaker_map_json, m.routing_status, m.vault_path,
               m.session_dir, m.audio_paths_json, m.asr_model, m.diarization_model,
               m.summary_model, m.pipeline_stage, m.created_at, m.updated_at,
               snippet(meetings_fts, -1, '[', ']', '...', 18) AS match_snippet,
               bm25(meetings_fts) AS rank
        FROM meetings_fts
        JOIN meetings m ON m.session_id = meetings_fts.session_id
        WHERE {' AND '.join(clauses)}
        ORDER BY rank ASC, COALESCE(m.started_at, m.date) DESC
        LIMIT ?
    """
    params.append(max(1, int(limit)))
    with connect(db_path) as conn:
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            like = f"%{query.strip()}%"
            rows = conn.execute(
                """
                SELECT session_id, source, trigger, started_at, date, duration_ms, title,
                       meet_code, category, confidence, team_key, participants_json,
                       speaker_map_json, routing_status, vault_path, session_dir,
                       audio_paths_json, asr_model, diarization_model, summary_model,
                       pipeline_stage, created_at, updated_at, '' AS match_snippet, 0.0 AS rank
                FROM meetings
                WHERE title LIKE ? OR summary_md LIKE ? OR transcript_md LIKE ?
                ORDER BY COALESCE(started_at, date) DESC
                LIMIT ?
                """,
                (like, like, like, max(1, int(limit))),
            ).fetchall()
    return [_row_to_meeting(row) for row in rows]


def log_dictation(
    db_path: str | Path | None = None,
    *,
    raw_text: str,
    polished_text: str | None = None,
    app_name: str = "",
    app_bundle_id: str = "",
    model: str = "",
    audio_seconds: float = 0.0,
    created_at: str | None = None,
) -> int:
    init_db(db_path)
    ts = created_at or utc_now_iso()
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO dictations (
                created_at, app_name, app_bundle_id, model, audio_seconds, raw_text, polished_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                app_name or "",
                app_bundle_id or "",
                model or "",
                float(audio_seconds or 0.0),
                raw_text,
                polished_text,
            ),
        )
        dict_id = int(cur.lastrowid)
        conn.execute(
            """
            INSERT INTO dictations_fts (dictation_id, app_name, raw_text, polished_text)
            VALUES (?, ?, ?, ?)
            """,
            (dict_id, app_name or "", raw_text, polished_text or ""),
        )
    return dict_id


def search_dictations(
    db_path: str | Path | None = None,
    query: str = "",
    *,
    limit: int = 20,
) -> list[dict]:
    init_db(db_path)
    fts_q = _fts_query(query)
    with connect(db_path) as conn:
        if not fts_q:
            rows = conn.execute(
                """
                SELECT id, created_at, app_name, app_bundle_id, model, audio_seconds, raw_text, polished_text
                FROM dictations
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT d.id, d.created_at, d.app_name, d.app_bundle_id, d.model, d.audio_seconds,
                       d.raw_text, d.polished_text, bm25(dictations_fts) AS rank
                FROM dictations_fts
                JOIN dictations d ON d.id = dictations_fts.dictation_id
                WHERE dictations_fts MATCH ?
                ORDER BY rank ASC, d.created_at DESC
                LIMIT ?
                """,
                (fts_q, max(1, int(limit))),
            ).fetchall()
    return [dict(row) for row in rows]
