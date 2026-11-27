#!/usr/bin/env python3
"""Local stdio MCP server for Fluid Dictation meetings & dictation archive.

Pure Python stdlib (JSON-RPC 2.0 over newline-delimited stdio). Never prints
non-JSON lines to stdout; diagnostics go to stderr.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HOTKEY_DIR = Path(__file__).resolve().parent
if str(HOTKEY_DIR) not in sys.path:
    sys.path.insert(0, str(HOTKEY_DIR))

import archive_db  # noqa: E402
import meeting_pipeline as mp  # noqa: E402
import meetings_cli  # noqa: E402

SERVER_INFO = {"name": "fluid-meetings", "version": "1.0.0"}
SUPPORTED_PROTOCOL = "2024-11-05"

TOOLS = [
    {
        "name": "list_meetings",
        "description": "List locally recorded meetings from Fluid Dictation archive, optionally filtered by date, category, EF team_key, routing_status, or search query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "Inclusive YYYY-MM-DD lower bound"},
                "end_date": {"type": "string", "description": "Inclusive YYYY-MM-DD upper bound"},
                "category": {
                    "type": "string",
                    "enum": list(mp.VALID_CATEGORIES),
                    "description": "Meeting category filter",
                },
                "team_key": {"type": "string", "description": "EF team registry key filter"},
                "routing_status": {
                    "type": "string",
                    "description": "e.g. needs_review, routed_personal_wiki, routed_edge_inference, archive_only",
                },
                "query": {"type": "string", "description": "Optional full-text query"},
                "limit": {"type": "integer", "default": 20},
            },
        },
    },
    {
        "name": "get_meeting",
        "description": "Fetch full metadata, summary, and optionally raw transcript for a meeting by session_id (full UUID or unambiguous prefix/suffix).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "include_transcript": {"type": "boolean", "default": True},
                "include_segments": {"type": "boolean", "default": False},
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "search_meetings",
        "description": "Full-text search across meeting titles, participants, summaries, and transcripts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "category": {"type": "string", "enum": list(mp.VALID_CATEGORIES)},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "reclassify_meeting",
        "description": "Manually reclassify a meeting (e.g. from needs_review) and optionally route its markdown note to Personal_Wiki or Edge Inference Vault.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "category": {"type": "string", "enum": list(mp.VALID_CATEGORIES)},
                "title": {"type": "string"},
                "team_key": {"type": "string"},
                "route": {"type": "boolean", "default": False},
            },
            "required": ["session_id", "category"],
        },
    },
    {
        "name": "search_dictations",
        "description": "Search local Fn dictation history (raw and polished text, active app).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "default": ""},
                "limit": {"type": "integer", "default": 20},
            },
        },
    },
]


def _default_db_path() -> Path:
    settings = mp.load_settings()
    return archive_db.resolve_db_path(
        settings.get("archive_db") or settings.get("configured_archive_db")
    )


def call_tool(name: str, arguments: dict | None, *, db_path: Path | None = None, settings: dict | None = None) -> dict:
    args = arguments or {}
    db = db_path or _default_db_path()

    if name == "list_meetings":
        rows = archive_db.list_meetings(
            db,
            start_date=args.get("start_date"),
            end_date=args.get("end_date"),
            category=args.get("category"),
            team_key=args.get("team_key"),
            routing_status=args.get("routing_status"),
            query=args.get("query"),
            limit=int(args.get("limit") or 20),
        )
        return {"meetings": rows}

    if name == "get_meeting":
        session_id = str(args["session_id"])
        rec = archive_db.get_meeting(
            db,
            session_id,
            include_transcript=bool(args.get("include_transcript", True)),
            include_segments=bool(args.get("include_segments", False)),
        )
        if rec is None:
            raise ValueError(f"Meeting not found: {session_id}")
        return rec

    if name == "search_meetings":
        rows = archive_db.search_meetings(
            db,
            query=str(args["query"]),
            category=args.get("category"),
            start_date=args.get("start_date"),
            end_date=args.get("end_date"),
            limit=int(args.get("limit") or 10),
        )
        return {"results": rows}

    if name == "reclassify_meeting":
        return meetings_cli.reclassify_and_route(
            session_id=str(args["session_id"]),
            category=str(args["category"]),
            title=args.get("title"),
            team_key=args.get("team_key"),
            route=bool(args.get("route", False)),
            db_path=db,
            settings=settings or mp.load_settings(),
        )

    if name == "search_dictations":
        rows = archive_db.search_dictations(
            db,
            query=str(args.get("query") or ""),
            limit=int(args.get("limit") or 20),
        )
        return {"dictations": rows}

    raise KeyError(f"Unknown tool: {name}")


def handle_request(message: dict, *, db_path: Path | None = None, settings: dict | None = None) -> dict | None:
    req_id = message.get("id")
    method = message.get("method")
    params = message.get("params") or {}

    # Notifications have no id and expect no response.
    if req_id is None:
        return None

    if method == "initialize":
        client_proto = str(params.get("protocolVersion") or SUPPORTED_PROTOCOL)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": client_proto,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        tool_name = str(params.get("name") or "")
        tool_args = params.get("arguments") or {}
        try:
            payload = call_tool(tool_name, tool_args, db_path=db_path, settings=settings)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}],
                    "isError": False,
                },
            }
        except KeyError as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": str(exc)},
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def serve_stdio(stdin=None, stdout=None, *, db_path: Path | None = None) -> None:
    inp = stdin or sys.stdin
    out = stdout or sys.stdout
    for raw_line in inp:
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError as exc:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Parse error: {exc}"}}
            out.write(json.dumps(resp) + "\n")
            out.flush()
            continue
        resp = handle_request(message, db_path=db_path)
        if resp is not None:
            out.write(json.dumps(resp, ensure_ascii=False) + "\n")
            out.flush()


if __name__ == "__main__":
    serve_stdio()
