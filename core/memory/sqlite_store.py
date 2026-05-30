import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    trace_id: str
    created_at: str
    updated_at: str
    memory_type: str
    importance: str
    workspace_root: Optional[str]
    project_scope: Optional[str]
    source_component: Optional[str]
    user_intent: Optional[str]
    summary: str
    content_preview: Optional[str]
    metadata_json: str
    embedding_json: Optional[str]
    embedding_model: Optional[str]
    token_estimate: int
    access_count: int
    last_accessed_at: Optional[str]


class SQLiteMemoryStore:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path).expanduser()

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_items (
                    id TEXT PRIMARY KEY,
                    trace_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    importance TEXT NOT NULL,
                    workspace_root TEXT,
                    project_scope TEXT,
                    source_component TEXT,
                    user_intent TEXT,
                    summary TEXT NOT NULL,
                    content_preview TEXT,
                    metadata_json TEXT,
                    embedding_json TEXT,
                    embedding_model TEXT,
                    token_estimate INTEGER,
                    access_count INTEGER DEFAULT 0,
                    last_accessed_at TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_events (
                    id TEXT PRIMARY KEY,
                    memory_id TEXT,
                    event_type TEXT,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_items_trace_id ON memory_items(trace_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_items_created_at ON memory_items(created_at);")
            conn.commit()
        finally:
            conn.close()

    def close(self) -> None:
        return None

    def insert_memory(self, item: Dict[str, Any]) -> None:
        now = utc_now()
        values = {
            "created_at": now,
            "updated_at": now,
            "metadata_json": "{}",
            "token_estimate": 0,
            "access_count": 0,
            "last_accessed_at": None,
            **item,
        }
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_items (
                    id, trace_id, created_at, updated_at, memory_type, importance,
                    workspace_root, project_scope, source_component, user_intent,
                    summary, content_preview, metadata_json, embedding_json,
                    embedding_model, token_estimate, access_count, last_accessed_at
                )
                VALUES (
                    :id, :trace_id, :created_at, :updated_at, :memory_type, :importance,
                    :workspace_root, :project_scope, :source_component, :user_intent,
                    :summary, :content_preview, :metadata_json, :embedding_json,
                    :embedding_model, :token_estimate, :access_count, :last_accessed_at
                );
                """,
                values,
            )
            conn.commit()
        finally:
            conn.close()

    def list_memories(self, limit: int = 1000) -> list[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM memory_items ORDER BY created_at DESC LIMIT ?;",
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        return [dict(row) for row in rows]

    def iter_searchable_memories(self, limit: int = 1000) -> Iterable[Dict[str, Any]]:
        return self.list_memories(limit=limit)

    def mark_accessed(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        now = utc_now()
        conn = self._connect()
        try:
            conn.executemany(
                """
                UPDATE memory_items
                SET access_count = access_count + 1,
                    last_accessed_at = ?
                WHERE id = ?;
                """,
                [(now, memory_id) for memory_id in memory_ids],
            )
            conn.commit()
        finally:
            conn.close()

    def counts(self) -> Dict[str, int]:
        conn = self._connect()
        try:
            item_count = conn.execute("SELECT COUNT(*) FROM memory_items;").fetchone()[0]
            embedded_count = conn.execute(
                "SELECT COUNT(*) FROM memory_items WHERE embedding_json IS NOT NULL;"
            ).fetchone()[0]
        finally:
            conn.close()
        return {"item_count": int(item_count), "embedded_count": int(embedded_count)}

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn


def decode_embedding(row: Dict[str, Any]) -> Optional[list[float]]:
    raw = row.get("embedding_json")
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list):
        return None
    return [float(item) for item in value]
