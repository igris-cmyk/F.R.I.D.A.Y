import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from core.memory.migrations import (
    CURRENT_SCHEMA_VERSION,
    MigrationError,
    UnsupportedSchemaVersion,
    ensure_schema,
    get_schema_version,
)


logger = logging.getLogger("friday.memory.sqlite_store")


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
        self.schema_version: Optional[int] = None
        self.target_schema_version = CURRENT_SCHEMA_VERSION
        self.migration_status = "unknown"
        self.migration_error: Optional[str] = None

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            self.schema_version = ensure_schema(conn, target_version=self.target_schema_version)
            self.migration_status = "ok"
            self.migration_error = None
        except UnsupportedSchemaVersion as exc:
            self.schema_version = self._safe_schema_version(conn)
            self.migration_status = "unsupported"
            self.migration_error = str(exc)
            raise
        except MigrationError as exc:
            self.schema_version = self._safe_schema_version(conn)
            self.migration_status = "failed"
            self.migration_error = str(exc)
            raise
        except Exception as exc:
            self.schema_version = self._safe_schema_version(conn)
            self.migration_status = "failed"
            self.migration_error = str(exc)
            logger.exception("SQLiteMemoryStore: migration failed: %s", exc)
            raise
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

    def migration_health(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_schema_version": self.target_schema_version,
            "migration_status": self.migration_status,
            "migration_error": self.migration_error,
        }

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _safe_schema_version(self, conn: sqlite3.Connection) -> Optional[int]:
        try:
            return get_schema_version(conn)
        except Exception:
            return None


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
