import asyncio
import json
import logging
import math
import re
import urllib.request
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from core.config import (
    FRIDAY_EMBEDDING_MODEL,
    FRIDAY_MEMORY_BACKEND,
    FRIDAY_MEMORY_DB_PATH,
    OLLAMA_BASE_URL,
)
from core.memory.redaction import redact_text
from core.memory.sqlite_store import SQLiteMemoryStore, decode_embedding

logger = logging.getLogger("friday.memory.manager")


class MemoryHealthState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    RECOVERING = "recovering"


class MemoryImportance(Enum):
    TRANSIENT = "transient"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    CRITICAL = "critical"


class MemoryType(Enum):
    EPISODIC = "EPISODIC"
    SEMANTIC = "SEMANTIC"
    PROJECT = "PROJECT"
    DEBUG = "DEBUG"
    USER_PREFERENCE = "USER_PREFERENCE"


class MemoryManager:
    """Public persistence boundary for FRIDAY memory."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        embedding_model: str = FRIDAY_EMBEDDING_MODEL,
        ollama_base_url: str = OLLAMA_BASE_URL,
        backend: str = FRIDAY_MEMORY_BACKEND,
    ):
        self.requested_backend = (backend or "sqlite").lower()
        self.backend = "sqlite"
        self.db_path = db_path or FRIDAY_MEMORY_DB_PATH
        self.embedding_model = embedding_model
        self.ollama_base_url = ollama_base_url
        self.store = SQLiteMemoryStore(self.db_path)
        self.health_state: MemoryHealthState = MemoryHealthState.OFFLINE
        self.degraded_reason: Optional[str] = None
        self.embedding_available = False
        self.max_queue_size = 100
        self.retry_queue = asyncio.Queue(maxsize=self.max_queue_size)

    async def initialize(self) -> None:
        if self.requested_backend not in {"sqlite", "auto", "postgres", "postgresql"}:
            logger.warning(
                "MemoryManager: Unknown backend '%s'; falling back to SQLite.",
                self.requested_backend,
            )

        if self.requested_backend in {"postgres", "postgresql"}:
            logger.warning(
                "MemoryManager: PostgreSQL backend requested but local SQLite is the active runtime backend; "
                "falling back to SQLite."
            )

        try:
            self.store.initialize()
            self.embedding_available = await self._check_embedding_model_available()
            self.health_state = MemoryHealthState.HEALTHY if self.embedding_available else MemoryHealthState.DEGRADED
            self.degraded_reason = None if self.embedding_available else f"embedding_model_unavailable:{self.embedding_model}"
            self.backend = "sqlite"
            logger.info(
                "MemoryManager: SQLite backend active path=%s health=%s",
                self.db_path,
                self.health_state.value,
            )
        except Exception as exc:
            self.health_state = MemoryHealthState.OFFLINE
            self.degraded_reason = str(exc)
            logger.warning("MemoryManager SQLite initialization failed: %s", exc)

    async def connect(self) -> None:
        await self.initialize()

    async def close(self) -> None:
        self.store.close()

    async def health(self) -> Dict[str, Any]:
        counts = {"item_count": 0, "embedded_count": 0}
        migration_health = self.store.migration_health()
        if self.health_state != MemoryHealthState.OFFLINE:
            try:
                counts = self.store.counts()
            except Exception as exc:
                self.health_state = MemoryHealthState.DEGRADED
                self.degraded_reason = str(exc)
        if migration_health.get("migration_status") in {"failed", "unsupported"}:
            self.degraded_reason = migration_health.get("migration_error") or self.degraded_reason
        return {
            "backend": self.backend,
            "requested_backend": self.requested_backend,
            "db_path": str(Path(self.db_path)),
            "item_count": counts["item_count"],
            "embedded_count": counts["embedded_count"],
            "health_state": self.health_state.value,
            "degraded_reason": self.degraded_reason,
            "embedding_model": self.embedding_model,
            "embedding_available": self.embedding_available,
            **migration_health,
        }

    async def persist_episodic_trace(
        self,
        trace_id: str,
        intent: str,
        importance: MemoryImportance,
        workflow_summary: str,
        environment_context: Dict[str, Any],
        metadata: Dict[str, Any],
        embedding: Optional[list[float]] = None,
    ) -> Dict[str, Any]:
        if importance == MemoryImportance.TRANSIENT:
            return {"persisted": False, "embedded": False, "degraded_reason": "transient", "memory_id": None}

        if self.health_state == MemoryHealthState.OFFLINE:
            return {
                "persisted": False,
                "embedded": False,
                "degraded_reason": self.degraded_reason or "offline",
                "memory_id": None,
            }

        summary = redact_text(workflow_summary, max_chars=2000)
        content_preview = redact_text(metadata.get("result_preview", workflow_summary), max_chars=4000)
        embedded = False
        degraded_reason = None

        if embedding is None and self.embedding_available:
            embedding = await self.generate_embedding(summary)
            if embedding is None:
                degraded_reason = f"embedding_failed:{self.embedding_model}"

        if embedding:
            embedded = True

        env = environment_context or {}
        now = datetime.now(timezone.utc).isoformat()
        memory_id = str(uuid.uuid4())
        item = {
            "id": memory_id,
            "trace_id": trace_id,
            "created_at": now,
            "updated_at": now,
            "memory_type": self._memory_type_for_importance(importance).value,
            "importance": importance.value,
            "workspace_root": env.get("working_directory") or env.get("workspace_root"),
            "project_scope": env.get("project") or env.get("active_app"),
            "source_component": metadata.get("source_component") or metadata.get("intent_type") or intent,
            "user_intent": metadata.get("command") or intent,
            "summary": summary,
            "content_preview": content_preview,
            "metadata_json": json.dumps(self._redact_metadata(metadata), sort_keys=True),
            "embedding_json": json.dumps(embedding) if embedding else None,
            "embedding_model": self.embedding_model if embedding else None,
            "token_estimate": max(1, len(summary.split())),
            "access_count": 0,
            "last_accessed_at": None,
        }

        try:
            self.store.insert_memory(item)
            if degraded_reason:
                self.health_state = MemoryHealthState.DEGRADED
                self.degraded_reason = degraded_reason
            logger.info("MemoryManager: Persisted memory item %s trace=%s backend=%s", memory_id, trace_id, self.backend)
            return {
                "persisted": True,
                "embedded": embedded,
                "degraded_reason": degraded_reason,
                "memory_id": memory_id,
            }
        except Exception as exc:
            self.health_state = MemoryHealthState.DEGRADED
            self.degraded_reason = str(exc)
            logger.warning("Memory persistence failed for trace %s: %s", trace_id, exc)
            return {"persisted": False, "embedded": False, "degraded_reason": str(exc), "memory_id": None}

    async def retrieve_relevant_context(
        self,
        query: str | list[float],
        limit: int = 5,
        min_score: float = 0.15,
    ) -> list[Dict[str, Any]]:
        if self.health_state == MemoryHealthState.OFFLINE:
            return []

        rows = list(self.store.iter_searchable_memories(limit=1000))
        if not rows:
            return []

        query_embedding = query if isinstance(query, list) else None
        query_text = "" if isinstance(query, list) else str(query)
        if query_embedding is None and self.embedding_available:
            query_embedding = await self.generate_embedding(query_text)

        if query_embedding:
            scored = self._score_by_embedding(query_embedding, rows)
        else:
            scored = self._score_by_keywords(query_text, rows)

        filtered = [item for item in scored if item["score"] >= min_score]
        results = filtered[:limit]
        self.store.mark_accessed([item["id"] for item in results])
        return results

    async def generate_embedding(self, text: str) -> Optional[list[float]]:
        if not text:
            return None
        try:
            return self._generate_embedding_sync(text[:2000])
        except Exception as exc:
            logger.warning("Embedding generation failed: %s", exc)
            self.embedding_available = False
            if self.health_state == MemoryHealthState.HEALTHY:
                self.health_state = MemoryHealthState.DEGRADED
            self.degraded_reason = f"embedding_failed:{exc}"
            return None

    def _generate_embedding_sync(self, text: str) -> list[float]:
        url = f"{self.ollama_base_url.rstrip('/')}/api/embeddings"
        payload = json.dumps({"model": self.embedding_model, "prompt": text}).encode("utf-8")
        request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=5.0) as response:
            body = json.loads(response.read().decode("utf-8"))
        embedding = body.get("embedding")
        if not isinstance(embedding, list):
            raise ValueError("Ollama embeddings response did not include an embedding list.")
        return [float(value) for value in embedding]

    async def _check_embedding_model_available(self) -> bool:
        try:
            return self._check_embedding_model_available_sync()
        except Exception as exc:
            logger.info("Embedding model availability check failed: %s", exc)
            return False

    def _check_embedding_model_available_sync(self) -> bool:
        url = f"{self.ollama_base_url.rstrip('/')}/api/tags"
        with urllib.request.urlopen(url, timeout=1.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        for item in payload.get("models", []):
            name = item.get("model") or item.get("name")
            if name == self.embedding_model or name == f"{self.embedding_model}:latest":
                return True
        return False

    def _score_by_embedding(self, query_embedding: list[float], rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        scored = []
        for row in rows:
            embedding = decode_embedding(row)
            if not embedding:
                continue
            score = cosine_similarity(query_embedding, embedding)
            scored.append(self._result_from_row(row, score))
        return sorted(scored, key=lambda item: (-item["score"], item["created_at"]))

    def _score_by_keywords(self, query: str, rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        terms = extract_memory_keywords(query)
        if not terms:
            return []
        scored = []
        for row in rows:
            haystack = f"{row.get('summary', '')} {row.get('content_preview', '')} {row.get('user_intent', '')}".lower()
            matches = sum(1 for term in terms if term in haystack)
            if matches:
                scored.append(self._result_from_row(row, matches / max(len(terms), 1)))
        return sorted(scored, key=lambda item: (-item["score"], item["created_at"]))

    def _result_from_row(self, row: Dict[str, Any], score: float) -> Dict[str, Any]:
        metadata = {}
        try:
            metadata = json.loads(row.get("metadata_json") or "{}")
        except json.JSONDecodeError:
            metadata = {}
        return {
            "id": row["id"],
            "trace_id": row.get("trace_id"),
            "score": round(float(score), 6),
            "summary": row.get("summary"),
            "workflow_summary": row.get("summary"),
            "memory_type": row.get("memory_type"),
            "importance": row.get("importance"),
            "created_at": row.get("created_at"),
            "source_metadata": metadata,
            "intent": row.get("user_intent") or row.get("source_component"),
        }

    def _redact_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        redacted: Dict[str, Any] = {}
        for key, value in (metadata or {}).items():
            if isinstance(value, str):
                redacted[key] = redact_text(value, max_chars=1000)
            else:
                redacted[key] = value
        return redacted

    def _memory_type_for_importance(self, importance: MemoryImportance) -> MemoryType:
        if importance == MemoryImportance.SEMANTIC:
            return MemoryType.SEMANTIC
        if importance == MemoryImportance.CRITICAL:
            return MemoryType.DEBUG
        return MemoryType.EPISODIC


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


MEMORY_STOPWORDS = {
    "about",
    "again",
    "did",
    "does",
    "earlier",
    "happen",
    "happened",
    "inspect",
    "inspected",
    "just",
    "recent",
    "recently",
    "recall",
    "remind",
    "summarize",
    "the",
    "this",
    "what",
    "when",
    "with",
    "work",
    "worked",
}


def extract_memory_keywords(query: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9_./-]+", query.lower()))
    return {
        token
        for token in tokens
        if len(token) > 2 and token not in MEMORY_STOPWORDS
    }


memory_manager = MemoryManager()
