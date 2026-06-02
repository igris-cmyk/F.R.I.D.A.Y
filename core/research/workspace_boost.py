import sqlite3
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from core.workspace.indexer import WorkspaceIndexer
from core.workspace.models import DEFAULT_WORKSPACE_INDEX_DB_PATH


MAX_WORKSPACE_INDEX_BOOST = 60.0


@dataclass(frozen=True)
class WorkspaceBoost:
    path: str
    score: float
    reasons: list[str]
    role_tags: list[str] = field(default_factory=list)
    summary: str | None = None
    symbols: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WorkspaceBoostDiagnostics:
    available: bool
    applied_count: int = 0
    top_path: str | None = None
    top_score: float = 0.0
    reason: str = "unavailable"
    stale: bool = False


class WorkspaceIndexBoost:
    """Optional research-ranking adapter over the deterministic workspace index."""

    def __init__(self, workspace_root: str | Path, db_path: str | Path | None = None):
        self.workspace_root = Path(workspace_root).resolve()
        if db_path is None:
            self.db_path = self.workspace_root / DEFAULT_WORKSPACE_INDEX_DB_PATH
        else:
            candidate = Path(db_path)
            self.db_path = candidate if candidate.is_absolute() else self.workspace_root / candidate
        self.indexer = WorkspaceIndexer(self.workspace_root, db_path=self.db_path)
        self.last_diagnostics = WorkspaceBoostDiagnostics(available=False)

    def available(self) -> bool:
        if not self.db_path.exists():
            self.last_diagnostics = WorkspaceBoostDiagnostics(
                available=False,
                reason="missing_index_db",
            )
            return False

        try:
            status = self.indexer.status()
        except (OSError, RuntimeError, sqlite3.Error, ValueError) as exc:
            self.last_diagnostics = WorkspaceBoostDiagnostics(
                available=False,
                reason=f"index_unavailable:{type(exc).__name__}",
            )
            return False

        if status.get("migration_status") != "ok":
            self.last_diagnostics = WorkspaceBoostDiagnostics(
                available=False,
                reason="migration_not_ok",
            )
            return False
        if not status.get("latest_run") or int(status.get("file_count", 0)) <= 0:
            self.last_diagnostics = WorkspaceBoostDiagnostics(
                available=False,
                reason="empty_index",
            )
            return False

        self.last_diagnostics = WorkspaceBoostDiagnostics(
            available=True,
            reason="ok",
        )
        return True

    def search(self, query: str, limit: int = 20) -> list[WorkspaceBoost]:
        if not self.available():
            return []

        try:
            results = self.indexer.search(query, limit=limit)
        except (OSError, RuntimeError, sqlite3.Error, ValueError) as exc:
            self.last_diagnostics = WorkspaceBoostDiagnostics(
                available=False,
                reason=f"search_failed:{type(exc).__name__}",
            )
            return []

        boosts: list[WorkspaceBoost] = []
        for result in results:
            if not self._safe_existing_workspace_file(result.path):
                continue
            record = self._safe_show(result.path)
            boosts.append(
                WorkspaceBoost(
                    path=result.path,
                    score=normalize_index_score(result.score),
                    reasons=[f"workspace_index:{reason}" for reason in result.reasons],
                    role_tags=list(result.role_tags),
                    summary=result.summary,
                    symbols=[symbol.name for symbol in record.symbols] if record else [],
                )
            )

        top = boosts[0] if boosts else None
        self.last_diagnostics = WorkspaceBoostDiagnostics(
            available=True,
            applied_count=len(boosts),
            top_path=top.path if top else None,
            top_score=top.score if top else 0.0,
            reason="ok",
        )
        return boosts

    def boost_map(self, query: str, limit: int = 20) -> dict[str, WorkspaceBoost]:
        return {boost.path: boost for boost in self.search(query, limit=limit)}

    def _safe_show(self, path: str):
        try:
            return self.indexer.show(path)
        except (OSError, RuntimeError, sqlite3.Error, ValueError):
            return None

    def _safe_existing_workspace_file(self, raw_path: str) -> bool:
        pure_path = PurePosixPath(raw_path.replace("\\", "/"))
        if pure_path.is_absolute() or ".." in pure_path.parts:
            return False
        try:
            candidate = (self.workspace_root / pure_path.as_posix()).resolve()
            candidate.relative_to(self.workspace_root)
        except (OSError, ValueError):
            return False
        return candidate.is_file()


def normalize_index_score(score: float) -> float:
    return round(min(MAX_WORKSPACE_INDEX_BOOST, max(0.0, 8.0 + (float(score) / 6.0))), 3)
