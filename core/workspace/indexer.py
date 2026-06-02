import hashlib
import os
import re
from pathlib import Path

from core.workspace.extractors import extract_file_record
from core.workspace.models import (
    DEFAULT_WORKSPACE_INDEX_DB_PATH,
    EXCLUDED_DIRS,
    INDEXABLE_EXTENSIONS,
    SECRET_EXTENSIONS,
    SECRET_FILENAMES,
    WorkspaceFileRecord,
    WorkspaceSearchResult,
)
from core.workspace.store import WorkspaceIndexStore


MAX_INDEX_FILE_BYTES = 1024 * 1024
QUERY_TOKEN_RE = re.compile(r"[a-z0-9_]+")

DOMAIN_PRIORITIES = {
    "planner": [
        "core/agents/planner.py",
        "core/agents/router.py",
        "core/main.py",
        "core/capabilities/registry.py",
    ],
    "router": ["core/agents/router.py", "core/agents/planner.py", "core/main.py"],
    "memory": [
        "core/memory/manager.py",
        "core/memory/pipeline.py",
        "core/memory/retriever.py",
        "core/agents/memory_agent.py",
    ],
    "security": ["core/security/permissions.py", "core/security/approval.py", "core/main.py"],
    "approval": ["core/security/approval.py", "core/security/permissions.py", "core/main.py"],
    "capability": [
        "core/capabilities/executor.py",
        "core/capabilities/registry.py",
        "core/capabilities/contracts.py",
    ],
    "nats": ["core/main.py", "apps/desktop/src/main.js", "core/schemas/events.py"],
    "stream": ["core/main.py", "apps/desktop/src/main.js", "core/schemas/events.py"],
    "research": ["core/research/ranker.py", "core/research/context_builder.py", "core/agents/research.py"],
}


class WorkspaceIndexer:
    def __init__(
        self,
        workspace_root: str | Path,
        db_path: str | Path | None = None,
        store: WorkspaceIndexStore | None = None,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        if db_path is None:
            db_path = self.workspace_root / DEFAULT_WORKSPACE_INDEX_DB_PATH
        else:
            db_path = Path(db_path)
            if not db_path.is_absolute():
                db_path = self.workspace_root / db_path
        self.store = store or WorkspaceIndexStore(db_path)

    def build(self) -> dict:
        self.store.initialize()
        records: list[WorkspaceFileRecord] = []
        errors: list[dict[str, str]] = []
        for path in self._iter_indexable_paths():
            relative_path = self._relative_path(path)
            try:
                record = self._index_file(path, relative_path)
            except Exception as exc:
                errors.append({"path": relative_path, "error": str(exc)})
                continue
            if record:
                records.append(record)

        records.sort(key=lambda item: item.path)
        self.store.replace_index(str(self.workspace_root), records, errors)
        status = self.store.status()
        return {
            "workspace_root": str(self.workspace_root),
            "db_path": str(self.store.db_path),
            "indexed_files": len(records),
            "errors": errors,
            "schema_version": status["schema_version"],
        }

    def status(self) -> dict:
        self.store.initialize()
        return self.store.status()

    def search(self, query: str, limit: int = 10) -> list[WorkspaceSearchResult]:
        self.store.initialize()
        tokens = _query_tokens(query)
        docs = self.store.searchable_documents()
        results: list[WorkspaceSearchResult] = []
        for doc in docs:
            score, reasons = _score_document(tokens, query, doc)
            if score > 0:
                results.append(
                    WorkspaceSearchResult(
                        path=doc["path"],
                        score=round(score, 3),
                        role_tags=doc["roles"],
                        summary=doc["summary"],
                        reasons=reasons,
                    )
                )
        return sorted(results, key=lambda item: (-item.score, item.path))[:limit]

    def show(self, path: str) -> WorkspaceFileRecord | None:
        self.store.initialize()
        return self.store.get_file(path)

    def _iter_indexable_paths(self):
        for current_root, dirnames, filenames in os.walk(self.workspace_root, topdown=True):
            current_path = Path(current_root)
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if not self._is_excluded_directory(current_path / dirname)
            ]
            for filename in filenames:
                path = current_path / filename
                if self._should_index_file(path):
                    yield path

    def _is_excluded_directory(self, path: Path) -> bool:
        return path.name in EXCLUDED_DIRS or self._is_outside_workspace(path)

    def _should_index_file(self, path: Path) -> bool:
        if self._is_outside_workspace(path):
            return False
        if any(part in EXCLUDED_DIRS for part in path.relative_to(self.workspace_root).parts):
            return False
        if path.name in SECRET_FILENAMES or path.suffix.lower() in SECRET_EXTENSIONS:
            return False
        if path.suffix.lower() not in INDEXABLE_EXTENSIONS:
            return False
        try:
            return path.is_file() and path.stat().st_size <= MAX_INDEX_FILE_BYTES
        except OSError:
            return False

    def _is_outside_workspace(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.workspace_root)
            return False
        except ValueError:
            return True

    def _relative_path(self, path: Path) -> str:
        return path.resolve().relative_to(self.workspace_root).as_posix()

    def _index_file(self, path: Path, relative_path: str) -> WorkspaceFileRecord | None:
        stat = path.stat()
        content = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return extract_file_record(
            relative_path=relative_path,
            content=content,
            size=stat.st_size,
            mtime=stat.st_mtime,
            sha256=digest,
        )


def _query_tokens(query: str) -> set[str]:
    return {token for token in QUERY_TOKEN_RE.findall(query.lower()) if len(token) > 2}


def _score_document(tokens: set[str], query: str, doc: dict) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    path = doc["path"]
    path_lower = path.lower()
    role_tags = set(doc["roles"])
    symbols = {symbol.lower() for symbol in doc["symbols"]}
    imports = {item.lower() for item in doc["imports"]}
    subjects = {item.lower() for item in doc["nats_subjects"]}
    capabilities = {item.lower() for item in doc["capabilities"]}
    terms = set(doc["terms"])
    summary = doc["summary"].lower()

    for token in sorted(tokens):
        if token in role_tags:
            score += 50.0
            reasons.append(f"role:{token}")
        if token in path_lower:
            score += 35.0
            reasons.append(f"path:{token}")
        if token in symbols:
            score += 25.0
            reasons.append(f"symbol:{token}")
        if any(token in item for item in imports):
            score += 10.0
            reasons.append(f"import:{token}")
        if any(token in item for item in subjects):
            score += 22.0
            reasons.append(f"nats:{token}")
        if any(token in item for item in capabilities):
            score += 22.0
            reasons.append(f"capability:{token}")
        if token in terms:
            score += 8.0
            reasons.append(f"term:{token}")
        elif token in summary:
            score += 4.0
            reasons.append(f"summary:{token}")

    query_lower = query.lower()
    for trigger, priorities in DOMAIN_PRIORITIES.items():
        if trigger not in tokens and trigger not in query_lower:
            continue
        if path in priorities:
            score += 120.0 - (priorities.index(path) * 10.0)
            reasons.append(f"priority:{trigger}")
        elif trigger in role_tags:
            score += 30.0
            reasons.append(f"role_priority:{trigger}")

    if "subsystem" in tokens and "memory" in tokens and "memory" in role_tags:
        score += 20.0
        reasons.append("domain:memory_subsystem")
    if "policy" in tokens and "security" in role_tags:
        score += 20.0
        reasons.append("domain:security_policy")
    if "tests" not in tokens and "test" in role_tags:
        score -= 30.0
        reasons.append("penalty:test")

    deduped_reasons = list(dict.fromkeys(reasons))
    return max(score, 0.0), deduped_reasons
