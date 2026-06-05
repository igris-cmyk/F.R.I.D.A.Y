from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from core.research.context_builder import ContextBudget, context_budget_used
from core.research.ranker import RankedFile, extract_keywords, is_excluded_path
from core.research.workspace_boost import WorkspaceBoost
from core.workspace.models import WorkspaceFileRecord


LOCATION_QUERY_MARKERS = (
    "where is",
    "where are",
    "which files",
    "which file",
    "what files",
    "implemented",
    "define capabilities",
    "handle nats",
)


@dataclass(frozen=True)
class ContextSelectionDecision:
    path: str
    source: str
    reason: str
    budget_chars: int
    matched_terms: list[str] = field(default_factory=list)
    role_tags: list[str] = field(default_factory=list)
    summary: str | None = None
    symbols: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    nats_subjects: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OptimizedContextPlan:
    decisions: list[ContextSelectionDecision]
    budget: ContextBudget
    source_mix: dict[str, int]
    avoided_duplicate_reads: int = 0

    @property
    def selected_paths(self) -> list[str]:
        return [decision.path for decision in self.decisions]

    @property
    def read_paths(self) -> list[str]:
        return [decision.path for decision in self.decisions if decision.source == "file_read"]


class ContextOptimizer:
    def __init__(
        self,
        intent: str,
        workspace_root: str | Path,
        workspace_boosts: Mapping[str, WorkspaceBoost] | None = None,
    ):
        self.intent = intent
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_boosts = dict(workspace_boosts or {})
        self.keywords = extract_keywords(intent)
        self.query_lower = " ".join(intent.lower().split())

    def adaptive_budget(self) -> ContextBudget:
        if self._is_location_query():
            return ContextBudget(max_files=4, max_chars_per_file=800, max_total_chars=3200)
        if "architecture" in self.keywords or "repository" in self.keywords:
            return ContextBudget(max_files=4, max_chars_per_file=2500, max_total_chars=10000)
        if "subsystem" in self.keywords or "memory" in self.keywords:
            return ContextBudget(max_files=4, max_chars_per_file=2000, max_total_chars=8000)
        return ContextBudget(max_files=4, max_chars_per_file=1800, max_total_chars=7200)

    def select(
        self,
        ranked_files: Iterable[RankedFile],
        index_records: Mapping[str, WorkspaceFileRecord] | None = None,
        existing_context: Iterable[dict] | None = None,
    ) -> OptimizedContextPlan:
        budget = self.adaptive_budget()
        records = dict(index_records or {})
        existing_paths = {
            str(item.get("path", ""))
            for item in (existing_context or [])
            if isinstance(item, dict) and item.get("path")
        }
        decisions: list[ContextSelectionDecision] = []
        seen: set[str] = set()
        avoided_duplicates = 0

        for ranked_file in ranked_files:
            path = ranked_file.path
            if path in seen or is_excluded_path(path) or not self._safe_workspace_path(path):
                continue
            seen.add(path)

            boost = self.workspace_boosts.get(path)
            record = records.get(path)
            if self._is_location_query() and self.workspace_boosts and not (boost or record):
                continue
            if path in existing_paths:
                avoided_duplicates += 1
                decision = self._decision_from_metadata(
                    path=path,
                    source="cached_preview",
                    reason="already_read_in_workflow",
                    budget_chars=budget.max_chars_per_file,
                    boost=boost,
                    record=record,
                )
            elif self._should_use_index_summary(boost, record):
                decision = self._decision_from_metadata(
                    path=path,
                    source="index_summary",
                    reason="strong_index_match_for_location_query",
                    budget_chars=budget.max_chars_per_file,
                    boost=boost,
                    record=record,
                )
            else:
                decision = self._decision_from_metadata(
                    path=path,
                    source="file_read",
                    reason="source_preview_required_for_grounded_synthesis",
                    budget_chars=budget.max_chars_per_file,
                    boost=boost,
                    record=record,
                )

            decisions.append(decision)
            if len(decisions) >= budget.max_files:
                break

        return OptimizedContextPlan(
            decisions=decisions,
            budget=budget,
            source_mix=_source_mix(decisions),
            avoided_duplicate_reads=avoided_duplicates,
        )

    def build_index_context_record(self, decision: ContextSelectionDecision) -> dict[str, Any]:
        content = "\n".join(_compact_lines([
            f"Indexed context for {decision.path}",
            f"Reason: {decision.reason}",
            f"Roles: {', '.join(decision.role_tags)}" if decision.role_tags else "",
            f"Summary: {decision.summary}" if decision.summary else "",
            f"Symbols: {', '.join(decision.symbols[:12])}" if decision.symbols else "",
            f"Imports: {', '.join(decision.imports[:8])}" if decision.imports else "",
            f"NATS subjects: {', '.join(decision.nats_subjects[:8])}" if decision.nats_subjects else "",
            f"Capabilities: {', '.join(decision.capabilities[:8])}" if decision.capabilities else "",
            f"Matched terms: {', '.join(decision.matched_terms[:12])}" if decision.matched_terms else "",
        ]))
        content = content[:decision.budget_chars]
        return {
            "path": decision.path,
            "content": content,
            "content_preview": content,
            "size": len(content),
            "truncated": False,
            "context_type": "index_summary",
            "source": decision.source,
            "summary": decision.summary,
            "role_tags": decision.role_tags,
            "symbols": decision.symbols,
            "imports": decision.imports,
            "nats_subjects": decision.nats_subjects,
            "capabilities": decision.capabilities,
            "matched_terms": decision.matched_terms,
        }

    def optimize_read_context(self, decision: ContextSelectionDecision, read_data: dict[str, Any]) -> dict[str, Any]:
        content = str(read_data.get("content", ""))
        preview = extract_targeted_preview(
            content=content,
            matched_terms=[*decision.matched_terms, *decision.symbols, *decision.capabilities, *decision.nats_subjects],
            max_chars=decision.budget_chars,
        )
        return {
            "path": read_data.get("path") or decision.path,
            "content": preview,
            "content_preview": preview,
            "size": int(read_data.get("size", len(content))),
            "truncated": bool(read_data.get("truncated", False)) or len(preview) < len(content),
            "redacted": bool(read_data.get("redacted", False)),
            "context_type": "targeted_file_read",
            "source": decision.source,
            "summary": decision.summary,
            "role_tags": decision.role_tags,
            "symbols": decision.symbols,
            "imports": decision.imports,
            "nats_subjects": decision.nats_subjects,
            "capabilities": decision.capabilities,
            "matched_terms": decision.matched_terms,
        }

    def _decision_from_metadata(
        self,
        path: str,
        source: str,
        reason: str,
        budget_chars: int,
        boost: WorkspaceBoost | None,
        record: WorkspaceFileRecord | None,
    ) -> ContextSelectionDecision:
        role_tags = list(record.role_tags if record else (boost.role_tags if boost else []))
        symbols = [symbol.name for symbol in record.symbols] if record else list(boost.symbols if boost else [])
        imports = [
            f"{item.module}.{item.name}".strip(".") if item.name else item.module
            for item in record.imports
        ] if record else []
        nats_subjects = [item.value for item in record.nats_subjects] if record else []
        capabilities = [item.value for item in record.capabilities] if record else []
        summary = record.summary if record else (boost.summary if boost else None)
        terms = set(record.terms if record else [])
        matched_terms = sorted(
            keyword
            for keyword in self.keywords
            if keyword in terms
            or keyword in path.lower()
            or keyword in " ".join(role_tags).lower()
            or keyword in " ".join(symbols).lower()
            or keyword in " ".join(capabilities).lower()
            or keyword in " ".join(nats_subjects).lower()
        )
        return ContextSelectionDecision(
            path=path,
            source=source,
            reason=reason,
            budget_chars=budget_chars,
            matched_terms=matched_terms,
            role_tags=role_tags,
            summary=summary,
            symbols=symbols,
            imports=imports,
            nats_subjects=nats_subjects,
            capabilities=capabilities,
        )

    def _should_use_index_summary(self, boost: WorkspaceBoost | None, record: WorkspaceFileRecord | None) -> bool:
        if not self._is_location_query() or not (boost or record):
            return False
        if "memory" in self.keywords and "subsystem" in self.keywords:
            return False
        return True

    def _is_location_query(self) -> bool:
        return any(marker in self.query_lower for marker in LOCATION_QUERY_MARKERS)

    def _safe_workspace_path(self, raw_path: str) -> bool:
        pure_path = PurePosixPath(raw_path.replace("\\", "/"))
        if pure_path.is_absolute() or ".." in pure_path.parts:
            return False
        try:
            candidate = (self.workspace_root / pure_path.as_posix()).resolve()
            candidate.relative_to(self.workspace_root)
        except (OSError, ValueError):
            return False
        return candidate.is_file()


def extract_targeted_preview(content: str, matched_terms: list[str], max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    normalized_terms = [
        term.lower()
        for term in matched_terms
        if term and len(term) > 2
    ]
    if not normalized_terms:
        return content[:max_chars]

    lines = content.splitlines()
    selected_indexes: set[int] = set()
    for index, line in enumerate(lines):
        lowered = line.lower()
        if any(term in lowered for term in normalized_terms):
            for nearby in range(max(0, index - 2), min(len(lines), index + 3)):
                selected_indexes.add(nearby)

    if not selected_indexes:
        return content[:max_chars]

    chunks: list[str] = []
    previous = -2
    for index in sorted(selected_indexes):
        if index != previous + 1 and chunks:
            chunks.append("...")
        chunks.append(f"{index + 1}: {lines[index]}")
        previous = index
        if sum(len(chunk) + 1 for chunk in chunks) >= max_chars:
            break
    return "\n".join(chunks)[:max_chars]


def _source_mix(decisions: list[ContextSelectionDecision]) -> dict[str, int]:
    mix = {"file_read": 0, "index_summary": 0, "cached_preview": 0}
    for decision in decisions:
        mix[decision.source] = mix.get(decision.source, 0) + 1
    return mix


def _compact_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if line]
