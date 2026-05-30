from dataclasses import dataclass
from typing import Iterable

from core.config import (
    FRIDAY_RESEARCH_MAX_CHARS_PER_FILE,
    FRIDAY_RESEARCH_MAX_FILES,
    FRIDAY_RESEARCH_MAX_TOTAL_CHARS,
)
from core.research.ranker import RankedFile, is_excluded_path


@dataclass(frozen=True)
class ContextBudget:
    max_files: int = FRIDAY_RESEARCH_MAX_FILES
    max_chars_per_file: int = FRIDAY_RESEARCH_MAX_CHARS_PER_FILE
    max_total_chars: int = FRIDAY_RESEARCH_MAX_TOTAL_CHARS


@dataclass(frozen=True)
class ContextFile:
    path: str
    content: str
    size: int
    truncated: bool


def select_ranked_files(
    ranked_files: Iterable[RankedFile],
    budget: ContextBudget | None = None,
) -> list[str]:
    active_budget = budget or ContextBudget()
    selected: list[str] = []
    seen: set[str] = set()

    for ranked_file in ranked_files:
        path = ranked_file.path
        if path in seen or is_excluded_path(path):
            continue
        selected.append(path)
        seen.add(path)
        if len(selected) >= active_budget.max_files:
            break

    return selected


def build_context_file(
    path: str,
    content: str,
    size: int,
    truncated: bool,
    used_chars: int,
    budget: ContextBudget | None = None,
) -> ContextFile | None:
    active_budget = budget or ContextBudget()
    if is_excluded_path(path):
        return None

    remaining = active_budget.max_total_chars - used_chars
    if remaining <= 0:
        return None

    char_limit = min(active_budget.max_chars_per_file, remaining)
    bounded_content = content[:char_limit]
    return ContextFile(
        path=path,
        content=bounded_content,
        size=size,
        truncated=truncated or len(content) > len(bounded_content),
    )


def context_budget_used(context_files: Iterable[dict | ContextFile]) -> int:
    total = 0
    for item in context_files:
        if isinstance(item, ContextFile):
            total += len(item.content)
        else:
            total += len(str(item.get("content", "")))
    return total


def context_has_truncation(context_files: Iterable[dict | ContextFile]) -> bool:
    for item in context_files:
        if isinstance(item, ContextFile):
            if item.truncated:
                return True
        elif item.get("truncated"):
            return True
    return False
