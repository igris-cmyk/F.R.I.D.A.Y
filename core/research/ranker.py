from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from typing import Mapping

from core.research.workspace_boost import WorkspaceBoost, WorkspaceIndexBoost


EXCLUDED_DIRS = {
    ".git",
    ".friday",
    ".venv",
    "__pycache__",
    "node_modules",
    "target",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".next",
    ".turbo",
    "coverage",
}

SECRET_FILENAMES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.test",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}

SECRET_EXTENSIONS = {".key", ".pem", ".p12", ".pfx"}

SOURCE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".rs",
    ".json",
    ".conf",
    ".toml",
    ".yml",
    ".yaml",
    ".md",
}

STOP_WORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "how",
    "in",
    "of",
    "on",
    "show",
    "the",
    "to",
}

DOMAIN_PROFILES = (
    {
        "name": "memory",
        "triggers": {"memory", "recall", "episodic", "embedding", "retriever"},
        "priorities": (
            "core/memory/manager.py",
            "core/memory/pipeline.py",
            "core/memory/retriever.py",
            "core/agents/memory_agent.py",
            "core/config.py",
        ),
        "prefixes": ("core/memory/",),
    },
    {
        "name": "approval",
        "triggers": {"approval", "permission", "security", "deny", "blocked", "risk", "dangerous"},
        "priorities": (
            "core/security/approval.py",
            "core/security/permissions.py",
            "core/main.py",
            "core/schemas/events.py",
            "apps/desktop/src/main.js",
        ),
        "prefixes": ("core/security/",),
    },
    {
        "name": "planner",
        "triggers": {"planner", "planning", "route", "router", "intent", "capability plan"},
        "priorities": (
            "core/agents/planner.py",
            "core/agents/router.py",
            "core/main.py",
            "core/capabilities/registry.py",
            "core/capabilities/executor.py",
        ),
        "prefixes": ("core/agents/",),
    },
    {
        "name": "capability",
        "triggers": {"capability", "executor", "tool", "registry", "execute"},
        "priorities": (
            "core/capabilities/executor.py",
            "core/capabilities/registry.py",
            "core/capabilities/contracts.py",
            "core/security/permissions.py",
            "core/main.py",
        ),
        "prefixes": ("core/capabilities/",),
    },
    {
        "name": "frontend",
        "triggers": {"frontend", "ui", "stream", "streaming", "tauri", "desktop", "palette", "nats websocket"},
        "priorities": (
            "apps/desktop/src/main.js",
            "apps/desktop/src/index.html",
            "apps/desktop/src/styles.css",
            "apps/desktop/src-tauri/src/lib.rs",
            "core/main.py",
            "core/schemas/events.py",
        ),
        "prefixes": ("apps/desktop/src/", "apps/desktop/src-tauri/src/"),
    },
    {
        "name": "architecture",
        "triggers": {"architecture", "repository", "project structure", "system design", "overview"},
        "priorities": (
            "core/main.py",
            "core/agents/planner.py",
            "core/capabilities/executor.py",
            "core/security/permissions.py",
            "core/memory/manager.py",
            "apps/desktop/src/main.js",
            "apps/desktop/src-tauri/src/lib.rs",
            "infra/docker-compose.yml",
        ),
        "prefixes": ("core/", "apps/desktop/"),
    },
)


@dataclass(frozen=True)
class RankedFile:
    path: str
    score: float
    reasons: list[str]


def is_excluded_path(path: str) -> bool:
    normalized = path.replace("\\", "/").strip()
    pure_path = PurePosixPath(normalized)
    parts = pure_path.parts
    if pure_path.is_absolute() or ".." in parts:
        return True
    if any(part in EXCLUDED_DIRS for part in parts):
        return True
    if pure_path.name in SECRET_FILENAMES or pure_path.suffix.lower() in SECRET_EXTENSIONS:
        return True
    return False


def extract_keywords(intent: str) -> set[str]:
    normalized = intent.lower()
    words = {
        word
        for word in re.findall(r"[a-z0-9_]+", normalized)
        if len(word) > 2 and word not in STOP_WORDS
    }
    phrases = {
        phrase
        for profile in DOMAIN_PROFILES
        for phrase in profile["triggers"]
        if " " in phrase and phrase in normalized
    }
    return words | phrases


def _active_profiles(intent: str, keywords: set[str]) -> list[dict]:
    normalized = intent.lower()
    active = []
    for profile in DOMAIN_PROFILES:
        if any(trigger in keywords or trigger in normalized for trigger in profile["triggers"]):
            active.append(profile)
    return active


def _path_terms(path: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9_]+", path.lower())
        if len(term) > 1
    }


def rank_research_files(
    intent: str,
    candidate_files: list[str],
    workspace_root: str | None = None,
    workspace_boosts: Mapping[str, WorkspaceBoost] | None = None,
) -> list[RankedFile]:
    keywords = extract_keywords(intent)
    active_profiles = _active_profiles(intent, keywords)
    query_mentions_tests = any(word in keywords for word in {"test", "tests", "testing"})
    query_mentions_docs = any(word in keywords for word in {"doc", "docs", "readme", "overview"})
    query_mentions_verification = any(word in keywords for word in {"verify", "verification", "smoke"})
    boosts = dict(workspace_boosts or {})
    if workspace_root and workspace_boosts is None:
        boosts = WorkspaceIndexBoost(workspace_root).boost_map(intent)

    ranked: list[RankedFile] = []
    candidates = list(dict.fromkeys([*candidate_files, *boosts.keys()]))
    for raw_path in candidates:
        path = raw_path.replace("\\", "/").strip()
        if not path or is_excluded_path(path):
            continue

        suffix = PurePosixPath(path).suffix.lower()
        score = 0.0
        reasons: list[str] = []

        if suffix in SOURCE_EXTENSIONS:
            score += 5.0
            reasons.append("source_file")

        terms = _path_terms(path)
        for keyword in sorted(keywords):
            keyword_parts = set(keyword.split())
            if keyword in terms:
                score += 12.0
                reasons.append(f"keyword:{keyword}")
            elif keyword_parts and keyword_parts.issubset(terms):
                score += 10.0
                reasons.append(f"phrase:{keyword}")
            elif keyword in path.lower():
                score += 5.0
                reasons.append(f"path_contains:{keyword}")

        for profile in active_profiles:
            priorities = profile["priorities"]
            if path in priorities:
                score += 120.0 - (priorities.index(path) * 10.0)
                reasons.append(f"domain:{profile['name']}:exact")
            elif any(path.startswith(prefix) for prefix in profile["prefixes"]):
                score += 35.0
                reasons.append(f"domain:{profile['name']}:prefix")

        lowered = path.lower()
        if not query_mentions_tests and ("/test" in lowered or lowered.startswith("tests/") or "_test." in lowered):
            score -= 30.0
            reasons.append("penalty:test")
        if not query_mentions_verification and ("verify" in lowered or "smoke" in lowered):
            score -= 15.0
            reasons.append("penalty:verification")
        if not query_mentions_docs and PurePosixPath(path).name.lower().startswith("readme"):
            score -= 10.0
            reasons.append("penalty:readme")

        boost = boosts.get(path)
        if boost:
            score += boost.score
            reasons.extend(boost.reasons)
            reasons.append(f"workspace_index_boost:{boost.score}")

        if score > 0:
            ranked.append(RankedFile(path=path, score=round(score, 3), reasons=list(dict.fromkeys(reasons))))

    return sorted(ranked, key=lambda item: (-item.score, item.path))
