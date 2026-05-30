from dataclasses import dataclass
from pathlib import PurePosixPath
import re


EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "target",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

SOURCE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".rs",
    ".json",
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
    parts = path.replace("\\", "/").split("/")
    return any(part in EXCLUDED_DIRS for part in parts)


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


def rank_research_files(intent: str, candidate_files: list[str]) -> list[RankedFile]:
    keywords = extract_keywords(intent)
    active_profiles = _active_profiles(intent, keywords)
    query_mentions_tests = any(word in keywords for word in {"test", "tests", "testing"})
    query_mentions_docs = any(word in keywords for word in {"doc", "docs", "readme", "overview"})
    query_mentions_verification = any(word in keywords for word in {"verify", "verification", "smoke"})

    ranked: list[RankedFile] = []
    for raw_path in dict.fromkeys(candidate_files):
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

        if score > 0:
            ranked.append(RankedFile(path=path, score=round(score, 3), reasons=reasons))

    return sorted(ranked, key=lambda item: (-item.score, item.path))
