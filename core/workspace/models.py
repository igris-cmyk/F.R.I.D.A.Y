from dataclasses import dataclass, field
from pathlib import PurePosixPath


CURRENT_SCHEMA_VERSION = 1
DEFAULT_WORKSPACE_INDEX_DB_PATH = ".friday/workspace/workspace_index.sqlite3"

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

INDEXABLE_EXTENSIONS = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".py",
    ".rs",
    ".sh",
    ".toml",
    ".ts",
    ".yaml",
    ".yml",
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


@dataclass(frozen=True)
class SymbolRecord:
    name: str
    kind: str
    line: int


@dataclass(frozen=True)
class ImportRecord:
    module: str
    name: str | None = None
    line: int = 0


@dataclass(frozen=True)
class TextMatchRecord:
    value: str
    line: int = 0


@dataclass
class WorkspaceFileRecord:
    path: str
    size: int
    mtime: float
    sha256: str
    extension: str
    language: str
    summary: str
    role_tags: list[str] = field(default_factory=list)
    symbols: list[SymbolRecord] = field(default_factory=list)
    imports: list[ImportRecord] = field(default_factory=list)
    nats_subjects: list[TextMatchRecord] = field(default_factory=list)
    capabilities: list[TextMatchRecord] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)

    @property
    def is_test(self) -> bool:
        path = self.path.lower()
        name = PurePosixPath(path).name
        return path.startswith("tests/") or name.startswith("test_") or name.endswith("_test.py")


@dataclass(frozen=True)
class WorkspaceSearchResult:
    path: str
    score: float
    role_tags: list[str]
    summary: str
    reasons: list[str]
