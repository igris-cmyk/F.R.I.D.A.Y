import ast
import re
from pathlib import PurePosixPath

from core.workspace.models import ImportRecord, SymbolRecord, TextMatchRecord, WorkspaceFileRecord


LANGUAGE_BY_EXTENSION = {
    ".css": "css",
    ".conf": "config",
    ".html": "html",
    ".js": "javascript",
    ".json": "json",
    ".md": "markdown",
    ".py": "python",
    ".rs": "rust",
    ".sh": "shell",
    ".toml": "toml",
    ".ts": "typescript",
    ".yaml": "yaml",
    ".yml": "yaml",
}

KNOWN_CAPABILITY_IDS = {
    "filesystem.search",
    "filesystem.read",
    "memory.recall",
    "git.status",
    "system.monitor",
    "research.synthesize",
    "shell.execute",
    "test.medium_action",
}
CAPABILITY_RE = re.compile(r"\b(?:filesystem|memory|git|system|research|shell|test)\.[a-zA-Z0-9_]+\b")
CAPABILITY_ASSIGNMENT_RE = re.compile(r"capability_id\s*=\s*[\"']([^\"']+)[\"']")
NATS_SUBJECT_RE = re.compile(r"friday\.[A-Za-z0-9_.*${}_-]+(?:\.[A-Za-z0-9_.*${}_-]+)*")
NATS_SUBJECT_PREFIXES = (
    "friday.intent.",
    "friday.permission.",
    "friday.stream.",
    "friday.system.",
)
TOKEN_RE = re.compile(r"[a-z0-9_]+")

STOP_WORDS = {
    "and",
    "are",
    "async",
    "await",
    "class",
    "const",
    "def",
    "else",
    "for",
    "from",
    "import",
    "into",
    "return",
    "self",
    "that",
    "the",
    "this",
    "with",
}


def language_for_extension(extension: str) -> str:
    return LANGUAGE_BY_EXTENSION.get(extension, "text")


def extract_file_record(
    relative_path: str,
    content: str,
    size: int,
    mtime: float,
    sha256: str,
) -> WorkspaceFileRecord:
    extension = PurePosixPath(relative_path).suffix.lower()
    language = language_for_extension(extension)
    symbols: list[SymbolRecord] = []
    imports: list[ImportRecord] = []

    if extension == ".py":
        symbols, imports = _extract_python_structure(content)

    nats_subjects = _extract_nats_subjects(content)
    capabilities = _extract_capabilities(content)
    role_tags = infer_role_tags(relative_path, content, symbols, imports, nats_subjects, capabilities)
    terms = extract_terms(relative_path, content, role_tags, symbols, imports, nats_subjects, capabilities)
    summary = build_summary(relative_path, role_tags, symbols, imports, nats_subjects, capabilities)

    return WorkspaceFileRecord(
        path=relative_path,
        size=size,
        mtime=mtime,
        sha256=sha256,
        extension=extension,
        language=language,
        summary=summary,
        role_tags=role_tags,
        symbols=symbols,
        imports=imports,
        nats_subjects=nats_subjects,
        capabilities=capabilities,
        terms=terms,
    )


def _extract_python_structure(content: str) -> tuple[list[SymbolRecord], list[ImportRecord]]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return [], []

    symbols: list[SymbolRecord] = []
    imports: list[ImportRecord] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            symbols.append(SymbolRecord(name=node.name, kind="class", line=node.lineno))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
            symbols.append(SymbolRecord(name=node.name, kind=kind, line=node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(ImportRecord(module=alias.name, line=node.lineno))
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                imports.append(ImportRecord(module=module, name=alias.name, line=node.lineno))

    return (
        sorted(symbols, key=lambda item: (item.line, item.name)),
        sorted(imports, key=lambda item: (item.line, item.module, item.name or "")),
    )


def _extract_matches(content: str, pattern: re.Pattern[str]) -> list[TextMatchRecord]:
    matches: dict[str, int] = {}
    line_starts = _line_start_offsets(content)
    for match in pattern.finditer(content):
        value = match.group(0).rstrip("`'\"),;")
        if value not in matches:
            matches[value] = _line_for_offset(line_starts, match.start())
    return [
        TextMatchRecord(value=value, line=line)
        for value, line in sorted(matches.items(), key=lambda item: (item[1], item[0]))
    ]


def _extract_nats_subjects(content: str) -> list[TextMatchRecord]:
    return [
        item
        for item in _extract_matches(content, NATS_SUBJECT_RE)
        if item.value.startswith(NATS_SUBJECT_PREFIXES)
    ]


def _extract_capabilities(content: str) -> list[TextMatchRecord]:
    line_starts = _line_start_offsets(content)
    matches: dict[str, int] = {}
    for match in CAPABILITY_ASSIGNMENT_RE.finditer(content):
        value = match.group(1)
        matches[value] = _line_for_offset(line_starts, match.start())
    for item in _extract_matches(content, CAPABILITY_RE):
        if item.value in KNOWN_CAPABILITY_IDS and item.value not in matches:
            matches[item.value] = item.line
    return [
        TextMatchRecord(value=value, line=line)
        for value, line in sorted(matches.items(), key=lambda item: (item[1], item[0]))
    ]


def _line_start_offsets(content: str) -> list[int]:
    offsets = [0]
    for match in re.finditer("\n", content):
        offsets.append(match.end())
    return offsets


def _line_for_offset(line_starts: list[int], offset: int) -> int:
    low = 0
    high = len(line_starts) - 1
    while low <= high:
        middle = (low + high) // 2
        if line_starts[middle] <= offset:
            low = middle + 1
        else:
            high = middle - 1
    return high + 1


def infer_role_tags(
    path: str,
    content: str,
    symbols: list[SymbolRecord],
    imports: list[ImportRecord],
    nats_subjects: list[TextMatchRecord],
    capabilities: list[TextMatchRecord],
) -> list[str]:
    tags: set[str] = set()
    lowered_path = path.lower()
    lowered_content = content.lower()
    symbol_names = {symbol.name.lower() for symbol in symbols}

    if path in {"core/main.py", "apps/desktop/src/main.js", "apps/desktop/src-tauri/src/lib.rs"}:
        tags.add("entry_point")
    if lowered_path.startswith("core/agents/planner") or "planner" in symbol_names:
        tags.add("planner")
    if lowered_path.startswith("core/agents/router") or "router" in lowered_path:
        tags.add("router")
    if lowered_path.startswith("core/memory/") or "memory" in lowered_path:
        tags.add("memory")
    if lowered_path.startswith("core/security/") or "securitypolicy" in lowered_content:
        tags.add("security")
    if lowered_path.startswith("core/capabilities/") or capabilities:
        tags.add("capability")
    if lowered_path.startswith("core/research/") or "research" in lowered_path:
        tags.add("research")
    if nats_subjects or "nats" in lowered_content or "nats" in lowered_path:
        tags.add("nats")
    if "friday.stream" in lowered_content or ("nats" in lowered_path and "websocket" in lowered_content):
        tags.add("nats_streaming")
    if lowered_path.startswith("tests/") or PurePosixPath(lowered_path).name.startswith("test_"):
        tags.add("test")
    if lowered_path.startswith("core/tools/") or lowered_path.startswith("scripts/"):
        tags.add("tool")
    if lowered_path.startswith("scripts/"):
        tags.add("script")
    if lowered_path.startswith("docs/") or lowered_path.endswith(".md"):
        tags.add("docs")
    if lowered_path.startswith("apps/desktop/"):
        tags.add("frontend")
    if lowered_path.endswith(("pyproject.toml", "package.json", "tauri.conf.json", ".conf")):
        tags.add("config")
    if lowered_path.startswith("core/evals/") or "eval_harness" in lowered_path:
        tags.add("eval")
    if any(record.module.startswith("core.schemas") for record in imports):
        tags.add("schema")

    return sorted(tags)


def extract_terms(
    path: str,
    content: str,
    role_tags: list[str],
    symbols: list[SymbolRecord],
    imports: list[ImportRecord],
    nats_subjects: list[TextMatchRecord],
    capabilities: list[TextMatchRecord],
) -> list[str]:
    values = [path, *role_tags]
    values.extend(symbol.name for symbol in symbols)
    values.extend(record.module for record in imports)
    values.extend(record.name for record in imports if record.name)
    values.extend(record.value for record in nats_subjects)
    values.extend(record.value for record in capabilities)

    trimmed_content = content[:20000]
    values.append(trimmed_content)

    terms = {
        token
        for value in values
        for token in TOKEN_RE.findall(str(value).lower())
        if len(token) > 2 and token not in STOP_WORDS
    }
    return sorted(terms)


def build_summary(
    path: str,
    role_tags: list[str],
    symbols: list[SymbolRecord],
    imports: list[ImportRecord],
    nats_subjects: list[TextMatchRecord],
    capabilities: list[TextMatchRecord],
) -> str:
    parts = [f"{path}"]
    if role_tags:
        parts.append(f"roles: {', '.join(role_tags)}")
    if symbols:
        parts.append("symbols: " + ", ".join(symbol.name for symbol in symbols[:5]))
    if imports:
        parts.append("imports: " + ", ".join(_format_import(item) for item in imports[:5]))
    if nats_subjects:
        parts.append("NATS: " + ", ".join(item.value for item in nats_subjects[:5]))
    if capabilities:
        parts.append("capabilities: " + ", ".join(item.value for item in capabilities[:5]))
    return "; ".join(parts)


def _format_import(record: ImportRecord) -> str:
    if record.name:
        return f"{record.module}.{record.name}".strip(".")
    return record.module
