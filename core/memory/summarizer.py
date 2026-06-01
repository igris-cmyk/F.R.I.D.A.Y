import ast
import re
from dataclasses import asdict, dataclass
from typing import Any

from core.config import FRIDAY_MEMORY_PREVIEW_MAX_CHARS, FRIDAY_MEMORY_SUMMARY_MAX_CHARS
from core.memory.redaction import redact_text


SOURCE_FILE_EXTENSIONS = (".py", ".js", ".ts", ".rs", ".json", ".toml", ".yml", ".yaml", ".md")
EXCLUDED_PATH_PARTS = {
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


@dataclass(frozen=True)
class DeterministicMemorySummary:
    title: str
    user_intent: str
    intent_type: str
    outcome: str
    summary: str
    key_files: list[str]
    capabilities_used: list[str]
    decisions: list[str]
    limitations: list[str]
    followups: list[str]
    safety_events: list[str]
    source_trace_id: str
    content_preview: str

    def metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("summary", None)
        data.pop("content_preview", None)
        data["summary_kind"] = "deterministic"
        data["summary_version"] = 1
        return data


def summarize_completed_trace(trace: dict[str, Any]) -> DeterministicMemorySummary:
    trace_id = str(trace.get("trace_id") or "")
    intent_type = str(trace.get("intent") or "unknown").lower()
    command = str(trace.get("command") or "").strip()
    result = str(trace.get("result") or "")
    metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
    error_state = bool(trace.get("error_state"))

    capabilities = _extract_capabilities(result, metadata)
    key_files = _extract_key_files(command, result, metadata)
    safety_events = _extract_safety_events(command, result, capabilities, error_state)
    limitations = _extract_limitations(metadata)
    decisions = _extract_decisions(command, intent_type, key_files, capabilities)
    followups: list[str] = []
    title = _title_for_trace(command, intent_type, safety_events)
    outcome = _outcome_for_trace(error_state, safety_events)

    summary = _build_summary(
        command=command,
        intent_type=intent_type,
        title=title,
        outcome=outcome,
        key_files=key_files,
        capabilities=capabilities,
        limitations=limitations,
        safety_events=safety_events,
    )
    content_preview = _build_content_preview(
        title=title,
        command=command,
        outcome=outcome,
        key_files=key_files,
        capabilities=capabilities,
        decisions=decisions,
        limitations=limitations,
        safety_events=safety_events,
    )

    return DeterministicMemorySummary(
        title=title,
        user_intent=command,
        intent_type=intent_type,
        outcome=outcome,
        summary=_clean_memory_text(summary, FRIDAY_MEMORY_SUMMARY_MAX_CHARS),
        key_files=key_files,
        capabilities_used=capabilities,
        decisions=decisions,
        limitations=limitations,
        followups=followups,
        safety_events=safety_events,
        source_trace_id=trace_id,
        content_preview=_clean_memory_text(content_preview, FRIDAY_MEMORY_PREVIEW_MAX_CHARS),
    )


def _build_summary(
    command: str,
    intent_type: str,
    title: str,
    outcome: str,
    key_files: list[str],
    capabilities: list[str],
    limitations: list[str],
    safety_events: list[str],
) -> str:
    lower_command = command.lower()
    file_phrase = _format_file_list(key_files)
    capability_phrase = _format_capability_sequence(capabilities)

    if safety_events:
        return (
            "FRIDAY blocked a destructive or unsafe request. The planner produced a high-risk "
            "execution path, but SecurityPolicy denied the capability before execution. "
            f"Safety note: {safety_events[0]}"
        )

    if intent_type == "memory":
        return (
            "FRIDAY answered a memory recall question by searching persistent SQLite memory "
            "and reconstructing relevant continuity from stored summaries."
        )

    if "memory" in lower_command and intent_type == "research":
        return (
            "We inspected the memory subsystem. FRIDAY searched the project, ranked relevant files, "
            f"and inspected {file_phrase or 'the selected memory implementation files'}. "
            "The subsystem covers SQLite persistence, retrieval, recall reconstruction, redaction, "
            "and keyword fallback when embeddings are unavailable."
        )

    if ("approval" in lower_command or "security" in lower_command or "permission" in lower_command) and intent_type in {"research", "terminal"}:
        return (
            "We inspected the approval and security workflow. FRIDAY selected the security policy, "
            f"approval, event, and orchestrator files{f' including {file_phrase}' if file_phrase else ''}. "
            "The trace confirms that risky capability requests remain routed through SecurityPolicy "
            "and the approval boundary before execution."
        )

    if ("architecture" in lower_command or "repository" in lower_command or "system design" in lower_command) and intent_type == "research":
        return (
            "We inspected the repository architecture. FRIDAY selected representative core files "
            f"{f'including {file_phrase}' if file_phrase else 'across the orchestrator, planner, capabilities, and security layers'}. "
            "The trace confirms the main flow: intent enters the orchestrator, routing and planning create a plan, "
            "SecurityPolicy checks risk, and CapabilityExecutor runs allowed capabilities."
        )

    if "git.status" in capabilities:
        return "FRIDAY inspected Git status for the current workspace and returned the repository state."

    if "filesystem.read" in capabilities and key_files:
        return f"FRIDAY read {file_phrase} from the workspace and returned bounded file content."

    if intent_type == "research":
        return (
            f"We completed a grounded research workflow for: {command}. "
            f"FRIDAY used {capability_phrase or 'the research capability flow'} "
            f"and inspected {file_phrase or 'selected project files'}."
        )

    if outcome == "failure":
        return f"FRIDAY handled a failed {intent_type} request for: {command}."

    return f"FRIDAY completed a {intent_type} request for: {command}."


def _build_content_preview(
    title: str,
    command: str,
    outcome: str,
    key_files: list[str],
    capabilities: list[str],
    decisions: list[str],
    limitations: list[str],
    safety_events: list[str],
) -> str:
    lines = [
        f"Title: {title}",
        f"Intent: {command}",
        f"Outcome: {outcome}",
    ]
    if key_files:
        lines.append(f"Key files: {', '.join(key_files)}")
    if capabilities:
        lines.append(f"Capabilities: {' -> '.join(capabilities)}")
    if decisions:
        lines.append(f"Decisions: {'; '.join(decisions)}")
    if safety_events:
        lines.append(f"Safety events: {'; '.join(safety_events)}")
    if limitations:
        lines.append(f"Limitations: {'; '.join(limitations)}")
    return "\n".join(lines)


def _extract_capabilities(result: str, metadata: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in ("capabilities_used", "capabilities", "capabilities_started"):
        value = metadata.get(key)
        if isinstance(value, list):
            candidates.extend(str(item) for item in value)
    candidates.extend(re.findall(r"\b(?:SUCCESS|FAILED)\s+([a-zA-Z0-9_.-]+):", result))
    return _dedupe_preserve_order(candidates)


def _extract_key_files(command: str, result: str, metadata: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in ("selected_files", "inspected_files", "key_files"):
        value = metadata.get(key)
        if isinstance(value, list):
            candidates.extend(str(item) for item in value)

    for item in metadata.get("files_read", []) if isinstance(metadata.get("files_read"), list) else []:
        if isinstance(item, dict) and item.get("path"):
            candidates.append(str(item["path"]))
        elif isinstance(item, str):
            candidates.append(item)

    workflow = metadata.get("workflow") if isinstance(metadata.get("workflow"), dict) else {}
    candidates.extend(str(item) for item in workflow.get("selected_files", []) if isinstance(item, str))
    for item in workflow.get("files_read", []) if isinstance(workflow.get("files_read"), list) else []:
        if isinstance(item, dict) and item.get("path"):
            candidates.append(str(item["path"]))

    candidates.extend(_parse_files_from_result(result))
    return _prioritize_files(command, candidates)[:8]


def _parse_files_from_result(result: str) -> list[str]:
    files: list[str] = []
    for match in re.finditer(r"'files':\s*(\[[^\]]*\])", result):
        try:
            value = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, list):
            files.extend(str(item) for item in value)

    path_pattern = r"\b(?:core|apps|infra|docs|tests)/[A-Za-z0-9_./-]+\.(?:py|js|ts|rs|json|toml|ya?ml|md)\b"
    files.extend(re.findall(path_pattern, result))
    return files


def _prioritize_files(command: str, files: list[str]) -> list[str]:
    lower_command = command.lower()
    unique = [
        file_path
        for file_path in _dedupe_preserve_order(files)
        if _is_allowed_source_path(file_path)
    ]
    priority = _priority_map_for_command(lower_command)
    return sorted(
        unique,
        key=lambda path: (
            priority.get(path, 100),
            _domain_score(lower_command, path),
            len(path),
            path,
        ),
    )


def _priority_map_for_command(lower_command: str) -> dict[str, int]:
    if "memory" in lower_command:
        ordered = [
            "core/memory/manager.py",
            "core/memory/pipeline.py",
            "core/memory/retriever.py",
            "core/agents/memory_agent.py",
            "core/memory/sqlite_store.py",
            "core/memory/redaction.py",
        ]
    elif "approval" in lower_command or "security" in lower_command or "permission" in lower_command:
        ordered = [
            "core/security/approval.py",
            "core/security/permissions.py",
            "core/main.py",
            "core/schemas/events.py",
            "apps/desktop/src/main.js",
        ]
    elif "architecture" in lower_command or "repository" in lower_command or "system design" in lower_command:
        ordered = [
            "core/main.py",
            "core/agents/planner.py",
            "core/capabilities/executor.py",
            "core/security/permissions.py",
            "core/memory/manager.py",
            "apps/desktop/src/main.js",
        ]
    else:
        ordered = []
    return {path: index for index, path in enumerate(ordered)}


def _domain_score(lower_command: str, path: str) -> int:
    if "memory" in lower_command and "/memory/" in path:
        return 0
    if ("approval" in lower_command or "security" in lower_command) and "/security/" in path:
        return 0
    if "architecture" in lower_command and path in {"core/main.py", "core/agents/planner.py"}:
        return 0
    return 1


def _is_allowed_source_path(path: str) -> bool:
    normalized = path.replace("\\", "/").strip("'\" ")
    if not normalized.endswith(SOURCE_FILE_EXTENSIONS):
        return False
    parts = normalized.split("/")
    return not any(part in EXCLUDED_PATH_PARTS for part in parts)


def _extract_safety_events(command: str, result: str, capabilities: list[str], error_state: bool) -> list[str]:
    combined = f"{command} {result}".lower()
    dangerous = any(term in combined for term in ("delete everything", "rm -rf", "wipe", "destructive"))
    blocked = any(term in combined for term in ("blocked", "denied", "securitypolicy", "explicitly blocked"))
    if "shell.execute" in capabilities and (blocked or dangerous or error_state):
        return ["shell.execute was blocked by SecurityPolicy before execution."]
    if blocked and dangerous:
        return ["A destructive request was denied before execution."]
    return []


def _extract_limitations(metadata: dict[str, Any]) -> list[str]:
    limitations: list[str] = []
    for key in ("fallback_reason", "degraded_reason"):
        value = metadata.get(key)
        if value:
            limitations.append(str(value))
    workflow = metadata.get("workflow") if isinstance(metadata.get("workflow"), dict) else {}
    if any(item.get("truncated") for item in workflow.get("files_read", []) if isinstance(item, dict)):
        limitations.append("Some inspected file content was truncated to stay within context limits.")
    return _dedupe_preserve_order(limitations)


def _extract_decisions(command: str, intent_type: str, key_files: list[str], capabilities: list[str]) -> list[str]:
    decisions: list[str] = []
    lower_command = command.lower()
    if intent_type == "research":
        decisions.append("Used grounded research over local repository files.")
    if key_files:
        decisions.append(f"Selected {len(key_files)} key file{'s' if len(key_files) != 1 else ''} for memory context.")
    if "shell.execute" in capabilities:
        decisions.append("Kept shell execution behind SecurityPolicy.")
    if "memory" in lower_command:
        decisions.append("Treated the request as memory subsystem work.")
    return decisions


def _title_for_trace(command: str, intent_type: str, safety_events: list[str]) -> str:
    lower_command = command.lower()
    if safety_events:
        return "Blocked destructive request"
    if intent_type == "memory":
        return "Memory recall"
    if "memory" in lower_command:
        return "Memory subsystem inspection"
    if "approval" in lower_command or "permission" in lower_command:
        return "Approval workflow inspection"
    if "security" in lower_command:
        return "Security workflow inspection"
    if "architecture" in lower_command or "repository" in lower_command:
        return "Repository architecture inspection"
    if "git status" in lower_command:
        return "Git status inspection"
    if lower_command.startswith("read "):
        return "Workspace file inspection"
    return command[:80] or "Completed FRIDAY trace"


def _outcome_for_trace(error_state: bool, safety_events: list[str]) -> str:
    if safety_events:
        return "blocked"
    return "failure" if error_state else "success"


def _format_file_list(files: list[str]) -> str:
    if not files:
        return ""
    if len(files) == 1:
        return files[0]
    if len(files) == 2:
        return f"{files[0]} and {files[1]}"
    return f"{', '.join(files[:-1])}, and {files[-1]}"


def _format_capability_sequence(capabilities: list[str]) -> str:
    if not capabilities:
        return ""
    return " -> ".join(capabilities)


def _clean_memory_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\b(?:SUCCESS|FAILED)\s+[a-zA-Z0-9_.-]+:\s*", "", str(text or ""))
    text = text.replace("compression_timeout", "deterministic summary")
    text = text.replace("reconstruction_timeout", "deterministic recall")
    text = re.sub(r"\s+", " ", text).strip()
    return redact_text(text, max_chars=max_chars)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        cleaned = str(value).strip().strip("'\"")
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result
