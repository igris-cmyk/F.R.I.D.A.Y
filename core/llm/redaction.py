import re
from dataclasses import dataclass, field

from core.memory.redaction import redact_text


SECRET_KEY_RE = re.compile(
    r"\b(?:database_url|auth_secret|nextauth_secret|nextauth_url|github_token|openai_api_key|"
    r"deepseek_api_key|api[_-]?key|token|password|secret|client_secret)\b",
    re.IGNORECASE,
)
ENV_ASSIGNMENT_RE = re.compile(r"^\s*[A-Z0-9_]{3,}\s*=\s*.+$", re.IGNORECASE | re.MULTILINE)
PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE)
BEARER_RE = re.compile(r"authorization:\s*bearer\s+", re.IGNORECASE)
DATABASE_URL_RE = re.compile(r"\b[a-z]+://[^@\s]+:[^@\s]+@", re.IGNORECASE)


@dataclass
class CloudRedactionResult:
    text: str
    cloud_safe: bool
    redacted: bool
    reasons: list[str] = field(default_factory=list)


def redact_for_cloud(text: str) -> CloudRedactionResult:
    raw = str(text or "")
    reasons: list[str] = []

    if SECRET_KEY_RE.search(raw):
        reasons.append("secret_key")
    if PRIVATE_KEY_RE.search(raw):
        reasons.append("private_key")
    if BEARER_RE.search(raw):
        reasons.append("bearer_token")
    if DATABASE_URL_RE.search(raw):
        reasons.append("credential_url")

    env_assignments = ENV_ASSIGNMENT_RE.findall(raw)
    if len(env_assignments) >= 2:
        reasons.append("env_like_content")

    redacted = redact_text(raw, max_chars=max(len(raw), 4000))
    return CloudRedactionResult(
        text=redacted,
        cloud_safe=not reasons,
        redacted=redacted != raw,
        reasons=sorted(set(reasons)),
    )


def redact_messages_for_cloud(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], CloudRedactionResult]:
    redacted_messages: list[dict[str, str]] = []
    aggregate_reasons: list[str] = []
    any_redacted = False

    for message in messages:
        content = str(message.get("content", ""))
        redaction = redact_for_cloud(content)
        aggregate_reasons.extend(redaction.reasons)
        any_redacted = any_redacted or redaction.redacted
        redacted_messages.append({
            "role": str(message.get("role", "user")),
            "content": redaction.text,
        })

    return redacted_messages, CloudRedactionResult(
        text="\n".join(item["content"] for item in redacted_messages),
        cloud_safe=not aggregate_reasons,
        redacted=any_redacted,
        reasons=sorted(set(aggregate_reasons)),
    )
