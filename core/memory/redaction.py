import re


SECRET_PATTERNS = (
    re.compile(r"(authorization:\s*bearer\s+)[^\s]+", re.IGNORECASE),
    re.compile(r"((?:api[_-]?key|token|password|secret)\s*=\s*)[^\s]+", re.IGNORECASE),
    re.compile(r"((?:api[_-]?key|token|password|secret)\s*:\s*)[^\s]+", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
    re.compile(r"\b[A-Za-z0-9_\-]{40,}\b"),
)


def redact_text(text: str, max_chars: int = 4000) -> str:
    redacted = str(text or "")
    for pattern in SECRET_PATTERNS:
        if pattern.pattern.startswith("(authorization"):
            redacted = pattern.sub(r"\1[REDACTED]", redacted)
        elif pattern.pattern.startswith("((?:api"):
            redacted = pattern.sub(r"\1[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)

    if len(redacted) > max_chars:
        return redacted[: max_chars // 2] + "\n...[TRUNCATED]...\n" + redacted[-max_chars // 2 :]
    return redacted
