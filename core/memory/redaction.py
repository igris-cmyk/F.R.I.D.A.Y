import re


SECRET_PATTERNS = (
    re.compile(r"(authorization:\s*bearer\s+)[^\s]+", re.IGNORECASE),
    re.compile(
        r"((?:[a-z0-9_./-]*api[_-]?key|[a-z0-9_./-]*token|[a-z0-9_./-]*password|"
        r"[a-z0-9_./-]*secret|database_url)\s*=\s*)[^\s]+",
        re.IGNORECASE,
    ),
    re.compile(
        r"((?:[a-z0-9_./-]*api[_-]?key|[a-z0-9_./-]*token|[a-z0-9_./-]*password|"
        r"[a-z0-9_./-]*secret|database_url)\s*:\s*)[^\s]+",
        re.IGNORECASE,
    ),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
    re.compile(
        r"\b(?=[A-Za-z0-9_\-]{40,}\b)(?=[A-Za-z0-9_\-]*[A-Z])(?=[A-Za-z0-9_\-]*[a-z])"
        r"(?=[A-Za-z0-9_\-]*\d)[A-Za-z0-9_\-]+\b"
    ),
)


def redact_text(text: str, max_chars: int = 4000) -> str:
    redacted = str(text or "")
    for pattern in SECRET_PATTERNS:
        if pattern.pattern.startswith("(authorization"):
            redacted = pattern.sub(r"\1[REDACTED]", redacted)
        elif pattern.pattern.startswith("((?:"):
            redacted = pattern.sub(r"\1[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)

    if len(redacted) > max_chars:
        return redacted[: max_chars // 2] + "\n...[TRUNCATED]...\n" + redacted[-max_chars // 2 :]
    return redacted
