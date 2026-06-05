from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMRequest:
    messages: list[dict[str, str]]
    system_prompt: str = ""
    temperature: float = 0.2
    max_tokens: int = 800
    timeout_seconds: float | None = None
    purpose: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    success: bool
    latency_ms: int = 0
    error_type: str | None = None
    error_message: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    redacted: bool = False


@dataclass
class ProviderHealth:
    provider: str
    configured: bool
    available: bool
    model: str
    message: str
    error_type: str | None = None
