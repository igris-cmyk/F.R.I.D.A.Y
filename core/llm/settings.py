from dataclasses import dataclass

from core.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_TIMEOUT_SECONDS,
    ENABLE_LOCAL_LLM,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)


@dataclass(frozen=True)
class LLMSettings:
    provider: str = LLM_PROVIDER
    deepseek_api_key: str = DEEPSEEK_API_KEY
    deepseek_base_url: str = DEEPSEEK_BASE_URL
    deepseek_model: str = DEEPSEEK_MODEL
    deepseek_timeout_seconds: float = DEEPSEEK_TIMEOUT_SECONDS
    enable_local_llm: bool = ENABLE_LOCAL_LLM
    ollama_base_url: str = OLLAMA_BASE_URL
    ollama_model: str = OLLAMA_MODEL
