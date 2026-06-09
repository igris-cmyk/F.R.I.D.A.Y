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
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OPENAI_TIMEOUT_SECONDS,
)


@dataclass(frozen=True)
class LLMSettings:
    provider: str = LLM_PROVIDER
    openai_api_key: str = OPENAI_API_KEY
    openai_base_url: str = OPENAI_BASE_URL
    openai_model: str = OPENAI_MODEL
    openai_timeout_seconds: float = OPENAI_TIMEOUT_SECONDS
    deepseek_api_key: str = DEEPSEEK_API_KEY
    deepseek_base_url: str = DEEPSEEK_BASE_URL
    deepseek_model: str = DEEPSEEK_MODEL
    deepseek_timeout_seconds: float = DEEPSEEK_TIMEOUT_SECONDS
    enable_local_llm: bool = ENABLE_LOCAL_LLM
    ollama_base_url: str = OLLAMA_BASE_URL
    ollama_model: str = OLLAMA_MODEL
