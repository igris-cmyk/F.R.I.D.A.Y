from core.llm.redaction import redact_messages_for_cloud
from core.llm.settings import LLMSettings
from core.llm.types import LLMRequest, LLMResponse
from core.llm.providers.deepseek_provider import DeepSeekProvider
from core.llm.providers.ollama_provider import OllamaProvider
from core.llm.providers.openai_provider import OpenAIProvider


class ModelRouter:
    def __init__(
        self,
        settings: LLMSettings | None = None,
        deepseek_provider: DeepSeekProvider | None = None,
        ollama_provider: OllamaProvider | None = None,
        openai_provider: OpenAIProvider | None = None,
    ):
        self.settings = settings or LLMSettings()
        self.openai_provider = openai_provider or OpenAIProvider(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            model=self.settings.openai_model,
            timeout_seconds=self.settings.openai_timeout_seconds,
        )
        self.deepseek_provider = deepseek_provider or DeepSeekProvider(
            api_key=self.settings.deepseek_api_key,
            base_url=self.settings.deepseek_base_url,
            model=self.settings.deepseek_model,
            timeout_seconds=self.settings.deepseek_timeout_seconds,
        )
        self.ollama_provider = ollama_provider or OllamaProvider(
            base_url=self.settings.ollama_base_url,
            model=self.settings.ollama_model,
            enabled=self.settings.enable_local_llm,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        provider_name = self.settings.provider
        if provider_name == "openai":
            return await self._generate_cloud(
                request=request,
                provider_name="openai",
                model=self.settings.openai_model,
                provider=self.openai_provider,
            )

        if provider_name == "deepseek":
            return await self._generate_cloud(
                request=request,
                provider_name="deepseek",
                model=self.settings.deepseek_model,
                provider=self.deepseek_provider,
            )

        if provider_name == "ollama":
            if not self.settings.enable_local_llm:
                return LLMResponse(
                    text="Local LLM provider is disabled. Set ENABLE_LOCAL_LLM=true to use Ollama.",
                    provider="ollama",
                    model=self.settings.ollama_model,
                    success=False,
                    error_type="disabled",
                    error_message="ENABLE_LOCAL_LLM is false.",
                )
            return await self.ollama_provider.generate(request)

        return LLMResponse(
            text=f"Configured LLM provider '{provider_name}' is not supported.",
            provider=provider_name,
            model="unknown",
            success=False,
            error_type="unsupported_provider",
            error_message=f"Unsupported provider: {provider_name}",
        )

    async def _generate_cloud(self, request: LLMRequest, provider_name: str, model: str, provider) -> LLMResponse:
        redacted_messages, redaction = redact_messages_for_cloud(request.messages)
        system_redaction = redact_messages_for_cloud([
            {"role": "system", "content": request.system_prompt}
        ])[1] if request.system_prompt else None
        reasons = redaction.reasons + (system_redaction.reasons if system_redaction else [])
        if reasons:
            return LLMResponse(
                text="This request appears to contain secrets. Cloud reasoning was not used.",
                provider=provider_name,
                model=model,
                success=False,
                error_type="privacy_blocked",
                error_message="Cloud request blocked by privacy guard.",
                redacted=redaction.redacted or bool(system_redaction and system_redaction.redacted),
            )
        safe_request = LLMRequest(
            messages=redacted_messages,
            system_prompt=system_redaction.text if system_redaction else request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            timeout_seconds=request.timeout_seconds,
            purpose=request.purpose,
            metadata=request.metadata,
        )
        response = await provider.generate(safe_request)
        response.redacted = redaction.redacted or response.redacted
        return response

    def selected_model(self) -> str:
        if self.settings.provider == "openai":
            return self.settings.openai_model
        if self.settings.provider == "deepseek":
            return self.settings.deepseek_model
        if self.settings.provider == "ollama":
            return self.settings.ollama_model
        return "unknown"

    async def health(self):
        if self.settings.provider == "openai":
            return await self.openai_provider.health()
        if self.settings.provider == "ollama":
            return await self.ollama_provider.health()
        return await self.deepseek_provider.health()


model_router = ModelRouter()
