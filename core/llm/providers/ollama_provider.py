import asyncio
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from core.llm.providers.base import LLMProvider
from core.llm.types import LLMRequest, LLMResponse, ProviderHealth


class OllamaProvider(LLMProvider):
    provider_name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        enabled: bool,
        urlopen: Callable[..., Any] = urllib.request.urlopen,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.enabled = enabled
        self._urlopen = urlopen

    async def health(self) -> ProviderHealth:
        if not self.enabled:
            return ProviderHealth(
                provider=self.provider_name,
                configured=False,
                available=False,
                model=self.model,
                message="Local LLM provider disabled.",
                error_type="disabled",
            )
        try:
            await self._run_blocking(self._get_tags)
            return ProviderHealth(
                provider=self.provider_name,
                configured=True,
                available=True,
                model=self.model,
                message="Ollama reachable.",
            )
        except Exception:
            return ProviderHealth(
                provider=self.provider_name,
                configured=True,
                available=False,
                model=self.model,
                message="Ollama unavailable.",
                error_type="ollama_unavailable",
            )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if not self.enabled:
            return LLMResponse(
                text="Local LLM provider is disabled. Set ENABLE_LOCAL_LLM=true to use Ollama.",
                provider=self.provider_name,
                model=self.model,
                success=False,
                error_type="disabled",
                error_message="ENABLE_LOCAL_LLM is false.",
            )

        start = time.time()
        try:
            prompt = self._prompt(request)
            timeout = request.timeout_seconds or 12.0
            payload = await self._run_blocking(self._post_generate, prompt, timeout)
            text = str(payload.get("response", "")).strip()
            if not text:
                return self._failure("empty_response", "Ollama returned an empty response.", start)
            return LLMResponse(
                text=text,
                provider=self.provider_name,
                model=self.model,
                success=True,
                latency_ms=int((time.time() - start) * 1000),
            )
        except TimeoutError:
            return self._failure("timeout", "Ollama request timed out.", start)
        except Exception:
            return self._failure("ollama_unavailable", "Ollama request failed.", start)

    def _get_tags(self) -> dict[str, Any]:
        with self._urlopen(f"{self.base_url}/api/tags", timeout=1.0) as response:
            return json.loads(response.read().decode("utf-8"))

    def _post_generate(self, prompt: str, timeout: float) -> dict[str, Any]:
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _prompt(self, request: LLMRequest) -> str:
        parts = []
        if request.system_prompt:
            parts.append(request.system_prompt)
        parts.extend(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in request.messages)
        return "\n\n".join(parts)

    def _failure(self, error_type: str, message: str, start: float) -> LLMResponse:
        return LLMResponse(
            text=f"{message} Local deterministic systems remain active.",
            provider=self.provider_name,
            model=self.model,
            success=False,
            latency_ms=int((time.time() - start) * 1000),
            error_type=error_type,
            error_message=message,
        )

    async def _run_blocking(self, func, *args):
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, lambda: func(*args))
