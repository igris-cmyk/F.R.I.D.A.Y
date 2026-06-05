import asyncio
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from core.llm.providers.base import LLMProvider
from core.llm.types import LLMRequest, LLMResponse, ProviderHealth


class DeepSeekProvider(LLMProvider):
    provider_name = "deepseek"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float,
        urlopen: Callable[..., Any] = urllib.request.urlopen,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._urlopen = urlopen

    async def health(self) -> ProviderHealth:
        if not self.api_key:
            return ProviderHealth(
                provider=self.provider_name,
                configured=False,
                available=False,
                model=self.model,
                message="DeepSeek API key missing.",
                error_type="missing_api_key",
            )
        return ProviderHealth(
            provider=self.provider_name,
            configured=True,
            available=True,
            model=self.model,
            message="DeepSeek API key present. Live health not tested.",
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if not self.api_key:
            return LLMResponse(
                text="DeepSeek provider is not configured. Set DEEPSEEK_API_KEY.",
                provider=self.provider_name,
                model=self.model,
                success=False,
                error_type="missing_api_key",
                error_message="DEEPSEEK_API_KEY is missing.",
                redacted=False,
            )

        start = time.time()
        try:
            payload = {
                "model": self.model,
                "messages": self._messages(request),
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }
            timeout = request.timeout_seconds or self.timeout_seconds
            response_payload = await self._run_blocking(self._post_chat_completions, payload, timeout)
            text = self._extract_text(response_payload)
            if not text:
                return self._failure("empty_response", "DeepSeek returned an empty response.", start)
            return LLMResponse(
                text=text,
                provider=self.provider_name,
                model=self.model,
                success=True,
                latency_ms=int((time.time() - start) * 1000),
                usage=response_payload.get("usage", {}) if isinstance(response_payload, dict) else {},
            )
        except TimeoutError:
            return self._failure("timeout", "DeepSeek request timed out.", start)
        except urllib.error.HTTPError as exc:
            return self._http_failure(exc, start)
        except urllib.error.URLError as exc:
            return self._failure("network_error", f"DeepSeek network error: {exc.reason}", start)
        except json.JSONDecodeError:
            return self._failure("bad_json", "DeepSeek returned invalid JSON.", start)
        except Exception as exc:
            return self._failure("exception", f"DeepSeek request failed: {exc.__class__.__name__}", start)

    def _messages(self, request: LLMRequest) -> list[dict[str, str]]:
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.extend(request.messages)
        return messages

    def _post_chat_completions(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with self._urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    async def _run_blocking(self, func, *args):
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, lambda: func(*args))

    def _extract_text(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices", []) if isinstance(payload, dict) else []
        if not choices:
            return ""
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        return str(message.get("content", "")).strip()

    def _http_failure(self, exc: urllib.error.HTTPError, start: float) -> LLMResponse:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            body = ""
        lowered = body.lower()
        if exc.code in {401, 403}:
            error_type = "auth_error"
            message = "DeepSeek API returned an authentication error."
        elif exc.code == 429:
            error_type = "rate_limit"
            message = "DeepSeek rate limit reached."
        elif "insufficient" in lowered and "balance" in lowered:
            error_type = "insufficient_balance"
            message = "DeepSeek API reported insufficient balance."
        elif exc.code >= 500:
            error_type = "server_error"
            message = "DeepSeek API returned a server error."
        else:
            error_type = "http_error"
            message = f"DeepSeek API returned HTTP {exc.code}."
        return self._failure(error_type, message, start)

    def _failure(self, error_type: str, message: str, start: float) -> LLMResponse:
        return LLMResponse(
            text=f"{message} Cloud reasoning unavailable; local deterministic systems remain active.",
            provider=self.provider_name,
            model=self.model,
            success=False,
            latency_ms=int((time.time() - start) * 1000),
            error_type=error_type,
            error_message=message,
        )
