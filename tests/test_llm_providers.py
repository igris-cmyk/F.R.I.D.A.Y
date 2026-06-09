import asyncio
import importlib
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import AsyncMock, patch

from core.agents.planner import CognitivePlanner
from core.agents.router import classify_intent
from core.capabilities.contracts import CapabilityExecutionContext, CapabilityInvocation
from core.capabilities.executor import CapabilityExecutor
from core.capabilities.registry import CapabilityRegistry
from core.llm.model_router import ModelRouter
from core.llm.providers.deepseek_provider import DeepSeekProvider
from core.llm.providers.ollama_provider import OllamaProvider
from core.llm.providers.openai_provider import OpenAIProvider
from core.llm.redaction import redact_for_cloud
from core.llm.settings import LLMSettings
from core.llm.types import LLMRequest, LLMResponse
from core.main import execute_conversation
from core.security.permissions import SecurityPolicy


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        if isinstance(self.payload, bytes):
            return self.payload
        return json.dumps(self.payload).encode("utf-8")

    def close(self):
        return None


class FakeNATS:
    def __init__(self):
        self.published = []

    async def publish(self, subject, payload):
        self.published.append((subject, payload))


class FakeProvider:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def generate(self, request):
        self.calls.append(request)
        return self.response

    async def health(self):
        return None


class TestLLMProviderConfig(unittest.TestCase):
    def _reload_config_and_settings(self):
        import core.config as config_module
        import core.llm.settings as settings_module

        importlib.reload(config_module)
        importlib.reload(settings_module)
        return config_module, settings_module

    def test_openai_provider_settings_default(self):
        with patch.dict(os.environ, {}, clear=True):
            config_module, settings_module = self._reload_config_and_settings()
            settings = settings_module.LLMSettings()

        self.assertEqual(config_module.LLM_PROVIDER, "openai")
        self.assertEqual(settings.provider, "openai")
        self.assertEqual(settings.openai_base_url, "https://api.openai.com/v1")
        self.assertEqual(settings.openai_model, "gpt-5.4-mini")
        self.assertEqual(settings.openai_timeout_seconds, 30.0)
        self.assertEqual(settings.deepseek_model, "deepseek-v4-flash")
        self.assertFalse(settings.enable_local_llm)

    def test_openai_env_settings_override_defaults(self):
        env = {
            "OPENAI_API_KEY": "openai-key",
            "OPENAI_BASE_URL": "https://example.test/v1",
            "OPENAI_MODEL": "openai:test",
            "OPENAI_TIMEOUT_SECONDS": "12.5",
        }
        with patch.dict(os.environ, env, clear=True):
            _, settings_module = self._reload_config_and_settings()
            settings = settings_module.LLMSettings()

        self.assertEqual(settings.openai_api_key, "openai-key")
        self.assertEqual(settings.openai_base_url, "https://example.test/v1")
        self.assertEqual(settings.openai_model, "openai:test")
        self.assertEqual(settings.openai_timeout_seconds, 12.5)

    def test_ollama_can_be_selected_explicitly(self):
        env = {"LLM_PROVIDER": "ollama", "ENABLE_LOCAL_LLM": "true", "OLLAMA_MODEL": "local:test"}
        with patch.dict(os.environ, env, clear=True):
            _, settings_module = self._reload_config_and_settings()
            settings = settings_module.LLMSettings()

        self.assertEqual(settings.provider, "ollama")
        self.assertTrue(settings.enable_local_llm)
        self.assertEqual(settings.ollama_model, "local:test")


class TestOpenAIProvider(unittest.IsolatedAsyncioTestCase):
    async def test_missing_key_returns_clean_error(self):
        provider = OpenAIProvider(
            api_key="",
            base_url="https://api.openai.com/v1",
            model="openai-test",
            timeout_seconds=1,
        )

        response = await provider.generate(LLMRequest(messages=[{"role": "user", "content": "hello"}]))

        self.assertFalse(response.success)
        self.assertEqual(response.error_type, "missing_api_key")
        self.assertIn("OPENAI_API_KEY", response.text)

    async def test_success_response_is_parsed(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["authorization"] = request.headers.get("Authorization")
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            return FakeResponse({
                "choices": [{"message": {"content": "hello from openai"}}],
                "usage": {"total_tokens": 7},
            })

        provider = OpenAIProvider(
            api_key="secret-key",
            base_url="https://api.openai.com/v1",
            model="openai-test",
            timeout_seconds=3,
            urlopen=fake_urlopen,
        )

        response = await provider.generate(LLMRequest(messages=[{"role": "user", "content": "hello"}]))

        self.assertTrue(response.success)
        self.assertEqual(response.text, "hello from openai")
        self.assertEqual(response.usage["total_tokens"], 7)
        self.assertEqual(captured["authorization"], "Bearer secret-key")
        self.assertEqual(captured["url"], "https://api.openai.com/v1/chat/completions")
        self.assertEqual(captured["timeout"], 3)

    async def test_timeout_is_handled_cleanly(self):
        def fake_urlopen(request, timeout):
            raise TimeoutError()

        provider = OpenAIProvider("key", "https://api.openai.com/v1", "openai-test", 1, urlopen=fake_urlopen)
        response = await provider.generate(LLMRequest(messages=[{"role": "user", "content": "hello"}]))

        self.assertFalse(response.success)
        self.assertEqual(response.error_type, "timeout")
        self.assertNotIn("key", response.text)

    async def test_auth_rate_limit_quota_and_server_errors_are_handled(self):
        cases = (
            (401, {"error": {"message": "bad auth"}}, "auth_error"),
            (403, {"error": {"message": "forbidden"}}, "auth_error"),
            (429, {"error": {"message": "rate limited"}}, "rate_limit"),
            (429, {"error": {"message": "insufficient_quota billing required"}}, "insufficient_quota"),
            (500, {"error": {"message": "server down"}}, "server_error"),
        )
        for status, payload, expected_type in cases:
            def fake_urlopen(request, timeout, status=status, payload=payload):
                raise urllib.error.HTTPError(
                    url="https://api.openai.com/v1/chat/completions",
                    code=status,
                    msg="error",
                    hdrs={},
                    fp=FakeResponse(payload),
                )

            provider = OpenAIProvider("key", "https://api.openai.com/v1", "openai-test", 1, urlopen=fake_urlopen)
            response = await provider.generate(LLMRequest(messages=[{"role": "user", "content": "hello"}]))
            self.assertFalse(response.success)
            self.assertEqual(response.error_type, expected_type)
            self.assertNotIn("key", response.text)
            self.assertNotIn("key", response.error_message or "")

    async def test_bad_json_and_empty_response_are_handled(self):
        provider = OpenAIProvider(
            "key",
            "https://api.openai.com/v1",
            "openai-test",
            1,
            urlopen=lambda request, timeout: FakeResponse(b"{bad-json"),
        )
        bad_json = await provider.generate(LLMRequest(messages=[{"role": "user", "content": "hello"}]))
        self.assertEqual(bad_json.error_type, "bad_json")

        provider = OpenAIProvider(
            "key",
            "https://api.openai.com/v1",
            "openai-test",
            1,
            urlopen=lambda request, timeout: FakeResponse({"choices": []}),
        )
        empty = await provider.generate(LLMRequest(messages=[{"role": "user", "content": "hello"}]))
        self.assertEqual(empty.error_type, "empty_response")


class TestDeepSeekProvider(unittest.IsolatedAsyncioTestCase):
    async def test_missing_key_returns_clean_error(self):
        provider = DeepSeekProvider(
            api_key="",
            base_url="https://api.deepseek.com",
            model="deepseek-test",
            timeout_seconds=1,
        )

        response = await provider.generate(LLMRequest(messages=[{"role": "user", "content": "hello"}]))

        self.assertFalse(response.success)
        self.assertEqual(response.error_type, "missing_api_key")
        self.assertIn("DEEPSEEK_API_KEY", response.text)

    async def test_success_response_is_parsed(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["authorization"] = request.headers.get("Authorization")
            captured["timeout"] = timeout
            return FakeResponse({
                "choices": [{"message": {"content": "hello from deepseek"}}],
                "usage": {"total_tokens": 5},
            })

        provider = DeepSeekProvider(
            api_key="secret-key",
            base_url="https://api.deepseek.com",
            model="deepseek-test",
            timeout_seconds=3,
            urlopen=fake_urlopen,
        )

        response = await provider.generate(LLMRequest(messages=[{"role": "user", "content": "hello"}]))

        self.assertTrue(response.success)
        self.assertEqual(response.text, "hello from deepseek")
        self.assertEqual(response.usage["total_tokens"], 5)
        self.assertEqual(captured["authorization"], "Bearer secret-key")
        self.assertEqual(captured["timeout"], 3)

    async def test_timeout_is_handled_cleanly(self):
        def fake_urlopen(request, timeout):
            raise TimeoutError()

        provider = DeepSeekProvider("key", "https://api.deepseek.com", "deepseek-test", 1, urlopen=fake_urlopen)
        response = await provider.generate(LLMRequest(messages=[{"role": "user", "content": "hello"}]))

        self.assertFalse(response.success)
        self.assertEqual(response.error_type, "timeout")
        self.assertNotIn("key", response.text)

    async def test_rate_limit_and_server_errors_are_handled(self):
        for status, expected_type in ((429, "rate_limit"), (500, "server_error")):
            def fake_urlopen(request, timeout, status=status):
                raise urllib.error.HTTPError(
                    url="https://api.deepseek.com/chat/completions",
                    code=status,
                    msg="error",
                    hdrs={},
                    fp=FakeResponse({"error": {"message": "no secrets here"}}),
                )

            provider = DeepSeekProvider("key", "https://api.deepseek.com", "deepseek-test", 1, urlopen=fake_urlopen)
            response = await provider.generate(LLMRequest(messages=[{"role": "user", "content": "hello"}]))
            self.assertFalse(response.success)
            self.assertEqual(response.error_type, expected_type)
            self.assertNotIn("key", response.text)

    async def test_bad_json_and_empty_response_are_handled(self):
        provider = DeepSeekProvider(
            "key",
            "https://api.deepseek.com",
            "deepseek-test",
            1,
            urlopen=lambda request, timeout: FakeResponse(b"{bad-json"),
        )
        bad_json = await provider.generate(LLMRequest(messages=[{"role": "user", "content": "hello"}]))
        self.assertEqual(bad_json.error_type, "bad_json")

        provider = DeepSeekProvider(
            "key",
            "https://api.deepseek.com",
            "deepseek-test",
            1,
            urlopen=lambda request, timeout: FakeResponse({"choices": []}),
        )
        empty = await provider.generate(LLMRequest(messages=[{"role": "user", "content": "hello"}]))
        self.assertEqual(empty.error_type, "empty_response")


class TestLLMRedactionAndRouter(unittest.IsolatedAsyncioTestCase):
    def test_redaction_blocks_secret_heavy_content(self):
        redaction = redact_for_cloud(
            "DATABASE_URL=postgres://user:pass@host/db\n"
            "AUTH_SECRET=abc123\n"
            "Authorization: Bearer tokenvalue"
        )

        self.assertFalse(redaction.cloud_safe)
        self.assertIn("[REDACTED]", redaction.text)
        self.assertNotIn("postgres://user:pass", redaction.text)
        self.assertIn("env_like_content", redaction.reasons)

    def test_normal_prompt_is_cloud_safe(self):
        redaction = redact_for_cloud("Explain what a planner does in this architecture.")

        self.assertTrue(redaction.cloud_safe)
        self.assertFalse(redaction.redacted)

    async def test_cloud_safe_conversation_uses_deepseek(self):
        deepseek = FakeProvider(LLMResponse(
            text="provider answer",
            provider="deepseek",
            model="deepseek-test",
            success=True,
        ))
        router = ModelRouter(
            settings=LLMSettings(provider="deepseek", deepseek_api_key="key", deepseek_model="deepseek-test"),
            deepseek_provider=deepseek,
        )

        response = await router.generate(LLMRequest(messages=[{"role": "user", "content": "hello"}]))

        self.assertTrue(response.success)
        self.assertEqual(response.text, "provider answer")
        self.assertEqual(len(deepseek.calls), 1)

    async def test_cloud_safe_conversation_uses_openai(self):
        openai = FakeProvider(LLMResponse(
            text="openai answer",
            provider="openai",
            model="openai-test",
            success=True,
        ))
        router = ModelRouter(
            settings=LLMSettings(provider="openai", openai_api_key="key", openai_model="openai-test"),
            openai_provider=openai,
        )

        response = await router.generate(LLMRequest(messages=[{"role": "user", "content": "hello"}]))

        self.assertTrue(response.success)
        self.assertEqual(response.text, "openai answer")
        self.assertEqual(len(openai.calls), 1)

    async def test_secret_prompt_blocks_cloud_call(self):
        openai = FakeProvider(LLMResponse(text="should not happen", provider="openai", model="x", success=True))
        router = ModelRouter(
            settings=LLMSettings(provider="openai", openai_api_key="key", openai_model="openai-test"),
            openai_provider=openai,
        )

        response = await router.generate(LLMRequest(
            messages=[{"role": "user", "content": "DATABASE_URL=postgres://user:pass@host/db explain this"}]
        ))

        self.assertFalse(response.success)
        self.assertEqual(response.error_type, "privacy_blocked")
        self.assertEqual(response.provider, "openai")
        self.assertEqual(len(openai.calls), 0)

    async def test_ollama_selected_only_when_enabled(self):
        ollama = FakeProvider(LLMResponse(text="local answer", provider="ollama", model="local", success=True))
        disabled_router = ModelRouter(
            settings=LLMSettings(provider="ollama", enable_local_llm=False, ollama_model="local"),
            ollama_provider=ollama,
        )
        disabled = await disabled_router.generate(LLMRequest(messages=[{"role": "user", "content": "hello"}]))
        self.assertFalse(disabled.success)
        self.assertEqual(disabled.error_type, "disabled")
        self.assertEqual(len(ollama.calls), 0)

        enabled_router = ModelRouter(
            settings=LLMSettings(provider="ollama", enable_local_llm=True, ollama_model="local"),
            ollama_provider=ollama,
        )
        enabled = await enabled_router.generate(LLMRequest(messages=[{"role": "user", "content": "hello"}]))
        self.assertTrue(enabled.success)
        self.assertEqual(len(ollama.calls), 1)

    async def test_execute_conversation_streams_provider_updates(self):
        nc = FakeNATS()
        router = ModelRouter(
            settings=LLMSettings(provider="openai", openai_api_key="key", openai_model="openai-test"),
            openai_provider=FakeProvider(LLMResponse(
                text="conversation answer",
                provider="openai",
                model="openai-test",
                success=True,
            )),
        )

        output = await execute_conversation(nc, "trace-llm", "hello", router=router)

        self.assertEqual(output, "conversation answer")
        messages = [json.loads(payload.decode())["payload"]["message"] for _, payload in nc.published]
        self.assertIn("[LLM] Using provider openai model=openai-test", messages)
        self.assertIn("[LLM] Provider response received.", messages)


class TestLLMSafetyRegression(unittest.IsolatedAsyncioTestCase):
    async def test_known_deterministic_command_still_skips_provider_path(self):
        planner = CognitivePlanner(registry=CapabilityRegistry())
        planner._generate_llm_plan = AsyncMock(side_effect=AssertionError("Provider path should not run"))

        plan = await planner.generate_plan("show git status")

        self.assertEqual(plan.validation.source, "deterministic")
        self.assertEqual(plan.steps[0].capability_id, "git.status")

    async def test_ambiguous_router_skips_local_llm_when_disabled(self):
        state = await classify_intent({
            "raw_command": "coordinate something nuanced",
            "environment": {},
            "intent": "",
            "parameters": {},
            "error": "",
            "routing_metadata": {},
        })

        self.assertEqual(state["intent"], "conversation")
        self.assertEqual(state["routing_metadata"].get("reason"), "local_llm_disabled")

    async def test_dangerous_delete_still_blocked_by_security_policy(self):
        registry = CapabilityRegistry()
        executor = CapabilityExecutor(registry, SecurityPolicy())
        invocation = CapabilityInvocation(
            capability_id="shell.execute",
            input_payload={"command": "rm -rf ."},
            context=CapabilityExecutionContext(trace_id="llm-security", source_intent="delete everything"),
        )

        result = await executor.execute(invocation)

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "SECURITY_DENIAL")


if __name__ == "__main__":
    unittest.main()
