import unittest
import asyncio
import importlib
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch
from core.capabilities.contracts import CapabilityInvocation, CapabilityExecutionContext
from core.capabilities.registry import CapabilityRegistry, CapabilityDefinition
from core.security.permissions import SecurityPolicy, RiskLevel
from core.capabilities.executor import CapabilityExecutor
from core.agents.planner import CognitivePlanner
from core.agents.router import classify_intent


PROJECT_ROOT = Path(__file__).resolve().parents[1]

class TestCapabilityFramework(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        self.registry = CapabilityRegistry()
        self.security = SecurityPolicy()
        self.executor = CapabilityExecutor(self.registry, self.security)
        self.planner = CognitivePlanner(registry=self.registry)
        self.planner._generate_llm_plan = AsyncMock(side_effect=asyncio.TimeoutError())

    def _reload_config_and_planner(self):
        import core.config as config_module
        import core.agents.planner as planner_module

        importlib.reload(config_module)
        importlib.reload(planner_module)
        return config_module, planner_module

    def _reload_config(self):
        import core.config as config_module

        return importlib.reload(config_module)

    def test_default_role_model_mapping_is_low_resource_friendly(self):
        with patch.dict(os.environ, {}, clear=True):
            config_module = self._reload_config()
            self.assertEqual(config_module.FRIDAY_PLANNER_MODEL, "qwen2.5:1.5b")
            self.assertEqual(config_module.FRIDAY_ROUTER_MODEL, "qwen2.5:1.5b")
            self.assertEqual(config_module.FRIDAY_RESEARCH_MODEL, "qwen2.5:3b")
            self.assertEqual(config_module.FRIDAY_MEMORY_MODEL, "qwen2.5:1.5b")
            self.assertEqual(config_module.FRIDAY_CODE_MODEL, "qwen2.5-coder:1.5b")
            self.assertEqual(config_module.FRIDAY_EMBEDDING_MODEL, "nomic-embed-text")
            self.assertEqual(config_module.FRIDAY_PLANNER_TIMEOUT_SECONDS, 6.0)
            self.assertEqual(config_module.FRIDAY_RESEARCH_TIMEOUT_SECONDS, 12.0)
            self.assertEqual(config_module.FRIDAY_MEMORY_TIMEOUT_SECONDS, 8.0)
            self.assertEqual(config_module.FRIDAY_RESEARCH_MAX_FILES, 4)
            self.assertEqual(config_module.FRIDAY_RESEARCH_MAX_CHARS_PER_FILE, 3500)
            self.assertEqual(config_module.FRIDAY_RESEARCH_MAX_TOTAL_CHARS, 12000)
        self._reload_config()

    def test_default_planner_model_is_laptop_friendly(self):
        with patch.dict(os.environ, {}, clear=True):
            config_module, planner_module = self._reload_config_and_planner()
            planner = planner_module.CognitivePlanner(registry=self.registry)
            self.assertEqual(config_module.FRIDAY_PLANNER_MODEL, "qwen2.5:1.5b")
            self.assertEqual(planner.model, "qwen2.5:1.5b")
        self._reload_config_and_planner()

    def test_default_planner_timeout_is_laptop_friendly(self):
        with patch.dict(os.environ, {}, clear=True):
            config_module, planner_module = self._reload_config_and_planner()
            planner = planner_module.CognitivePlanner(registry=self.registry)
            self.assertEqual(config_module.FRIDAY_PLANNER_TIMEOUT_SECONDS, 6.0)
            self.assertEqual(planner.timeout_seconds, 6.0)
        self._reload_config_and_planner()

    def test_friday_planner_model_overrides_default(self):
        with patch.dict(os.environ, {"FRIDAY_PLANNER_MODEL": "qwen2.5:14b"}, clear=True):
            config_module, planner_module = self._reload_config_and_planner()
            planner = planner_module.CognitivePlanner(registry=self.registry)
            self.assertEqual(config_module.FRIDAY_PLANNER_MODEL, "qwen2.5:14b")
            self.assertEqual(planner.model, "qwen2.5:14b")
        self._reload_config_and_planner()

    def test_planner_timeout_env_overrides_default(self):
        with patch.dict(os.environ, {"FRIDAY_PLANNER_TIMEOUT_SECONDS": "8.5"}, clear=True):
            config_module, planner_module = self._reload_config_and_planner()
            planner = planner_module.CognitivePlanner(registry=self.registry)
            self.assertEqual(config_module.FRIDAY_PLANNER_TIMEOUT_SECONDS, 8.5)
            self.assertEqual(planner.timeout_seconds, 8.5)
        self._reload_config_and_planner()

    def test_planner_prompt_is_compact_for_local_models(self):
        planner = CognitivePlanner(registry=self.registry)
        self.assertLess(len(planner.prompt.template), 700)
        self.assertIn("Allowed capabilities:", planner.prompt.template)
        self.assertIn("Intent:\n{intent}", planner.prompt.template)
        self.assertNotIn("{context}", planner.prompt.template)

    def test_legacy_ollama_planner_model_fallback_still_works(self):
        with patch.dict(os.environ, {"OLLAMA_PLANNER_MODEL": "qwen2.5:legacy"}, clear=True):
            config_module, planner_module = self._reload_config_and_planner()
            planner = planner_module.CognitivePlanner(registry=self.registry)
            self.assertEqual(config_module.FRIDAY_PLANNER_MODEL, "qwen2.5:legacy")
            self.assertEqual(planner.model, "qwen2.5:legacy")
        self._reload_config_and_planner()

    def test_friday_planner_model_precedes_legacy_env(self):
        env = {
            "FRIDAY_PLANNER_MODEL": "qwen2.5:primary",
            "OLLAMA_PLANNER_MODEL": "qwen2.5:legacy",
        }
        with patch.dict(os.environ, env, clear=True):
            config_module, planner_module = self._reload_config_and_planner()
            planner = planner_module.CognitivePlanner(registry=self.registry)
            self.assertEqual(config_module.FRIDAY_PLANNER_MODEL, "qwen2.5:primary")
            self.assertEqual(planner.model, "qwen2.5:primary")
        self._reload_config_and_planner()

    def test_role_model_env_overrides_are_applied(self):
        env = {
            "FRIDAY_ROUTER_MODEL": "router:test",
            "FRIDAY_RESEARCH_MODEL": "research:test",
            "FRIDAY_MEMORY_MODEL": "memory:test",
            "FRIDAY_CODE_MODEL": "code:test",
            "FRIDAY_EMBEDDING_MODEL": "embedding:test",
            "FRIDAY_RESEARCH_TIMEOUT_SECONDS": "11.5",
            "FRIDAY_MEMORY_TIMEOUT_SECONDS": "7.5",
            "FRIDAY_RESEARCH_MAX_FILES": "3",
            "FRIDAY_RESEARCH_MAX_CHARS_PER_FILE": "1200",
            "FRIDAY_RESEARCH_MAX_TOTAL_CHARS": "3000",
        }
        with patch.dict(os.environ, env, clear=True):
            config_module = self._reload_config()
            self.assertEqual(config_module.FRIDAY_ROUTER_MODEL, "router:test")
            self.assertEqual(config_module.FRIDAY_RESEARCH_MODEL, "research:test")
            self.assertEqual(config_module.FRIDAY_MEMORY_MODEL, "memory:test")
            self.assertEqual(config_module.FRIDAY_CODE_MODEL, "code:test")
            self.assertEqual(config_module.FRIDAY_EMBEDDING_MODEL, "embedding:test")
            self.assertEqual(config_module.FRIDAY_RESEARCH_TIMEOUT_SECONDS, 11.5)
            self.assertEqual(config_module.FRIDAY_MEMORY_TIMEOUT_SECONDS, 7.5)
            self.assertEqual(config_module.FRIDAY_RESEARCH_MAX_FILES, 3)
            self.assertEqual(config_module.FRIDAY_RESEARCH_MAX_CHARS_PER_FILE, 1200)
            self.assertEqual(config_module.FRIDAY_RESEARCH_MAX_TOTAL_CHARS, 3000)
        self._reload_config()

    def test_agent_llm_modules_do_not_hardcode_local_model_literals(self):
        checked_files = [
            PROJECT_ROOT / "core" / "agents" / "router.py",
            PROJECT_ROOT / "core" / "agents" / "research.py",
            PROJECT_ROOT / "core" / "agents" / "memory_agent.py",
            PROJECT_ROOT / "core" / "agents" / "planner.py",
            PROJECT_ROOT / "core" / "memory" / "pipeline.py",
            PROJECT_ROOT / "core" / "capabilities" / "executor.py",
        ]
        disallowed_literals = [
            "qwen2.5:7b",
            "qwen2.5:3b",
            "qwen2.5:1.5b",
            "qwen2.5-coder:1.5b",
            "nomic-embed-text",
        ]
        for path in checked_files:
            content = path.read_text(encoding="utf-8")
            for literal in disallowed_literals:
                self.assertNotIn(literal, content)

    async def test_planner_output(self):
        plan = await self.planner.generate_plan("summarize python files")
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].capability_id, "filesystem.search")
        self.assertEqual(plan.estimated_risk, "LOW")
        self.assertTrue(plan.validation.valid)
        self.assertEqual(plan.validation.source, "deterministic")
        self.assertFalse(plan.validation.fallback_used)

    async def test_known_operational_prompts_skip_llm(self):
        for command, capability_id in (
            ("show git status", "git.status"),
            ("find python files", "filesystem.search"),
            ("read apps/desktop/package.json", "filesystem.read"),
        ):
            self.planner._generate_llm_plan.reset_mock()
            plan = await self.planner.generate_plan(command)
            self.assertEqual(plan.steps[0].capability_id, capability_id)
            self.assertEqual(plan.validation.source, "deterministic")
            self.assertFalse(plan.validation.fallback_used)
            self.planner._generate_llm_plan.assert_not_awaited()

    async def test_known_research_prompts_skip_llm(self):
        for command in (
            "analyze repository architecture",
            "explain memory subsystem",
            "show approval workflow",
        ):
            self.planner._generate_llm_plan.reset_mock()
            plan = await self.planner.generate_plan(command)
            self.assertEqual(plan.steps[-1].capability_id, "research.synthesize")
            self.assertEqual(plan.validation.source, "deterministic")
            self.assertFalse(plan.validation.fallback_used)
            self.planner._generate_llm_plan.assert_not_awaited()

    async def test_ambiguous_prompt_still_attempts_llm_planner(self):
        planner = CognitivePlanner(registry=self.registry)
        planner._generate_llm_plan = AsyncMock(side_effect=asyncio.TimeoutError())

        plan = await planner.generate_plan("coordinate a nuanced multi step repository investigation")
        planner._generate_llm_plan.assert_awaited_once()
        self.assertEqual(plan.validation.source, "deterministic")
        self.assertTrue(plan.validation.fallback_used)
        self.assertEqual(plan.validation.fallback_reason, "timeout")

    async def test_git_status_routes_to_capability(self):
        router_state = await classify_intent({
            "raw_command": "show git status",
            "environment": {},
            "intent": "",
            "parameters": {},
            "error": "",
            "routing_metadata": {},
        })
        self.assertEqual(router_state["intent"], "terminal")

        plan = await self.planner.generate_plan("show git status")
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].capability_id, "git.status")
        self.assertEqual(plan.validation.source, "deterministic")
        self.assertFalse(plan.validation.fallback_used)

    async def test_find_python_files_routes_to_filesystem_search(self):
        router_state = await classify_intent({
            "raw_command": "find python files",
            "environment": {},
            "intent": "",
            "parameters": {},
            "error": "",
            "routing_metadata": {},
        })
        self.assertEqual(router_state["intent"], "terminal")

        plan = await self.planner.generate_plan("find python files")
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].capability_id, "filesystem.search")
        self.assertEqual(plan.validation.source, "deterministic")
        self.assertFalse(plan.validation.fallback_used)

    async def test_read_package_json_routes_to_filesystem_read(self):
        router_state = await classify_intent({
            "raw_command": "read package.json",
            "environment": {},
            "intent": "",
            "parameters": {},
            "error": "",
            "routing_metadata": {},
        })
        self.assertEqual(router_state["intent"], "terminal")

        plan = await self.planner.generate_plan("read package.json")
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].capability_id, "filesystem.read")
        self.assertEqual(plan.validation.source, "deterministic")
        self.assertFalse(plan.validation.fallback_used)

    async def test_read_nested_package_json_routes_to_filesystem_read(self):
        plan = await self.planner.generate_plan("read apps/desktop/package.json")
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].capability_id, "filesystem.read")
        self.assertEqual(plan.steps[0].input["path"], "apps/desktop/package.json")
        self.assertEqual(plan.validation.source, "deterministic")
        self.assertFalse(plan.validation.fallback_used)

    async def test_hello_routes_to_conversation(self):
        router_state = await classify_intent({
            "raw_command": "hello friday",
            "environment": {},
            "intent": "",
            "parameters": {},
            "error": "",
            "routing_metadata": {},
        })
        self.assertEqual(router_state["intent"], "conversation")

    async def test_research_requests_route_without_llm_dependency(self):
        for command in (
            "analyze repository architecture",
            "explain memory subsystem",
        ):
            router_state = await classify_intent({
                "raw_command": command,
                "environment": {},
                "intent": "",
                "parameters": {},
                "error": "",
                "routing_metadata": {},
            })
            self.assertEqual(router_state["intent"], "research")

    async def test_delete_everything_is_not_conversation(self):
        router_state = await classify_intent({
            "raw_command": "delete everything in this folder",
            "environment": {},
            "intent": "",
            "parameters": {},
            "error": "",
            "routing_metadata": {},
        })
        self.assertEqual(router_state["intent"], "terminal")

        self.planner._generate_llm_plan.reset_mock()
        plan = await self.planner.generate_plan("delete everything in this folder")
        self.assertEqual(plan.steps[0].capability_id, "shell.execute")
        self.assertEqual(plan.estimated_risk, "CRITICAL")
        self.assertEqual(plan.validation.source, "deterministic")
        self.assertFalse(plan.validation.fallback_used)
        self.planner._generate_llm_plan.assert_not_awaited()

    async def test_remove_all_files_is_denied(self):
        router_state = await classify_intent({
            "raw_command": "remove all files",
            "environment": {},
            "intent": "",
            "parameters": {},
            "error": "",
            "routing_metadata": {},
        })
        self.assertEqual(router_state["intent"], "terminal")

        plan = await self.planner.generate_plan("remove all files")
        inv = CapabilityInvocation(
            capability_id=plan.steps[0].capability_id,
            input_payload=plan.steps[0].input,
            context=CapabilityExecutionContext(trace_id="test", source_intent="remove all files")
        )
        res = await self.executor.execute(inv)
        self.assertFalse(res.success)
        self.assertEqual(res.error_code, "SECURITY_DENIAL")

    async def test_rm_rf_is_denied(self):
        router_state = await classify_intent({
            "raw_command": "rm -rf .",
            "environment": {},
            "intent": "",
            "parameters": {},
            "error": "",
            "routing_metadata": {},
        })
        self.assertEqual(router_state["intent"], "terminal")

        plan = await self.planner.generate_plan("rm -rf .")
        inv = CapabilityInvocation(
            capability_id=plan.steps[0].capability_id,
            input_payload=plan.steps[0].input,
            context=CapabilityExecutionContext(trace_id="test", source_intent="rm -rf .")
        )
        res = await self.executor.execute(inv)
        self.assertFalse(res.success)
        self.assertEqual(res.error_code, "SECURITY_DENIAL")

    async def test_wipe_this_project_is_denied(self):
        router_state = await classify_intent({
            "raw_command": "wipe this project",
            "environment": {},
            "intent": "",
            "parameters": {},
            "error": "",
            "routing_metadata": {},
        })
        self.assertEqual(router_state["intent"], "terminal")

        plan = await self.planner.generate_plan("wipe this project")
        inv = CapabilityInvocation(
            capability_id=plan.steps[0].capability_id,
            input_payload=plan.steps[0].input,
            context=CapabilityExecutionContext(trace_id="test", source_intent="wipe this project")
        )
        res = await self.executor.execute(inv)
        self.assertFalse(res.success)
        self.assertEqual(res.error_code, "SECURITY_DENIAL")

    async def test_system_monitor_routes_to_capability(self):
        router_state = await classify_intent({
            "raw_command": "system monitor",
            "environment": {},
            "intent": "",
            "parameters": {},
            "error": "",
            "routing_metadata": {},
        })
        self.assertEqual(router_state["intent"], "terminal")

        plan = await self.planner.generate_plan("system monitor")
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].capability_id, "system.monitor")
        self.assertEqual(plan.validation.source, "deterministic")
        self.assertFalse(plan.validation.fallback_used)

    async def test_research_requests_generate_multi_step_plans(self):
        for command in (
            "analyze repository architecture",
            "explain memory subsystem",
            "show approval workflow",
        ):
            plan = await self.planner.generate_plan(command)
            self.assertGreaterEqual(len(plan.steps), 2)
            self.assertEqual(plan.steps[-1].capability_id, "research.synthesize")
            self.assertEqual(plan.validation.source, "deterministic")
            self.assertFalse(plan.validation.fallback_used)

    async def test_planner_critical_intent(self):
        self.planner._generate_llm_plan.reset_mock()
        plan = await self.planner.generate_plan("delete everything")
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].capability_id, "shell.execute")
        self.assertEqual(plan.estimated_risk, "CRITICAL")
        self.assertTrue(plan.requires_confirmation)
        self.planner._generate_llm_plan.assert_not_awaited()

    async def test_shell_execute_remains_disabled_or_gated(self):
        inv = CapabilityInvocation(
            capability_id="shell.execute",
            input_payload={"command": "rm -rf ."},
            context=CapabilityExecutionContext(trace_id="test", source_intent="rm -rf .")
        )
        res = await self.executor.execute(inv)
        self.assertFalse(res.success)
        self.assertIn(res.error_code, {"SECURITY_DENIAL", "CAPABILITY_DISABLED"})

    async def test_planner_invalid_json_falls_back(self):
        planner = CognitivePlanner(registry=self.registry)
        planner._generate_llm_plan = AsyncMock(side_effect=ValueError("LLM returned invalid JSON"))

        plan = await planner.generate_plan("coordinate a nuanced multi step repository investigation")
        self.assertEqual(plan.steps[0].capability_id, "system.monitor")
        self.assertEqual(plan.validation.source, "deterministic")
        self.assertTrue(plan.validation.fallback_used)
        self.assertEqual(plan.validation.fallback_reason, "invalid_json")
        self.assertIn("invalid JSON", plan.validation.errors[0])
        planner._generate_llm_plan.assert_awaited_once()

    async def test_planner_unknown_capability_falls_back(self):
        planner = CognitivePlanner(registry=self.registry)
        planner._generate_llm_plan = AsyncMock(return_value=planner._parse_llm_output(
            '{"steps":[{"capability_id":"unknown.capability","reason":"bad","input":{}}],"estimated_risk":"LOW","requires_confirmation":false}'
        ))

        plan = await planner.generate_plan("coordinate a nuanced multi step repository investigation")
        self.assertEqual(plan.steps[0].capability_id, "system.monitor")
        self.assertEqual(plan.validation.source, "deterministic")
        self.assertTrue(plan.validation.fallback_used)
        self.assertEqual(plan.validation.fallback_reason, "unknown_capability")
        self.assertIn("Unknown capability", plan.validation.errors[0])
        planner._generate_llm_plan.assert_awaited_once()

    async def test_planner_timeout_falls_back(self):
        planner = CognitivePlanner(registry=self.registry)
        planner._generate_llm_plan = AsyncMock(side_effect=asyncio.TimeoutError())

        plan = await planner.generate_plan("coordinate a nuanced multi step repository investigation")
        self.assertEqual(plan.steps[0].capability_id, "system.monitor")
        self.assertEqual(plan.validation.source, "deterministic")
        self.assertTrue(plan.validation.fallback_used)
        self.assertEqual(plan.validation.fallback_reason, "timeout")
        planner._generate_llm_plan.assert_awaited_once()

    async def test_planner_missing_required_input_falls_back(self):
        planner = CognitivePlanner(registry=self.registry)
        planner._generate_llm_plan = AsyncMock(return_value=planner._parse_llm_output(
            '{"steps":[{"capability_id":"filesystem.search","reason":"bad","input":{}}],"estimated_risk":"SAFE","requires_confirmation":false}'
        ))

        plan = await planner.generate_plan("coordinate a nuanced multi step repository investigation")
        self.assertEqual(plan.steps[0].capability_id, "system.monitor")
        self.assertEqual(plan.validation.source, "deterministic")
        self.assertTrue(plan.validation.fallback_used)
        self.assertEqual(plan.validation.fallback_reason, "missing_required_input")
        planner._generate_llm_plan.assert_awaited_once()

    async def test_planner_fallback_reason_is_exposed(self):
        plan = await self.planner.generate_plan("coordinate a nuanced multi step repository investigation")
        self.assertTrue(plan.validation.fallback_used)
        self.assertEqual(plan.validation.fallback_reason, "timeout")
        self.assertTrue(plan.validation.errors)

    async def test_model_missing_falls_back_with_visible_reason(self):
        planner = CognitivePlanner(registry=self.registry)
        planner._fetch_ollama_tags = lambda: {
            "models": [{"name": "qwen2.5:other", "model": "qwen2.5:other"}]
        }

        plan = await planner.generate_plan("coordinate a nuanced multi step repository investigation")
        self.assertEqual(plan.steps[0].capability_id, "system.monitor")
        self.assertEqual(plan.validation.source, "deterministic")
        self.assertTrue(plan.validation.fallback_used)
        self.assertEqual(plan.validation.fallback_reason, "ollama_unavailable")
        self.assertIn("model_missing", plan.validation.errors[0])
        self.assertIn(planner.model, plan.validation.errors[0])

    async def test_llm_plan_uses_configured_model_when_available(self):
        planner = CognitivePlanner(registry=self.registry)
        planner._fetch_ollama_tags = lambda: {
            "models": [{"name": planner.model, "model": planner.model}]
        }
        planner._invoke_llm = AsyncMock(return_value=(
            '{"steps":[{"capability_id":"git.status","reason":"Inspect repository status.","input":{"directory":"."}}],'
            '"estimated_risk":"LOW","requires_confirmation":false}'
        ))

        plan = await planner.generate_plan("coordinate a nuanced multi step repository investigation")
        self.assertEqual(plan.validation.source, "llm")
        self.assertFalse(plan.validation.fallback_used)
        self.assertIsNone(plan.validation.fallback_reason)
        self.assertEqual(plan.steps[0].capability_id, "git.status")
        planner._invoke_llm.assert_awaited_once()

    async def test_unknown_capability_rejection(self):
        inv = CapabilityInvocation(
            capability_id="unknown.capability",
            input_payload={},
            context=CapabilityExecutionContext(trace_id="test", source_intent="test")
        )
        res = await self.executor.execute(inv)
        self.assertFalse(res.success)
        self.assertEqual(res.error_code, "CAPABILITY_NOT_FOUND")

    async def test_security_policy_rejection_critical(self):
        inv = CapabilityInvocation(
            capability_id="shell.execute",
            input_payload={"command": "rm -rf"},
            context=CapabilityExecutionContext(trace_id="test", source_intent="test")
        )
        # Manually enable the capability in registry to isolate the security policy check
        self.registry.get("shell.execute").enabled = True
        
        res = await self.executor.execute(inv)
        self.assertFalse(res.success)
        self.assertEqual(res.error_code, "SECURITY_DENIAL")

    async def test_malformed_invocation_schema(self):
        inv = CapabilityInvocation(
            capability_id="filesystem.search",
            input_payload={}, # Missing 'pattern'
            context=CapabilityExecutionContext(trace_id="test", source_intent="test")
        )
        res = await self.executor.execute(inv)
        self.assertFalse(res.success)
        self.assertEqual(res.error_code, "INVALID_SCHEMA")

    async def test_filesystem_search_excludes_generated_dependency_dirs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / "core").mkdir()
            (workspace / "core" / "main.py").write_text("print('source')", encoding="utf-8")
            (workspace / ".venv" / "lib").mkdir(parents=True)
            (workspace / ".venv" / "lib" / "pollution.py").write_text("print('venv')", encoding="utf-8")
            (workspace / "node_modules" / "pkg").mkdir(parents=True)
            (workspace / "node_modules" / "pkg" / "pollution.py").write_text("print('node')", encoding="utf-8")
            (workspace / "target" / "debug").mkdir(parents=True)
            (workspace / "target" / "debug" / "pollution.py").write_text("print('target')", encoding="utf-8")

            inv = CapabilityInvocation(
                capability_id="filesystem.search",
                input_payload={"pattern": "*.py", "root": "."},
                context=CapabilityExecutionContext(
                    trace_id="test",
                    source_intent="find python files",
                    workspace_root=str(workspace)
                )
            )
            res = await self.executor.execute(inv)

            self.assertTrue(res.success)
            self.assertIn("core/main.py", res.data["files"])
            self.assertFalse(any(".venv/" in path for path in res.data["files"]))
            self.assertFalse(any("node_modules/" in path for path in res.data["files"]))
            self.assertFalse(any("target/" in path for path in res.data["files"]))
            self.assertEqual(res.data["count"], 1)
            self.assertFalse(res.data["truncated"])

    async def test_filesystem_search_truncates_at_max_results(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            for index in range(3):
                (workspace / f"file_{index}.py").write_text("print('x')", encoding="utf-8")

            inv = CapabilityInvocation(
                capability_id="filesystem.search",
                input_payload={"pattern": "*.py", "root": ".", "max_results": 2},
                context=CapabilityExecutionContext(
                    trace_id="test",
                    source_intent="find python files",
                    workspace_root=str(workspace)
                )
            )
            res = await self.executor.execute(inv)

            self.assertTrue(res.success)
            self.assertEqual(res.data["count"], 2)
            self.assertTrue(res.data["truncated"])

    async def test_research_synthesize_uses_grounded_context_without_simulation(self):
        original_availability = self.executor._ollama_model_available
        self.executor._ollama_model_available = lambda model: False
        try:
            inv = CapabilityInvocation(
                capability_id="research.synthesize",
                input_payload={
                    "topic": "analyze repository architecture",
                    "context": [
                        {
                            "path": "core/main.py",
                            "content": "NATS ACTIVE_TRACES planner capability execution streaming",
                            "size": 56,
                            "truncated": False,
                        }
                    ],
                },
                context=CapabilityExecutionContext(trace_id="test", source_intent="analyze repository architecture")
            )
            res = await self.executor.execute(inv)
        finally:
            self.executor._ollama_model_available = original_availability

        self.assertTrue(res.success)
        self.assertNotIn("Simulated synthesis", res.data["synthesis"])
        self.assertIn("core/main.py", res.data["synthesis"])
        self.assertTrue(res.data["grounded"])
        self.assertFalse(res.data["llm_used"])

    async def test_research_synthesize_context_respects_max_file_count(self):
        original_availability = self.executor._ollama_model_available
        self.executor._ollama_model_available = lambda model: False
        try:
            inv = CapabilityInvocation(
                capability_id="research.synthesize",
                input_payload={
                    "topic": "test",
                    "context": [
                        {"path": f"core/file_{index}.py", "content": "print('x')", "size": 10, "truncated": False}
                        for index in range(12)
                    ],
                },
                context=CapabilityExecutionContext(trace_id="test", source_intent="test")
            )
            res = await self.executor.execute(inv)
        finally:
            self.executor._ollama_model_available = original_availability

        self.assertTrue(res.success)
        self.assertEqual(res.data["context_file_count"], self.executor.SYNTHESIS_MAX_FILES)
        self.assertNotIn("core/file_8.py", res.data["inspected_files"])

    async def test_research_synthesize_context_respects_max_total_chars(self):
        original_availability = self.executor._ollama_model_available
        self.executor._ollama_model_available = lambda model: False
        try:
            inv = CapabilityInvocation(
                capability_id="research.synthesize",
                input_payload={
                    "topic": "test",
                    "context": [
                        {"path": f"core/file_{index}.py", "content": "x" * 5000, "size": 5000, "truncated": False}
                        for index in range(8)
                    ],
                },
                context=CapabilityExecutionContext(trace_id="test", source_intent="test")
            )
            res = await self.executor.execute(inv)
        finally:
            self.executor._ollama_model_available = original_availability

        self.assertTrue(res.success)
        self.assertEqual(res.data["context_file_count"], 5)
        self.assertIn("Content was truncated", res.data["synthesis"])

    async def test_invalid_file_read_fails_safely(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            inv = CapabilityInvocation(
                capability_id="filesystem.read",
                input_payload={"path": "missing.py"},
                context=CapabilityExecutionContext(
                    trace_id="test",
                    source_intent="read missing.py",
                    workspace_root=tmp_dir
                )
            )
            res = await self.executor.execute(inv)

        self.assertFalse(res.success)
        self.assertEqual(res.error_code, "EXECUTION_ERROR")
        self.assertIn("File not found", res.message)
        
    async def test_capability_timeout_enforcement(self):
        # Register a fake capability that hangs
        self.registry.register(CapabilityDefinition(
            capability_id="fake.hang",
            human_name="Hang",
            description="Hangs forever.",
            risk_level=RiskLevel.SAFE,
            input_schema={},
            timeout_seconds=1 # 1 second timeout
        ))
        
        # We must monkeypatch the executor's implementation specifically for this test
        original_impl = self.executor._execute_implementation
        async def slow_impl(inv, cid):
            if cid == "fake.hang":
                await asyncio.sleep(5)
                return {}
            return await original_impl(inv, cid)
        
        self.executor._execute_implementation = slow_impl
        
        try:
            inv = CapabilityInvocation(
                capability_id="fake.hang",
                input_payload={},
                context=CapabilityExecutionContext(trace_id="test", source_intent="test")
            )
            res = await self.executor.execute(inv)
            
            self.assertFalse(res.success)
            self.assertEqual(res.error_code, "TIMEOUT")
        finally:
            self.executor._execute_implementation = original_impl

    async def test_workspace_containment_rejects_sibling_prefix_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "project"
            sibling = Path(tmp_dir) / "project2"
            workspace.mkdir()
            sibling.mkdir()
            outside_file = sibling / "secret.txt"
            outside_file.write_text("secret", encoding="utf-8")

            inv = CapabilityInvocation(
                capability_id="filesystem.read",
                input_payload={"path": str(outside_file)},
                context=CapabilityExecutionContext(
                    trace_id="test",
                    source_intent="test",
                    workspace_root=str(workspace)
                )
            )
            res = await self.executor.execute(inv)
            self.assertFalse(res.success)
            self.assertEqual(res.error_code, "EXECUTION_ERROR")
            self.assertIn("outside of workspace scope", res.message)

    async def test_workspace_containment_allows_nested_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "project"
            workspace.mkdir()
            nested_file = workspace / "notes.txt"
            nested_file.write_text("hello", encoding="utf-8")

            inv = CapabilityInvocation(
                capability_id="filesystem.read",
                input_payload={"path": "notes.txt"},
                context=CapabilityExecutionContext(
                    trace_id="test",
                    source_intent="test",
                    workspace_root=str(workspace)
                )
            )
            res = await self.executor.execute(inv)
            self.assertTrue(res.success)
            self.assertEqual(res.data["content"], "hello")

if __name__ == '__main__':
    unittest.main()
