import unittest
import asyncio
import tempfile
from pathlib import Path
from core.capabilities.contracts import CapabilityInvocation, CapabilityExecutionContext
from core.capabilities.registry import CapabilityRegistry, CapabilityDefinition
from core.security.permissions import SecurityPolicy, RiskLevel
from core.capabilities.executor import CapabilityExecutor
from core.agents.planner import CognitivePlanner
from core.agents.router import classify_intent

class TestCapabilityFramework(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        self.registry = CapabilityRegistry()
        self.security = SecurityPolicy()
        self.executor = CapabilityExecutor(self.registry, self.security)
        self.planner = CognitivePlanner()

    async def test_planner_output(self):
        plan = await self.planner.generate_plan("summarize python files")
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].capability_id, "filesystem.search")
        self.assertEqual(plan.estimated_risk, "LOW")

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

        plan = await self.planner.generate_plan("delete everything in this folder")
        self.assertEqual(plan.steps[0].capability_id, "shell.execute")
        self.assertEqual(plan.estimated_risk, "CRITICAL")

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

    async def test_planner_critical_intent(self):
        plan = await self.planner.generate_plan("delete everything")
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].capability_id, "shell.execute")
        self.assertEqual(plan.estimated_risk, "CRITICAL")
        self.assertTrue(plan.requires_confirmation)

    async def test_shell_execute_remains_disabled_or_gated(self):
        inv = CapabilityInvocation(
            capability_id="shell.execute",
            input_payload={"command": "rm -rf ."},
            context=CapabilityExecutionContext(trace_id="test", source_intent="rm -rf .")
        )
        res = await self.executor.execute(inv)
        self.assertFalse(res.success)
        self.assertIn(res.error_code, {"SECURITY_DENIAL", "CAPABILITY_DISABLED"})

    async def test_planner_malformed_fallback(self):
        with self.assertRaises(ValueError):
            await self.planner.generate_plan("malformed json")

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
