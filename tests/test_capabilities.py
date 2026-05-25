import unittest
import asyncio
import tempfile
from pathlib import Path
from core.capabilities.contracts import CapabilityInvocation, CapabilityExecutionContext
from core.capabilities.registry import CapabilityRegistry, CapabilityDefinition
from core.security.permissions import SecurityPolicy, RiskLevel
from core.capabilities.executor import CapabilityExecutor
from core.agents.planner import CognitivePlanner

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

    async def test_planner_critical_intent(self):
        plan = await self.planner.generate_plan("delete everything")
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].capability_id, "shell.execute")
        self.assertEqual(plan.estimated_risk, "CRITICAL")
        self.assertTrue(plan.requires_confirmation)

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
