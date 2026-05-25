import unittest
import asyncio
import tempfile
import subprocess
from core.capabilities.registry import CapabilityRegistry, CapabilityDefinition
from core.security.permissions import SecurityPolicy, RiskLevel, SecurityEvaluation
from core.security.approval import PendingApproval, matches_pending_approval
from core.capabilities.executor import CapabilityExecutor
from core.capabilities.contracts import CapabilityInvocation, CapabilityExecutionContext, CapabilityRequiresApproval, CapabilityFailure, CapabilityResult

class TestCapabilityApproval(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.registry = CapabilityRegistry()
        self.security_policy = SecurityPolicy()
        self.executor = CapabilityExecutor(self.registry, self.security_policy)
        
        self.context = CapabilityExecutionContext(
            trace_id="test-trace-123",
            source_intent="Test intent",
            workspace_root="."
        )

    async def test_safe_capability_executes(self):
        invocation = CapabilityInvocation(
            capability_id="filesystem.search",
            input_payload={"pattern": "*.py", "root": "."},
            context=self.context
        )
        res = await self.executor.execute(invocation)
        self.assertIsInstance(res, CapabilityResult)
        self.assertEqual(res.success, True)
        self.assertIn("files", res.data)

    async def test_low_capability_executes(self):
        with tempfile.TemporaryDirectory() as repo_dir:
            subprocess.run(["git", "init"], cwd=repo_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            invocation = CapabilityInvocation(
                capability_id="git.status",
                input_payload={"directory": "."},
                context=CapabilityExecutionContext(
                    trace_id="test-trace-123",
                    source_intent="Test intent",
                    workspace_root=repo_dir
                )
            )
            res = await self.executor.execute(invocation)
            self.assertIsInstance(res, CapabilityResult)
            self.assertEqual(res.success, True)
            self.assertIn("status", res.data)

    async def test_medium_safe_returns_requires_approval(self):
        invocation = CapabilityInvocation(
            capability_id="test.medium_action",
            input_payload={},
            context=self.context
        )
        res = await self.executor.execute(invocation)
        self.assertIsInstance(res, CapabilityRequiresApproval)
        self.assertEqual(res.status, "REQUIRES_APPROVAL")
        self.assertEqual(res.risk_level, "MEDIUM")

    async def test_critical_capability_denied(self):
        invocation = CapabilityInvocation(
            capability_id="shell.execute",
            input_payload={"command": "ls"},
            context=self.context
        )
        res = await self.executor.execute(invocation)
        self.assertIsInstance(res, CapabilityFailure)
        self.assertEqual(res.error_code, "CAPABILITY_DISABLED")
        # Even if enabled, it would be denied by security policy. Let's force enable to test security policy.
        self.registry.get("shell.execute").enabled = True
        res = await self.executor.execute(invocation)
        self.assertIsInstance(res, CapabilityFailure)
        self.assertEqual(res.error_code, "SECURITY_DENIAL")
        self.assertIn("blocked", res.message.lower())

    async def test_approved_permission_continues_execution(self):
        invocation = CapabilityInvocation(
            capability_id="test.medium_action",
            input_payload={},
            context=self.context
        )
        # Bypassing the check as if the user approved
        
        # Note: We need a mock implementation in executor for test.medium_action
        # Since it's a mock, it will raise ValueError("No implementation found") from _execute_implementation
        # But we can catch that error and ensure it bypassed the security policy.
        res = await self.executor.execute(invocation, human_approved=True)
        self.assertIsInstance(res, CapabilityFailure)
        self.assertEqual(res.error_code, "EXECUTION_ERROR")
        self.assertIn("No implementation found", res.message)

    async def test_pending_approval_rejects_wrong_capability(self):
        future = asyncio.get_running_loop().create_future()
        pending = PendingApproval(future=future, capability_id="expected.capability", expires_at=9999999999.0)
        self.assertFalse(matches_pending_approval(pending, "trace-1", "wrong.capability", now=0.0))

    async def test_pending_approval_rejects_unknown_trace(self):
        self.assertFalse(matches_pending_approval(None, "unknown-trace", "expected.capability", now=0.0))

    async def test_pending_approval_rejects_expired_and_duplicate(self):
        future = asyncio.get_running_loop().create_future()
        pending = PendingApproval(future=future, capability_id="expected.capability", expires_at=1.0)
        self.assertFalse(matches_pending_approval(pending, "trace-1", "expected.capability", now=2.0))

        future.set_result("resolved")
        pending = PendingApproval(future=future, capability_id="expected.capability", expires_at=9999999999.0)
        self.assertFalse(matches_pending_approval(pending, "trace-1", "expected.capability", now=0.0))

if __name__ == '__main__':
    unittest.main()
