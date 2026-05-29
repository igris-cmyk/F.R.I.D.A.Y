import json
import unittest
import asyncio

from core.agents.planner import Plan, PlanStep, PlanValidation
from core.main import publish_failure_result, run_planner_with_progress


class FakeNATS:
    def __init__(self):
        self.published = []

    async def publish(self, subject, payload):
        self.published.append((subject, payload))


class FakeRecord:
    def __init__(self):
        self.stages = []

    def bump_heartbeat(self, stage):
        self.stages.append(stage)


class SlowPlanner:
    async def generate_plan(self, intent):
        await asyncio.sleep(2.1)
        return Plan(
            steps=[PlanStep(capability_id="git.status", reason="Inspect repository.", input={"directory": "."})],
            estimated_risk="LOW",
            requires_confirmation=False,
            validation=PlanValidation(valid=True, source="llm", fallback_used=False),
        )


class TimeoutFallbackPlanner:
    async def generate_plan(self, intent):
        await asyncio.sleep(2.1)
        return Plan(
            steps=[PlanStep(capability_id="git.status", reason="Inspect repository.", input={"directory": "."})],
            estimated_risk="LOW",
            requires_confirmation=False,
            validation=PlanValidation(
                valid=True,
                source="deterministic",
                fallback_used=True,
                fallback_reason="timeout",
                errors=["Planner LLM inference timed out."],
            ),
        )


class TestRuntimeFailures(unittest.IsolatedAsyncioTestCase):
    async def test_publish_failure_result_emits_execution_result_event(self):
        nc = FakeNATS()

        await publish_failure_result(
            nc,
            trace_id="trace-123",
            error=RuntimeError("boom"),
            execution_time_ms=12,
        )

        self.assertEqual(len(nc.published), 1)
        subject, payload = nc.published[0]
        self.assertEqual(subject, "friday.stream.trace-123")

        event = json.loads(payload.decode())
        self.assertEqual(event["metadata"]["trace_id"], "trace-123")
        self.assertEqual(event["payload"]["status"], "failure")
        self.assertEqual(event["payload"]["error"], "boom")
        self.assertEqual(event["payload"]["execution_time_ms"], 12)

    async def test_run_planner_with_progress_emits_waiting_updates(self):
        nc = FakeNATS()
        record = FakeRecord()

        plan = await run_planner_with_progress(
            nc=nc,
            trace_id="trace-456",
            planner=SlowPlanner(),
            intent="analyze repository architecture",
            record=record,
            heartbeat_interval_seconds=0.5,
        )

        self.assertEqual(plan.validation.source, "llm")
        events = [json.loads(payload.decode()) for _, payload in nc.published]
        messages = [event["payload"]["message"] for event in events]
        self.assertIn("[PLANNER] Local model planning started...", messages)
        self.assertIn("[PLANNER] Waiting for local model...", messages)
        self.assertIn("planning_wait", record.stages)

    async def test_run_planner_with_progress_emits_timeout_fallback_update(self):
        nc = FakeNATS()
        record = FakeRecord()

        plan = await run_planner_with_progress(
            nc=nc,
            trace_id="trace-789",
            planner=TimeoutFallbackPlanner(),
            intent="analyze repository architecture",
            record=record,
            heartbeat_interval_seconds=0.5,
        )

        self.assertTrue(plan.validation.fallback_used)
        self.assertEqual(plan.validation.fallback_reason, "timeout")
        events = [json.loads(payload.decode()) for _, payload in nc.published]
        messages = [event["payload"]["message"] for event in events]
        self.assertIn("[PLANNER] Local planner timed out; using deterministic fallback.", messages)


if __name__ == "__main__":
    unittest.main()
