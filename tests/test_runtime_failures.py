import json
import unittest
import asyncio
import tempfile
from pathlib import Path

from core.agents.planner import Plan, PlanStep, PlanValidation
from core.capabilities.contracts import CapabilityExecutionContext
from core.capabilities.executor import CapabilityExecutor
from core.capabilities.registry import CapabilityRegistry
from core.main import (
    WORKFLOW_MAX_FILES_TO_READ,
    WORKFLOW_MAX_TOTAL_CONTEXT_CHARS,
    build_synthesis_payload,
    capture_workflow_result,
    create_workflow_context,
    publish_failure_result,
    read_selected_files_for_workflow,
    run_planner_with_progress,
    select_files_for_synthesis,
)
from core.security.permissions import SecurityPolicy


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

    async def test_filesystem_search_output_is_captured_in_workflow_context(self):
        context = create_workflow_context("trace-ctx", "analyze repository architecture")

        capture_workflow_result(
            context,
            "filesystem.search",
            {"files": ["core/main.py", "core/.venv/lib/site.py"], "count": 2},
        )

        self.assertEqual(context["trace_id"], "trace-ctx")
        self.assertIn("core/main.py", context["files_found"])
        self.assertIn("core/.venv/lib/site.py", context["files_found"])
        self.assertEqual(context["last_result"]["capability_id"], "filesystem.search")

    async def test_filesystem_read_consumes_selected_search_results(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / "core").mkdir()
            (workspace / "core" / "main.py").write_text("orchestrator = True", encoding="utf-8")
            (workspace / "core" / "agents").mkdir()
            (workspace / "core" / "agents" / "planner.py").write_text("planner = True", encoding="utf-8")

            workflow_context = create_workflow_context("trace-read", "analyze repository architecture")
            capture_workflow_result(
                workflow_context,
                "filesystem.search",
                {"files": ["core/main.py", "core/agents/planner.py"], "count": 2},
            )

            reads = await read_selected_files_for_workflow(
                nc=FakeNATS(),
                trace_id="trace-read",
                executor=CapabilityExecutor(CapabilityRegistry(), SecurityPolicy()),
                workflow_context=workflow_context,
                capability_context=CapabilityExecutionContext(
                    trace_id="trace-read",
                    source_intent="analyze repository architecture",
                    workspace_root=str(workspace),
                ),
                record=FakeRecord(),
            )

            self.assertEqual(len(reads), 2)
            self.assertEqual([item["path"] for item in workflow_context["files_read"]], [
                "core/main.py",
                "core/agents/planner.py",
            ])

    async def test_research_synthesize_receives_prior_file_contents(self):
        workflow_context = create_workflow_context("trace-synth", "analyze repository architecture")
        capture_workflow_result(
            workflow_context,
            "filesystem.read",
            {"path": "core/main.py", "content": "NATS ACTIVE_TRACES planner executor", "size": 35, "truncated": False},
        )

        payload = build_synthesis_payload(
            workflow_context=workflow_context,
            topic="analyze repository architecture",
            step_input={"topic": "analyze repository architecture"},
        )

        self.assertEqual(payload["context"][0]["path"], "core/main.py")
        self.assertIn("NATS", payload["context"][0]["content"])
        self.assertEqual(payload["previous_results"][0]["capability_id"], "filesystem.read")

    async def test_synthesis_context_respects_max_file_count_and_total_chars(self):
        files = [
            "core/main.py",
            "core/agents/router.py",
            "core/agents/planner.py",
            "core/capabilities/executor.py",
            "core/security/permissions.py",
            "core/memory/manager.py",
            "apps/desktop/src/main.js",
            "apps/desktop/src-tauri/src/main.rs",
            "core/extra.py",
            "node_modules/pkg/index.py",
        ]

        selected = select_files_for_synthesis(files)
        self.assertEqual(len(selected), WORKFLOW_MAX_FILES_TO_READ)
        self.assertNotIn("node_modules/pkg/index.py", selected)

        workflow_context = create_workflow_context("trace-budget", "test")
        for index in range(WORKFLOW_MAX_FILES_TO_READ + 2):
            added = capture_workflow_result(
                workflow_context,
                "filesystem.read",
                {
                    "path": f"core/file_{index}.py",
                    "content": "x" * 5000,
                    "size": 5000,
                    "truncated": False,
                },
            )

        total_chars = sum(len(item["content"]) for item in workflow_context["files_read"])
        self.assertLessEqual(total_chars, WORKFLOW_MAX_TOTAL_CONTEXT_CHARS)
        self.assertTrue(any(item["truncated"] for item in workflow_context["files_read"]))

    async def test_workflow_context_does_not_leak_between_traces(self):
        first = create_workflow_context("trace-one", "first")
        second = create_workflow_context("trace-two", "second")

        capture_workflow_result(first, "filesystem.search", {"files": ["core/main.py"]})

        self.assertEqual(first["files_found"], ["core/main.py"])
        self.assertEqual(second["files_found"], [])


if __name__ == "__main__":
    unittest.main()
