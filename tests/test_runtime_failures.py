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
    execute_memory_recall,
    log_memory_pipeline_task_result,
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


class HighConfidencePlanner:
    model = "qwen2.5:test"
    timeout_seconds = 6.0

    def generate_high_confidence_plan(self, intent):
        return Plan(
            steps=[PlanStep(capability_id="git.status", reason="Inspect repository.", input={"directory": "."})],
            estimated_risk="LOW",
            requires_confirmation=False,
            validation=PlanValidation(valid=True, source="deterministic", fallback_used=False),
        )

    async def generate_plan(self, intent):
        raise AssertionError("LLM planner path should not run for high-confidence plans.")


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


class FakeMemoryAgent:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def recall_context(self, query, policy, current_workspace=None):
        self.calls.append((query, policy, current_workspace))
        return self.result


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

    async def test_memory_pipeline_task_exception_is_observed(self):
        async def failing_task():
            raise RuntimeError("memory boom")

        task = asyncio.create_task(failing_task())
        await asyncio.sleep(0)

        with self.assertLogs("friday.core", level="ERROR") as logs:
            log_memory_pipeline_task_result(task)

        self.assertTrue(any("Background task failed" in line for line in logs.output))

    async def test_execute_memory_recall_returns_stored_memory_and_streams(self):
        nc = FakeNATS()
        agent = FakeMemoryAgent({
            "narrative": "We inspected the memory subsystem implementation.",
            "lineage": {
                "source_trace_ids": ["trace-memory"],
                "candidate_count": 1,
            },
        })

        output = await execute_memory_recall(
            nc,
            trace_id="trace-recall",
            query="what did we just inspect about memory?",
            environment={"working_directory": "/repo"},
            agent=agent,
        )

        self.assertIn("memory subsystem", output)
        self.assertIn("trace-memory", output)
        self.assertEqual(len(agent.calls), 1)
        self.assertEqual(len(nc.published), 2)
        messages = [json.loads(payload.decode())["payload"]["message"] for _, payload in nc.published]
        self.assertIn("[MEMORY] Searching persistent memory...", messages)
        self.assertIn("[MEMORY] Retrieved 1 relevant memories.", messages)

    async def test_execute_memory_recall_empty_memory_is_explicit(self):
        nc = FakeNATS()
        agent = FakeMemoryAgent(None)

        output = await execute_memory_recall(
            nc,
            trace_id="trace-empty",
            query="what did we do earlier?",
            environment={},
            agent=agent,
        )

        self.assertEqual(output, "No relevant continuity found.")
        messages = [json.loads(payload.decode())["payload"]["message"] for _, payload in nc.published]
        self.assertIn("[MEMORY] No relevant continuity found.", messages)

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
        self.assertTrue(any(message.startswith("[PLANNER] Local model planning started...") for message in messages))
        self.assertTrue(any("model=unknown" in message and "timeout=unknown" in message for message in messages))
        self.assertIn("[PLANNER] Waiting for local model...", messages)
        self.assertIn("planning_wait", record.stages)

    async def test_run_planner_with_progress_uses_deterministic_high_confidence_plan(self):
        nc = FakeNATS()
        record = FakeRecord()

        plan = await run_planner_with_progress(
            nc=nc,
            trace_id="trace-det",
            planner=HighConfidencePlanner(),
            intent="show git status",
            record=record,
            heartbeat_interval_seconds=0.5,
        )

        self.assertEqual(plan.validation.source, "deterministic")
        self.assertFalse(plan.validation.fallback_used)
        events = [json.loads(payload.decode()) for _, payload in nc.published]
        messages = [event["payload"]["message"] for event in events]
        self.assertIn(
            "[PLANNER] Using deterministic high-confidence plan. source=deterministic model=qwen2.5:test timeout=6.0",
            messages,
        )
        self.assertNotIn("[PLANNER] Waiting for local model...", messages)
        self.assertIn("planning_deterministic", record.stages)

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

    async def test_workflow_uses_ranked_selected_files_not_first_eight(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            paths = [
                "tests/test_noise.py",
                "core/test_event_loop.py",
                "core/verify_approval.py",
                "core/memory/manager.py",
                "core/memory/pipeline.py",
                "core/memory/retriever.py",
                "core/agents/memory_agent.py",
                "core/main.py",
            ]
            for path in paths:
                target = workspace / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(f"# {path}", encoding="utf-8")

            workflow_context = create_workflow_context("trace-ranked", "explain memory subsystem")
            capture_workflow_result(
                workflow_context,
                "filesystem.search",
                {"files": paths, "count": len(paths)},
            )

            reads = await read_selected_files_for_workflow(
                nc=FakeNATS(),
                trace_id="trace-ranked",
                executor=CapabilityExecutor(CapabilityRegistry(), SecurityPolicy()),
                workflow_context=workflow_context,
                capability_context=CapabilityExecutionContext(
                    trace_id="trace-ranked",
                    source_intent="explain memory subsystem",
                    workspace_root=str(workspace),
                ),
                record=FakeRecord(),
            )

            self.assertEqual([item["path"] for item in reads], [
                "core/memory/manager.py",
                "core/memory/pipeline.py",
                "core/memory/retriever.py",
                "core/agents/memory_agent.py",
            ])
            self.assertEqual(workflow_context["metadata"]["selected_files"], [
                "core/memory/manager.py",
                "core/memory/pipeline.py",
                "core/memory/retriever.py",
                "core/agents/memory_agent.py",
            ])

    async def test_research_telemetry_is_emitted_during_binding(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            for path in (
                "core/main.py",
                "core/agents/planner.py",
                "core/capabilities/executor.py",
                "core/security/permissions.py",
            ):
                target = workspace / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("x" * 100, encoding="utf-8")

            nc = FakeNATS()
            workflow_context = create_workflow_context("trace-telemetry", "analyze repository architecture")
            capture_workflow_result(
                workflow_context,
                "filesystem.search",
                {
                    "files": [
                        "core/main.py",
                        "core/agents/planner.py",
                        "core/capabilities/executor.py",
                        "core/security/permissions.py",
                    ],
                    "count": 4,
                },
            )

            await read_selected_files_for_workflow(
                nc=nc,
                trace_id="trace-telemetry",
                executor=CapabilityExecutor(CapabilityRegistry(), SecurityPolicy()),
                workflow_context=workflow_context,
                capability_context=CapabilityExecutionContext(
                    trace_id="trace-telemetry",
                    source_intent="analyze repository architecture",
                    workspace_root=str(workspace),
                ),
                record=FakeRecord(),
            )

            events = [json.loads(payload.decode()) for _, payload in nc.published]
            messages = [event["payload"]["message"] for event in events]
            self.assertTrue(any(message.startswith("[RESEARCH] Ranked 4 files") for message in messages))
            self.assertTrue(any(message.startswith("[RESEARCH] Top file: core/main.py") for message in messages))
            self.assertIn("[RESEARCH] Selected 4 files for grounded synthesis.", messages)
            self.assertTrue(any(message.startswith("[RESEARCH] Context budget: 400/12000 chars") for message in messages))

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
