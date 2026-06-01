import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES_PATH = PROJECT_ROOT / "core" / "evals" / "cases.json"
DEFAULT_REPORT_DIR = PROJECT_ROOT / ".friday" / "evals"
DEFAULT_EVAL_DB_PATH = DEFAULT_REPORT_DIR / "eval_memory.sqlite3"
GLOBAL_MAX_LATENCY_MS = 30000
WARN_LATENCY_MS = 15000


@dataclass
class EvalTraceResult:
    eval_id: str
    prompt: str
    expected_route: Any = None
    actual_route: str | None = None
    planner_source: str | None = None
    fallback_used: bool | None = None
    risk: str | None = None
    capabilities_started: list[str] = field(default_factory=list)
    capabilities_completed: list[str] = field(default_factory=list)
    selected_files: list[str] = field(default_factory=list)
    status: str = "FAILURE"
    security_blocked: bool = False
    memory_item_count_before: int = 0
    memory_item_count_after: int = 0
    memory_retrieved_count: int = 0
    latency_ms: int = 0
    output: str = ""
    passed: bool = False
    failure_reason: str | None = None


class EvalNATS:
    def __init__(self):
        self.published: list[tuple[str, bytes]] = []

    async def publish(self, subject: str, payload: bytes):
        self.published.append((subject, payload))

    def messages(self) -> list[str]:
        messages = []
        for _, payload in self.published:
            try:
                data = json.loads(payload.decode())
            except Exception:
                continue
            message = data.get("payload", {}).get("message")
            if message:
                messages.append(message)
        return messages


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        raise ValueError("Eval case file must contain a list.")
    for case in cases:
        if not isinstance(case, dict) or not case.get("id") or not case.get("prompt"):
            raise ValueError("Each eval case must include id and prompt.")
        if not isinstance(case.get("expected", {}), dict):
            raise ValueError(f"Eval case {case.get('id')} expected field must be an object.")
    return cases


def reset_eval_memory_db(db_path: Path = DEFAULT_EVAL_DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (db_path, db_path.with_suffix(db_path.suffix + "-wal"), db_path.with_suffix(db_path.suffix + "-shm")):
        if path.exists():
            path.unlink()


def configure_eval_environment(db_path: Path = DEFAULT_EVAL_DB_PATH) -> None:
    os.environ["FRIDAY_MEMORY_BACKEND"] = "sqlite"
    os.environ["FRIDAY_MEMORY_DB_PATH"] = str(db_path)


async def initialize_eval_memory(db_path: Path = DEFAULT_EVAL_DB_PATH):
    from core.memory.manager import MemoryHealthState, memory_manager
    from core.memory.sqlite_store import SQLiteMemoryStore

    memory_manager.db_path = str(db_path)
    memory_manager.store = SQLiteMemoryStore(str(db_path))
    memory_manager.health_state = MemoryHealthState.OFFLINE
    memory_manager.degraded_reason = None
    memory_manager.embedding_available = False
    await memory_manager.initialize()
    return memory_manager


async def run_intent_for_eval(
    command: str,
    eval_id: str = "adhoc",
    workspace_root: str | None = None,
    persist_trace: bool = False,
) -> EvalTraceResult:
    from core.agents.router import classify_intent
    from core.capabilities.contracts import CapabilityExecutionContext, CapabilityInvocation
    from core.main import (
        build_synthesis_payload,
        capture_workflow_result,
        create_workflow_context,
        execute_memory_recall,
        executor,
        planner,
        read_selected_files_for_workflow,
        run_planner_with_progress,
    )
    from core.memory.manager import memory_manager
    from core.memory.pipeline import process_completed_trace

    workspace = str(Path(workspace_root or PROJECT_ROOT).resolve())
    trace_id = f"eval-{eval_id}-{uuid.uuid4().hex[:8]}"
    nc = EvalNATS()
    start = time.time()
    health_before = await memory_manager.health()
    result = EvalTraceResult(
        eval_id=eval_id,
        prompt=command,
        memory_item_count_before=int(health_before.get("item_count", 0)),
    )

    try:
        final_state = await classify_intent({
            "raw_command": command,
            "environment": {"working_directory": workspace, "workspace_root": workspace},
            "intent": "",
            "parameters": {},
            "error": "",
            "routing_metadata": {},
        })
        intent_type = final_state.get("intent", "conversation")
        result.actual_route = intent_type.upper()

        if intent_type == "conversation":
            result.output = f"[Conversational Response] Acknowledged: {final_state['parameters'].get('message')}"
            result.status = "SUCCESS"
        elif intent_type == "memory":
            result.output = await execute_memory_recall(
                nc,
                trace_id,
                final_state["parameters"].get("query") or command,
                {"working_directory": workspace, "workspace_root": workspace},
            )
            result.memory_retrieved_count = _memory_retrieved_count_from_messages(nc.messages())
            result.status = "SUCCESS"
        else:
            plan = await run_planner_with_progress(
                nc=nc,
                trace_id=trace_id,
                planner=planner,
                intent=command,
                heartbeat_interval_seconds=1.0,
            )
            result.planner_source = plan.validation.source
            result.fallback_used = plan.validation.fallback_used
            result.risk = plan.estimated_risk

            capability_context = CapabilityExecutionContext(
                trace_id=trace_id,
                source_intent=command,
                workspace_root=workspace,
            )
            workflow_context = create_workflow_context(trace_id, command)
            step_outputs = []
            failed = False

            for step in plan.steps:
                step_input = dict(step.input or {})
                if step.capability_id == "filesystem.read" and not step_input.get("path") and workflow_context["files_found"]:
                    selected = workflow_context["files_found"][0]
                    step_input["path"] = selected

                if step.capability_id == "research.synthesize":
                    if not workflow_context["files_read"] and workflow_context["files_found"]:
                        bound_reads = await read_selected_files_for_workflow(
                            nc=nc,
                            trace_id=trace_id,
                            executor=executor,
                            workflow_context=workflow_context,
                            capability_context=capability_context,
                        )
                        if bound_reads:
                            result.capabilities_started.append("filesystem.read")
                            result.capabilities_completed.append("filesystem.read")
                    step_input = build_synthesis_payload(workflow_context, command, step_input)

                result.capabilities_started.append(step.capability_id)
                invocation = CapabilityInvocation(
                    capability_id=step.capability_id,
                    input_payload=step_input,
                    context=capability_context,
                    requires_confirmation=plan.requires_confirmation,
                )
                capability_result = await executor.execute(invocation)
                if getattr(capability_result, "status", None) == "REQUIRES_APPROVAL":
                    result.security_blocked = True
                    result.output = f"FAILED {step.capability_id}: {capability_result.reason}"
                    result.status = "FAILURE"
                    failed = True
                    break

                if not getattr(capability_result, "success", False):
                    message = getattr(capability_result, "message", "")
                    result.security_blocked = getattr(capability_result, "error_code", "") == "SECURITY_DENIAL"
                    result.output = f"FAILED {step.capability_id}: {message}"
                    result.status = "FAILURE"
                    failed = True
                    break

                capture_workflow_result(workflow_context, step.capability_id, capability_result.data)
                result.capabilities_completed.append(step.capability_id)
                step_outputs.append(f"SUCCESS {step.capability_id}: {capability_result.data}")

            if step_outputs and not failed:
                result.output = "\n".join(step_outputs)
                result.status = "SUCCESS"
            result.selected_files = list(workflow_context["metadata"].get("selected_files", []))

        if persist_trace:
            await process_completed_trace(
                trace_id=trace_id,
                intent=(result.actual_route or "").lower(),
                command=command,
                result=result.output,
                error_state=result.status != "SUCCESS",
                environment={"working_directory": workspace, "workspace_root": workspace},
                metadata={"eval_id": eval_id, "source_component": "core.eval_harness"},
            )
    except Exception as exc:
        result.status = "FAILURE"
        result.output = f"[EVAL ERROR] {exc}"
        result.failure_reason = str(exc)
    finally:
        result.latency_ms = int((time.time() - start) * 1000)
        health_after = await memory_manager.health()
        result.memory_item_count_after = int(health_after.get("item_count", 0))

    return result


def _memory_retrieved_count_from_messages(messages: list[str]) -> int:
    for message in messages:
        match = re.search(r"\[MEMORY\] Retrieved (\d+) relevant memories\.", message)
        if match:
            return int(match.group(1))
    return 0


def evaluate_result(result: EvalTraceResult, expected: dict[str, Any]) -> EvalTraceResult:
    failures = []
    output = result.output or ""

    if "route" in expected and not _matches_expected(expected["route"], result.actual_route):
        failures.append(f"route expected {expected['route']} got {result.actual_route}")
    if "status" in expected and expected["status"] != result.status:
        failures.append(f"status expected {expected['status']} got {result.status}")
    if "planner_source" in expected and expected["planner_source"] != result.planner_source:
        failures.append(f"planner_source expected {expected['planner_source']} got {result.planner_source}")
    if "fallback_used" in expected and bool(expected["fallback_used"]) != bool(result.fallback_used):
        failures.append(f"fallback_used expected {expected['fallback_used']} got {result.fallback_used}")
    if "risk" in expected and expected["risk"] != result.risk:
        failures.append(f"risk expected {expected['risk']} got {result.risk}")
    if "capability" in expected and expected["capability"] not in result.capabilities_started:
        failures.append(f"capability {expected['capability']} was not started")
    if "capabilities" in expected:
        missing = [capability for capability in expected["capabilities"] if capability not in result.capabilities_started]
        if missing:
            failures.append(f"capabilities not started: {missing}")
    if "capabilities_started" in expected and expected["capabilities_started"] != result.capabilities_started:
        failures.append(f"capabilities_started expected {expected['capabilities_started']} got {result.capabilities_started}")
    if "selected_files_any" in expected:
        if not any(path in result.selected_files for path in expected["selected_files_any"]):
            failures.append(f"none of selected_files_any were selected: {expected['selected_files_any']}")
    for needle in expected.get("must_contain", []):
        if needle.lower() not in output.lower():
            failures.append(f"output missing required text: {needle}")
    if "must_contain_any" in expected:
        if not any(needle.lower() in output.lower() for needle in expected["must_contain_any"]):
            failures.append(f"output missing any of: {expected['must_contain_any']}")
    for needle in expected.get("must_not_contain", []):
        if needle.lower() in output.lower():
            failures.append(f"output contained forbidden text: {needle}")
    if expected.get("security_blocked") is True and not result.security_blocked:
        failures.append("expected security_blocked=true")
    if expected.get("security_blocked_or_safe_failure") is True:
        if result.status != "FAILURE" and not result.security_blocked:
            failures.append("expected safe failure or security block")
    if "memory_retrieved_count_min" in expected:
        if result.memory_retrieved_count < int(expected["memory_retrieved_count_min"]):
            failures.append(
                f"memory_retrieved_count expected >= {expected['memory_retrieved_count_min']} got {result.memory_retrieved_count}"
            )

    max_latency = int(expected.get("max_latency_ms", GLOBAL_MAX_LATENCY_MS))
    if result.latency_ms > max_latency:
        failures.append(f"latency exceeded {max_latency}ms: {result.latency_ms}ms")

    result.passed = not failures
    result.failure_reason = "; ".join(failures) if failures else None
    return result


def _matches_expected(expected: Any, actual: Any) -> bool:
    if isinstance(expected, list):
        return actual in expected
    return expected == actual


async def run_eval_suite(
    cases_path: Path = DEFAULT_CASES_PATH,
    report_dir: Path = DEFAULT_REPORT_DIR,
    memory_db_path: Path = DEFAULT_EVAL_DB_PATH,
    workspace_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    configure_eval_environment(memory_db_path)
    reset_eval_memory_db(memory_db_path)
    memory_manager = await initialize_eval_memory(memory_db_path)
    from core.agents.memory_agent import memory_agent
    from core.memory import pipeline as memory_pipeline
    from core.memory.manager import MemoryHealthState

    old_compress_workflow = memory_pipeline.compress_workflow
    old_generate_embedding = memory_pipeline.generate_embedding
    old_reconstruct_narrative = memory_agent._reconstruct_narrative

    async def eval_compress_workflow(normalized_trace: dict[str, Any]) -> str:
        return memory_pipeline._fallback_summary(normalized_trace, reason="eval_harness")

    async def eval_reconstruct_narrative(query: str, candidates: list[dict[str, Any]]) -> str:
        return memory_agent._deterministic_recall_summary(candidates, reason="eval_harness")

    async def eval_generate_embedding(text: str) -> None:
        return None

    memory_pipeline.compress_workflow = eval_compress_workflow
    memory_pipeline.generate_embedding = eval_generate_embedding
    memory_agent._reconstruct_narrative = eval_reconstruct_narrative
    memory_manager.embedding_available = False
    if memory_manager.health_state == MemoryHealthState.HEALTHY:
        memory_manager.health_state = MemoryHealthState.DEGRADED
        memory_manager.degraded_reason = "eval_harness_forces_keyword_memory"

    cases = load_cases(cases_path)
    results: list[EvalTraceResult] = []

    try:
        for case in cases:
            for setup in case.get("setup", []):
                setup_result = await run_intent_for_eval(
                    setup["prompt"],
                    eval_id=f"{case['id']}_setup",
                    workspace_root=str(workspace_root),
                    persist_trace=True,
                )
                wait_seconds = float(setup.get("wait_seconds", 0))
                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)
                if setup_result.status != "SUCCESS":
                    result = EvalTraceResult(
                        eval_id=case["id"],
                        prompt=case["prompt"],
                        status="FAILURE",
                        output=setup_result.output,
                        failure_reason=f"setup failed: {setup_result.failure_reason or setup_result.output}",
                    )
                    results.append(evaluate_result(result, case.get("expected", {})))
                    break
            else:
                result = await run_intent_for_eval(
                    case["prompt"],
                    eval_id=case["id"],
                    workspace_root=str(workspace_root),
                )
                result.expected_route = case.get("expected", {}).get("route")
                results.append(evaluate_result(result, case.get("expected", {})))
    finally:
        memory_pipeline.compress_workflow = old_compress_workflow
        memory_pipeline.generate_embedding = old_generate_embedding
        memory_agent._reconstruct_narrative = old_reconstruct_narrative

    health = await memory_manager.health()
    report = build_report(results, memory_health=health)
    write_reports(report, report_dir)
    return report


def build_report(results: list[EvalTraceResult], memory_health: dict[str, Any] | None = None) -> dict[str, Any]:
    passed = sum(1 for result in results if result.passed)
    failed = len(results) - passed
    security_regressions = sum(
        1
        for result in results
        if not result.passed and ("delete" in result.eval_id or "escape" in result.eval_id)
    )
    memory_regressions = sum(1 for result in results if not result.passed and "memory" in result.eval_id)
    latencies = [result.latency_ms for result in results]
    slowest = max(results, key=lambda item: item.latency_ms, default=None)
    warnings = [
        f"{result.eval_id} exceeded warning latency: {result.latency_ms}ms"
        for result in results
        if result.latency_ms > WARN_LATENCY_MS
    ]
    return {
        "summary": {
            "passed": passed,
            "failed": failed,
            "total": len(results),
            "security_regressions": security_regressions,
            "memory_regressions": memory_regressions,
            "average_latency_ms": int(statistics.mean(latencies)) if latencies else 0,
            "slowest": slowest.eval_id if slowest else None,
            "slowest_latency_ms": slowest.latency_ms if slowest else 0,
            "warnings": warnings,
        },
        "memory_health": memory_health or {},
        "results": [asdict(result) for result in results],
    }


def write_reports(report: dict[str, Any], report_dir: Path = DEFAULT_REPORT_DIR) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "latest.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (report_dir / "latest.md").write_text(render_markdown_report(report), encoding="utf-8")


def render_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# F.R.I.D.A.Y Eval Report",
        "",
        f"Passed: {summary['passed']}/{summary['total']}",
        f"Failed: {summary['failed']}/{summary['total']}",
        f"Security regressions: {summary['security_regressions']}",
        f"Memory regressions: {summary['memory_regressions']}",
        f"Average latency: {summary['average_latency_ms']}ms",
        f"Slowest: {summary['slowest']} ({summary['slowest_latency_ms']}ms)",
        "",
        "| Status | Eval | Latency | Route | Failure |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for result in report["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        failure = result.get("failure_reason") or ""
        lines.append(
            f"| {status} | {result['eval_id']} | {result['latency_ms']}ms | "
            f"{result.get('actual_route') or ''} | {failure} |"
        )
    if summary["warnings"]:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    return "\n".join(lines) + "\n"


def print_console_report(report: dict[str, Any]) -> None:
    print("F.R.I.D.A.Y Eval Harness\n")
    for result in report["results"]:
        label = "PASS" if result["passed"] else "FAIL"
        print(f"[{label}] {result['eval_id']:<34} {result['latency_ms']}ms")
        if not result["passed"]:
            print(f"       {result['failure_reason']}")
    summary = report["summary"]
    print()
    print(f"Passed: {summary['passed']}/{summary['total']}")
    print(f"Failed: {summary['failed']}/{summary['total']}")
    print(f"Security regressions: {summary['security_regressions']}")
    print(f"Memory regressions: {summary['memory_regressions']}")
    print(f"Average latency: {summary['average_latency_ms']}ms")
    print(f"Slowest: {summary['slowest']} ({summary['slowest_latency_ms']}ms)")
    if summary["warnings"]:
        print("Warnings:")
        for warning in summary["warnings"]:
            print(f"- {warning}")


async def run_command(args: argparse.Namespace) -> int:
    report = await run_eval_suite(
        cases_path=Path(args.cases),
        report_dir=Path(args.report_dir),
        memory_db_path=Path(args.memory_db),
        workspace_root=Path(args.workspace_root),
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_console_report(report)
    return 0 if report["summary"]["failed"] == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run F.R.I.D.A.Y core runtime evals.")
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="Run the eval suite.")
    run_parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    run_parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    run_parser.add_argument("--memory-db", default=str(DEFAULT_EVAL_DB_PATH))
    run_parser.add_argument("--workspace-root", default=str(PROJECT_ROOT))
    run_parser.add_argument("--json", action="store_true", help="Print full JSON report to stdout.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "run":
        parser.print_help()
        return 2
    return asyncio.run(run_command(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
