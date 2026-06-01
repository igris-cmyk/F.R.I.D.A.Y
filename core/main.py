import asyncio
import contextlib
import os
import signal
import json
import logging
from nats.aio.client import Client as NATS
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
import time

from core.schemas.events import (
    CommandIntentEvent, 
    ExecutionResultEvent, 
    ExecutionResultPayload, 
    EventMetadata,
    TaskAcknowledgedEvent,
    TaskAcknowledgedPayload,
    ExecutionUpdateEvent,
    ExecutionUpdatePayload,
    CapabilityPermissionRequestEvent,
    CapabilityPermissionRequestPayload,
    CapabilityPermissionResponseEvent,
    CapabilityPermissionTimeoutEvent,
    CapabilityPermissionTimeoutPayload,
    CapabilityDeniedEvent,
    CapabilityDeniedPayload
)
from core.agents.router import intent_router
from core.agents.memory_agent import RetrievalPolicy, memory_agent
from core.memory.manager import memory_manager
from core.memory.pipeline import process_completed_trace
from core.supervision.heartbeat import HeartbeatMonitor, TraceRecord, SupervisionState
from core.api.health import broadcast_health_telemetry
from core.agents.planner import CognitivePlanner
from core.capabilities.registry import CapabilityRegistry
from core.security.permissions import SecurityPolicy
from core.security.approval import PendingApproval, matches_pending_approval
from core.capabilities.executor import CapabilityExecutor
from core.capabilities.contracts import CapabilityInvocation, CapabilityExecutionContext, CapabilityFailure
from core.config import (
    FRIDAY_RESEARCH_MAX_CHARS_PER_FILE,
    FRIDAY_RESEARCH_MAX_FILES,
    FRIDAY_RESEARCH_MAX_TOTAL_CHARS,
)
from core.research.context_builder import (
    ContextBudget,
    build_context_file,
    context_budget_used,
    context_has_truncation,
    select_ranked_files,
)
from core.research.ranker import rank_research_files

# Global Capability Engine
registry = CapabilityRegistry()
security_policy = SecurityPolicy()
executor = CapabilityExecutor(registry, security_policy)
planner = CognitivePlanner(registry=registry)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("friday.core")

# Global Active Trace Registry
ACTIVE_TRACES = {}
PENDING_APPROVALS: Dict[str, PendingApproval] = {}

WORKFLOW_MAX_FILES_TO_READ = FRIDAY_RESEARCH_MAX_FILES
WORKFLOW_MAX_CHARS_PER_FILE = FRIDAY_RESEARCH_MAX_CHARS_PER_FILE
WORKFLOW_MAX_TOTAL_CONTEXT_CHARS = FRIDAY_RESEARCH_MAX_TOTAL_CHARS
WORKFLOW_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "target",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
def create_workflow_context(trace_id: str, original_intent: str) -> Dict[str, Any]:
    return {
        "trace_id": trace_id,
        "original_intent": original_intent,
        "step_results": [],
        "last_result": None,
        "files_found": [],
        "files_read": [],
        "synthesis_inputs": [],
        "metadata": {},
    }


def is_generated_or_dependency_path(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    return any(part in WORKFLOW_EXCLUDED_DIRS for part in parts)


def select_files_for_synthesis(files: List[str], max_files: int = WORKFLOW_MAX_FILES_TO_READ) -> List[str]:
    budget = ContextBudget(max_files=max_files)
    ranked_files = rank_research_files("", files)
    return select_ranked_files(ranked_files, budget=budget)


def add_file_to_workflow_context(
    workflow_context: Dict[str, Any],
    path: str,
    content: str,
    size: int,
    truncated: bool,
) -> Optional[Dict[str, Any]]:
    context_file = build_context_file(
        path=path,
        content=content,
        size=size,
        truncated=truncated,
        used_chars=context_budget_used(workflow_context["files_read"]),
        budget=ContextBudget(),
    )
    if not context_file:
        return None
    file_record = {
        "path": context_file.path,
        "content": context_file.content,
        "size": context_file.size,
        "truncated": context_file.truncated,
    }
    workflow_context["files_read"].append(file_record)
    workflow_context["synthesis_inputs"] = workflow_context["files_read"]
    return file_record


def capture_workflow_result(
    workflow_context: Dict[str, Any],
    capability_id: str,
    data: Dict[str, Any],
) -> None:
    result_record = {"capability_id": capability_id, "data": data}
    workflow_context["step_results"].append(result_record)
    workflow_context["last_result"] = result_record

    if capability_id == "filesystem.search":
        files = data.get("files", [])
        workflow_context["files_found"].extend(
            file_path for file_path in files
            if file_path not in workflow_context["files_found"]
        )

    if capability_id == "filesystem.read" and data.get("content") is not None:
        path = data.get("path")
        if path:
            add_file_to_workflow_context(
                workflow_context=workflow_context,
                path=path,
                content=data.get("content", ""),
                size=data.get("size", 0),
                truncated=data.get("truncated", False),
            )


def build_synthesis_payload(
    workflow_context: Dict[str, Any],
    topic: str,
    step_input: Dict[str, Any],
) -> Dict[str, Any]:
    payload = dict(step_input)
    payload["topic"] = payload.get("topic") or topic
    payload["goal"] = payload.get("goal") or workflow_context["original_intent"]
    if workflow_context["files_read"]:
        payload["context"] = workflow_context["files_read"]
    payload["previous_results"] = workflow_context["step_results"]
    return payload


async def read_selected_files_for_workflow(
    nc: NATS,
    trace_id: str,
    executor: CapabilityExecutor,
    workflow_context: Dict[str, Any],
    capability_context: CapabilityExecutionContext,
    record: Optional[TraceRecord] = None,
) -> List[Dict[str, Any]]:
    ranked_files = rank_research_files(
        workflow_context["original_intent"],
        workflow_context["files_found"],
    )
    selected_files = select_ranked_files(ranked_files, budget=ContextBudget())
    workflow_context["metadata"]["ranked_files"] = [
        {"path": item.path, "score": item.score, "reasons": item.reasons}
        for item in ranked_files
    ]
    workflow_context["metadata"]["selected_files"] = selected_files

    if not selected_files:
        return []

    await publish_execution_update(
        nc,
        trace_id=trace_id,
        source_component="core.research",
        stage="planning",
        message=f"[RESEARCH] Ranked {len(ranked_files)} files for intent relevance.",
    )
    if ranked_files:
        top_file = ranked_files[0]
        await publish_execution_update(
            nc,
            trace_id=trace_id,
            source_component="core.research",
            stage="planning",
            message=f"[RESEARCH] Top file: {top_file.path} score={top_file.score}",
        )
    await publish_execution_update(
        nc,
        trace_id=trace_id,
        source_component="core.research",
        stage="planning",
        message=f"[RESEARCH] Selected {len(selected_files)} files for grounded synthesis.",
    )

    bound_reads = []
    for path in selected_files:
        await publish_execution_update(
            nc,
            trace_id=trace_id,
            source_component="core.executor",
            stage="capability_execution",
            message="[CAPABILITY] filesystem.read started: Reading selected files for grounded synthesis.",
        )
        if record:
            record.bump_heartbeat("workflow_read_binding")

        invocation = CapabilityInvocation(
            capability_id="filesystem.read",
            input_payload={"path": path},
            context=capability_context,
        )
        result = await executor.execute(invocation)
        if not getattr(result, "success", False):
            logger.warning("Workflow-bound filesystem.read failed for %s: %s", path, getattr(result, "message", ""))
            continue

        capture_workflow_result(workflow_context, "filesystem.read", result.data)
        bound_reads.append(result.data)
        await publish_execution_update(
            nc,
            trace_id=trace_id,
            source_component="core.executor",
            stage="capability_execution",
            message="[CAPABILITY] filesystem.read completed successfully.",
        )

    budget_used = context_budget_used(workflow_context["files_read"])
    truncation_note = " truncated=true" if context_has_truncation(workflow_context["files_read"]) else ""
    await publish_execution_update(
        nc,
        trace_id=trace_id,
        source_component="core.research",
        stage="planning",
        message=(
            f"[RESEARCH] Context budget: "
            f"{budget_used}/{WORKFLOW_MAX_TOTAL_CONTEXT_CHARS} chars.{truncation_note}"
        ),
    )

    return bound_reads


async def publish_execution_update(
    nc: NATS,
    trace_id: str,
    source_component: str,
    stage: str,
    message: str,
    progress_percentage: Optional[int] = None,
):
    event = ExecutionUpdateEvent(
        metadata=EventMetadata(trace_id=trace_id, source_component=source_component),
        payload=ExecutionUpdatePayload(
            stage=stage,
            message=message,
            progress_percentage=progress_percentage,
        ),
    )
    await nc.publish(f"friday.stream.{trace_id}", event.model_dump_json().encode())


async def run_planner_with_progress(
    nc: NATS,
    trace_id: str,
    planner: CognitivePlanner,
    intent: str,
    record: Optional[TraceRecord] = None,
    heartbeat_interval_seconds: float = 2.0,
):
    model_name = getattr(planner, "model", "unknown")
    timeout_seconds = getattr(planner, "timeout_seconds", "unknown")
    generate_high_confidence_plan = getattr(planner, "generate_high_confidence_plan", None)
    deterministic_plan = (
        generate_high_confidence_plan(intent)
        if callable(generate_high_confidence_plan)
        else None
    )
    if deterministic_plan:
        await publish_execution_update(
            nc,
            trace_id=trace_id,
            source_component="core.planner",
            stage="planning",
            message=(
                "[PLANNER] Using deterministic high-confidence plan. "
                f"source=deterministic model={model_name} timeout={timeout_seconds}"
            ),
        )
        if record:
            record.bump_heartbeat("planning_deterministic")
        return deterministic_plan

    await publish_execution_update(
        nc,
        trace_id=trace_id,
        source_component="core.planner",
        stage="planning",
        message=(
            "[PLANNER] Local model planning started... "
            f"model={model_name} timeout={timeout_seconds}"
        ),
    )
    if record:
        record.bump_heartbeat("planning_start")

    planner_task = asyncio.create_task(planner.generate_plan(intent))

    try:
        while True:
            try:
                plan = await asyncio.wait_for(asyncio.shield(planner_task), timeout=heartbeat_interval_seconds)
                break
            except asyncio.TimeoutError:
                await publish_execution_update(
                    nc,
                    trace_id=trace_id,
                    source_component="core.planner",
                    stage="planning",
                    message="[PLANNER] Waiting for local model...",
                )
                if record:
                    record.bump_heartbeat("planning_wait")
    except Exception:
        if not planner_task.done():
            planner_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await planner_task
        raise

    if plan.validation.fallback_used and plan.validation.fallback_reason == "timeout":
        await publish_execution_update(
            nc,
            trace_id=trace_id,
            source_component="core.planner",
            stage="planning",
            message="[PLANNER] Local planner timed out; using deterministic fallback.",
        )
        if record:
            record.bump_heartbeat("planning_timeout_fallback")

    return plan


async def publish_failure_result(
    nc: NATS,
    trace_id: str,
    error: Exception,
    intent: Optional[CommandIntentEvent] = None,
    execution_time_ms: int = 0,
    source_component: str = "core.orchestrator",
):
    if not trace_id:
        return

    result_metadata = EventMetadata(
        trace_id=trace_id,
        source_component=source_component,
        target_component=intent.metadata.source_component if intent else None,
        priority=intent.metadata.priority if intent else "normal",
    )
    result_event = ExecutionResultEvent(
        metadata=result_metadata,
        payload=ExecutionResultPayload(
            status="failure",
            output="[ORCHESTRATOR] Internal execution failure.",
            error=str(error),
            execution_time_ms=execution_time_ms,
        ),
    )
    await nc.publish(f"friday.stream.{trace_id}", result_event.model_dump_json().encode())


async def execute_memory_recall(
    nc: NATS,
    trace_id: str,
    query: str,
    environment: Dict[str, Any],
    agent=memory_agent,
) -> str:
    stream_subject = f"friday.stream.{trace_id}"
    await nc.publish(stream_subject, ExecutionUpdateEvent(
        metadata=EventMetadata(trace_id=trace_id, source_component="core.agent.memory"),
        payload=ExecutionUpdatePayload(
            stage="memory_retrieval",
            message="[MEMORY] Searching persistent memory...",
            progress_percentage=75,
        )
    ).model_dump_json().encode())

    recall_result = await agent.recall_context(
        query,
        RetrievalPolicy.PROJECT_RECALL,
        current_workspace=(environment or {}).get("working_directory") or (environment or {}).get("workspace_root"),
    )

    if not recall_result:
        await nc.publish(stream_subject, ExecutionUpdateEvent(
            metadata=EventMetadata(trace_id=trace_id, source_component="core.agent.memory"),
            payload=ExecutionUpdatePayload(
                stage="memory_retrieval",
                message="[MEMORY] No relevant continuity found.",
                progress_percentage=90,
            )
        ).model_dump_json().encode())
        return "No relevant continuity found."

    retrieved_count = recall_result.get("lineage", {}).get("candidate_count", 0)
    await nc.publish(stream_subject, ExecutionUpdateEvent(
        metadata=EventMetadata(trace_id=trace_id, source_component="core.agent.memory"),
        payload=ExecutionUpdatePayload(
            stage="memory_retrieval",
            message=f"[MEMORY] Retrieved {retrieved_count} relevant memories.",
            progress_percentage=90,
        )
    ).model_dump_json().encode())
    source_trace_ids = recall_result.get("lineage", {}).get("source_trace_ids", [])
    sources = ", ".join(source_trace_ids) if source_trace_ids else "none"
    return f"{recall_result['narrative']}\n\nSources: {sources}"


def log_memory_pipeline_task_result(task: asyncio.Task):
    try:
        result = task.result()
        logger.info("[MEMORY PIPELINE] Background task completed: %s", result)
    except asyncio.CancelledError:
        logger.warning("[MEMORY PIPELINE] Background task was cancelled.")
    except Exception as exc:
        logger.exception("[MEMORY PIPELINE] Background task failed: %s", exc)

async def supervision_loop(nc: NATS):
    """
    Background worker that scans ACTIVE_TRACES every 5 seconds.
    Enforces TTL rules and cleans up Zombie tasks.
    """
    while True:
        await asyncio.sleep(5)
        current_time = time.time()
        
        # We iterate over a copy of keys to avoid runtime mutation errors
        for trace_id in list(ACTIVE_TRACES.keys()):
            record = ACTIVE_TRACES.get(trace_id)
            if not record:
                continue
                
            new_state = HeartbeatMonitor.evaluate_state(record.last_heartbeat, current_time)
            
            if new_state != record.status:
                logger.warning(f"[SUPERVISION] Trace {trace_id} transition: {record.status.value.upper()} -> {new_state.value.upper()}")
                record.status = new_state
                
            if new_state == SupervisionState.FAILED:
                logger.error(f"[ZOMBIE CLEANUP] Force terminating stalled trace {trace_id}")
                
                # Cancel the actual asyncio.Task
                record.task.cancel()
                
                # Emit Failure Event to cleanup frontend
                fail_event = ExecutionResultEvent(
                    metadata=EventMetadata(trace_id=trace_id, source_component="core.orchestrator.supervision"),
                    payload=ExecutionResultPayload(
                        status="failure",
                        output="[ORCHESTRATOR] Task forcefully terminated due to heartbeat timeout (Zombie).",
                        execution_time_ms=int((current_time - record.started_at)*1000)
                    )
                )
                try:
                    await nc.publish(f"friday.stream.{trace_id}", fail_event.model_dump_json().encode())
                except Exception as e:
                    logger.error(f"Failed to publish Zombie cleanup event: {e}")
                    
                # Clean registry
                del ACTIVE_TRACES[trace_id]

async def main():
    nc = NATS()
    
    # Initialize Memory Connections
    await memory_manager.connect()
    
    # Wait for NATS to be available
    # The default NATS port is 4222
    nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
    
    logger.info(f"Connecting to NATS at {nats_url}...")
    try:
        await nc.connect(nats_url)
        logger.info("Connected to NATS event bus.")
    except Exception as e:
        logger.error(f"Failed to connect to NATS: {e}")
        return

    # Start Health Telemetry and Supervision loops
    asyncio.create_task(broadcast_health_telemetry(nc, ACTIVE_TRACES))
    asyncio.create_task(supervision_loop(nc))

    # Actual intent execution logic (Supervised)
    async def execute_supervised_intent(msg, intent, trace_id, stream_subject, reply):
        record = ACTIVE_TRACES.get(trace_id)
        start_time = time.time()
        
        try:
            # Enforce schema on inbound intent
            data_dict = json.loads(msg.data.decode())
            intent = CommandIntentEvent(**data_dict)
            trace_id = intent.metadata.trace_id
            record = ACTIVE_TRACES.get(trace_id)
            
            # Observability Hook: Structured Execution Tracing
            logger.info(f"[{trace_id}] Received intent on {msg.subject}: {intent.payload.raw_command}")
            
            # Trace-Bound Streaming Channel
            stream_subject = f"friday.stream.{trace_id}"
            
            # 1. Acknowledge the intent instantly
            ack_event = TaskAcknowledgedEvent(
                metadata=EventMetadata(trace_id=trace_id, source_component="core.orchestrator"),
                payload=TaskAcknowledgedPayload(intent_type="unknown", message="Routing intent...")
            )
            await nc.publish(stream_subject, ack_event.model_dump_json().encode())
            
            # 2. Publish Routing Update
            await nc.publish(stream_subject, ExecutionUpdateEvent(
                metadata=EventMetadata(trace_id=trace_id, source_component="core.orchestrator.router"),
                payload=ExecutionUpdatePayload(stage="routing", message="Analyzing semantics via LangGraph...", progress_percentage=10)
            ).model_dump_json().encode())
            if record: record.bump_heartbeat("routing")
            
            # Run intent through LangGraph router
            router_state = {
                "raw_command": intent.payload.raw_command, 
                "environment": intent.payload.environment.model_dump() if intent.payload.environment else {},
                "intent": "", 
                "parameters": {}, 
                "error": "", 
                "routing_metadata": {}
            }
            final_state = await intent_router.ainvoke(router_state)
            intent_type = final_state.get("intent", "conversation")
            routing_metadata = final_state.get("routing_metadata", {})
            
            # 3. Publish Execution Update
            await nc.publish(stream_subject, ExecutionUpdateEvent(
                metadata=EventMetadata(trace_id=trace_id, source_component=f"core.agent.{intent_type}"),
                payload=ExecutionUpdatePayload(stage="executing", message=f"Dispatched to {intent_type.upper()} agent.", progress_percentage=50)
            ).model_dump_json().encode())
            if record: record.bump_heartbeat(f"agent_{intent_type}")
            
            success = False
            output = ""
            exec_ms = 0
            trace_memory_metadata: Dict[str, Any] = {"routing": routing_metadata}
            
            if intent_type == "memory":
                query = final_state["parameters"].get("query") or intent.payload.raw_command
                environment = intent.payload.environment.model_dump() if intent.payload.environment else {}
                output = await execute_memory_recall(nc, trace_id, query, environment)
                success = True
            elif intent_type == "conversation":
                output = f"[Conversational Response] Acknowledged: {final_state['parameters'].get('message')}"
                success = True
            else:
                # 4. Invoke Cognitive Planner
                await nc.publish(stream_subject, ExecutionUpdateEvent(
                    metadata=EventMetadata(trace_id=trace_id, source_component="core.planner"),
                    payload=ExecutionUpdatePayload(stage="planning", message="[PLANNER] Analyzing intent...")
                ).model_dump_json().encode())
                if record: record.bump_heartbeat("planning")
                
                plan_start = time.time()
                plan = None
                success = False
                output = ""

                await nc.publish(stream_subject, ExecutionUpdateEvent(
                    metadata=EventMetadata(trace_id=trace_id, source_component="core.planner"),
                    payload=ExecutionUpdatePayload(stage="planning", message="[PLANNER] Constructing execution plan...")
                ).model_dump_json().encode())

                try:
                    plan = await run_planner_with_progress(
                        nc=nc,
                        trace_id=trace_id,
                        planner=planner,
                        intent=intent.payload.raw_command,
                        record=record,
                    )
                except ValueError as e:
                    await nc.publish(stream_subject, ExecutionUpdateEvent(
                        metadata=EventMetadata(trace_id=trace_id, source_component="core.planner"),
                        payload=ExecutionUpdatePayload(stage="planning", message=f"[PLANNER] Invalid plan: {e}")
                    ).model_dump_json().encode())
                    output = f"Plan generation failed: {e}"
                    exec_ms = int((time.time() - plan_start) * 1000)
                
                if plan:
                    await nc.publish(stream_subject, ExecutionUpdateEvent(
                        metadata=EventMetadata(trace_id=trace_id, source_component="core.planner"),
                        payload=ExecutionUpdatePayload(
                            stage="planning",
                            message=(
                                f"[PLANNER] Plan validated... "
                                f"steps={len(plan.steps)} risk={plan.estimated_risk} "
                                f"model={planner.model} timeout={planner.timeout_seconds} "
                                f"source={plan.validation.source} fallback={plan.validation.fallback_used} "
                                f"fallback_reason={plan.validation.fallback_reason or 'none'}"
                            )
                        )
                    ).model_dump_json().encode())
                    
                    # 5. Execute Plan Sequentially
                    results = []
                    success = True
                    workflow_context = create_workflow_context(
                        trace_id=trace_id,
                        original_intent=intent.payload.raw_command,
                    )
                    
                    for step in plan.steps:
                        step_input = dict(step.input)
                        capability_context = CapabilityExecutionContext(
                            trace_id=trace_id,
                            source_intent=intent.payload.raw_command,
                            workspace_root=intent.payload.working_directory or "."
                        )

                        if step.capability_id == "filesystem.read" and not step_input.get("path"):
                            bound_reads = await read_selected_files_for_workflow(
                                nc=nc,
                                trace_id=trace_id,
                                executor=executor,
                                workflow_context=workflow_context,
                                capability_context=capability_context,
                                record=record,
                            )
                            results.append(f"SUCCESS filesystem.read: bound {len(bound_reads)} files for synthesis")
                            continue

                        if step.capability_id == "research.synthesize":
                            if workflow_context["files_found"] and not workflow_context["files_read"]:
                                await publish_execution_update(
                                    nc,
                                    trace_id=trace_id,
                                    source_component="core.workflow",
                                    stage="planning",
                                    message="[WORKFLOW] Binding search results into synthesis context...",
                                )
                                await read_selected_files_for_workflow(
                                    nc=nc,
                                    trace_id=trace_id,
                                    executor=executor,
                                    workflow_context=workflow_context,
                                    capability_context=capability_context,
                                    record=record,
                                )

                            step_input = build_synthesis_payload(
                                workflow_context=workflow_context,
                                topic=intent.payload.raw_command,
                                step_input=step_input,
                            )
                            start_message = (
                                f"[CAPABILITY] research.synthesize started: "
                                f"Grounded synthesis from {len(workflow_context['files_read'])} files."
                            )
                        else:
                            start_message = f"[CAPABILITY] {step.capability_id} started: {step.reason}"

                        await nc.publish(stream_subject, ExecutionUpdateEvent(
                            metadata=EventMetadata(trace_id=trace_id, source_component="core.executor"),
                            payload=ExecutionUpdatePayload(stage="capability_execution", message=start_message)
                        ).model_dump_json().encode())
                        if record: record.bump_heartbeat(f"capability:{step.capability_id}")
                        
                        invocation = CapabilityInvocation(
                            capability_id=step.capability_id,
                            input_payload=step_input,
                            context=capability_context,
                            requires_confirmation=plan.requires_confirmation
                        )
                        
                        res = await executor.execute(invocation)
                        
                        if getattr(res, "status", None) == "REQUIRES_APPROVAL":
                            await nc.publish(stream_subject, ExecutionUpdateEvent(
                                metadata=EventMetadata(trace_id=trace_id, source_component="core.security"),
                                payload=ExecutionUpdatePayload(stage="security_check", message=f"[SECURITY] Approval required")
                            ).model_dump_json().encode())
                            
                            req = CapabilityPermissionRequestEvent(
                                metadata=EventMetadata(trace_id=trace_id, source_component="core.orchestrator"),
                                payload=CapabilityPermissionRequestPayload(
                                    trace_id=trace_id,
                                    capability_id=step.capability_id,
                                    human_name=step.capability_id,
                                    risk_level=res.risk_level,
                                    reason=res.reason,
                                    requested_action_summary=step.reason,
                                    input_preview=json.dumps(step_input),
                                    timeout_seconds=30,
                                    expires_at=(datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
                                )
                            )
                            await nc.publish(f"friday.permission.request.{trace_id}", req.model_dump_json().encode())
                            if record: record.bump_heartbeat("awaiting_approval")
                            
                            future = asyncio.get_running_loop().create_future()
                            PENDING_APPROVALS[trace_id] = PendingApproval(
                                future=future,
                                capability_id=step.capability_id,
                                expires_at=time.time() + 30.0,
                            )
                            
                            try:
                                response_event = await asyncio.wait_for(future, timeout=30.0)
                                if response_event.payload.approved:
                                    res = await executor.execute(invocation, human_approved=True)
                                else:
                                    res = CapabilityFailure(
                                        capability_id=step.capability_id,
                                        error_code="DENIED_BY_USER",
                                        message="Capability denied by user",
                                        trace_id=trace_id
                                    )
                            except asyncio.TimeoutError:
                                await nc.publish(stream_subject, ExecutionUpdateEvent(
                                    metadata=EventMetadata(trace_id=trace_id, source_component="core.orchestrator"),
                                    payload=ExecutionUpdatePayload(stage="security_check", message=f"[SECURITY] Approval timed out. Capability denied.")
                                ).model_dump_json().encode())
                                
                                res = CapabilityFailure(
                                    capability_id=step.capability_id,
                                    error_code="APPROVAL_TIMEOUT",
                                    message="Approval timed out",
                                    trace_id=trace_id
                                )
                            finally:
                                if trace_id in PENDING_APPROVALS:
                                    del PENDING_APPROVALS[trace_id]
                        
                        if not getattr(res, "success", False):
                            await nc.publish(stream_subject, ExecutionUpdateEvent(
                                metadata=EventMetadata(trace_id=trace_id, source_component="core.security"),
                                payload=ExecutionUpdatePayload(stage="security_check", message=f"[SECURITY] Blocked/Failed capability: {step.capability_id}. Reason: {getattr(res, 'message', 'Execution error')}")
                            ).model_dump_json().encode())
                            results.append(f"FAILED {step.capability_id}: {getattr(res, 'message', '')}")
                            success = False
                            break
                        else:
                            capture_workflow_result(workflow_context, step.capability_id, getattr(res, "data", {}))
                            if step.capability_id == "filesystem.search":
                                await publish_execution_update(
                                    nc,
                                    trace_id=trace_id,
                                    source_component="core.workflow",
                                    stage="planning",
                                    message="[WORKFLOW] Captured filesystem.search output.",
                                )
                            await nc.publish(stream_subject, ExecutionUpdateEvent(
                                metadata=EventMetadata(trace_id=trace_id, source_component="core.executor"),
                                payload=ExecutionUpdatePayload(stage="capability_execution", message=f"[CAPABILITY] {step.capability_id} completed successfully.")
                            ).model_dump_json().encode())
                            results.append(f"SUCCESS {step.capability_id}: {getattr(res, 'data', {})}")
                    
                    output = "\n".join(results)
                    exec_ms = int((time.time() - plan_start) * 1000)
                    trace_memory_metadata.update({
                        "workflow": {
                            "selected_files": workflow_context["metadata"].get("selected_files", []),
                            "ranked_files": workflow_context["metadata"].get("ranked_files", []),
                            "files_read": [
                                {
                                    "path": item.get("path"),
                                    "size": item.get("size"),
                                    "truncated": item.get("truncated", False),
                                }
                                for item in workflow_context.get("files_read", [])
                            ],
                            "step_results": [
                                {"capability_id": item.get("capability_id")}
                                for item in workflow_context.get("step_results", [])
                            ],
                        },
                        "selected_files": workflow_context["metadata"].get("selected_files", []),
                        "files_read": [
                            {
                                "path": item.get("path"),
                                "size": item.get("size"),
                                "truncated": item.get("truncated", False),
                            }
                            for item in workflow_context.get("files_read", [])
                        ],
                        "capabilities_used": [
                            item.get("capability_id")
                            for item in workflow_context.get("step_results", [])
                            if item.get("capability_id")
                        ],
                    })
            
            # Memory Hook: Trigger asynchronous cognitive compression pipeline
            # This is fire-and-forget; it must NOT block the orchestrator or routing loops.
            memory_task = asyncio.create_task(
                process_completed_trace(
                    trace_id=trace_id,
                    intent=intent_type,
                    command=intent.payload.raw_command,
                    result=output,
                    error_state=not success,
                    environment=intent.payload.environment.model_dump() if intent.payload.environment else {},
                    metadata={**trace_memory_metadata, "execution_ms": exec_ms}
                )
            )
            memory_task.add_done_callback(log_memory_pipeline_task_result)
            
            # 4. Final Result Event
            result_metadata = EventMetadata(
                trace_id=trace_id,
                source_component="core.orchestrator",
                target_component=intent.metadata.source_component,
                priority=intent.metadata.priority
            )
            
            result_event = ExecutionResultEvent(
                metadata=result_metadata,
                payload=ExecutionResultPayload(
                    status="success" if success else "failure",
                    output=output,
                    execution_time_ms=exec_ms
                )
            )
            
            await nc.publish(stream_subject, result_event.model_dump_json().encode())
            logger.info(f"[{trace_id}] Published final result to {stream_subject} (success={success}, ms={exec_ms})")
            
            # Legacy reply acknowledgment for callers still using Request/Reply
            if reply:
                await nc.publish(reply, b"ack")

        except asyncio.CancelledError:
            # Re-raise so the task actually stops if killed by the supervision loop
            raise
        except Exception as e:
            logger.exception("Execution logic failed")
            execution_time_ms = int((time.time() - start_time) * 1000)
            try:
                await publish_failure_result(nc, trace_id, e, intent=intent, execution_time_ms=execution_time_ms)
            except Exception:
                logger.exception("Failed to publish structured execution failure")
        finally:
            # Auto-purge completed traces
            if trace_id in ACTIVE_TRACES:
                del ACTIVE_TRACES[trace_id]
                
    # Entrypoint message handler
    async def message_handler(msg):
        try:
            data_dict = json.loads(msg.data.decode())
            intent = CommandIntentEvent(**data_dict)
            trace_id = intent.metadata.trace_id
            stream_subject = f"friday.stream.{trace_id}"
            
            # Spawn supervised task
            task = asyncio.create_task(execute_supervised_intent(msg, intent, trace_id, stream_subject, msg.reply))
            
            # Register in ACTIVE_TRACES
            ACTIVE_TRACES[trace_id] = TraceRecord(trace_id=trace_id, task=task, agent="router")
            
        except Exception as e:
            logger.error(f"Event schema validation failed on inbound message: {e}")

    # Subscribe to MVP command intent subject
    await nc.subscribe("friday.intent.command", cb=message_handler)
    logger.info("Listening for command intents on 'friday.intent.command'")
    
    async def permission_response_handler(msg):
        try:
            data_dict = json.loads(msg.data.decode())
            response = CapabilityPermissionResponseEvent(**data_dict)
            trace_id = response.payload.trace_id
            
            pending = PENDING_APPROVALS.get(trace_id)
            if not matches_pending_approval(pending, trace_id, response.payload.capability_id):
                return

            pending.future.set_result(response)
        except Exception as e:
            logger.error(f"Failed to process permission response: {e}")

    await nc.subscribe("friday.permission.response.*", cb=permission_response_handler)
    logger.info("Listening for UI permission responses on 'friday.permission.response.*'")

    # Keep the process running
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Shutdown signal received.")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass # Windows compatibility if needed later

    await stop_event.wait()

    # Clean up
    await nc.drain()
    logger.info("Disconnected from NATS. Shutdown complete.")

if __name__ == '__main__':
    asyncio.run(main())
