import uuid
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, Literal

class EventMetadata(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_component: str
    target_component: Optional[str] = None
    priority: Literal["low", "normal", "high", "critical"] = "normal"

class BaseEvent(BaseModel):
    metadata: EventMetadata
    payload: Dict[str, Any]

class EnvironmentalContext(BaseModel):
    active_app: Optional[str] = None
    window_title: Optional[str] = None
    selected_text: Optional[str] = None
    ingestion_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class CommandIntentPayload(BaseModel):
    raw_command: str
    environment: Optional[EnvironmentalContext] = None
    working_directory: Optional[str] = None

class CommandIntentEvent(BaseEvent):
    payload: CommandIntentPayload

class ExecutionResultPayload(BaseModel):
    status: Literal["success", "failure", "pending_approval"]
    output: str
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None
    
class ExecutionResultEvent(BaseEvent):
    payload: ExecutionResultPayload

# --- Supervision & Telemetry Events ---

class HeartbeatPayload(BaseModel):
    status: Literal["running", "stalling", "recovering", "failed", "completed"]
    stage: str
    active_agent: str
    time_since_last_update_ms: int

class HeartbeatEvent(BaseModel):
    metadata: EventMetadata
    payload: HeartbeatPayload

class TraceStalledPayload(BaseModel):
    reason: str
    silence_duration_ms: int

class TraceStalledEvent(BaseModel):
    metadata: EventMetadata
    payload: TraceStalledPayload

class ExecutionFailurePayload(BaseModel):
    reason: str
    fatal: bool
    cleanup_successful: bool

class ExecutionFailureEvent(BaseModel):
    metadata: EventMetadata
    payload: ExecutionFailurePayload

class CoreHealthPayload(BaseModel):
    status: Literal["healthy", "degraded", "recovering"]
    active_traces_count: int
    stalled_traces_count: int
    memory_queue_depth: int
    ollama_available: bool

class CoreHealthEvent(BaseModel):
    metadata: EventMetadata
    payload: CoreHealthPayload

# --- Streaming Execution Lifecycle Events ---

class TaskAcknowledgedPayload(BaseModel):
    intent_type: str
    message: str = "Task acknowledged and routed."

class TaskAcknowledgedEvent(BaseEvent):
    payload: TaskAcknowledgedPayload

class ExecutionUpdatePayload(BaseModel):
    stage: Literal["routing", "executing", "memory_retrieval", "agent_transition", "synthesizing", "planning", "capability_execution", "security_check"]
    message: str
    progress_percentage: Optional[int] = None

class ExecutionUpdateEvent(BaseEvent):
    payload: ExecutionUpdatePayload

# --- Permission & Approval Events ---

class CapabilityPermissionRequestPayload(BaseModel):
    trace_id: str
    capability_id: str
    human_name: str
    risk_level: str
    reason: str
    requested_action_summary: str
    input_preview: str
    timeout_seconds: int
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: str

class CapabilityPermissionRequestEvent(BaseEvent):
    payload: CapabilityPermissionRequestPayload

class CapabilityPermissionResponsePayload(BaseModel):
    trace_id: str
    capability_id: str
    approved: bool
    user_decision: Literal["approved", "denied"]
    response_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_component: str

class CapabilityPermissionResponseEvent(BaseEvent):
    payload: CapabilityPermissionResponsePayload

class CapabilityPermissionTimeoutPayload(BaseModel):
    trace_id: str
    capability_id: str
    timeout_seconds: int

class CapabilityPermissionTimeoutEvent(BaseEvent):
    payload: CapabilityPermissionTimeoutPayload

class CapabilityDeniedPayload(BaseModel):
    trace_id: str
    capability_id: str
    reason: str

class CapabilityDeniedEvent(BaseEvent):
    payload: CapabilityDeniedPayload
