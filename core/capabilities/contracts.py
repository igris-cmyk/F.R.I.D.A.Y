from typing import Any, Dict, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class CapabilityExecutionContext(BaseModel):
    """Metadata preserving trace lineage and execution scope."""
    trace_id: str
    source_intent: str
    workspace_root: str = Field(default=".")
    
class CapabilityInvocation(BaseModel):
    """Strict payload for invoking a capability."""
    capability_id: str
    input_payload: Dict[str, Any]
    context: CapabilityExecutionContext
    timeout_seconds: Optional[int] = None
    requires_confirmation: bool = False

class CapabilityResult(BaseModel):
    """Structured success output from a capability."""
    capability_id: str
    success: Literal[True] = True
    data: Dict[str, Any]
    execution_time_ms: float
    trace_id: str

class CapabilityFailure(BaseModel):
    """Structured error boundary from a capability."""
    capability_id: str
    success: Literal[False] = False
    error_code: str
    message: str
    stack_trace: Optional[str] = None
    trace_id: str
    recovery_hint: Optional[str] = None

class CapabilityRequiresApproval(BaseModel):
    """Structured state when a capability requires human approval."""
    capability_id: str
    status: Literal["REQUIRES_APPROVAL"] = "REQUIRES_APPROVAL"
    reason: str
    risk_level: str
    trace_id: str

class CapabilityPermissionRequest(BaseModel):
    """Event emitted when a capability requires human approval."""
    capability_id: str
    risk_level: str
    reason: str
    invocation: CapabilityInvocation
    trace_id: str
