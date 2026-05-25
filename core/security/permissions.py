from enum import Enum
from pydantic import BaseModel
from typing import Optional, List, Dict, Literal
from core.capabilities.contracts import CapabilityInvocation, CapabilityFailure

class RiskLevel(str, Enum):
    SAFE = "SAFE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class SecurityEvaluation(BaseModel):
    """Structured response for capability security evaluation."""
    status: Literal["ALLOW", "DENY", "REQUIRES_APPROVAL"]
    reason: str
    risk_level: RiskLevel

class SecurityPolicy:
    """Human-authoritative security gatekeeper for capability execution."""
    
    def __init__(self):
        # Explicit MVP Deny List for mutation tools
        self.blocked_capabilities = [
            "git.commit",
            "git.push",
            "shell.execute",
            "filesystem.write",
            "system.update"
        ]

    def evaluate_invocation(self, invocation: CapabilityInvocation, risk_level: RiskLevel, mutation_allowed: bool) -> SecurityEvaluation:
        """
        Evaluate if an invocation is allowed.
        """
        # 1. Block explicitly dangerous/mutating capabilities in this MVP
        if invocation.capability_id in self.blocked_capabilities:
            return SecurityEvaluation(
                status="DENY",
                reason=f"Capability '{invocation.capability_id}' is explicitly blocked.",
                risk_level=risk_level
            )
            
        # 2. Risk Level Rules
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            return SecurityEvaluation(
                status="DENY",
                reason=f"Risk level {risk_level.value} is auto-rejected.",
                risk_level=risk_level
            )
            
        if risk_level == RiskLevel.MEDIUM:
            if mutation_allowed:
                return SecurityEvaluation(
                    status="DENY",
                    reason="Mutating MEDIUM capabilities are DENIED for this MVP.",
                    risk_level=risk_level
                )
            else:
                return SecurityEvaluation(
                    status="REQUIRES_APPROVAL",
                    reason="MEDIUM risk actions require explicit user approval.",
                    risk_level=risk_level
                )
                
        # SAFE and LOW
        return SecurityEvaluation(
            status="ALLOW",
            reason=f"Risk level {risk_level.value} is automatically allowed.",
            risk_level=risk_level
        )
