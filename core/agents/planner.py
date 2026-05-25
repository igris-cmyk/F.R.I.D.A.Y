from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json

class CapabilityStep(BaseModel):
    capability_id: str
    reason: str
    input: Dict[str, Any]

class CognitivePlan(BaseModel):
    steps: List[CapabilityStep]
    estimated_risk: str
    requires_confirmation: bool

class CognitivePlanner:
    """Decomposes intent into a structured capability plan."""
    
    def __init__(self):
        pass
        
    async def generate_plan(self, intent: str, context: Optional[str] = None) -> CognitivePlan:
        """
        Invoke the local LLM to plan the execution.
        For this MVP, we simulate the LLM extraction logic to prove the architecture.
        """
        
        # Simulated LLM Parsing based on intent string
        normalized_intent = intent.lower().strip()
        tokens = normalized_intent.split()
        destructive_phrases = [
            "delete",
            "remove",
            "wipe",
            "rm -rf",
        ]
        if any(phrase in normalized_intent for phrase in destructive_phrases) or "rm" in tokens:
            return CognitivePlan(
                steps=[
                    CapabilityStep(
                        capability_id="shell.execute",
                        reason="Destructive filesystem deletion was requested.",
                        input={"command": "rm -rf target"}
                    )
                ],
                estimated_risk="CRITICAL",
                requires_confirmation=True
            )
            
        elif "git status" in normalized_intent:
            return CognitivePlan(
                steps=[
                    CapabilityStep(
                        capability_id="git.status",
                        reason="Inspect repository status as requested.",
                        input={"directory": "."}
                    )
                ],
                estimated_risk="LOW",
                requires_confirmation=False
            )

        elif normalized_intent.startswith("find ") and "python" in normalized_intent and "file" in normalized_intent:
            return CognitivePlan(
                steps=[
                    CapabilityStep(
                        capability_id="filesystem.search",
                        reason="Find Python files in the current project.",
                        input={"pattern": "*.py", "root": "."}
                    )
                ],
                estimated_risk="SAFE",
                requires_confirmation=False
            )

        elif normalized_intent.startswith("read ") and "package.json" in normalized_intent:
            return CognitivePlan(
                steps=[
                    CapabilityStep(
                        capability_id="filesystem.read",
                        reason="Read the requested package manifest.",
                        input={"path": "package.json"}
                    )
                ],
                estimated_risk="SAFE",
                requires_confirmation=False
            )

        elif normalized_intent in {"system monitor", "monitor system"}:
            return CognitivePlan(
                steps=[
                    CapabilityStep(
                        capability_id="system.monitor",
                        reason="Inspect system status.",
                        input={}
                    )
                ],
                estimated_risk="LOW",
                requires_confirmation=False
            )

        elif "summarize" in normalized_intent and "python" in normalized_intent:
            return CognitivePlan(
                steps=[
                    CapabilityStep(
                        capability_id="filesystem.search",
                        reason="Find Python files in the current project.",
                        input={"pattern": "*.py", "root": "."}
                    ),
                    CapabilityStep(
                        capability_id="research.synthesize",
                        reason="Summarize discovered files.",
                        input={"topic": "Architecture Summary", "context": ""}
                    )
                ],
                estimated_risk="LOW",
                requires_confirmation=False
            )
            
        elif "malformed" in intent.lower():
            # Force a schema error by returning garbage that fails Pydantic validation
            # In a real scenario, the LLM output might fail json parsing
            raise ValueError("LLM returned invalid JSON structure.")
            
        # Default fallback
        return CognitivePlan(
            steps=[
                CapabilityStep(
                    capability_id="system.monitor",
                    reason="Check system status.",
                    input={}
                )
            ],
            estimated_risk="LOW",
            requires_confirmation=False
        )
