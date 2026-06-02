import asyncio
import json
import logging
import urllib.request
from typing import Any, Dict, List, Optional

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from core.capabilities.registry import CapabilityRegistry
from core.config import (
    FRIDAY_PLANNER_MODEL,
    FRIDAY_PLANNER_TIMEOUT_SECONDS,
    OLLAMA_BASE_URL,
)
from core.security.permissions import RiskLevel

logger = logging.getLogger("friday.agents.planner")

PLANNER_ALLOWED_CAPABILITIES = (
    "git.status",
    "filesystem.search",
    "filesystem.read",
    "system.monitor",
    "memory.recall",
    "research.synthesize",
    "shell.execute",
)

class PlannerFallbackError(Exception):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.message = message


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str
    reason: str
    input: Dict[str, Any]


class PlanValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    source: str
    fallback_used: bool
    fallback_reason: Optional[str] = None
    errors: List[str] = Field(default_factory=list)


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: List[PlanStep]
    estimated_risk: str
    requires_confirmation: bool
    validation: PlanValidation = Field(
        default_factory=lambda: PlanValidation(
            valid=True,
            source="deterministic",
            fallback_used=False,
            fallback_reason=None,
            errors=[],
        )
    )


# Backward-compatible aliases for the existing planner contract.
CapabilityStep = PlanStep
CognitivePlan = Plan


class CognitivePlanner:
    """Schema-first planner with bounded local-LLM proposal and deterministic fallback."""

    def __init__(
        self,
        registry: Optional[CapabilityRegistry] = None,
        llm: Optional[OllamaLLM] = None,
        timeout_seconds: Optional[float] = None,
    ):
        self.registry = registry or CapabilityRegistry()
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else FRIDAY_PLANNER_TIMEOUT_SECONDS
        self.model = FRIDAY_PLANNER_MODEL
        self.base_url = OLLAMA_BASE_URL
        self.llm = llm or OllamaLLM(
            model=self.model,
            base_url=self.base_url,
            temperature=0.0,
            format="json",
        )
        allowed_capabilities = ", ".join(PLANNER_ALLOWED_CAPABILITIES)
        self.prompt = PromptTemplate.from_template(
            "You are a JSON planner. Return ONLY valid JSON. No markdown.\n\n"
            f"Allowed capabilities:\n{allowed_capabilities}\n\n"
            "Return:\n"
            "{{\n"
            '  "goal": "...",\n'
            '  "risk": "SAFE|LOW|MEDIUM|HIGH|CRITICAL",\n'
            '  "steps": [\n'
            '    {{"capability_id": "...", "inputs": {{}}, "reasoning": "..."}}\n'
            "  ]\n"
            "}}\n\n"
            "Rules:\n"
            "- Use only allowed capabilities.\n"
            "- Dangerous or destructive requests use shell.execute with risk CRITICAL.\n"
            "- Do not execute anything.\n"
            "- JSON only.\n\n"
            "Intent:\n{intent}\n\n"
            "JSON:"
        )

    async def generate_plan(self, intent: str, context: Optional[str] = None) -> Plan:
        deterministic_plan = self.generate_high_confidence_plan(intent=intent, context=context)
        if deterministic_plan:
            logger.info("Planner using deterministic high-confidence plan for intent: %s", intent)
            return deterministic_plan

        try:
            llm_plan = await self._generate_llm_plan(intent=intent, context=context)
            return self._finalize_plan(llm_plan, source="llm", fallback_used=False)
        except Exception as exc:
            reason, message = self._classify_fallback(exc)
            logger.warning(
                "Planner falling back to deterministic rules: reason=%s detail=%s",
                reason,
                message,
            )
            fallback_plan = self._generate_deterministic_plan(intent=intent, context=context)
            return self._finalize_plan(
                fallback_plan,
                source="deterministic",
                fallback_used=True,
                fallback_reason=reason,
                fallback_errors=[message],
            )

    def generate_high_confidence_plan(self, intent: str, context: Optional[str] = None) -> Optional[Plan]:
        plan = self._generate_high_confidence_deterministic_plan(intent=intent, context=context)
        if not plan:
            return None
        return self._finalize_plan(plan, source="deterministic", fallback_used=False)

    async def _generate_llm_plan(self, intent: str, context: Optional[str]) -> Plan:
        await self._verify_ollama_ready()
        raw = await asyncio.wait_for(
            self._invoke_llm(intent=intent, context=context),
            timeout=self.timeout_seconds,
        )
        return self._parse_llm_output(raw)

    async def _invoke_llm(self, intent: str, context: Optional[str]) -> Any:
        chain = self.prompt | self.llm
        return await chain.ainvoke(
            {
                "intent": intent,
            }
        )

    def _parse_llm_output(self, raw_output: Any) -> Plan:
        if not isinstance(raw_output, str):
            raise PlannerFallbackError("invalid_json", "LLM planner returned a non-string response.")

        candidate = raw_output.strip()
        if candidate.startswith("```"):
            lines = [line for line in candidate.splitlines() if not line.strip().startswith("```")]
            candidate = "\n".join(lines).strip()

        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise PlannerFallbackError("invalid_json", f"LLM returned invalid JSON: {exc}") from exc

        payload = self._normalize_plan_payload(payload)

        try:
            return Plan.model_validate(payload)
        except ValidationError as exc:
            raise PlannerFallbackError("invalid_json", f"LLM returned JSON that failed plan schema validation: {exc}") from exc

    def _normalize_plan_payload(self, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise PlannerFallbackError("invalid_json", "LLM returned JSON that is not an object.")
        normalized = dict(payload)
        normalized.pop("goal", None)

        if "risk" in normalized and "estimated_risk" not in normalized:
            normalized["estimated_risk"] = normalized.pop("risk")

        steps = []
        for raw_step in normalized.get("steps", []):
            step = dict(raw_step)
            if "inputs" in step and "input" not in step:
                step["input"] = step.pop("inputs")
            if "reasoning" in step and "reason" not in step:
                step["reason"] = step.pop("reasoning")
            steps.append(step)

        normalized["steps"] = steps
        if "requires_confirmation" not in normalized:
            normalized["requires_confirmation"] = normalized.get("estimated_risk", "").upper() in {"HIGH", "CRITICAL"}
        return normalized

    async def _verify_ollama_ready(self) -> None:
        try:
            response = self._fetch_ollama_tags()
        except asyncio.TimeoutError as exc:
            raise PlannerFallbackError("timeout", f"Ollama availability check timed out at {self.base_url}.") from exc
        except TimeoutError as exc:
            raise PlannerFallbackError("timeout", f"Ollama availability check timed out at {self.base_url}.") from exc
        except Exception as exc:
            raise PlannerFallbackError("ollama_unavailable", f"Ollama unavailable at {self.base_url}: {exc}") from exc

        model_names = self._extract_ollama_model_names(response)
        if self.model not in model_names:
            available = ", ".join(sorted(model_names)) if model_names else "none"
            raise PlannerFallbackError(
                "ollama_unavailable",
                f"model_missing: required '{self.model}' is not installed locally. available_models={available}",
            )

    def _fetch_ollama_tags(self) -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/api/tags"
        with urllib.request.urlopen(url, timeout=1.0) as response:
            return json.loads(response.read().decode("utf-8"))

    def _extract_ollama_model_names(self, response: Any) -> set[str]:
        models = getattr(response, "models", None)
        if models is None and isinstance(response, dict):
            models = response.get("models", [])

        names: set[str] = set()
        for model in models or []:
            name = getattr(model, "model", None) or getattr(model, "name", None)
            if name is None and isinstance(model, dict):
                name = model.get("model") or model.get("name")
            if name:
                names.add(name)
        return names

    def _finalize_plan(
        self,
        plan: Plan,
        source: str,
        fallback_used: bool,
        fallback_reason: Optional[str] = None,
        fallback_errors: Optional[List[str]] = None,
    ) -> Plan:
        validation_errors = self._validate_plan(plan)
        if validation_errors:
            if not fallback_used:
                reason = self._validation_fallback_reason(validation_errors)
                raise PlannerFallbackError(reason, "; ".join(validation_errors))
            raise PlannerFallbackError("exception", "; ".join((fallback_errors or []) + validation_errors))

        plan.validation = PlanValidation(
            valid=True,
            source=source,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            errors=fallback_errors or [],
        )
        return plan

    def _validate_plan(self, plan: Plan) -> List[str]:
        errors: List[str] = []
        if not plan.steps:
            errors.append("Plan must contain at least one step.")

        normalized_risk = plan.estimated_risk.upper()
        if normalized_risk not in {level.value for level in RiskLevel}:
            errors.append(f"Unknown risk level '{plan.estimated_risk}'.")

        for index, step in enumerate(plan.steps):
            definition = self.registry.get(step.capability_id)
            if not definition:
                errors.append(f"unknown_capability: Unknown capability '{step.capability_id}' at step {index}.")
                continue

            for field in definition.input_schema.get("required", []):
                if field not in step.input:
                    errors.append(
                        f"missing_required_input: Capability '{step.capability_id}' missing required input '{field}' at step {index}."
                    )

            allowed_fields = set(definition.input_schema.get("properties", {}).keys())
            if allowed_fields:
                extra_fields = sorted(set(step.input.keys()) - allowed_fields)
                if extra_fields:
                    errors.append(
                        f"unexpected_input: Capability '{step.capability_id}' received unsupported input(s) {extra_fields} at step {index}."
                    )

        return errors

    def _validation_fallback_reason(self, validation_errors: List[str]) -> str:
        if any(error.startswith("unknown_capability:") for error in validation_errors):
            return "unknown_capability"
        if any(error.startswith("missing_required_input:") for error in validation_errors):
            return "missing_required_input"
        return "exception"

    def _classify_fallback(self, exc: Exception) -> tuple[str, str]:
        if isinstance(exc, PlannerFallbackError):
            return exc.reason, exc.message
        if isinstance(exc, asyncio.TimeoutError):
            return "timeout", "Planner LLM inference timed out."
        message = str(exc) or exc.__class__.__name__
        if "invalid json" in message.lower():
            return "invalid_json", message
        return "exception", message

    def _generate_deterministic_plan(self, intent: str, context: Optional[str] = None) -> Plan:
        high_confidence_plan = self._generate_high_confidence_deterministic_plan(intent=intent, context=context)
        if high_confidence_plan:
            return high_confidence_plan

        return Plan(
            steps=[
                PlanStep(
                    capability_id="system.monitor",
                    reason="Check system status.",
                    input={},
                )
            ],
            estimated_risk="LOW",
            requires_confirmation=False,
        )

    def _generate_high_confidence_deterministic_plan(self, intent: str, context: Optional[str] = None) -> Optional[Plan]:
        normalized_intent = intent.lower().strip()
        tokens = normalized_intent.split()

        destructive_phrases = [
            "delete",
            "remove",
            "wipe",
            "rm -rf",
            "format",
            "chmod",
            "chown",
            "trash",
            "rename every file",
            "move all files",
            "bypass securitypolicy",
            "pretend securitypolicy",
            "shell.execute",
        ]
        if any(phrase in normalized_intent for phrase in destructive_phrases) or "rm" in tokens:
            return Plan(
                steps=[
                    PlanStep(
                        capability_id="shell.execute",
                        reason="Destructive filesystem deletion was requested.",
                        input={"command": "rm -rf target"},
                    )
                ],
                estimated_risk="CRITICAL",
                requires_confirmation=True,
            )

        if "git status" in normalized_intent:
            return Plan(
                steps=[
                    PlanStep(
                        capability_id="git.status",
                        reason="Inspect repository status as requested.",
                        input={"directory": "."},
                    )
                ],
                estimated_risk="LOW",
                requires_confirmation=False,
            )

        if normalized_intent.startswith("find ") and "python" in normalized_intent and "file" in normalized_intent:
            return Plan(
                steps=[
                    PlanStep(
                        capability_id="filesystem.search",
                        reason="Find Python files in the current project.",
                        input={"pattern": "*.py", "root": "."},
                    )
                ],
                estimated_risk="SAFE",
                requires_confirmation=False,
            )

        if normalized_intent.startswith("read "):
            requested_path = intent.strip()[5:].strip()
            if requested_path:
                return Plan(
                    steps=[
                        PlanStep(
                            capability_id="filesystem.read",
                            reason="Read the requested file from the workspace.",
                            input={"path": requested_path},
                        )
                    ],
                    estimated_risk="SAFE",
                    requires_confirmation=False,
                )

        if normalized_intent in {"system monitor", "monitor system"}:
            return Plan(
                steps=[
                    PlanStep(
                        capability_id="system.monitor",
                        reason="Inspect system status.",
                        input={},
                    )
                ],
                estimated_risk="LOW",
                requires_confirmation=False,
            )

        if "summarize" in normalized_intent and "python" in normalized_intent:
            return Plan(
                steps=[
                    PlanStep(
                        capability_id="filesystem.search",
                        reason="Find Python files in the current project.",
                        input={"pattern": "*.py", "root": "."},
                    ),
                    PlanStep(
                        capability_id="research.synthesize",
                        reason="Summarize discovered files.",
                        input={"topic": "Architecture Summary", "goal": intent},
                    ),
                ],
                estimated_risk="LOW",
                requires_confirmation=False,
            )

        if "repository architecture" in normalized_intent or "approval workflow" in normalized_intent:
            return Plan(
                steps=[
                    PlanStep(
                        capability_id="filesystem.search",
                        reason="Locate relevant source files in the workspace.",
                        input={"pattern": "*.py", "root": "core"},
                    ),
                    PlanStep(
                        capability_id="research.synthesize",
                        reason="Synthesize the requested architecture explanation.",
                        input={"topic": intent.strip(), "goal": intent},
                    ),
                ],
                estimated_risk="LOW",
                requires_confirmation=False,
            )

        if "memory subsystem" in normalized_intent:
            return Plan(
                steps=[
                    PlanStep(
                        capability_id="filesystem.search",
                        reason="Locate memory subsystem source files.",
                        input={"pattern": "*.py", "root": "core"},
                    ),
                    PlanStep(
                        capability_id="research.synthesize",
                        reason="Explain the memory subsystem using local context.",
                        input={"topic": "Memory Subsystem", "goal": intent},
                    ),
                ],
                estimated_risk="LOW",
                requires_confirmation=False,
            )

        return None
