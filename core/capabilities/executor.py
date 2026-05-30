import asyncio
import fnmatch
import json
import time
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Dict
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from core.capabilities.contracts import CapabilityInvocation, CapabilityResult, CapabilityFailure, CapabilityRequiresApproval
from core.capabilities.registry import CapabilityRegistry
from core.config import (
    FRIDAY_RESEARCH_MODEL,
    FRIDAY_RESEARCH_TIMEOUT_SECONDS,
    OLLAMA_BASE_URL,
)
from core.security.permissions import SecurityPolicy, RiskLevel

class CapabilityExecutor:
    """Deterministic execution boundary for system capabilities."""

    DEFAULT_SEARCH_EXCLUDED_DIRS = {
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
    DEFAULT_SEARCH_MAX_RESULTS = 500
    SYNTHESIS_MAX_FILES = 8
    SYNTHESIS_MAX_CHARS_PER_FILE = 4000
    SYNTHESIS_MAX_TOTAL_CHARS = 20000
    SYNTHESIS_LLM_TIMEOUT_SECONDS = FRIDAY_RESEARCH_TIMEOUT_SECONDS
    
    def __init__(self, registry: CapabilityRegistry, security_policy: SecurityPolicy):
        self.registry = registry
        self.security_policy = security_policy

    def _resolve_workspace_path(self, workspace_root_str: str, candidate_str: str) -> tuple[Path, Path]:
        workspace_root = Path(workspace_root_str).resolve()
        candidate_path = Path(candidate_str)
        if not candidate_path.is_absolute():
            candidate_path = workspace_root / candidate_path
        candidate_path = candidate_path.resolve()

        try:
            candidate_path.relative_to(workspace_root)
        except ValueError as exc:
            raise PermissionError("Access denied: Path outside of workspace scope.") from exc

        return workspace_root, candidate_path

    async def execute(self, invocation: CapabilityInvocation, human_approved: bool = False) -> CapabilityResult | CapabilityFailure | CapabilityRequiresApproval:
        """
        Validate, authorize, and execute a capability safely.
        """
        start_time = time.time()
        
        # 1. Lookup Capability
        definition = self.registry.get(invocation.capability_id)
        if not definition:
            return CapabilityFailure(
                capability_id=invocation.capability_id,
                error_code="CAPABILITY_NOT_FOUND",
                message=f"Capability '{invocation.capability_id}' is not registered.",
                trace_id=invocation.context.trace_id,
                recovery_hint="Check the capability ID and ensure it exists in the registry."
            )
            
        # 2. Security & Permission Validation
        evaluation = self.security_policy.evaluate_invocation(invocation, definition.risk_level, definition.mutation_allowed)

        if evaluation.status == "DENY":
            return CapabilityFailure(
                capability_id=invocation.capability_id,
                error_code="SECURITY_DENIAL",
                message=evaluation.reason,
                trace_id=invocation.context.trace_id,
                recovery_hint="Action requires human approval or is explicitly blocked."
            )

        if not definition.enabled:
            return CapabilityFailure(
                capability_id=invocation.capability_id,
                error_code="CAPABILITY_DISABLED",
                message=f"Capability '{invocation.capability_id}' is currently disabled.",
                trace_id=invocation.context.trace_id
            )

        # 3. Schema Validation (Basic check for MVP)
        required_fields = definition.input_schema.get("required", [])
        for field in required_fields:
            if field not in invocation.input_payload:
                return CapabilityFailure(
                    capability_id=invocation.capability_id,
                    error_code="INVALID_SCHEMA",
                    message=f"Missing required field: '{field}' in payload.",
                    trace_id=invocation.context.trace_id
                )

        # 4. Approval Validation
        if evaluation.status == "REQUIRES_APPROVAL" and not human_approved:
            return CapabilityRequiresApproval(
                capability_id=invocation.capability_id,
                reason=evaluation.reason,
                risk_level=definition.risk_level.value,
                trace_id=invocation.context.trace_id
            )
            
        # 5. Execution with Timeout (ALLOW status)
        timeout = invocation.timeout_seconds or definition.timeout_seconds
        
        try:
            # Wrap actual implementation in wait_for
            result_data = await asyncio.wait_for(
                self._execute_implementation(invocation, definition.capability_id),
                timeout=timeout
            )
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            return CapabilityResult(
                capability_id=invocation.capability_id,
                data=result_data,
                execution_time_ms=execution_time_ms,
                trace_id=invocation.context.trace_id
            )
            
        except asyncio.TimeoutError:
            return CapabilityFailure(
                capability_id=invocation.capability_id,
                error_code="TIMEOUT",
                message=f"Capability execution exceeded {timeout} seconds.",
                trace_id=invocation.context.trace_id
            )
        except Exception as e:
            # Catch-all boundary to prevent orchestrator crash
            return CapabilityFailure(
                capability_id=invocation.capability_id,
                error_code="EXECUTION_ERROR",
                message=str(e),
                trace_id=invocation.context.trace_id
            )
            
    async def _execute_implementation(self, invocation: CapabilityInvocation, capability_id: str) -> Dict[str, Any]:
        """
        Safe implementation router for MVP phase.
        """
        if capability_id == "filesystem.search":
            pattern = invocation.input_payload.get("pattern", "")
            root = invocation.input_payload.get("root", invocation.context.workspace_root)
            max_results = invocation.input_payload.get("max_results", self.DEFAULT_SEARCH_MAX_RESULTS)
            try:
                max_results = int(max_results)
            except (TypeError, ValueError):
                max_results = self.DEFAULT_SEARCH_MAX_RESULTS
            max_results = max(1, min(max_results, self.DEFAULT_SEARCH_MAX_RESULTS))

            workspace_root, search_root = self._resolve_workspace_path(invocation.context.workspace_root, root)

            results = []
            truncated = False
            for current_root, dirnames, filenames in os.walk(search_root, topdown=True):
                dirnames[:] = [
                    dirname for dirname in dirnames
                    if dirname not in self.DEFAULT_SEARCH_EXCLUDED_DIRS
                ]

                for filename in filenames:
                    if not fnmatch.fnmatch(filename, pattern):
                        continue

                    path = Path(current_root) / filename
                    results.append(str(path.relative_to(workspace_root)))
                    if len(results) >= max_results:
                        truncated = True
                        break

                if truncated:
                    break

            return {
                "files": results,
                "count": len(results),
                "truncated": truncated,
                "max_results": max_results,
                "excluded_dirs": sorted(self.DEFAULT_SEARCH_EXCLUDED_DIRS),
            }
            
        elif capability_id == "filesystem.read":
            path_str = invocation.input_payload.get("path", "")
            workspace_root, target_path = self._resolve_workspace_path(invocation.context.workspace_root, path_str)
                
            if not target_path.exists() or not target_path.is_file():
                raise FileNotFoundError(f"File not found: {path_str}")
                
            file_size = target_path.stat().st_size
            if file_size > 1024 * 1024: # 1MB limit
                raise ValueError("File too large. Maximum size is 1MB.")
                
            try:
                content = target_path.read_text(encoding="utf-8")
                return {"path": str(target_path.relative_to(workspace_root)), "content": content, "size": file_size, "truncated": False}
            except UnicodeDecodeError:
                raise ValueError("Binary files are not supported.")
                
        elif capability_id == "git.status":
            repo_dir = invocation.input_payload.get("directory", invocation.context.workspace_root)
            _, target_dir = self._resolve_workspace_path(invocation.context.workspace_root, repo_dir)
                
            process = await asyncio.create_subprocess_exec(
                "git", "status", "--short",
                cwd=target_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                raise RuntimeError(f"Git status failed: {stderr.decode()}")
            return {"status": stdout.decode().strip(), "directory": str(target_dir)}
            
        elif capability_id == "system.monitor":
            try:
                load1, load5, load15 = os.getloadavg()
            except AttributeError:
                load1, load5, load15 = (0.0, 0.0, 0.0) # Windows fallback
            return {
                "cpu_load_1m": load1,
                "cpu_load_5m": load5,
                "cpu_load_15m": load15,
                "status": "healthy"
            }
            
        elif capability_id == "memory.recall":
            query = invocation.input_payload.get("query")
            return {"memory": f"Simulated recall for: {query}", "confidence": 0.8}
            
        elif capability_id == "research.synthesize":
            return await self._execute_research_synthesis(invocation)
            
        else:
            raise ValueError(f"No implementation found for {capability_id}")

    async def _execute_research_synthesis(self, invocation: CapabilityInvocation) -> Dict[str, Any]:
        topic = invocation.input_payload.get("topic", invocation.context.source_intent)
        goal = invocation.input_payload.get("goal", topic)
        context = self._normalize_synthesis_context(invocation.input_payload.get("context", []))
        previous_results = invocation.input_payload.get("previous_results", [])

        if context and self._ollama_model_available(FRIDAY_RESEARCH_MODEL):
            try:
                synthesis = await self._synthesize_with_llm(topic=topic, goal=goal, context=context)
                return {
                    "synthesis": synthesis,
                    "grounded": True,
                    "llm_used": True,
                    "fallback_reason": None,
                    "inspected_files": [item["path"] for item in context],
                    "context_file_count": len(context),
                    "previous_result_count": len(previous_results),
                }
            except Exception as exc:
                fallback = self._deterministic_grounded_summary(topic, context, reason=str(exc))
                fallback["previous_result_count"] = len(previous_results)
                return fallback

        fallback_reason = "no_context" if not context else f"local_model_unavailable:{FRIDAY_RESEARCH_MODEL}"
        fallback = self._deterministic_grounded_summary(topic, context, reason=fallback_reason)
        fallback["previous_result_count"] = len(previous_results)
        return fallback

    def _normalize_synthesis_context(self, raw_context: Any) -> list[Dict[str, Any]]:
        if not isinstance(raw_context, list):
            return []

        normalized: list[Dict[str, Any]] = []
        total_chars = 0
        for item in raw_context[:self.SYNTHESIS_MAX_FILES]:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "unknown"))
            if any(part in self.DEFAULT_SEARCH_EXCLUDED_DIRS for part in path.replace("\\", "/").split("/")):
                continue

            content = str(item.get("content", ""))
            remaining = self.SYNTHESIS_MAX_TOTAL_CHARS - total_chars
            if remaining <= 0:
                break

            limit = min(self.SYNTHESIS_MAX_CHARS_PER_FILE, remaining)
            bounded_content = content[:limit]
            total_chars += len(bounded_content)
            normalized.append({
                "path": path,
                "content": bounded_content,
                "size": int(item.get("size", len(content))),
                "truncated": bool(item.get("truncated", False)) or len(content) > len(bounded_content),
            })

        return normalized

    def _ollama_model_available(self, model: str) -> bool:
        try:
            with urllib.request.urlopen(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=0.5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            model_names = {
                entry.get("model") or entry.get("name")
                for entry in payload.get("models", [])
                if isinstance(entry, dict)
            }
            return model in model_names
        except Exception:
            return False

    async def _synthesize_with_llm(self, topic: str, goal: str, context: list[Dict[str, Any]]) -> str:
        context_block = "\n\n".join(
            f"File: {item['path']}\nSize: {item['size']}\nTruncated: {item['truncated']}\nContent:\n{item['content']}"
            for item in context
        )
        prompt = PromptTemplate.from_template(
            "You are F.R.I.D.A.Y. Produce a concise, grounded technical synthesis.\n"
            "Use only the provided repository context. Cite file paths when making claims.\n\n"
            "Topic: {topic}\n"
            "Goal: {goal}\n\n"
            "Repository context:\n{context_block}\n\n"
            "Synthesis:"
        )
        chain = prompt | OllamaLLM(
            model=FRIDAY_RESEARCH_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
        )
        result = await asyncio.wait_for(
            chain.ainvoke({"topic": topic, "goal": goal, "context_block": context_block}),
            timeout=self.SYNTHESIS_LLM_TIMEOUT_SECONDS,
        )
        return result.strip()

    def _deterministic_grounded_summary(
        self,
        topic: str,
        context: list[Dict[str, Any]],
        reason: str,
    ) -> Dict[str, Any]:
        lines = [f"Grounded repository summary for: {topic}", "", "Inspected files:"]
        if not context:
            lines.extend([
                "- No file contents were available for synthesis.",
                "",
                "Limitations:",
                f"- Local semantic synthesis unavailable: {reason}.",
            ])
            return {
                "synthesis": "\n".join(lines),
                "grounded": False,
                "llm_used": False,
                "fallback_reason": reason,
                "inspected_files": [],
                "context_file_count": 0,
            }

        for index, item in enumerate(context, start=1):
            lines.append(f"{index}. {item['path']}")
            lines.append(f"   - {self._describe_file(item['path'], item['content'])}")
            if item["truncated"]:
                lines.append("   - Content was truncated to stay within the synthesis budget.")

        lines.extend([
            "",
            "Limitations:",
            f"- Local semantic synthesis unavailable: {reason}.",
            "- Summary is based on bounded file previews and path-level structure.",
        ])
        return {
            "synthesis": "\n".join(lines),
            "grounded": True,
            "llm_used": False,
            "fallback_reason": reason,
            "inspected_files": [item["path"] for item in context],
            "context_file_count": len(context),
        }

    def _describe_file(self, path: str, content: str) -> str:
        lowered = content.lower()
        if path == "core/main.py" or "nats" in lowered or "active_traces" in lowered:
            return "Orchestrator flow, NATS streaming, trace lifecycle, planning, and capability execution."
        if "capabilityexecutor" in lowered or "security_policy" in lowered:
            return "Capability execution boundary with registry lookup, policy checks, timeout handling, and implementations."
        if "securitypolicy" in lowered or "risklevel" in lowered:
            return "Security policy and risk evaluation for capability authorization."
        if "memory" in path or "memory" in lowered:
            return "Memory retrieval, compression, persistence, or reconstruction logic."
        if "router" in path or "classify_intent" in lowered:
            return "Intent routing and classification logic."
        if "planner" in path or "generate_plan" in lowered:
            return "Planner schema, fallback behavior, and capability plan construction."
        return "Repository source file included in the bounded synthesis context."
