import asyncio
import time
import os
import subprocess
from pathlib import Path
from typing import Any, Dict
from core.capabilities.contracts import CapabilityInvocation, CapabilityResult, CapabilityFailure, CapabilityRequiresApproval
from core.capabilities.registry import CapabilityRegistry
from core.security.permissions import SecurityPolicy, RiskLevel

class CapabilityExecutor:
    """Deterministic execution boundary for system capabilities."""
    
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
            workspace_root, search_root = self._resolve_workspace_path(invocation.context.workspace_root, root)
                
            results = []
            for path in search_root.rglob(pattern):
                if path.is_file():
                    results.append(str(path.relative_to(workspace_root)))
            return {"files": results, "count": len(results)}
            
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
                return {"content": content, "size": file_size, "truncated": False}
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
            topic = invocation.input_payload.get("topic")
            return {"synthesis": f"Simulated synthesis of topic: {topic}", "tokens": 42}
            
        else:
            raise ValueError(f"No implementation found for {capability_id}")
