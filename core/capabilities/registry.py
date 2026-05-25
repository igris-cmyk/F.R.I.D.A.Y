from typing import Dict, Any, Type, Optional
from pydantic import BaseModel, Field
from core.security.permissions import RiskLevel

class CapabilityDefinition(BaseModel):
    """Authoritative definition of a system capability."""
    capability_id: str
    human_name: str
    description: str
    risk_level: RiskLevel
    timeout_seconds: int = 30
    input_schema: Dict[str, Any]  # JSON schema representation
    enabled: bool = True
    mutation_allowed: bool = False

class CapabilityRegistry:
    """The authoritative index of all executable system capabilities."""
    
    def __init__(self):
        self._capabilities: Dict[str, CapabilityDefinition] = {}
        self._seed_registry()
        
    def _seed_registry(self):
        """Seed the registry with initial capabilities."""
        # 1. Filesystem Search (SAFE)
        self.register(CapabilityDefinition(
            capability_id="filesystem.search",
            human_name="Search Filesystem",
            description="Search for files or directories by pattern within the workspace.",
            risk_level=RiskLevel.SAFE,
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g. *.py)"},
                    "root": {"type": "string", "description": "Root directory to search from"}
                },
                "required": ["pattern"]
            },
            enabled=True,
            mutation_allowed=False
        ))
        
        # 2. Filesystem Read (SAFE)
        self.register(CapabilityDefinition(
            capability_id="filesystem.read",
            human_name="Read File",
            description="Read the contents of a specific file.",
            risk_level=RiskLevel.SAFE,
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path to the file"}
                },
                "required": ["path"]
            },
            enabled=True,
            mutation_allowed=False
        ))
        
        # 3. Memory Recall (SAFE)
        self.register(CapabilityDefinition(
            capability_id="memory.recall",
            human_name="Recall Memory",
            description="Query the episodic memory engine for context.",
            risk_level=RiskLevel.SAFE,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Semantic search query"},
                    "policy": {"type": "string", "description": "Retrieval policy (e.g., DEEP_RESEARCH)"}
                },
                "required": ["query"]
            },
            enabled=True,
            mutation_allowed=False
        ))
        
        # 4. Git Status (LOW)
        self.register(CapabilityDefinition(
            capability_id="git.status",
            human_name="Git Status",
            description="Check the current git repository status.",
            risk_level=RiskLevel.LOW,
            input_schema={
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Target git repository directory"}
                },
                "required": []
            },
            enabled=True,
            mutation_allowed=False
        ))
        
        # 5. System Monitor (LOW)
        self.register(CapabilityDefinition(
            capability_id="system.monitor",
            human_name="System Monitor",
            description="Check system health, cpu, and memory usage.",
            risk_level=RiskLevel.LOW,
            input_schema={
                "type": "object",
                "properties": {},
                "required": []
            },
            enabled=True,
            mutation_allowed=False
        ))
        
        # 6. Research Synthesize (SAFE)
        self.register(CapabilityDefinition(
            capability_id="research.synthesize",
            human_name="Synthesize Research",
            description="Synthesize context into a structured summary using the LLM.",
            risk_level=RiskLevel.SAFE,
            timeout_seconds=120, # Higher timeout for LLM inference
            input_schema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to synthesize"},
                    "context": {"type": "string", "description": "Raw context to synthesize from"}
                },
                "required": ["topic"]
            },
            enabled=True,
            mutation_allowed=False
        ))
        
        # 7. Shell Execute (CRITICAL - Disabled by default)
        self.register(CapabilityDefinition(
            capability_id="shell.execute",
            human_name="Execute Shell Command",
            description="Execute arbitrary shell commands. EXTREMELY DANGEROUS.",
            risk_level=RiskLevel.CRITICAL,
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Raw bash command to execute"}
                },
                "required": ["command"]
            },
            enabled=False, # Explicitly disabled for MVP
            mutation_allowed=False # Explicitly denied mutation for MVP
        ))
        
        # 8. Mock MEDIUM Action for Testing
        self.register(CapabilityDefinition(
            capability_id="test.medium_action",
            human_name="Test Medium Action",
            description="Mock capability for testing explicit approval requirement.",
            risk_level=RiskLevel.MEDIUM,
            input_schema={"type": "object", "properties": {}},
            enabled=True,
            mutation_allowed=False
        ))

    def register(self, definition: CapabilityDefinition):
        self._capabilities[definition.capability_id] = definition
        
    def get(self, capability_id: str) -> Optional[CapabilityDefinition]:
        return self._capabilities.get(capability_id)
        
    def get_all(self) -> Dict[str, CapabilityDefinition]:
        return self._capabilities.copy()
