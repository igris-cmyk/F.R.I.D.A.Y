import logging
from typing import Any, Dict, List

from core.memory.manager import memory_manager

logger = logging.getLogger("friday.memory.retriever")


class MemoryRetriever:
    async def get_relevant_context(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Compatibility wrapper over MemoryManager retrieval."""
        try:
            return await memory_manager.retrieve_relevant_context(query, limit=limit)
        except Exception as exc:
            logger.warning("Memory retrieval failed: %s", exc)
            return []


memory_retriever = MemoryRetriever()
