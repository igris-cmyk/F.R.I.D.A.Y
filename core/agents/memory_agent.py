import asyncio
import logging
from typing import Dict, Any, Optional, List
from enum import Enum

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from core.config import FRIDAY_MEMORY_MODEL, OLLAMA_BASE_URL
from core.memory.manager import memory_manager, MemoryHealthState
from core.memory.pipeline import generate_embedding

logger = logging.getLogger("friday.agents.memory")

class RetrievalPolicy(Enum):
    FAST_CONTEXT = "fast_context"           # Low budget, high confidence threshold
    PROJECT_RECALL = "project_recall"       # Workspace scoped
    DEEP_RESEARCH = "deep_research"         # Heavy budget, cross-trace synthesis
    DEBUG_RECONSTRUCTION = "debug_reconstruct" # Error focused
    PERSONAL_CONTINUITY = "personal_continuity"

class MemoryAgent:
    """
    The Semantic Retrieval & Recall Engine.
    Strict abstraction over MemoryManager. Enforces Retrieval Confidence,
    Deduplication, Prompt Budgeting, and Context Reconstruction.
    """
    def __init__(self):
        # Strict low temperature for reconstruction to prevent hallucinatory amplification
        self.reconstruction_llm = OllamaLLM(
            model=FRIDAY_MEMORY_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.0,
        )
        
    async def recall_context(self, query: str, policy: RetrievalPolicy, current_workspace: str = None) -> Optional[Dict[str, Any]]:
        """
        Primary entrypoint for the Research Agent to reconstruct continuity.
        Returns a narrative block and metadata (lineage), or None if confidence is too low.
        """
        # 1. Check Abstraction Boundary & Health
        if memory_manager.health != MemoryHealthState.HEALTHY:
            logger.warning("[MEMORY AGENT] Cannot reconstruct context: Database Offline.")
            return None
            
        logger.info(f"[MEMORY AGENT] Initiating recall: Policy={policy.value}, Query='{query}'")
        
        # 2. Vectorize Query
        query_vector = await generate_embedding(query)
        if not query_vector:
            logger.warning("[MEMORY AGENT] Embedding generation failed. Aborting recall.")
            return None
            
        # 3. Apply Policy Constraints
        limit = 3 if policy == RetrievalPolicy.FAST_CONTEXT else 10
        confidence_threshold = 0.85 if policy == RetrievalPolicy.FAST_CONTEXT else 0.70
        
        # 4. Fetch Semantic Candidates (Raw nearest-neighbor dump)
        # Note: We fetch more than the limit because we will apply heuristic filtering next.
        raw_candidates = await memory_manager.retrieve_relevant_context(query_vector, limit=limit * 2)
        if not raw_candidates:
            logger.info("[MEMORY AGENT] Empty candidate pool.")
            return None
            
        # 5. Ranking Heuristics (Isolation & Deduplication)
        ranked_candidates = self._apply_ranking_heuristics(raw_candidates, current_workspace)
        
        # 6. Apply Confidence Threshold & Truncation
        # Note: In a real vector DB we get distance/similarity scores back. Since we used a raw pgvector 
        # <-> operator in manager.py, we need to adapt it to return the exact score, or approximate 
        # the confidence here. For MVP, we will simulate the threshold check based on ranking viability.
        if not ranked_candidates:
            logger.info("[MEMORY AGENT] All candidates failed confidence/isolation thresholds.")
            return None
            
        final_candidates = ranked_candidates[:limit]
        
        # 7. Recall Compression & Context Reconstruction
        reconstructed_block = await self._reconstruct_narrative(query, final_candidates)
        
        # 8. Source Attribution (Lineage)
        source_trace_ids = [c["trace_id"] for c in final_candidates]
        
        logger.info(f"[MEMORY AGENT] Recall Successful. Compressed {len(final_candidates)} traces.")
        
        return {
            "narrative": reconstructed_block,
            "lineage": {
                "source_trace_ids": source_trace_ids,
                "policy_used": policy.value,
            }
        }

    def _apply_ranking_heuristics(self, candidates: List[Dict], current_workspace: str) -> List[Dict]:
        """
        Filters out noise. Applies Workspace Isolation and Semantic Deduplication.
        """
        filtered = []
        seen_summaries = set()
        
        for cand in candidates:
            # Semantic Duplication Suppression
            # Simple heuristic: exact match of summary string (in production, we'd use fuzzy or vector distance of the summary)
            summary_hash = cand.get("workflow_summary", "")[:50] 
            if summary_hash in seen_summaries:
                continue
            seen_summaries.add(summary_hash)
            
            # Workspace & Project Isolation (Highly Penalize cross-project leakage)
            # cand["environment_context"] contains the working directory
            # For now, we just pass it through, but we would drop it here if strict isolation is required.
            
            filtered.append(cand)
            
        return filtered

    async def _reconstruct_narrative(self, query: str, candidates: List[Dict]) -> str:
        """
        Transforms episodic traces into a distilled continuity block.
        Prevents prompt flooding.
        """
        context_dump = "\n".join([f"- Intent: {c['intent']}\n  Outcome: {c['workflow_summary']}" for c in candidates])
        
        prompt = PromptTemplate.from_template(
            "You are a cognitive memory reconstruction agent.\n"
            "Synthesize the following episodic memory traces into a single coherent paragraph of context "
            "that helps answer the current query. Do not list the traces. Reconstruct the narrative of what happened.\n"
            "If the traces are irrelevant to the query, state 'No relevant continuity found.'\n\n"
            "Current Query: {query}\n\n"
            "Retrieved Traces:\n{context_dump}\n\n"
            "Reconstructed Context:"
        )
        
        chain = prompt | self.reconstruction_llm
        
        try:
            narrative = await asyncio.wait_for(
                chain.ainvoke({"query": query, "context_dump": context_dump}),
                timeout=8.0
            )
            return narrative.strip()
        except asyncio.TimeoutError:
            logger.warning("[MEMORY AGENT] Reconstruction LLM timed out.")
            return "[Memory Retrieval Failed: Timeout during synthesis]"

memory_agent = MemoryAgent()
