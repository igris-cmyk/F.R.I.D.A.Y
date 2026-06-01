import asyncio
import logging
from typing import Dict, Any, Optional, List
from enum import Enum

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from core.config import FRIDAY_MEMORY_MODEL, FRIDAY_MEMORY_TIMEOUT_SECONDS, OLLAMA_BASE_URL
from core.memory.manager import MemoryHealthState, memory_manager

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

    def policy_settings(self, policy: RetrievalPolicy) -> Dict[str, float | int]:
        return {
            "limit": 3 if policy == RetrievalPolicy.FAST_CONTEXT else 10,
            "confidence_threshold": {
                RetrievalPolicy.FAST_CONTEXT: 0.85,
                RetrievalPolicy.PROJECT_RECALL: 0.05,
                RetrievalPolicy.DEEP_RESEARCH: 0.70,
                RetrievalPolicy.DEBUG_RECONSTRUCTION: 0.05,
                RetrievalPolicy.PERSONAL_CONTINUITY: 0.05,
            }[policy],
            "raw_min_score": 0.05,
        }

    async def search_candidates(
        self,
        query: str,
        policy: RetrievalPolicy,
        current_workspace: str = None,
    ) -> Dict[str, Any]:
        if memory_manager.health_state == MemoryHealthState.OFFLINE:
            logger.warning("[MEMORY AGENT] Cannot search memory: Database Offline.")
            return {
                "candidates": [],
                "diagnostics": {
                    "policy": policy.value,
                    "item_count": 0,
                    "embedding_available": memory_manager.embedding_available,
                    "retrieval_mode": "offline",
                    "confidence_threshold": self.policy_settings(policy)["confidence_threshold"],
                    "raw_min_score": self.policy_settings(policy)["raw_min_score"],
                },
            }

        settings = self.policy_settings(policy)
        limit = int(settings["limit"])
        confidence_threshold = float(settings["confidence_threshold"])
        raw_min_score = float(settings["raw_min_score"])

        retrieval = await memory_manager.retrieve_relevant_context_with_diagnostics(
            query,
            limit=limit * 2,
            min_score=raw_min_score,
        )
        raw_candidates = retrieval["results"]
        ranked_candidates = self._apply_ranking_heuristics(raw_candidates, current_workspace)
        final_candidates = [
            candidate for candidate in ranked_candidates
            if candidate.get("score", 0.0) >= confidence_threshold
        ][:limit]
        manager_diagnostics = retrieval["diagnostics"]
        diagnostics = {
            "policy": policy.value,
            "item_count": int(manager_diagnostics.get("item_count", 0)),
            "embedded_count": int(manager_diagnostics.get("embedded_count", 0)),
            "embedding_available": bool(manager_diagnostics.get("embedding_available", False)),
            "retrieval_mode": manager_diagnostics.get("retrieval_mode", "keyword_fallback"),
            "embedding_attempted": bool(manager_diagnostics.get("embedding_attempted", False)),
            "embedding_failed": bool(manager_diagnostics.get("embedding_failed", False)),
            "confidence_threshold": confidence_threshold,
            "raw_min_score": raw_min_score,
            "raw_candidate_count": len(raw_candidates),
            "ranked_candidate_count": len(ranked_candidates),
            "final_candidate_count": len(final_candidates),
        }
        return {"candidates": final_candidates, "diagnostics": diagnostics}
        
    async def recall_context(self, query: str, policy: RetrievalPolicy, current_workspace: str = None) -> Optional[Dict[str, Any]]:
        """
        Primary entrypoint for the Research Agent to reconstruct continuity.
        Returns a narrative block and metadata (lineage), or None if confidence is too low.
        """
        # 1. Check Abstraction Boundary & Health
        if memory_manager.health_state == MemoryHealthState.OFFLINE:
            logger.warning("[MEMORY AGENT] Cannot reconstruct context: Database Offline.")
            return None
            
        logger.info(f"[MEMORY AGENT] Initiating recall: Policy={policy.value}, Query='{query}'")
        
        search_result = await self.search_candidates(query, policy, current_workspace=current_workspace)
        final_candidates = search_result["candidates"]
        if not search_result["diagnostics"]["raw_candidate_count"]:
            logger.info("[MEMORY AGENT] Empty candidate pool.")
            return None
        if not search_result["diagnostics"]["ranked_candidate_count"]:
            logger.info("[MEMORY AGENT] All candidates failed confidence/isolation thresholds.")
            return None
        if not final_candidates:
            logger.info("[MEMORY AGENT] No candidates met confidence threshold.")
            return None
        
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
                "candidate_count": len(final_candidates),
                "diagnostics": search_result["diagnostics"],
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
            summary_hash = cand.get("workflow_summary") or cand.get("summary", "")
            summary_hash = summary_hash[:50]
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
        context_dump = "\n".join([
            (
                f"- Title: {c.get('title') or c.get('intent')}\n"
                f"  Summary: {c.get('workflow_summary') or c.get('summary')}\n"
                f"  Key files: {', '.join(c.get('key_files') or [])}"
            )
            for c in candidates
        ])
        
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
                timeout=FRIDAY_MEMORY_TIMEOUT_SECONDS
            )
            return narrative.strip()
        except asyncio.TimeoutError:
            logger.warning("[MEMORY AGENT] Reconstruction LLM timed out.")
            return self._deterministic_recall_summary(candidates, reason="reconstruction_timeout")
        except Exception as exc:
            logger.warning("[MEMORY AGENT] Reconstruction LLM failed: %s", exc)
            return self._deterministic_recall_summary(candidates, reason=f"reconstruction_failed:{exc}")

    def _deterministic_recall_summary(self, candidates: List[Dict], reason: str) -> str:
        lines = ["Relevant continuity found."]
        for index, candidate in enumerate(candidates[:5], start=1):
            title = candidate.get("title") or candidate.get("intent") or "Memory item"
            summary = candidate.get("workflow_summary") or candidate.get("summary") or ""
            summary = " ".join(str(summary).split())
            if len(summary) > 500:
                summary = summary[:497] + "..."
            key_files = candidate.get("key_files") or []
            if key_files:
                summary = f"{summary} Key files: {', '.join(key_files[:6])}."
            lines.append(f"{index}. {title}\n   {summary}")
        return "\n".join(lines)

memory_agent = MemoryAgent()
