import asyncio
import logging
import json
import time
from typing import Dict, Any, Optional

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from core.config import (
    FRIDAY_MEMORY_MODEL,
    FRIDAY_MEMORY_TIMEOUT_SECONDS,
    OLLAMA_BASE_URL,
)
from core.memory.manager import MemoryHealthState, memory_manager, MemoryImportance

logger = logging.getLogger("friday.memory.pipeline")

# Primary Cognitive Model for Compression
compression_llm = OllamaLLM(
    model=FRIDAY_MEMORY_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.1,
)

async def score_relevance(normalized_trace: Dict[str, Any]) -> MemoryImportance:
    """
    Fast Heuristic Stage: Determine Importance Class.
    Prevents meaningless traces from reaching the LLM for compression.
    """
    intent = normalized_trace.get("intent", "unknown")
    command = normalized_trace.get("command", "")
    output = normalized_trace.get("result", "")
    
    # 1. Fast Path: Low-Signal Terminals
    if intent == "terminal":
        # Discard routine navigations or simple commands
        if command.startswith("cd ") or command.startswith("ls ") or command == "pwd" or command == "clear":
            return MemoryImportance.TRANSIENT
            
        # Error states usually represent active debugging workflow -> Higher value
        if normalized_trace.get("error_state"):
            return MemoryImportance.EPISODIC
            
        # Large outputs without errors might be data extraction, moderate value
        if len(output) > 500:
            return MemoryImportance.EPISODIC
            
    # 2. Fast Path: Research and complex workflows
    if intent == "research":
        return MemoryImportance.SEMANTIC
        
    return MemoryImportance.TRANSIENT

async def compress_workflow(normalized_trace: Dict[str, Any]) -> str:
    """
    Compression Stage: Synthesizes the normalized artifact into a semantic memory.
    """
    prompt = PromptTemplate.from_template(
        "You are an analytical memory compressor for an operating system.\n"
        "Summarize the following workflow execution into a highly condensed semantic summary.\n"
        "Focus on the 'outcome', 'errors encountered', and 'structural impact'.\n"
        "Do not use conversational text. Write as a concise log entry.\n\n"
        "Intent: {intent}\n"
        "Command/Query: {command}\n"
        "Execution Outcome: {result_summary}\n\n"
        "Semantic Summary:"
    )
    
    # Truncate result to avoid blowing out context window
    result_str = normalized_trace.get("result", "")
    if len(result_str) > 2000:
        result_str = result_str[:1000] + "\n...[TRUNCATED]...\n" + result_str[-1000:]
        
    chain = prompt | compression_llm
    
    try:
        # Wrap in timeout to prevent LLM hanging
        summary = await asyncio.wait_for(
            chain.ainvoke({
                "intent": normalized_trace.get("intent"),
                "command": normalized_trace.get("command"),
                "result_summary": result_str
            }),
            timeout=FRIDAY_MEMORY_TIMEOUT_SECONDS
        )
        return summary.strip()
    except asyncio.TimeoutError:
        logger.warning("MemoryPipeline: Compression LLM timed out.")
        return _fallback_summary(normalized_trace, reason="compression_timeout")
    except Exception as exc:
        logger.warning("MemoryPipeline: Compression LLM failed: %s", exc)
        return _fallback_summary(normalized_trace, reason=f"compression_failed:{exc}")


def _fallback_summary(normalized_trace: Dict[str, Any], reason: str) -> str:
    command = normalized_trace.get("command", "")
    result = normalized_trace.get("result", "")
    if len(result) > 1200:
        result = result[:600] + "\n...[TRUNCATED]...\n" + result[-600:]
    return (
        f"Intent: {normalized_trace.get('intent', 'unknown')}. "
        f"Command: {command}. "
        f"Outcome summary generated deterministically because {reason}.\n"
        f"Result preview:\n{result}"
    ).strip()

async def generate_embedding(text: str) -> Optional[list[float]]:
    """
    Validation Stage: Checks if embedding model is available and generates vector.
    """
    return await memory_manager.generate_embedding(text)

async def process_completed_trace(trace_id: str, intent: str, command: str, result: str, error_state: bool, environment: Dict[str, Any], metadata: Dict[str, Any]):
    """
    The orchestrator-controlled Memory Lifecycle Pipeline.
    Runs strictly as a fire-and-forget background task.
    """
    start_time = time.time()
    logger.info("[MEMORY PIPELINE] Processing completed trace %s intent=%s command=%r", trace_id, intent, command)

    try:
        if memory_manager.health_state == MemoryHealthState.OFFLINE:
            logger.info("[MEMORY PIPELINE] MemoryManager offline; initializing before persistence.")
            await memory_manager.initialize()

        normalized_trace = {
            "trace_id": trace_id,
            "intent": intent,
            "command": command,
            "result": result,
            "error_state": error_state
        }

        importance = await score_relevance(normalized_trace)
        if importance == MemoryImportance.TRANSIENT:
            logger.info("[MEMORY PIPELINE] Skipped transient trace %s.", trace_id)
            return {"persisted": False, "embedded": False, "degraded_reason": "transient", "memory_id": None}

        logger.info("[MEMORY PIPELINE] Trace %s importance=%s", trace_id, importance.value.upper())

        compression_start = time.time()
        workflow_summary = await compress_workflow(normalized_trace)
        compression_ms = int((time.time() - compression_start) * 1000)

        embedding_start = time.time()
        embedding_vector = await generate_embedding(workflow_summary)
        embedding_ms = int((time.time() - embedding_start) * 1000) if embedding_vector else 0

        persistence_result = await memory_manager.persist_episodic_trace(
            trace_id=trace_id,
            intent=intent,
            importance=importance,
            workflow_summary=workflow_summary,
            environment_context=environment,
            metadata={
                "compression_ms": compression_ms,
                "embedding_ms": embedding_ms,
                "command": command,
                "intent_type": intent,
                "result_preview": result[:2000],
                **metadata,
            },
            embedding=embedding_vector
        )

        total_ms = int((time.time() - start_time) * 1000)
        if persistence_result.get("persisted"):
            logger.info(
                "[MEMORY PIPELINE] Persisted memory item trace=%s memory_id=%s embedded=%s total_ms=%s",
                trace_id,
                persistence_result.get("memory_id"),
                persistence_result.get("embedded"),
                total_ms,
            )
        else:
            logger.warning(
                "[MEMORY PIPELINE] Persistence failed: trace=%s reason=%s",
                trace_id,
                persistence_result.get("degraded_reason"),
            )
        return persistence_result
    except Exception as exc:
        logger.exception("[MEMORY PIPELINE] Persistence failed: trace=%s error=%s", trace_id, exc)
        return {"persisted": False, "embedded": False, "degraded_reason": str(exc), "memory_id": None}
