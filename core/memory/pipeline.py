import asyncio
import logging
import json
import time
from typing import Dict, Any, Optional

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings

from core.config import FRIDAY_MEMORY_MODEL, OLLAMA_BASE_URL
from core.memory.manager import memory_manager, MemoryHealthState, MemoryImportance

logger = logging.getLogger("friday.memory.pipeline")

# Primary Cognitive Model for Compression
compression_llm = OllamaLLM(
    model=FRIDAY_MEMORY_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.1,
)

# Embedding Model (Must be verified before use)
try:
    embedding_llm = OllamaEmbeddings(model="nomic-embed-text")
except Exception as e:
    embedding_llm = None
    logger.warning(f"MemoryPipeline: Failed to initialize OllamaEmbeddings: {e}")

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
            timeout=5.0
        )
        return summary.strip()
    except asyncio.TimeoutError:
        logger.warning("MemoryPipeline: Compression LLM timed out.")
        return f"Timeout during summarization of intent: {normalized_trace.get('intent')}"

async def generate_embedding(text: str) -> Optional[list[float]]:
    """
    Validation Stage: Checks if embedding model is available and generates vector.
    """
    if not embedding_llm:
        logger.warning("MemoryPipeline: Embeddings unavailable. Skipping.")
        return None
        
    try:
        # Note: langchain-ollama embeddings usually use sync embed_query.
        # For true async in production, we should wrap in run_in_executor.
        # Or use aiohttp directly against the Ollama API.
        loop = asyncio.get_event_loop()
        vector = await loop.run_in_executor(None, embedding_llm.embed_query, text)
        return vector
    except Exception as e:
        logger.warning(f"MemoryPipeline: Embedding generation failed: {e}")
        return None

async def process_completed_trace(trace_id: str, intent: str, command: str, result: str, error_state: bool, environment: Dict[str, Any], metadata: Dict[str, Any]):
    """
    The orchestrator-controlled Memory Lifecycle Pipeline.
    Runs strictly as a fire-and-forget background task.
    """
    start_time = time.time()
    
    # Stage 1: Trace Normalization
    normalized_trace = {
        "trace_id": trace_id,
        "intent": intent,
        "command": command,
        "result": result,
        "error_state": error_state
    }
    
    # Stage 2: Relevance Scoring
    importance = await score_relevance(normalized_trace)
    if importance == MemoryImportance.TRANSIENT:
        logger.info(f"[MEMORY PIPELINE] Trace {trace_id} -> REJECTED (Relevance: TRANSIENT)")
        return
        
    logger.info(f"[MEMORY PIPELINE] Trace {trace_id} -> IMPORTANCE: {importance.value.upper()}")
    
    # Stage 3: Adaptive Degradation & Stage 4: Compression
    # (If we were severely degraded, we could skip compression, but for now we proceed)
    compression_start = time.time()
    workflow_summary = await compress_workflow(normalized_trace)
    compression_ms = int((time.time() - compression_start) * 1000)
    
    # Stage 5: Embedding Validation
    embedding_start = time.time()
    embedding_vector = await generate_embedding(workflow_summary)
    embedding_ms = int((time.time() - embedding_start) * 1000) if embedding_vector else 0
    
    # Stage 6: Persistence to MemoryManager
    await memory_manager.persist_episodic_trace(
        trace_id=trace_id,
        intent=intent,
        importance=importance,
        workflow_summary=workflow_summary,
        environment_context=environment,
        metadata={"compression_ms": compression_ms, "embedding_ms": embedding_ms, **metadata},
        embedding=embedding_vector
    )
    
    total_ms = int((time.time() - start_time) * 1000)
    logger.info(f"[MEMORY PIPELINE] Trace {trace_id} -> PERSISTED (Total: {total_ms}ms, Comp: {compression_ms}ms, Embed: {embedding_ms}ms)")
