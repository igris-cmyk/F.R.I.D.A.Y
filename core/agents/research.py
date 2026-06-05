import asyncio
import time
import logging
import json
from typing import AsyncGenerator, Dict, Any
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from core.agents.memory_agent import memory_agent, RetrievalPolicy
from core.config import ENABLE_LOCAL_LLM, FRIDAY_RESEARCH_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger("friday.agents.research")

llm = (
    OllamaLLM(
        model=FRIDAY_RESEARCH_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.4 # Slightly creative for synthesis, but still bounded
    )
    if ENABLE_LOCAL_LLM
    else None
)

async def execute_research(query: str, environment: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Pure Cognition Worker Pipeline.
    This operates strictly as an async generator, yielding structured progress updates.
    It does NOT touch the event bus directly. The orchestrator maps yields to events.
    """
    logger.info(f"Starting research pipeline for query: {query}")
    
    env = environment or {}
    active_app = env.get("active_app", "unknown")
    window_title = env.get("window_title", "unknown")
    
    context_str = ""
    if active_app != "unknown":
        context_str = f"[AMBIENT CONTEXT: User is actively looking at '{active_app}' with window title '{window_title}']\n"
    
    # Stage 1: Retrieval
    yield {"stage": "retrieval", "message": "Initializing cognitive parameters...", "progress_percentage": 10}
    await asyncio.sleep(0.5) # Simulate setup
    
    yield {"stage": "retrieval", "message": f"Querying knowledge sources for: {query}", "progress_percentage": 30}
    # Future: Actual vector DB & Web Search integration happens here
    await asyncio.sleep(1.0) # Simulate IO bounds
    
    # Stage 2: Synthesis
    yield {"stage": "synthesizing", "message": "Analyzing context...", "progress_percentage": 60}
    
    # Memory recall
    recall_result = await memory_agent.recall_context(
        query=query, 
        policy=RetrievalPolicy.DEEP_RESEARCH,
        current_workspace=environment.get("working_directory") if environment else None
    )
    
    cognitive_context = recall_result["narrative"] if recall_result else "No relevant continuity found."
    
    prompt = PromptTemplate.from_template(
        "You are F.R.I.D.A.Y, an analytical research assistant.\n"
        "Synthesize a clear, highly concise, operational response to the user's query.\n"
        "Do not use conversational filler. Be direct and technical.\n"
        "If Ambient Context is provided, use it if it seems relevant to the query.\n\n"
        "Memory Continuity: {cognitive_context}\n\n"
        "{context}"
        "Query: {query}\n\n"
        "Synthesis:"
    )
    
    yield {"stage": "synthesizing", "message": "Generating structured synthesis...", "progress_percentage": 80}

    if llm is None:
        yield {
            "stage": "completed",
            "message": "Synthesis complete using deterministic local fallback.",
            "progress_percentage": 100,
            "final_result": (
                "Local research LLM is disabled. "
                f"Memory continuity: {cognitive_context}\nQuery: {query}"
            ),
            "success": True,
            "latency_ms": 0,
        }
        return

    chain = prompt | llm
    
    try:
        # In a fully streaming setup, we would use astream() here.
        # For this MVP step, we'll await ainvoke to generate the full block.
        start_time = time.time()
        result = await chain.ainvoke({"query": query, "context": context_str, "cognitive_context": cognitive_context})
        latency = int((time.time() - start_time) * 1000)
        
        # Stage 3: Memory Extraction
        yield {"stage": "memory_retrieval", "message": "Persisting synthesis to working memory...", "progress_percentage": 95}
        await asyncio.sleep(0.5) # Simulate extraction processing
        
        # Stage 4: Completion marker
        yield {
            "stage": "completed", 
            "message": f"Synthesis complete ({latency}ms).", 
            "progress_percentage": 100, 
            "final_result": result.strip(), 
            "success": True,
            "latency_ms": latency
        }
        
    except Exception as e:
        logger.error(f"Research synthesis failed: {e}")
        yield {
            "stage": "failed", 
            "message": "Cognitive inference failed.", 
            "progress_percentage": 100, 
            "final_result": str(e), 
            "success": False,
            "latency_ms": 0
        }
