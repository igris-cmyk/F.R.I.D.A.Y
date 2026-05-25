import logging
import asyncio
import time
from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger("friday.agents.router")

OLLAMA_MODEL = "qwen2.5:7b" # Target local model
OLLAMA_BASE_URL = "http://localhost:11434"

# We instantiate the LLM, but it is strictly controlled
llm = OllamaLLM(
    model=OLLAMA_MODEL, 
    base_url=OLLAMA_BASE_URL,
    temperature=0.0 # Strict determinism
)

class RouterState(TypedDict):
    raw_command: str
    environment: Dict[str, Any]
    intent: str  # "terminal", "memory", "research", "conversation"
    parameters: Dict[str, Any]
    error: str
    routing_metadata: Dict[str, Any]

async def classify_intent(state: RouterState) -> RouterState:
    """
    Hybrid Intent Classifier.
    Prioritizes fast-path heuristics for operational fluidity.
    Falls back to cognitive inference (LLM) for ambiguous natural language.
    Degrades gracefully if LLM times out or errors.
    """
    cmd = state["raw_command"].strip()
    start_time = time.time()
    
    # 1. Fast-Path Heuristic Evaluation (Sub-millisecond)
    terminal_indicators = ["run", "execute", "ls", "pwd", "echo", "cat", "git", "docker", "npm", "cargo", "pkill"]
    memory_indicators = ["remember", "what did i do", "search", "find", "recall"]
    
    if any(cmd.lower().startswith(indicator) for indicator in terminal_indicators):
        state["intent"] = "terminal"
        # Naive extraction for fast path
        state["parameters"] = {"executable_command": cmd[4:].strip() if cmd.lower().startswith("run ") else cmd}
        state["routing_metadata"] = {"source": "heuristic", "confidence": 1.0, "latency_ms": int((time.time() - start_time)*1000)}
        logger.info(f"Router classified (heuristic): TERMINAL -> {state['parameters']['executable_command']}")
        return state

    if any(cmd.lower().startswith(ind) for ind in memory_indicators):
        state["intent"] = "memory"
        state["parameters"] = {"query": cmd}
        state["routing_metadata"] = {"source": "heuristic", "confidence": 1.0, "latency_ms": int((time.time() - start_time)*1000)}
        logger.info("Router classified (heuristic): MEMORY")
        return state

    # 2. Cognitive Classification Fallback (Bounded Local LLM Inference)
    logger.info(f"Heuristics bypassed. Engaging cognitive intent classification for: '{cmd}'")
    
    env = state.get("environment") or {}
    active_app = env.get("active_app", "unknown")
    window_title = env.get("window_title", "unknown")
    
    context_str = ""
    if active_app != "unknown":
        context_str = f"Ambient Context: The user is currently looking at the application '{active_app}' with the window titled '{window_title}'.\n"

    prompt = PromptTemplate.from_template(
        "You are an OS intent router. Classify the user's intent strictly into exactly ONE of the following categories: 'terminal', 'memory', 'research', 'conversation'.\n\n"
        "Rules:\n"
        "- If they ask to run a shell command, script, or manage a process, answer: terminal\n"
        "- If they ask about past events, logs, or stored information, answer: memory\n"
        "- If they ask a complex question requiring synthesis or analysis, answer: research\n"
        "- If they ask a simple greeting or casual chat, answer: conversation\n\n"
        "{context}"
        "User Command: {command}\n\n"
        "Category:"
    )
    
    try:
        # Enforce strict 2.0s latency bound on inference
        chain = prompt | llm
        result = await asyncio.wait_for(
            chain.ainvoke({"command": cmd, "context": context_str}), 
            timeout=2.0
        )
        
        classification = result.strip().lower()
        # Clean up formatting if the model hallucinated punctuation
        for valid_intent in ["terminal", "memory", "research", "conversation"]:
            if valid_intent in classification:
                state["intent"] = valid_intent
                state["parameters"] = {"message": cmd, "executable_command": cmd, "query": cmd} # Pass raw context everywhere
                state["routing_metadata"] = {
                    "source": "cognitive_llm", 
                    "confidence": 0.85, 
                    "model": OLLAMA_MODEL,
                    "latency_ms": int((time.time() - start_time)*1000)
                }
                logger.info(f"Router classified (llm): {valid_intent.upper()} in {state['routing_metadata']['latency_ms']}ms")
                return state
                
        # If the LLM returns garbage, trigger graceful degradation
        raise ValueError(f"LLM returned invalid classification: {classification}")
        
    except asyncio.TimeoutError:
        logger.warning("Cognitive classification timeout (2.0s). Degrading to fallback conversation.")
        state["error"] = "llm_timeout"
    except Exception as e:
        logger.error(f"Cognitive classification failed: {e}. Degrading to fallback conversation.")
        state["error"] = f"llm_error: {str(e)}"
        
    # 3. Graceful Degradation / Absolute Fallback
    state["intent"] = "conversation"
    state["parameters"] = {"message": cmd}
    state["routing_metadata"] = {"source": "fallback", "confidence": 0.1, "latency_ms": int((time.time() - start_time)*1000)}
    logger.info("Router classified (fallback): CONVERSATION")
    return state

def route_decision(state: RouterState) -> str:
    """Returns the name of the next node based on the intent."""
    return state.get("intent", "conversation")

# Build the LangGraph
workflow = StateGraph(RouterState)

# Nodes
workflow.add_node("classify", classify_intent)
workflow.add_node("terminal", lambda x: x) # Pass-through for orchestrator dispatch
workflow.add_node("memory", lambda x: x) # Pass-through
workflow.add_node("research", lambda x: x) # Pass-through
workflow.add_node("conversation", lambda x: x) # Pass-through

# Edges
workflow.set_entry_point("classify")
workflow.add_conditional_edges(
    "classify",
    route_decision,
    {
        "terminal": "terminal",
        "memory": "memory",
        "research": "research",
        "conversation": "conversation"
    }
)
workflow.add_edge("terminal", END)
workflow.add_edge("memory", END)
workflow.add_edge("research", END)
workflow.add_edge("conversation", END)

intent_router = workflow.compile()
