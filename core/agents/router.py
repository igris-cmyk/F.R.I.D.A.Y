import logging
import asyncio
import time
from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from core.config import FRIDAY_ROUTER_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger("friday.agents.router")

# We instantiate the LLM, but it is strictly controlled
llm = OllamaLLM(
    model=FRIDAY_ROUTER_MODEL,
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


def is_operational_command(cmd: str) -> bool:
    normalized = cmd.strip().lower()
    dangerous_terms = [
        "delete",
        "remove",
        "wipe",
        "rm -rf",
        "format",
        "chmod",
        "chown",
        "trash",
        "rename every file",
        "move all files",
        "bypass securitypolicy",
        "pretend securitypolicy",
        "shell.execute",
    ]

    direct_prefixes = [
        "run",
        "execute",
        "ls",
        "pwd",
        "echo",
        "cat",
        "git",
        "docker",
        "npm",
        "cargo",
        "pkill",
        "read ",
        "show ",
    ]
    if any(normalized.startswith(prefix) for prefix in direct_prefixes):
        return True

    if any(term in normalized for term in dangerous_terms):
        return True

    if "git status" in normalized:
        return True

    if normalized.startswith("find ") and " file" in normalized:
        return True

    if normalized in {"system monitor", "monitor system"}:
        return True

    return False


def is_conversational_greeting(cmd: str) -> bool:
    normalized = cmd.strip().lower()
    greetings = [
        "hello",
        "hello friday",
        "hi",
        "hi friday",
        "hey",
        "hey friday",
    ]
    return normalized in greetings


def is_general_conversation_request(cmd: str) -> bool:
    normalized = " ".join(cmd.strip().lower().split())
    conversation_prefixes = [
        "how are you",
        "tell me a joke",
        "what are ",
        "what is ",
        "explain what ",
        "explain how ",
    ]
    return any(normalized.startswith(prefix) for prefix in conversation_prefixes)


def is_memory_recall_request(cmd: str) -> bool:
    normalized = " ".join(cmd.strip().lower().split())
    recall_phrases = [
        "what did we just",
        "what did we work on",
        "what did we do",
        "what did i do",
        "what changed with",
        "what happened with",
        "what did we inspect",
        "what did we look at",
        "recall ",
        "remember ",
        "remind me",
        "summarize our recent",
        "our recent",
        "recently",
        "earlier",
    ]
    if any(phrase in normalized for phrase in recall_phrases):
        return True

    topic_terms = [
        "planner timeout",
        "approval workflow",
        "memory work",
        "memory subsystem",
        "repository architecture",
        "research ranking",
    ]
    return any(term in normalized for term in topic_terms) and any(
        cue in normalized for cue in ["what", "recall", "remind", "changed", "happened", "inspect"]
    )


def is_research_request(cmd: str) -> bool:
    normalized = cmd.strip().lower()
    research_phrases = [
        "analyze repository architecture",
        "explain memory subsystem",
        "show approval workflow",
    ]
    return normalized in research_phrases

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
    if is_operational_command(cmd):
        state["intent"] = "terminal"
        state["parameters"] = {"executable_command": cmd}
        state["routing_metadata"] = {"source": "heuristic", "confidence": 1.0, "latency_ms": int((time.time() - start_time)*1000)}
        logger.info(f"Router classified (heuristic): TERMINAL -> {state['parameters']['executable_command']}")
        return state

    if is_conversational_greeting(cmd) or is_general_conversation_request(cmd):
        state["intent"] = "conversation"
        state["parameters"] = {"message": cmd}
        state["routing_metadata"] = {"source": "heuristic", "confidence": 1.0, "latency_ms": int((time.time() - start_time)*1000)}
        logger.info("Router classified (heuristic): CONVERSATION")
        return state

    if is_memory_recall_request(cmd):
        state["intent"] = "memory"
        state["parameters"] = {"query": cmd}
        state["routing_metadata"] = {"source": "heuristic", "confidence": 1.0, "latency_ms": int((time.time() - start_time)*1000)}
        logger.info("Router classified (heuristic): MEMORY")
        return state

    if is_research_request(cmd):
        state["intent"] = "research"
        state["parameters"] = {"message": cmd, "query": cmd}
        state["routing_metadata"] = {"source": "heuristic", "confidence": 1.0, "latency_ms": int((time.time() - start_time)*1000)}
        logger.info("Router classified (heuristic): RESEARCH")
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
                    "model": FRIDAY_ROUTER_MODEL,
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
