import asyncio
import json
import logging
from typing import Dict, Any
from datetime import datetime, timezone
from nats.aio.client import Client as NATS

from core.schemas.events import CoreHealthEvent, CoreHealthPayload, EventMetadata
from core.memory.manager import memory_manager, MemoryHealthState

logger = logging.getLogger("friday.api.health")

async def broadcast_health_telemetry(nc: NATS, active_traces: Dict[str, Any]):
    """
    Background worker that broadcasts core health telemetry every 10 seconds.
    The Tauri System Tray daemon will subscribe to this to render the health icon.
    """
    subject = "friday.system.health"
    
    while True:
        try:
            await asyncio.sleep(10)
            if not nc.is_connected:
                continue
                
            stalled_count = sum(1 for t in active_traces.values() if t.status.value in ["stalling", "recovering"])
            
            # Simple heuristic for Ollama availability
            # In a full prod app we would ping the Ollama local HTTP endpoint
            ollama_ok = True 
            
            # Memory Health Mapping
            sys_status = "healthy"
            if memory_manager.health_state == MemoryHealthState.OFFLINE:
                sys_status = "degraded"
            elif stalled_count > 0:
                sys_status = "recovering"
                
            event = CoreHealthEvent(
                metadata=EventMetadata(
                    trace_id="system-telemetry",
                    source_component="core.api.health",
                    timestamp=datetime.now(timezone.utc).isoformat()
                ),
                payload=CoreHealthPayload(
                    status=sys_status,
                    active_traces_count=len(active_traces),
                    stalled_traces_count=stalled_count,
                    memory_queue_depth=memory_manager.retry_queue.qsize(),
                    ollama_available=ollama_ok
                )
            )
            
            await nc.publish(subject, event.model_dump_json().encode())
            logger.debug(f"[TELEMETRY] Broadcasted Health State: {sys_status.upper()}")
            
        except Exception as e:
            logger.error(f"[TELEMETRY] Failed to broadcast health: {e}")
