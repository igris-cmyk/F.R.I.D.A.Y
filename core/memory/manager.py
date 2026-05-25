import asyncio
import logging
import json
import os
from enum import Enum
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import asyncpg

logger = logging.getLogger("friday.memory.manager")

class MemoryHealthState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    RECOVERING = "recovering"

class MemoryImportance(Enum):
    TRANSIENT = "transient"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    CRITICAL = "critical"

class MemoryManager:
    """
    Abstracted persistence boundary.
    No other subsystem interacts directly with pgvector.
    Provides Graceful Degradation and Bounded Retry Queues.
    """
    def __init__(self):
        self.db_url = os.environ.get("POSTGRES_URL", "postgresql://friday:friday_password@localhost:5432/friday_memory")
        self.pool: Optional[asyncpg.Pool] = None
        self.health: MemoryHealthState = MemoryHealthState.OFFLINE
        
        # Bounded Retry Queue (Prevents infinite memory leaks)
        self.max_queue_size = 100
        self.retry_queue = asyncio.Queue(maxsize=self.max_queue_size)
        
    async def connect(self):
        """Initializes connection pool and validates schema."""
        try:
            self.pool = await asyncpg.create_pool(self.db_url, min_size=1, max_size=10)
            self.health = MemoryHealthState.HEALTHY
            logger.info("MemoryManager: Connected to PostgreSQL.")
            await self._initialize_schema()
            
            # Start background retry processor
            asyncio.create_task(self._process_retry_queue())
        except Exception as e:
            self.health = MemoryHealthState.OFFLINE
            logger.warning(f"MemoryManager: PostgreSQL connection failed: {e}. Degrading to OFFLINE state.")

    async def _initialize_schema(self):
        """Idempotent setup for episodic memory table and pgvector."""
        if self.health != MemoryHealthState.HEALTHY or not self.pool:
            return
            
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS episodic_memory (
                        trace_id UUID PRIMARY KEY,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        intent TEXT NOT NULL,
                        importance_class TEXT NOT NULL,
                        workflow_summary TEXT NOT NULL,
                        environment_context JSONB,
                        metadata JSONB,
                        embedding vector(768) -- nomic-embed-text is 768 dims
                    );
                """)
                logger.info("MemoryManager: Schema verified.")
        except Exception as e:
            logger.error(f"MemoryManager: Failed to initialize schema: {e}")
            self.health = MemoryHealthState.DEGRADED
            
    def _enqueue_for_retry(self, trace_id: str, intent: str, importance: MemoryImportance, summary: str, env: Dict, metadata: Dict, embedding: List[float]):
        """Handles offline fallback by queueing strictly high-value traces."""
        if importance == MemoryImportance.TRANSIENT:
            logger.info(f"MemoryManager: Discarding TRANSIENT trace {trace_id} (DB offline).")
            return
            
        try:
            record = (trace_id, intent, importance, summary, env, metadata, embedding)
            self.retry_queue.put_nowait(record)
            logger.info(f"MemoryManager: Queued trace {trace_id} for retry (Queue depth: {self.retry_queue.qsize()}/{self.max_queue_size}).")
        except asyncio.QueueFull:
            # Eviction Policy: FIFO. We discard the oldest and queue the newest.
            try:
                dropped = self.retry_queue.get_nowait()
                logger.warning(f"MemoryManager: Retry queue full. Evicted oldest trace {dropped[0]}.")
                self.retry_queue.put_nowait(record)
            except Exception:
                pass

    async def _process_retry_queue(self):
        """Background worker to drain the retry queue when HEALTHY."""
        while True:
            await asyncio.sleep(10)
            if self.health == MemoryHealthState.HEALTHY and self.pool and not self.retry_queue.empty():
                try:
                    record = await self.retry_queue.get()
                    trace_id, intent, importance, summary, env, metadata, embedding = record
                    
                    # Attempt persistence
                    await self.persist_episodic_trace(trace_id, intent, importance, summary, env, metadata, embedding)
                    self.retry_queue.task_done()
                    logger.info(f"MemoryManager: Recovered trace {trace_id} from retry queue.")
                except Exception as e:
                    logger.error(f"MemoryManager: Retry loop error: {e}")

    async def persist_episodic_trace(self, trace_id: str, intent: str, importance: MemoryImportance, 
                                     workflow_summary: str, environment_context: Dict[str, Any], 
                                     metadata: Dict[str, Any], embedding: List[float] = None):
        """Primary abstraction boundary for writing to Episodic Memory."""
        
        # Telemetry
        logger.info(f"MemoryManager: Persistence Attempt -> Trace {trace_id} [{importance.value.upper()}]")
        
        # Degradation Check
        if self.health != MemoryHealthState.HEALTHY or not self.pool:
            self._enqueue_for_retry(trace_id, intent, importance, workflow_summary, environment_context, metadata, embedding)
            
            # Temporary fallback logging to stdout so we can observe the workflow compression
            fallback_log = {
                "trace_id": trace_id,
                "importance": importance.value,
                "workflow_summary": workflow_summary,
                "context": environment_context
            }
            logger.info(f"FALLBACK MEMORY LOG: {json.dumps(fallback_log)}")
            return
            
        try:
            # Parse embedding if provided
            embedding_str = f"[{','.join(map(str, embedding))}]" if embedding else None
            
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO episodic_memory 
                    (trace_id, intent, importance_class, workflow_summary, environment_context, metadata, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (trace_id) DO NOTHING;
                    """,
                    trace_id, intent, importance.value, workflow_summary, 
                    json.dumps(environment_context), json.dumps(metadata), embedding_str
                )
        except Exception as e:
            logger.error(f"MemoryManager: Failed to write trace {trace_id} to DB: {e}")
            self._enqueue_for_retry(trace_id, intent, importance, workflow_summary, environment_context, metadata, embedding)

    async def retrieve_relevant_context(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Semantic Retrieval boundary. Prevent nearest-neighbor flooding by capping limits."""
        if self.health != MemoryHealthState.HEALTHY or not self.pool:
            logger.warning("MemoryManager: Cannot retrieve context while OFFLINE.")
            return []
            
        try:
            embedding_str = f"[{','.join(map(str, query_embedding))}]"
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT trace_id, intent, importance_class, workflow_summary, created_at
                    FROM episodic_memory
                    ORDER BY embedding <-> $1
                    LIMIT $2;
                    """,
                    embedding_str, limit
                )
                
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"MemoryManager: Semantic retrieval failed: {e}")
            return []

# Global singleton equivalent to execution_logger
memory_manager = MemoryManager()
