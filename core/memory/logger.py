import asyncio
import logging
import json
from datetime import datetime, timezone
import asyncpg
import os
from typing import Dict, Any, Optional

logger = logging.getLogger("friday.memory.logger")

class ExecutionLogger:
    def __init__(self):
        self.db_url = os.environ.get("POSTGRES_URL", "postgresql://friday:friday_password@localhost:5432/friday_memory")
        self.pool: Optional[asyncpg.Pool] = None
        
    async def connect(self):
        try:
            self.pool = await asyncpg.create_pool(self.db_url)
            logger.info("Connected to PostgreSQL Execution Log.")
        except Exception as e:
            logger.warning(f"Database connection failed: {e}. Execution logs will degrade to stdout temporarily.")

    async def log_execution(self, trace_id: str, agent: str, action_type: str, 
                            target: str, input_summary: str, output_summary: str, 
                            duration_ms: int, success: bool, error_message: str = None, 
                            metadata: Dict[str, Any] = None):
        """
        Persists execution events to the DB to build the continuous operating memory.
        Fallback to stdout if DB is unavailable.
        """
        record = {
            "trace_id": trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": agent,
            "action_type": action_type,
            "target": target,
            "input_summary": input_summary,
            "output_summary": output_summary,
            "duration_ms": duration_ms,
            "success": success,
            "error_message": error_message,
            "metadata": metadata or {}
        }

        if self.pool:
            try:
                async with self.pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO execution_log 
                        (trace_id, agent, action_type, target, input_summary, output_summary, duration_ms, success, error_message, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        """,
                        trace_id, agent, action_type, target, input_summary, output_summary, 
                        duration_ms, success, error_message, json.dumps(record["metadata"])
                    )
            except Exception as e:
                logger.error(f"Failed to write execution log to DB: {e}")
                logger.info(f"FALLBACK LOG: {json.dumps(record)}")
        else:
            # Degrade gracefully
            logger.info(f"TRACE_LOG [{trace_id}]: {action_type} via {agent} -> Success: {success} ({duration_ms}ms)")

# Global singleton
execution_logger = ExecutionLogger()
