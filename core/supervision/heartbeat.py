from enum import Enum
import time
from typing import Dict, Any
import asyncio

class SupervisionState(Enum):
    RUNNING = "running"
    STALLING = "stalling"
    RECOVERING = "recovering"
    FAILED = "failed"
    COMPLETED = "completed"

class HeartbeatMonitor:
    """
    Orchestrator-controlled Trace Supervision Rules.
    Defines TTL thresholds for detecting zombie tasks.
    """
    STALL_THRESHOLD_SEC = 15.0
    RECOVER_THRESHOLD_SEC = 30.0
    TERMINATE_THRESHOLD_SEC = 60.0

    @classmethod
    def evaluate_state(cls, last_heartbeat: float, current_time: float) -> SupervisionState:
        silence_duration = current_time - last_heartbeat
        
        if silence_duration > cls.TERMINATE_THRESHOLD_SEC:
            return SupervisionState.FAILED
        elif silence_duration > cls.RECOVER_THRESHOLD_SEC:
            return SupervisionState.RECOVERING
        elif silence_duration > cls.STALL_THRESHOLD_SEC:
            return SupervisionState.STALLING
            
        return SupervisionState.RUNNING

class TraceRecord:
    def __init__(self, trace_id: str, task: asyncio.Task, agent: str):
        self.trace_id = trace_id
        self.task = task
        self.agent = agent
        self.started_at = time.time()
        self.last_heartbeat = time.time()
        self.status = SupervisionState.RUNNING
        self.stage = "initializing"
        
    def bump_heartbeat(self, stage: str):
        self.last_heartbeat = time.time()
        self.stage = stage
        self.status = SupervisionState.RUNNING
