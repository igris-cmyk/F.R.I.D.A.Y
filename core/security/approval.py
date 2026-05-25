from dataclasses import dataclass
from typing import Optional
import asyncio
import time


@dataclass
class PendingApproval:
    future: asyncio.Future
    capability_id: str
    expires_at: float

    def is_expired(self, now: Optional[float] = None) -> bool:
        current_time = time.time() if now is None else now
        return current_time >= self.expires_at


def matches_pending_approval(
    pending: Optional[PendingApproval],
    trace_id: str,
    capability_id: str,
    now: Optional[float] = None,
) -> bool:
    if pending is None:
        return False

    if pending.future.done():
        return False

    if pending.is_expired(now):
        return False

    return pending.capability_id == capability_id and bool(trace_id)
