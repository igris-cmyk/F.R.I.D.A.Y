import asyncio
import json
import uuid
from datetime import datetime, timezone
from nats.aio.client import Client as NATS

async def main():
    nc = NATS()
    await nc.connect("nats://localhost:4222")
    
    trace_id = str(uuid.uuid4())
    print(f"Testing end-to-end event loop. Trace ID: {trace_id}")
    
    payload = {
        "metadata": {
            "trace_id": trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_component": "test.cli",
            "priority": "high"
        },
        "payload": {
            "raw_command": "echo 'F.R.I.D.A.Y event loop is fully operational'",
            "context_window": "terminal",
            "working_directory": "/tmp"
        }
    }
    
    try:
        msg = await nc.request(
            "friday.intent.command", 
            json.dumps(payload).encode(), 
            timeout=5.0
        )
        response = json.loads(msg.data.decode())
        print("Received execution result:")
        print(json.dumps(response, indent=2))
    except Exception as e:
        print(f"Event loop validation failed: {e}")
        
    await nc.drain()

if __name__ == '__main__':
    asyncio.run(main())
