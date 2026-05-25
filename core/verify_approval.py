import asyncio
import json
import uuid
from nats.aio.client import Client as NATS

async def run_manual_test():
    nc = NATS()
    await nc.connect("nats://localhost:4222")
    
    trace_id = str(uuid.uuid4())
    
    # 1. Listen for stream events
    async def stream_handler(msg):
        event = json.loads(msg.data.decode())
        print(f"\n[STREAM] {event.get('payload', {}).get('message', event.get('payload', {}).get('output', ''))}")
        
    await nc.subscribe(f"friday.stream.{trace_id}", cb=stream_handler)
    
    # 2. Listen for permission requests
    async def permission_handler(msg):
        event = json.loads(msg.data.decode())
        print(f"\n[UI] Received Approval Request: {event['payload']['human_name']}")
        print("[UI] Auto-denying to test failure path...")
        
        # Send denial
        response = {
            "metadata": {
                "trace_id": trace_id,
                "timestamp": "now",
                "source_component": "test"
            },
            "payload": {
                "trace_id": trace_id,
                "capability_id": event['payload']['capability_id'],
                "approved": False,
                "user_decision": "denied",
                "response_timestamp": "now",
                "source_component": "test"
            }
        }
        await nc.publish(f"friday.permission.response.{trace_id}", json.dumps(response).encode())

    await nc.subscribe(f"friday.permission.request.{trace_id}", cb=permission_handler)
    
    # Send test intent
    print(f"Sending intent with trace_id={trace_id}...")
    intent = {
        "metadata": {
            "trace_id": trace_id,
            "timestamp": "now",
            "source_component": "test",
            "priority": "normal"
        },
        "payload": {
            "raw_command": "Run test medium action",
            "working_directory": "."
        }
    }
    
    await nc.publish("friday.intent.command", json.dumps(intent).encode())
    
    await asyncio.sleep(5)
    await nc.drain()

if __name__ == "__main__":
    asyncio.run(run_manual_test())
