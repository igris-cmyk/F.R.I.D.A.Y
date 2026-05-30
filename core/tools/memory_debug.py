import asyncio
import json
import sys

from core.memory.manager import memory_manager


async def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "health"
    await memory_manager.initialize()

    if command == "health":
        print(json.dumps(await memory_manager.health(), indent=2, sort_keys=True))
        return 0

    if command == "list":
        rows = await asyncio.to_thread(memory_manager.store.list_memories, 20)
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0

    if command == "search":
        query = " ".join(sys.argv[2:]).strip()
        if not query:
            print("Usage: core/.venv/bin/python -m core.tools.memory_debug search \"query\"", file=sys.stderr)
            return 2
        results = await memory_manager.retrieve_relevant_context(query, limit=5, min_score=0.05)
        print(json.dumps(results, indent=2, sort_keys=True))
        return 0

    print("Usage: memory_debug.py [health|list|search QUERY]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
