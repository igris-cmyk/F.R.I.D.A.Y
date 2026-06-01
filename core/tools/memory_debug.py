import asyncio
import json
import sys

from core.memory.manager import memory_manager
from core.memory.summarizer import summarize_completed_trace


def compact_memory_row(row: dict, verbose: bool = False) -> dict:
    metadata = row.get("source_metadata") or {}
    if "metadata_json" in row:
        try:
            metadata = json.loads(row.get("metadata_json") or "{}")
        except json.JSONDecodeError:
            metadata = {}

    summary = row.get("summary")
    title = row.get("title") or metadata.get("title")
    key_files = row.get("key_files") or metadata.get("key_files", [])
    if _looks_like_legacy_summary(summary):
        display_summary = summarize_completed_trace({
            "trace_id": row.get("trace_id"),
            "intent": metadata.get("intent_type") or row.get("source_component") or "",
            "command": row.get("user_intent") or metadata.get("command") or row.get("intent") or "",
            "result": f"{row.get('summary', '')}\n{row.get('content_preview', '')}",
            "error_state": row.get("importance") == "critical",
            "metadata": metadata,
        })
        summary = display_summary.summary
        title = title or display_summary.title
        key_files = key_files or display_summary.key_files

    compact = {
        "title": title,
        "summary": summary,
        "key_files": key_files,
        "trace_id": row.get("trace_id"),
        "score": row.get("score"),
        "created_at": row.get("created_at"),
        "memory_type": row.get("memory_type"),
        "importance": row.get("importance"),
    }
    if verbose:
        compact["metadata"] = metadata
        compact["content_preview"] = row.get("content_preview")
    return compact


def _looks_like_legacy_summary(summary: object) -> bool:
    text = str(summary or "")
    return any(
        marker in text
        for marker in (
            "SUCCESS filesystem.search",
            "compression_timeout",
            "Outcome summary generated deterministically",
            "Result preview:",
        )
    )


async def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "health"
    verbose = "--verbose" in sys.argv
    args = [arg for arg in sys.argv[2:] if arg != "--verbose"]
    await memory_manager.initialize()

    if command == "health":
        print(json.dumps(await memory_manager.health(), indent=2, sort_keys=True))
        return 0

    if command == "list":
        rows = await asyncio.to_thread(memory_manager.store.list_memories, 20)
        print(json.dumps([compact_memory_row(row, verbose=verbose) for row in rows], indent=2, sort_keys=True))
        return 0

    if command == "search":
        query = " ".join(args).strip()
        if not query:
            print("Usage: core/.venv/bin/python -m core.tools.memory_debug search \"query\" [--verbose]", file=sys.stderr)
            return 2
        results = await memory_manager.retrieve_relevant_context(query, limit=5, min_score=0.05)
        print(json.dumps([compact_memory_row(row, verbose=verbose) for row in results], indent=2, sort_keys=True))
        return 0

    print("Usage: memory_debug.py [health|list [--verbose]|search QUERY [--verbose]]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
