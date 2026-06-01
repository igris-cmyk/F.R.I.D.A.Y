import asyncio
import argparse
import json
import sys

from core.agents.memory_agent import RetrievalPolicy, memory_agent
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
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:])
    command = args.command
    await memory_manager.initialize()

    if command == "health":
        print(json.dumps(await memory_manager.health(), indent=2, sort_keys=True))
        return 0

    if command == "list":
        rows = await asyncio.to_thread(memory_manager.store.list_memories, 20)
        print(json.dumps([compact_memory_row(row, verbose=args.verbose) for row in rows], indent=2, sort_keys=True))
        return 0

    if command == "search":
        query = " ".join(args.query).strip()
        if not query:
            parser.error("search requires a query")
        policy = parse_policy(args.policy)
        search_result = await memory_agent.search_candidates(query, policy)
        results = search_result["candidates"]
        print(json.dumps([compact_memory_row(row, verbose=args.verbose) for row in results], indent=2, sort_keys=True))
        if not results:
            diagnostics = search_result["diagnostics"]
            print(
                (
                    "[INFO] No memory results. "
                    f"item_count={diagnostics['item_count']} "
                    f"embedded_count={diagnostics.get('embedded_count', 0)} "
                    f"embedding_available={str(diagnostics['embedding_available']).lower()} "
                    f"retrieval_mode={diagnostics['retrieval_mode']} "
                    f"threshold={diagnostics['confidence_threshold']} "
                    f"raw_candidates={diagnostics['raw_candidate_count']} "
                    f"ranked_candidates={diagnostics['ranked_candidate_count']}"
                ),
                file=sys.stderr,
            )
        return 0

    parser.print_help(sys.stderr)
    return 2


def parse_policy(value: str) -> RetrievalPolicy:
    normalized = (value or "project_recall").strip().lower()
    if normalized == "default":
        normalized = RetrievalPolicy.PROJECT_RECALL.value
    for policy in RetrievalPolicy:
        if policy.value == normalized:
            return policy
    raise ValueError(f"Unknown policy: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect FRIDAY persistent memory.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("health", help="Show memory backend health.")

    list_parser = subparsers.add_parser("list", help="List recent memory items.")
    list_parser.add_argument("--verbose", action="store_true", help="Include metadata and previews.")

    search_parser = subparsers.add_parser("search", help="Search memory items.")
    search_parser.add_argument("query", nargs="+", help="Memory search query.")
    search_parser.add_argument(
        "--policy",
        default="project_recall",
        help="Retrieval policy: project_recall, deep_research, fast_context, debug_reconstruct, personal_continuity, or default.",
    )
    search_parser.add_argument("--verbose", action="store_true", help="Include metadata and previews.")
    return parser


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
