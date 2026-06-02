import argparse
import json
import sys
from pathlib import Path

from core.workspace.indexer import WorkspaceIndexer
from core.workspace.store import dumps_record


def main() -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:])
    indexer = WorkspaceIndexer(Path.cwd(), db_path=args.db_path)

    if args.command == "build":
        print(json.dumps(indexer.build(), indent=2, sort_keys=True))
        return 0

    if args.command == "status":
        print(json.dumps(indexer.status(), indent=2, sort_keys=True))
        return 0

    if args.command == "search":
        query = " ".join(args.query).strip()
        if not query:
            parser.error("search requires a query")
        results = indexer.search(query, limit=args.limit)
        print(json.dumps([result.__dict__ for result in results], indent=2, sort_keys=True))
        return 0

    if args.command == "show":
        record = indexer.show(args.path)
        if not record:
            print(json.dumps({"error": "not_found", "path": args.path}, indent=2, sort_keys=True))
            return 1
        print(dumps_record(record))
        return 0

    parser.print_help(sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and inspect FRIDAY's deterministic workspace index.")
    parser.add_argument(
        "--db-path",
        default=None,
        help="Workspace index DB path. Relative paths are resolved under the current workspace.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("build", help="Rebuild the local workspace index.")
    subparsers.add_parser("status", help="Show index health and counts.")

    search_parser = subparsers.add_parser("search", help="Search indexed project intelligence.")
    search_parser.add_argument("query", nargs="+", help="Search query.")
    search_parser.add_argument("--limit", type=int, default=10, help="Maximum results.")

    show_parser = subparsers.add_parser("show", help="Show extracted facts for one indexed file.")
    show_parser.add_argument("path", help="Workspace-relative file path.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
