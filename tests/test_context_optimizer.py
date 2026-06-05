import tempfile
import unittest
from pathlib import Path

from core.research.context_optimizer import ContextOptimizer, extract_targeted_preview
from core.research.ranker import rank_research_files
from core.research.workspace_boost import WorkspaceIndexBoost
from core.workspace.indexer import WorkspaceIndexer


def write_file(root: Path, relative_path: str, content: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestContextOptimizer(unittest.TestCase):
    def test_uses_index_summaries_for_location_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_file(root, "core/agents/planner.py", "class CognitivePlanner:\n    def generate_plan(self):\n        pass\n")
            WorkspaceIndexer(root).build()
            boosts = WorkspaceIndexBoost(root).boost_map("where is planner implemented?")
            ranked = rank_research_files("where is planner implemented?", [], workspace_boosts=boosts)

            optimizer = ContextOptimizer("where is planner implemented?", root, workspace_boosts=boosts)
            plan = optimizer.select(ranked, index_records={"core/agents/planner.py": WorkspaceIndexer(root).show("core/agents/planner.py")})

        self.assertEqual(plan.selected_paths[0], "core/agents/planner.py")
        self.assertEqual(plan.decisions[0].source, "index_summary")
        self.assertEqual(plan.source_mix["index_summary"], 1)

    def test_selects_expected_memory_files_for_subsystem_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for path in (
                "core/memory/manager.py",
                "core/memory/pipeline.py",
                "core/memory/retriever.py",
                "core/agents/memory_agent.py",
            ):
                write_file(root, path, "class MemoryManager:\n    pass\n")
            WorkspaceIndexer(root).build()
            boosts = WorkspaceIndexBoost(root).boost_map("explain memory subsystem")
            ranked = rank_research_files("explain memory subsystem", [], workspace_boosts=boosts)

            optimizer = ContextOptimizer("explain memory subsystem", root, workspace_boosts=boosts)
            plan = optimizer.select(ranked, index_records={path: WorkspaceIndexer(root).show(path) for path in boosts})

        self.assertEqual(plan.selected_paths[:4], [
            "core/memory/manager.py",
            "core/memory/pipeline.py",
            "core/memory/retriever.py",
            "core/agents/memory_agent.py",
        ])
        self.assertTrue(all(decision.source == "file_read" for decision in plan.decisions))

    def test_selects_expected_nats_files_for_location_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_file(root, "core/main.py", "subject = f'friday.stream.{trace_id}'\n")
            write_file(root, "apps/desktop/src/main.js", "nc.subscribe(`friday.stream.${traceId}`)\n")
            write_file(root, "infra/nats.conf", "websocket {\n  port: 9222\n}\n")
            WorkspaceIndexer(root).build()
            boosts = WorkspaceIndexBoost(root).boost_map("which files handle NATS streaming?")
            ranked = rank_research_files("which files handle NATS streaming?", [], workspace_boosts=boosts)

            optimizer = ContextOptimizer("which files handle NATS streaming?", root, workspace_boosts=boosts)
            plan = optimizer.select(ranked, index_records={path: WorkspaceIndexer(root).show(path) for path in boosts})

        self.assertIn("core/main.py", plan.selected_paths)
        self.assertIn("apps/desktop/src/main.js", plan.selected_paths)
        self.assertIn("infra/nats.conf", plan.selected_paths)
        self.assertTrue(all(decision.source == "index_summary" for decision in plan.decisions[:3]))

    def test_deduplicates_repeated_paths_and_respects_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_file(root, "core/main.py", "architecture\n")
            ranked = rank_research_files("analyze repository architecture", ["core/main.py", "core/main.py"])
            optimizer = ContextOptimizer("analyze repository architecture", root)

            plan = optimizer.select(ranked)

        self.assertEqual(plan.selected_paths, ["core/main.py"])
        self.assertEqual(plan.budget.max_total_chars, 10000)

    def test_falls_back_without_index_and_skips_secret_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_file(root, "core/main.py", "architecture\n")
            write_file(root, ".env", "SECRET=value\n")
            ranked = rank_research_files("analyze repository architecture", ["core/main.py", ".env"])
            optimizer = ContextOptimizer("analyze repository architecture", root)

            plan = optimizer.select(ranked)

        self.assertEqual(plan.selected_paths, ["core/main.py"])
        self.assertEqual(plan.decisions[0].source, "file_read")

    def test_targeted_preview_respects_per_file_budget(self):
        content = "\n".join([
            "ignore me",
            "class SecurityPolicy:",
            "    def evaluate_invocation(self):",
            "        return 'safe'",
            "ignore me too",
        ])

        preview = extract_targeted_preview(content, ["SecurityPolicy"], max_chars=80)

        self.assertLessEqual(len(preview), 80)
        self.assertIn("SecurityPolicy", preview)


if __name__ == "__main__":
    unittest.main()
