import unittest

from core.research.context_builder import (
    ContextBudget,
    build_context_file,
    context_budget_used,
    select_ranked_files,
)
from core.research.ranker import rank_research_files


class TestResearchRanking(unittest.TestCase):
    def test_memory_query_ranks_memory_files_first(self):
        ranked = rank_research_files("explain memory subsystem", [
            "core/main.py",
            "core/memory/pipeline.py",
            "core/memory/manager.py",
            "core/memory/retriever.py",
            "core/agents/memory_agent.py",
            "core/capabilities/executor.py",
        ])
        self.assertEqual([item.path for item in ranked[:4]], [
            "core/memory/manager.py",
            "core/memory/pipeline.py",
            "core/memory/retriever.py",
            "core/agents/memory_agent.py",
        ])

    def test_approval_query_ranks_security_files_first(self):
        ranked = rank_research_files("show approval workflow", [
            "core/main.py",
            "core/security/permissions.py",
            "core/security/approval.py",
            "core/schemas/events.py",
            "apps/desktop/src/main.js",
        ])
        self.assertEqual([item.path for item in ranked[:4]], [
            "core/security/approval.py",
            "core/security/permissions.py",
            "core/main.py",
            "core/schemas/events.py",
        ])

    def test_planner_query_ranks_planner_router_files_first(self):
        ranked = rank_research_files("explain planner router intent flow", [
            "core/main.py",
            "core/agents/router.py",
            "core/agents/planner.py",
            "core/capabilities/executor.py",
        ])
        self.assertEqual([item.path for item in ranked[:3]], [
            "core/agents/planner.py",
            "core/agents/router.py",
            "core/main.py",
        ])

    def test_capability_query_ranks_capability_files_first(self):
        ranked = rank_research_files("explain capability executor registry", [
            "core/main.py",
            "core/security/permissions.py",
            "core/capabilities/registry.py",
            "core/capabilities/executor.py",
        ])
        self.assertEqual([item.path for item in ranked[:3]], [
            "core/capabilities/executor.py",
            "core/capabilities/registry.py",
            "core/security/permissions.py",
        ])

    def test_frontend_streaming_query_ranks_frontend_and_events(self):
        ranked = rank_research_files("frontend streaming tauri ui", [
            "core/main.py",
            "core/schemas/events.py",
            "apps/desktop/src/main.js",
            "apps/desktop/src-tauri/src/lib.rs",
            "core/capabilities/executor.py",
        ])
        self.assertEqual([item.path for item in ranked[:4]], [
            "apps/desktop/src/main.js",
            "apps/desktop/src-tauri/src/lib.rs",
            "core/main.py",
            "core/schemas/events.py",
        ])

    def test_architecture_query_produces_representative_core_files(self):
        ranked = rank_research_files("analyze repository architecture", [
            "core/main.py",
            "core/agents/planner.py",
            "core/capabilities/executor.py",
            "core/security/permissions.py",
            "core/memory/manager.py",
            "tests/test_capabilities.py",
        ])
        self.assertEqual([item.path for item in ranked[:4]], [
            "core/main.py",
            "core/agents/planner.py",
            "core/capabilities/executor.py",
            "core/security/permissions.py",
        ])

    def test_tests_are_penalized_unless_query_mentions_tests(self):
        without_tests = rank_research_files("analyze repository architecture", [
            "core/main.py",
            "tests/test_capabilities.py",
        ])
        with_tests = rank_research_files("analyze tests for capabilities", [
            "core/main.py",
            "tests/test_capabilities.py",
        ])
        self.assertNotIn("tests/test_capabilities.py", [item.path for item in without_tests])
        self.assertEqual(with_tests[0].path, "tests/test_capabilities.py")

    def test_ranking_is_deterministic_and_excludes_generated_paths(self):
        candidates = [
            "core/main.py",
            "core/.venv/lib/site.py",
            "node_modules/pkg/index.js",
            "core/agents/planner.py",
        ]
        first = rank_research_files("repository architecture", candidates)
        second = rank_research_files("repository architecture", candidates)
        self.assertEqual(first, second)
        self.assertFalse(any(".venv" in item.path or "node_modules" in item.path for item in first))


class TestBoundedContextBuilder(unittest.TestCase):
    def test_selects_max_four_files_by_default(self):
        ranked = rank_research_files("repository architecture", [
            "core/main.py",
            "core/agents/planner.py",
            "core/capabilities/executor.py",
            "core/security/permissions.py",
            "core/memory/manager.py",
        ])
        self.assertEqual(len(select_ranked_files(ranked)), 4)

    def test_respects_per_file_and_total_char_budgets(self):
        budget = ContextBudget(max_files=4, max_chars_per_file=5, max_total_chars=8)
        first = build_context_file("core/main.py", "abcdefghij", 10, False, used_chars=0, budget=budget)
        second = build_context_file("core/agents/planner.py", "abcdefghij", 10, False, used_chars=len(first.content), budget=budget)
        third = build_context_file("core/capabilities/executor.py", "abc", 3, False, used_chars=context_budget_used([first, second]), budget=budget)

        self.assertEqual(first.content, "abcde")
        self.assertTrue(first.truncated)
        self.assertEqual(second.content, "abc")
        self.assertTrue(second.truncated)
        self.assertIsNone(third)
        self.assertEqual(context_budget_used([first, second]), 8)

    def test_does_not_include_excluded_paths(self):
        ranked = rank_research_files("repository architecture", [
            "core/main.py",
            "core/.venv/lib/site.py",
            "target/debug/build.rs",
        ])
        selected = select_ranked_files(ranked)
        self.assertEqual(selected, ["core/main.py"])
        self.assertIsNone(build_context_file("node_modules/pkg/index.js", "x", 1, False, 0))


if __name__ == "__main__":
    unittest.main()
