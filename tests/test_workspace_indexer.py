import tempfile
import unittest
from pathlib import Path

from core.workspace.indexer import WorkspaceIndexer


class TestWorkspaceIndexer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._write("core/agents/planner.py", "class PlannerAgent:\n    def generate_plan(self):\n        return 'filesystem.search'\n")
        self._write("core/agents/router.py", "class RouterAgent:\n    pass\n")
        self._write("core/memory/manager.py", "class MemoryManager:\n    def retrieve_relevant_context(self):\n        pass\n")
        self._write("core/security/permissions.py", "class SecurityPolicy:\n    pass\n")
        self._write("core/main.py", "subject = f'friday.stream.{trace_id}'\ncommand = 'friday.intent.command'\n")
        self._write("apps/desktop/src/main.js", "nc.publish('friday.intent.command', payload)\n")
        self._write("tests/test_memory.py", "def test_memory():\n    pass\n")
        self._write(".env", "SECRET=value\n")
        self._write("node_modules/pkg/index.js", "secret\n")
        self.indexer = WorkspaceIndexer(self.root, db_path=self.root / ".friday/workspace/workspace_index.sqlite3")

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, relative_path: str, content: str) -> None:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def test_build_indexes_project_files_and_skips_excluded_paths(self):
        result = self.indexer.build()
        self.assertEqual(result["schema_version"], 1)
        status = self.indexer.status()
        self.assertGreaterEqual(status["file_count"], 7)

        indexed_paths = {record.path for record in self.indexer.store.list_files()}
        self.assertIn("core/agents/planner.py", indexed_paths)
        self.assertIn("core/main.py", indexed_paths)
        self.assertNotIn(".env", indexed_paths)
        self.assertNotIn("node_modules/pkg/index.js", indexed_paths)

    def test_search_finds_domain_files(self):
        self.indexer.build()
        self.assertEqual(self.indexer.search("planner")[0].path, "core/agents/planner.py")
        self.assertEqual(self.indexer.search("memory subsystem")[0].path, "core/memory/manager.py")
        self.assertEqual(self.indexer.search("security policy")[0].path, "core/security/permissions.py")

    def test_show_returns_static_extraction_facts(self):
        self.indexer.build()
        record = self.indexer.show("core/main.py")
        self.assertIsNotNone(record)
        self.assertIn("entry_point", record.role_tags)
        self.assertIn("nats_streaming", record.role_tags)
        self.assertIn("friday.intent.command", [item.value for item in record.nats_subjects])


if __name__ == "__main__":
    unittest.main()
