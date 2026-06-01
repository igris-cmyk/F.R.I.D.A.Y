import asyncio
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from core.agents.memory_agent import MemoryAgent, RetrievalPolicy
from core.memory.migrations import apply_migrations, get_schema_version
from core.memory.manager import MemoryHealthState, MemoryImportance, MemoryManager, cosine_similarity
from core.memory import pipeline as memory_pipeline
from core.memory.redaction import redact_text
from core.memory.summarizer import summarize_completed_trace
from core.memory.sqlite_store import SQLiteMemoryStore
from core.tools.memory_debug import compact_memory_row, parse_policy


class TestSQLiteMemoryBackend(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "memory.sqlite3")
        self.manager = MemoryManager(db_path=self.db_path)
        self.manager._check_embedding_model_available = AsyncMock(return_value=False)
        await self.manager.initialize()

    async def asyncTearDown(self):
        await self.manager.close()
        self.tmp.cleanup()

    async def test_initialize_creates_schema_and_health_counts(self):
        self.assertTrue(Path(self.db_path).exists())
        health = await self.manager.health()
        self.assertEqual(health["backend"], "sqlite")
        self.assertEqual(health["requested_backend"], "sqlite")
        self.assertEqual(health["schema_version"], 1)
        self.assertEqual(health["target_schema_version"], 1)
        self.assertEqual(health["migration_status"], "ok")
        self.assertEqual(health["item_count"], 0)
        self.assertEqual(health["embedded_count"], 0)
        self.assertEqual(health["health_state"], MemoryHealthState.DEGRADED.value)

    async def test_fresh_db_creates_schema_migrations_and_base_tables(self):
        conn = sqlite3.connect(self.db_path)
        try:
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table';").fetchall()
            }
            version = get_schema_version(conn)
        finally:
            conn.close()

        self.assertIn("schema_migrations", tables)
        self.assertIn("memory_items", tables)
        self.assertIn("memory_events", tables)
        self.assertEqual(version, 1)

    async def test_postgres_requested_falls_back_to_sqlite_not_offline(self):
        manager = MemoryManager(
            db_path=str(Path(self.tmp.name) / "postgres-fallback.sqlite3"),
            backend="postgres",
        )
        manager._check_embedding_model_available = AsyncMock(return_value=False)

        with self.assertLogs("friday.memory.manager", level="WARNING") as logs:
            await manager.initialize()

        health = await manager.health()
        self.assertEqual(health["backend"], "sqlite")
        self.assertEqual(health["requested_backend"], "postgres")
        self.assertEqual(health["health_state"], MemoryHealthState.DEGRADED.value)
        self.assertNotEqual(manager.health_state, MemoryHealthState.OFFLINE)
        self.assertTrue(any("falling back to SQLite" in line for line in logs.output))

    async def test_persists_memory_item_with_embedding_json(self):
        with self.assertLogs("friday.memory.manager", level="INFO") as logs:
            result = await self.manager.persist_episodic_trace(
                trace_id="trace-1",
                intent="research",
                importance=MemoryImportance.SEMANTIC,
                workflow_summary="Analyzed memory subsystem architecture.",
                environment_context={"working_directory": "/repo"},
                metadata={"command": "explain memory subsystem"},
                embedding=[1.0, 0.0, 0.0],
            )

        self.assertTrue(result["persisted"])
        self.assertTrue(result["embedded"])
        self.assertTrue(any("Persisted memory item" in line for line in logs.output))
        health = await self.manager.health()
        self.assertEqual(health["item_count"], 1)
        self.assertEqual(health["embedded_count"], 1)
        row = self.manager.store.list_memories()[0]
        self.assertEqual(json.loads(row["embedding_json"]), [1.0, 0.0, 0.0])

    async def test_keyword_fallback_works_without_embeddings(self):
        await self.manager.persist_episodic_trace(
            trace_id="trace-memory",
            intent="research",
            importance=MemoryImportance.SEMANTIC,
            workflow_summary="Memory subsystem uses SQLite persistence and local embeddings.",
            environment_context={},
            metadata={"command": "explain memory subsystem"},
        )

        results = await self.manager.retrieve_relevant_context("memory embeddings", limit=3)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["trace_id"], "trace-memory")

    async def test_repository_architecture_memory_found_when_embedded_count_zero(self):
        self.manager.embedding_available = True
        self.manager.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        await self.manager.persist_episodic_trace(
            trace_id="trace-architecture",
            intent="research",
            importance=MemoryImportance.SEMANTIC,
            workflow_summary="We inspected the repository architecture and orchestrator flow.",
            environment_context={},
            metadata={
                "command": "analyze repository architecture",
                "title": "Repository architecture inspection",
                "key_files": ["core/main.py", "core/agents/planner.py"],
            },
        )
        row = self.manager.store.list_memories()[0]
        row["embedding_json"] = None
        self.manager.store.insert_memory(row)

        retrieval = await self.manager.retrieve_relevant_context_with_diagnostics(
            "repository architecture",
            limit=5,
            min_score=0.05,
        )

        self.assertEqual(retrieval["results"][0]["trace_id"], "trace-architecture")
        self.assertEqual(retrieval["diagnostics"]["embedded_count"], 0)
        self.assertEqual(retrieval["diagnostics"]["retrieval_mode"], "keyword_fallback")

    async def test_cosine_similarity_ranks_relevant_memory_higher(self):
        await self.manager.persist_episodic_trace(
            trace_id="trace-memory",
            intent="research",
            importance=MemoryImportance.SEMANTIC,
            workflow_summary="Memory subsystem details.",
            environment_context={},
            metadata={"command": "memory"},
            embedding=[1.0, 0.0],
        )
        await self.manager.persist_episodic_trace(
            trace_id="trace-frontend",
            intent="research",
            importance=MemoryImportance.SEMANTIC,
            workflow_summary="Frontend details.",
            environment_context={},
            metadata={"command": "frontend"},
            embedding=[0.0, 1.0],
        )

        results = await self.manager.retrieve_relevant_context([0.9, 0.1], limit=2, min_score=0.0)
        self.assertEqual(results[0]["trace_id"], "trace-memory")
        self.assertGreater(results[0]["score"], results[1]["score"])

    async def test_redaction_removes_secrets_before_storage(self):
        await self.manager.persist_episodic_trace(
            trace_id="trace-secret",
            intent="terminal",
            importance=MemoryImportance.EPISODIC,
            workflow_summary="password=supersecret Authorization: Bearer abcdefghijklmnopqrstuvwxyz1234567890TOKEN",
            environment_context={},
            metadata={"result_preview": "api_key=secretvalue"},
        )

        row = self.manager.store.list_memories()[0]
        self.assertNotIn("supersecret", row["summary"])
        self.assertNotIn("secretvalue", row["content_preview"])
        self.assertIn("[REDACTED]", row["summary"])

    async def test_embedding_failure_still_persists_without_embedding(self):
        self.manager.embedding_available = True
        self.manager.generate_embedding = AsyncMock(return_value=None)

        result = await self.manager.persist_episodic_trace(
            trace_id="trace-no-embed",
            intent="research",
            importance=MemoryImportance.SEMANTIC,
            workflow_summary="Repository architecture summary.",
            environment_context={},
            metadata={"command": "analyze repository architecture"},
        )

        self.assertTrue(result["persisted"])
        self.assertFalse(result["embedded"])
        health = await self.manager.health()
        self.assertEqual(health["item_count"], 1)
        self.assertEqual(health["embedded_count"], 0)

    async def test_embedding_available_stores_embedding_when_generated(self):
        self.manager.embedding_available = True
        self.manager.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])

        result = await self.manager.persist_episodic_trace(
            trace_id="trace-embedded",
            intent="research",
            importance=MemoryImportance.SEMANTIC,
            workflow_summary="Memory subsystem architecture.",
            environment_context={},
            metadata={"command": "explain memory subsystem"},
        )

        self.assertTrue(result["persisted"])
        self.assertTrue(result["embedded"])
        health = await self.manager.health()
        self.assertEqual(health["embedded_count"], 1)

    async def test_embedding_timeout_falls_back_to_keyword_retrieval(self):
        self.manager.embedding_available = True
        self.manager.generate_embedding = AsyncMock(return_value=None)
        await self.manager.persist_episodic_trace(
            trace_id="trace-fallback",
            intent="research",
            importance=MemoryImportance.SEMANTIC,
            workflow_summary="We inspected the memory subsystem and repository architecture.",
            environment_context={},
            metadata={
                "command": "explain memory subsystem",
                "title": "Memory subsystem inspection",
                "key_files": ["core/memory/manager.py"],
            },
            embedding=[0.9, 0.1],
        )

        retrieval = await self.manager.retrieve_relevant_context_with_diagnostics(
            "memory subsystem",
            limit=5,
            min_score=0.05,
        )

        self.assertEqual(retrieval["results"][0]["trace_id"], "trace-fallback")
        self.assertEqual(retrieval["diagnostics"]["retrieval_mode"], "keyword_fallback")
        self.assertTrue(retrieval["diagnostics"]["embedding_attempted"])
        self.assertTrue(retrieval["diagnostics"]["embedding_failed"])

    async def test_empty_db_returns_no_relevant_memory(self):
        results = await self.manager.retrieve_relevant_context("planner timeout")
        self.assertEqual(results, [])

    async def test_health_sees_item_count_after_runtime_insert(self):
        result = await self.manager.persist_episodic_trace(
            trace_id="trace-runtime",
            intent="research",
            importance=MemoryImportance.SEMANTIC,
            workflow_summary="Runtime inserted memory item.",
            environment_context={},
            metadata={"command": "runtime insert"},
        )

        self.assertTrue(result["persisted"])
        health = await self.manager.health()
        self.assertEqual(health["backend"], "sqlite")
        self.assertEqual(health["item_count"], 1)


class TestSQLiteMemoryMigrations(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "memory.sqlite3"

    def tearDown(self):
        self.tmp.cleanup()

    def _create_legacy_base_schema(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE memory_items (
                    id TEXT PRIMARY KEY,
                    trace_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    importance TEXT NOT NULL,
                    workspace_root TEXT,
                    project_scope TEXT,
                    source_component TEXT,
                    user_intent TEXT,
                    summary TEXT NOT NULL,
                    content_preview TEXT,
                    metadata_json TEXT,
                    embedding_json TEXT,
                    embedding_model TEXT,
                    token_estimate INTEGER,
                    access_count INTEGER DEFAULT 0,
                    last_accessed_at TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE memory_events (
                    id TEXT PRIMARY KEY,
                    memory_id TEXT,
                    event_type TEXT,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT
                );
                """
            )
            conn.execute(
                """
                INSERT INTO memory_items (
                    id, trace_id, created_at, updated_at, memory_type, importance,
                    summary, metadata_json, token_estimate, access_count
                )
                VALUES (
                    'memory-1', 'trace-legacy', '2026-01-01T00:00:00+00:00',
                    '2026-01-01T00:00:00+00:00', 'EPISODIC', 'episodic',
                    'Legacy memory row', '{}', 3, 0
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    def test_existing_db_without_migration_table_is_adopted_preserving_rows(self):
        self._create_legacy_base_schema()

        store = SQLiteMemoryStore(str(self.db_path))
        store.initialize()

        conn = sqlite3.connect(self.db_path)
        try:
            version = get_schema_version(conn)
            count = conn.execute("SELECT COUNT(*) FROM memory_items;").fetchone()[0]
        finally:
            conn.close()

        self.assertEqual(version, 1)
        self.assertEqual(count, 1)
        self.assertEqual(store.migration_status, "ok")

    def test_ensure_schema_is_idempotent(self):
        self._create_legacy_base_schema()
        store = SQLiteMemoryStore(str(self.db_path))

        store.initialize()
        store.initialize()

        conn = sqlite3.connect(self.db_path)
        try:
            version = get_schema_version(conn)
            migration_records = conn.execute(
                "SELECT COUNT(*) FROM schema_migrations WHERE version = 1;"
            ).fetchone()[0]
            row_count = conn.execute("SELECT COUNT(*) FROM memory_items;").fetchone()[0]
        finally:
            conn.close()

        self.assertEqual(version, 1)
        self.assertEqual(migration_records, 1)
        self.assertEqual(row_count, 1)

    def test_newer_schema_version_is_unsupported_and_preserves_data(self):
        self._create_legacy_base_schema()
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE schema_migrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version INTEGER NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "INSERT INTO schema_migrations (version, name, applied_at) VALUES (999, 'future', '2026-01-01');"
            )
            conn.commit()
        finally:
            conn.close()

        store = SQLiteMemoryStore(str(self.db_path))
        with self.assertRaises(Exception):
            store.initialize()

        conn = sqlite3.connect(self.db_path)
        try:
            version = get_schema_version(conn)
            row_count = conn.execute("SELECT COUNT(*) FROM memory_items;").fetchone()[0]
        finally:
            conn.close()

        self.assertEqual(version, 999)
        self.assertEqual(row_count, 1)
        self.assertEqual(store.migration_status, "unsupported")

    def test_failed_migration_rolls_back_version(self):
        store = SQLiteMemoryStore(str(self.db_path))
        store.initialize()

        def failing_migration(conn):
            conn.execute("CREATE TABLE should_rollback (id TEXT);")
            raise RuntimeError("forced migration failure")

        conn = sqlite3.connect(self.db_path)
        try:
            with self.assertRaises(Exception):
                apply_migrations(
                    conn,
                    current_version=1,
                    target_version=2,
                    migrations={2: ("forced_failure", failing_migration)},
                )
            version = get_schema_version(conn)
            rollback_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='should_rollback';"
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(version, 1)
        self.assertIsNone(rollback_table)


class TestMemoryRedaction(unittest.TestCase):
    def test_redactor_removes_common_secret_shapes(self):
        redacted = redact_text(
            "token=abc123 password=hunter2 Authorization: Bearer secretbearer "
            "-----BEGIN PRIVATE KEY-----abc-----END PRIVATE KEY-----"
        )
        self.assertNotIn("hunter2", redacted)
        self.assertNotIn("secretbearer", redacted)
        self.assertIn("[REDACTED]", redacted)

    def test_cosine_similarity(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertEqual(cosine_similarity([1.0], [1.0, 0.0]), 0.0)


class TestDeterministicMemorySummarizer(unittest.TestCase):
    def test_research_trace_summary_is_human_readable(self):
        summary = summarize_completed_trace({
            "trace_id": "trace-arch",
            "intent": "research",
            "command": "analyze repository architecture",
            "result": "SUCCESS filesystem.search: {'files': ['core/main.py', 'core/agents/planner.py']}",
            "error_state": False,
            "metadata": {
                "selected_files": [
                    "core/main.py",
                    "core/agents/planner.py",
                    "core/capabilities/executor.py",
                    "core/security/permissions.py",
                ],
                "capabilities_used": ["filesystem.search", "filesystem.read", "research.synthesize"],
            },
        })

        self.assertEqual(summary.title, "Repository architecture inspection")
        self.assertIn("We inspected the repository architecture", summary.summary)
        self.assertIn("core/main.py", summary.key_files)
        self.assertNotIn("SUCCESS filesystem.search", summary.summary)

    def test_memory_subsystem_summary_extracts_key_memory_files(self):
        summary = summarize_completed_trace({
            "trace_id": "trace-memory",
            "intent": "research",
            "command": "explain memory subsystem",
            "result": (
                "SUCCESS filesystem.search: {'files': ['core/main.py', "
                "'core/memory/manager.py', 'core/memory/pipeline.py', "
                "'core/memory/retriever.py', 'core/agents/memory_agent.py']}"
            ),
            "error_state": False,
            "metadata": {},
        })

        self.assertEqual(summary.key_files[:4], [
            "core/memory/manager.py",
            "core/memory/pipeline.py",
            "core/memory/retriever.py",
            "core/agents/memory_agent.py",
        ])
        self.assertIn("SQLite persistence", summary.summary)

    def test_approval_workflow_summary_extracts_security_files(self):
        summary = summarize_completed_trace({
            "trace_id": "trace-approval",
            "intent": "research",
            "command": "show approval workflow",
            "result": (
                "SUCCESS filesystem.search: {'files': ['core/main.py', "
                "'core/security/permissions.py', 'core/security/approval.py', "
                "'core/schemas/events.py']}"
            ),
            "error_state": False,
            "metadata": {},
        })

        self.assertEqual(summary.title, "Approval workflow inspection")
        self.assertIn("core/security/approval.py", summary.key_files)
        self.assertIn("SecurityPolicy", summary.summary)

    def test_security_denial_summary_is_safety_oriented(self):
        summary = summarize_completed_trace({
            "trace_id": "trace-denial",
            "intent": "terminal",
            "command": "delete everything in this folder",
            "result": "FAILED shell.execute: explicitly blocked by SecurityPolicy",
            "error_state": True,
            "metadata": {},
        })

        self.assertEqual(summary.title, "Blocked destructive request")
        self.assertIn("blocked", summary.summary.lower())
        self.assertIn("SecurityPolicy", summary.summary)
        self.assertNotIn("SUCCESS filesystem.search", summary.summary)

    def test_summary_respects_limits_and_redaction(self):
        summary = summarize_completed_trace({
            "trace_id": "trace-secret",
            "intent": "research",
            "command": "explain memory subsystem",
            "result": "token=supersecret " + ("SUCCESS filesystem.search " * 200),
            "error_state": False,
            "metadata": {},
        })

        self.assertLessEqual(len(summary.summary), 1200 + len("\n...[TRUNCATED]...\n"))
        self.assertNotIn("supersecret", summary.summary)
        self.assertNotIn("compression_timeout", summary.summary)


class TestMemoryAgentBackend(unittest.IsolatedAsyncioTestCase):
    async def test_does_not_hallucinate_when_no_memory_exists(self):
        agent = MemoryAgent()
        original = agent._reconstruct_narrative
        try:
            import core.agents.memory_agent as memory_agent_module

            old_manager = memory_agent_module.memory_manager
            manager = MemoryManager(db_path=str(Path(tempfile.mkdtemp()) / "empty.sqlite3"))
            manager._check_embedding_model_available = AsyncMock(return_value=False)
            await manager.initialize()
            memory_agent_module.memory_manager = manager
            result = await agent.recall_context("missing memory", RetrievalPolicy.DEEP_RESEARCH)
            self.assertIsNone(result)
        finally:
            agent._reconstruct_narrative = original
            memory_agent_module.memory_manager = old_manager

    async def test_uses_retrieved_memory_when_available(self):
        agent = MemoryAgent()
        agent._reconstruct_narrative = AsyncMock(return_value="Recovered memory context.")
        import core.agents.memory_agent as memory_agent_module

        old_manager = memory_agent_module.memory_manager
        try:
            manager = MemoryManager(db_path=str(Path(tempfile.mkdtemp()) / "agent.sqlite3"))
            manager._check_embedding_model_available = AsyncMock(return_value=False)
            await manager.initialize()
            await manager.persist_episodic_trace(
                trace_id="trace-agent",
                intent="research",
                importance=MemoryImportance.SEMANTIC,
                workflow_summary="Memory subsystem persistence was implemented.",
                environment_context={},
                metadata={
                    "command": "explain memory subsystem",
                    "title": "Memory subsystem inspection",
                    "key_files": ["core/memory/manager.py"],
                },
            )
            memory_agent_module.memory_manager = manager
            result = await agent.recall_context("memory subsystem persistence", RetrievalPolicy.DEEP_RESEARCH)
        finally:
            memory_agent_module.memory_manager = old_manager

        self.assertIsNotNone(result)
        self.assertEqual(result["narrative"], "Recovered memory context.")
        self.assertEqual(result["lineage"]["source_trace_ids"], ["trace-agent"])

    async def test_natural_recall_returns_stored_matching_memory(self):
        agent = MemoryAgent()
        agent._reconstruct_narrative = AsyncMock(return_value="We inspected the memory subsystem files.")
        import core.agents.memory_agent as memory_agent_module

        old_manager = memory_agent_module.memory_manager
        try:
            manager = MemoryManager(db_path=str(Path(tempfile.mkdtemp()) / "natural.sqlite3"))
            manager._check_embedding_model_available = AsyncMock(return_value=False)
            await manager.initialize()
            await manager.persist_episodic_trace(
                trace_id="trace-natural",
                intent="research",
                importance=MemoryImportance.SEMANTIC,
                workflow_summary=(
                    "Explained memory subsystem using core/memory/manager.py, "
                    "core/memory/pipeline.py, core/memory/retriever.py, and "
                    "core/agents/memory_agent.py."
                ),
                environment_context={},
                metadata={
                    "command": "explain memory subsystem",
                    "title": "Memory subsystem inspection",
                    "key_files": ["core/memory/manager.py", "core/memory/pipeline.py"],
                },
            )
            memory_agent_module.memory_manager = manager
            result = await agent.recall_context(
                "what did we just inspect about memory?",
                RetrievalPolicy.PROJECT_RECALL,
            )
        finally:
            memory_agent_module.memory_manager = old_manager

        self.assertIsNotNone(result)
        self.assertEqual(result["narrative"], "We inspected the memory subsystem files.")
        self.assertEqual(result["lineage"]["source_trace_ids"], ["trace-natural"])

    async def test_recall_fallback_uses_clean_summaries(self):
        agent = MemoryAgent()
        output = agent._deterministic_recall_summary([
            {
                "title": "Memory subsystem inspection",
                "summary": "We inspected the memory subsystem.",
                "workflow_summary": "We inspected the memory subsystem.",
                "key_files": ["core/memory/manager.py", "core/memory/pipeline.py"],
                "trace_id": "trace-memory",
            }
        ], reason="reconstruction_timeout")

        self.assertIn("Relevant continuity found.", output)
        self.assertIn("Memory subsystem inspection", output)
        self.assertIn("core/memory/manager.py", output)
        self.assertNotIn("reconstruction_timeout", output)
        self.assertNotIn("SUCCESS filesystem.search", output)

    async def test_search_candidates_matches_project_recall_retrieval(self):
        agent = MemoryAgent()
        import core.agents.memory_agent as memory_agent_module

        old_manager = memory_agent_module.memory_manager
        try:
            manager = MemoryManager(db_path=str(Path(tempfile.mkdtemp()) / "search.sqlite3"))
            manager._check_embedding_model_available = AsyncMock(return_value=False)
            await manager.initialize()
            await manager.persist_episodic_trace(
                trace_id="trace-search",
                intent="research",
                importance=MemoryImportance.SEMANTIC,
                workflow_summary="We inspected the memory subsystem and repository architecture.",
                environment_context={},
                metadata={
                    "command": "explain memory subsystem",
                    "title": "Memory subsystem inspection",
                    "key_files": ["core/memory/manager.py", "core/memory/pipeline.py"],
                },
            )
            memory_agent_module.memory_manager = manager

            recall = await agent.recall_context(
                "what did we just inspect about memory?",
                RetrievalPolicy.PROJECT_RECALL,
            )
            search = await agent.search_candidates(
                "memory subsystem",
                RetrievalPolicy.PROJECT_RECALL,
            )
        finally:
            memory_agent_module.memory_manager = old_manager

        self.assertIsNotNone(recall)
        self.assertEqual(len(search["candidates"]), 1)
        self.assertEqual(search["candidates"][0]["trace_id"], "trace-search")
        self.assertEqual(search["diagnostics"]["policy"], RetrievalPolicy.PROJECT_RECALL.value)
        self.assertEqual(search["diagnostics"]["retrieval_mode"], "keyword_fallback")

    async def test_search_candidates_reports_hybrid_fallback_when_embedding_yields_no_matches(self):
        agent = MemoryAgent()
        import core.agents.memory_agent as memory_agent_module

        old_manager = memory_agent_module.memory_manager
        try:
            manager = MemoryManager(db_path=str(Path(tempfile.mkdtemp()) / "hybrid.sqlite3"))
            manager._check_embedding_model_available = AsyncMock(return_value=True)
            await manager.initialize()
            manager.embedding_available = True
            manager.generate_embedding = AsyncMock(return_value=[1.0, 0.0])
            await manager.persist_episodic_trace(
                trace_id="trace-hybrid",
                intent="research",
                importance=MemoryImportance.SEMANTIC,
                workflow_summary="We inspected the repository architecture.",
                environment_context={},
                metadata={
                    "command": "analyze repository architecture",
                    "title": "Repository architecture inspection",
                    "key_files": ["core/main.py"],
                },
                embedding=[0.0, 1.0],
            )
            memory_agent_module.memory_manager = manager
            result = await agent.search_candidates("repository architecture", RetrievalPolicy.PROJECT_RECALL)
        finally:
            memory_agent_module.memory_manager = old_manager

        self.assertEqual(result["candidates"][0]["trace_id"], "trace-hybrid")
        self.assertEqual(result["diagnostics"]["retrieval_mode"], "hybrid_fallback")

    async def test_respects_confidence_threshold(self):
        agent = MemoryAgent()
        import core.agents.memory_agent as memory_agent_module

        old_manager = memory_agent_module.memory_manager
        try:
            manager = MemoryManager(db_path=str(Path(tempfile.mkdtemp()) / "threshold.sqlite3"))
            manager._check_embedding_model_available = AsyncMock(return_value=False)
            await manager.initialize()
            await manager.persist_episodic_trace(
                trace_id="trace-weak",
                intent="research",
                importance=MemoryImportance.SEMANTIC,
                workflow_summary="Only one planner term appears here.",
                environment_context={},
                metadata={"command": "planner"},
            )
            memory_agent_module.memory_manager = manager
            result = await agent.recall_context("planner timeout memory architecture", RetrievalPolicy.DEEP_RESEARCH)
        finally:
            memory_agent_module.memory_manager = old_manager

        self.assertIsNone(result)

    async def test_search_candidates_empty_db_returns_empty_list(self):
        agent = MemoryAgent()
        import core.agents.memory_agent as memory_agent_module

        old_manager = memory_agent_module.memory_manager
        try:
            manager = MemoryManager(db_path=str(Path(tempfile.mkdtemp()) / "empty-search.sqlite3"))
            manager._check_embedding_model_available = AsyncMock(return_value=False)
            await manager.initialize()
            memory_agent_module.memory_manager = manager
            result = await agent.search_candidates("memory subsystem", RetrievalPolicy.PROJECT_RECALL)
        finally:
            memory_agent_module.memory_manager = old_manager

        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["diagnostics"]["item_count"], 0)


class TestCompletedTracePersistence(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.manager = MemoryManager(db_path=str(Path(self.tmp.name) / "pipeline.sqlite3"))
        self.manager._check_embedding_model_available = AsyncMock(return_value=False)
        await self.manager.initialize()
        self.old_manager = memory_pipeline.memory_manager
        self.old_compress = memory_pipeline.compress_workflow
        self.old_generate_embedding = memory_pipeline.generate_embedding
        memory_pipeline.memory_manager = self.manager
        memory_pipeline.compress_workflow = AsyncMock(return_value="This should not be the primary summary.")
        memory_pipeline.generate_embedding = AsyncMock(return_value=None)

    async def asyncTearDown(self):
        memory_pipeline.memory_manager = self.old_manager
        memory_pipeline.compress_workflow = self.old_compress
        memory_pipeline.generate_embedding = self.old_generate_embedding
        self.tmp.cleanup()

    async def test_completed_meaningful_trace_persists_memory_item(self):
        result = await memory_pipeline.process_completed_trace(
            trace_id="trace-meaningful",
            intent="research",
            command="analyze repository architecture",
            result=(
                "SUCCESS filesystem.search: {'files': ['core/main.py']}\n"
                "SUCCESS filesystem.read: bound 1 files for synthesis\n"
                "SUCCESS research.synthesize: grounded architecture summary"
            ),
            error_state=False,
            environment={"working_directory": "/repo"},
            metadata={"execution_ms": 123},
        )

        self.assertTrue(result["persisted"])
        rows = self.manager.store.list_memories()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["user_intent"], "analyze repository architecture")
        self.assertIn("repository architecture", rows[0]["summary"])
        self.assertNotIn("SUCCESS filesystem.search", rows[0]["summary"])
        metadata = json.loads(rows[0]["metadata_json"])
        self.assertEqual(metadata["title"], "Repository architecture inspection")
        self.assertIn("filesystem.search", metadata["capabilities_used"])

    async def test_trivial_conversation_does_not_persist(self):
        result = await memory_pipeline.process_completed_trace(
            trace_id="trace-hello",
            intent="conversation",
            command="hello friday",
            result="[Conversational Response] Acknowledged: hello friday",
            error_state=False,
            environment={},
            metadata={},
        )

        self.assertFalse(result["persisted"])
        self.assertEqual(result["degraded_reason"], "transient")
        self.assertEqual(self.manager.store.counts()["item_count"], 0)

    async def test_security_denial_persists_as_important_event(self):
        result = await memory_pipeline.process_completed_trace(
            trace_id="trace-denial",
            intent="terminal",
            command="delete everything in this folder",
            result="FAILED shell.execute: blocked by SecurityPolicy",
            error_state=True,
            environment={},
            metadata={"execution_ms": 20},
        )

        self.assertTrue(result["persisted"])
        row = self.manager.store.list_memories()[0]
        self.assertEqual(row["importance"], MemoryImportance.EPISODIC.value)
        self.assertEqual(row["user_intent"], "delete everything in this folder")
        self.assertIn("blocked", row["summary"].lower())

    async def test_memory_manager_initialized_before_persist(self):
        self.manager.health_state = MemoryHealthState.OFFLINE
        self.manager.initialize = AsyncMock(side_effect=self.manager.initialize)

        result = await memory_pipeline.process_completed_trace(
            trace_id="trace-init",
            intent="research",
            command="show approval workflow",
            result="SUCCESS research.synthesize: approval workflow",
            error_state=False,
            environment={},
            metadata={},
        )

        self.assertTrue(result["persisted"])
        self.manager.initialize.assert_awaited_once()

    async def test_persistence_exceptions_do_not_crash_pipeline(self):
        self.manager.persist_episodic_trace = AsyncMock(side_effect=RuntimeError("db write failed"))

        result = await memory_pipeline.process_completed_trace(
            trace_id="trace-error",
            intent="research",
            command="explain memory subsystem",
            result="SUCCESS research.synthesize: memory subsystem",
            error_state=False,
            environment={},
            metadata={},
        )

        self.assertFalse(result["persisted"])
        self.assertIn("db write failed", result["degraded_reason"])

    async def test_pipeline_does_not_report_persisted_for_queue_only_fallback(self):
        self.manager.persist_episodic_trace = AsyncMock(
            return_value={
                "persisted": False,
                "embedded": False,
                "degraded_reason": "queued_retry",
                "memory_id": None,
            }
        )

        with self.assertLogs("friday.memory.pipeline", level="INFO") as logs:
            result = await memory_pipeline.process_completed_trace(
                trace_id="trace-queued",
                intent="research",
                command="analyze repository architecture",
                result="SUCCESS research.synthesize: architecture",
                error_state=False,
                environment={},
                metadata={},
            )

        self.assertFalse(result["persisted"])
        self.assertIn("queued_retry", result["degraded_reason"])
        self.assertFalse(any("Persisted memory item" in line for line in logs.output))

    async def test_llm_compression_failure_does_not_degrade_deterministic_summary(self):
        memory_pipeline.compress_workflow = AsyncMock(side_effect=RuntimeError("ollama broke"))

        result = await memory_pipeline.process_completed_trace(
            trace_id="trace-compress-fail",
            intent="research",
            command="explain memory subsystem",
            result="SUCCESS research.synthesize: memory subsystem",
            error_state=False,
            environment={},
            metadata={},
        )

        self.assertTrue(result["persisted"])
        row = self.manager.store.list_memories()[0]
        self.assertIn("memory subsystem", row["summary"])
        self.assertNotIn("ollama broke", row["summary"])
        self.assertNotIn("compression_timeout", row["summary"])


class TestMemoryDebugFormatting(unittest.TestCase):
    def test_compact_memory_row_hides_raw_metadata_by_default(self):
        row = {
            "trace_id": "trace-memory",
            "summary": "We inspected the memory subsystem.",
            "metadata_json": json.dumps({
                "title": "Memory subsystem inspection",
                "key_files": ["core/memory/manager.py"],
                "result_preview": "SUCCESS filesystem.search: raw dump",
            }),
            "created_at": "2026-01-01T00:00:00+00:00",
            "memory_type": "SEMANTIC",
            "importance": "semantic",
        }

        compact = compact_memory_row(row)

        self.assertEqual(compact["title"], "Memory subsystem inspection")
        self.assertEqual(compact["key_files"], ["core/memory/manager.py"])
        self.assertNotIn("metadata", compact)

    def test_compact_memory_row_verbose_keeps_metadata(self):
        row = {
            "trace_id": "trace-memory",
            "summary": "We inspected the memory subsystem.",
            "metadata_json": json.dumps({"title": "Memory subsystem inspection"}),
            "content_preview": "Title: Memory subsystem inspection",
        }

        compact = compact_memory_row(row, verbose=True)

        self.assertIn("metadata", compact)
        self.assertIn("content_preview", compact)

    def test_parse_policy_accepts_default_alias(self):
        self.assertEqual(parse_policy("default"), RetrievalPolicy.PROJECT_RECALL)


if __name__ == "__main__":
    unittest.main()
