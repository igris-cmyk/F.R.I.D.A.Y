import logging
import sqlite3
from datetime import datetime, timezone
from typing import Callable


logger = logging.getLogger("friday.memory.migrations")

CURRENT_SCHEMA_VERSION = 1
BASE_SCHEMA_NAME = "base_memory_schema"


class MigrationError(RuntimeError):
    """Base error for SQLite memory schema migration failures."""


class UnsupportedSchemaVersion(MigrationError):
    """Raised when the DB schema is newer than this code understands."""


MigrationFn = Callable[[sqlite3.Connection], None]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?;",
        (table_name,),
    ).fetchone()
    return row is not None


def has_base_tables(conn: sqlite3.Connection) -> bool:
    return table_exists(conn, "memory_items") and table_exists(conn, "memory_events")


def create_schema_migrations_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version INTEGER NOT NULL UNIQUE,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        );
        """
    )


def create_base_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_items (
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
        CREATE TABLE IF NOT EXISTS memory_events (
            id TEXT PRIMARY KEY,
            memory_id TEXT,
            event_type TEXT,
            created_at TEXT NOT NULL,
            metadata_json TEXT
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_items_trace_id ON memory_items(trace_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_items_created_at ON memory_items(created_at);")


def get_schema_version(conn: sqlite3.Connection) -> int:
    if not table_exists(conn, "schema_migrations"):
        return 0
    row = conn.execute("SELECT MAX(version) FROM schema_migrations;").fetchone()
    if not row or row[0] is None:
        return 0
    return int(row[0])


def record_migration(conn: sqlite3.Connection, version: int, name: str) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO schema_migrations (version, name, applied_at)
        VALUES (?, ?, ?);
        """,
        (version, name, utc_now()),
    )


def ensure_schema(
    conn: sqlite3.Connection,
    target_version: int = CURRENT_SCHEMA_VERSION,
    migrations: dict[int, tuple[str, MigrationFn]] | None = None,
) -> int:
    """Ensure the memory schema exists and is migrated to target_version.

    Future migrations can be added with:

    MIGRATIONS = {
        2: ("add_session_id_to_memory_items", add_session_id_to_memory_items),
    }
    """
    migrations = migrations or {}

    migration_table_exists = table_exists(conn, "schema_migrations")
    base_tables_exist = has_base_tables(conn)
    current_version = get_schema_version(conn) if migration_table_exists else 0

    if current_version > target_version:
        message = (
            f"Database schema version is newer than this code supports: "
            f"current={current_version} target={target_version}"
        )
        logger.warning("SQLiteMemoryStore: unsupported future schema version %s", message)
        raise UnsupportedSchemaVersion(message)

    try:
        with conn:
            if not migration_table_exists and not base_tables_exist:
                create_schema_migrations_table(conn)
                create_base_schema(conn)
                record_migration(conn, 1, BASE_SCHEMA_NAME)
                current_version = 1
                logger.info("SQLiteMemoryStore: initialized fresh schema version 1")
            elif not migration_table_exists and base_tables_exist:
                create_schema_migrations_table(conn)
                record_migration(conn, 1, BASE_SCHEMA_NAME)
                current_version = 1
                logger.info("SQLiteMemoryStore: adopted existing schema as version 1")
            elif migration_table_exists and current_version == 0 and base_tables_exist:
                record_migration(conn, 1, BASE_SCHEMA_NAME)
                current_version = 1
                logger.info("SQLiteMemoryStore: adopted existing schema as version 1")
            elif migration_table_exists and current_version == 0:
                create_base_schema(conn)
                record_migration(conn, 1, BASE_SCHEMA_NAME)
                current_version = 1
                logger.info("SQLiteMemoryStore: initialized fresh schema version 1")

            if current_version < target_version:
                apply_migrations(conn, current_version, target_version, migrations)
                current_version = get_schema_version(conn)

    except UnsupportedSchemaVersion:
        raise
    except Exception as exc:
        logger.exception("SQLiteMemoryStore: migration failed: %s", exc)
        raise MigrationError(str(exc)) from exc

    logger.info("SQLiteMemoryStore: schema version current=%s target=%s", current_version, target_version)
    return current_version


def apply_migrations(
    conn: sqlite3.Connection,
    current_version: int,
    target_version: int,
    migrations: dict[int, tuple[str, MigrationFn]],
) -> None:
    for version in range(current_version + 1, target_version + 1):
        if version not in migrations:
            raise MigrationError(f"No migration registered for schema version {version}.")
        name, migration = migrations[version]
        savepoint = f"memory_migration_{version}"
        try:
            conn.execute(f"SAVEPOINT {savepoint};")
            migration(conn)
            record_migration(conn, version, name)
            conn.execute(f"RELEASE SAVEPOINT {savepoint};")
        except Exception as exc:
            try:
                conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint};")
                conn.execute(f"RELEASE SAVEPOINT {savepoint};")
            except sqlite3.Error:
                pass
            logger.exception("SQLiteMemoryStore: migration failed version=%s name=%s", version, name)
            raise MigrationError(str(exc)) from exc
