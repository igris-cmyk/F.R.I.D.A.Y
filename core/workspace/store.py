import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.workspace.models import (
    CURRENT_SCHEMA_VERSION,
    ImportRecord,
    SymbolRecord,
    TextMatchRecord,
    WorkspaceFileRecord,
)


BASE_SCHEMA_NAME = "base_workspace_schema"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkspaceIndexStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()
        self.schema_version: int | None = None
        self.target_schema_version = CURRENT_SCHEMA_VERSION

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            self.schema_version = ensure_schema(conn, self.target_schema_version)
        finally:
            conn.close()

    def replace_index(self, workspace_root: str, files: list[WorkspaceFileRecord], errors: list[dict[str, str]]) -> None:
        run_id = utc_now()
        conn = self._connect()
        try:
            with conn:
                conn.execute("DELETE FROM file_terms;")
                conn.execute("DELETE FROM file_capabilities;")
                conn.execute("DELETE FROM file_nats_subjects;")
                conn.execute("DELETE FROM file_imports;")
                conn.execute("DELETE FROM file_symbols;")
                conn.execute("DELETE FROM file_roles;")
                conn.execute("DELETE FROM workspace_files;")
                conn.execute("DELETE FROM index_errors;")

                conn.execute(
                    """
                    INSERT INTO index_runs (
                        id, created_at, workspace_root, file_count, error_count
                    ) VALUES (?, ?, ?, ?, ?);
                    """,
                    (run_id, run_id, workspace_root, len(files), len(errors)),
                )
                for error in errors:
                    conn.execute(
                        """
                        INSERT INTO index_errors (run_id, path, error)
                        VALUES (?, ?, ?);
                        """,
                        (run_id, error.get("path"), error.get("error")),
                    )
                for record in files:
                    self._insert_file(conn, record)
        finally:
            conn.close()

    def status(self) -> dict[str, Any]:
        conn = self._connect()
        try:
            schema_version = get_schema_version(conn)
            latest_run = conn.execute(
                """
                SELECT id, created_at, workspace_root, file_count, error_count
                FROM index_runs
                ORDER BY created_at DESC
                LIMIT 1;
                """
            ).fetchone()
            tables = {
                "file_count": "workspace_files",
                "role_count": "file_roles",
                "symbol_count": "file_symbols",
                "import_count": "file_imports",
                "nats_subject_count": "file_nats_subjects",
                "capability_count": "file_capabilities",
                "term_count": "file_terms",
            }
            counts = {
                key: int(conn.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0])
                for key, table in tables.items()
            }
        finally:
            conn.close()

        return {
            "db_path": str(self.db_path),
            "schema_version": schema_version,
            "target_schema_version": self.target_schema_version,
            "migration_status": "ok" if schema_version == self.target_schema_version else "out_of_date",
            "latest_run": dict(latest_run) if latest_run else None,
            **counts,
        }

    def list_files(self) -> list[WorkspaceFileRecord]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT path, size, mtime, sha256, extension, language, summary
                FROM workspace_files
                ORDER BY path;
                """
            ).fetchall()
            return [self._record_from_row(conn, row) for row in rows]
        finally:
            conn.close()

    def get_file(self, path: str) -> WorkspaceFileRecord | None:
        normalized = path.replace("\\", "/").strip("/")
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT path, size, mtime, sha256, extension, language, summary
                FROM workspace_files
                WHERE path = ?;
                """,
                (normalized,),
            ).fetchone()
            if not row:
                return None
            return self._record_from_row(conn, row)
        finally:
            conn.close()

    def searchable_documents(self) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT path, summary, language
                FROM workspace_files
                ORDER BY path;
                """
            ).fetchall()
            docs = []
            for row in rows:
                path = row["path"]
                docs.append({
                    "path": path,
                    "summary": row["summary"],
                    "language": row["language"],
                    "roles": self._values(conn, "file_roles", path, "role"),
                    "symbols": self._values(conn, "file_symbols", path, "name"),
                    "imports": self._import_values(conn, path),
                    "nats_subjects": self._values(conn, "file_nats_subjects", path, "subject"),
                    "capabilities": self._values(conn, "file_capabilities", path, "capability_id"),
                    "terms": self._values(conn, "file_terms", path, "term"),
                })
            return docs
        finally:
            conn.close()

    def _insert_file(self, conn: sqlite3.Connection, record: WorkspaceFileRecord) -> None:
        conn.execute(
            """
            INSERT INTO workspace_files (
                path, size, mtime, sha256, extension, language, summary, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                record.path,
                record.size,
                record.mtime,
                record.sha256,
                record.extension,
                record.language,
                record.summary,
                utc_now(),
            ),
        )
        conn.executemany(
            "INSERT INTO file_roles (file_path, role) VALUES (?, ?);",
            [(record.path, role) for role in record.role_tags],
        )
        conn.executemany(
            """
            INSERT INTO file_symbols (file_path, name, kind, line)
            VALUES (?, ?, ?, ?);
            """,
            [(record.path, symbol.name, symbol.kind, symbol.line) for symbol in record.symbols],
        )
        conn.executemany(
            """
            INSERT INTO file_imports (file_path, module, name, line)
            VALUES (?, ?, ?, ?);
            """,
            [(record.path, item.module, item.name, item.line) for item in record.imports],
        )
        conn.executemany(
            """
            INSERT INTO file_nats_subjects (file_path, subject, line)
            VALUES (?, ?, ?);
            """,
            [(record.path, item.value, item.line) for item in record.nats_subjects],
        )
        conn.executemany(
            """
            INSERT INTO file_capabilities (file_path, capability_id, line)
            VALUES (?, ?, ?);
            """,
            [(record.path, item.value, item.line) for item in record.capabilities],
        )
        conn.executemany(
            "INSERT INTO file_terms (file_path, term) VALUES (?, ?);",
            [(record.path, term) for term in record.terms],
        )

    def _record_from_row(self, conn: sqlite3.Connection, row: sqlite3.Row) -> WorkspaceFileRecord:
        path = row["path"]
        return WorkspaceFileRecord(
            path=path,
            size=int(row["size"]),
            mtime=float(row["mtime"]),
            sha256=row["sha256"],
            extension=row["extension"],
            language=row["language"],
            summary=row["summary"],
            role_tags=self._values(conn, "file_roles", path, "role"),
            symbols=[
                SymbolRecord(name=item["name"], kind=item["kind"], line=int(item["line"]))
                for item in self._rows(conn, "file_symbols", path, "line, name")
            ],
            imports=[
                ImportRecord(module=item["module"], name=item["name"], line=int(item["line"]))
                for item in self._rows(conn, "file_imports", path, "line, module, name")
            ],
            nats_subjects=[
                TextMatchRecord(value=item["subject"], line=int(item["line"]))
                for item in self._rows(conn, "file_nats_subjects", path, "line, subject")
            ],
            capabilities=[
                TextMatchRecord(value=item["capability_id"], line=int(item["line"]))
                for item in self._rows(conn, "file_capabilities", path, "line, capability_id")
            ],
            terms=self._values(conn, "file_terms", path, "term"),
        )

    def _rows(self, conn: sqlite3.Connection, table: str, path: str, order_by: str) -> list[sqlite3.Row]:
        return conn.execute(
            f"SELECT * FROM {table} WHERE file_path = ? ORDER BY {order_by};",
            (path,),
        ).fetchall()

    def _values(self, conn: sqlite3.Connection, table: str, path: str, column: str) -> list[str]:
        return [
            row[0]
            for row in conn.execute(
                f"SELECT {column} FROM {table} WHERE file_path = ? ORDER BY {column};",
                (path,),
            ).fetchall()
        ]

    def _import_values(self, conn: sqlite3.Connection, path: str) -> list[str]:
        rows = conn.execute(
            """
            SELECT module, name FROM file_imports
            WHERE file_path = ?
            ORDER BY module, name;
            """,
            (path,),
        ).fetchall()
        return [
            f"{row['module']}.{row['name']}".strip(".") if row["name"] else row["module"]
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn


def ensure_schema(conn: sqlite3.Connection, target_version: int = CURRENT_SCHEMA_VERSION) -> int:
    create_schema_migrations_table(conn)
    current = get_schema_version(conn)
    if current > target_version:
        raise RuntimeError(
            f"Workspace index schema is newer than this code supports: current={current} target={target_version}"
        )
    with conn:
        if current == 0:
            create_base_schema(conn)
            record_migration(conn, 1, BASE_SCHEMA_NAME)
            current = 1
    return current


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
        CREATE TABLE IF NOT EXISTS index_runs (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            workspace_root TEXT NOT NULL,
            file_count INTEGER NOT NULL,
            error_count INTEGER NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS index_errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            path TEXT,
            error TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS workspace_files (
            path TEXT PRIMARY KEY,
            size INTEGER NOT NULL,
            mtime REAL NOT NULL,
            sha256 TEXT NOT NULL,
            extension TEXT NOT NULL,
            language TEXT NOT NULL,
            summary TEXT NOT NULL,
            indexed_at TEXT NOT NULL
        );
        """
    )
    conn.execute("CREATE TABLE IF NOT EXISTS file_roles (file_path TEXT NOT NULL, role TEXT NOT NULL);")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS file_symbols (
            file_path TEXT NOT NULL,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            line INTEGER NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS file_imports (
            file_path TEXT NOT NULL,
            module TEXT NOT NULL,
            name TEXT,
            line INTEGER NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS file_nats_subjects (
            file_path TEXT NOT NULL,
            subject TEXT NOT NULL,
            line INTEGER NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS file_capabilities (
            file_path TEXT NOT NULL,
            capability_id TEXT NOT NULL,
            line INTEGER NOT NULL
        );
        """
    )
    conn.execute("CREATE TABLE IF NOT EXISTS file_terms (file_path TEXT NOT NULL, term TEXT NOT NULL);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_workspace_files_language ON workspace_files(language);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_file_roles_role ON file_roles(role);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_file_terms_term ON file_terms(term);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_file_symbols_name ON file_symbols(name);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_file_imports_module ON file_imports(module);")


def get_schema_version(conn: sqlite3.Connection) -> int:
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


def record_to_dict(record: WorkspaceFileRecord) -> dict[str, Any]:
    return {
        "path": record.path,
        "size": record.size,
        "mtime": record.mtime,
        "sha256": record.sha256,
        "extension": record.extension,
        "language": record.language,
        "summary": record.summary,
        "role_tags": record.role_tags,
        "symbols": [symbol.__dict__ for symbol in record.symbols],
        "imports": [item.__dict__ for item in record.imports],
        "nats_subjects": [item.__dict__ for item in record.nats_subjects],
        "capabilities": [item.__dict__ for item in record.capabilities],
        "terms": record.terms,
    }


def dumps_record(record: WorkspaceFileRecord) -> str:
    return json.dumps(record_to_dict(record), indent=2, sort_keys=True)
