"""SQLite-backed repository for document records.

Each record is stored as a JSON blob, with the columns we filter or sort on
(sha256, status, created_at) kept alongside it. A single connection guarded by
a lock is enough here: writes are tiny and infrequent, and it keeps the
check-then-update operations (e.g. "re-index only if not busy") atomic.
"""

import sqlite3
import threading
from collections.abc import Collection
from enum import StrEnum
from pathlib import Path
from typing import Any

from tio2agent.domain import TERMINAL_STATUSES, DocumentRecord, DocumentStatus, utcnow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id         TEXT PRIMARY KEY,
    sha256     TEXT NOT NULL UNIQUE,
    status     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
"""


class DuplicateDocumentError(Exception):
    def __init__(self, existing_id: str) -> None:
        self.existing_id = existing_id
        super().__init__(existing_id)


class Outcome(StrEnum):
    OK = "ok"
    MISSING = "missing"
    BLOCKED = "blocked"


class DocumentRepository:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._lock = threading.RLock()

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()

    # --- reads -----------------------------------------------------------------

    def get(self, doc_id: str) -> DocumentRecord | None:
        with self._lock:
            return self._get(doc_id)

    def get_by_sha256(self, sha256: str) -> DocumentRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM documents WHERE sha256 = ?", (sha256,)
            ).fetchone()
        return DocumentRecord.model_validate_json(row[0]) if row else None

    def list_page(
        self,
        *,
        status: DocumentStatus | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[DocumentRecord], int]:
        where, params = ("WHERE status = ?", [status.value]) if status else ("", [])
        with self._lock:
            total = self._conn.execute(
                f"SELECT COUNT(*) FROM documents {where}", params
            ).fetchone()[0]
            rows = self._conn.execute(
                f"SELECT data FROM documents {where} "
                "ORDER BY created_at DESC, id LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return [DocumentRecord.model_validate_json(r[0]) for r in rows], total

    def list_unfinished(self) -> list[DocumentRecord]:
        placeholders = ",".join("?" for _ in TERMINAL_STATUSES)     # "?,?,?"
        with self._lock:
            rows = self._conn.execute(
                f"SELECT data FROM documents WHERE status NOT IN ({placeholders}) "
                "ORDER BY created_at",
                [s.value for s in TERMINAL_STATUSES],
            ).fetchall()
        return [DocumentRecord.model_validate_json(r[0]) for r in rows]

    # --- writes ----------------------------------------------------------------

    def insert(self, record: DocumentRecord) -> None:
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO documents (id, sha256, status, created_at, data) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        record.id,
                        record.sha256,
                        record.status.value,
                        record.created_at.isoformat(),
                        record.model_dump_json(),
                    ),
                )
            # 如果发现 sha256 已经存在，则抛出自定义的 DuplicateDocumentError，实现文档去重。
            except sqlite3.IntegrityError:
                row = self._conn.execute(
                    "SELECT id FROM documents WHERE sha256 = ?", (record.sha256,)
                ).fetchone()
                if row is None:
                    raise
                raise DuplicateDocumentError(row[0]) from None

    def update(self, doc_id: str, **changes: Any) -> DocumentRecord | None:
        """Apply field changes. Returns the new record, or None if it no longer exists."""
        with self._lock:
            record = self._get(doc_id)
            if record is None:
                return None
            return self._write(record, changes)

    def transition(
        self,
        doc_id: str,
        *,
        allowed_from: Collection[DocumentStatus],
        **changes: Any,
    ) -> tuple[Outcome, DocumentRecord | None]:
        """Apply changes only if the current status is in ``allowed_from``."""
        with self._lock:
            record = self._get(doc_id)
            if record is None:
                return Outcome.MISSING, None
            if record.status not in allowed_from:
                return Outcome.BLOCKED, record
            return Outcome.OK, self._write(record, changes)

    def delete(
        self, doc_id: str, *, allowed_from: Collection[DocumentStatus]
    ) -> tuple[Outcome, DocumentRecord | None]:
        with self._lock:
            record = self._get(doc_id)
            if record is None:
                return Outcome.MISSING, None
            if record.status not in allowed_from:
                return Outcome.BLOCKED, record
            self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            return Outcome.OK, record

    # --- internals (caller holds the lock) -------------------------------------

    def _get(self, doc_id: str) -> DocumentRecord | None:
        row = self._conn.execute("SELECT data FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return DocumentRecord.model_validate_json(row[0]) if row else None

    def _write(self, record: DocumentRecord, changes: dict[str, Any]) -> DocumentRecord:
        updated = DocumentRecord.model_validate(
            {**record.model_dump(), **changes, "updated_at": utcnow()}
        )
        self._conn.execute(
            "UPDATE documents SET status = ?, data = ? WHERE id = ?",
            (updated.status.value, updated.model_dump_json(), updated.id),
        )
        return updated
