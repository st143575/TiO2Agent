"""SQLite storage for extracted facts (table ``facts``, same database file).

Facts are stored as JSON with the columns needed for filtering pulled out, so the
Agent's tools can ask "all 温度 facts of 实施例1" without loading every document.
"""

import sqlite3
import threading
from pathlib import Path

from tio2agent.domain import Fact, FactKind

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id            TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL,
    ord           INTEGER NOT NULL,
    kind          TEXT NOT NULL,
    subject       TEXT NOT NULL,
    subject_label TEXT NOT NULL,
    name          TEXT NOT NULL,
    name_raw      TEXT,
    step          TEXT,
    data          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_doc ON facts(doc_id, ord);
CREATE INDEX IF NOT EXISTS idx_facts_name ON facts(name);
"""


class FactRepository:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._lock = threading.RLock()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- writes ----------------------------------------------------------------

    def replace_document(self, doc_id: str, facts: list[Fact]) -> None:
        """Atomically replace all facts of a document."""
        rows = [
            (f.id, doc_id, i, f.kind.value, f.subject, f.subject_label, f.name, f.name_raw, f.step,
             f.model_dump_json())
            for i, f in enumerate(facts)
        ]
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.execute("DELETE FROM facts WHERE doc_id = ?", (doc_id,))
                self._conn.executemany(
                    "INSERT INTO facts (id, doc_id, ord, kind, subject, subject_label, name, name_raw, step, data)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            return self._conn.execute("DELETE FROM facts WHERE doc_id = ?", (doc_id,)).rowcount

    # --- reads -----------------------------------------------------------------

    def list_by_doc(self, doc_id: str) -> list[Fact]:
        return self.query(doc_ids=[doc_id])

    def query(
        self,
        *,
        doc_ids: list[str] | None = None,
        kind: FactKind | None = None,
        subject: str | None = None,
        name: str | None = None,
        step: str | None = None,
        limit: int | None = None,
    ) -> list[Fact]:
        """Filter facts. ``subject`` matches the id or the label ("E1" / "实施例1" /
        "claim:1" / "权利要求1"); ``name`` is a substring of the canonical or written name."""
        where, params = [], []
        if doc_ids is not None:
            if not doc_ids:
                return []
            where.append(f"doc_id IN ({','.join('?' * len(doc_ids))})")
            params += doc_ids
        if kind is not None:
            where.append("kind = ?")
            params.append(FactKind(kind).value)
        if subject:
            where.append("(subject = ? OR subject_label = ?)")
            params += [subject, subject]
        if name:
            where.append("(instr(name, ?) > 0 OR instr(coalesce(name_raw, ''), ?) > 0)")
            params += [name, name]
        if step:
            where.append("step = ?")
            params.append(step)
        sql = "SELECT data FROM facts"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY doc_id, ord"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [Fact.model_validate_json(r[0]) for r in rows]

    def vocabulary(self, doc_ids: list[str] | None = None, kind: FactKind | None = None) -> dict[str, list[str]]:
        """Distinct subjects and names, in document order: what a query can match."""
        where, params = [], []
        if doc_ids is not None:
            if not doc_ids:
                return {"subjects": [], "names": []}
            where.append(f"doc_id IN ({','.join('?' * len(doc_ids))})")
            params += doc_ids
        if kind is not None:
            where.append("kind = ?")
            params.append(FactKind(kind).value)
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        with self._lock:
            rows = self._conn.execute(f"SELECT subject_label, name FROM facts{clause} ORDER BY doc_id, ord", params).fetchall()
        return {
            "subjects": list(dict.fromkeys(r[0] for r in rows)),
            "names": list(dict.fromkeys(r[1] for r in rows)),
        }

    def count(self, doc_id: str | None = None) -> int:
        with self._lock:
            if doc_id is None:
                return self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            return self._conn.execute("SELECT COUNT(*) FROM facts WHERE doc_id = ?", (doc_id,)).fetchone()[0]
