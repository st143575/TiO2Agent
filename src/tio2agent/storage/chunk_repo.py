"""SQLite storage for chunks and their embeddings.

Chunks live in the same database file as document records (table ``chunks``).
Embeddings are stored next to the chunk as float32 bytes, tagged with the model
that produced them, so changing the embedding model never mixes vector spaces.

``generation`` increases on every write; in-memory search indexes compare it to
decide when to rebuild.
"""

import sqlite3
import threading
from pathlib import Path

import numpy as np

from tio2agent.domain import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id              TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL,
    ord             INTEGER NOT NULL,
    data            TEXT NOT NULL,
    embedding       BLOB,
    embedding_model TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, ord);
"""


class ChunkRepository:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._lock = threading.RLock()
        self._generation = 0

    @property
    def generation(self) -> int:
        return self._generation

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- writes ----------------------------------------------------------------

    def replace_document(
        self,
        doc_id: str,
        chunks: list[Chunk],
        *,
        vectors: np.ndarray | None = None,
        model: str | None = None,
    ) -> None:
        """Atomically replace all chunks of a document (used by indexing and re-indexing)."""
        if vectors is not None and len(vectors) != len(chunks):
            raise ValueError("one vector per chunk is required")
        rows = []
        for i, chunk in enumerate(chunks):
            # 把高精度的 NumPy 数组强制转为 32 位浮点数（节省一半存储空间），然后压扁成原始的内存字节流（bytes），以便存入 SQLite 的 BLOB 列。
            blob = vectors[i].astype(np.float32).tobytes() if vectors is not None else None
            rows.append((chunk.id, doc_id, chunk.order, chunk.model_dump_json(), blob, model if blob else None))
        # 原子事务(Transactions)
        # 使用 BEGIN、COMMIT 和 ROLLBACK 将“删除旧块”和“插入新块”包裹在同一个事务中。
        # 如果在插入中途程序崩溃或断电，整个操作会回滚，绝对不会出现“一个文档只被存入了一半的 Chunk”这种脏数据状态。
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                self._conn.executemany(
                    "INSERT INTO chunks (id, doc_id, ord, data, embedding, embedding_model) VALUES (?, ?, ?, ?, ?, ?)",
                    rows,
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._generation += 1

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            removed = self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,)).rowcount
            self._generation += 1
            return removed

    # --- reads -----------------------------------------------------------------

    def list_by_doc(self, doc_id: str) -> list[Chunk]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM chunks WHERE doc_id = ? ORDER BY ord", (doc_id,)
            ).fetchall()
        return [Chunk.model_validate_json(r[0]) for r in rows]

    def all_chunks(self) -> list[Chunk]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM chunks ORDER BY doc_id, ord").fetchall()
        return [Chunk.model_validate_json(r[0]) for r in rows]

    def vectors(self, model: str) -> tuple[list[str], np.ndarray | None]:
        """Chunk ids and their embeddings (one row each) for ``model``."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, embedding FROM chunks WHERE embedding_model = ? ORDER BY doc_id, ord", (model,)
            ).fetchall()
        if not rows:
            return [], None
        return [r[0] for r in rows], np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])

    def count(self, doc_id: str | None = None) -> int:
        with self._lock:
            if doc_id is None:
                return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            return self._conn.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)).fetchone()[0]
