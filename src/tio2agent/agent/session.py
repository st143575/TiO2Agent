"""Chat sessions: each turn's question and full response, in SQLite.

A session is created by its first turn. Later turns send the earlier questions and
answer texts to the LLM as context ("那实施例2呢？"); sources are not carried over,
the Agent queries the tools again.
"""

import json
import secrets
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from tio2agent.domain import utcnow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_sessions (
    id         TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_turns (
    session_id TEXT NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
    idx        INTEGER NOT NULL,
    question   TEXT NOT NULL,
    response   TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (session_id, idx)
);
"""


def new_session_id() -> str:
    return f"ses_{secrets.token_hex(6)}"


class TurnRecord(BaseModel):
    index: int
    question: str
    response: dict[str, Any]
    created_at: datetime

    @property
    def answer_text(self) -> str:
        return (self.response.get("answer") or {}).get("answer", "")


class SessionRecord(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime
    turns: list[TurnRecord]


class SessionStore:
    """
    把每一轮问答存进同一个 SQLite 文件，追问时会带上最近 4 轮问答作为上下文。
    """
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._lock = threading.RLock()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def exists(self, session_id: str) -> bool:
        with self._lock:
            return self._conn.execute("SELECT 1 FROM agent_sessions WHERE id = ?", (session_id,)).fetchone() is not None

    def append_turn(self, session_id: str, question: str, response: dict[str, Any]) -> int:
        """Store a turn (creating the session on its first turn). Returns the turn index (1-based)."""
        now = utcnow().isoformat()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT INTO agent_sessions (id, created_at, updated_at) VALUES (?, ?, ?)"
                    " ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at",
                    (session_id, now, now),
                )
                idx = self._conn.execute(
                    "SELECT COALESCE(MAX(idx), 0) + 1 FROM agent_turns WHERE session_id = ?", (session_id,)
                ).fetchone()[0]
                self._conn.execute(
                    "INSERT INTO agent_turns (session_id, idx, question, response, created_at) VALUES (?, ?, ?, ?, ?)",
                    (session_id, idx, question, json.dumps(response, ensure_ascii=False), now),
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return idx

    def history(self, session_id: str, limit: int) -> list[tuple[str, str]]:
        """The last ``limit`` turns as (question, answer text), oldest first."""
        if limit <= 0:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT question, response FROM agent_turns WHERE session_id = ? ORDER BY idx DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [(q, (json.loads(r).get("answer") or {}).get("answer", "")) for q, r in reversed(rows)]

    def get(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            head = self._conn.execute(
                "SELECT id, created_at, updated_at FROM agent_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if head is None:
                return None
            rows = self._conn.execute(
                "SELECT idx, question, response, created_at FROM agent_turns WHERE session_id = ? ORDER BY idx",
                (session_id,),
            ).fetchall()
        turns = [TurnRecord(index=i, question=q, response=json.loads(r), created_at=c) for i, q, r, c in rows]
        return SessionRecord(id=head[0], created_at=head[1], updated_at=head[2], turns=turns)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._conn.execute("DELETE FROM agent_sessions WHERE id = ?", (session_id,)).rowcount > 0
