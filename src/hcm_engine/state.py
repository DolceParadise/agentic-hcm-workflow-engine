from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from hcm_engine.models import Message, SessionState


class SQLiteStateStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    employee_id TEXT NOT NULL,
                    messages_json TEXT NOT NULL,
                    pending_intent TEXT,
                    slots_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def load(self, session_id: str, employee_id: str) -> SessionState:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if not row:
            return SessionState(session_id=session_id, employee_id=employee_id)
        return SessionState(
            session_id=row["session_id"],
            employee_id=row["employee_id"],
            messages=[Message(**item) for item in json.loads(row["messages_json"])],
            pending_intent=row["pending_intent"],
            slots=json.loads(row["slots_json"]),
        )

    def save(self, state: SessionState) -> None:
        messages = json.dumps(
            [{"role": message.role, "content": message.content} for message in state.messages]
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions
                    (session_id, employee_id, messages_json, pending_intent, slots_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    employee_id=excluded.employee_id,
                    messages_json=excluded.messages_json,
                    pending_intent=excluded.pending_intent,
                    slots_json=excluded.slots_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    state.session_id,
                    state.employee_id,
                    messages,
                    state.pending_intent,
                    json.dumps(state.slots),
                ),
            )

