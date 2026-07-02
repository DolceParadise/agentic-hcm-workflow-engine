from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from models import Anomaly, GraphTransition, Message, SessionState


class SQLiteStateStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def connection(self) -> sqlite3.Connection:
        """Return a configured connection for collaborating persistence components."""
        return self._connect()

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
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS transitions (
                    run_id TEXT NOT NULL, sequence INTEGER NOT NULL, node TEXT NOT NULL,
                    event TEXT NOT NULL, state_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS incidents (
                    anomaly_id TEXT PRIMARY KEY, employee_id TEXT NOT NULL, category TEXT NOT NULL,
                    description TEXT NOT NULL, confidence REAL NOT NULL,
                    proposed_action TEXT NOT NULL, evidence_json TEXT NOT NULL,
                    status TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS approvals (
                    anomaly_id TEXT PRIMARY KEY, reason TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, anomaly_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    proposed_action TEXT NOT NULL, decision TEXT NOT NULL, chosen_action TEXT,
                    comment TEXT NOT NULL DEFAULT '', created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS rl_experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, anomaly_id TEXT NOT NULL,
                    category TEXT NOT NULL, action TEXT NOT NULL, reward REAL NOT NULL,
                    source TEXT NOT NULL, vetoed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, anomaly_id TEXT NOT NULL,
                    context TEXT NOT NULL, vector_json TEXT NOT NULL, action TEXT NOT NULL,
                    outcome TEXT NOT NULL, reward REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
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

    def record_transition(self, transition: GraphTransition) -> None:
        persisted_state = dict(transition.state)
        persisted_state["_observability"] = {
            "agent": transition.node,
            "input": transition.input,
            "output": transition.output,
            "latency_ms": transition.latency_ms,
            "input_tokens": transition.input_tokens,
            "output_tokens": transition.output_tokens,
            "cost_usd": transition.cost_usd,
            "rl_action": transition.rl_action,
            "reward": transition.reward,
            "tool_calls": transition.tool_calls,
        }
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO transitions
                (run_id, sequence, node, event, state_json) VALUES (?, ?, ?, ?, ?)""",
                (
                    transition.run_id,
                    transition.sequence,
                    transition.node,
                    transition.event,
                    json.dumps(persisted_state),
                ),
            )

    def save_incident(self, anomaly: Anomaly) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO incidents
                (anomaly_id, employee_id, category, description, confidence, proposed_action,
                 evidence_json, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    anomaly.anomaly_id,
                    anomaly.employee_id,
                    anomaly.category,
                    anomaly.description,
                    anomaly.confidence,
                    anomaly.recommended_action,
                    json.dumps(anomaly.evidence),
                    anomaly.status,
                ),
            )

    def queue_approval(self, anomaly_id: str, reason: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO approvals
                (anomaly_id, reason, status) VALUES (?, ?, 'pending')""",
                (anomaly_id, reason),
            )

    def record_feedback(
        self,
        anomaly_id: str,
        decision: str,
        chosen_action: str | None = None,
        comment: str = "",
    ) -> tuple[Anomaly, str, float]:
        if decision not in {"approved", "rejected", "modified"}:
            raise ValueError("decision must be approved, rejected, or modified")
        with self._connect() as connection:
            incident = connection.execute(
                "SELECT * FROM incidents WHERE anomaly_id = ?",
                (anomaly_id,),
            ).fetchone()
            if not incident:
                raise ValueError(f"Unknown anomaly: {anomaly_id}")
            connection.execute(
                """INSERT INTO feedback
                (anomaly_id, category, proposed_action, decision, chosen_action, comment)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    anomaly_id,
                    incident["category"],
                    incident["proposed_action"],
                    decision,
                    chosen_action,
                    comment,
                ),
            )
            connection.execute(
                "UPDATE approvals SET status = ? WHERE anomaly_id = ?", (decision, anomaly_id)
            )
            action = chosen_action or incident["proposed_action"]
            reward = {"approved": 1.0, "modified": 0.5, "rejected": -1.0}[decision]
            connection.execute(
                """INSERT INTO rl_experiences
                (anomaly_id, category, action, reward, source, vetoed)
                VALUES (?, ?, ?, ?, 'human', 0)""",
                (anomaly_id, incident["category"], action, reward),
            )
            anomaly = Anomaly(
                anomaly_id=incident["anomaly_id"],
                employee_id=incident["employee_id"],
                category=incident["category"],
                description=incident["description"],
                confidence=incident["confidence"],
                recommended_action=action,
                evidence=json.loads(incident["evidence_json"]),
                status=incident["status"],
            )
        return anomaly, action, reward

    def record_rl_experience(
        self,
        anomaly: Anomaly,
        reward: float,
        source: str,
        *,
        vetoed: bool = False,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO rl_experiences
                (anomaly_id, category, action, reward, source, vetoed)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    anomaly.anomaly_id,
                    anomaly.category,
                    anomaly.recommended_action,
                    reward,
                    source,
                    int(vetoed),
                ),
            )

    def rl_diagnostics(self) -> dict:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT action, reward, vetoed FROM rl_experiences ORDER BY id"
            ).fetchall()
        cumulative = []
        total = 0.0
        distribution: dict[str, int] = {}
        for row in rows:
            total += float(row["reward"])
            cumulative.append(round(total, 3))
            distribution[row["action"]] = distribution.get(row["action"], 0) + 1
        return {
            "cumulative_reward": cumulative,
            "action_distribution": distribution,
            "veto_count": sum(int(row["vetoed"]) for row in rows),
        }
