from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Intent = Literal["policy", "leave_balance", "leave_apply", "payslip", "unknown"]


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class SessionState:
    session_id: str
    employee_id: str
    messages: list[Message] = field(default_factory=list)
    pending_intent: Intent | None = None
    slots: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyChunk:
    chunk_id: str
    heading: str
    text: str
    source: str


@dataclass(frozen=True)
class SearchResult:
    chunk: PolicyChunk
    score: float


@dataclass
class TraceStep:
    name: str
    kind: Literal["agent", "tool", "retrieval", "llm", "system"]
    input: dict[str, Any]
    output: dict[str, Any]
    latency_ms: float
    status: Literal["ok", "error"] = "ok"
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunResult:
    response: str
    session_id: str
    intent: Intent
    trace: list[TraceStep]
    citations: list[SearchResult] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)
    cost: dict[str, float] = field(default_factory=dict)
