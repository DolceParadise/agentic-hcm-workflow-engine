from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Intent = Literal["policy", "leave_balance", "leave_apply", "payslip", "unknown"]
TriggerType = Literal["reactive", "scheduled", "system"]
RecommendedAction = Literal[
    "auto-correct", "escalate-to-manager", "escalate-to-HR", "flag-for-audit", "no-action"
]


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
class WorkflowTrigger:
    trigger_id: str
    trigger_type: TriggerType
    payload: dict[str, Any]
    source: str = "user"


@dataclass
class Anomaly:
    anomaly_id: str
    employee_id: str
    category: Literal["payroll", "leave", "compliance"]
    description: str
    confidence: float
    recommended_action: RecommendedAction
    evidence: dict[str, Any] = field(default_factory=dict)
    risk: Literal["low", "medium", "high"] = "medium"
    status: str = "detected"


@dataclass
class GraphTransition:
    run_id: str
    sequence: int
    node: str
    event: str
    state: dict[str, Any]
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    rl_action: RecommendedAction | None = None
    reward: float | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class WorkflowResult:
    run_id: str
    trigger: WorkflowTrigger
    anomalies: list[Anomaly]
    transitions: list[GraphTransition]
    approvals_queued: int = 0
    actions_executed: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, float] = field(default_factory=dict)


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
