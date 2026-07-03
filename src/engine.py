from __future__ import annotations

import uuid
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.action_agent import ActionAgent
from agents.anomaly_detection_agent import AnomalyDetectionAgent, EmployeeDataRepository
from agents.compliance_agent import ComplianceAgent
from agents.policy_agent import PolicyAgent
from agents.supervisor_agent import SupervisorAgent
from config import Settings
from llm import LLMError, OpenRouterClient
from memory import EpisodicVectorMemory
from models import Message, RunResult, TriggerType, WorkflowResult, WorkflowTrigger
from rag import PolicyIndex, QwenEmbedder
from state import SQLiteStateStore
from tools import MockHRTools
from tracing import TraceCollector
from workflow import SelfHealingWorkflow


class ConversationGraphState(TypedDict, total=False):
    message: str
    session_state: Any
    trace: TraceCollector
    intent: str
    response: str
    citations: list


class WorkflowEngine:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: OpenRouterClient | None = None,
        index: PolicyIndex | None = None,
        state_store: SQLiteStateStore | None = None,
        tools: MockHRTools | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.llm = llm or OpenRouterClient(
            self.settings.openrouter_api_key, self.settings.openrouter_model
        )
        self.index = index or PolicyIndex(
            self.settings.policy_path,
            self.settings.index_path,
            QwenEmbedder(self.settings.embedding_model),
        )
        self.state_store = state_store or SQLiteStateStore(self.settings.db_path)
        self.tools = tools or MockHRTools()
        self.supervisor_agent = SupervisorAgent(self.llm)
        self.policy_agent = PolicyAgent(
            self.index,
            self.llm,
            self.settings.retrieval_threshold,
            self.settings.retrieval_top_k,
        )
        self.action_agent = ActionAgent(self.tools, self.llm)
        self.episodic_memory = EpisodicVectorMemory(self.state_store.connection)
        self.self_healing = SelfHealingWorkflow(
            AnomalyDetectionAgent(EmployeeDataRepository(self.settings.employee_data_path)),
            ComplianceAgent(self.settings.compliance_rules_path),
            self.state_store,
            self.settings.auto_action_threshold,
            self.episodic_memory,
        )
        self.conversation_graph = self._build_conversation_graph()

    def _build_conversation_graph(self):
        builder = StateGraph(ConversationGraphState)
        builder.add_node("supervisor_agent", self._supervisor_agent_node)
        builder.add_node("policy_agent", self._policy_agent_node)
        builder.add_node("action_agent", self._action_agent_node)
        builder.add_edge(START, "supervisor_agent")
        builder.add_conditional_edges(
            "supervisor_agent",
            lambda state: "policy_agent" if state["intent"] == "policy" else "action_agent",
            {"policy_agent": "policy_agent", "action_agent": "action_agent"},
        )
        builder.add_edge("policy_agent", END)
        builder.add_edge("action_agent", END)
        return builder.compile()

    def _supervisor_agent_node(self, graph_state: ConversationGraphState) -> dict:
        session_state = graph_state["session_state"]
        message = graph_state["message"]
        trace = graph_state["trace"]
        with trace.span(
            "supervisor_agent",
            "agent",
            {"message": message, "pending_intent": session_state.pending_intent},
        ) as span:
            decision = self.supervisor_agent.route(message, session_state, trace)
            span["output"] = {
                "intent": decision.intent,
                "strategy": decision.strategy,
            }
        return {"intent": decision.intent}

    def _policy_agent_node(self, graph_state: ConversationGraphState) -> dict:
        response, citations = self.policy_agent.run(graph_state["message"], graph_state["trace"])
        return {"response": response, "citations": citations}

    def _action_agent_node(self, graph_state: ConversationGraphState) -> dict:
        response = self.action_agent.run(
            graph_state["intent"],
            graph_state["message"],
            graph_state["session_state"],
            graph_state["trace"],
        )
        return {"response": response, "citations": []}

    def process_trigger(
        self,
        trigger_type: TriggerType,
        payload: dict | None = None,
        *,
        source: str = "user",
    ) -> WorkflowResult:
        """Process scheduled, system, or structured reactive signals through the graph."""
        trigger = WorkflowTrigger(str(uuid.uuid4()), trigger_type, payload or {}, source)
        return self.self_healing.run(trigger)

    def run_scheduled_scan(self) -> WorkflowResult:
        return self.process_trigger("scheduled", source="scheduler")

    def record_feedback(
        self,
        anomaly_id: str,
        decision: str,
        chosen_action: str | None = None,
        comment: str = "",
    ) -> None:
        anomaly, action, reward = self.state_store.record_feedback(
            anomaly_id, decision, chosen_action, comment
        )
        self.episodic_memory.add(anomaly, action, decision, reward)

    def record_outcome_feedback(
        self,
        anomaly_id: str,
        outcome: str,
        *,
        recurrence: bool = False,
        false_positive: bool = False,
        comment: str = "",
    ) -> None:
        anomaly, action, reward = self.state_store.record_outcome_feedback(
            anomaly_id,
            outcome,
            recurrence=recurrence,
            false_positive=false_positive,
            comment=comment,
        )
        self.episodic_memory.add(anomaly, action, outcome, reward)

    def rl_diagnostics(self) -> dict:
        return self.state_store.rl_diagnostics()

    def pending_approvals(self) -> list[dict[str, str]]:
        return self.state_store.pending_approvals()

    def recent_incidents(self, limit: int = 20) -> list[dict[str, str]]:
        return self.state_store.recent_incidents(limit)

    def recent_experiences(self, limit: int = 20) -> list[dict[str, str]]:
        return self.state_store.recent_experiences(limit)

    def simulate_learning_cycle(
        self,
        anomaly_payload: dict[str, Any] | None = None,
        *,
        feedback_decision: str = "approved",
        outcome: str = "resolved",
        recurrence: bool = False,
    ) -> dict[str, Any]:
        payload = anomaly_payload or {
            "anomaly_id": str(uuid.uuid4()),
            "employee_id": "E-DEMO",
            "category": "leave",
            "description": "Repeated Q1 leave threshold anomaly",
            "confidence": 0.84,
            "recommended_action": "auto-correct",
            "evidence": {"leave_days": 18, "review_threshold": 15},
            "risk": "low",
        }
        first = self.process_trigger("system", {"anomaly": payload}, source="rl-demo")
        queued = self.pending_approvals()
        if queued:
            anomaly_id = queued[0]["anomaly_id"]
            self.record_feedback(anomaly_id, feedback_decision)
            self.record_outcome_feedback(
                anomaly_id,
                outcome,
                recurrence=recurrence,
                comment="learning-cycle-demo",
            )
        second_payload = dict(payload)
        second_payload["anomaly_id"] = str(uuid.uuid4())
        second = self.process_trigger("system", {"anomaly": second_payload}, source="rl-demo")
        return {
            "first": first,
            "second": second,
            "diagnostics": self.rl_diagnostics(),
            "pending_approvals": self.pending_approvals(),
        }

    def run(self, message: str, session_id: str | None = None) -> RunResult:
        session_id = session_id or str(uuid.uuid4())
        state = self.state_store.load(session_id, self.settings.employee_id)
        trace = TraceCollector()
        state.messages.append(Message(role="user", content=message))

        try:
            graph_result = self.conversation_graph.invoke(
                {"message": message, "session_state": state, "trace": trace}
            )
            intent = graph_result["intent"]
            response = graph_result["response"]
            citations = graph_result.get("citations", [])
        except LLMError as exc:
            intent = "unknown"
            citations = []
            response = f"The language model is temporarily unavailable: {exc}"

        state.messages.append(Message(role="assistant", content=response))
        self.state_store.save(state)

        input_tokens = sum(step.input_tokens for step in trace.steps)
        output_tokens = sum(step.output_tokens for step in trace.steps)
        actual_cost = sum(step.cost_usd for step in trace.steps)
        llm_calls = sum(step.kind == "llm" for step in trace.steps)
        naive_calls = 3
        naive_tokens = max(input_tokens, 450) * naive_calls
        optimized_tokens = input_tokens + output_tokens
        savings = 100.0 * (naive_tokens - optimized_tokens) / naive_tokens
        return RunResult(
            response=response,
            session_id=session_id,
            intent=intent,
            trace=trace.steps,
            citations=citations,
            token_usage={
                "input": input_tokens,
                "output": output_tokens,
                "total": optimized_tokens,
                "llm_calls": llm_calls,
                "naive_baseline_estimate": naive_tokens,
            },
            cost={
                "actual_usd": round(actual_cost, 8),
                "token_savings_percent": round(max(savings, 0.0), 1),
            },
        )
