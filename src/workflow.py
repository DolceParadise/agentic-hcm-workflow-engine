from __future__ import annotations

import uuid
from dataclasses import asdict
from time import perf_counter
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.anomaly_detection_agent import AnomalyDetectionAgent
from agents.compliance_agent import ComplianceAgent, ComplianceDecision
from learning import FeedbackLearner
from memory import EpisodicVectorMemory
from models import Anomaly, GraphTransition, WorkflowResult, WorkflowTrigger
from state import SQLiteStateStore


class WorkflowGraphState(TypedDict, total=False):
    run_id: str
    trigger: WorkflowTrigger
    anomalies: list[Anomaly]
    decisions: list[ComplianceDecision]
    transitions: list[GraphTransition]
    shared_state: dict[str, Any]
    last_transition_at: float
    approvals_queued: int
    actions_executed: int


class SelfHealingWorkflow:
    """LangGraph supervisor workflow with communication only through graph state."""

    def __init__(
        self,
        anomaly_detection_agent: AnomalyDetectionAgent,
        compliance_agent: ComplianceAgent,
        store: SQLiteStateStore,
        auto_action_threshold: float,
        memory: EpisodicVectorMemory | None = None,
    ) -> None:
        self.anomaly_detection_agent = anomaly_detection_agent
        self.compliance_agent = compliance_agent
        self.store = store
        self.feedback_learner = FeedbackLearner(store.connection)
        self.auto_action_threshold = auto_action_threshold
        self.memory = memory or EpisodicVectorMemory(store.connection)
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(WorkflowGraphState)
        builder.add_node("supervisor_agent", self._supervisor_agent)
        builder.add_node("anomaly_detection_agent", self._anomaly_detection_agent)
        builder.add_node("memory_agent", self._memory_agent)
        builder.add_node("compliance_agent", self._compliance_agent)
        builder.add_node("action_agent", self._action_agent)
        builder.add_node("supervisor_finalize_agent", self._supervisor_finalize_agent)
        builder.add_edge(START, "supervisor_agent")
        builder.add_edge("supervisor_agent", "anomaly_detection_agent")
        builder.add_edge("anomaly_detection_agent", "memory_agent")
        builder.add_edge("memory_agent", "compliance_agent")
        builder.add_edge("compliance_agent", "action_agent")
        builder.add_edge("action_agent", "supervisor_finalize_agent")
        builder.add_edge("supervisor_finalize_agent", END)
        return builder.compile()

    def _transition(
        self,
        state: WorkflowGraphState,
        node: str,
        event: str,
        *,
        rl_action=None,
        reward=None,
        tool_calls=None,
        **updates: Any,
    ) -> None:
        shared_state = state["shared_state"]
        inputs = dict(shared_state)
        shared_state.update(updates)
        now = perf_counter()
        transition = GraphTransition(
            state["run_id"],
            len(state["transitions"]) + 1,
            node,
            event,
            dict(shared_state),
            input=inputs,
            output=dict(updates),
            latency_ms=round((now - state["last_transition_at"]) * 1000, 3),
            rl_action=rl_action,
            reward=reward,
            tool_calls=tool_calls or [],
        )
        state["last_transition_at"] = now
        state["transitions"].append(transition)
        self.store.record_transition(transition)

    def _supervisor_agent(self, state: WorkflowGraphState) -> dict[str, Any]:
        self._transition(state, "supervisor_agent", "trigger_received")
        return {
            "transitions": state["transitions"],
            "shared_state": state["shared_state"],
            "last_transition_at": state["last_transition_at"],
        }

    def _anomaly_detection_agent(self, state: WorkflowGraphState) -> dict[str, Any]:
        trigger = state["trigger"]
        self._transition(state, "anomaly_detection_agent", "scan_started")
        if trigger.payload.get("anomalies") is not None:
            anomalies = [Anomaly(**item) for item in trigger.payload["anomalies"]]
        elif trigger.payload.get("anomaly") is not None:
            anomalies = [Anomaly(**trigger.payload["anomaly"])]
        else:
            anomalies = self.anomaly_detection_agent.scan()
        self._transition(
            state,
            "anomaly_detection_agent",
            "scan_completed",
            anomalies=[asdict(item) for item in anomalies],
        )
        return {
            "anomalies": anomalies,
            "transitions": state["transitions"],
            "shared_state": state["shared_state"],
            "last_transition_at": state["last_transition_at"],
        }

    def _memory_agent(self, state: WorkflowGraphState) -> dict[str, Any]:
        for anomaly in state["anomalies"]:
            matches = self.memory.search(anomaly)
            learned = self.feedback_learner.propose(anomaly, matches)
            anomaly.recommended_action = learned.action
            anomaly.confidence = round(
                min(1.0, max(0.0, anomaly.confidence + learned.confidence_adjustment)), 3
            )
            self._transition(
                state,
                "memory_agent",
                "proposal_warm_started",
                memory_matches=[asdict(item) for item in matches],
                active_anomaly=asdict(anomaly),
                feedback_samples=learned.sample_count,
                rl_action=learned.action,
            )
        return {
            "anomalies": state["anomalies"],
            "transitions": state["transitions"],
            "shared_state": state["shared_state"],
            "last_transition_at": state["last_transition_at"],
        }

    def _compliance_agent(self, state: WorkflowGraphState) -> dict[str, Any]:
        decisions = []
        for anomaly in state["anomalies"]:
            decision = self.compliance_agent.evaluate(anomaly)
            decisions.append(decision)
            self._transition(
                state,
                "compliance_agent",
                "hard_veto_evaluated",
                compliance={
                    "allowed": decision.allowed,
                    "reason": decision.reason,
                    "violated_rule_ids": decision.violated_rule_ids,
                },
                rl_action=anomaly.recommended_action,
                reward=-1.0 if not decision.allowed else None,
            )
            if not decision.allowed:
                self.store.record_rl_experience(anomaly, -1.0, "compliance-veto", vetoed=True)
                self.memory.add(anomaly, anomaly.recommended_action, "compliance-veto", -1.0)
        return {
            "decisions": decisions,
            "transitions": state["transitions"],
            "shared_state": state["shared_state"],
            "last_transition_at": state["last_transition_at"],
        }

    def _action_agent(self, state: WorkflowGraphState) -> dict[str, Any]:
        approvals = 0
        actions = 0
        for anomaly, decision in zip(state["anomalies"], state["decisions"], strict=True):
            self.store.save_incident(anomaly)
            needs_approval = not decision.allowed or anomaly.confidence < self.auto_action_threshold
            if needs_approval:
                anomaly.status = "pending-human-review"
                self.store.save_incident(anomaly)
                self.store.queue_approval(anomaly.anomaly_id, decision.reason)
                approvals += 1
                self._transition(
                    state,
                    "action_agent",
                    "human_approval_queued",
                    rl_action=anomaly.recommended_action,
                    tool_calls=[{"name": "queue_approval", "anomaly_id": anomaly.anomaly_id}],
                )
            else:
                status_map = {
                    "auto-correct": "auto-corrected",
                    "escalate-to-manager": "auto-routed-to-manager",
                    "escalate-to-HR": "auto-routed-to-hr",
                    "flag-for-audit": "flagged-for-audit",
                    "no-action": "closed-no-action",
                }
                tool_name_map = {
                    "auto-correct": "auto_correct",
                    "escalate-to-manager": "notify_manager",
                    "escalate-to-HR": "escalate_to_hr",
                    "flag-for-audit": "flag_for_audit",
                    "no-action": "no_action",
                }
                anomaly.status = status_map.get(anomaly.recommended_action, "auto-processed")
                self.store.save_incident(anomaly)
                reward_map = {
                    "auto-correct": 0.25,
                    "escalate-to-manager": 0.18,
                    "escalate-to-HR": 0.15,
                    "flag-for-audit": 0.10,
                    "no-action": 0.05,
                }
                reward = reward_map.get(anomaly.recommended_action, 0.1)
                self.store.record_rl_experience(anomaly, reward, "safe-execution")
                actions += 1
                self._transition(
                    state,
                    "action_agent",
                    "guarded_action_executed",
                    rl_action=anomaly.recommended_action,
                    reward=reward,
                    tool_calls=[
                        {
                            "name": tool_name_map.get(anomaly.recommended_action, "auto_process"),
                            "anomaly_id": anomaly.anomaly_id,
                        }
                    ],
                )
        return {
            "anomalies": state["anomalies"],
            "approvals_queued": approvals,
            "actions_executed": actions,
            "transitions": state["transitions"],
            "shared_state": state["shared_state"],
            "last_transition_at": state["last_transition_at"],
        }

    def _supervisor_finalize_agent(self, state: WorkflowGraphState) -> dict[str, Any]:
        self._transition(state, "supervisor_finalize_agent", "workflow_completed")
        return {
            "transitions": state["transitions"],
            "shared_state": state["shared_state"],
            "last_transition_at": state["last_transition_at"],
        }

    def run(self, trigger: WorkflowTrigger) -> WorkflowResult:
        run_id = str(uuid.uuid4())
        initial: WorkflowGraphState = {
            "run_id": run_id,
            "trigger": trigger,
            "anomalies": [],
            "decisions": [],
            "transitions": [],
            "shared_state": {"trigger": asdict(trigger), "anomalies": []},
            "last_transition_at": perf_counter(),
            "approvals_queued": 0,
            "actions_executed": 0,
        }
        final = self.graph.invoke(initial)
        diagnostics = self.store.rl_diagnostics()
        cost = {
            "actual_usd": 0.0,
            "actual_tokens": 0,
            "naive_baseline_tokens": 1350,
            "token_savings_percent": 100.0,
        }
        return WorkflowResult(
            run_id,
            trigger,
            final["anomalies"],
            final["transitions"],
            final["approvals_queued"],
            final["actions_executed"],
            diagnostics,
            cost,
        )

    def record_outcome_feedback(
        self,
        anomaly_id: str,
        outcome: str,
        *,
        recurrence: bool = False,
        false_positive: bool = False,
        comment: str = "",
    ) -> None:
        anomaly, action, reward = self.store.record_outcome_feedback(
            anomaly_id,
            outcome,
            recurrence=recurrence,
            false_positive=false_positive,
            comment=comment,
        )
        self.memory.add(anomaly, action, outcome, reward)

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
        first = self.run(WorkflowTrigger(str(uuid.uuid4()), "system", {"anomaly": payload}))
        if first.anomalies and first.anomalies[0].status == "pending-human-review":
            anomaly_id = first.anomalies[0].anomaly_id
            anomaly, action, reward = self.store.record_feedback(anomaly_id, feedback_decision)
            self.memory.add(anomaly, action, feedback_decision, reward)
            self.record_outcome_feedback(
                anomaly_id,
                outcome,
                recurrence=recurrence,
                comment="learning-cycle-demo",
            )
        second_payload = dict(payload)
        second_payload["anomaly_id"] = str(uuid.uuid4())
        second = self.run(WorkflowTrigger(str(uuid.uuid4()), "system", {"anomaly": second_payload}))
        return {
            "first": first,
            "second": second,
            "diagnostics": self.store.rl_diagnostics(),
            "pending_approvals": self.store.pending_approvals(),
        }
