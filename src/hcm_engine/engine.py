from __future__ import annotations

import uuid

from hcm_engine.agents import ActionAgent, OrchestratorAgent, PolicyAgent
from hcm_engine.config import Settings
from hcm_engine.llm import LLMError, OpenRouterClient
from hcm_engine.models import Message, RunResult
from hcm_engine.rag import PolicyIndex, QwenEmbedder
from hcm_engine.state import SQLiteStateStore
from hcm_engine.tools import MockHRTools
from hcm_engine.tracing import TraceCollector


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
        self.orchestrator = OrchestratorAgent()
        self.policy_agent = PolicyAgent(
            self.index,
            self.llm,
            self.settings.retrieval_threshold,
            self.settings.retrieval_top_k,
        )
        self.action_agent = ActionAgent(self.tools, self.llm)

    def run(self, message: str, session_id: str | None = None) -> RunResult:
        session_id = session_id or str(uuid.uuid4())
        state = self.state_store.load(session_id, self.settings.employee_id)
        trace = TraceCollector()
        state.messages.append(Message(role="user", content=message))

        with trace.span(
            "orchestrator",
            "agent",
            {"message": message, "pending_intent": state.pending_intent},
        ) as span:
            intent = self.orchestrator.route(message, state)
            span["output"] = {"intent": intent, "strategy": "deterministic_rule_first"}

        citations = []
        try:
            if intent == "policy":
                response, citations = self.policy_agent.run(message, trace)
            else:
                response = self.action_agent.run(intent, message, state, trace)
        except LLMError as exc:
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

