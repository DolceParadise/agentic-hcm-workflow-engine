import pytest
from conftest import FakeLLM

from engine import WorkflowEngine
from llm import LLMResponse
from tools import MockHRTools


def make_engine(settings, fake_index, state_store):
    return WorkflowEngine(
        settings,
        llm=FakeLLM(),
        index=fake_index,
        state_store=state_store,
        tools=MockHRTools(),
    )


def test_policy_agent_is_grounded_and_traced(settings, fake_index, state_store):
    result = make_engine(settings, fake_index, state_store).run(
        "What is the harassment policy?", "policy-session"
    )
    assert "[policy-005]" in result.response
    assert result.citations
    assert [step.name for step in result.trace] == [
        "supervisor_agent",
        "policy_retrieval",
        "grounded_policy_answer",
    ]
    assert result.token_usage["llm_calls"] == 1
    assert result.cost["token_savings_percent"] >= 20


@pytest.mark.parametrize(
    "query",
    [
        "What are the rules for overtime?",
        "What approvals are required for payroll corrections?",
    ],
)
def test_operational_policy_queries_route_to_policy_agent(query, settings, fake_index, state_store):
    result = make_engine(settings, fake_index, state_store).run(query)

    assert result.intent == "policy"
    assert result.citations
    assert any(step.name == "grounded_policy_answer" for step in result.trace)


class SemanticRoutingLLM(FakeLLM):
    def chat(self, messages, **kwargs) -> LLMResponse:
        if kwargs.get("json_mode"):
            return LLMResponse(
                content='{"intent":"policy"}',
                input_tokens=45,
                output_tokens=6,
                cost_usd=0.0,
            )
        return super().chat(messages, **kwargs)


def test_ambiguous_hr_request_uses_semantic_routing(settings, fake_index, state_store):
    engine = WorkflowEngine(
        settings,
        llm=SemanticRoutingLLM(),
        index=fake_index,
        state_store=state_store,
        tools=MockHRTools(),
    )

    result = engine.run(
        "Is my employer permitted to schedule me for sixty hours each week?",
        "semantic-routing-session",
    )

    assert result.intent == "policy"
    assert any(step.name == "semantic_intent_classification" for step in result.trace)
    assert result.token_usage["llm_calls"] == 2


def test_multi_turn_leave_application_persists_slots(settings, fake_index, state_store):
    engine = make_engine(settings, fake_index, state_store)
    first = engine.run("Apply for annual leave starting 2026-07-06", "leave-session")
    assert "end date" in first.response

    second = engine.run("It ends 2026-07-08", "leave-session")
    assert "LV-1001" in second.response
    tool_step = next(step for step in second.trace if step.name == "apply_leave")
    assert tool_step.input["start_date"] == "2026-07-06"
    assert tool_step.input["end_date"] == "2026-07-08"


def test_unknown_request_returns_capabilities(settings, fake_index, state_store):
    result = make_engine(settings, fake_index, state_store).run("Hello", "unknown-session")
    assert "leave balances" in result.response
    assert result.token_usage["llm_calls"] == 0


class NullContentLLM(FakeLLM):
    def chat(self, messages, **kwargs) -> LLMResponse:
        return LLMResponse(  # type: ignore[arg-type]
            content=None, input_tokens=0, output_tokens=0, cost_usd=0.0
        )


def test_null_slot_extraction_falls_back_gracefully(settings, fake_index, state_store):
    engine = WorkflowEngine(
        settings,
        llm=NullContentLLM(),
        index=fake_index,
        state_store=state_store,
        tools=MockHRTools(),
    )

    result = engine.run("Apply for annual leave starting 2026-07-06", "null-content-session")

    assert "end date" in result.response
    extraction = next(step for step in result.trace if step.name == "leave_slot_extraction")
    assert extraction.status == "error"
