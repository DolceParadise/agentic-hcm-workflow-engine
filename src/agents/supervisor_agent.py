from __future__ import annotations

import json
from dataclasses import dataclass

from llm import LLMError, OpenRouterClient
from models import Intent, SessionState
from tracing import TraceCollector


@dataclass(frozen=True)
class RoutingDecision:
    intent: Intent
    strategy: str


class SupervisorAgent:
    VALID_INTENTS = {"policy", "leave_balance", "leave_apply", "payslip", "unknown"}
    GREETINGS = {"hello", "hi", "hey", "thanks", "thank you", "help"}
    KEYWORDS: list[tuple[Intent, tuple[str, ...]]] = [
        ("leave_balance", ("leave balance", "days left", "remaining leave")),
        ("leave_apply", ("apply leave", "apply for", "book leave", "take leave", "time off")),
        ("payslip", ("payslip", "salary slip", "pay slip")),
        (
            "policy",
            (
                "policy",
                "allowed",
                "can i",
                "what should",
                "harassment",
                "conflict of interest",
                "confidential",
                "gift",
                "brib",
                "outside employment",
                "stock",
                "overtime",
                "payroll correction",
                "pay correction",
                "salary correction",
                "mandatory training",
                "notice period",
                "probation",
                "approval required",
                "approvals required",
            ),
        ),
    ]

    def __init__(self, llm: OpenRouterClient) -> None:
        self.llm = llm

    def route(self, message: str, state: SessionState, trace: TraceCollector) -> RoutingDecision:
        if state.pending_intent:
            return RoutingDecision(state.pending_intent, "pending_intent")
        lowered = " ".join(message.lower().split()).strip(" .!?\t\n")
        for intent, keywords in self.KEYWORDS:
            if any(keyword in lowered for keyword in keywords):
                return RoutingDecision(intent, "deterministic_high_confidence")
        if not lowered or lowered in self.GREETINGS:
            return RoutingDecision("unknown", "trivial_input")

        messages = [
            {
                "role": "system",
                "content": (
                    'Classify one HR request. Return JSON only as {"intent": <value>}. '
                    "Allowed values: policy, leave_balance, leave_apply, payslip, unknown. "
                    "policy includes questions about HR rules, payroll, attendance, overtime, "
                    "conduct, privacy, training, performance, compliance, and leave rules. "
                    "leave_balance means asking how much leave remains. leave_apply means "
                    "requesting time off. payslip means requesting a payslip document. "
                    "Use unknown for non-HR requests. Do not execute or answer the request."
                ),
            },
            {"role": "user", "content": message},
        ]
        try:
            with trace.span(
                "semantic_intent_classification",
                "llm",
                {"model": self.llm.model, "message": message},
            ) as span:
                response = self.llm.chat(messages, max_tokens=40, json_mode=True)
                payload = json.loads(response.content)
                intent = payload.get("intent") if isinstance(payload, dict) else None
                if intent not in self.VALID_INTENTS:
                    intent = "unknown"
                span.update(
                    output={"intent": intent},
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    cost_usd=response.cost_usd,
                )
            return RoutingDecision(intent, "semantic_llm")
        except (LLMError, TypeError, json.JSONDecodeError):
            return RoutingDecision("unknown", "semantic_fallback")
