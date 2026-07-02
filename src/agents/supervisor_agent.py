from __future__ import annotations

from models import Intent, SessionState


class SupervisorAgent:
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
            ),
        ),
    ]

    def route(self, message: str, state: SessionState) -> Intent:
        if state.pending_intent:
            return state.pending_intent
        lowered = message.lower()
        for intent, keywords in self.KEYWORDS:
            if any(keyword in lowered for keyword in keywords):
                return intent
        return "unknown"
