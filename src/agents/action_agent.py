from __future__ import annotations

import json
import re
from datetime import date, datetime

from llm import LLMError, OpenRouterClient
from models import Intent, SessionState
from tools import MockHRTools, ToolError, call_with_retry
from tracing import TraceCollector


class ActionAgent:
    def __init__(self, tools: MockHRTools, llm: OpenRouterClient) -> None:
        self.tools = tools
        self.llm = llm

    @staticmethod
    def _iso_dates(message: str) -> list[str]:
        return re.findall(r"\b\d{4}-\d{2}-\d{2}\b", message)

    @staticmethod
    def _month(message: str) -> str | None:
        numeric = re.search(r"\b(20\d{2})-(0[1-9]|1[0-2])\b", message)
        if numeric:
            return numeric.group(0)
        for fmt in ("%B %Y", "%b %Y"):
            for match in re.finditer(r"\b[A-Za-z]{3,9}\s+20\d{2}\b", message):
                try:
                    return datetime.strptime(match.group(0), fmt).strftime("%Y-%m")
                except ValueError:
                    pass
        return None

    def _extract_leave_slots(
        self, message: str, state: SessionState, trace: TraceCollector
    ) -> None:
        dates = self._iso_dates(message)
        if dates:
            if state.slots.get("start_date") and not state.slots.get("end_date"):
                state.slots["end_date"] = dates[0]
            else:
                state.slots["start_date"] = dates[0]
            if len(dates) > 1:
                state.slots["end_date"] = dates[1]
        leave_match = re.search(r"\b(annual|sick|casual)\s+leave\b", message.lower())
        if leave_match:
            state.slots["leave_type"] = leave_match.group(1)
        if all(key in state.slots for key in ("start_date", "end_date", "leave_type")):
            return
        prompt = [
            {
                "role": "system",
                "content": (
                    "Extract HR leave fields from the user message. Return JSON only with any "
                    "explicitly stated start_date, end_date (YYYY-MM-DD), leave_type "
                    "(annual/sick/casual), and reason. Do not infer missing values. "
                    f"Today is {date.today().isoformat()}."
                ),
            },
            {"role": "user", "content": message},
        ]
        try:
            with trace.span("leave_slot_extraction", "llm", {"model": self.llm.model}) as span:
                response = self.llm.chat(prompt, max_tokens=160, json_mode=True)
                extracted = json.loads(response.content)
                if not isinstance(extracted, dict):
                    raise ValueError("Leave slot extraction must return a JSON object.")
                for key in ("start_date", "end_date", "leave_type", "reason"):
                    if extracted.get(key):
                        state.slots[key] = extracted[key]
                span.update(
                    output={"extracted": extracted},
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    cost_usd=response.cost_usd,
                )
        except (LLMError, TypeError, ValueError, json.JSONDecodeError):
            return

    def run(self, intent: Intent, message: str, state: SessionState, trace: TraceCollector) -> str:
        if intent == "leave_balance":
            return self._call_tool(
                "check_leave_balance",
                self.tools.check_leave_balance,
                {"employee_id": state.employee_id},
                trace,
                lambda result: (
                    "Your available leave is: "
                    + ", ".join(
                        f"{name} {days:g} days" for name, days in result["balances_days"].items()
                    )
                    + "."
                ),
            )
        if intent == "payslip":
            month = self._month(message) or state.slots.get("month")
            if not month:
                state.pending_intent = "payslip"
                return (
                    "Which payslip month do you need? Please use YYYY-MM or a month like June 2026."
                )
            state.pending_intent = None
            state.slots.clear()
            return self._call_tool(
                "fetch_payslip",
                self.tools.fetch_payslip,
                {"employee_id": state.employee_id, "month": month},
                trace,
                lambda result: (
                    f"Your {result['label']} payslip is available at `{result['document_url']}`."
                ),
            )
        if intent == "leave_apply":
            self._extract_leave_slots(message, state, trace)
            missing = [
                key for key in ("start_date", "end_date", "leave_type") if not state.slots.get(key)
            ]
            if missing:
                state.pending_intent = "leave_apply"
                labels = {
                    "start_date": "start date (YYYY-MM-DD)",
                    "end_date": "end date (YYYY-MM-DD)",
                    "leave_type": "leave type (annual, sick, or casual)",
                }
                return "Please provide the " + ", ".join(labels[item] for item in missing) + "."
            arguments = {
                "employee_id": state.employee_id,
                "start_date": state.slots["start_date"],
                "end_date": state.slots["end_date"],
                "leave_type": state.slots["leave_type"],
                "reason": state.slots.get("reason", ""),
            }
            response = self._call_tool(
                "apply_leave",
                self.tools.apply_leave,
                arguments,
                trace,
                lambda result: (
                    f"Leave request {result['request_id']} was submitted for "
                    f"{result['working_days']} working day(s), from {result['start_date']} "
                    f"to {result['end_date']}."
                ),
            )
            state.pending_intent = None
            state.slots.clear()
            return response
        return (
            "I can check leave balances, apply for leave, fetch payslips, "
            "or answer policy questions."
        )

    @staticmethod
    def _call_tool(name, function, arguments, trace, formatter) -> str:
        try:
            with trace.span(name, "tool", arguments) as span:
                result = call_with_retry(function, arguments)
                span["output"] = result
            return formatter(result)
        except ToolError as exc:
            return f"I couldn't complete that action: {exc}"
