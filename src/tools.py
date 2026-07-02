from __future__ import annotations

import calendar
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any


class ToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]


TOOL_DEFINITIONS = [
    ToolDefinition(
        name="check_leave_balance",
        description="Return the employee's available leave balances.",
        input_schema={
            "type": "object",
            "properties": {"employee_id": {"type": "string"}},
            "required": ["employee_id"],
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        name="apply_leave",
        description="Apply for leave after dates and type are confirmed.",
        input_schema={
            "type": "object",
            "properties": {
                "employee_id": {"type": "string"},
                "start_date": {"type": "string", "format": "date"},
                "end_date": {"type": "string", "format": "date"},
                "leave_type": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["employee_id", "start_date", "end_date", "leave_type"],
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        name="fetch_payslip",
        description="Fetch payslip metadata for an employee and month.",
        input_schema={
            "type": "object",
            "properties": {
                "employee_id": {"type": "string"},
                "month": {"type": "string", "pattern": "^\\d{4}-\\d{2}$"},
            },
            "required": ["employee_id", "month"],
            "additionalProperties": False,
        },
    ),
]


class MockHRTools:
    def __init__(self) -> None:
        self.balances = {
            "E001": {"annual": 18.0, "sick": 8.0, "casual": 5.0},
        }
        self.applications: list[dict[str, Any]] = []

    def check_leave_balance(self, employee_id: str) -> dict[str, Any]:
        balances = self.balances.get(employee_id)
        if balances is None:
            raise ToolError(f"No leave record found for employee {employee_id}.")
        return {"employee_id": employee_id, "balances_days": balances}

    def apply_leave(
        self,
        employee_id: str,
        start_date: str,
        end_date: str,
        leave_type: str,
        reason: str = "",
    ) -> dict[str, Any]:
        try:
            start = date.fromisoformat(start_date)
            end = date.fromisoformat(end_date)
        except ValueError as exc:
            raise ToolError("Dates must use YYYY-MM-DD format.") from exc
        if end < start:
            raise ToolError("End date cannot be before start date.")
        days = sum(
            1
            for offset in range((end - start).days + 1)
            if (date.fromordinal(start.toordinal() + offset)).weekday() < 5
        )
        balance = self.balances.get(employee_id, {}).get(leave_type.lower())
        if balance is None:
            raise ToolError(f"Unknown leave type: {leave_type}.")
        if days > balance:
            raise ToolError(
                f"Insufficient {leave_type} leave: {days} days requested, {balance:g} available."
            )
        request_id = f"LV-{len(self.applications) + 1001}"
        application = {
            "request_id": request_id,
            "employee_id": employee_id,
            "start_date": start_date,
            "end_date": end_date,
            "working_days": days,
            "leave_type": leave_type.lower(),
            "reason": reason,
            "status": "submitted",
        }
        self.applications.append(application)
        self.balances[employee_id][leave_type.lower()] -= days
        return application

    def fetch_payslip(self, employee_id: str, month: str) -> dict[str, Any]:
        try:
            parsed = datetime.strptime(month, "%Y-%m")
        except ValueError as exc:
            raise ToolError("Payslip month must use YYYY-MM format.") from exc
        if parsed.date() > date.today().replace(day=1):
            raise ToolError("A payslip is not available for a future month.")
        label = f"{calendar.month_name[parsed.month]} {parsed.year}"
        return {
            "employee_id": employee_id,
            "month": month,
            "label": label,
            "document_url": f"mock://payslips/{employee_id}/{month}.pdf",
            "status": "available",
        }


def call_with_retry(
    function: Callable[..., dict[str, Any]],
    kwargs: dict[str, Any],
    *,
    attempts: int = 2,
    delay_seconds: float = 0.05,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return function(**kwargs)
        except (TimeoutError, ConnectionError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(delay_seconds * (attempt + 1))
        except ToolError:
            raise
    raise ToolError(f"Tool remained unavailable after {attempts} attempts: {last_error}")


def definitions_as_dicts() -> list[dict[str, Any]]:
    return [asdict(definition) for definition in TOOL_DEFINITIONS]
