from __future__ import annotations

import csv
import math
import statistics
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from models import Anomaly


def _first(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return ""


def _number(row: dict[str, str], *names: str) -> float | None:
    try:
        value = _first(row, *names)
        return float(value) if value else None
    except ValueError:
        return None


class EmployeeDataRepository:
    """Normalizes the supplied CSV while tolerating common HRIS column names."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[dict[str, Any]]:
        with self.path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return [
            {
                "employee_id": _first(row, "Employee_ID", "employee_id", "id"),
                "department": _first(row, "Department", "department") or "Unknown",
                "experience": _number(row, "Experience_Years", "experience_years") or 0.0,
                "salary": _number(row, "Salary_USD", "salary", "gross_pay"),
                "leave_days": _number(row, "Leave", "leave_days", "q1_leave_days"),
                "overtime_hours": _number(row, "Overtime_Hours", "overtime_hours"),
                "training_complete": _first(row, "Mandatory_Training_Complete", "training_complete")
                .strip()
                .lower(),
                "raw": row,
            }
            for row in rows
        ]


class AnomalyDetectionAgent:
    """Deterministic, explainable cohort and policy-limit anomaly detector.

    Payroll uses a robust modified z-score within department and five-year experience
    bands. Leave and compliance confidence grow with the amount over a documented limit.
    """

    def __init__(self, repository: EmployeeDataRepository) -> None:
        self.repository = repository

    @staticmethod
    def _id(employee_id: str, category: str) -> str:
        return f"AN-{category[:3].upper()}-{employee_id}-{uuid.uuid4().hex[:8]}"

    def scan(self) -> list[Anomaly]:
        records = self.repository.load()
        anomalies = self._payroll(records)
        anomalies.extend(self._leave(records))
        anomalies.extend(self._compliance(records))
        return sorted(anomalies, key=lambda item: item.confidence, reverse=True)

    def _payroll(self, records: list[dict[str, Any]]) -> list[Anomaly]:
        cohorts: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            if record["salary"] is not None:
                cohorts[(record["department"], int(record["experience"] // 5))].append(record)
        output: list[Anomaly] = []
        for cohort, members in cohorts.items():
            if len(members) < 5:
                continue
            salaries = [member["salary"] for member in members]
            median = statistics.median(salaries)
            mad = statistics.median(abs(value - median) for value in salaries)
            scale = max(1.4826 * mad, median * 0.03, 1.0)
            for member in members:
                z_score = abs(member["salary"] - median) / scale
                # 2.5 is a review threshold, not a finding of incorrect pay. It catches
                # extreme tails in this synthetic data while the action remains audit-only.
                if z_score < 2.5:
                    continue
                confidence = min(0.99, 0.72 + 0.06 * (z_score - 2.5))
                output.append(
                    Anomaly(
                        anomaly_id=self._id(member["employee_id"], "payroll"),
                        employee_id=member["employee_id"],
                        category="payroll",
                        description="Salary is a robust outlier within its peer cohort.",
                        confidence=round(confidence, 3),
                        recommended_action="flag-for-audit",
                        risk="high",
                        evidence={
                            "salary": member["salary"],
                            "cohort_median": median,
                            "modified_z_score": round(z_score, 2),
                            "cohort": {"department": cohort[0], "experience_band": cohort[1]},
                            "policy_refs": ["Payroll Accuracy", "Payroll Adjustments"],
                        },
                    )
                )
        return output

    def _leave(self, records: list[dict[str, Any]]) -> list[Anomaly]:
        output = []
        for record in records:
            days = record["leave_days"]
            if days is None or days <= 15:
                continue
            confidence = min(0.98, 0.65 + (days - 15) / 30)
            output.append(
                Anomaly(
                    anomaly_id=self._id(record["employee_id"], "leave"),
                    employee_id=record["employee_id"],
                    category="leave",
                    description="Leave usage exceeds the Q1 review threshold of 15 days.",
                    confidence=round(confidence, 3),
                    recommended_action="escalate-to-manager",
                    evidence={
                        "leave_days": days,
                        "review_threshold": 15,
                        "policy_refs": ["Annual Leave"],
                    },
                )
            )
        return output

    def _compliance(self, records: list[dict[str, Any]]) -> list[Anomaly]:
        output = []
        for record in records:
            training = record["training_complete"]
            overtime = record["overtime_hours"]
            if training in {"no", "false", "0", "overdue", "missing"}:
                output.append(
                    Anomaly(
                        anomaly_id=self._id(record["employee_id"], "compliance"),
                        employee_id=record["employee_id"],
                        category="compliance",
                        description="Mandatory training is incomplete.",
                        confidence=0.99,
                        recommended_action="escalate-to-HR",
                        risk="high",
                        evidence={
                            "rule": "mandatory_training_required",
                            "policy_refs": ["Mandatory Learning"],
                        },
                    )
                )
            if overtime is not None and overtime > 48:
                confidence = min(0.99, 0.88 + math.log1p(overtime - 48) / 20)
                output.append(
                    Anomaly(
                        anomaly_id=self._id(record["employee_id"], "compliance"),
                        employee_id=record["employee_id"],
                        category="compliance",
                        description="Recorded overtime exceeds the 48-hour cycle cap.",
                        confidence=round(confidence, 3),
                        recommended_action="escalate-to-HR",
                        risk="high",
                        evidence={
                            "overtime_hours": overtime,
                            "cap": 48,
                            "policy_refs": ["Overtime"],
                        },
                    )
                )
        return output
