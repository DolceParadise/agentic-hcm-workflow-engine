from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models import Anomaly


@dataclass(frozen=True)
class ComplianceDecision:
    allowed: bool
    reason: str
    violated_rule_ids: tuple[str, ...] = ()


class ComplianceAgent:
    """File-backed hard-veto engine; prompts and learned policies cannot override it."""

    def __init__(self, rules_path: Path | None = None) -> None:
        rules_path = (
            rules_path or Path(__file__).resolve().parents[2] / "data/compliance_rules.json"
        )
        document = json.loads(rules_path.read_text(encoding="utf-8"))
        rules = document.get("rules")
        if not isinstance(rules, list) or not 10 <= len(rules) <= 15:
            raise ValueError("Compliance ruleset must contain 10–15 rules")
        self.rules = rules
        self.version = str(document.get("version", "unknown"))

    @staticmethod
    def _value(context: dict[str, Any], path: str) -> Any:
        value: Any = context
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    @classmethod
    def _matches(cls, context: dict[str, Any], condition: dict[str, Any]) -> bool:
        actual = cls._value(context, condition["field"])
        expected = condition.get("value")
        operator = condition["op"]
        if actual is None:
            return False
        if operator == "eq":
            return actual == expected
        if operator == "ne":
            return actual != expected
        if operator == "gt":
            return float(actual) > float(expected)
        if operator == "lt":
            return float(actual) < float(expected)
        if operator == "in":
            return actual in expected
        if operator == "not_in":
            return actual not in expected
        raise ValueError(f"Unsupported compliance operator: {operator}")

    def evaluate(self, anomaly: Anomaly) -> ComplianceDecision:
        context = {
            "category": anomaly.category,
            "action": anomaly.recommended_action,
            "risk": anomaly.risk,
            "evidence": anomaly.evidence,
        }
        violations = [
            rule
            for rule in self.rules
            if all(self._matches(context, condition) for condition in rule["conditions"])
        ]
        if not violations:
            return ComplianceDecision(True, "No hard-veto rule matched.")
        return ComplianceDecision(
            False,
            " ".join(rule["veto"] for rule in violations),
            tuple(rule["id"] for rule in violations),
        )
