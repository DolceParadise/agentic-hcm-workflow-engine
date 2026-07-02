from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    test: str
    reasoning: str


CASES: list[EvaluationCase] = [
    EvaluationCase(
        "grounded-policy",
        "tests/test_engine.py::test_policy_agent_is_grounded_and_traced",
        "RAG response cites retrieved policy evidence.",
    ),
    EvaluationCase(
        "multi-turn-action",
        "tests/test_engine.py::test_multi_turn_leave_application_persists_slots",
        "Shared state preserves action slots across turns.",
    ),
    EvaluationCase(
        "unknown-input",
        "tests/test_engine.py::test_unknown_request_returns_capabilities",
        "Unknown input fails safely without an LLM call.",
    ),
    EvaluationCase(
        "null-llm",
        "tests/test_engine.py::test_null_slot_extraction_falls_back_gracefully",
        "Malformed model output does not trigger an unsafe tool call.",
    ),
    EvaluationCase(
        "policy-chunking",
        "tests/test_rag.py::test_policy_is_chunked_by_heading",
        "Policy corpus remains section-addressable.",
    ),
    EvaluationCase(
        "adversarial-veto",
        "tests/test_self_healing.py::test_compliance_vetoes_high_confidence_system_alert",
        "Hard veto overrides high confidence and auto-correct.",
    ),
    EvaluationCase(
        "detector-coverage",
        "tests/test_self_healing.py::test_detector_finds_leave_training_and_overtime",
        "Leave and both compliance detector paths emit bounded scores.",
    ),
    EvaluationCase(
        "external-rules",
        "tests/test_self_healing.py::test_ruleset_contains_twelve_external_rules",
        "The versioned file contains the required 10–15 rules.",
    ),
    EvaluationCase(
        "veto-penalty",
        "tests/test_self_healing.py::test_veto_is_a_negative_rl_experience",
        "A veto contributes negative reward to the policy.",
    ),
    EvaluationCase(
        "semantic-memory",
        "tests/test_self_healing.py::test_episode_retrieval_returns_semantically_similar_incident",
        "A recurring incident retrieves its prior resolution.",
    ),
    EvaluationCase(
        "warm-start",
        "tests/test_self_healing.py::test_second_occurrence_is_higher_confidence_and_faster_to_resolution",
        "Positive memory raises confidence and removes HITL on recurrence.",
    ),
    EvaluationCase(
        "structured-trace",
        "tests/test_self_healing.py::test_structured_trace_contains_rl_reward_tools_and_latency",
        "Trace exposes latency, tool calls, RL action, and reward.",
    ),
    EvaluationCase(
        "rl-diagnostics",
        "tests/test_self_healing.py::test_diagnostics_report_reward_curve_and_action_shift",
        "Diagnostics expose cumulative reward and action counts.",
    ),
    EvaluationCase(
        "cost-reduction",
        "tests/test_self_healing.py::test_deterministic_scan_cost_beats_naive_baseline",
        "Deterministic specialists beat the all-LLM baseline by at least 20%.",
    ),
    EvaluationCase(
        "persisted-observability",
        "tests/test_self_healing.py::test_persisted_transition_contains_observability_payload",
        "The audit store retains structured observability fields.",
    ),
]


def main() -> None:
    results = []
    for case in CASES:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", case.test, "-q"],
            check=False,
            capture_output=True,
            text=True,
        )
        results.append(
            {
                **asdict(case),
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "detail": (completed.stdout + completed.stderr).strip(),
            }
        )
    print(
        json.dumps(
            {
                "passed": sum(r["status"] == "PASS" for r in results),
                "total": len(results),
                "cases": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
