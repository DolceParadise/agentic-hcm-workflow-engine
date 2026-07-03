import csv
import json

from agents.anomaly_detection_agent import AnomalyDetectionAgent, EmployeeDataRepository
from agents.compliance_agent import ComplianceAgent
from memory import EpisodicVectorMemory
from models import Anomaly, WorkflowTrigger
from state import SQLiteStateStore
from workflow import SelfHealingWorkflow


def test_compliance_vetoes_high_confidence_system_alert(tmp_path):
    data_path = tmp_path / "employees.csv"
    data_path.write_text("Employee_ID,Department,Salary_USD,Leave\n1,HR,50000,2\n")
    store = SQLiteStateStore(tmp_path / "state.db")
    workflow = SelfHealingWorkflow(
        AnomalyDetectionAgent(EmployeeDataRepository(data_path)),
        ComplianceAgent(),
        store,
        0.9,
    )
    anomaly = {
        "anomaly_id": "upstream-1",
        "employee_id": "1",
        "category": "compliance",
        "description": "Training overdue",
        "confidence": 0.99,
        "recommended_action": "auto-correct",
        "evidence": {},
        "risk": "high",
    }

    result = workflow.run(WorkflowTrigger("t-1", "system", {"anomaly": anomaly}))

    assert result.actions_executed == 0
    assert result.approvals_queued == 1
    assert result.anomalies[0].status == "pending-human-review"
    assert any(step.event == "hard_veto_evaluated" for step in result.transitions)


def test_detector_finds_leave_training_and_overtime(tmp_path):
    data_path = tmp_path / "employees.csv"
    fields = [
        "Employee_ID",
        "Department",
        "Experience_Years",
        "Salary_USD",
        "Leave",
        "Mandatory_Training_Complete",
        "Overtime_Hours",
    ]
    with data_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "Employee_ID": "1",
                "Department": "Engineering",
                "Experience_Years": 3,
                "Salary_USD": 70000,
                "Leave": 20,
                "Mandatory_Training_Complete": "No",
                "Overtime_Hours": 55,
            }
        )

    anomalies = AnomalyDetectionAgent(EmployeeDataRepository(data_path)).scan()

    assert {item.category for item in anomalies} == {"leave", "compliance"}
    assert len(anomalies) == 3
    assert all(0 <= item.confidence <= 1 for item in anomalies)


def _workflow(tmp_path):
    data_path = tmp_path / "employees.csv"
    data_path.write_text("Employee_ID,Department,Salary_USD,Leave\n1,HR,50000,2\n")
    store = SQLiteStateStore(tmp_path / "state.db")
    memory = EpisodicVectorMemory(store.connection)
    workflow = SelfHealingWorkflow(
        AnomalyDetectionAgent(EmployeeDataRepository(data_path)),
        ComplianceAgent(),
        store,
        0.9,
        memory,
    )
    return workflow, store, memory


def _leave_anomaly(anomaly_id, confidence=0.85):
    return {
        "anomaly_id": anomaly_id,
        "employee_id": "E-9",
        "category": "leave",
        "description": "Repeated Q1 leave threshold anomaly",
        "confidence": confidence,
        "recommended_action": "auto-correct",
        "evidence": {"leave_days": 18, "review_threshold": 15},
        "risk": "low",
    }


def test_ruleset_contains_twelve_external_rules():
    agent = ComplianceAgent()
    assert len(agent.rules) == 12
    assert agent.version == "1.0"


def test_veto_is_a_negative_rl_experience(tmp_path):
    workflow, store, _ = _workflow(tmp_path)
    alert = _leave_anomaly("pay-1", 0.99)
    alert.update(category="payroll", risk="high")

    workflow.run(WorkflowTrigger("t", "system", {"anomaly": alert}))

    diagnostics = store.rl_diagnostics()
    assert diagnostics["cumulative_reward"] == [-1.0]
    assert diagnostics["veto_count"] == 1


def test_outcome_feedback_persists_as_rl_signal(tmp_path):
    workflow, store, _ = _workflow(tmp_path)
    alert = _leave_anomaly("leave-3", 0.84)

    workflow.run(WorkflowTrigger("first", "system", {"anomaly": alert}))
    store.record_feedback("leave-3", "approved")
    workflow.record_outcome_feedback("leave-3", "resolved")

    diagnostics = store.rl_diagnostics()
    assert diagnostics["reward_by_source"]["human"] == 1.0
    assert diagnostics["reward_by_source"]["outcome"] == 0.35
    assert diagnostics["cumulative_reward"][-1] == 1.35


def test_simulated_learning_cycle_shifts_confidence_after_feedback(tmp_path):
    workflow, _, _ = _workflow(tmp_path)

    demo = workflow.simulate_learning_cycle()

    assert demo["first"].approvals_queued == 1
    assert demo["second"].actions_executed == 1
    assert demo["second"].anomalies[0].confidence > demo["first"].anomalies[0].confidence


def test_episode_retrieval_returns_semantically_similar_incident(tmp_path):
    _, _, memory = _workflow(tmp_path)
    memory.add(Anomaly(**_leave_anomaly("leave-1")), "auto-correct", "approved", 1.0)

    matches = memory.search(Anomaly(**_leave_anomaly("leave-2")))

    assert matches[0].anomaly_id == "leave-1"
    assert matches[0].similarity > 0.9


def test_second_occurrence_is_higher_confidence_and_faster_to_resolution(tmp_path):
    workflow, store, memory = _workflow(tmp_path)
    first = workflow.run(WorkflowTrigger("first", "system", {"anomaly": _leave_anomaly("leave-1")}))
    assert first.approvals_queued == 1
    anomaly, action, reward = store.record_feedback("leave-1", "approved")
    memory.add(anomaly, action, "approved", reward)

    second = workflow.run(
        WorkflowTrigger("second", "system", {"anomaly": _leave_anomaly("leave-2")})
    )

    assert second.anomalies[0].confidence > first.anomalies[0].confidence
    assert second.actions_executed == 1
    assert second.approvals_queued == 0


def test_structured_trace_contains_rl_reward_tools_and_latency(tmp_path):
    workflow, _, _ = _workflow(tmp_path)
    result = workflow.run(
        WorkflowTrigger("t", "system", {"anomaly": _leave_anomaly("leave-1", 0.95)})
    )
    action_step = next(step for step in result.transitions if step.node == "action_agent")
    assert action_step.latency_ms >= 0
    assert action_step.rl_action == "auto-correct"
    assert action_step.reward == 0.25
    assert action_step.tool_calls[0]["name"] == "auto_correct"


def test_diagnostics_report_reward_curve_and_action_shift(tmp_path):
    workflow, _, _ = _workflow(tmp_path)
    result = workflow.run(
        WorkflowTrigger("t", "system", {"anomaly": _leave_anomaly("leave-1", 0.95)})
    )
    assert result.diagnostics["cumulative_reward"] == [0.25]
    assert result.diagnostics["action_distribution"] == {"auto-correct": 1}


def test_deterministic_scan_cost_beats_naive_baseline(tmp_path):
    workflow, _, _ = _workflow(tmp_path)
    result = workflow.run(WorkflowTrigger("t", "scheduled", {}))
    assert result.cost["actual_tokens"] == 0
    assert result.cost["naive_baseline_tokens"] == 1350
    assert result.cost["token_savings_percent"] >= 20


def test_persisted_transition_contains_observability_payload(tmp_path):
    workflow, store, _ = _workflow(tmp_path)
    result = workflow.run(
        WorkflowTrigger("t", "system", {"anomaly": _leave_anomaly("leave-1", 0.95)})
    )
    with store.connection() as connection:
        row = connection.execute(
            "SELECT state_json FROM transitions WHERE run_id = ? ORDER BY sequence DESC LIMIT 1",
            (result.run_id,),
        ).fetchone()
    payload = json.loads(row["state_json"])
    assert payload["_observability"]["agent"] == "supervisor_finalize_agent"
    assert "latency_ms" in payload["_observability"]
