from __future__ import annotations

import json
import subprocess
import sys
import uuid

import streamlit as st
from dotenv import load_dotenv

from engine import WorkflowEngine
from evaluation import CASES

load_dotenv()
st.set_page_config(page_title="Self-Healing HR Ops Platform", page_icon="🧭", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(41, 98, 255, 0.18), transparent 30%),
                radial-gradient(circle at top right, rgba(255, 122, 24, 0.14), transparent 26%),
                linear-gradient(180deg, #09111f 0%, #0d1526 42%, #f5f7fb 42%, #f5f7fb 100%);
        }
        .hero-card {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.92));
            color: white;
            border-radius: 24px;
            padding: 24px 26px;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 24px 60px rgba(2, 6, 23, 0.28);
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.82);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 20px;
            padding: 18px 20px;
            box-shadow: 0 18px 35px rgba(15, 23, 42, 0.08);
        }
        .badge {
            display: inline-block;
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
            font-size: 0.77rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            background: rgba(59, 130, 246, 0.12);
            color: #1d4ed8;
        }
        .muted { color: #64748b; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_engine() -> WorkflowEngine:
    return WorkflowEngine()


def _default_learning_anomaly() -> dict[str, object]:
    return {
        "anomaly_id": str(uuid.uuid4()),
        "employee_id": "E-DEMO",
        "category": "leave",
        "description": "Repeated Q1 leave threshold anomaly",
        "confidence": 0.84,
        "recommended_action": "auto-correct",
        "evidence": {"leave_days": 18, "review_threshold": 15, "policy_refs": ["Annual Leave"]},
        "risk": "low",
    }


def _render_trace(result) -> None:
    cols = st.columns(3)
    cols[0].metric("LLM calls", result.token_usage.get("llm_calls", 0))
    cols[1].metric("Token saving", f"{result.cost.get('token_savings_percent', 0)}%")
    cols[2].metric("Trace steps", len(result.trace))
    st.caption(
        f"Tokens: {result.token_usage.get('total', 0)} | Reported cost: ${result.cost.get('actual_usd', 0.0):.6f}"
    )
    for step in result.trace:
        icon = {"agent": "A", "retrieval": "R", "llm": "L", "tool": "T"}.get(step.kind, "S")
        with st.expander(f"{icon} {step.name} · {step.latency_ms:.0f} ms", expanded=False):
            st.json(
                {
                    "kind": step.kind,
                    "status": step.status,
                    "input": step.input,
                    "output": step.output,
                    "tokens": {"input": step.input_tokens, "output": step.output_tokens},
                    "cost_usd": step.cost_usd,
                }
            )


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "scan_result" not in st.session_state:
    st.session_state.scan_result = None
if "learning_result" not in st.session_state:
    st.session_state.learning_result = None
if "evaluation_report" not in st.session_state:
    st.session_state.evaluation_report = None

engine = get_engine()

if st.sidebar.button("Run proactive workforce scan", type="primary", use_container_width=True):
    with st.spinner("Scanning workforce data..."):
        st.session_state.scan_result = engine.run_scheduled_scan()
if st.sidebar.button("Run RL learning demo", use_container_width=True):
    with st.spinner("Running two feedback cycles..."):
        st.session_state.learning_result = engine.simulate_learning_cycle(
            _default_learning_anomaly(),
            feedback_decision="approved",
            outcome="resolved",
        )
if st.sidebar.button("Run evaluation harness", use_container_width=True):
    with st.spinner("Executing the 15-case evaluation harness..."):
        completed = subprocess.run(
            [sys.executable, "-m", "evaluation"],
            cwd=str(engine.settings.project_root / "src"),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            st.session_state.evaluation_report = json.loads(completed.stdout)
        else:
            st.session_state.evaluation_report = {
                "error": completed.stderr.strip() or completed.stdout.strip(),
            }
if st.sidebar.button("New conversation", use_container_width=True):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.last_result = None
    st.rerun()

pending_approvals = engine.pending_approvals()
diagnostics = engine.rl_diagnostics()
recent_incidents = engine.recent_incidents()
recent_experiences = engine.recent_experiences()

st.markdown(
    """
    <div class="hero-card">
        <span class="badge">Self-healing HR ops</span>
        <h1 style="margin: 0.55rem 0 0.3rem 0; font-size: 2.25rem;">Self-Healing HR Ops Platform</h1>
        <p style="margin: 0; max-width: 72ch; color: rgba(226, 232, 240, 0.88);">
            Reactive requests, scheduled scans, and upstream alerts all flow through the same graph:
            grounded policy retrieval, compliance vetoes, human approvals, episodic memory, and a
            persisted RL layer that changes future proposals after feedback.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

overview_cols = st.columns(5)
overview_cols[0].metric("Pending approvals", len(pending_approvals))
overview_cols[1].metric("Incidents stored", len(recent_incidents))
overview_cols[2].metric("RL experiences", len(recent_experiences))
overview_cols[3].metric("Compliance vetoes", diagnostics.get("veto_count", 0))
overview_cols[4].metric("Policy cases", len(CASES))

tabs = st.tabs(["Overview", "Reactive", "Proactive", "Learning Lab", "Observability"])

with tabs[0]:
    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown("### How the graph behaves")
        st.markdown(
            """
            - Reactive chat routes to policy or action specialists through the supervisor.
            - Scheduled scans and system alerts share one workflow state and one persistence layer.
            - Compliance rules are hard vetoes before any action is executed.
            - Human decisions and outcome feedback are stored as RL rewards and persist across restarts.
            """
        )
        st.info(
            f"Database: `{engine.settings.db_path}` | Policy index: `{engine.settings.index_path}` | "
            f"Auto-action threshold: `{engine.settings.auto_action_threshold:.2f}`"
        )
    with right:
        st.markdown("### Evaluation harness")
        st.dataframe(
            [
                {"case": case.name, "test": case.test, "reasoning": case.reasoning}
                for case in CASES
            ],
            use_container_width=True,
            hide_index=True,
        )
        if st.session_state.evaluation_report:
            report = st.session_state.evaluation_report
            if "error" in report:
                st.error(report["error"])
            else:
                st.success(f"{report['passed']}/{report['total']} cases passed")

with tabs[1]:
    st.markdown("### Reactive requests")
    st.caption("Ask a policy question or request an HR action. The response is grounded and traced.")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a policy question or request an HR action"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Working..."):
                result = engine.run(prompt, st.session_state.session_id)
            st.markdown(result.response)
            if result.citations:
                with st.expander("Retrieved policy evidence"):
                    for item in result.citations:
                        st.markdown(
                            f"**{item.chunk.heading}** · `{item.chunk.chunk_id}` · score `{item.score:.3f}`"
                        )
                        st.write(item.chunk.text)
        st.session_state.messages.append({"role": "assistant", "content": result.response})
        st.session_state.last_result = result
        st.rerun()

    result = st.session_state.last_result
    if result:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        _render_trace(result)
        st.markdown("</div>", unsafe_allow_html=True)

with tabs[2]:
    st.markdown("### Proactive scans")
    scan_result = st.session_state.scan_result
    if scan_result:
        top_cols = st.columns(4)
        top_cols[0].metric("Detected", len(scan_result.anomalies))
        top_cols[1].metric("Pending approval", scan_result.approvals_queued)
        top_cols[2].metric("Auto-executed", scan_result.actions_executed)
        top_cols[3].metric("Transitions", len(scan_result.transitions))
        st.caption(
            f"Token optimization: {scan_result.cost['actual_tokens']} actual vs {scan_result.cost['naive_baseline_tokens']} naive "
            f"({scan_result.cost['token_savings_percent']:.1f}% reduction)"
        )
        if scan_result.diagnostics.get("cumulative_reward"):
            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.markdown("**Cumulative RL reward**")
                st.line_chart(scan_result.diagnostics["cumulative_reward"])
            with chart_cols[1]:
                st.markdown("**Action distribution**")
                st.bar_chart(scan_result.diagnostics["action_distribution"])

        filtered_category = st.selectbox(
            "Filter anomalies by category",
            ["all", "payroll", "leave", "compliance"],
            index=0,
            key="scan_category_filter",
        )
        anomalies = [
            anomaly
            for anomaly in scan_result.anomalies
            if filtered_category == "all" or anomaly.category == filtered_category
        ]
        for anomaly in anomalies[:30]:
            st.markdown(
                f"**{anomaly.category.title()} · {anomaly.employee_id}**  \n"
                f"{anomaly.description}  \n"
                f"Confidence `{anomaly.confidence:.3f}` · Proposal `{anomaly.recommended_action}` · Status `{anomaly.status}`"
            )
            st.json(anomaly.evidence)
    else:
        st.info("Run a scheduled scan from the sidebar to populate proactive anomalies.")

with tabs[3]:
    st.markdown("### Learning lab")
    st.caption(
        "Use the same anomaly twice to see the contextual bandit warm-start from memory and human feedback."
    )
    demo_anomaly = _default_learning_anomaly()
    if st.button("Generate first occurrence", use_container_width=True, key="learning_first"):
        st.session_state.learning_result = {
            "first": engine.process_trigger("system", {"anomaly": demo_anomaly}, source="ui-demo"),
            "second": None,
            "anomaly": demo_anomaly,
        }

    learning_result = st.session_state.learning_result
    if learning_result and learning_result.get("first"):
        first = learning_result["first"]
        first_anomaly = first.anomalies[0]
        first_cols = st.columns(4)
        first_cols[0].metric("First confidence", f"{first_anomaly.confidence:.3f}")
        first_cols[1].metric("First action", first_anomaly.recommended_action)
        first_cols[2].metric("Queued approvals", first.approvals_queued)
        first_cols[3].metric(
            "Reward",
            first.diagnostics["cumulative_reward"][-1] if first.diagnostics["cumulative_reward"] else 0,
        )
        st.json(first_anomaly.evidence)

        pending = engine.pending_approvals()
        if pending:
            selected = st.selectbox(
                "Pending approval",
                pending,
                format_func=lambda item: f"{item['anomaly_id']} · {item['status']} · {item['reason']}",
                key="learning_pending_selector",
            )
            decision = st.selectbox(
                "Decision",
                ["approved", "rejected", "modified"],
                key="learning_decision",
            )
            chosen_action = st.selectbox(
                "If modified, new action",
                [
                    "auto-correct",
                    "escalate-to-manager",
                    "escalate-to-HR",
                    "flag-for-audit",
                    "no-action",
                ],
                key="learning_action_choice",
            )
            if st.button("Record human feedback", use_container_width=True, key="learning_feedback"):
                engine.record_feedback(
                    selected["anomaly_id"],
                    decision,
                    chosen_action if decision == "modified" else None,
                    comment="streamlit-feedback",
                )
                engine.record_outcome_feedback(
                    selected["anomaly_id"],
                    "resolved" if decision != "rejected" else "false_positive",
                    false_positive=decision == "rejected",
                    comment="streamlit-feedback",
                )
                st.success("Feedback stored and fed into the RL layer.")
                learning_result = st.session_state.learning_result
                learning_result["second"] = engine.process_trigger(
                    "system",
                    {"anomaly": {**learning_result["anomaly"], "anomaly_id": str(uuid.uuid4())}},
                    source="ui-demo",
                )
                st.session_state.learning_result = learning_result
                st.rerun()

        if learning_result.get("second"):
            second = learning_result["second"]
            second_anomaly = second.anomalies[0]
            comparison_cols = st.columns(4)
            comparison_cols[0].metric("Second confidence", f"{second_anomaly.confidence:.3f}")
            comparison_cols[1].metric("Second action", second_anomaly.recommended_action)
            comparison_cols[2].metric("Actioned", second.actions_executed)
            comparison_cols[3].metric(
                "Confidence lift", f"{second_anomaly.confidence - first_anomaly.confidence:+.3f}"
            )
            st.line_chart(
                {
                    "first": first.diagnostics["cumulative_reward"],
                    "second": second.diagnostics["cumulative_reward"],
                }
            )
            st.caption(
                "The second occurrence should show a higher confidence and a lower need for HITL after the reward update."
            )

    st.markdown("### Pending approvals inbox")
    if pending_approvals:
        st.dataframe(pending_approvals, use_container_width=True, hide_index=True)
    else:
        st.info("No pending approvals right now.")

with tabs[4]:
    st.markdown("### Observability and RL diagnostics")
    cols = st.columns(4)
    cols[0].metric("Cumulative reward steps", len(diagnostics.get("cumulative_reward", [])))
    cols[1].metric("Action types", len(diagnostics.get("action_distribution", {})))
    cols[2].metric("Human reward", diagnostics.get("reward_by_source", {}).get("human", 0.0))
    cols[3].metric("Outcome reward", diagnostics.get("reward_by_source", {}).get("outcome", 0.0))
    if diagnostics.get("cumulative_reward"):
        charts = st.columns(2)
        with charts[0]:
            st.markdown("**Cumulative reward curve**")
            st.line_chart(diagnostics["cumulative_reward"])
        with charts[1]:
            st.markdown("**Action distribution**")
            st.bar_chart(diagnostics["action_distribution"])
    st.markdown("**Reward by source**")
    st.bar_chart(diagnostics.get("reward_by_source", {}))
    st.markdown("**Recent RL experiences**")
    st.dataframe(recent_experiences, use_container_width=True, hide_index=True)
    st.markdown("**Recent incidents**")
    st.dataframe(recent_incidents, use_container_width=True, hide_index=True)
