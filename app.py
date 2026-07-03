# ruff: noqa: E501
from __future__ import annotations

import uuid
from dataclasses import asdict

import streamlit as st
from dotenv import load_dotenv

from engine import WorkflowEngine

load_dotenv()
st.set_page_config(page_title="Self-Healing HR Ops", page_icon="✦", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
    :root { --ink:#16261f; --muted:#64736c; --green:#176b4d; --mint:#dff4e8; --paper:#f6f5ef; --line:#dfe4dc; --amber:#e3a72f; }
    .stApp { background: var(--paper); color: var(--ink); font-family:'DM Sans',sans-serif; }
    h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; }
    [data-testid="stSidebar"] { background:#10271e; border-right:0; }
    [data-testid="stSidebar"] * { color:#eef8f1; }
    [data-testid="stSidebar"] .stButton button { background:#f4fff7; color:#123d2c !important; border:1px solid #b9efca; font-weight:700; }
    [data-testid="stSidebar"] .stButton button * { color:#123d2c !important; }
    [data-testid="stSidebar"] .stButton button:hover { background:#b9efca; border-color:#b9efca; }
    [data-testid="stSidebar"] .stButton button[kind="primary"] { background:#b9efca; color:#10271e !important; border:1px solid #b9efca; }
    .stButton button, .stFormSubmitButton button { background:#176b4d; color:#fff; border:1px solid #176b4d; font-weight:700; }
    .stButton button:hover, .stFormSubmitButton button:hover { background:#0f5038; color:#fff; border-color:#0f5038; }
    [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); padding:15px 17px; border-radius:14px; box-shadow:0 4px 18px rgba(24,46,36,.04); }
    [data-testid="stMetricLabel"] { color:var(--muted); }
    [data-testid="stMetricValue"] { font-family:'Manrope',sans-serif; color:var(--ink); }
    .hero { background:#173d2e; color:white; border-radius:22px; padding:30px 34px; margin:6px 0 20px; position:relative; overflow:hidden; }
    .hero:after { content:'✦'; position:absolute; right:34px; top:5px; font-size:130px; color:rgba(185,239,202,.12); }
    .eyebrow { color:#9fe1b7; font-size:.72rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
    .hero h1 { color:white; margin:.4rem 0 .45rem; font-size:2.35rem; }
    .hero p { color:#cfe1d7; max-width:72ch; margin:0; font-size:1rem; }
    .section-kicker { color:var(--green); font-size:.72rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; margin-bottom:.1rem; }
    .agent-strip { display:grid; grid-template-columns:repeat(5,1fr); gap:8px; margin:12px 0 22px; }
    .agent-node { background:#fff; border:1px solid var(--line); border-radius:12px; padding:12px; font-size:.82rem; }
    .agent-node b { display:block; color:var(--green); margin-bottom:2px; }
    .callout { background:#eef7f1; border-left:4px solid #3b9a6c; padding:14px 16px; border-radius:0 12px 12px 0; margin:12px 0; }
    .signal { display:inline-block; padding:4px 9px; background:#e6f2ea; color:#176b4d; border-radius:99px; font-size:.72rem; font-weight:700; }
    div[data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid var(--line); }
    button[data-baseweb="tab"] { background:#e5ece7 !important; color:#29483b !important; border-radius:10px 10px 0 0; padding:10px 18px !important; opacity:1 !important; }
    button[data-baseweb="tab"] p { color:#29483b !important; font-weight:700 !important; }
    button[data-baseweb="tab"][aria-selected="true"] { background:#176b4d !important; color:#fff !important; }
    button[data-baseweb="tab"][aria-selected="true"] p { color:#fff !important; }
    button[data-baseweb="tab"]:hover { background:#cfe2d6 !important; }
    button[data-baseweb="tab"][aria-selected="true"]:hover { background:#176b4d !important; }
    [data-testid="stRadio"] > label p { color:#16261f !important; font-weight:700 !important; }
    [data-testid="stRadio"] [role="radiogroup"] label { background:#fff; border:1px solid #b7c6bd; border-radius:10px; padding:8px 14px; margin-right:8px; }
    [data-testid="stRadio"] [role="radiogroup"] label p { color:#16261f !important; font-weight:700 !important; }
    .stDataFrame { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_engine() -> WorkflowEngine:
    return WorkflowEngine()


def demo_anomaly(*, veto: bool = False) -> dict[str, object]:
    if veto:
        return {
            "anomaly_id": str(uuid.uuid4()), "employee_id": "EMP-042", "category": "payroll",
            "description": "AED 800 payroll correction requested without manager-tier approval",
            "confidence": 0.97, "recommended_action": "auto-correct", "risk": "high",
            "evidence": {"correction_amount": 800, "approval_tier": "analyst", "retroactive_days": 12},
        }
    return {
        "anomaly_id": str(uuid.uuid4()), "employee_id": "EMP-218", "category": "leave",
        "description": "Repeated Q1 leave threshold anomaly", "confidence": 0.84,
        "recommended_action": "auto-correct", "risk": "low",
        "evidence": {"leave_days": 18, "review_threshold": 15, "policy_refs": ["Annual Leave"]},
    }


def anomaly_rows(result) -> list[dict]:
    rows = []
    for anomaly in result.anomalies:
        evidence = anomaly.evidence
        current_payroll = evidence.get("salary") if anomaly.category == "payroll" else None
        mean_payroll = evidence.get("cohort_mean") if anomaly.category == "payroll" else None
        leave_days = evidence.get("leave_days") if anomaly.category == "leave" else None
        rows.append(
            {
                "Employee": anomaly.employee_id,
                "Signal": anomaly.category.title(),
                "Existing payroll": f"${current_payroll:,.0f}" if current_payroll else "—",
                "Mean payroll": f"${mean_payroll:,.0f}" if mean_payroll else "—",
                "Leave days": f"{leave_days:g}" if leave_days is not None else "—",
                "Confidence": f"{anomaly.confidence:.0%}",
                "Recommended action": anomaly.recommended_action,
                "Status": anomaly.status,
                "Finding": anomaly.description,
            }
        )
    return rows


def render_workflow(result) -> None:
    st.markdown("#### Processing history")
    for transition in result.transitions:
        reward = "" if transition.reward is None else f" · reward {transition.reward:+.2f}"
        with st.expander(f"Step {transition.sequence:02d} — {transition.event.replace('_', ' ').title()}{reward}"):
            st.json({
                "latency_ms": transition.latency_ms, "rl_action": transition.rl_action,
                "tool_calls": transition.tool_calls, "output": transition.output,
            })


def render_request_trace(result) -> None:
    a, b, c, d = st.columns(4)
    a.metric("Request type", result.intent.replace("_", " ").title())
    b.metric("AI calls", result.token_usage.get("llm_calls", 0))
    c.metric("Processing saved", f"{result.cost.get('token_savings_percent', 0):.0f}%")
    d.metric("Processing steps", len(result.trace))
    for step in result.trace:
        with st.expander(f"{step.name.replace('_', ' ').title()} · {step.latency_ms:.0f} ms"):
            st.json(step.to_dict())


defaults = {
    "session_id": str(uuid.uuid4()), "messages": [], "last_result": None,
    "scan_result": None, "system_result": None, "learning_result": None,
    "request_text": "",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

engine = get_engine()

st.sidebar.markdown("## ✦ HR OPS")
st.sidebar.caption("Self-healing workforce operations")
st.sidebar.markdown("---")
st.sidebar.markdown("**Quick actions**")
if st.sidebar.button("Simulate a payroll alert", width="stretch"):
    with st.spinner("Routing payroll-engine alert…"):
        st.session_state.system_result = engine.process_trigger(
            "system", {"anomaly": demo_anomaly(veto=True)}, source="payroll-engine"
        )
if st.sidebar.button("Run two-cycle learning demo", width="stretch"):
    with st.spinner("Applying feedback and replaying incident…"):
        st.session_state.learning_result = engine.simulate_learning_cycle(
            demo_anomaly(), feedback_decision="approved", outcome="resolved"
        )
st.sidebar.markdown("---")

pending = engine.pending_approvals()
diagnostics = engine.rl_diagnostics()
incidents = engine.recent_incidents()
experiences = engine.recent_experiences()

st.markdown("""
<div class="hero"><h1>Self-Healing HR Operations</h1></div>
""", unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Workforce", "1,000")
m2.metric("Open reviews", len(pending))
m3.metric("Incidents learned", len(incidents))
m4.metric("Policy rewards", len(experiences))
m5.metric("Hard vetoes", diagnostics.get("veto_count", 0))

tabs = st.tabs(["Ask HR", "Detection", "Review queue", "Learning"])

with tabs[0]:
    st.markdown("### Ask a grounded HR question")
    with st.form("request_form"):
        prompt = st.text_area("Request", key="request_text", height=90)
        submitted = st.form_submit_button("Submit request", type="primary")
    if submitted and prompt.strip():
        with st.spinner("Retrieving policy and selecting a safe action…"):
            result = engine.run(prompt, st.session_state.session_id)
        st.session_state.last_result = result
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": result.response})
    if st.session_state.last_result:
        result = st.session_state.last_result
        st.markdown("#### Agent response")
        st.success(result.response)
        if result.citations:
            st.markdown("#### Grounding evidence")
            for hit in result.citations:
                with st.expander(f"{hit.chunk.heading} · {hit.chunk.chunk_id} · similarity {hit.score:.3f}"):
                    st.write(hit.chunk.text)
        render_request_trace(result)

with tabs[1]:
    st.markdown("### Proactive anomaly detection")
    if st.button("Run workforce scan", type="primary", width="stretch"):
        with st.spinner("Scanning 1,000 workforce records…"):
            st.session_state.scan_result = engine.run_scheduled_scan()
        st.rerun()
    result = st.session_state.scan_result
    if result:
        a, b, c, d = st.columns(4)
        a.metric("Signals detected", len(result.anomalies))
        b.metric("Auto-executed", result.actions_executed)
        c.metric("Human review", result.approvals_queued)
        d.metric("Processing saved", f"{result.cost['token_savings_percent']:.0f}%")
        category = st.segmented_control("Signal type", ["All", "Payroll", "Leave", "Compliance"], default="All")
        rows = anomaly_rows(result)
        if category != "All":
            rows = [row for row in rows if row["Signal"] == category]
        st.dataframe(rows, width="stretch", hide_index=True, height=360)
        with st.expander("View processing details"):
            render_workflow(result)
    else:
        st.info("Select **Run workforce scan** to check the 1,000-record employee dataset.")
    if st.session_state.system_result:
        st.markdown("### Latest upstream alert")
        system_result = st.session_state.system_result
        alert = system_result.anomalies[0]
        st.error(f"{alert.description} — {alert.status}")
        st.json(asdict(alert))
        render_workflow(system_result)

with tabs[2]:
    st.markdown("### Review queue")
    pending = engine.pending_approvals()
    if pending:
        selected = st.selectbox("Select a case", pending, format_func=lambda x: f"{x['anomaly_id'][:8]} · {x['reason']}")
        matching = next((item for item in engine.recent_incidents(100) if item["anomaly_id"] == selected["anomaly_id"]), None)
        if matching:
            c1, c2, c3 = st.columns([0.65, 1.05, 1.8])
            c1.metric("Confidence", f"{float(matching['confidence']):.0%}")
            c2.metric("Proposal", matching["proposed_action"])
            c3.metric("Status", matching["status"])
            st.markdown(f"**Context:** {matching['description']}")
        decision = st.radio(
            "Decision",
            ["Approved", "Modified", "Rejected"],
            horizontal=True,
            key="review_decision",
        ).lower()
        with st.form("approval_form"):
            modified_action = None
            if decision == "modified":
                modified_action = st.selectbox(
                    "Replacement action",
                    [
                        "flag-for-audit",
                        "escalate-to-manager",
                        "escalate-to-HR",
                        "no-action",
                        "auto-correct",
                    ],
                )
            note_label = "Rejection reason" if decision == "rejected" else "Reviewer note"
            note_placeholder = (
                "Explain why this proposal should not proceed"
                if decision == "rejected"
                else "Add context for this decision (optional)"
            )
            reason = st.text_input(note_label, placeholder=note_placeholder)
            save = st.form_submit_button("Submit decision", type="primary")
        if save:
            engine.record_feedback(selected["anomaly_id"], decision, modified_action, reason)
            engine.record_outcome_feedback(
                selected["anomaly_id"], "false_positive" if decision == "rejected" else "resolved",
                false_positive=decision == "rejected", comment=reason,
            )
            st.success("Decision saved. The system will use this feedback in future cases.")
            st.rerun()
    else:
        st.info("No cases are waiting. Run a scan or inject the payroll alert to create review work.")

with tabs[3]:
    st.markdown("### The policy changes when humans teach it")
    learning = st.session_state.learning_result
    if learning:
        first, second = learning["first"], learning["second"]
        a1, a2 = first.anomalies[0], second.anomalies[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Before confidence", f"{a1.confidence:.0%}")
        c2.metric("After confidence", f"{a2.confidence:.0%}")
        c3.metric("Before", a1.status)
        c4.metric("After", a2.status)
        st.markdown("<div class='callout'><b>What changed?</b><br>The system remembered the approved resolution. When a similar case appeared again, it raised its confidence and handled the case with less manual review.</div>", unsafe_allow_html=True)
    else:
        st.info("Use **Run two-cycle learning demo** in the sidebar for a before/after comparison.")
    diagnostics = engine.rl_diagnostics()
    if diagnostics.get("cumulative_reward"):
        left, right = st.columns(2)
        with left:
            st.markdown("#### Learning progress")
            st.line_chart(
                diagnostics["cumulative_reward"],
                x_label="Feedback cycle",
                y_label="Cumulative reward",
                height=260,
            )
        with right:
            st.markdown("#### Recommended actions")
            st.bar_chart(diagnostics.get("action_distribution", {}), height=260)
    st.markdown("#### Recent feedback history")
    st.dataframe(engine.recent_experiences(), width="stretch", hide_index=True)
