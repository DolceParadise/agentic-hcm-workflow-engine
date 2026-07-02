from __future__ import annotations

import uuid

import streamlit as st
from dotenv import load_dotenv

from engine import WorkflowEngine

load_dotenv()
st.set_page_config(page_title="Darwinbox HR Agent", page_icon="DB", layout="wide")


@st.cache_resource
def get_engine() -> WorkflowEngine:
    return WorkflowEngine()


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "scan_result" not in st.session_state:
    st.session_state.scan_result = None

with st.sidebar:
    st.title("Run trace")
    if st.button("Run proactive workforce scan", type="primary", use_container_width=True):
        with st.spinner("Scanning workforce data..."):
            st.session_state.scan_result = get_engine().run_scheduled_scan()
    if st.button("New conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()
    result = st.session_state.last_result
    if result:
        cols = st.columns(2)
        cols[0].metric("LLM calls", result.token_usage["llm_calls"])
        cols[1].metric("Token saving", f"{result.cost['token_savings_percent']}%")
        st.caption(
            f"Tokens: {result.token_usage['total']} | "
            f"Reported cost: ${result.cost['actual_usd']:.6f}"
        )
        for step in result.trace:
            icon = {"agent": "A", "retrieval": "R", "llm": "L", "tool": "T"}.get(step.kind, "S")
            with st.expander(f"{icon} {step.name} · {step.latency_ms:.0f} ms"):
                st.json(
                    {
                        "kind": step.kind,
                        "status": step.status,
                        "input": step.input,
                        "output": step.output,
                        "tokens": {
                            "input": step.input_tokens,
                            "output": step.output_tokens,
                        },
                        "cost_usd": step.cost_usd,
                    }
                )

st.title("Darwinbox HR Agent")
st.caption(
    "Self-healing HR operations: grounded policy answers, proactive anomaly detection, "
    "compliance vetoes, and human approvals."
)

scan_result = st.session_state.scan_result
if scan_result:
    with st.expander(
        f"Latest proactive scan · {len(scan_result.anomalies)} anomalies",
        expanded=True,
    ):
        cols = st.columns(3)
        cols[0].metric("Detected", len(scan_result.anomalies))
        cols[1].metric("Pending approval", scan_result.approvals_queued)
        cols[2].metric("Auto-corrected", scan_result.actions_executed)
        st.caption(
            f"Token optimization: {scan_result.cost['actual_tokens']} actual vs "
            f"{scan_result.cost['naive_baseline_tokens']} naive "
            f"({scan_result.cost['token_savings_percent']:.1f}% reduction)"
        )
        diagnostics = scan_result.diagnostics
        if diagnostics.get("cumulative_reward"):
            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.markdown("**Cumulative RL reward**")
                st.line_chart(diagnostics["cumulative_reward"])
            with chart_cols[1]:
                st.markdown("**Action distribution**")
                st.bar_chart(diagnostics["action_distribution"])
        st.caption(f"Compliance vetoes learned from: {diagnostics.get('veto_count', 0)}")
        for anomaly in scan_result.anomalies[:50]:
            st.markdown(
                f"**{anomaly.category.title()} · Employee {anomaly.employee_id}** — "
                f"{anomaly.description}  \n"
                f"Confidence `{anomaly.confidence:.3f}` · "
                f"Proposal `{anomaly.recommended_action}` · Status `{anomaly.status}`"
            )
            st.json(anomaly.evidence)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a policy question or request an HR action"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Working..."):
            result = get_engine().run(prompt, st.session_state.session_id)
        st.markdown(result.response)
        if result.citations:
            with st.expander("Retrieved policy evidence"):
                for item in result.citations:
                    st.markdown(
                        f"**{item.chunk.heading}** · `{item.chunk.chunk_id}` · "
                        f"score `{item.score:.3f}`"
                    )
                    st.write(item.chunk.text)
    st.session_state.messages.append({"role": "assistant", "content": result.response})
    st.session_state.last_result = result
    st.rerun()
