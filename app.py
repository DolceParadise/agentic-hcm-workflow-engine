from __future__ import annotations

import uuid

import streamlit as st
from dotenv import load_dotenv

from hcm_engine import WorkflowEngine

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

with st.sidebar:
    st.title("Run trace")
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
            icon = {"agent": "A", "retrieval": "R", "llm": "L", "tool": "T"}.get(
                step.kind, "S"
            )
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
st.caption("Policy answers are grounded in the supplied corpus. Actions use mock HR APIs.")

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

