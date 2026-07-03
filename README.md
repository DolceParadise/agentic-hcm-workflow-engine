# Agentic HCM Workflow Engine

A reusable, traceable self-healing HR operations engine. It supports conversational,
scheduled, and upstream-system triggers and persists every graph transition, incident,
approval, and feedback outcome.

## What is included

- **`supervisor_agent`:** deterministic, rule-first routing for policy, leave, and payslip intents.
- **Policy agent:** section-aware retrieval with `Qwen/Qwen3-Embedding-8B`, cosine similarity,
  relevance gating, and citation-constrained generation.
- **Action agent:** leave balance, leave application, and payslip tools with JSON schemas,
  validation, retry handling, and meaningful errors.
- **Anomaly agent:** robust peer-cohort payroll scoring, leave-limit review signals, mandatory
  training checks, and overtime-cap checks.
- **Compliance agent:** deterministic hard vetoes that model confidence and feedback cannot
  override.
- **HITL and learning:** SQLite approval queue plus a conservative contextual bandit that uses
  human approvals, rejection reasons, and outcome feedback to adjust proposals.
- **Episodic vector memory:** dependency-free SQLite vector store using stable feature-hashed
  embeddings; stores incident context, action, outcome, and reward and retrieves similar cases.
- **Conversation memory:** SQLite-backed messages, pending intent, and collected slots.
- **Observability:** per-step agent/retrieval/LLM/tool trace with input, output, status, latency,
  tokens, reported OpenRouter cost, and an optimization comparison.
- **Interfaces:** Streamlit dashboard with chat, proactive scans, HITL inbox, RL learning lab,
  live run trace, and a CLI for automation.

All LLM inference uses OpenRouter's free `openai/gpt-oss-120b:free` model. All embeddings use
`Qwen/Qwen3-Embedding-8B` through `sentence-transformers`.

## Architecture

```text
Reactive / Scheduled / System trigger
  |
  v
LangGraph `supervisor_agent` + shared graph state
  |-- `policy_agent` (RAG)
  |-- `anomaly_detection_agent` (scan and score)
  |-- `memory_agent` (episodic retrieval + RL warm start)
  |-- `compliance_agent` (hard veto)
  `-- `action_agent` (guarded execution / HITL queue)
                  |
      SQLite transitions + incidents + feedback
```

The policy corpus is split at Markdown section headings. Oversized sections are windowed at
220 words with a 35-word overlap. Heading text is embedded with each body chunk so short,
topic-oriented questions retain strong semantic signals. Embeddings are normalized, persisted
to `data/policy_index.npz`, and rebuilt only when the corpus or embedding model changes.

## Setup

Python 3.11 or 3.12 is recommended. Qwen3-Embedding-8B is a large model, so the first run
downloads substantial model weights and works best on a machine with adequate RAM or GPU memory.

```bash
cd /Users/pratik/Documents/agentic-hcm-workflow-engine
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env`, then run:

```bash
streamlit run app.py
```

Open the URL printed by Streamlit, normally `http://localhost:8501`.

CLI usage:

```bash
hcm-agent "What is the policy on outside employment?" --trace
hcm-agent "Apply for annual leave from 2026-07-06 to 2026-07-08" \
  --session-id demo-session --trace
```

Use the same `--session-id` on later commands to continue a conversation.

Programmatic trigger usage:

```python
engine = WorkflowEngine()
scheduled = engine.run_scheduled_scan()
system = engine.process_trigger("system", upstream_alert_payload, source="payroll-engine")
engine.record_feedback(anomaly_id, "approved")
```

The upstream payload contract is documented in `data/mock_api_spec.yaml`; the 12 hard-veto
rules live in `data/compliance_rules.json`, never in a prompt.

## Anomaly scoring and action policy

- Payroll compares salary within department and five-year experience bands. A robust modified
  z-score of 2.5 starts an audit-only signal; confidence rises from 0.72 with extremity.
- Leave over 15 days in the supplied review period starts at 0.65 confidence and increases
  with excess days. It is explicitly a review signal, not a misconduct finding.
- Missing mandatory training scores 0.99. Overtime above the 48-hour cycle cap starts at 0.88.
- The auto-action threshold is 0.90 (`HCM_AUTO_ACTION_THRESHOLD`). Anything below it enters
  HITL. Payroll changes, compliance incidents, high-risk changes, and non-correction proposals
  always enter HITL regardless of confidence.

Compliance vetoes receive reward `-1.0`, rejected proposals `-1.0`, modified proposals `+0.5`,
approved proposals `+1.0`, safe automated execution `+0.05` to `+0.25` depending on the action,
and outcome feedback rewards or penalizes recurrence and false positives. Positive, semantically
similar episodes can raise recurrence confidence by up to 0.12; negative/vetoed episodes suppress
the same proposal. Compliance is evaluated after learning, so no learned policy can bypass a veto.

## Evaluation and Loom diagnostics

Run the 15-case evaluation harness (happy paths, edge cases, adversarial input, memory, and RL):

```bash
PYTHONPATH=src python -m evaluation
```

It prints pass/fail plus reasoning for every case. The full pytest suite currently contains 20
tests. The Streamlit proactive-scan panel plots cumulative reward and action distribution and
shows the per-run token baseline. Deterministic scans use zero LLM tokens versus the explicit
naive baseline of three 450-token calls (1,350 tokens), a 100% reduction; conversational RAG
runs continue reporting measured provider tokens and cost.

For the recurrence walkthrough, submit a low-confidence `auto-correct` leave anomaly, approve
it with `engine.record_feedback(...)`, then submit the semantically identical anomaly under a
new ID. The first queues for HITL at 0.85; the second retrieves the approved episode, rises above
0.90, and resolves without another approval. This behavior is covered by the `warm-start` case.

## Example conversations

```text
User: Apply for annual leave starting 2026-07-06
Agent: Please provide the end date (YYYY-MM-DD).
User: It ends 2026-07-08
Agent: Leave request LV-1001 was submitted for 3 working day(s), from 2026-07-06 to 2026-07-08.
```

```text
User: Can I use confidential company information for another employer?
Agent: No... [policy-003] [policy-010]
```

Policy evidence is displayed below the answer in the UI. If retrieval scores do not meet the
configured threshold, the agent refuses to guess and directs the employee to HR.

## Tool contracts

Tool definitions live in `src/tools.py` as OpenAI-style JSON schemas. Tool calls and
results are recorded in the trace. Transient `TimeoutError` and `ConnectionError` failures are
retried; validation and business errors are returned as actionable messages without retrying.

The implementation is intentionally mock-only: no real leave, payroll, or employee system is
called. Replace `MockHRTools` behind the same method contracts to connect production services.

## Cost optimization

A naive agentic baseline calls an LLM for:

1. intent routing,
2. specialist execution or tool selection,
3. response composition.

This engine removes calls 1 and 3 for recognizable actions and removes call 1 for policy
questions. It also sends only the top relevant policy chunks, not full conversation and corpus.
Each run reports actual OpenRouter usage plus this transparent estimate:

```text
naive baseline tokens = max(observed input tokens, 450) x 3 calls
optimized tokens      = observed input + output tokens
saving %              = (naive - optimized) / naive x 100
```

For a representative policy run with 600 prompt tokens and 120 completion tokens:

```text
Naive:      600 x 3 = 1,800 tokens
Optimized:  600 + 120 = 720 tokens
Reduction:  60.0%
```

Action requests that are fully structured use zero LLM calls. Ambiguous leave messages use one
small JSON extraction call. The UI shows the measured values for every run, making the claimed
reduction inspectable rather than static.

## Tests

Tests use a deterministic fake embedder and LLM, so they do not download model weights or call
OpenRouter.

```bash
pytest
ruff check .
```

Coverage includes chunking and retrieval quality, tool schemas, leave balance mutation,
business errors, transient retries, policy traces, no-LLM routing, and multi-turn slot memory.

## Project structure

```text
data/                      Supplied PDF and normalized policy corpus
app.py                     Streamlit UI
src/
  agents/
    supervisor_agent.py      Trigger routing agent
    policy_agent.py          Grounded RAG agent
    action_agent.py          Guarded HR action agent
    anomaly_detection_agent.py  Workforce anomaly detection agent
    compliance_agent.py      File-backed hard-veto agent
  config.py                Environment-backed settings
  engine.py                Public facade + conversational LangGraph
  llm.py                   OpenRouter client
  rag.py                   Chunking, Qwen embeddings, persisted vector index
  state.py                 SQLite conversation state
  tools.py                 Mock APIs and JSON schemas
  tracing.py               Per-step observability
  workflow.py              Self-healing LangGraph
tests/                     Offline deterministic test suite
```

## Safety and production notes

- The included document is Apple's 2017 business-conduct policy, not a comprehensive HR
  benefits policy. Questions such as maternity leave duration are correctly rejected when the
  answer is absent.
- Real deployments should add authentication, employee authorization, encryption, PII
  redaction, audit retention controls, tool idempotency keys, human approval for sensitive
  actions, and a production vector database.
- Free OpenRouter model availability and rate limits are controlled by OpenRouter. The client
  returns a clear model-unavailable message when the request cannot be completed.
