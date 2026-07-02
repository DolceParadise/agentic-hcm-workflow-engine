# Agentic HCM Workflow Engine

A reusable, traceable multi-agent engine for conversational HR operations. It routes requests
to a grounded policy RAG agent or an action agent backed by structured mock HR tools, while
persisting context across turns.

## What is included

- **Orchestrator agent:** deterministic, rule-first routing for policy, leave, and payslip intents.
- **Policy agent:** section-aware retrieval with `Qwen/Qwen3-Embedding-8B`, cosine similarity,
  relevance gating, and citation-constrained generation.
- **Action agent:** leave balance, leave application, and payslip tools with JSON schemas,
  validation, retry handling, and meaningful errors.
- **Conversation memory:** SQLite-backed messages, pending intent, and collected slots.
- **Observability:** per-step agent/retrieval/LLM/tool trace with input, output, status, latency,
  tokens, reported OpenRouter cost, and an optimization comparison.
- **Interfaces:** Streamlit chat UI with a live run trace and a CLI for automation.

All LLM inference uses OpenRouter's free `openai/gpt-oss-120b:free` model. All embeddings use
`Qwen/Qwen3-Embedding-8B` through `sentence-transformers`.

## Architecture

```text
User
  |
  v
Orchestrator (rule-first, no LLM)
  |-------------------------------|
  v                               v
Policy Agent                  Action Agent
  |                               |
Qwen embedding retrieval      Structured mock tools
  |                               |
gpt-oss grounded answer       balance / leave / payslip
  |-------------------------------|
                  |
           SQLite state + trace
```

The policy corpus is split at Markdown section headings. Oversized sections are windowed at
220 words with a 35-word overlap. Heading text is embedded with each body chunk so short,
topic-oriented questions retain strong semantic signals. Embeddings are normalized, persisted
to `.runtime/policy_index.npz`, and rebuilt only when the corpus or embedding model changes.

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

Tool definitions live in `src/hcm_engine/tools.py` as OpenAI-style JSON schemas. Tool calls and
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
app.py                     Streamlit UI
data/                      Supplied PDF and normalized policy corpus
src/hcm_engine/
  agents.py                Orchestrator, policy, and action agents
  config.py                Environment-backed settings
  engine.py                Public workflow facade
  llm.py                   OpenRouter client
  rag.py                   Chunking, Qwen embeddings, persisted vector index
  state.py                 SQLite conversation state
  tools.py                 Mock APIs and JSON schemas
  tracing.py               Per-step observability
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
