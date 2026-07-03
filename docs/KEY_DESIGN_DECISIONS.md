# Key Design Decisions

This document records the choices that shape the Self-Healing HR Ops platform, why they were
made, and the tradeoffs they introduce. Configuration values described here are defaults and can
be overridden where an environment variable is available.

## 1. Shared-state multi-agent orchestration

The platform uses LangGraph to coordinate a supervisor and specialist agents. Agents never call
one another directly; every handoff passes through shared graph state. This makes routing and
intermediate decisions observable, reproducible, and persistable rather than hiding them inside
an unstructured chain of model calls.

Two related graphs are used:

- Conversational requests route from the supervisor to either the policy or action specialist.
- Scheduled scans and upstream alerts move through detection, memory and learning, compliance,
  guarded action, and supervisor finalization.

SQLite stores graph transitions with agent input, output, latency, token use, cost, selected RL
action, reward, and tool calls.

## 2. Deterministic routing before model routing

Recognizable policy, leave, balance, and payslip intents are routed with deterministic rules.
The language model is used only when language is ambiguous or grounded response generation adds
value. This lowers latency and cost and reduces the chance that a model selects an unsafe tool.

## 3. Policy retrieval and refusal behavior

Policy retrieval uses `Qwen/Qwen3-Embedding-8B` embeddings and cosine similarity. The corpus is
split on Markdown headings; long sections use 220-word windows with a 35-word overlap. Headings
are embedded with the body because short policy questions often resemble section names more than
individual sentences.

The default retrieval settings are:

- top results: `4`
- minimum relevance score: `0.28`
- persisted index: `data/policy_index.npz`

If retrieval does not meet the relevance threshold, the platform refuses to invent an answer and
directs the employee to HR. The supplied policy is Apple's 2017 business-conduct policy, not a
complete benefits handbook, so unsupported topics such as maternity-leave duration should fail
safely.

## 4. Explainable anomaly thresholds

Detection is deterministic so every anomaly can be explained and reproduced.

### Payroll

Employees are compared within department and five-year experience bands. Cohorts with fewer than
five members are ignored. A robust deviation score is calculated using the cohort median and
median absolute deviation, with a minimum scale of three percent of the median to avoid unstable
scores in tightly grouped cohorts.

- detection threshold: modified z-score `2.5`
- starting confidence: `0.72`
- confidence growth: `0.06 × (score - 2.5)`
- maximum confidence: `0.99`
- initial action: `flag-for-audit`

Payroll signals are review indicators, not proof that compensation is incorrect. The UI shows
the employee's current payroll and cohort mean to give reviewers immediate context.

### Leave

- review threshold: more than `15` leave days in the supplied period
- starting confidence: `0.65`
- confidence growth: `(leave_days - 15) / 30`
- maximum confidence: `0.98`
- initial action: `escalate-to-manager`

Crossing the threshold triggers review; it is not classified as misconduct.

### Compliance

- incomplete mandatory training confidence: `0.99`
- overtime threshold: more than `48` hours per cycle
- overtime starting confidence: `0.88`
- overtime confidence grows logarithmically with excess hours and is capped at `0.99`
- initial action: `escalate-to-HR`

## 5. Automatic action threshold and human review

The default automatic-action threshold is `0.90`, configurable through
`HCM_AUTO_ACTION_THRESHOLD`. Proposals below it enter the human review queue. A compliance veto
also forces review regardless of confidence.

Each new full-workforce scan supersedes the previous active scan queue so Open Reviews represents
current findings instead of accumulating every historic scan. Superseded incidents remain in the
database for audit.

Reviewers can approve, reject, or modify a proposal and attach a reason. The selected decision,
replacement action, comment, and eventual outcome are persisted and become learning signals.

## 6. Compliance is a hard veto after learning

Compliance rules live in `data/compliance_rules.json`, not in model prompts. The file contains 12
versioned rules covering payroll approvals, high-risk automation, overtime, training, leave,
probation, bank details, termination, medical information, and retroactive corrections.

Compliance evaluates the action after memory and learning adjust the proposal. This ordering is
intentional: a learned policy can improve recommendations but can never bypass a hard rule. A
veto queues human review and contributes a `-1.0` learning reward.

## 7. Contextual bandit instead of full policy training

Action selection is a small discrete decision with sparse, immediate feedback. A reward-weighted
contextual bandit is therefore more sample-efficient, inspectable, and practical than PPO or
fine-tuning a language model.

The learner groups experience by anomaly category and action. Candidate scores use mean reward
with one unit of conservative smoothing. Only positively scoring actions replace the detector's
initial proposal. Learned confidence adjustments are capped to `±0.12` so limited feedback cannot
overwhelm detector evidence.

Reward signals are:

| Signal | Reward |
| --- | ---: |
| Human approved | `+1.0` |
| Human modified | `+0.5` |
| Human rejected | `-1.0` |
| Compliance veto | `-1.0` |
| Outcome resolved | `+0.35` |
| Outcome validated | `+0.20` |
| Outcome recurred | `-0.80` |
| Outcome false positive | `-0.60` |
| Explicit recurrence flag | additional `-0.10` |
| Explicit false-positive flag | additional `-0.15` |
| Safe automatic execution | `+0.05` to `+0.25`, depending on action |

## 8. Episodic memory warm-starts recurring cases

Resolved incidents are stored with context, selected action, outcome, and reward. A stable
256-dimensional feature-hashed vector represents each incident in SQLite. At proposal time, the
three most similar incidents are retrieved from the latest 500 records. Matches with similarity
of at least `0.55` contribute reward-weighted evidence to the contextual bandit.

This dependency-free store keeps the demo portable while preserving a replaceable vector-store
contract. A production deployment can substitute Chroma or Qdrant without changing the workflow.

## 9. Guarded tool contracts and failure handling

HR actions use OpenAI-style JSON schemas defined in `src/tools.py`. Arguments are validated before
execution, every call and result is traced, and transient `TimeoutError` or `ConnectionError`
failures are retried. Validation and business-rule failures return actionable errors without
repeating an unsafe request.

The included implementation is mock-only: it does not call a real payroll, leave, or employee
system. `MockHRTools` can be replaced behind the same contracts. Production tools should also add
idempotency keys and authorization checks.

## 10. Persistence and queue semantics

SQLite is used for conversation state, incidents, approvals, feedback, outcome feedback, RL
experiences, episodic memory, and graph traces. This gives the demo restart-safe behavior without
an external service. It also makes feedback effects inspectable with ordinary SQL.

For production volume and concurrency, separate transactional storage and a dedicated vector
database would be more appropriate.

## 11. Observability and evaluation

Every workflow transition records the responsible agent, input, output, latency, model tokens,
cost, selected action, reward, and tool calls. The UI plots cumulative reward and action counts.

The fifteen-case evaluation harness covers grounded policy answers, multi-turn actions, malformed
input, adversarial vetoes, detector coverage, external rules, veto penalties, semantic memory,
learning warm-start, structured traces, diagnostics, cost reduction, and persistence. The full
test suite currently contains 25 deterministic offline tests.

## 12. Cost optimization baseline

The naive baseline assumes three model calls: routing, specialist/tool selection, and response
composition. The engine removes routing and composition calls for recognizable structured actions
and sends only top policy chunks rather than the full corpus and conversation.

```text
naive baseline tokens = max(observed input tokens, 450) × 3
optimized tokens      = observed input + output tokens
saving percentage     = (naive - optimized) / naive × 100
```

Scheduled scans use deterministic specialists and therefore report zero model tokens against the
1,350-token baseline. Fully structured action requests can also use zero model calls; ambiguous
requests use one constrained extraction call.

## 13. Production boundaries

Before connecting this project to real HR systems, add authentication, employee-level
authorization, encryption, PII redaction, audit-retention controls, tool idempotency, secrets
management, rate limiting, and human approval for sensitive actions. Replace the bundled policy
with an authoritative HR corpus and validate jurisdiction-specific employment rules.

The configured OpenRouter model is `openai/gpt-oss-120b:free`; availability and rate limits are
controlled by OpenRouter. The client returns a clear failure when the model is unavailable rather
than silently continuing with an ungrounded action.
