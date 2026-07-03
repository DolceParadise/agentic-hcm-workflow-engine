# Self-Healing HR Ops demo transcript

Target length: 7–9 minutes. Text in quotation marks is narration. Bracketed text is an on-screen action.

## 1. Introduction — Ask HR

[Start on **Ask HR**.]

“This is the Self-Healing HR Operations platform. It answers employee questions, monitors workforce data, identifies potential issues, coordinates human review, and learns from previous decisions.

The summary at the top shows the one-thousand-person workforce, open reviews, learned incidents, policy rewards, and compliance vetoes.

Work enters the platform in three ways: a request from an employee or HR user, an automatic workforce scan, or an alert from another system such as payroll. Each follows the same controlled process, with policy checks, compliance safeguards, recorded decisions, and human approval where needed.”

## 2. Employee request and policy evidence — Ask HR

[Enter “Can I use confidential company information for another employer?” and click **Submit request**.]

“I’ll begin with a policy question. The platform finds the relevant sections of the HR policy and uses only that evidence to prepare its response.

The answer appears here. Under Grounding evidence, I can inspect the exact policy section, reference identifier, similarity score, and supporting text. This makes the answer verifiable instead of relying only on general AI knowledge.

The request summary also shows how the request was classified, how many AI calls were used, the processing saved, and the recorded processing steps.”

[Replace the request with “Apply for annual leave from 2026-07-06 to 2026-07-08” and click **Submit request**.]

“The same page can complete an HR action. This leave request has clear dates, so the platform validates the details and submits it through the mock HR system. Routine structured requests avoid unnecessary AI calls.”

## 3. Automatic workforce scan — Detection

[Open **Detection** and click **Run workforce scan**.]

“Now I’ll check the one-thousand-record workforce dataset.

The scan looks for unusual payroll values within comparable departments and experience groups, leave above the review threshold, incomplete mandatory training, and overtime above the allowed cap.

When a new scan runs, it replaces the previous active scan queue. Older cases remain in history, but Open Reviews reflects the current scan rather than continuously accumulating old results.

The summary separates detected signals, automatically processed cases, and cases requiring a person. Every finding has a confidence score, recommended action, and current status.

The table provides useful context directly in the result. Payroll findings show the employee’s existing payroll and the mean payroll for their peer group. Leave findings show the number of leave days. I can filter the table by Payroll, Leave, or Compliance.”

[Select **Payroll**, then **Leave**, then return to **All**.]

“For routine scanning, the platform uses deterministic checks instead of sending every employee record through an AI model. This scan therefore uses zero AI tokens compared with the 1,350-token baseline shown by the processing-saved metric.”

[Expand **View processing details** and open two or three steps.]

“The processing history shows how the signal moved through detection, previous-case lookup, compliance review, and action selection. It records timing, the recommended action, any reward, and any action taken.”

## 4. Payroll alert and compliance protection

[Click **Simulate a payroll alert** under Quick actions, then return to **Detection** and scroll to **Latest upstream alert**.]

“This demonstrates an alert coming from an upstream payroll system. The alert proposes an eight-hundred-dirham payroll correction without the required manager-level approval.

The alert is high confidence, but confidence cannot override compliance. The platform checks twelve rules stored outside the AI prompt. The payroll approval rule and the high-risk automation rule block the correction, place it in the review queue, and record a negative learning signal so similar unsafe proposals become less likely.”

## 5. Human decision — Review queue

[Open **Review queue** and select the newly created payroll case.]

“The reviewer can see the case context, confidence, proposed action, status, and the reason it needs attention.

The decision options are Approved, Modified, and Rejected. The form adapts to the choice. Approved requires no replacement action. Modified reveals the Replacement action field. Rejected removes that field and asks for a rejection reason.”

[Choose **Modified**, select **escalate-to-HR**, enter “Manager-level approval required”, and click **Submit decision**.]

“I’ll modify this proposal and escalate it to HR. The decision and reviewer note are persisted. An approval creates positive feedback, a rejection creates negative feedback, and a modification creates partial feedback. The final outcome also teaches the platform whether the issue was resolved, repeated, or proved to be a false positive.”

## 6. Two-cycle learning demonstration — Learning

[Click **Run two-cycle learning demo** under Quick actions, then open **Learning**.]

“This demonstration runs the same type of leave case twice.

On the first occurrence, confidence begins at eighty-four percent, below the ninety-percent automatic-action threshold, so the case requires review. The simulated reviewer approves the proposal, and the resolved outcome creates additional positive feedback.

When a similar case appears again, the platform finds the earlier resolution and uses it to improve the new proposal. The Before and After cards show whether confidence and handling changed between the two cycles.

The Learning Progress chart plots feedback cycle on the horizontal axis and cumulative reward on the vertical axis. The Recommended Actions chart shows how action choices shift as feedback accumulates. Recent Feedback History lists the stored learning events.”

## 7. Close

“In this walkthrough, we used all three entry points: an employee request, an automatic workforce scan, and a payroll-system alert.

We also demonstrated policy-backed answers, payroll and leave anomaly context, safe automatic processing, firm compliance controls, human approval, stored feedback, and measurable improvement across two learning cycles.

The result is an HR operations platform that can detect, act, remember, and improve while keeping people in control of sensitive decisions.”
