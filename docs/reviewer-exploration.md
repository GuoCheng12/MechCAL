# Reviewer exploration mode (experimental)

The existing Reviewer has two separately addressed roles: argument audit and
independent exploration. They may use the same model, but never share their
conversation, judgments or retrieval cache. Enable `command_reviewer_exploration`
with `command_investigation_policy`; the older coupled
`command_experiment_options` mode must be off. Default profiles keep this feature
off pending scientific validation.

## Memory boundary

| Information | Exploration | Planner / audit |
| --- | --- | --- |
| Current SMILES and mechanism definitions | Shared | Shared |
| Raw calculation records, parameters, failures and artifacts | Readable | Readable |
| Planner rankings, confidence, arguments and task rationale | Hidden | Existing access |
| Worker explanations and audit opinions | Hidden | Existing access |
| Exploration notes, original proposals and query receipts | Private | Hidden |
| Optional exploration proposals copied to the question store | Original copy | Public advice |
| Proposal-linked executions | Result ID, tool and status | Existing access |
| Reference labels and external judge feedback | Hidden | Not inference input |

The runtime constructs a fresh, read-only case projection with an explicit
field allowlist. It does not pass the main case and merely ask the model to
ignore other roles. Known IDs outside that projection cannot be read. Calculator
payloads retain their method limitations and tool-generated descriptors; this
is not a ban on scientifically meaningful facts. Agent-written interpretations
are kept separately and excluded.

Private state lives in `command_protocol.reviewer_exploration`, bound to the case.
It includes the current note, original proposals, seen source versions, completed
attempts and its own retrieval session/cache. Older notes are retained in private
audit logs, not accumulated in each prompt. The latest note replaces the working
note. Original proposals are not overwritten when Planner edits the shared copy.
Actual execution receipts return without Planner's rationale or confidence.

`private:reviewer_exploration` is readable only inside the projected case. It is
context, not evidence, and cannot be cited as an experiment's factual source.
Private notes are not present in the ordinary record index used by Planner/audit.
Logs remain available to the operator for diagnosis, not as an agent input.

This removes direct inheritance of other agents' judgments. It does not make the
model statistically independent: the facts were still collected by previous
Planner choices, and the same model may reach similar conclusions independently.

## Scheduling and control

An episode is due only with unseen factual source versions, at one of these
boundaries:

1. The first usable calculation result (success or partial).
2. A question is resolved, deferred or assessed as distinguished/inconclusive,
   with an unseen linked calculation result, including failed attempts.
3. Planner requests conclusion with unseen facts.

A round number, changed confidence or wording edit alone does not trigger it.
There is no forced discovery quota or required mechanism. The reason for scheduling
is an operator log field, not an input revealing Planner's intention to exploration.

Exploration reads a compact catalog and up to three new result excerpts. All raw
fact records remain queryable. Each episode permits two read/description turns,
up to three optional experiments, and two local correction attempts. Format
correction does not reset reads. No scientific tool can be executed by this role.
The existing scoped read, reduce, compare and description interfaces are reused.

Only the optional proposals are published to the existing question store with
origin `exploration_reviewer`. They are advice, not evidence or mandatory tasks.
Planner may select, revise, defer or decline them, and retains all calculation and
ranking authority. A new question invalidates a prior investigation choice; a
queued computation is not silently executed after the new advice. Planner may
renew its choice in the same response as its next operation.

New proposal IDs also create a durable public `pending_proposals` handoff in the
investigation state. This is a planning event, not a synthetic calculation result.
It selects the full `exploration_decision` interface even when old arguments and
review feedback exist. Parameter descriptions, delegation and loaded computation
contracts stay subject to ordinary permissions and limits. An open invalid
judgment draft is repaired first without losing the pending handoff.

The Planner sees complete proposal comparisons and possible outcomes. Prior
argument review is shown as historical metadata, with its exact text readable at
`review:feedback`; outstanding factual findings remain visible. A valid
investigation choice consumes the handoff and records its identity in the choice
history, whether the decision is to investigate or conclude. Reads, local errors
and belief-only updates do not consume it. Identical proposals do not reopen a
consumed handoff. A current investigate choice retains the full interface until
the next action, including when selection and execution use separate turns.

Route diagnostics distinguish overall `tool_status` from the status of the
proposal's selected inputs. An incompatible selected artifact remains rejected,
but other case-local inputs passing basic checks are advertised as explicit
options. They are not substituted automatically and may differ in scientific
conditions. Disabled backends remain disabled. Both full and compact views retain
this distinction. Calculation success is never promised by a readiness check.

The episode is checkpointed across reads and corrections. Resume checks case,
role and source versions. Exhausted optional review is logged as unavailable,
not scientific approval, and is not retried against identical facts. Provider or
state failures preserve the checkpoint under the normal interruption policy.
No rank, confidence, calculation count or evidence record changes from exploration
alone. Query calls and usage are logged under `exploration_reviewer`.

## Acceptance

Offline acceptance includes marker-based non-leakage and input invariance,
out-of-scope reads, cross-case/session mismatch, durable private retrieval,
Planner edits, failed computations, repeated facts, query/correction limits,
real loop scheduling/resume, combined native-call receipts and unchanged defaults.

Online acceptance starts with the existing Case 0003 observations, fixed code,
the current DeepSeek Flash/high configuration and a bounded query/decision probe.
No reference labels, judge or new scientific calculations are required for this
first test. Record proposals, exact source/route validity, Planner's response,
local corrections, usage and latency. A changed label or more proposals is not
itself a scientific benefit. A later execution test must establish whether a
proposed contrast actually returns informative results and affects the relevant
judgments. Broader coverage and score gains require controlled multi-case tests.

## First online result (2026-11-11)

Full offline regression passed 3,465 tests with 4 skips, including 26 new
memory-isolation and scheduling tests. Repository Ruff and diff checks passed.

The bounded Case 0003 terminal-checkpoint probe completed with three exploration
calls and two Planner calls, using 118,006 input and 27,693 output tokens in
209.33 seconds. Exploration proposed same-grid state descriptors, a second dimer
pose comparison and an implicit-solvent contrast. The suggestions were accepted
as advisory records without modifying scientific state; Planner saw all three
without receiving the private note or caches.

Two imperfect source queries returned partial reads with explicit errors. Planner
also needed one local correction for an extra field in its investigation choice.
No computation or accepted evidence was lost. Original source hashes were unchanged.

Planner considered the suggestions but selected conclusion. The probe stopped
there without executing new chemistry, finalizing an answer or calling a judge.
The private memory and proposal publication work; broader exploration and scientific gains
are not yet demonstrated. Subsequent diagnosis found that the downstream task
router treated new advice as old-answer revision, folded the test outcomes away,
and retained the prior review as current conversational context. The solvent
proposal also referenced a bundle rather than the available prepared structure;
its selected-input failure was misread as tool unavailability. The explicit
decision handoff above addresses these integration problems. Useful next checks are earlier question-boundary
decisions, better source navigation, and distinguishing an informative limited
comparison from a computation that would conclusively identify an emitting
mechanism. Keep this feature off by default.

## Decision-handoff correction (2026-11-11)

Full regression passed 3,476 tests with 4 skips; the focused set passed 105.
The same terminal checkpoint and exactly the same three public proposals were
replayed with DeepSeek Flash/high. No proposals were regenerated and no new
chemistry or external judging was performed.

Planner first described the solvent tool, then selected investigation rather
than conclusion. It changed the proposed input from the baseline bundle to a
registered monomer structure, kept the second-dimer comparison open and deferred
the same-grid consistency check. Two calls completed without local repairs,
using 84,721 input and 4,444 output tokens, 29.57 seconds of model time and 52.91
seconds wall time. The earlier Planner-only segment used 58,944 tokens over two
calls; restoring scientific context increased input cost in this probe.

This validates handoff behavior, not scientific improvement. In particular, the
selected prepared structure is upstream of the baseline bundle's additional
optimization, not identical to its final analysis geometry. Planner's proposed
gas-phase comparison needs a matched-geometry check before interpreting any
solvent shift. An alternate population partition is also not automatically an
independent physical confirmation. No calculation was executed on these proposed
assumptions. Broader exploration, runtime and score effects remain unmeasured.
