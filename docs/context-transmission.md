# Task Memory and Message Transmission

This change reduces repeated background in model requests. It does not change
scientific tools, ranking rules, judge settings or the evidence rubric. Raw
records and execution receipts remain authoritative.

## What Is Sent

| Role | Default input | Exact background on demand |
| --- | --- | --- |
| Planner | Current task, candidate confidences, new-result references, open issues, bounded comparison frontier, short artifact index and available-tool directory | Historical comparisons, full explanations, pending experiment details, full source indexes and complete capability descriptions |
| Worker | Current result, execution conditions and bounded retained facts | Other permitted source fields and large background sections |
| Impact Reviewer | New observations and current assessment task | Full question history, question-to-evidence links and source directory |
| Exploration Agent | Permitted facts, its own current note and bounded proposal navigation | Its own older proposals and permitted historical source directory, never Planner opinions |
| Scientific Reviewer | Current claims, source conditions, numerical facts and outstanding issues | Historical background; this change does not defer the facts needed to check the current claim |

An omitted entry is not a rejected hypothesis or missing evidence. Display limits
do not change read permissions or scientific priority. All known candidate
confidences remain visible to Planner; newly proposed and currently selected
questions receive explicit navigation entries.

## Delegation Input Handoff

Planner keeps at most 12 artifact addresses (ID, kind and status) in its working
view. Small indexes are shown in full. Larger indexes retain starting structures,
recently referenced items and recent products, with counts and the exact complete
index in `context:planner`. No coordinates or large metadata are in this view.

A delegated task grants access only to its explicit `input_refs`, their existing
execution provenance, and the task's own later results. Before starting a Worker,
the runtime checks for a starting path using registered artifact requirements,
alternative source kinds, declared outputs and enabled backends. Intermediate
products need not exist yet. Potential reachability is not execution success or
scientific support. A usable saved result can also be inspected and delivered
without starting new computation. Missing access returns an ordinary operation
error with candidate addresses for Planner to select, never implicit permission.

If a Worker selects a route whose inputs cannot be reached with its authorized
tools, the task returns an input handoff to Planner. A second identical missing
artifact error with unchanged inputs also returns a handoff, even after metadata
reads or checkpoint restoration. Changed inputs can be reconsidered. This rule
does not limit numerical-convergence retries. Successful results, partial results
and existing scientific interpretations are retained; runtime handoff notices do
not become scientific interpretations or evidence.

## Storage and Retrieval

Planner background uses the existing case-local `context:planner` record in the
checkpoint. Scoped agents receive a role-local `context:task:<content hash>`
record. The latter is stored in `memory/task_contexts/` and retained in the active
task continuation. Identity includes the case, role and exact field contents.
Neither kind of context record can be cited as scientific evidence.
Role-private context IDs stay in local navigation history and are excluded from
cross-agent read-source lists and evidence handoffs.

Each view includes exact record IDs, field paths and read instructions. Agents
use the existing `read_context` or `read_evidence` operations. Large lists support
page reads. Full source directories retain column names, defaults and path sets;
navigation previews do not replace the permission-checked directory.

There is no new summarizing model or embedding service. Old data are stored
exactly. Only navigation text is previewed, with the full original available on
read. Exploration's private memory is built from its already-isolated input,
not by selecting arbitrary fields from Planner state.

## Continuation Rules

- Exhausting Planner's two context reads restores the permitted operation
  interface, not the entire background payload. Further reads follow the
  existing action policy.
- Reading a value does not mean it remains visible forever. A request retains
  bounded working facts and its latest tool exchange. Necessary older reads
  remain in the working-memory view, within its existing size limit.
- If a local correction replaces the latest exchange with an error, previously
  read facts are restored to working memory rather than incorrectly marked as
  still delivered by that exchange.
- Identical large values shared by a tool result and the current request are
  serialized once. Tool messages point to the exact value in the same user
  message. Every original tool-call ID still receives a paired result.
- Shared-value and shared-text compression are preserved when another view
  processes the request. They must not expand an old shared table into multiple
  copies. Original provider responses, operation results and provenance are
  never rewritten by this wire-only projection.

## Configuration and Observation

Planner task views use the existing `command_compact_context` and
`command_task_context` flags with native commands and the review workflow.
Scoped-agent views follow their compact-context path; the independent Exploration
Agent already uses that path. The shared commentary function's existing
`review_task_context` switch can disable its task projection.
No model, inference limit or production configuration is changed here.
Exact request-local deduplication runs in the native transport independently.

`history/context_views.jsonl` records scoped-agent before/after payload bytes and
deferred fields. These values exclude system messages and tool definitions.
`history/command_requests.jsonl` remains the source for actual sent messages and
provider-reported usage. Bytes are not tokens, and cached input tokens must not
be counted again as additional input.

## Offline Acceptance

Tests cover exact readback, bounded historical growth, ownership, role isolation,
metadata citation rejection, checkpoint continuation, local correction, source
directory visibility and both supported transport protocols. Request-local
deduplication must preserve numerical values, source IDs and call/result pairs.
The initial memory-change regression run passed 3,587 tests with four skips. The 236 targeted
context, retrieval, transport and role-isolation tests also passed after the last
private-ID handoff adjustment. `git diff --check` passed. Ruff was not available
in the verification environment.

`examples/audit_context_transmission.py` compares two source checkouts against
one checkpoint and its saved capability registry without external model calls.
It also projects recorded scoped-agent messages. The report stores checkpoint
and catalog hashes so the comparison can be checked for identical inputs.

The 2026-11-11 offline comparison used the saved 0003 reviewfix run. At the saved
checkpoint, Planner's task payload fell from 105,344 to 60,385 JSON bytes. Its
request including prompt and tool definitions fell from 144,918 to about 98,200
bytes. Historical impact-review and exploration-review request projections fell
by approximately 35% and 26%; Worker requests were nearly unchanged.

These are transmission measurements, not an online token or quality result.
Historical projections hold model decisions fixed and exclude extra retrieval
calls that the new view may induce. No claim of reduced total runtime or stable
ES-nDCG follows from this audit alone.

## Online Acceptance

Use a fresh output directory and unchanged model/tool/judge settings. First test
a small number of 0003 rounds, including at least one historical read and one
review continuation. Check:

1. No hidden source, ownership error or orphaned tool result.
2. Necessary values are available after read, correction and resume.
3. The same large table is not repeated in both receipt and working view.
4. Record model calls, retrieval calls, input/output/cache tokens and time per
   scientific round, rather than counting a smaller request as a complete win.
5. Preserve raw results and inspect scientific decisions before a longer smoke.

### Input Handoff, 2026-11-11

Two bounded Case 0003 runs used the same baseline configuration: DeepSeek Flash,
high reasoning, at most 40 model calls and three scientific computations, and a
250,000 observed-token stop threshold. The threshold is checked after responses,
so one response may exceed it. Both runs paused cleanly with exit code zero;
neither finalized a scientific answer or called the judge.

Local records under the workspace's `var/experiments/`:

- Before: `mechcal_memory_smoke_0003_20261111_214747`.
- After: `mechcal_input_handoff_0003_20261111_230658`.

Each directory retains source hashes, configuration, exact requests, usage,
checkpoints, execution records and the acceptance stop reason. No credentials
are required in this document.

| Observed quantity | Before | After |
| --- | ---: | ---: |
| Model calls through first valid geometry-computation request | 20 | 5 |
| Tokens through that request | 231,742 | 115,486 |
| Seconds from process start through that request | 124.8 | 76.5 |
| Total model calls before acceptance pause | 22 | 15 |
| Total input tokens | 232,758 | 224,339 |
| Total output tokens | 17,682 | 28,846 |
| Total tokens | 250,440 | 253,185 |
| Subject runtime, seconds | 312.7 | 377.1 |
| Retained successful scientific computations | 1 | 1 |
| Planner operation rejections | 2 | 0 |
| Worker missing-input issue occurrences | 5 | 0 |
| Worker out-of-scope issue occurrences | 3 | 0 |

The earlier first Worker task used all 12 calls without starting a computation.
Its only delegated input was `input:smiles`; the initial-structure address had
been hidden in deferred context. In the new run, Planner saw `structure:initial`
immediately and directly requested geometry optimization. That computation
succeeded, preserving its raw results and prepared-structure artifact. Planner
then explicitly delegated the prepared geometry, the execution record and the
initial structure to a Worker. Its `input_access` marked the baseline bundle
ready and the two downstream analyses potentially reachable. The Worker loaded
all three tool descriptions without an input or permission rejection. The token
threshold stopped the run before that task's first computation.

This demonstrates address visibility and valid online delegation, not completed
downstream science. The improved prefix is a single observed trajectory, not an
average speedup: direct computation versus delegation and the number of review
calls differed. Total runtime did not decrease. The new run spent the remaining
allowance on interpretation, impact review and exploration review rather than
the old missing-input loop. Scientific scores are unavailable, not zero.

Offline acceptance additionally replays the earlier real delegation against its
saved registry and checkpoint. The unmodified SMILES-only contract now returns
`delegation_input_gap` before any Worker call; explicitly adding the initial
structure makes all three declared steps reachable. Focused tests cover
cross-case references, disabled tools, dependency cycles, intermediate products,
partial results, repeated input errors across reads/resume, corrected requests,
and preserved interpretations. The online run did not intentionally force an
invalid Worker request; repeated-error handoff is verified offline.

The 118 targeted tests passed after a final index-boundary correction: six recent
read references cannot displace the starting structure in a large index. The
online run used the frozen build before that boundary correction; its small
artifact indexes are unaffected. Running code was never changed in place.
The final source then passed the full regression suite: 3,604 passed, four
skipped, with one existing dependency-deprecation warning in 368.34 seconds.
`git diff --check` passed. No merge or push was performed.

## Task Packets, 2026-11-12

Splitting Scientific Reviewer and Exploration Agent does not by itself reduce
the background sent to either agent. The first five model calls of the input
handoff run used 104,976 input tokens out of 115,486 total tokens, before a
scientific result was available. The three Planner requests repeated a
30,962-byte system prompt, the same 16,338-byte compact tool catalog, the same
5,122-byte review feedback and the same 1,692-byte mechanism definitions. The two
coverage-review requests repeated the same 32,922-byte compact `sections`
object. These are measured byte counts, not token estimates. Some repeated
input was cached, which must be reported separately from uncached cost.

The implemented change uses task-specific input construction, not deletion of
historical data or an assumption that a stateless model remembers an old request:

1. Planner's normal task-memory view uses a short resident operational policy.
   The exact existing manuals remain in `context:planner`, under
   `/fields/operation_guides/<name>`. The payload lists available guide names.
   These records are instructions, not scientific evidence. Legacy full-context
   mode and the existing judgment-edit prompt are unchanged.
2. After a read or description query, the capability directory keeps every tool
   ID, purpose and live status. Full inputs, outputs and preconditions stay
   readable through the saved catalog or `describe_tools`. Reviewer/Worker task
   continuations use the same directory projection. Initial task packets retain
   the richer short directory.
3. Long review prose becomes an explicitly marked preview with an exact read
   address. Structured findings and status remain inline. Reading the full
   review is possible; its preview does not replace the scientific advice.
4. Candidate confidences, current claims, numerical facts, source conditions,
   open issues, artifact addresses and bounded working facts are preserved.
   The offered operations and their argument schemas are unchanged. The runtime
   neither chooses a tool nor fills scientific arguments.
5. Query continuations retain the same background snapshot and read allowance.
   Role-private notes remain scoped to their owner. Earlier reads are not
   assumed visible: the existing latest-exchange and working-memory rules still
   determine what is actually sent.

Planner already may submit hypothesis updates and a computation in one response;
the observed extra decision call was not a runtime requirement. This change adds
no mandatory summarization or routing model call.

At the same saved input-handoff checkpoint, the complete Planner request fell
from 113,604 to 86,785 JSON bytes (23.6%). Its system text fell from 30,962 to
7,177 bytes. Tool schemas grew from 28,673 to 29,136 bytes because they now explain
the provisional-update field boundary and enumerate the currently readable
context fields. Exact local background storage
grew because it now retains manuals. This checkpoint was not a read
continuation, so these figures do not include the slimmer continuation catalog.

This is not a claim that every agent input is minimal. Function schemas and
current scientific working data remain substantial. Acceptance checks exact
readback, permissions, preserved facts and stable query allowances, followed by
a bounded online prefix. Online calls, usage and retrieval behavior must be
reported separately; smaller fixed packets alone do not prove lower total cost
or preserved scientific quality.

### Online Findings and Corrections

Two fresh bounded runs used DeepSeek Flash/high with the previous model/tool
configuration, 20 model calls, three scientific computations and a 200,000
observed-token threshold. Neither called the judge. One response can exceed
that threshold; it is an acceptance stop, not scientific convergence.

The first run (`mechcal_background_0003_20261112`) retained one successful
calculation. It also exposed two rejected hypothesis updates: formal candidate
status fields were mixed into provisional confidence updates. The native field
description now distinguishes those commands. Validation still rejects mixed
records; it never guesses a confidence or silently converts the command.

The second run (`mechcal_background_verified_0003_20261112`) retained successful
geometry optimization and a vertical-state baseline bundle. It paused normally
after 14 model calls, 188,374 input tokens and 21,253 output tokens, in 383.96
seconds. Cached input was 78,592 tokens, already included in input totals. Its
Worker task was still active; this is not a finished case or final scientific
answer. That run had one Planner rejection and one Reviewer read rejection from
mixing a description query with record reads. A complete operation grouping is
now generated from the same available-operation set for Planner and scoped
commentary agents. Existing execution batching rules are unchanged.

An additional online replay found that `read_context` still advertised a fixed
old field enumeration while memory exposed new fields. That mismatch was fixed
at the interface: advertised names now come from the current `deferred_context`
directory, and execution validates against the case-local saved snapshot.
There is no separate manually maintained name list. Offline tests exercise
every advertised field, including guides and full review feedback. Two final
online replays of the saved failure contexts returned accepted operations with
no local syntax/grouping errors; they executed no scientific commands. These
last corrections were not followed by another complete scientific smoke.

The observed first Planner input fell from 20,511 to 15,294 tokens. Its query
continuation catalog fell from 16,338 to 10,879 JSON bytes. But the second run
used 130,935 total tokens over eight model calls before its first computation
request, versus 115,486 over five calls in the previous run. It chose Worker
delegation rather than direct computation, made additional reads, and incurred
a local correction. The prefix therefore did not show a net token saving.
Different stopping thresholds and downstream trajectories also prevent using
whole-run totals as a matched efficiency comparison. This change demonstrates
smaller transmitted background and working retrieval, not a guaranteed total
cost or scientific-score improvement.

Exact source/configuration records, requests, usage and checkpoints remain in
the two private experiment directories. The workspace audit directory
`var/audits/mechcal_background_20261112/` contains the fixed-checkpoint byte
comparison, online comparison and final contract replays. No secret, hidden
reference or judge feedback is supplied by this change.

Before integration on 2026-11-12, the full suite was rerun against the final
code: 3,629 tests passed with four skips and one existing dependency-deprecation
warning (366.57 seconds). A separate 56-test context/role suite had also passed,
including the added check that scoped commentary receives the same batching
classification used by its parser. `git diff --check` passed. This integration
check made no new model calls and did not resume any experiment; the online
validation limits above still apply.
