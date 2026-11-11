# Execution Facts and Recovery

This change separates three things: what a calculation returned, whether another
route can be attempted, and what those results mean scientifically. It does not
change mechanism labels, priorities, numerical acceptance thresholds, or the
external judge.

## Derived Status

With computational review enabled, Planner and Reviewer receive an execution
inventory derived from the existing case-local result records. It contains
per-capability attempt/status counts, up to three recent results, exact returned
member counts and source paths. An earlier usable result remains addressable if
more recent attempts all failed. It is not another evidence store.

Counts describe returned members of an execution, not unique conformers across
runs or independent scientific corroboration. A failed retry does not erase an
earlier success. An untried, ready route differs from a missing-input route, a
failed attempt, and a disabled calculation. The untried directory is an option
list, not an obligation to run every tool.

Ordinary reviews receive inventory for their cited capability histories; final
review can inspect the full inventory. Scope is explicitly capability-wide,
not a claim that every attempt used the same geometry or answered the same
question. Raw parameters and outputs remain in their original records.

## Changed Evidence and Prior Reviews

Review reuse depends on both the cited record versions and the versions of
related capability histories. A new related result makes an unchanged passage
eligible for review again. `execution_rechecks` identifies affected passages and
capabilities. It does not assert that the passage is wrong. Final narrative and
global limitations also depend on the current execution inventory.

Resolved factual issues with changed source histories become visible for review.
Program logic does not rewrite scientific claims, raise confidence, or substitute
a mechanism. Unrelated argument sections and exact source reads remain reusable.

With task-scoped context enabled, Planner receives only changed inventory groups
after successful delivery. The full inventory remains available through the
case-local context record. A transport failure, local normalization, or a narrow
format-repair request does not acknowledge delivery of the new context.

## Computation Failures and Time Limits

PySCF timeout reports retain completed stages, SCF timing, number of completed
states, and pointers to raw results and bounded logs. A completed ground-state
SCF with no excited states is a failed spin-state calculation, not triplet
evidence. Existing validated partial states remain retained under their original
limitations. This change does not implement SCF checkpoint reuse or guarantee
convergence for large metal-containing systems.

Hole-electron postprocessing keeps its 300-second default. The explicit
`max_wall_seconds` argument now permits 10-1800 seconds. Set
`MECHCAL_SPATIAL_MAX_WALL_SECONDS` to a value in that range to impose a lower
deployment ceiling; its default is 1800. Requests beyond the ceiling fail
before execution rather than being silently shortened. Worker delegation time
must also accommodate the requested calculation. MCP waits for a validated
whole-operation deadline plus delivery time unless an explicit transport policy
disables this behavior.

Longer time does not relax normalization, transition-dipole, provenance, or
geometry checks. An invalid numerical output remains a failed result. Recovery
is an explicit Worker/Planner decision, not an automatic repeated calculation.

## Verification

Focused offline coverage:

```bash
pytest -q tests/test_execution_inventory.py \
  tests/test_calculation_recovery_context.py tests/test_worker_tasks.py \
  tests/test_command_shape_repair.py tests/test_toolchain_expansion.py \
  tests/test_review_delta.py tests/test_reviewer_execution_context.py
```

Online acceptance should separately check factual correction, an executable
next-step choice, and explicit failure handling, with original records available
through bounded reads. Keep the current task and necessary state in each request;
the native transport replays the most recent exchange, not an unlimited chat
history. Do not put the result of `reviewer_scope` directly in a model message:
it is an internal map of authorized full records, not a serialized read directory.

Component probes are not end-to-end benchmark results. A valid command is not
proof of useful science. A successful contact-motion probe is a rigid modeled
geometry comparison, not a crystal structure or a nonradiative-rate calculation.

## Reviewer Delivery Boundary

Provider completion is not review completion. Native read calls use the actual
tool-call channel and exact call/result IDs. A serialized call envelope emitted
as text (for example, DSML or a JSON tool-call object) is rejected as commentary;
the runtime never parses it into an executable operation or accepts it as advice.
Ordinary scientific prose and mentions of tools remain allowed. This is a
transport-format check, not a test of scientific correctness.

Each commentary turn exposes its current delivery mode and offered query tools.
If a text-only response is rejected, its error is included in the next request,
even though there is no native call ID for a tool-result message. The original
response remains in the audit log. The existing allowance permits at most two
local corrections; corrections neither reset query turns nor consume scientific
rounds. Exhaustion produces unavailable feedback, not approval or a zero score.
The legacy one-shot prose path rejects such output without adding hidden retries.

When an error replaces the last query result, prior read handles remain saved,
and their values are no longer suppressed as already present in that result.
This keeps the necessary facts available during correction without re-querying
or making up sources. Structured `record_review` delivery and its ownership checks
remain unchanged.

```bash
pytest -q tests/test_commentary_delivery.py tests/test_review_wire.py \
  tests/test_review_link_repair.py tests/test_reviewer_text.py
```
