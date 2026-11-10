# Incremental Editing Contracts

The November 11 interface changes address rejected local edits, not chemical
methods or mechanism-ranking policy. They do not relax source ownership,
reference existence, execution permissions or atomic judgment submission.

## Stored Records And Model Edits

Comparison questions remain complete records in case storage. Model-facing
question updates are small deltas keyed by a case-local question ID:

- Omitted or null fields preserve saved values.
- New questions still require both `labels` and `question`.
- `clear_fields` explicitly clears `assessment`, `route` or
  `disposition_reason`; clearing and replacing the same field is rejected.
- Parent links, source references and all related updates are validated before
  the transaction is accepted. Unknown IDs cannot borrow another record's text.
- Updating a question retains its origin, trials and independent Reviewer memory.

## Draft Repair

`revise_judgment` and `amend_judgment` create transactions that may request review.
If validation rejects one, the pending draft retains its scientific text and
review intent. `repair_judgment` now accepts the same transaction metadata:

- `note` and `request_review` are optional overrides. Null or omission preserves
  the pending transaction's values. Explicit false is not permission to bypass
  mandatory review of changed evidence or arguments.
- `citation_path_edits` changes only a selected citation path, using its citation
  and path indices. It cannot be combined with replacing the full citation list
  in the same argument patch.
- A citation error displays the exact editable position and the source directory.
  The runtime does not guess a replacement path or select scientific evidence.
- Every repair revalidates the whole transaction. Invalid, stale, cross-case or
  duplicate edits cannot partially change accepted arguments or ranking.

## Validation Scope

Portable regression tests cover metadata preservation, strict and non-strict
tool contracts, single-path corrections, duplicate/conflicting edits, invalid
references, atomic rollback, partial question updates and explicit clearing.
Existing tests also cover truncated responses, checkpoint reload, stale drafts
and source ownership.

The full offline suite passed **3,493 tests with 4 skips**. Ruff and diff
whitespace checks passed. Private Case 0003 replay additionally confirmed that
the exact previously rejected repair response now commits and enters review,
without changing its scientific text or raw observations. An earlier response's
misspelled impact ID remains rejected; it is not silently corrected.

A frozen-runtime online continuation used `deepseek-flash` with high reasoning
on the existing Case 0003 draft and R014 results. Planner inspected the draft
and source directory, checked the values, and submitted a single-path repair.
One response contained an extra JSON closing brace; it was rejected without
mutation and corrected by the next model response. The transaction then committed
and Reviewer returned a valid report after two read turns. No human edited the
model output or bypassed validation.

This check used 7 calls (210,372 input and 16,314 output tokens), taking 535.59
seconds in total, including 89.92 seconds spent awaiting model responses. The
observed-token boundary was 200,000; the final allowed response crossed it to
226,686 and the run stopped after review. Raw results and the original checkpoint
were unchanged; there were no new calculations or judge calls. This is not a
zero-error claim: one syntax rejection recovered within the bounded loop.

Reviewer resolved two prior findings and raised three new scientific findings
about geometry provenance, unlike state-gap comparisons, and the strength of
the two-pose generalization. Those are retained as scientific feedback, not
silently fixed or counted as interface failures. A full-case continuation and
external scoring were outside this acceptance check.

Private acceptance artifacts, credentials and raw case data are not part of the
public repository. These checks establish interface behavior, not benchmark
performance or the scientific correctness of the model's interpretation.
