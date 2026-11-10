# Evidence Retrieval and Investigation Follow-up

This change concerns retrieval and task views, not scientific calculations,
mechanism scores, reference labels, or the external judge.

## Complete Values Versus Partial Views

Every `read_evidence` response includes coverage for each requested JSON pointer.

- `selected_value`: the whole selected value, not necessarily the whole source.
- `complete_array`: a page starting at zero that includes the entire array.
- `partial_array`: only the reported rows. Even a last or empty page is partial.
- `directory_only`: names and sizes, not observations.

Exact prior reads and Reviewer fact packets reconstruct the same coverage from
their original source. Old checkpoints do not need new persisted fields.
Byte-limited pages report the actual returned row count, not the requested count.
Missing paths remain local errors; they are never silently replaced with data.

`reduce_evidence` supports `min`, `max`, `max_abs`, and `sum` over up to 100,000
rows. Null `row_indices` selects the complete original array. An explicit subset
is identified as such and still allows at most 512 distinct indices.
`max_abs` returns a magnitude; matching rows retain their original signed values,
source paths, and scalar row context. This is useful for signed coefficients.

All selected values must be finite numeric scalars. Missing values, strings and
booleans are rejected rather than skipped or coerced. Responses include the
source version, array size and number of values considered. At most 32 matching
indices and four detailed matching rows are returned; omitted counts are explicit.
This bounds ties without hiding the total number of matches.

Numerical receipts are read-only, source-version-bound arithmetic, not new or
independent evidence. Their query scope and allowances are unchanged. A maximum
does not establish a mechanism, comparable conditions or a physical state match.

## Investigation Versus Confidence

Planner retains control of confidence, ranking, test selection and stopping.
Comparison questions now expose deduplicated counts of explicitly linked results
by execution status. These are neither support scores nor a complete inventory
of tests relevant to a mechanism: unlinked calculations are not inferred.

During repair, revision and delivery, a bounded `investigation_frontier` retains
unresolved question IDs, labels, execution counts and current route readiness.
Short question/test/outcome text is exact; longer fields remain addressable in
`followup:state`. The preview favors label coverage, not a scientific priority.
Omitted-question counts and a route to the full options remain visible.

Compact question links are resolved before their targets are deferred. Therefore
reading `comparison_options` during delivery returns self-contained questions,
not pointers to a question list that has disappeared from the task view.
`read_context` can restore the existing scientific interface as before.

No candidate-specific tool map, mandatory test sequence, new Agent or required
model-output field was added. Failure does not refute a mechanism; readiness
does not guarantee success. Open questions do not automatically block delivery.

## Verification

Run the portable regression suite:

```bash
python -m pytest -q
```

Focused coverage is in `tests/test_evidence_selection_coverage.py` and
`tests/test_investigation_frontier.py`. It includes long signed arrays, ties,
invalid values, explicit subsets, last/empty pages, byte-limited pages, scoped
reads, source-version checks, old-handle reconstruction, task-view transitions,
disabled routes, and unchanged scientific state.

A bounded online review can additionally present a numerical disagreement and
an explicitly partial page, allow the normal read-only operations, then check
the returned source and conclusion. This verifies retrieval behavior, not
end-to-end ranking performance. Full-case comparisons still need separate runs.
