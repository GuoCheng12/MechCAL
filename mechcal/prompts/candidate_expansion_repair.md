Correct the rejected candidate proposal batch using only the supplied computations.
Return JSON matching the correction output_contract, not a new CandidateExpansion.
Each repairs entry contains an allowed path and its replacement value.

Read the indexed issues. Preserve every retained_proposals item unchanged in the
replacement proposals list. Correct only invalid items. The proposals list has
zero to two items; two is a maximum, not a target. If an invalid explanation rests
only on old observations, either explain a genuine connection to newly delivered
observations and cite those sources, or withdraw that item by omitting it from
the replacement list. Withdrawal is your explicit scientific decision, not a
default required by the program. Explain the withdrawal in rationale. If every
item is invalid and none can be grounded, an explicit empty list with a reason is
valid. Never discard a retained valid item to clear the error.

For example, if only the second proposal lacks a new observation, the first must
remain unchanged. Do not rewrite both proposals or merely restate the old second
proposal. Each submitted item must independently cite a new_observation_refs ID.
Do not invent calculations, copy a citation without a real connection, or treat
an old observation as false just because it does not qualify for this batch.

If last_failed_correction is present, fix that correction's format using the same
rejected_response and allowed paths. This uses the existing finite retry allowance.
No tools, rankings, evaluator rubrics, private references or external judge feedback
are available. Proposed explanations and corrections are not new evidence.
