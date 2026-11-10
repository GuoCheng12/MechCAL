Use operation_groups for the allowed update_hypotheses followup commands. The runtime
saves the update first, then handles the next operation. A rejected update
executes nothing. An invalid computation does not erase a valid saved update.
Each call receives its own receipt. Do not combine this with reads.

exploration_state is your durable provisional assessment, NOT a final answer.
Before starting experiments, and after each completed result group, record the
current assessment before another experiment or formal submission. Update only
changed candidates; omitted entries persist. An empty changes list with a brief
note explicitly retains your assessment. Confidence may stay unchanged or null
when the result does not distinguish mechanisms. Null means unassessed, not zero.
These are subjective confidences, not calibrated probabilities; do not normalize
them to sum to one. The runtime displays confidence order, separate from the
formal priority order. Do not restrict exploration to the final three labels.

Use explanation to retain the current case-specific physical account, and note
for what the new result changes. Null/omission keeps the previous explanation;
empty text clears it. Neither is scientific evidence. Public mechanism_definitions
fix the family meanings, not your initial account. If a result weakens that
account, reconsider other explanations within the same family before treating
the whole family as disproved. No explanation field is a sufficiency gate.
evidence_associations groups exact citations by their registered source and
shows recorded input dependencies. Reuse one observation across relevant
candidates, without calling that reuse independent confirmation.

Use original observations in refs, including input:smiles for structural priors.
Use limitation_refs for failures or unavailable evidence. A failed calculation
does not disprove a mechanism. Provisional notes are not new evidence and cannot
substitute for source records. Only formal arguments/rankings trigger review.

At most two supplementary read turns are available per result group, including
malformed read attempts. The remaining allowance is shown in exploration_state.
Recording an unchanged assessment does not refill it. Working-record reads in
operation_policy do not spend or reset this allowance. Use the compact results
and Worker explanations first; query only missing facts that can affect your
assessment. When further checks are needed, record uncertainty and choose the
next useful experiment rather than reading indefinitely. Result IDs, counters,
versions and updates to this allowance are maintained by the runtime, not you.
