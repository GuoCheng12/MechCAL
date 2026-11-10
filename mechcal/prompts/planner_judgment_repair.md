# Planner: repair an uncommitted judgment draft

Return only repair_judgment using output_contract, not a new scientific command.
Echo draft_id exactly. Select the supplied target_id and its authorized operation.
For replace, provide the replacement value. For remove, omit value.

The runtime preserves every other part of the draft. A missing field in this
repair response does not delete it from the draft. Do not recreate the complete
answer, alter unrelated scores, invent evidence, or add a second interpretation
agent. This remains the same Planner correcting its own submission.

Each target identifies a field or record in THIS draft. Stable record selectors
identify the affected question, mechanism or argument. Never copy a snapshot
array index into an edit. The draft_id prevents application to a different draft,
case, or accepted revision.

For a scientific inconsistency or missing required content, scientific_context
supplies the public case context and candidate_command is the uncommitted draft.
Use both to ground the correction; the candidate is not accepted evidence.
The authorized target may encompass a record or command when the
issue cannot be resolved at one field. Make the scientific decision explicitly;
the runtime does not invent values or select a mechanism. Unresolved conclusions
remain allowed. Retrieve omitted case records when genuinely needed.

All edits are revalidated together against the complete command and the accepted
case. An applied draft edit is not an accepted judgment, evidence, or approval.
Use validation_history and repair_failure to avoid repeating invalid edits.
