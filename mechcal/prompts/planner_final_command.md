# Final command

Return action=revise_final with update, final_action and audit_responses.
update uses the same JudgmentProposal interface as earlier stages, not a complete
snapshot. There are no pending calculations or candidate proposals, so update
dispositions and candidate_responses are empty. Do not perform scientific tools.

Complete evidence_submission once per final Top-3 label, in any array order. For each
label choose submitted with its actual current support_ids, or no_evidence
with a specific reason. This is a delivery decision, not a support grade.
If your explanation relies on a justified structural or computed finding,
write its finding, warrant, boundary and observation_refs as an argument in
update.mechanism_support_arguments, then list its ID. Hypothesis prose,
question answers and the final draft are not exported as evidence. Do not
omit a bounded argument only because it does not establish experimental truth.
Do not fabricate a link to avoid an empty submission. Explicitly withdraw
unjustified arguments before declaring no_evidence for their label.

Respond to each pre-revision support_id in the outer audit_responses, including
withdrawn arguments. Correct unsupported links or factual errors without changing
the source observations. Retain justified disagreement and unresolved limitations.
Missing experimental proof does not itself erase bounded computational support.

final_action follows its output_contract. Set its case_id to the supplied case
and revision to the input revision + 1. Retain the prior finalize/stop action,
with no dispatch_requests or investigation_plans. Include the consistent final
answer, ranking justification, question dispositions and unresolved gaps.
State ranking_uncertainty_material and retain continuation_assessment when required.
Rank-bound comparisons will be requested against the frozen merged proposal if
the judgment changes. Do not submit those comparisons here or invent evidence.

If review reveals a useful remaining test that prevents completion, use stop,
judgment_state=unresolved and finish_basis=null. Preserve the incomplete answer
and useful route; do not pretend convergence or reopen tool execution here.
