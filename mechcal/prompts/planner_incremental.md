# Incremental judgment contract

This mode replaces explicit retained_judgment lists. Return retained_judgment=null.
At revision 0 produce one complete initial snapshot without delta. Thereafter
use delta with the provided base_revision and a concise scientific reason.

Omission means keep the accepted record unchanged. The program automatically
expands all retained hypotheses, arguments, questions and candidate bases before
checking the full judgment. Submit full records only for new or changed content.
Use delta.hypothesis_scores for score-only changes. Empty hypotheses/questions/
mechanism_support_arguments arrays mean no changes, not deletion.
Withdraw arguments only through delta.withdrawn_support_ids. Keep prior
questions; mark a question deferred or revise it instead of deleting it.
Never rewrite unchanged rationales just to demonstrate that you reviewed them.

Keep portfolio.current and runner_up explicit. A null ranking_update declares
the order and previous candidate comparisons unchanged, with delta.reason as
the explanation. If evidence changes a comparison or the ranking cutoff, submit
the new ranking_update, but include only changed/new candidate_bases.
No mechanism is protected from reconsideration; update any affected records.
In decision_assessment, briefly state what changed, what did not, and why.

For local_audit feedback, use delta.scope=local_audit and respond to the specified
support_ids. Change only those arguments; do not resubmit unaffected records.
If the feedback affects ranking, interpretation elsewhere, or investigation,
use scope=global and explain why. Never retain an unsound conclusion to minimize
response length. Final revision still requires a complete final_action and all
required audit_responses, consistent with the expanded judgment.
