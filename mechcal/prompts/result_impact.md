# Scope the effect of this evidence batch

Also return impact_review for the supplied impact_context snapshot. Preserve its
snapshot_id and all result_ids. checked_support_ids must cover every supplied
claim, not only claims cited by the Worker. Mark affected_support_ids and
affected_labels when new results could strengthen, weaken, contradict or change
the conditions of a claim. Include affected_comparison_ids when a comparison may
need revision. Examine alternatives outside the leaders and uncited counterevidence.

Do not infer no impact merely because a claim does not yet cite the new evidence.
Separate an unchanged structural fact from its mechanism interpretation and its
position in the ranking. An empty affected set is an explicit judgment that this
batch does not require reopening those claims, not approval of their truth.
Use global_review=true if effects cannot be localized from the supplied material.
Explain the scope in reason. Do not assign priorities, rewrite arguments, select
tools or consult hidden references/rubric. Planner can widen the scope afterward.
