# Investigation choice workflow (experimental)

The opt-in `command_investigation_policy` keeps candidate beliefs and experiment
selection separate. It requires the existing hypothesis, question and final
computational-review workflow. The default profile is unchanged.

## State and messages

- `exploration.entries` remains the single candidate-belief store.
- `evidence_followup.questions` remains the single comparison/experiment store.
- `update_hypotheses.investigation` records an investigate/conclude decision,
  a selected question (nullable for preparation/open exploration), alternatives
  actually considered, and a short natural-language reason. It can share a native
  response with computation, delegation or finalization; no extra agent is added.
- Runtime binds the decision to case-local observations, artifacts, candidate
  judgments, questions and impact responses. Reads and prose corrections retain
  it. A new result requires reassessment and another decision. The code does not
  rank experiments, infer negative evidence from failures or enforce score moves.
- The decision history is readable at `investigation:state`, but is not evidence.
  Exact executable parameters still use the existing computation interfaces.
- The existing final Reviewer receives the choice, alternatives, actual attempted
  routes and remaining comparisons. Its version-bound computational advice and
  disagreement rules are reused; this is not another review loop.
- Pending result-to-candidate impacts remain visible until explicitly addressed.
  No automatic claim of scientific coverage is made from saving a choice.

## Acceptance, fixed before online tests

1. Offline: native schema/restore consistency, atomic edits, local unknown IDs,
   cross-case isolation, failed results, unchanged beliefs, lower-ranked questions,
   duplicate cached computations, same-response selection and execution, stale
   choice handling, task views and unchanged final-review reuse.
2. Online first: matched decision probes from identical historical Planner inputs,
   using the same model, effort, tools and observations. No hidden reference or
   judge. Preserve all replies, choices, validation failures, usage and latency.
   A decision-only probe tests selection and interface behavior, not tool outcomes
   or end-to-end scientific quality. Stop on unhandled errors, do not auto-restart.
3. Only after those pass: short forked rollouts (2-3 new computations) with the
   same starting checkpoint and equal computation/call allowances. Do not force
   the same tools, mechanism labels or rank changes. Compare redundant confirmation,
   actual distinguishing observations, affected-candidate updates and cost.
4. Full development cases 0003/0004/0005 and preselected unused cases follow only
   if the short rollouts justify the cost. The three development cases alone are
   not evidence of generalization. GT and judge are evaluation-only.

Interface success, descriptive experiment-selection reasons and changed rankings
are not scientific-success criteria. A stable ranking or justified early stop may
be correct. Report unresolved issues and overhead, including failures.

## First validation outcome (2026-11-10)

- Full offline regression: 3,422 passed, 4 skipped. Ruff and diff checks passed.
- Matched historical decision probes used `deepseek-flash` with high reasoning,
  no new computations, no judge and no reference labels. Both workflows accepted
  an assessment at the first point. The second new-workflow assessment required
  one ordinary format correction. Raw failures were retained, not counted as
  accepted decisions or zero scientific scores.
- An initial restricted probe exposed a contract-advertising mismatch: its tools
  allowed only assessment, while its operation-group description advertised reads.
  Operation groups now reflect the actual exposed functions. The restricted
  probe also explicitly removes its historical read allowance. Full-workflow
  reads remain available under the existing policy.
- At both points, old and new workflows favored related dimer electronic-state
  analyses. The new workflow recorded alternatives and a solvent-polarity test,
  but did not propose contact-versus-isolated torsional constraints. This does
  not establish that it overcomes evidence-availability bias.
- At the second point the new arm used 113,870 tokens including correction,
  versus 52,512 for the accepted old-arm attempt. These exploratory probes do not
  estimate average overhead, and failed pilot attempts add further cost.

At the end of the first validation, no chemical forks or complete case reruns had
been launched. The feature remains experimental. Before a long run, the open design question is
whether experiment-option generation still inherits the initial candidate focus;
recording a choice alone does not establish broader scientific exploration.

## Optional experiment generation

`command_experiment_options` requires `command_investigation_policy` and an
enabled result-impact Reviewer. Both features remain off in the default profile.
When enabled, the existing result-impact Reviewer
receives the live compact tool catalog and can include up to three optional
experiment proposals in the same report. There is no additional review call or
agent. Full tool parameters are still loaded on demand by the executing agent.

A proposal contains its motivating source IDs, affected labels, comparison,
possible outcomes, and an optional route/control design. The runtime gives it a
case-local ID in the existing question store and marks its origin as Reviewer
advice. Planner sees it beside its own questions and may select, edit, defer or
decline it. No score, ranking or computation changes when advice is recorded.

Source and route identities are validated against the case and the review's read
scope. Unknown references receive the existing bounded local correction. Failed
calculations may motivate an alternative test, but never become supporting or
negative physical evidence. A proposal is not permission to execute a disabled
tool; normal prerequisites and execution checks remain independent.

Identical proposals do not reopen a question after Planner changes its status.
Semantically similar but differently worded proposals are not silently merged by
code: previous proposals are supplied to Reviewer, and redundancy must be measured
in validation. The experiment list is advice, not a requirement to exhaust it.

The second diagnostic used two isolated continuations of the same actual saved
terminal checkpoint. Delivery is explicitly reopened and the last impact review
replayed in both arms; observations and scientific judgments are preserved.
This is not a mid-run checkpoint or a fresh benchmark run. Each arm permits two
new computations, twenty model calls and an observed 450,000-token boundary
(one response can overshoot). No judge or hidden reference is loaded. A voluntary
early conclusion, a limited probe, and a runtime failure are reported separately.

## Second validation outcome (2026-11-10)

Full offline regression before the frozen online comparison: 3,439 passed and
4 skipped. The feature leaves scientific scores, rankings and execution permissions
unchanged when advice is recorded. Tests cover case ownership, local reference
correction, duplicate suggestions, resume, Planner edits and the disabled mode.
After adding the configuration dependency guard, the full regression again
passed with the same 3,439 passes and 4 skips. Ruff and diff checks passed.

Both online arms used the same actual terminal checkpoint, model (`deepseek-flash`,
high reasoning), tools and allowances. The control retained investigation choices;
the treatment additionally enabled experiment proposals. Thus this comparison
isolates proposal availability, not the entire new workflow versus a historical
release. It is a diagnostic continuation, not a complete benchmark evaluation.

| Observed quantity | Choices only | Choices plus experiment proposals |
| --- | ---: | ---: |
| New calculations | 2 (1 success, 1 failure) | 1 (success) |
| Model calls | 20 | 13 |
| Input tokens | 427,168 | 452,271 |
| Output tokens | 41,054 | 32,748 |
| Total tokens | 468,222 | 485,019 |
| Wall time | 20 min 10 s | 9 min 34 s |
| Acceptance boundary | Model-call allowance | Observed-token allowance |

The control checked a second dimer pose and then attempted bright-state spatial
analysis. A Worker task implicitly locked the default state root even though the
delegation requested selecting the brightest state. After two local rejections,
Planner requested that state directly. The numerical analysis then failed its
normalization/dipole checks; its values were not accepted as evidence.

The treatment generated one accepted advisory question, and Planner selected and
executed it. This was also the second-dimer-pose check, not a new mechanism route.
A later proposal contained a mistyped geometry ID. Local correction retained the
calculation and review but the model dropped the proposal instead of repairing it.
The selected output setting (`td_tout=1` rather than the control's `10`) also left
the new bundle unsuitable for the proposed downstream spatial analysis.

Both arms raised aggregate confidence from 0.37 to 0.40; Packing stayed at 0.20,
and the other candidate confidences were unchanged. The treatment subsequently
requested conclusion, claiming no remaining useful feasible contrast. The compact
catalog did contain contact-versus-isolated torsional comparison as an available
route, but neither arm proposed or tested it. Availability alone does not establish
that this route would succeed or change the ranking.

Both runs paused at predefined probe boundaries without an unhandled exception;
neither finalized or called a judge. The control had not finished assimilating its
last failed result. The treatment had recorded a conclude choice, but had not
completed final delivery. The shorter treatment duration is not an efficiency win:
it performed less computation and consumed 3.6% more tokens. The first treatment
review alone used 9,295 more input tokens after adding the catalog and proposal
contract. One run per arm cannot estimate variance or generalization.

The original checkpoint and artifacts were hash-verified unchanged. Frozen code,
raw responses, local errors, parameter choices, setup-only failures and both
continuations are retained in the private audit, not the public repository.

Conclusion: the optional proposal-to-selection interface works, but broader
scientific exploration and lower cost are not established. Keep the feature off
by default. Address proposal diversity, explicit versus implicit Worker parameter
constraints, and downstream artifact requirements before a larger rollout.
