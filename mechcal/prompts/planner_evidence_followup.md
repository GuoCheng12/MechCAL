evidence_followup retains result-to-candidate suggestions and comparison questions.
These are interpretations, not observations. An overall assessment does NOT clear
pending associations. Address relevant suggestions in update_hypotheses.responses:
reviewed (including unchanged confidence), deferred, or dismissed with a short
reason. Update affected candidate entries in the same call when the judgment
changes. An omitted response stays pending; this does not forbid continuing or
ending with an explicit uncertainty. You can add affected candidates yourself.

Maintain a few useful comparison questions using update_hypotheses.questions.
Each question names candidates, the unresolved issue, a possible test, and how
different outcomes would change your assessment (or remain inconclusive). Reuse
the same ID while refining it. These are scientific notes, not required checklists.
Use question_id on a computation to link it to a saved question; the runtime
records the exact question version and actual result. Preparatory/exploratory
calculations may omit it. Tools are not dispatched from natural-language next_test.
A finished computation does not automatically resolve its question.

Separate a broad mechanism question (scope=mechanism) from a narrower test
(scope=test, parent_id=the broader question ID) when the test only checks part
of an explanation. Resolving the child leaves the parent unchanged. A result
about optical allowedness, contact feasibility or state character answers that
specific question, not automatically emission, restriction or decay. Refine the
remaining parent question, or defer it with the actual computational limitation.
Legacy questions have unspecified scope; do not infer that they are settled.

For a useful controlled comparison, contrast can record varied, held_fixed,
readout and existing baseline_refs; outcomes describes how different results
would affect the explanations. These are optional design notes, not extra tool
parameters. Use normal computation calls for exact inputs and settings. The
runtime links results and execution conditions; verify that the intended controls
actually held. Unknown or changed conditions limit interpretation, not necessarily
the usefulness of the calculation. Exploratory/preparatory work needs no design.
Use relation=alternatives, complementary or unknown for candidates in a question.
A result may support a shared premise without distinguishing the mechanisms.
Shared observations are not independent confirmations of separate labels.
Null/omitted scope, parent_id, relation or contrast retains an existing value.
Empty parent_id detaches; an empty contrast object removes the design notes.

Choose tests that can advance unresolved comparisons, not just strengthen the
current leader. Confidence is not evidence strength, labels need not be exclusive,
and unavailable or unsuccessful tests do not disprove a mechanism. Do not force
new labels, score movements, exhaustive tests, or a particular final ranking.

Current candidate reasons live in exploration_state. Formal arguments are linked
to the judgment version at submission; stale_formal_labels means to check them,
not automatically rewrite or reject them. Read followup:state for omitted items
or complete trial history. Resolve questions explicitly or retain their uncertainty
when finalizing. No extra review or computation is needed just to close a note.
Before stopping, consider unresolved mechanism questions outside the top three
as well. Distinguish a useful executable contrast, a largely redundant test and
a capability gap. Capability gaps are neither negative evidence nor automatic
high ranks. Do not demand wet-lab proof to offer a bounded computational judgment.
