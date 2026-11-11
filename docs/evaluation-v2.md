# Evaluation v2 Design (ACL/ARR 2027 revision)

Status: design, 2026-11-11. Branch `eval-v2`. Supersedes nothing yet; v2/v3 judges stay
available for historical comparison.

## 1. Why

The AAAI-26 reviews (R1) identified that the evidence score measured *rubric-conforming
prose*, not a verified evidence chain. Evaluation v2 keeps the task (SMILES only, fixed
11-family pool, top-3 with evidence) and the story ("evidence should help discover and
compare mechanisms, not only explain them"), and changes what is scored and how its
validity is demonstrated.

| R1 concern | v2 response |
| --- | --- |
| Judge sees evidence text, not artifacts; fabricated numbers get credit | Layer A deterministic grounding against recorded artifacts; Layer B judge sees verified artifact excerpts |
| Evidence "fits the rubric" | Rubric v4 frozen before new system runs; perturbation suite proves the score reacts to falsified evidence and not to paraphrase |
| Support@Top1 thresholds partial = sufficient | Report Sufficient@1 (e=1) and Partial+@1 (e>=0.5) separately |
| Same model judges and is evaluated | Judge from a different model family than every evaluated backbone; second judge for agreement; human subset |
| No CIs, label imbalance | Paper-clustered paired bootstrap; per-family and macro metrics; SMILES-blind floors in every table |
| Not compute-matched | Uniform cost accounting per method; equal-budget baselines (see experiments plan) |
| SMILES-only vs condition-specific labels; contamination | Dataset analyses: series/near-duplicate and label-conflict tables, leave-one-paper-out structural kNN floor, memorization probes |

## 2. Submission contract v4

Every method (MechCAL, ReAct, Codex, Direct LLM) submits the same object. Evidence items
become typed claims so that grounding can be checked by code rather than inferred from prose.

```json
{
  "mechanism_predictions": [
    {"label": "ICT_TICT_CT", "rank": 1, "evidence": [
      {"statement": "S1 has 0.82 hole on the triphenylamine and 0.71 electron on the dicyanovinyl.",
       "kind": "computed",
       "quantities": [{"name": "hole_fraction_donor", "value": 0.82, "unit": "",
                       "source": {"artifact_id": "A017", "pointer": "/fragments/donor/hole"}}]},
      {"statement": "Donor N1 is conjugated to acceptor C21 through the phenylene bridge.",
       "kind": "structural", "atom_ids": [1, 14, 21]}
    ], "limitations": ["Vertical excitation only; no relaxed S1."]}
  ],
  "artifacts": "path or inline list of artifact records (tool, inputs incl. canonical SMILES, status, outputs)"
}
```

`kind` is one of `structural` (checkable against the SMILES graph), `computed` (must cite
an artifact), or `prior` (general chemical knowledge; allowed, never counts as case-specific
support). Free-text-only submissions remain scorable: their claims are classified as
`unsourced` and can reach at most the rubric's structural/partial routes.

Artifact records are produced by a thin recorder around tool calls, so ReAct/Codex/ChemGraph
baselines are graded on the same footing as MechCAL. A method that never calls tools has an
empty artifact list; that is a property of the method, not a parse failure.

## 3. Layer A — deterministic grounding (no LLM)

Per claim, the checker emits exactly one outcome:

| Outcome | Rule |
| --- | --- |
| `grounded` | artifact exists, belongs to this case (canonical SMILES/InChIKey match), status success, every quantity matches the pointed value within tolerance |
| `value_mismatch` | artifact valid but a cited value differs beyond tolerance |
| `wrong_molecule` | cited artifact was computed for a different structure |
| `failed_artifact` | cited artifact is a failed/partial calculation used as if successful |
| `missing_artifact` | cited ID does not exist |
| `unsourced` | `computed` claim with no source, or a number in a structural/prior claim |
| `structural_ok` / `structural_bad` | atom IDs exist and the asserted motif holds (RDKit check where a checker exists; otherwise `structural_unchecked`) |

Tolerance: relative 2% or the printed precision of the artifact, whichever is larger;
units normalized for eV/nm/Debye/Hartree. Layer A outputs per-prediction counts and a
`fabrication_rate` = (value_mismatch + wrong_molecule + missing_artifact) / computed claims.

## 4. Layer B — rubric v4 sufficiency judge

- Starts from rubric v3 (route `any_of` → condition `all_of`, cited findings, validator).
- Only `grounded` computed claims and `structural_ok` claims may satisfy a route condition.
  The validator enforces this in code, as v3 already enforces quote provenance.
- The judge sees: SMILES graph facts, the claims with Layer A outcomes, and bounded excerpts
  of the cited artifacts (v3 `ToolSourcePacket`). It never sees rank, method identity,
  internal confidence/claim status, hidden reference or source paper.
- Rubric refinement: each family states explicitly (a) what a structure-only argument can
  reach (ceiling), (b) which computed observables qualify for partial vs strong, and
  (c) known proxy overreach. Rubric, prompt and schema hashes are frozen and recorded
  before any MechCAL v2 run; later edits create v5, never a silent change.
- Final e_j ∈ {0, 0.5, 1}. A prediction with any `value_mismatch`/`wrong_molecule` claim used
  in its selected route is capped at 0 (fabrication is not partial support).

## 5. Metric validity: perturbation suite

Applied to real submissions from several methods; each operator edits one prediction.

| Operator | Edit | Expected |
| --- | --- | --- |
| `tamper_value` | multiply a key cited quantity (×10 or sign flip) in the statement only | drop |
| `swap_molecule` | replace cited artifacts with the same tool's output for another case | drop |
| `flip_conclusion` | invert the stated direction (bright→dark, CT→local) | drop |
| `strip_artifacts` | keep text, remove artifact links | drop to structural ceiling |
| `inject_fabrication` | add a plausible, uncited numeric claim | no gain |
| `paraphrase` (control) | meaning-preserving rewrite | no change |
| `reorder` (control) | reorder evidence items | no change |

Reported per judge configuration (v2 text-only, v3, v4): detection rate (score decreases),
mean Δe, false-alarm rate on controls. The headline is the contrast: text-only judging is
fooled by tampering; v4 is not, and both are stable on controls.

## 6. Statistics and reporting

- Every table: constant top-3 floor and frequency floor, plus leave-one-paper-out kNN floor.
- Per-family recall/precision@3 and macro averages alongside micro means.
- Paired bootstrap resampling **source papers**, not cases (74/119 cases share a paper).
- Repeated runs (≥3 seeds) on a fixed subset; report run-to-run spread.
- Judge agreement: Cohen's κ between two judges; human subset with confusion matrix and
  false-sufficient rate.

## 7. Dataset analyses (answer R1 rebuttal Q1/Q2)

- Series/near-duplicate clusters (shared DOI, Morgan Tanimoto ≥ 0.7) and within-cluster
  label agreement; list structurally similar pairs with different primary labels.
- SMILES-only ceiling estimates: leave-one-paper-out nearest-neighbour label transfer.
- Contamination probes (API needed): name/source recall from SMILES, randomized-SMILES
  control (same atoms, scrambled graph), performance by publication year.

## 8. Cost accounting

Each run records LLM calls, input/output/cached tokens per role, tool calls, tool wall time
and dollar cost under a fixed price table. Comparisons are reported at matched budgets.

## Implementation order

1. Offline statistics + dataset analyses (`mechcal/eval/stats.py`, `mechcal/eval/dataset_analysis.py`).
2. Submission contract v4 + Layer A checker (`mechcal/eval/grounding.py`), with tests on synthetic artifacts.
3. Perturbation operators (`mechcal/eval/perturb.py`).
4. Rubric v4 + Layer B validator changes; freeze manifest.
5. API-dependent: judge runs, probes, human-subset export.
