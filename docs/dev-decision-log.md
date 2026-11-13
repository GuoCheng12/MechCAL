# Development decision log (release 0.3.0)

Every design decision taken during development, and whether it was informed by results on
development cases (`benchmarks/photomechbench/splits/raw_v2_119_dev_test.json`). Test cases were
never run. "Dev-informed" means the decision used development runs or judge feedback on them;
it does not mean the decision encodes any case's answer.

| Date | Decision | Basis | Dev-informed |
|---|---|---|---|
| 2026-11-12 | Professor-student protocol (scientific settings with the Planner, technical with the Worker; interceptions, retries, report checks) | Design goal; failure modes seen in 0003 smokes | yes (0003) |
| 2026-11-12 | Gateway: duplicate-request guard, run wall budget, lineage, warnings, failure classes | Repeated identical PySCF calls in 0003 smoke 3 | yes (0003) |
| 2026-11-12 | Cost visibility and delegation cost check | PBE0 chosen without cost in 0003 smoke 6 | yes (0003) |
| 2026-11-12 | Native Amesp S0 optimizer, per-backend threads | 11 benchmark molecules (timing/energy only, no labels) | no labels used |
| 2026-11-12 | Capability cards replace truncated descriptions; recipes listed as plain combinations; applicability from SMILES | Tool-use statistics over dev runs (2/3 of delegations via recipes; tools never chosen) | yes (tool usage, not labels) |
| 2026-11-12 | `fixed_args` (scientific) vs `suggested_args` (technical) | Needless interceptions in dev runs (0141, 0047) | yes (protocol events) |
| 2026-11-12 | 12 overlapping or unused tools removed from the agents' view (8 `run_targeted_*`, bright/dark ordering, torsion snapshots, conformer state probe, CT descriptor reader) | Zero use in dev runs and duplicated outputs | yes (tool usage) |
| 2026-11-12 | Derived descriptors in results: radiative-rate estimate, state gaps, torsion S0/S1 profile, two-fragment transfer share | Standard photophysics observables; the judge's notes on dev runs showed these were missing | yes (judge notes on dev runs) |
| 2026-11-12 | `dimer_excited_states` recipe | Standard intermolecular excited-state analysis; judge notes on dev runs | yes (judge notes on dev runs) |
| 2026-11-12 | PySCF: level shift, SCF non-convergence code, completion status, time limit from cost model; cost model scaled by basis size with a metal caveat | PySCF failures in dev runs (0141 Fe, 0003 Au) | yes (failures, not labels) |
| 2026-11-12 | Torsion angle and dihedral-id spellings normalized; geometry measurement shorthand | Argument-format failures in dev runs | yes (failures) |
| 2026-11-12 | Dev/test split by paper cluster | Coverage of primary families in the dev set | yes (dev labels only) |
| 2026-11-13 | Two-fragment analysis reports net transfer and same-fragment share; the card explains that a symmetric delocalized excitation gives a 50 % inter-fragment share with zero net transfer (C1a) | Definition error found when reading dev runs: a 50 % IFCT share was read as charge transfer or coupling | yes (reading of dev runs; no labels) |
| 2026-11-13 | Recipes accept the settings their tools take (dihedral atoms, custom fragments, PySCF functional/basis) (C1a) | Recipe argument gaps seen in dev runs | yes (failures) |
| 2026-11-13 | Prior and evidence kept apart on the board: per-family prior with its basis; per-claim `discriminates` (yes / partly / no) judged by the Planner; Critic flags a top-3 family resting only on non-discriminating evidence, and missing priors (C1b) | W1 report: the Planner ranked by how easily a family could be supported, not by how likely it is to be reported | yes (per-family statistics on dev runs) |
| 2026-11-13 | Direct and ReAct baselines on the dev set with the same cards and budgets; perturbation runner on real submissions; a run without a valid answer is graded as an empty answer (C1c) | Evaluation design (§9) | no |
| 2026-11-13 | Cost log fix: each run appends only its own records (loaded records were written back by every run, so the log doubled and reached 17 GB, slowing each model step by about 60 s and weighting estimates toward the earliest records); the log was collapsed to unique records | Found when dev4a runs took about 60 min instead of 22 | no (software fault) |
| 2026-11-13 | Structure corrections: raw_v2_119 stays frozen; `corrections/structure_corrections.json` lists the four development SMILES found wrong against the source papers (0020 CF3 position, 0141 two H missing, 0152 methylchrysene for 2-methylbenz[a]anthracene, 0153 picene for benzo[a]pyrene); `examples/make_corrected_dataset.py` builds `raw_v2_119_corrected` (every case with a `structure_verification` record; references unchanged) and all readers use it via `mechcal/eval/dataset.py`. 0141 is marked not runnable (no SMILES gives an RDKit ferrocene sandwich; earlier runs used a sigma-bonded geometry) and is left out of default dev runs | SMILES verification of the 17 dev cases against the papers (formula/HRMS, OPSIN/PubChem names, figures, MolScribe), done without labels or agent results; requested by the author | no (structure verification; no labels) |
