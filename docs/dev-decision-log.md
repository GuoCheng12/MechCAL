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
