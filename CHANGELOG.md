# Changelog

## 0.2.0 (Unreleased)

- Complete the original PhotoMechBench raw-v2-119 snapshot: include all 249
  files unchanged, including JSONL/CSV exports and historical validation notes.
  Add an external full-file SHA-256 inventory and narrowly allow the two dataset
  JSONL files in Git and source distributions; runtime logs remain excluded.
- Publish reconstructed PhotoMechBench references under CC BY 4.0: 119 cases,
  401 reference mechanisms, 998 evidence records and 76 source DOIs, with
  unchanged case files, JSON Schema, provenance and original review notes.
  Keep PDFs, model runs and credentials excluded; references stay out of
  inference prompts. Add offline dataset integrity and loader checks.
- Split Scientific Reviewer and Exploration Agent into independent components,
  clients and private memory scopes; preserve published proposal handoff.
- Send short resident Planner rules and task-scoped background, with exact
  manuals and full review text available through case-local memory. Generate
  context-read fields and operation grouping from the live interface contract.
- Preserve explicit Worker input bindings and calculation provenance across
  delegation, partial failures and review. Record execution history without
  replacing earlier usable results with later failed attempts.
- Reject serialized tool-call envelopes as commentary and return local delivery
  errors without interpreting model text as an executed operation.
- Cache source fingerprints within an immutable case snapshot and persist large
  arrays once in content-addressed run-local files. Preserve historical reads,
  exact source values, and corruption detection. See runtime storage notes.
- Reuse unchanged computational review background through exact receipt paths
  and deltas instead of resending full question/comparison tables.

- Integrate the current native-command Planner/Worker workflow, case memory,
  incremental evidence review and autonomous worker tasks.
- Include expanded RDKit, Amesp, xTB, CREST, DFTB+, PySCF and PySOC adapters.
- Add the installed `mechcal` CLI with portable TOML configuration, local
  dependency/MCP checks, explicit connectivity probe and fresh-run entry.
- Keep credentials, inference histories, model weights
  and third-party binaries outside the code distribution.
- Deployment changes preserve the current scientific methods, mechanism pool
  and rubric. Include tested direct-dependency versions for Linux/Python 3.11.

- Add reconstructed PhotoMechBench inputs: 119 case IDs and SMILES, excluding
  historical case 0127. Preserve the remaining case IDs without renumbering.
- Include dataset format notes and offline input-integrity tests. This initial
  input-only export is now accompanied by the reference release listed above;
  model outputs remain excluded.

## 0.1.0

- Standalone MechCAL package and command-line interfaces.
- Planner, coverage review, evidence workers, and support audit.
- RDKit and optional Amesp tools, with local and MCP transports.
- Mechanism-ranking metrics, evidence rubric, and baseline adapters.
- Direct, tool-augmented, common-prior, and adapted ChemGraph evaluation.
- Source-only distribution with offline regression tests.

Version 0.1.0 packaged the existing research implementation. It did not
include benchmark cases, hidden references, inference records, weights,
credentials, or historical experiment results.
