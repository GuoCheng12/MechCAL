# Changelog

## 0.2.0 (Unreleased)

- Integrate the current native-command Planner/Worker workflow, case memory,
  incremental evidence review and autonomous worker tasks.
- Include expanded RDKit, Amesp, xTB, CREST, DFTB+, PySCF and PySOC adapters.
- Add the installed `mechcal` CLI with portable TOML configuration, local
  dependency/MCP checks, explicit connectivity probe and fresh-run entry.
- Keep credentials, private references, inference histories, model weights
  and third-party binaries outside the code distribution.
- Deployment changes preserve the current scientific methods, mechanism pool
  and rubric. Include tested direct-dependency versions for Linux/Python 3.11.

- Add reconstructed PhotoMechBench inputs: 119 case IDs and SMILES, excluding
  historical case 0127. Preserve the remaining case IDs without renumbering.
- Include dataset format notes and offline input-integrity tests. Hidden
  references, labels, evidence records and model outputs remain excluded.

## 0.1.0

- Standalone MechCAL package and command-line interfaces.
- Planner, coverage review, evidence workers, and support audit.
- RDKit and optional Amesp tools, with local and MCP transports.
- Mechanism-ranking metrics, evidence rubric, and baseline adapters.
- Direct, tool-augmented, common-prior, and adapted ChemGraph evaluation.
- Source-only distribution with offline regression tests.

This release packages the existing research implementation. It does not
include benchmark cases, hidden references, inference records, weights,
credentials, or historical experiment results.
