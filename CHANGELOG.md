# Changelog

## Unreleased

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
