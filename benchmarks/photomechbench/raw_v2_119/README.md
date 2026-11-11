# PhotoMechBench Raw Data V2

119-case extension base. Case 0127 is excluded; existing case IDs are retained.
This is the current reconstruction dataset, not a claim of independent human certification.
Scientific follow-up notes are retained in `audit/` and do not block inclusion.
No new inference or judge evaluation is performed when building this package.

## Layout

- `cases/`: authoritative full records, one JSON per case.
- `public_inputs.jsonl`: case ID and SMILES only, for model input.
- `inputs.csv`: the same IDs and SMILES; compatible with the existing CSV input reader.
- `references.jsonl`: hidden source evidence, mechanism labels and roles.
- `mechanism_pool.json`: shared eleven-family taxonomy.
- `schema/case.schema.json`: generated JSON Schema.
- `manifest.json`: case membership, statistics and checksums.
- `validation.json`: format check results.
- `audit/`: inherited review notes, separate from the stable case format.
- `excluded_cases.json`: excluded historical IDs; do not recycle them.
- `metadata_repairs.json`: source-title completions with PDF page provenance.

The full case contains `schema_version`, `case_id`, `public_input`,
`hidden_reference` and `provenance`. Its schema version is `photomechbench.raw.v2`.
The public SMILES is at `public_input.molecule.structure.value`.
Names, DOI, source evidence and reference labels are hidden, not model inputs.
The global mechanism pool and task prompt are supplied by the runner, not repeated per case.
These exports contain data, not a replacement for the shared task prompt/output schema.

`hidden_reference` contains `target_name`, `source_documents`,
`reference_evidence_units`, `reference_mechanisms` and `mechanism_decisions`.
Each source records its local source ID, title, DOI, PDF checksum and page count.
No machine-specific PDF path is required in a case. PDFs are not included in this package.
Page numbers are physical PDF pages starting at 1, not printed article pagination.
Source and evidence IDs are unique within a case, not necessarily across cases.

Each family has exactly one decision: `supported`, `not_established`, or `contradicted`.
Only supported decisions enter `reference_mechanisms`. Primary and co-primary roles
have gain 2; secondary roles have gain 1. The two representations must agree.
Preserve the source-supported number of labels; do not force exactly three reference labels.
Three is the model prediction count, not a constraint on the hidden reference.

## Validation

From the project root, with the reconstruction environment:

```bash
var/benchmark_rebuild/audit_env/bin/python -m aie_mas_v2.benchmark_rebuild.package_dataset validate PATH_TO_DATASET
```

Checks cover JSON types, unique case IDs, parseable SMILES, duplicate canonical graphs,
source/page references, evidence links, all eleven decisions, and role/gain consistency.
Validation also checks manifest statistics, case checksums and derived-export agreement.
Parsing verifies format and valence handling; it does not certify correspondence to a paper.
Existing material-review notes are not converted into automatic exclusions.
The validator runs in the reconstruction environment. Running the full MechCAL
application additionally requires its runtime environment (Python 3.11 or newer).

## Adding Cases

1. Work in a copy/new dataset version, preserving the current 119-case snapshot.
2. Allocate an unused `AIE_DDX_MECH_V4_NNNN` ID. IDs are identifiers, not sample counts.
   Check all historical IDs, including excluded ones; never renumber old cases.
3. Use an existing case as a field-layout example. Replace all molecule, source,
   evidence, decision and provenance values with the new case's actual records.
4. Store it as `cases/CASE_ID.json`. Provide a real source PDF checksum and page count.
   Provenance hashes refer to the new case's own upstream raw and label records,
   not those of the example. Reuse the construction pipeline to generate those records.
5. Run validation and refresh the derived files:

```bash
var/benchmark_rebuild/audit_env/bin/python -m aie_mas_v2.benchmark_rebuild.package_dataset validate PATH_TO_NEW_DATASET --refresh
```

The refresh command updates JSONL, CSV, case membership, statistics and checksums.
It accepts any nonzero case count, not a hardcoded 119. Update the dataset version
in `manifest.json` when creating a new version. Keep scientific annotation changes
separate from format-only edits. Never feed `cases/` or `references.jsonl` directly
to an inference prompt because they contain hidden answers.
