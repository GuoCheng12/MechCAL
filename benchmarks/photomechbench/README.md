# PhotoMechBench

This directory releases the reconstructed **119-case** PhotoMechBench dataset,
version `raw-v2-119-20261004`, including molecular inputs, reference mechanisms,
source-linked evidence and annotation provenance. Original annotations are
released under [CC BY 4.0](LICENSE.md); MechCAL code remains MIT licensed.

| Snapshot statistic | Count |
| --- | ---: |
| Cases / distinct canonical molecular structures | 119 |
| Source paper DOIs | 76 |
| Mechanism families | 11 |
| Reference mechanism assignments | 401 |
| Source-linked evidence records | 998 |

Case `AIE_DDX_MECH_V4_0127` is excluded because its source target and linkage
identity were unresolved. Remaining IDs are retained, not renumbered.
Cases can share a paper and have multiple reference mechanisms. There is no
requirement that a case have exactly three reference labels.

## Files

```text
raw_v2_119.csv                  Inference inputs: case_id,smiles only
manifest.json                  Existing CSV version and checksum
raw_v2_119/
  cases/<case_id>.json          Complete case and reference annotations
  audit/<case_id>.json          Preserved reconstruction/review notes
  schema/case.schema.json       JSON Schema for a complete case
  mechanism_pool.json          Fixed eleven-family taxonomy
  manifest.json                Membership, statistics and original checksums
  excluded_cases.json           Excluded case and reason
  metadata_repairs.json         Source-title repairs from reconstruction
LICENSE.md                     Annotation license and third-party notice
```

Paper/SI PDFs, full-text archives, model responses, inference histories,
credentials and provider settings are **not** distributed. Source records
provide titles, DOIs, page counts and PDF checksums. Evidence records include
short attributed source excerpts, not copies of the source publications.

## Input And Reference Separation

The UTF-8 CSV has two columns:

- `case_id`: stable case identifier.
- `smiles`: molecular structure for inference.

Use this CSV for inference. It contains no labels, evidence or paper metadata.
Only the current molecule's SMILES, the shared mechanism pool, and common
task/output instructions belong in the inference prompt.

Full JSON cases keep references in `hidden_reference`. Here **hidden means
withheld from the inference system**, not unavailable to dataset users.
Do not give a subject agent the complete case JSON or access to its reference
directory. Reference records also stay out of the evidence-support judge's
input; the judge assesses submitted evidence, and the metric separately uses
reference relevance. Public availability does not relax this closed-book rule.

## Case Format

Each case has five top-level fields:

- `schema_version`: `photomechbench.raw.v2`.
- `case_id`: stable identifier, matching the filename.
- `public_input.molecule.structure`: SMILES format and value.
- `hidden_reference`: target name, sources, evidence and mechanism annotations.
- `provenance`: reconstruction method, source snapshot and record checksums.

Within `hidden_reference`, `reference_mechanisms` supplies each supported
`label`, its `role`, relevance `gain`, `evidence_ids` and `rationale`.
`primary` and `co_primary` have gain 2; `secondary` has gain 1. These gains
are reference relevance weights, **not** probabilities or judge support scores.
`mechanism_decisions` contains one decision for each of the eleven families.
Only `supported` decisions become reference mechanisms; `not_established`
does not mean that a mechanism has been disproven.

Each `reference_evidence_units` record has a local evidence ID, source ID,
PDF page, quote, subject, scope, conditions, evidence kind, statement and
limitations. Pages are physical PDF pages, starting at 1. Source and evidence
IDs are case-local: use `(case_id, evidence_id)` when combining cases.

## Reconstruction Status

These are reconstructed inputs, not the original 120-case paper evaluation
set. Do not associate historical performance numbers with this version
without rerunning the corresponding experiment.

The saved annotation method is `assistant_source_reconstruction`. This release
preserves the existing records, including audit notes and pending human-review
fields. It does not assert a new independent expert certification. Some audit
notes describe earlier deferred structure checks; retain them as historical
provenance rather than silently treating them as completed reviews. SMILES
parseability and canonical uniqueness do not by themselves certify molecular
identity against a paper. Missing SI and case-specific scope limitations are
recorded alongside the evidence.

## Load And Evaluate

From a repository checkout in the MechCAL Python 3.11+ environment:

```python
from pathlib import Path
from mechcal.benchmark import load_benchmark_cases

cases = load_benchmark_cases(Path("benchmarks/photomechbench/raw_v2_119.csv"))
assert len(cases) == 119
```

Loading does not make model API calls. To infer a single molecule, pass its
SMILES and case ID to `mechcal ... run` as shown in the project README.
Read reference annotations separately, after inference:

```python
import json

case = json.loads(Path(
    "benchmarks/photomechbench/raw_v2_119/cases/AIE_DDX_MECH_V4_0003.json"
).read_text(encoding="utf-8"))
references = case["hidden_reference"]["reference_mechanisms"]
```

The existing evaluation loaders accept this case format directly. Set
`--case-dir benchmarks/photomechbench/raw_v2_119/cases` for the historical
evaluation entry points. These are not aliases for the current inference
workflow. See [Evaluation](../../docs/evaluation.md) for metric versions,
judge configuration and the synthetic offline scoring example. No model calls
or benchmark reruns were made to prepare this data release.

## Versioning And Checks

The original CSV and all complete case records are unchanged from the saved
reconstruction snapshot. The two manifests separately verify the CSV and case
files. The reference manifest also retains hashes of private construction
inputs for provenance; those source files and PDF binaries are not bundled.

From the repository root, run:

```bash
pytest -q tests/test_public_inputs.py tests/test_public_benchmark.py
```

These offline checks cover all cases, JSON Schema, source/page/evidence links,
roles and gains, taxonomy coverage, statistics, exclusion, CSV/reference
agreement, and compatibility with evaluation loaders. Install the `science`
extra to run the RDKit parseability and uniqueness checks as well. The checks
validate data integrity, not the truth of each scientific interpretation.

When extending the dataset, retain existing IDs and publish a new versioned
snapshot and manifest. Do not overwrite this snapshot or reuse excluded IDs.
The data are included in Git and the source distribution, not the Python wheel.

## Attribution

Credit the PhotoMechBench contributors, dataset name, version and repository:

> PhotoMechBench contributors. PhotoMechBench, reconstructed dataset
> raw-v2-119-20261004. https://github.com/GuoCheng12/MechCAL/tree/main/benchmarks/photomechbench.
> Original annotations licensed under CC BY 4.0.

Include the commit ID used in an experiment and indicate any annotation changes.
Retain attribution to the original papers via the per-case source records.
