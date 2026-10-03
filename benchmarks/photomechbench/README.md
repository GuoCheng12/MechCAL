# PhotoMechBench Molecular Inputs

`raw_v2_119.csv` contains the 119 molecular inputs from the reconstructed
PhotoMechBench dataset, version `raw-v2-119-20261004`.
Case `AIE_DDX_MECH_V4_0127` is excluded. The remaining case IDs are retained;
the numeric suffixes are identifiers, not consecutive row numbers.

## Format

The UTF-8 CSV has two columns:

- `case_id`: stable case identifier.
- `smiles`: molecular structure for inference.

The file contains no mechanism labels, hidden references, evidence text,
paper metadata, inference records, credentials or provider configuration.
All 119 SMILES parse with RDKit and yield distinct canonical structures.
The shared eleven-family mechanism pool remains in `mechcal/mechanism_pool.py`.

These are reconstructed inputs, not the original 120-case paper evaluation
set. Do not associate historical performance numbers with this version
without rerunning the corresponding experiment.

## Load Inputs

From a repository checkout in the MechCAL Python 3.11+ environment:

```python
from pathlib import Path
from mechcal.benchmark import load_benchmark_cases

cases = load_benchmark_cases(Path("benchmarks/photomechbench/raw_v2_119.csv"))
assert len(cases) == 119
```

Loading does not make model API calls. To infer a single molecule, pass its
SMILES and case ID to `mechcal-run-case` as shown in the project README.
The evaluation commands require separate case JSON files with hidden
references; this input-only CSV cannot compute Recall@3 or ES-nDCG@3.

## Versioning And Checks

`manifest.json` records the version, excluded ID, row count and CSV SHA-256.
Run `pytest -q tests/test_public_inputs.py` to check the public export.
Install the `science` extra to also run the RDKit structure checks.

When extending the dataset, retain existing IDs and publish a new versioned
CSV and manifest. Do not overwrite this snapshot or reuse excluded IDs.
The input files are included in the Git repository and source distribution,
but not in the Python wheel.
