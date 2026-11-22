"""Where benchmark inputs and references are read from: the rebuilt PhotoMechBench snapshot ``rebuilt_302``.

``benchmarks/photomechbench/rebuilt_302`` holds one file per case (public SMILES, structure
verification record and hidden reference with a supported / contradicted / not_established decision
for every family). It is read only. Agents get only the SMILES (``public_smiles`` returns that one
string); the evaluation side reads the case files. A case whose structure cannot yet be prepared
correctly is marked not runnable.

The earlier 119-case data stays in place for its own builders and tests: ``raw_v2_119`` is the frozen
original snapshot (its checksums are tested) and ``raw_v2_119_corrected`` (``CORRECTED_119``) is
generated from it and ``corrections/structure_corrections.json`` by ``examples/make_corrected_dataset.py``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmarks/photomechbench"
RAW = BENCH / "raw_v2_119"
CORRECTIONS = BENCH / "corrections/structure_corrections.json"
CORRECTED_119 = BENCH / "raw_v2_119_corrected"
DATASET = BENCH / "rebuilt_302"
CASES_DIR = DATASET / "cases"

NOT_RUNNABLE = "{case} cannot be run yet: {reason} Pass {flag} to run it anyway."


def public_smiles(case_id: str) -> str:
    try:
        return case_record(case_id)["public_input"]["molecule"]["structure"]["value"]
    except FileNotFoundError:
        raise KeyError(f"{case_id} not in the dataset") from None


@lru_cache(maxsize=512)
def case_record(case_id: str) -> dict:
    return json.loads((CASES_DIR / f"{case_id}.json").read_text(encoding="utf-8"))


def structure_verification(case_id: str) -> dict:
    path = CASES_DIR / f"{case_id}.json"
    return case_record(case_id).get("structure_verification", {}) if path.is_file() else {}


def runnable(case_id: str) -> tuple[bool, str]:
    """(True, "") unless the case's structure is marked as not yet preparable (e.g. a metallocene)."""
    v = structure_verification(case_id)
    return (v.get("runnable", True), v.get("not_runnable_reason", ""))
