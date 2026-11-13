"""Where benchmark inputs and references are read from: the structure-corrected copy of raw_v2_119.

``benchmarks/photomechbench/raw_v2_119`` stays the frozen original snapshot (its checksums are
tested). ``raw_v2_119_corrected`` is generated from it and from
``benchmarks/photomechbench/corrections/structure_corrections.json`` by
``examples/make_corrected_dataset.py``: SMILES that were checked against the source papers and found
wrong are replaced, every case carries a ``structure_verification`` record, and references and
labels are unchanged. Agents read only the SMILES (``inputs.csv``); the evaluation side reads the
case files. A case whose structure cannot yet be prepared correctly is marked not runnable.
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmarks/photomechbench"
RAW = BENCH / "raw_v2_119"
CORRECTIONS = BENCH / "corrections/structure_corrections.json"
DATASET = BENCH / "raw_v2_119_corrected"
CASES_DIR = DATASET / "cases"
PUBLIC_INPUTS = DATASET / "inputs.csv"

NOT_RUNNABLE = "{case} cannot be run yet: {reason} Pass {flag} to run it anyway."


@lru_cache(maxsize=1)
def _inputs() -> dict[str, str]:
    with PUBLIC_INPUTS.open(encoding="utf-8", newline="") as fh:
        return {row["case_id"]: row["smiles"] for row in csv.DictReader(fh)}


def public_smiles(case_id: str) -> str:
    try:
        return _inputs()[case_id]
    except KeyError:
        raise KeyError(f"{case_id} not in public inputs") from None


@lru_cache(maxsize=256)
def case_record(case_id: str) -> dict:
    return json.loads((CASES_DIR / f"{case_id}.json").read_text(encoding="utf-8"))


def structure_verification(case_id: str) -> dict:
    path = CASES_DIR / f"{case_id}.json"
    return case_record(case_id).get("structure_verification", {}) if path.is_file() else {}


def runnable(case_id: str) -> tuple[bool, str]:
    """(True, "") unless the case's structure is marked as not yet preparable (e.g. a metallocene)."""
    v = structure_verification(case_id)
    return (v.get("runnable", True), v.get("not_runnable_reason", ""))
