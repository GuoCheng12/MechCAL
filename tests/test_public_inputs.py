import csv
import hashlib
import json
import re
from pathlib import Path

import pytest


DATASET = Path(__file__).resolve().parents[1] / "benchmarks" / "photomechbench"


def read_inputs():
    with (DATASET / "raw_v2_119.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["case_id", "smiles"]
        return list(reader)


def test_public_input_contract():
    rows = read_inputs()
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    assert len(rows) == manifest["case_count"] == 119
    assert len({row["case_id"] for row in rows}) == 119
    assert not set(manifest["excluded_case_ids"]) & {row["case_id"] for row in rows}
    assert all(re.fullmatch(r"AIE_DDX_MECH_V4_\d{4}", row["case_id"]) for row in rows)
    assert all(set(row) == {"case_id", "smiles"} and row["smiles"] for row in rows)
    assert [row["case_id"] for row in rows] == sorted(row["case_id"] for row in rows)
    assert hashlib.sha256((DATASET / manifest["file"]).read_bytes()).hexdigest() == manifest["sha256"]


def test_structures_parse_and_are_unique():
    chem = pytest.importorskip("rdkit.Chem")
    graphs = []
    for row in read_inputs():
        molecule = chem.MolFromSmiles(row["smiles"])
        assert molecule is not None, row["case_id"]
        graphs.append(chem.MolToSmiles(molecule))
    assert len(set(graphs)) == 119


def test_existing_csv_reader():
    from mechcal.benchmark import load_benchmark_cases

    rows = read_inputs()
    cases = load_benchmark_cases(DATASET / "raw_v2_119.csv")
    assert [(c.case_id, c.smiles) for c in cases] == [
        (row["case_id"], row["smiles"]) for row in rows
    ]
