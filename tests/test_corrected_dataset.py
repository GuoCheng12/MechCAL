"""Structure-corrected copy of raw_v2_119: reproducible, only the listed SMILES change, readers use it."""

import importlib.util
import json
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from mechcal.eval import dataset
from mechcal.eval.splits import dev_cases, load_split

ROOT = Path(__file__).resolve().parents[1]
P = "AIE_DDX_MECH_V4_"


def builder():
    spec = importlib.util.spec_from_file_location("make_corrected", ROOT / "examples/make_corrected_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_copy_equals_a_fresh_build():
    files = builder().build()
    for path, text in files.items():
        assert (dataset.DATASET / path).read_text(encoding="utf-8") == text, path
    on_disk = {str(p.relative_to(dataset.DATASET)) for p in dataset.DATASET.rglob("*") if p.is_file()}
    assert on_disk == set(files)


def test_only_listed_smiles_change_and_references_are_identical():
    corrections = json.loads(dataset.CORRECTIONS.read_text())
    fixed = {c["case_id"]: c for c in corrections["corrections"]}
    assert set(fixed) == {P + "0020", P + "0141", P + "0152", P + "0153"}
    for raw_path in sorted((dataset.RAW / "cases").glob("*.json")):
        raw = json.loads(raw_path.read_text())
        new = json.loads((dataset.CASES_DIR / raw_path.name).read_text())
        assert new["hidden_reference"] == raw["hidden_reference"] and new["provenance"] == raw["provenance"]
        smiles_raw = raw["public_input"]["molecule"]["structure"]["value"]
        smiles_new = new["public_input"]["molecule"]["structure"]["value"]
        if raw["case_id"] in fixed:
            assert smiles_raw == fixed[raw["case_id"]]["old_smiles"] != smiles_new
            assert new["structure_verification"]["original_smiles"] == smiles_raw
        else:
            assert smiles_new == smiles_raw
        assert dataset.public_smiles(raw["case_id"]) == smiles_new


@pytest.mark.parametrize("case,formula", [("0020", "C18H8F6N2O2S2"), ("0141", "C19H16Fe"),
                                          ("0152", "C19H14"), ("0153", "C20H12")])
def test_corrected_formulas_match_the_papers(case, formula):
    assert CalcMolFormula(Chem.MolFromSmiles(dataset.public_smiles(P + case))) == formula


def test_verification_status_covers_every_case():
    counts = json.loads((dataset.DATASET / "manifest.json").read_text())["verification_counts"]
    assert counts == {"corrected": 4, "verified_consistent": 13, "unverified": 102}
    dev = {d["case_id"] for d in load_split()["dev"]}
    assert all(dataset.structure_verification(c)["status"] != "unverified" for c in dev)   # all 17 dev cases checked


def test_metallocene_case_is_not_runnable_until_its_geometry_can_be_prepared(tmp_path):
    ok, why = dataset.runnable(P + "0141")
    assert not ok and "sandwich" in why
    assert P + "0141" not in dev_cases() and P + "0141" in dev_cases(include_not_runnable=True)
    assert len(dev_cases()) == 16
    from mechcal.agent import cli
    with pytest.raises(SystemExit, match="cannot be run yet"):
        cli.main(["--case-id", P + "0141", "--out", str(tmp_path / "x")])


def test_batch_refuses_not_runnable_cases(tmp_path, monkeypatch):
    from mechcal.agent import batch
    monkeypatch.setenv("MECHCAL_OPENAI_API_KEY", "test")
    with pytest.raises(SystemExit, match="cannot be run yet"):
        batch.main(["--cases", "0141", "--out", str(tmp_path / "b")])
