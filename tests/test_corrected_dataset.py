"""Structure-corrected copy of raw_v2_119: reproducible, only the listed SMILES change, readers use it."""

import importlib.util
import json
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from mechcal.eval import dataset
from mechcal.eval.family_reduction import reduce_hidden_reference
from mechcal.mechanism_pool import ELEVEN_FAMILY_POOL, MECHANISM_POOL
from mechcal.eval.splits import dev_cases, load_split

ROOT = Path(__file__).resolve().parents[1]
P = "AIE_DDX_MECH_V4_"
CORRECTED = dataset.CORRECTED_119


def corrected_smiles(case_id: str) -> str:
    import csv
    with (CORRECTED / "inputs.csv").open(encoding="utf-8", newline="") as fh:
        return {r["case_id"]: r["smiles"] for r in csv.DictReader(fh)}[case_id]


def builder():
    spec = importlib.util.spec_from_file_location("make_corrected", ROOT / "examples/make_corrected_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_copy_equals_a_fresh_build():
    files = builder().build()
    for path, text in files.items():
        assert (CORRECTED / path).read_text(encoding="utf-8") == text, path
    on_disk = {str(p.relative_to(CORRECTED)) for p in CORRECTED.rglob("*") if p.is_file()}
    assert on_disk == set(files)


def test_only_listed_smiles_change_and_references_are_mapped_not_edited():
    corrections = json.loads(dataset.CORRECTIONS.read_text())
    fixed = {c["case_id"]: c for c in corrections["corrections"]}
    dev = {P + c for c in ("0020", "0141", "0152", "0153")}
    test = {P + c for c in ("0001 0002 0007 0008 0011 0012 0018 0025 0026 0027 0039 0048 0049 0051 0075 0088 "
                            "0102 0125").split()}
    assert set(fixed) == dev | test
    assert all(fixed[c]["split"] == ("dev" if c in dev else "test") for c in fixed)
    for raw_path in sorted((dataset.RAW / "cases").glob("*.json")):
        raw = json.loads(raw_path.read_text())
        new = json.loads((CORRECTED / "cases" / raw_path.name).read_text())
        assert new["provenance"] == raw["provenance"]
        assert new["hidden_reference"] == reduce_hidden_reference(raw["hidden_reference"])
        kept = {k: v for k, v in new["hidden_reference"].items()
                if k not in ("reference_mechanisms", "mechanism_decisions", "family_reduction")}
        assert kept == {k: v for k, v in raw["hidden_reference"].items()
                        if k not in ("reference_mechanisms", "mechanism_decisions")}
        assert new["hidden_reference"]["family_reduction"]["original_reference_mechanisms"] == \
            raw["hidden_reference"]["reference_mechanisms"]
        smiles_raw = raw["public_input"]["molecule"]["structure"]["value"]
        smiles_new = new["public_input"]["molecule"]["structure"]["value"]
        if raw["case_id"] in fixed:
            assert smiles_raw == fixed[raw["case_id"]]["old_smiles"] != smiles_new
            assert new["structure_verification"]["original_smiles"] == smiles_raw
        else:
            assert smiles_new == smiles_raw
        assert corrected_smiles(raw["case_id"]) == smiles_new


@pytest.mark.parametrize("case,formula", [("0020", "C18H8F6N2O2S2"), ("0141", "C19H16Fe"),
                                          ("0152", "C19H14"), ("0153", "C20H12")])
def test_corrected_formulas_match_the_papers(case, formula):
    assert CalcMolFormula(Chem.MolFromSmiles(corrected_smiles(P + case))) == formula


def test_verification_status_covers_every_case():
    counts = json.loads((CORRECTED / "manifest.json").read_text())["verification_counts"]
    assert counts == {"corrected": 22, "verified_consistent": 97, "unverified": 0}           # every case checked
    corrections = json.loads(dataset.CORRECTIONS.read_text())
    assert corrections["verification"]["labels_read"] is False
    assert corrections["verification"]["agent_runs_on_test_cases"] is False


def test_metallocene_case_is_not_runnable_until_its_geometry_can_be_prepared(tmp_path):
    ok, why = dataset.runnable(P + "0141")
    assert not ok and "sandwich" in why
    assert P + "0141" not in dev_cases() and P + "0141" in dev_cases(include_not_runnable=True)
    assert len(dev_cases()) == 28                     # rebuilt_302: 29 development cases, one not runnable
    from mechcal.agent import cli
    with pytest.raises(SystemExit, match="cannot be run yet"):
        cli.main(["--case-id", P + "0141", "--out", str(tmp_path / "x")])


def test_batch_refuses_not_runnable_cases(tmp_path, monkeypatch):
    from mechcal.agent import batch
    monkeypatch.setenv("MECHCAL_OPENAI_API_KEY", "test")
    with pytest.raises(SystemExit, match="cannot be run yet"):
        batch.main(["--cases", "0141", "--out", str(tmp_path / "b")])


def test_structure_checks_flag_the_patterns_that_were_systematically_wrong():
    from mechcal.eval.structure_checks import CHRYSENE, PYREN_2_YL, structure_warnings
    assert structure_warnings("Cc1cc2ccc3cccc4ccc(c1)c2c34") == [PYREN_2_YL]               # 2-methylpyrene
    for ok in ("Cc1ccc2ccc3cccc4ccc1c2c34", "Cc1cc2cccc3ccc4cccc1c4c32",                  # 1- and 4-methylpyrene
               "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34", "Cc1ccc2ccc3cc4ccccc4cc3c2c1"):        # benzo[a]pyrene, Me-benz[a]anthracene
        assert structure_warnings(ok) == [], ok
    assert structure_warnings("c1ccc2c(c1)ccc1c3ccccc3ccc21") == [CHRYSENE]               # chrysene
    assert structure_warnings("c1ccc2c(c1)ccc1c2ccc2c3ccccc3ccc21") == [CHRYSENE]         # picene


def test_original_records_are_flagged_exactly_where_they_were_corrected():
    import csv
    from mechcal.eval.structure_checks import structure_warnings
    with (dataset.RAW / "inputs.csv").open(encoding="utf-8", newline="") as fh:
        raw = {r["case_id"]: r["smiles"] for r in csv.DictReader(fh)}
    flagged = {c for c, s in raw.items() if structure_warnings(s)}
    fixed = {c["case_id"] for c in json.loads(dataset.CORRECTIONS.read_text())["corrections"]}
    assert flagged == {P + c for c in ("0001", "0002", "0007", "0011", "0152", "0153")} and flagged <= fixed
    assert not any(structure_warnings(corrected_smiles(c)) for c in raw)                # none left


def test_family_mapping_rules():
    def mech(label, role, gain, ev):
        return {"label": label, "role": role, "gain": gain, "evidence_ids": ev, "rationale": label}

    def dec(label, status, role=None):
        return {"label": label, "status": status, "role": role, "evidence_ids": [], "rationale": label}

    hidden = {"reference_mechanisms": [mech("RIM_RIR_RIV", "secondary", 1, ["E1"]),
                                       mech("RACI_CI_ACCESS", "primary", 2, ["E2"]),
                                       mech("PACKING_HOST_MATRIX_CONFINEMENT", "co_primary", 2, ["E3"]),
                                       mech("TRIPLET_METAL_ENERGY_TRANSFER", "secondary", 1, ["E4"]),
                                       mech("SOKR_ANTI_KASHA", "secondary", 1, ["E5"])],
              "mechanism_decisions": [dec("RIM_RIR_RIV", "not_established"), dec("RACI_CI_ACCESS", "supported", "primary"),
                                      dec("PACKING_HOST_MATRIX_CONFINEMENT", "supported", "co_primary"),
                                      dec("HOST_GUEST_INTERACTION", "not_established")]}
    out = reduce_hidden_reference(hidden)
    assert [(m["label"], m["role"], m["gain"], m["evidence_ids"]) for m in out["reference_mechanisms"]] == [
        ("RIM_RIR_RIV", "primary", 2, ["E1", "E2"]), ("TRIPLET_ISC_RTP_TADF", "secondary", 1, ["E4"])]
    assert [(d["label"], d["status"], d["role"]) for d in out["mechanism_decisions"]] == [
        ("RIM_RIR_RIV", "supported", "primary")]
    r = out["family_reduction"]
    assert [m["label"] for m in r["environment_context"]] == ["PACKING_HOST_MATRIX_CONFINEMENT"]
    assert [m["label"] for m in r["out_of_pool_labels"]] == ["SOKR_ANTI_KASHA"]
    assert r["original_reference_mechanisms"] == hidden["reference_mechanisms"] and not r["primary_lost"]


def test_every_mapped_label_is_in_the_seven_family_pool():
    assert len(MECHANISM_POOL) == 7 and len(ELEVEN_FAMILY_POOL) == 11
    for path in sorted((CORRECTED / "cases").glob("*.json")):
        hidden = json.loads(path.read_text())["hidden_reference"]
        assert {m["label"] for m in hidden["reference_mechanisms"]} <= set(MECHANISM_POOL)
        assert {d["label"] for d in hidden["mechanism_decisions"]} <= set(MECHANISM_POOL)
