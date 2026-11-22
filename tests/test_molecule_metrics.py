"""Per-molecule metrics on hand-computed examples (docs/evaluation-metrics-zh.md) and their floors on rebuilt_302."""

import json
import random
from statistics import mean

import pytest

from mechcal.eval import dataset
from mechcal.eval import molecule_metrics as mm
from mechcal.eval.splits import dev_cases, load_split, split_of
from mechcal.mechanism_pool import MECHANISM_POOL

REF = mm.Reference(frozenset({"ICT", "RIM"}), frozenset({"ICT"}), frozenset({"ESIPT"}))


def test_worked_example_of_the_metric_document():
    m = mm.molecule_metrics(["ICT", "AGG", "ESIPT"], REF, {"ICT": 1.0, "AGG": 0.5, "ESIPT": 0.0},
                            ["consistent"] * 5 + ["beyond_source"])
    assert m == {"es_rec": 0.5, "top1": 1.0, "evr": 1.0, "ep": pytest.approx(5 / 6), "rec": 0.5, "bal": 0.25}


def test_scope_of_each_metric():
    no_positive = mm.Reference(frozenset(), frozenset(), frozenset({"ESIPT"}))
    m = mm.molecule_metrics(["ICT"], no_positive)
    assert m["es_rec"] is None and m["rec"] is None and m["top1"] is None and m["bal"] is None and m["evr"] == 0.0
    no_exclusion = mm.Reference(frozenset({"ICT"}), frozenset(), frozenset())
    m = mm.molecule_metrics(["RIM", "ICT"], no_exclusion, {"ICT": 0.5})
    assert m["evr"] is None and m["bal"] is None and m["top1"] is None and m["ep"] is None
    assert m["es_rec"] == 0.5 and m["rec"] == 1.0
    assert mm.molecule_metrics(["RIM", "ICT"], REF)["top1"] == 0.0            # order matters: first claim is not primary
    assert mm.molecule_metrics([], REF) == {"es_rec": 0.0, "top1": 0.0, "evr": 0.0, "ep": None, "rec": 0.0, "bal": 0.5}


def test_repeats_are_averaged_within_a_molecule_first_and_units_resampled_together():
    rows = {"A": [mm.molecule_metrics(["ICT"], REF, {"ICT": 1.0}), mm.molecule_metrics(["RIM"], REF, {})],
            "B": [mm.molecule_metrics(["ICT", "RIM"], REF, {"ICT": 1.0, "RIM": 1.0})]}
    cases = mm.per_case(rows)
    assert cases["A"]["es_rec"] == 0.25 and cases["A"]["top1"] == 0.5 and "ep" not in cases["A"]
    assert mm.summarize(cases)["es_rec"] == pytest.approx((0.25 + 1.0) / 2) and mm.summarize(cases)["ep"] is None
    same = {c: {"rec": 1.0} for c in "ABC"}
    ci = mm.unit_bootstrap(same, {"A": "u1", "B": "u1", "C": "u2"}, n=200)
    assert ci["rec"] == (1.0, 1.0) and ci["evr"] is None


# ---- rebuilt_302: the data the metrics are computed on ---------------------------------------------

def references(case_ids):
    return {c: mm.reference_of(dataset.case_record(c)["hidden_reference"]) for c in case_ids}


def test_rebuilt_dataset_is_the_committed_snapshot_and_split():
    import hashlib
    manifest = json.loads((dataset.DATASET / "manifest.json").read_text())
    files = sorted(dataset.CASES_DIR.glob("*.json"))
    assert len(files) == manifest["case_count"] == 302
    for path in files:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["files_sha256"][f"cases/{path.name}"], path.name
    split = load_split()
    dev, test = {d["case_id"] for d in split["dev"]}, set(split["test"])
    assert len(dev) == 29 and len(test) == 273 and not dev & test and dev | test == {p.stem for p in files}
    assert split_of("EXPA_021") == "dev" and split_of(split["test"][0]) == "test" and split_of("custom") is None
    assert not {split["units"][c] for c in dev} & {split["units"][c] for c in test}      # paper units stay together
    assert len(dev_cases()) == 28 and "AIE_DDX_MECH_V4_0141" not in dev_cases()
    assert dataset.public_smiles("EXPA_021") == "Cn1cc(/C=C/c2ccc3ccccc3[n+]2C)c2ccccc21"
    with pytest.raises(KeyError):
        dataset.public_smiles("AIE_DDX_MECH_V4_0152")                                     # removed in the rebuild


def test_every_decision_is_in_the_pool_and_scopes_match_the_metric_document():
    split = load_split()
    refs = references(p.stem for p in dataset.CASES_DIR.glob("*.json"))
    for case, r in refs.items():
        assert r.supported | r.excluded <= set(MECHANISM_POOL) and not r.supported & r.excluded and r.primary <= r.supported, case
    def scope(ids):
        rs = [refs[c] for c in ids]
        return (sum(bool(r.supported) for r in rs), sum(bool(r.primary) for r in rs), sum(bool(r.excluded) for r in rs),
                sum(bool(r.supported and r.excluded) for r in rs))
    assert scope(d["case_id"] for d in split["dev"]) == (28, 28, 22, 21)
    assert scope(refs) == (288, 281, 222, 208)


def test_floors_on_the_development_set():
    refs = references(d["case_id"] for d in load_split()["dev"])
    def score(answer):
        return mm.summarize(mm.per_case({c: [mm.molecule_metrics(answer(), r)] for c, r in refs.items()}))
    everything = score(lambda: list(MECHANISM_POOL))
    assert (everything["rec"], everything["evr"], everything["bal"], everything["es_rec"]) == (1.0, 1.0, 0.5, 0.0)
    common = score(lambda: ["AGGREGATE_EXCITON_EXCIMER", "ICT_TICT_CT", "RADIATIVE_RATE_STATE_BALANCE"])
    assert (round(common["rec"], 2), round(common["evr"], 2), round(common["bal"], 2)) == (0.46, 0.02, 0.75)
    assert round(score(lambda: ["PET_ET"])["top1"], 2) == 0.21
    rng = random.Random(0)
    draws = [score(lambda: rng.sample(list(MECHANISM_POOL), 3)) for _ in range(300)]
    for metric, expected in (("rec", 3 / 7), ("evr", 3 / 7), ("bal", 0.5), ("top1", 0.17)):
        assert mean(d[metric] for d in draws) == pytest.approx(expected, abs=0.02), metric
