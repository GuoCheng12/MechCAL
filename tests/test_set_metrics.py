"""Set metrics on hand-computed examples (docs/evaluation-metrics-zh.md)."""

import pytest

from mechcal.eval import set_metrics as sm

POOL = ("ICT", "RIM", "AGG")


def test_es_f1_and_label_f1_one_molecule():
    # reference {ICT, RIM}; claimed {ICT, AGG}; ICT evidence 1 -> TP* 1, F1* = 2*1/(2+2)
    assert sm.es_f1(["ICT", "AGG"], {"ICT", "RIM"}, {"ICT": 1.0, "AGG": 1.0}) == 0.5
    assert sm.es_f1(["ICT", "AGG"], {"ICT", "RIM"}, {"ICT": 0.5}) == 0.25
    assert sm.label_f1(["ICT", "AGG"], {"ICT", "RIM"}) == 0.5
    assert sm.es_f1(["ICT"], {"ICT"}, {}) == 0.0                  # right family, no evidence


def test_top1_and_brier():
    assert sm.top1_correct(["ICT", "RIM"], {"ICT"}) == 1.0
    assert sm.top1_correct(["RIM", "ICT"], {"ICT"}) == 0.0
    assert sm.top1_correct(["ICT"], set()) is None                # no primary label: not counted
    assert sm.brier({"ICT": 1.0, "RIM": 0.5, "AGG": 0.0}, {"ICT", "RIM"}, POOL) == pytest.approx(0.25 / 3)


def test_family_mcc_matches_the_worked_example():
    # six molecules, family ICT: TP 2 (A, B), FN 1 (C), FP 1 (D), TN 2 (E, F) -> MCC 3/9
    rows = [sm.Row("A", ["ICT"], {"ICT"}, set()), sm.Row("B", ["ICT"], {"ICT"}, set()),
            sm.Row("C", ["RIM"], {"ICT"}, set()), sm.Row("D", ["ICT"], {"RIM"}, set()),
            sm.Row("E", ["RIM"], {"RIM"}, set()), sm.Row("F", ["RIM"], {"RIM"}, set())]
    assert sm.family_counts(rows, "ICT") == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}
    assert sm.family_mcc(rows, POOL)["ICT"] == pytest.approx(1 / 3)
    always = [sm.Row(r.case_id, ["ICT"], r.reference, set()) for r in rows]
    assert sm.family_mcc(always, POOL)["ICT"] == 0.0               # a constant answer scores 0
    assert "AGG" not in sm.family_mcc(rows, POOL)                  # no positive case: left out


def test_summary_averages_per_case_and_pass_k():
    rows = [sm.Row("A", ["ICT"], {"ICT"}, {"ICT"}, {"ICT": 1.0}), sm.Row("A", ["RIM"], {"ICT"}, {"ICT"}, {}),
            sm.Row("B", ["RIM"], {"RIM"}, {"RIM"}, {"RIM": 0.5})]
    s = sm.summarize(rows, POOL)
    assert s["es_f1"] == pytest.approx(((1.0 + 0.0) / 2 + 0.5) / 2)       # case A averaged first
    assert s["top1_primary"] == pytest.approx((0.5 + 1.0) / 2) and s["brier"] is None
    groups = {"A": rows[:2], "B": rows[2:]}
    assert sm.pass_k(groups) == {"cases": 1, "all_top1_correct": 0.0, "identical_sets": 0.0}


def test_cluster_bootstrap_moves_cases_of_a_paper_together():
    rows = [sm.Row(c, ["ICT"], {"ICT"}, set()) for c in ("A", "B", "C")]
    lo, hi = sm.cluster_bootstrap(rows, {"A": "p1", "B": "p1", "C": "p2"},
                                  lambda rs: sm.summarize(rs, POOL)["label_f1"], n=200)
    assert lo == hi == 1.0
