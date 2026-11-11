"""Benchmark statistics and dataset analyses on synthetic references (no API)."""

from pathlib import Path

import pytest

from mechcal.eval import dataset_analysis as da
from mechcal.eval.stats import (
    ReferenceCase, clustered_bootstrap, family_metrics, load_reference_cases,
    paired_clustered_bootstrap, paper_clusters, score_cases, summarize,
)

ROOT = Path(__file__).resolve().parents[1]


def ref(label, role="primary"):
    return {"label": label, "role": role, "gain": 2 if role in {"primary", "co_primary"} else 1}


def case(cid, smiles, refs, dois):
    return ReferenceCase(cid, smiles, tuple(refs), tuple(dois))


@pytest.fixture
def cases():
    return [
        case("A", "c1ccccc1O", [ref("RIM_RIR_RIV"), ref("ICT_TICT_CT", "secondary")], ["d1"]),
        case("B", "c1ccccc1N", [ref("RIM_RIR_RIV")], ["d1", "d2"]),
        case("C", "CCO", [ref("ESIPT_PT")], ["d2"]),
        case("D", "CCN", [ref("PET_ET"), ref("ICT_TICT_CT", "co_primary")], ["d3"]),
    ]


def test_paper_clusters_follow_shared_dois_transitively(cases):
    clusters = paper_clusters(cases)
    assert clusters["A"] == clusters["B"] == clusters["C"]
    assert clusters["D"] != clusters["A"]


def test_family_metrics_and_macro(cases):
    ranking = {"A": ["RIM_RIR_RIV", "PET_ET", "ESIPT_PT"], "B": ["ICT_TICT_CT", "RIM_RIR_RIV", "PET_ET"],
               "C": ["ESIPT_PT", "RIM_RIR_RIV", "PET_ET"], "D": ["ICT_TICT_CT", "RIM_RIR_RIV", "PET_ET"]}
    out = family_metrics(ranking, cases)
    f = out["families"]
    assert f["RIM_RIR_RIV"]["recall_at_3"] == 1.0
    assert f["ICT_TICT_CT"]["recall_at_3"] == 0.5          # missed in A, hit in D
    assert f["PET_ET"]["precision_at_3"] == 0.25           # predicted 4x, correct only in D
    assert f["RIM_RIR_RIV"]["primary_top1_recall"] == 0.5  # first in A, second in B
    assert f["ICT_TICT_CT"]["primary_top1_recall"] == 1.0  # co-primary in D, ranked first
    assert f["AGGREGATE_EXCITON_EXCIMER"]["recall_at_3"] is None
    recalls = [v["recall_at_3"] for v in f.values() if v["recall_at_3"] is not None]
    assert out["macro_recall_at_3"] == pytest.approx(sum(recalls) / len(recalls))


def test_missing_answer_scores_zero(cases):
    rows = score_cases({"A": ["RIM_RIR_RIV", "ICT_TICT_CT", "PET_ET"]}, cases)
    assert rows["A"]["recall_at_3"] == 1.0 and rows["A"]["top1_primary"] == 1.0
    assert rows["B"]["recall_at_3"] == 0.0 and rows["B"]["predicted"] == []


def test_sufficient_at_top1_separates_partial_from_full(cases):
    ranking = {c.case_id: ["RIM_RIR_RIV", "ICT_TICT_CT", "PET_ET"] for c in cases}
    support = {c.case_id: {"RIM_RIR_RIV": 0.5, "ICT_TICT_CT": 1.0, "PET_ET": 0.0} for c in cases}
    rows = score_cases(ranking, cases, support_scores=support)
    assert rows["A"]["support_at_top1"] == 1.0
    assert rows["A"]["sufficient_at_top1"] == 0.0


def test_clustered_bootstrap_is_deterministic_and_brackets_mean(cases):
    clusters = paper_clusters(cases)
    values = {"A": 1.0, "B": 0.0, "C": 1.0, "D": 0.0}
    first = clustered_bootstrap(values, clusters, n_resamples=500, seed=3)
    assert first == clustered_bootstrap(values, clusters, n_resamples=500, seed=3)
    assert first["n_clusters"] == 2 and first["mean"] == 0.5
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]


def test_constant_cluster_has_zero_width_interval(cases):
    out = clustered_bootstrap({c.case_id: 0.7 for c in cases}, paper_clusters(cases), n_resamples=200)
    assert out["ci_low"] == pytest.approx(0.7) == out["ci_high"]


def test_paired_bootstrap_detects_consistent_gain_and_rejects_mismatch(cases):
    clusters = {c.case_id: c.case_id for c in cases}  # independent cases
    a = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0}
    b = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
    out = paired_clustered_bootstrap(a, b, clusters, n_resamples=500)
    assert out["mean"] == 1.0 and out["p_two_sided"] == 0.0
    same = paired_clustered_bootstrap(a, a, clusters, n_resamples=200)
    assert same["p_two_sided"] == 1.0
    with pytest.raises(ValueError):
        paired_clustered_bootstrap(a, {"A": 1.0}, clusters)


def test_lopo_never_trains_on_the_same_paper(cases):
    clusters = paper_clusters(cases)
    seen = {}

    def spy(train, test):
        seen[test.case_id] = {t.case_id for t in train}
        return ["RIM_RIR_RIV", "ICT_TICT_CT", "PET_ET"]

    da.leave_one_paper_out(cases, spy)
    for cid, train in seen.items():
        assert all(clusters[t] != clusters[cid] for t in train)
    assert seen["A"] == {"D"}


def test_best_constant_matches_full_scorer(cases):
    perm = da.best_constant_ranking(cases)
    rows = score_cases({c.case_id: perm for c in cases}, cases)
    best = sum(r["ndcg_at_3"] for r in rows.values())
    import itertools
    from mechcal.mechanism_pool import MECHANISM_POOL
    for other in itertools.islice(itertools.permutations(MECHANISM_POOL, 3), 200):
        rows = score_cases({c.case_id: other for c in cases}, cases)
        assert sum(r["ndcg_at_3"] for r in rows.values()) <= best + 1e-9


def test_near_duplicates_and_knn_on_similar_series():
    series = [
        case("S1", "c1ccc(cc1)C=Cc1ccc(N(C)C)cc1", [ref("ICT_TICT_CT")], ["p1"]),
        case("S2", "c1ccc(cc1)C=Cc1ccc(N(CC)CC)cc1", [ref("ICT_TICT_CT")], ["p2"]),
        case("S3", "OC(=O)CCCCCCCC", [ref("HOST_GUEST_INTERACTION")], ["p3"]),
    ]
    sims = da.similarity_matrix(series)
    pairs = da.near_duplicate_pairs(series, sims, threshold=0.4)
    assert pairs and {pairs[0]["case_a"], pairs[0]["case_b"]} == {"S1", "S2"}
    assert pairs[0]["same_paper_cluster"] is False and pairs[0]["same_primary_set"]
    preds = da.leave_one_paper_out(series, da.knn_predictor(sims, k=1))
    assert preds["S1"][0] == "ICT_TICT_CT"


def test_real_benchmark_loads_and_summarizes():
    cases = load_reference_cases(ROOT / "benchmarks/photomechbench/raw_v2_119/cases")
    assert len(cases) == 119
    clusters = paper_clusters(cases)
    assert len(set(clusters.values())) < len(cases)
    rows = score_cases({c.case_id: da.frequency_ranking(cases) for c in cases}, cases)
    out = summarize(rows, clusters, n_resamples=200)
    assert 0 < out["recall_at_3"]["mean"] < 1
    assert out["recall_at_3"]["n_clusters"] == len(set(clusters.values()))
