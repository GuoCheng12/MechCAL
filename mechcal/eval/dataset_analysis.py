"""Offline PhotoMechBench analyses that need no model API (R1 rebuttal Q1, Q6).

- series / near-duplicate structure pairs and their label agreement
- SMILES-blind floors (constant, frequency) and a structure-only nearest-neighbour floor
  evaluated leave-one-paper-out, so no label of the same source paper is ever used.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from typing import Mapping, Sequence

from mechcal.eval.stats import (
    ReferenceCase, clustered_bootstrap, family_metrics, metric_column, paper_clusters, score_cases,
)
from mechcal.mechanism_pool import MECHANISM_POOL


def _fingerprints(cases: Sequence[ReferenceCase]):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = {}
    for case in cases:
        mol = Chem.MolFromSmiles(case.smiles)
        if mol is None:
            raise ValueError(f"Unparseable SMILES in {case.case_id}")
        fps[case.case_id] = generator.GetFingerprint(mol)
    return fps


def similarity_matrix(cases: Sequence[ReferenceCase]) -> dict[tuple[str, str], float]:
    from rdkit import DataStructs

    fps = _fingerprints(cases)
    ids = [c.case_id for c in cases]
    sims = {}
    for i, a in enumerate(ids):
        values = DataStructs.BulkTanimotoSimilarity(fps[a], [fps[b] for b in ids[i + 1:]])
        for b, s in zip(ids[i + 1:], values):
            sims[(a, b)] = sims[(b, a)] = s
    return sims


def _jaccard(x: set, y: set) -> float:
    return len(x & y) / len(x | y) if x | y else 1.0


def near_duplicate_pairs(
    cases: Sequence[ReferenceCase], sims: Mapping[tuple[str, str], float], *, threshold: float = 0.7,
) -> list[dict]:
    """Structurally similar pairs with their label overlap; sorted by similarity."""
    by_id = {c.case_id: c for c in cases}
    clusters = paper_clusters(cases)
    pairs = []
    for a, b in itertools.combinations(sorted(by_id), 2):
        s = sims[(a, b)]
        if s < threshold:
            continue
        ca, cb = by_id[a], by_id[b]
        pairs.append({
            "case_a": a, "case_b": b, "tanimoto": round(s, 3),
            "same_paper_cluster": clusters[a] == clusters[b],
            "label_jaccard": round(_jaccard(ca.labels, cb.labels), 3),
            "same_primary_set": ca.primary_labels == cb.primary_labels,
            "labels_a": sorted(ca.labels), "labels_b": sorted(cb.labels),
            "primary_a": sorted(ca.primary_labels), "primary_b": sorted(cb.primary_labels),
        })
    return sorted(pairs, key=lambda p: -p["tanimoto"])


def within_paper_agreement(cases: Sequence[ReferenceCase]) -> dict:
    """How often cases from the same paper cluster carry identical label sets."""
    clusters = paper_clusters(cases)
    groups = defaultdict(list)
    for c in cases:
        groups[clusters[c.case_id]].append(c)
    multi = [g for g in groups.values() if len(g) > 1]
    pair_j, identical = [], 0
    for g in multi:
        for x, y in itertools.combinations(g, 2):
            j = _jaccard(x.labels, y.labels)
            pair_j.append(j)
            identical += j == 1.0
    return {
        "clusters": len(groups), "multi_case_clusters": len(multi),
        "cases_in_multi_case_clusters": sum(len(g) for g in multi),
        "largest_cluster": max(len(g) for g in groups.values()),
        "within_cluster_pairs": len(pair_j),
        "within_cluster_identical_label_sets": identical,
        "within_cluster_mean_label_jaccard": sum(pair_j) / len(pair_j) if pair_j else None,
    }


def _gain_table(cases: Sequence[ReferenceCase]) -> Counter:
    table = Counter()
    for c in cases:
        for m in c.reference_mechanisms:
            table[m["label"]] += float(m.get("gain", 1))
    return table


def frequency_ranking(cases: Sequence[ReferenceCase]) -> list[str]:
    table = _gain_table(cases)
    return sorted(MECHANISM_POOL, key=lambda f: (-table[f], f))[:3]


_DISCOUNT = (1.0, 1 / math.log2(3), 0.5)


def _case_gains(case: ReferenceCase) -> tuple[dict[str, float], float]:
    gains: dict[str, float] = {}
    for m in case.reference_mechanisms:
        gains[m["label"]] = max(gains.get(m["label"], 0.0), float(m.get("gain", 1)))
    ideal = sorted(gains.values(), reverse=True)[:3]
    return gains, sum(g * d for g, d in zip(ideal, _DISCOUNT))


def best_constant_ranking(cases: Sequence[ReferenceCase]) -> list[str]:
    """Best fixed top-3 permutation by mean nDCG@3 on ``cases`` (in-sample oracle).

    Uses the same gain/discount definition as ``score_mechanism_ranking``; ties break
    lexicographically for determinism.
    """
    tables = [_case_gains(c) for c in cases]
    best, best_value = None, -1.0
    for perm in itertools.permutations(MECHANISM_POOL, 3):
        value = sum(
            sum(gains.get(f, 0.0) * d for f, d in zip(perm, _DISCOUNT)) / idcg
            for gains, idcg in tables if idcg
        )
        if value > best_value + 1e-12:
            best, best_value = list(perm), value
    return best


def leave_one_paper_out(cases: Sequence[ReferenceCase], predictor) -> dict[str, list[str]]:
    """Run ``predictor(train_cases, test_case) -> top3`` with the test case's paper cluster held out."""
    clusters = paper_clusters(cases)
    return {
        c.case_id: list(predictor([t for t in cases if clusters[t.case_id] != clusters[c.case_id]], c))
        for c in cases
    }


def knn_predictor(sims: Mapping[tuple[str, str], float], *, k: int = 5):
    """Similarity-weighted label transfer from the k most similar training molecules."""

    def predict(train: Sequence[ReferenceCase], case: ReferenceCase) -> list[str]:
        neighbours = sorted(train, key=lambda t: -sims[(case.case_id, t.case_id)])[:k]
        votes = Counter()
        for t in neighbours:
            w = sims[(case.case_id, t.case_id)]
            for m in t.reference_mechanisms:
                votes[m["label"]] += w * float(m.get("gain", 1))
        prior = _gain_table(train)
        return sorted(MECHANISM_POOL, key=lambda f: (-votes[f], -prior[f], f))[:3]

    return predict


def floors(cases: Sequence[ReferenceCase], sims=None, *, n_resamples: int = 2000) -> dict:
    """SMILES-blind and structure-only baselines, each with paper-clustered 95% CIs."""
    sims = sims or similarity_matrix(cases)
    clusters = paper_clusters(cases)
    methods = {
        "constant_in_sample_oracle": {c.case_id: best_constant_ranking(cases) for c in cases},
        "constant_lopo": leave_one_paper_out(cases, lambda train, _: best_constant_ranking(train)),
        "frequency_lopo": leave_one_paper_out(cases, lambda train, _: frequency_ranking(train)),
        "knn5_lopo": leave_one_paper_out(cases, knn_predictor(sims, k=5)),
        "knn1_lopo": leave_one_paper_out(cases, knn_predictor(sims, k=1)),
    }
    out = {}
    for name, ranking in methods.items():
        rows = score_cases(ranking, cases)
        out[name] = {
            **{key: clustered_bootstrap(metric_column(rows, key), clusters, n_resamples=n_resamples)
               for key in ("top1_primary", "recall_at_3", "ndcg_at_3")},
            **{k: v for k, v in family_metrics(ranking, cases).items() if k.startswith("macro")},
        }
    return out


def label_distribution(cases: Sequence[ReferenceCase]) -> dict:
    any_role, primary = Counter(), Counter()
    for c in cases:
        any_role.update(c.labels)
        primary.update(c.primary_labels)
    return {f: {"cases": any_role[f], "primary_cases": primary[f]} for f in MECHANISM_POOL}
