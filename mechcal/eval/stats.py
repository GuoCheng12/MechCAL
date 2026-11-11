"""Benchmark-level statistics for PhotoMechBench: per-family metrics and clustered bootstrap.

Cases that share a source paper are not independent, so resampling is done over
paper clusters (connected components of shared DOIs), never over individual cases.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from mechcal.eval.mechanism_ranking import score_mechanism_ranking
from mechcal.mechanism_pool import MECHANISM_POOL

PRIMARY_ROLES = {"primary", "co_primary"}


@dataclass(frozen=True)
class ReferenceCase:
    case_id: str
    smiles: str
    reference_mechanisms: tuple[dict, ...]
    dois: tuple[str, ...]

    @property
    def labels(self) -> set[str]:
        return {m["label"] for m in self.reference_mechanisms}

    @property
    def primary_labels(self) -> set[str]:
        return {m["label"] for m in self.reference_mechanisms if m.get("role") in PRIMARY_ROLES}


def load_reference_cases(cases_dir: Path) -> list[ReferenceCase]:
    """Load full raw-v2 case records (hidden references). Evaluation-side only."""
    cases = []
    for path in sorted(Path(cases_dir).glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        hidden = raw["hidden_reference"]
        cases.append(ReferenceCase(
            case_id=raw["case_id"],
            smiles=raw["public_input"]["molecule"]["structure"]["value"],
            reference_mechanisms=tuple(hidden["reference_mechanisms"]),
            dois=tuple(sorted({s["doi"].lower() for s in hidden["source_documents"] if s.get("doi")})),
        ))
    if not cases:
        raise ValueError(f"No case files under {cases_dir}")
    return cases


def paper_clusters(cases: Sequence[ReferenceCase]) -> dict[str, str]:
    """Map case_id -> cluster id; cases are joined when they share any DOI."""
    parent = {c.case_id: c.case_id for c in cases}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner: dict[str, str] = {}
    for case in cases:
        for doi in case.dois:
            if doi in owner:
                parent[find(case.case_id)] = find(owner[doi])
            else:
                owner[doi] = case.case_id
    return {c.case_id: find(c.case_id) for c in cases}


def as_predictions(labels: Sequence[str]) -> list[dict]:
    return [{"label": label, "rank": i} for i, label in enumerate(labels, start=1)]


def score_cases(
    ranked_labels: Mapping[str, Sequence[str]],
    cases: Sequence[ReferenceCase],
    *,
    support_scores: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, dict]:
    """Per-case metric rows. A case missing from ``ranked_labels`` scores as an empty answer."""
    rows = {}
    for case in cases:
        labels = list(ranked_labels.get(case.case_id, []))
        scored = score_mechanism_ranking(
            as_predictions(labels) if labels else [],
            list(case.reference_mechanisms),
            support_scores=dict(support_scores[case.case_id]) if support_scores else None,
        ) if labels else _empty_row(support_scores is not None)
        row = {k: v for k, v in scored.items() if k != "details"}
        row["predicted"] = labels[:3]
        if support_scores is not None and labels:
            e1 = scored["details"]["top1_support_score"]
            row["sufficient_at_top1"] = 1.0 if e1 is not None and e1 >= 1.0 else 0.0
        rows[case.case_id] = row
    return rows


def _empty_row(with_support: bool) -> dict:
    row = {"top1_primary": 0.0, "top1_any": 0.0, "recall_at_3": 0.0, "ndcg_at_3": 0.0}
    if with_support:
        row.update(es_ndcg_at_3=0.0, support_at_top1=0.0, sufficient_at_top1=0.0)
    return row


def family_metrics(
    ranked_labels: Mapping[str, Sequence[str]], cases: Sequence[ReferenceCase], *, k: int = 3,
) -> dict:
    """Per-family recall@k, precision@k and top-1-primary recall, plus macro averages.

    recall@k(f): among cases whose reference contains f, fraction with f in the top k.
    precision@k(f): among top-k predictions of f, fraction that are in the reference.
    primary_top1(f): among cases where f is primary/co-primary, fraction ranked first.
    """
    stats = {f: defaultdict(int) for f in MECHANISM_POOL}
    for case in cases:
        top = list(ranked_labels.get(case.case_id, []))[:k]
        for f in case.labels:
            stats[f]["ref"] += 1
            stats[f]["hit"] += f in top
        for f in case.primary_labels:
            stats[f]["primary"] += 1
            stats[f]["primary_top1"] += bool(top) and top[0] == f
        for f in top:
            if f in stats:
                stats[f]["pred"] += 1
                stats[f]["pred_correct"] += f in case.labels
    families = {}
    for f, s in stats.items():
        families[f] = {
            "reference_count": s["ref"],
            "prediction_count": s["pred"],
            f"recall_at_{k}": s["hit"] / s["ref"] if s["ref"] else None,
            f"precision_at_{k}": s["pred_correct"] / s["pred"] if s["pred"] else None,
            "primary_count": s["primary"],
            "primary_top1_recall": s["primary_top1"] / s["primary"] if s["primary"] else None,
        }

    def macro(key: str) -> float | None:
        values = [v[key] for v in families.values() if v[key] is not None]
        return sum(values) / len(values) if values else None

    return {
        "families": families,
        f"macro_recall_at_{k}": macro(f"recall_at_{k}"),
        # Families never predicted have undefined precision; macro precision treats them as 0
        # so that a method cannot inflate it by never predicting rare families.
        f"macro_precision_at_{k}": sum(
            (v[f"precision_at_{k}"] or 0.0) for v in families.values() if v["reference_count"]
        ) / max(1, sum(1 for v in families.values() if v["reference_count"])),
        "macro_primary_top1_recall": macro("primary_top1_recall"),
    }


def _cluster_index(case_ids: Sequence[str], clusters: Mapping[str, str]) -> list[list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, cid in enumerate(case_ids):
        groups[clusters.get(cid, cid)].append(i)
    return list(groups.values())


def clustered_bootstrap(
    values: Mapping[str, float], clusters: Mapping[str, str], *,
    n_resamples: int = 10_000, alpha: float = 0.05, seed: int = 0,
) -> dict:
    """Mean with a percentile CI from resampling paper clusters with replacement."""
    case_ids = sorted(values)
    data = [float(values[c]) for c in case_ids]
    groups = _cluster_index(case_ids, clusters)
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        total = count = 0
        for _ in range(len(groups)):
            g = groups[rng.randrange(len(groups))]
            total += sum(data[i] for i in g)
            count += len(g)
        means.append(total / count)
    means.sort()
    return {
        "mean": sum(data) / len(data),
        "ci_low": means[int(alpha / 2 * n_resamples)],
        "ci_high": means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))],
        "n_cases": len(data), "n_clusters": len(groups), "n_resamples": n_resamples,
    }


def paired_clustered_bootstrap(
    a: Mapping[str, float], b: Mapping[str, float], clusters: Mapping[str, str], *,
    n_resamples: int = 10_000, alpha: float = 0.05, seed: int = 0,
) -> dict:
    """Difference mean(a) - mean(b) on the same cases, resampling paper clusters.

    ``p_two_sided`` is the bootstrap estimate 2*min(P(diff<=0), P(diff>=0)), clipped to 1.
    """
    if set(a) != set(b):
        raise ValueError("Paired comparison requires the same case set")
    diffs = {c: float(a[c]) - float(b[c]) for c in a}
    boot = clustered_bootstrap(diffs, clusters, n_resamples=n_resamples, alpha=alpha, seed=seed)
    case_ids = sorted(diffs)
    groups = _cluster_index(case_ids, clusters)
    data = [diffs[c] for c in case_ids]
    rng = random.Random(seed + 1)
    le = ge = 0
    for _ in range(n_resamples):
        total = count = 0
        for _ in range(len(groups)):
            g = groups[rng.randrange(len(groups))]
            total += sum(data[i] for i in g)
            count += len(g)
        m = total / count
        le += m <= 0
        ge += m >= 0
    return {**boot, "p_two_sided": min(1.0, 2 * min(le, ge) / n_resamples)}


def metric_column(rows: Mapping[str, Mapping[str, float]], key: str) -> dict[str, float]:
    return {cid: float(row[key]) for cid, row in rows.items() if row.get(key) is not None}


def summarize(
    rows: Mapping[str, Mapping[str, float]], clusters: Mapping[str, str],
    keys: Iterable[str] = ("top1_primary", "recall_at_3", "ndcg_at_3"), **bootstrap_kwargs,
) -> dict:
    return {key: clustered_bootstrap(metric_column(rows, key), clusters, **bootstrap_kwargs)
            for key in keys if metric_column(rows, key)}
