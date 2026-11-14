"""Set-based metrics for the 7-family pool (docs/evaluation-metrics-zh.md). Evaluation side.

A system claims a set of families for a molecule (ordered: the first is its primary prediction),
optionally gives a probability for every family, and the judge gives each claimed family an
evidence score s in {0, 0.5, 1}. All functions are pure.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Callable, Sequence


@dataclass
class Row:
    """One run on one case."""
    case_id: str
    claimed: list[str]                         # ordered; first = primary prediction
    reference: set[str]
    primary: set[str]
    scores: dict[str, float] = field(default_factory=dict)    # judge evidence score per claimed family
    probs: dict[str, float] | None = None                      # probability per family (all families) or None


def es_f1(claimed: Sequence[str], reference: set[str], scores: dict[str, float]) -> float:
    """FEVER-style F1 for one molecule: a correct family counts only as far as its evidence score."""
    tp = sum(scores.get(f, 0.0) for f in set(claimed) & reference)
    return 2 * tp / (len(set(claimed)) + len(reference))


def label_f1(claimed: Sequence[str], reference: set[str]) -> float:
    return 2 * len(set(claimed) & reference) / (len(set(claimed)) + len(reference))


def top1_correct(claimed: Sequence[str], primary: set[str]) -> float | None:
    """1/0 whether the first claimed family is a primary reference family; None without a primary label."""
    if not primary:
        return None
    return float(bool(claimed) and claimed[0] in primary)


def brier(probs: dict[str, float], reference: set[str], pool: Sequence[str]) -> float:
    return mean((probs[f] - (f in reference)) ** 2 for f in pool)


def mcc(tp: float, fp: float, fn: float, tn: float) -> float:
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den else 0.0


def family_counts(rows: Sequence[Row], family: str) -> dict[str, int]:
    """TP, FP, FN, TN of one family over all rows."""
    c = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for r in rows:
        claimed, present = family in r.claimed, family in r.reference
        c["tp" if claimed and present else "fp" if claimed else "fn" if present else "tn"] += 1
    return c


def family_mcc(rows: Sequence[Row], pool: Sequence[str]) -> dict[str, float]:
    """MCC per family over all rows; families with no positive case in the rows are left out."""
    out = {}
    for f in pool:
        c = family_counts(rows, f)
        if c["tp"] + c["fn"]:
            out[f] = mcc(c["tp"], c["fp"], c["fn"], c["tn"])
    return out


def pass_k(groups: dict[str, list[Row]]) -> dict[str, float | None]:
    """Over cases run k >= 2 times: share whose primary prediction is right in every run (cases with a
    primary label), and share whose claimed set is identical in every run."""
    multi = {c: rs for c, rs in groups.items() if len(rs) >= 2}
    with_primary = [rs for rs in multi.values() if rs[0].primary]
    return {"cases": len(multi),
            "all_top1_correct": mean(all(top1_correct(r.claimed, r.primary) for r in rs) for rs in with_primary)
            if with_primary else None,
            "identical_sets": mean(len({frozenset(r.claimed) for r in rs}) == 1 for rs in multi.values())
            if multi else None}


def summarize(rows: Sequence[Row], pool: Sequence[str]) -> dict[str, float | None]:
    """All set metrics over rows. Per-molecule metrics are averaged per case first, then over cases."""
    by_case: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_case[r.case_id].append(r)

    def case_mean(fn):
        vals = [mean(v) for v in ([fn(r) for r in rs if fn(r) is not None] for rs in by_case.values()) if v]
        return mean(vals) if vals else None
    fam = family_mcc(rows, pool)
    return {
        "es_f1": case_mean(lambda r: es_f1(r.claimed, r.reference, r.scores)),
        "top1_primary": case_mean(lambda r: top1_correct(r.claimed, r.primary)),
        "macro_mcc": mean(fam.values()) if fam else None,
        "brier": case_mean(lambda r: brier(r.probs, r.reference, pool) if r.probs else None),
        "label_f1": case_mean(lambda r: label_f1(r.claimed, r.reference)),
    }


def cluster_bootstrap(rows: Sequence[Row], clusters: dict[str, str], stat: Callable[[Sequence[Row]], float | None],
                      n: int = 2000, seed: int = 0) -> tuple[float | None, float | None]:
    """95 % interval of ``stat`` from resampling paper clusters (all runs of a case move together)."""
    groups: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        groups[clusters.get(r.case_id, r.case_id)].append(r)
    keys, rng, vals = sorted(groups), random.Random(seed), []
    for _ in range(n):
        sample = [r for k in (rng.choice(keys) for _ in keys) for r in groups[k]]
        v = stat(sample)
        if v is not None:
            vals.append(v)
    if not vals:
        return None, None
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]
