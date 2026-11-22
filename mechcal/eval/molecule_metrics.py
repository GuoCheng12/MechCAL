"""Per-molecule metrics for the 7-family pool (docs/evaluation-metrics-zh.md). Evaluation side.

Every metric is computed for one molecule from its ordered claimed families, the judge's evidence
score per claimed family, the judge's verdict per evidence statement and the reference (supported
S, primary P within S, explicitly excluded X). A dataset value is the mean over molecules; repeated
runs of a molecule are averaged first. All functions are pure.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Mapping, Sequence

METRICS = ("es_rec", "top1", "evr", "ep", "rec", "bal")
PRIMARY_ROLES = {"primary", "co_primary"}


@dataclass(frozen=True)
class Reference:
    supported: frozenset[str]      # S: reported by the source
    primary: frozenset[str]        # P: primary or co-primary families of S
    excluded: frozenset[str]       # X: explicitly excluded by the source


def reference_of(hidden_reference: dict) -> Reference:
    decisions = hidden_reference["mechanism_decisions"]
    supported = [d for d in decisions if d["status"] == "supported"]
    return Reference(frozenset(d["label"] for d in supported),
                     frozenset(d["label"] for d in supported if d.get("role") in PRIMARY_ROLES),
                     frozenset(d["label"] for d in decisions if d["status"] == "contradicted"))


def molecule_metrics(claimed: Sequence[str], ref: Reference, scores: Mapping[str, float] | None = None,
                     verdicts: Sequence[str] = ()) -> dict[str, float | None]:
    """The six metrics for one run on one molecule; None where the molecule is outside a metric's scope.

    ``scores``: evidence score (0, 0.5, 1) per claimed family; ``verdicts``: the judge's outcome for
    every evidence statement of the submission ("consistent" or not).
    """
    c, s, x = set(claimed), ref.supported, ref.excluded
    scores = scores or {}
    rec = len(c & s) / len(s) if s else None
    evr = len(c & x) / len(x) if x else None
    return {
        "es_rec": sum(scores.get(f, 0.0) for f in c & s) / len(s) if s else None,
        "top1": float(bool(claimed) and claimed[0] in ref.primary) if ref.primary else None,
        "evr": evr,
        "ep": sum(v == "consistent" for v in verdicts) / len(verdicts) if verdicts else None,
        "rec": rec,
        "bal": (rec + 1 - evr) / 2 if s and x else None,
    }


def per_case(rows: Mapping[str, Sequence[Mapping[str, float | None]]]) -> dict[str, dict[str, float]]:
    """case -> metric -> mean over that case's runs (metrics out of scope for the case are left out)."""
    out = {}
    for case, runs in rows.items():
        out[case] = {m: mean(v) for m in METRICS if (v := [r[m] for r in runs if r.get(m) is not None])}
    return out


def summarize(case_values: Mapping[str, Mapping[str, float]]) -> dict[str, float | None]:
    """Mean over molecules of each metric, over the molecules in its scope."""
    return {m: mean(v) if (v := [c[m] for c in case_values.values() if m in c]) else None for m in METRICS}


def unit_bootstrap(case_values: Mapping[str, Mapping[str, float]], units: Mapping[str, str], *, n: int = 2000,
                   seed: int = 0) -> dict[str, tuple[float, float] | None]:
    """95 % interval of every metric from resampling paper units (cases of a unit move together)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for case in case_values:
        groups[units.get(case, case)].append(case)
    keys, rng = sorted(groups), random.Random(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(n):
        sample = [case_values[c] for k in (rng.choice(keys) for _ in keys) for c in groups[k]]
        for m in METRICS:
            if v := [c[m] for c in sample if m in c]:
                draws[m].append(mean(v))
    out = {}
    for m in METRICS:
        v = sorted(draws[m])
        out[m] = (v[int(0.025 * len(v))], v[min(len(v) - 1, int(0.975 * len(v)))]) if v else None
    return out
