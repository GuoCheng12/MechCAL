"""Compare methods on the same cases with paper-clustered bootstrap CIs (evaluation side).

    python examples/compare_methods.py mechcal_c1=outputs/agent/dev4a,outputs/agent/dev4b \
        direct=outputs/baselines/direct_dev,outputs/baselines/direct_dev_b --metrics ndcg_at_3 es_ndcg_at_3

Each method is one or more directories of graded runs (repeats). Per case, a method's value is
its mean over the repeats that have that case. Every pair of methods is compared on the cases
both have: mean difference, 95 % CI and two-sided p from resampling paper clusters
(mechcal.eval.stats.paired_clustered_bootstrap). Prints a JSON report.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from statistics import mean

from mechcal.eval.stats import clustered_bootstrap, load_reference_cases, paired_clustered_bootstrap, paper_clusters

ROOT = Path(__file__).resolve().parents[1]
from mechcal.eval.dataset import CASES_DIR as CASES  # noqa: E402  (structure-corrected cases)


def per_case(dirs: list[Path], metric: str) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for d in dirs:
        for g in sorted(d.glob("*/grade.json")):
            grade = json.loads(g.read_text())
            values.setdefault(grade["case_id"], []).append(float(grade["metrics"].get(metric) or 0.0))
    return {c: mean(v) for c, v in values.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("methods", nargs="+", help="name=dir[,dir...]")
    ap.add_argument("--metrics", nargs="+", default=["ndcg_at_3", "es_ndcg_at_3", "top1_primary", "recall_at_3"])
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--exclude", default="", help="comma-separated case numbers to leave out (e.g. 0020,0141)")
    args = ap.parse_args()
    skip = {"AIE_DDX_MECH_V4_" + c.strip().zfill(4) for c in args.exclude.split(",") if c.strip()}
    methods = {spec.split("=", 1)[0]: [Path(p) for p in spec.split("=", 1)[1].split(",")] for spec in args.methods}
    clusters = paper_clusters(load_reference_cases(CASES))
    report = {}
    for metric in args.metrics:
        cols = {name: {c: v for c, v in per_case(dirs, metric).items() if c not in skip} for name, dirs in methods.items()}
        out = {"single": {n: {k: round(v, 3) if isinstance(v, float) else v for k, v in
                              clustered_bootstrap(c, clusters, n_resamples=args.resamples).items()}
                          for n, c in cols.items() if c}, "paired": {}}
        for a, b in combinations(cols, 2):
            shared = sorted(set(cols[a]) & set(cols[b]))
            if not shared:
                continue
            r = paired_clustered_bootstrap({c: cols[a][c] for c in shared}, {c: cols[b][c] for c in shared}, clusters,
                                           n_resamples=args.resamples)
            out["paired"][f"{a} - {b}"] = {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}
        report[metric] = out
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
