"""Does evidence improve the ranking over the prior? Per case: final answer minus the prior-only ranking (evaluation side).

    python examples/evidence_effect.py c1=outputs/agent/dev4a,outputs/agent/dev4b c2=...

The prior-only ranking sorts the 11 families by the prior recorded on the run's board (set
before any calculation) and is scored like the answer. For each cycle: mean of (final - prior)
over cases (repeats averaged per case) with a paper-clustered bootstrap CI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from mechcal.eval.mechanism_ranking import score_mechanism_ranking
from mechcal.eval.stats import clustered_bootstrap, load_reference_cases, paper_clusters
from mechcal.mechanism_pool import MECHANISM_POOL

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "benchmarks/photomechbench/raw_v2_119/cases"
METRICS = ("ndcg_at_3", "top1_primary", "recall_at_3")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cycles", nargs="+", help="name=dir[,dir...]")
    args = ap.parse_args()
    clusters = paper_clusters(load_reference_cases(CASES))
    out = {}
    for spec in args.cycles:
        name, dirs = spec.split("=", 1)
        per = {m: {} for m in METRICS}
        prior_only = {m: {} for m in METRICS}
        final = {m: {} for m in METRICS}
        for d in dirs.split(","):
            for res_path in sorted(Path(d).glob("*/result.json")):
                grade_path = res_path.with_name("grade.json")
                if not grade_path.is_file():
                    continue
                res, grade = json.loads(res_path.read_text()), json.loads(grade_path.read_text())
                fam = res["board"]["families"]
                if any(f.get("prior") is None for f in fam.values()):
                    continue
                order = sorted(fam, key=lambda k: (-fam[k]["prior"], MECHANISM_POOL.index(k)))[:3]
                ref = json.loads((CASES / f"{res['case_id']}.json").read_text())["hidden_reference"]["reference_mechanisms"]
                p = score_mechanism_ranking([{"label": lab, "rank": i + 1} for i, lab in enumerate(order)], ref)
                for m in METRICS:
                    per[m].setdefault(res["case_id"], []).append(grade["metrics"][m] - p[m])
                    prior_only[m].setdefault(res["case_id"], []).append(p[m])
                    final[m].setdefault(res["case_id"], []).append(grade["metrics"][m])
        out[name] = {}
        for m in METRICS:
            diffs = {c: mean(v) for c, v in per[m].items()}
            boot = clustered_bootstrap(diffs, clusters)
            out[name][m] = {"prior_only": round(mean(mean(v) for v in prior_only[m].values()), 3),
                            "final": round(mean(mean(v) for v in final[m].values()), 3),
                            "final_minus_prior": round(boot["mean"], 3),
                            "ci": [round(boot["ci_low"], 3), round(boot["ci_high"], 3)], "n_cases": boot["n_cases"]}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
