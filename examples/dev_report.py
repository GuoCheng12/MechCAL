"""Summarize development runs of the MechCAL agent (one or more repeat directories of graded runs).

    python examples/dev_report.py outputs/agent/dev3a outputs/agent/dev3b [--compare outputs/agent/dev2]

Reports, over all runs found: ranking metrics (mean and spread across repeats), judge scores per
family, how often each family was tested (named in an experiment's `separates`) and predicted, tool
usage, wasted delegations (not applicable), interceptions by kind, self-repair, tokens and time.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
from pathlib import Path

SHORT = {"ICT_TICT_CT": "ICT", "RADIATIVE_RATE_STATE_BALANCE": "RAD", "TRIPLET_METAL_ENERGY_TRANSFER": "TRIP",
         "RIM_RIR_RIV": "RIM", "PACKING_HOST_MATRIX_CONFINEMENT": "PACK", "AGGREGATE_EXCITON_EXCIMER": "AGG",
         "ESIPT_PT": "ESIPT", "PET_ET": "PET", "HOST_GUEST_INTERACTION": "HG", "RACI_CI_ACCESS": "RACI",
         "SOKR_ANTI_KASHA": "SOKR"}


def runs(roots: list[Path]) -> list[dict]:
    out = []
    for root in roots:
        for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "grade.json").is_file()):
            ev = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines()]
            out.append({"root": root.name, "case": d.name, "grade": json.loads((d / "grade.json").read_text()),
                        "result": json.loads((d / "result.json").read_text()), "events": ev})
    return out


def summarize(rs: list[dict]) -> dict:
    metrics = collections.defaultdict(list)
    per_case = collections.defaultdict(list)
    fam_scores = collections.defaultdict(list)
    tested, predicted, referenced = collections.Counter(), collections.Counter(), collections.Counter()
    tools, intercepts, wasted = collections.Counter(), collections.Counter(), 0
    sys_fail = repaired = 0
    tokens, minutes = [], []
    for r in rs:
        m = r["grade"]["metrics"]
        for k in ("top1_primary", "top1_any", "recall_at_3", "ndcg_at_3", "es_ndcg_at_3"):
            metrics[k].append(m[k])
        per_case[r["case"]].append(m["ndcg_at_3"])
        for p in r["grade"]["judge"]:
            fam_scores[p["label"]].append(p["score"])
        predicted.update(r["grade"]["predicted"])
        referenced.update(lab for lab, _ in r["grade"]["reference"])
        for e in r["events"]:
            if e.get("role") == "task_created" or (e.get("role") == "action" and e["action"]["type"] in {"tool", "recipe"}):
                seps = (e.get("action") or {}).get("separates") if e.get("role") == "action" else None
                if seps:
                    tested.update(set(seps))
            if e.get("role") == "action" and e["action"]["type"] == "delegate":
                tested.update(set(e["action"].get("separates") or []))
            if e.get("role") == "attempt":
                tools[e["name"].split(".")[-1]] += 1
                if e.get("failure_class") == "not_applicable":
                    wasted += 1
            if e.get("role") == "interception":
                intercepts[e["code"]] += 1
        p = r["result"].get("protocol", {})
        sys_fail += p.get("system_failure_episodes", 0)
        repaired += p.get("self_repaired", 0)
        tokens.append(sum(v.get("total_tokens", 0) for v in r["result"]["usage"]["by_role"].values()))
        minutes.append(r["result"]["seconds"] / 60)
    by_root = collections.defaultdict(list)
    for r in rs:
        by_root[r["root"]].append(r["grade"]["metrics"]["ndcg_at_3"])
    return {
        "runs": len(rs), "cases": len(per_case),
        "metrics_mean": {k: round(st.mean(v), 3) for k, v in metrics.items()},
        "ndcg_by_repeat": {k: round(st.mean(v), 3) for k, v in by_root.items()},
        "ndcg_case_spread_mean_abs": round(st.mean(abs(v[0] - v[1]) for v in per_case.values() if len(v) > 1), 3)
        if any(len(v) > 1 for v in per_case.values()) else None,
        "judge_by_family": {SHORT[k]: {"n": len(v), "mean": round(st.mean(v), 2), "full": sum(x == 1 for x in v)}
                            for k, v in sorted(fam_scores.items())},
        "tested": {SHORT[k]: tested[k] for k in SHORT}, "predicted": {SHORT[k]: predicted[k] for k in SHORT},
        "in_reference": {SHORT[k]: referenced[k] for k in SHORT},
        "tool_attempts": dict(tools.most_common()), "not_applicable_attempts": wasted,
        "interceptions": dict(intercepts), "self_repair": f"{repaired}/{sys_fail}",
        "mean_tokens": round(st.mean(tokens)), "mean_minutes": round(st.mean(minutes), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--compare", nargs="*", type=Path, default=[])
    args = ap.parse_args()
    now = summarize(runs(args.roots))
    print(json.dumps(now, indent=1, ensure_ascii=False))
    if args.compare:
        before = runs(args.compare)
        cases = {r["case"] for r in runs(args.roots)}
        shared = [r for r in before if r["case"] in cases]
        print("\n== earlier runs on the same cases ==")
        print(json.dumps(summarize(shared), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
