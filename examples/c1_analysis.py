"""Cycle analysis of graded runs: agent repeats and baselines, with signals that do not depend on labels first.

    python examples/c1_analysis.py --agent outputs/agent/dev4a outputs/agent/dev4b \
        --before outputs/agent/dev3a outputs/agent/dev3b \
        --baseline react=outputs/baselines/react_dev direct=outputs/baselines/direct_dev --out outputs/analysis/c1.json

Per group of runs: ranking metrics (rough reference only), grounding and fabrication, judge score,
per-family precision / recall / predicted-to-reference ratio, and repeat stability (top-3 overlap
of the same case across repeats). For agent runs also: the prior-only ranking (families sorted by
the Planner's prior) scored like the answer, how often evidence changed the top three, the
evidence-availability test (how often a family is placed in the top three, split by whether it
is in the reference and whether it has computed claims; with priors also by prior top three), and how claims were marked (discriminates).
Evaluation side: reads hidden references; nothing goes back to any evaluated system.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
from pathlib import Path

from mechcal.eval.mechanism_ranking import score_mechanism_ranking
from mechcal.mechanism_pool import MECHANISM_POOL

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "benchmarks/photomechbench/raw_v2_119/cases"
SHORT = {"ICT_TICT_CT": "ICT", "RADIATIVE_RATE_STATE_BALANCE": "RAD", "TRIPLET_METAL_ENERGY_TRANSFER": "TRIP",
         "RIM_RIR_RIV": "RIM", "PACKING_HOST_MATRIX_CONFINEMENT": "PACK", "AGGREGATE_EXCITON_EXCIMER": "AGG",
         "ESIPT_PT": "ESIPT", "PET_ET": "PET", "HOST_GUEST_INTERACTION": "HG", "RACI_CI_ACCESS": "RACI",
         "SOKR_ANTI_KASHA": "SOKR"}
METRICS = ("top1_primary", "top1_any", "recall_at_3", "ndcg_at_3", "es_ndcg_at_3")


def load(root: Path) -> list[dict]:
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "grade.json").is_file()):
        out.append({"root": root.name, "case": d.name, "dir": d, "grade": json.loads((d / "grade.json").read_text()),
                    "result": json.loads((d / "result.json").read_text())})
    return out


def _mean(xs):
    xs = list(xs)
    return round(st.mean(xs), 3) if xs else None


def reference(case: str) -> list[dict]:
    data = json.loads((CASES / f"AIE_DDX_MECH_V4_{case}.json").read_text())
    return data["hidden_reference"]["reference_mechanisms"]


def common(rs: list[dict]) -> dict:
    metrics = {k: _mean((r["grade"]["metrics"].get(k) or 0.0) for r in rs) for k in METRICS}
    g = [r["grade"]["grounding"] for r in rs]
    computed = sum(x["computed_claims"] for x in g)
    judge = [p["score"] or 0.0 for r in rs for p in r["grade"]["judge"]]
    tp, pred, ref = collections.Counter(), collections.Counter(), collections.Counter()
    for r in rs:
        p, refs = set(r["grade"]["predicted"]), {lab for lab, _ in r["grade"]["reference"]}
        pred.update(p), ref.update(refs), tp.update(p & refs)
    family = {SHORT[f]: {"pred": pred[f], "ref": ref[f], "tp": tp[f],
                         "precision": round(tp[f] / pred[f], 2) if pred[f] else None,
                         "recall": round(tp[f] / ref[f], 2) if ref[f] else None,
                         "pred_over_ref": round(pred[f] / ref[f], 2) if ref[f] else None} for f in MECHANISM_POOL}
    by_case = collections.defaultdict(list)
    for r in rs:
        by_case[r["case"]].append(r)
    pairs = [(v[0], v[1]) for v in by_case.values() if len(v) > 1]
    stability = None
    if pairs:
        jac = [len(set(a["grade"]["predicted"]) & set(b["grade"]["predicted"])) /
               max(1, len(set(a["grade"]["predicted"]) | set(b["grade"]["predicted"]))) for a, b in pairs]
        stability = {"pairs": len(pairs), "top3_jaccard": _mean(jac),
                     "top1_same": _mean(float(a["grade"]["predicted"][:1] == b["grade"]["predicted"][:1]) for a, b in pairs),
                     "ndcg_abs_diff": _mean(abs(a["grade"]["metrics"]["ndcg_at_3"] - b["grade"]["metrics"]["ndcg_at_3"])
                                            for a, b in pairs)}
    return {"runs": len(rs), "cases": len(by_case),
            "no_submission": sum(bool(r["grade"].get("no_valid_submission")) for r in rs),
            "metrics": metrics,
            "grounding": {"computed_claims": computed,
                          "grounded_rate": round(sum(x["grounded_computed_claims"] for x in g) / computed, 3) if computed else None,
                          "fabricated": sum(x["fabricated_claims"] for x in g),
                          "unsourced": sum(x["unsourced_claims"] for x in g)},
            "judge_mean": _mean(judge), "judge_full_share": _mean(float(x == 1.0) for x in judge),
            "family": family, "stability": stability,
            "minutes_mean": _mean(r["result"].get("seconds", 0) / 60 for r in rs if r["result"].get("seconds")),
            "tool_seconds_mean": _mean((r["result"].get("tool_cost") or {}).get("tool_wall_seconds", 0) for r in rs)}


def agent_only(rs: list[dict]) -> dict:
    prior_metrics = collections.defaultdict(list)
    changed = with_prior = 0
    moves = collections.Counter()               # (family, in reference?, entered/left top 3 because of evidence)
    avail = collections.Counter()               # evidence-availability test: top-3 rate by reference x evidence
    marks = collections.Counter()
    top3_marks = collections.Counter()
    generic_top3 = 0
    proposals = collections.Counter()
    responses = collections.Counter()
    for r in rs:
        board = r["result"].get("board") or {}
        fams = board.get("families") or {}
        claims = board.get("claims") or []
        refs = {lab for lab, _ in r["grade"]["reference"]}
        top3 = r["grade"]["predicted"]
        priors = {f: s.get("prior") for f, s in fams.items()}
        if fams and all(v is not None for v in priors.values()):
            with_prior += 1
            order = sorted(priors, key=lambda f: (-priors[f], MECHANISM_POOL.index(f)))
            score = score_mechanism_ranking([{"label": f, "rank": i + 1} for i, f in enumerate(order[:3])],
                                            reference(r["case"]))
            for k in METRICS[:4]:
                prior_metrics[k].append(score[k])
            if set(order[:3]) != set(top3):
                changed += 1
            for f in set(top3) - set(order[:3]):
                moves[(SHORT[f], f in refs, "entered")] += 1
            for f in set(order[:3]) - set(top3):
                moves[(SHORT[f], f in refs, "left")] += 1
        computed_fams = {c["label"] for c in claims if c["claim"]["kind"] == "computed"}
        prior_top3 = set(order[:3]) if fams and all(v is not None for v in priors.values()) else None
        for f in MECHANISM_POOL:
            strata = ["all"] + ([] if prior_top3 is None else ["prior_top3" if f in prior_top3 else "prior_rest"])
            for stratum in strata:
                key = (stratum, "ref" if f in refs else "not_ref", "with_computed" if f in computed_fams else "without_computed")
                avail[key + ("families",)] += 1
                avail[key + ("in_top3",)] += f in top3
        for c in claims:
            if c["claim"]["kind"] == "computed":
                marks[c.get("discriminates") or "unmarked"] += 1
                if c["label"] in top3:
                    top3_marks[c.get("discriminates") or "unmarked"] += 1
        for f in top3:
            comp = [c for c in claims if c["label"] == f and c["claim"]["kind"] == "computed"]
            if comp and all(c.get("discriminates") == "no" for c in comp):
                generic_top3 += 1
        ev = r["dir"] / "events.jsonl"
        seen: set = set()
        if ev.is_file():
            for line in ev.read_text().splitlines():
                e = json.loads(line)
                if e.get("role") == "critic_summary":
                    for p in e.get("proposals") or []:
                        if p.get("proposal_id") not in seen:
                            seen.add(p.get("proposal_id"))
                            proposals[p.get("kind") or "other"] += 1
                if e.get("role") == "proposal_response":
                    responses[((e.get("proposal") or {}).get("kind") or "other", e.get("decision"))] += 1

    def rate(*key):
        n = avail[key + ("families",)]
        return f"{avail[key + ('in_top3',)]}/{n}" + (f" = {avail[key + ('in_top3',)] / n:.2f}" if n else "")

    availability = {f"{stratum} {ref} {comp}": rate(stratum, ref, comp)
                    for stratum in ("all", "prior_top3", "prior_rest") for ref in ("not_ref", "ref")
                    for comp in ("with_computed", "without_computed") if avail[(stratum, ref, comp, "families")]}

    return {"runs_with_full_prior": with_prior,
            "prior_only_ranking": {k: _mean(v) for k, v in prior_metrics.items()},
            "evidence_changed_top3": changed,
            "evidence_moves": {f"{f} {'ref' if inref else 'not-ref'} {d}": n for (f, inref, d), n in sorted(moves.items())},
            "availability_test": availability,
            "claim_marks_all": dict(marks), "claim_marks_top3": dict(top3_marks),
            "top3_families_generic_only": generic_top3,
            "critic_proposals": dict(proposals),
            "proposal_responses": {f"{k} {d}": n for (k, d), n in sorted(responses.items())}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", nargs="+", type=Path, required=True)
    ap.add_argument("--before", nargs="*", type=Path, default=[])
    ap.add_argument("--baseline", nargs="*", default=[], help="name=dir")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    agent = [r for root in args.agent for r in load(root)]
    report = {"agent": common(agent) | agent_only(agent)}
    cases = {r["case"] for r in agent}
    if args.before:
        before = [r for root in args.before for r in load(root) if r["case"] in cases]
        report["before"] = common(before) | agent_only(before)
    for spec in args.baseline:
        name, path = spec.split("=", 1)
        report[name] = common([r for r in load(Path(path)) if r["case"] in cases])
    text = json.dumps(report, indent=1, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()
