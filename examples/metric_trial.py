"""Trial of the 7-family set metrics on finished runs (evaluation side; calls the judge).

    python examples/metric_trial.py mechcal_c2=outputs/agent/dev5a,outputs/agent/dev5b direct=outputs/baselines/direct_dev \
        --exclude 0020,0141,0152,0153 --out outputs/metric_trial

The runs answered with a fixed top-3 over the old 11 families. Their answers are mapped onto the
7-family pool (mechcal.eval.family_reduction rules: RACI -> RIM, TRIPLET renamed, PACKING /
HOST_GUEST / SOKR dropped) and read as the claimed set; every claimed family is judged again with
the current evidence rubric. Board plausibilities (MechCAL only) give the probabilities for the
Brier score (RIM takes the larger of RIM and RACI). Judgements are cached under --out, and judge
calls start only in DeepSeek off-peak hours. Prints the metric tables (docs/evaluation-metrics-zh.md)
with paper-clustered 95 % intervals and writes them to <out>/summary.json.
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

from mechcal.agent.schedule import wait_for_off_peak
from mechcal.eval import set_metrics as sm
from mechcal.eval.dataset import CASES_DIR, public_smiles
from mechcal.eval.family_reduction import ENVIRONMENT, MERGED_INTO, OUT_OF_POOL
from mechcal.eval.grounding import check_submission
from mechcal.eval.stats import load_reference_cases, paper_clusters
from mechcal.eval.submission import ArtifactStore, Prediction, Submission
from mechcal.mechanism_pool import MECHANISM_POOL as POOL

SHORT = {"RIM_RIR_RIV": "RIM", "ICT_TICT_CT": "ICT", "ESIPT_PT": "ESIPT", "PET_ET": "PET",
         "AGGREGATE_EXCITON_EXCIMER": "AGG", "RADIATIVE_RATE_STATE_BALANCE": "RAD", "TRIPLET_ISC_RTP_TADF": "TRIP"}
CONSTANT = ["RIM_RIR_RIV", "ICT_TICT_CT", "RADIATIVE_RATE_STATE_BALANCE"]


def map_submission(raw: dict | None, case_id: str) -> Submission | None:
    """Old 11-family predictions -> 7-family predictions in the same order (merged families keep the
    earlier position and both evidence lists; dropped families vanish)."""
    if not raw:
        return None
    merged: dict[str, dict] = {}
    for p in sorted(raw["mechanism_predictions"], key=lambda p: p["rank"]):
        if p["label"] in ENVIRONMENT or p["label"] in OUT_OF_POOL:
            continue
        label = MERGED_INTO.get(p["label"], p["label"])
        if label in merged:
            merged[label]["evidence"] += p.get("evidence", [])
            merged[label]["limitations"] += p.get("limitations", [])
        else:
            merged[label] = {"label": label, "evidence": list(p.get("evidence", [])),
                             "limitations": list(p.get("limitations", []))}
    if not merged:
        return None
    preds = [Prediction(rank=i, **m) for i, m in enumerate(merged.values(), start=1)]
    return Submission(case_id=case_id, mechanism_predictions=preds)


def board_probs(result: dict) -> dict[str, float] | None:
    fams = (result.get("board") or {}).get("families")
    if not fams:
        return None
    p = {f: s["plausibility"] for f, s in fams.items()}
    return {"RIM_RIR_RIV": max(p.get("RIM_RIR_RIV", 0.0), p.get("RACI_CI_ACCESS", 0.0)),
            "ICT_TICT_CT": p["ICT_TICT_CT"], "ESIPT_PT": p["ESIPT_PT"], "PET_ET": p["PET_ET"],
            "AGGREGATE_EXCITON_EXCIMER": p["AGGREGATE_EXCITON_EXCIMER"],
            "RADIATIVE_RATE_STATE_BALANCE": p["RADIATIVE_RATE_STATE_BALANCE"],
            "TRIPLET_ISC_RTP_TADF": p.get("TRIPLET_ISC_RTP_TADF", p.get("TRIPLET_METAL_ENERGY_TRANSFER", 0.0))}


def system_tokens(run_dir: Path) -> int:
    path = run_dir / "api_telemetry.jsonl"
    if not path.is_file():
        return 0
    return sum((json.loads(l).get("usage") or {}).get("total_tokens", 0) for l in path.read_text().splitlines() if l.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("methods", nargs="+", help="name=dir[,dir...]")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exclude", default="", help="case numbers to leave out")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--no-judge", action="store_true", help="dry run: no judge calls (evidence scores 0)")
    ap.add_argument("--limit", type=int, default=0, help="dry run: at most this many runs per method")
    args = ap.parse_args()
    skip = {"AIE_DDX_MECH_V4_" + c.strip().zfill(4) for c in args.exclude.split(",") if c.strip()}
    refs = {c.case_id: c for c in load_reference_cases(CASES_DIR)}
    clusters = paper_clusters(list(refs.values()))
    judge = None
    if not args.no_judge:
        from mechcal.eval.evidence_judge import EvidenceJudge, JudgeClientAdapter
        from mechcal.eval.llm_settings import judge_settings, record_judge_usage
        from mechcal.runtime.llm import OpenAIJsonClient
        args.out.mkdir(parents=True, exist_ok=True)
        record_judge_usage(args.out / "judge_telemetry.jsonl")
        judge = EvidenceJudge(JudgeClientAdapter(OpenAIJsonClient(judge_settings())))
    gate = threading.Lock()

    def judged(cache: Path, pred: Prediction, store: ArtifactStore, smiles: str) -> dict:
        if cache.is_file():
            return json.loads(cache.read_text())
        if judge is None:
            return {"score": 0.0, "status": "not_judged"}
        with gate:
            wait_for_off_peak(10, log=lambda m: print(m, flush=True))
        out = judge.judge(pred, store, smiles)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out, default=str))
        return out

    report, constant_rows = {}, {}
    for spec in args.methods:
        name, dirs = spec.split("=", 1)
        runs = [d for root in dirs.split(",") for d in sorted(Path(root).glob("*/result.json"))]
        runs = [p for p in runs if json.loads(p.read_text())["case_id"] not in skip
                and refs.get(json.loads(p.read_text())["case_id"]) and refs[json.loads(p.read_text())["case_id"]].labels]
        if args.limit:
            runs = runs[:args.limit]

        def one(result_path: Path) -> tuple[sm.Row, dict]:
            run_dir = result_path.parent
            result = json.loads(result_path.read_text())
            cid = result["case_id"]
            ref = refs[cid]
            sub = map_submission(result.get("submission"), cid)
            claimed = [p.label for p in sub.mechanism_predictions] if sub else []
            scores, verdicts, grounded, computed = {}, [], 0, 0
            if sub:
                store = ArtifactStore.model_validate_json((run_dir / "artifacts.json").read_text())
                smiles = public_smiles(cid)
                check = check_submission(sub, store, smiles)
                computed, grounded = check.computed_claims, check.grounded_computed_claims
                for pred in sub.mechanism_predictions:
                    cache = args.out / "judgements" / name / run_dir.parent.name / run_dir.name / f"{pred.label}.json"
                    out = judged(cache, pred, store, smiles)
                    scores[pred.label] = out.get("score") or 0.0
                    verdicts += [v.get("outcome") for v in (out.get("judgement") or {}).get("verifications", [])]
            row = sm.Row(cid, claimed, set(ref.labels), set(ref.primary_labels), scores, board_probs(result))
            return row, {"verdicts": verdicts, "grounded": grounded, "computed": computed,
                         "tokens": system_tokens(run_dir)}

        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            done = list(pool.map(one, runs))
        rows = [r for r, _ in done]
        extra = [e for _, e in done]
        for r in rows:
            constant_rows.setdefault(r.case_id, sm.Row(r.case_id, CONSTANT, r.reference, r.primary, {}, None))
        report[name] = describe(rows, extra, clusters)
        print(f"{name}: {len(rows)} runs on {len({r.case_id for r in rows})} cases", flush=True)
    report["constant (RIM, ICT, RAD)"] = describe(list(constant_rows.values()), [], clusters)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(report, indent=1))
    print_tables(report)


def describe(rows, extra, clusters) -> dict:
    s = sm.summarize(rows, POOL)
    ci = {k: sm.cluster_bootstrap(rows, clusters, lambda rs, k=k: sm.summarize(rs, POOL)[k]) for k in s}
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r.case_id, []).append(r)
    verdicts = [v for e in extra for v in e["verdicts"]]
    return {"runs": len(rows), "cases": len(groups), "metrics": s, "ci": ci,
            "family_mcc": sm.family_mcc(rows, POOL),
            "family_counts": {f: sm.family_counts(rows, f) for f in POOL},
            "pass_k": sm.pass_k(groups),
            "grounded_rate": (sum(e["grounded"] for e in extra) / sum(e["computed"] for e in extra))
            if extra and sum(e["computed"] for e in extra) else None,
            "evidence_precision": (verdicts.count("consistent") / len(verdicts)) if verdicts else None,
            "mean_claimed": mean(len(r.claimed) for r in rows),
            "mean_system_tokens": mean(e["tokens"] for e in extra) if extra else None}


def print_tables(report: dict) -> None:
    fmt = lambda v: "  -  " if v is None else f"{v:5.2f}"
    def cell(d, k):
        lo, hi = d["ci"][k]
        return f"{fmt(d['metrics'][k])} [{fmt(lo)},{fmt(hi)}]"
    print("\nMain table (95% paper-clustered intervals)")
    print(f"{'method':26s} {'ES-F1':19s} {'Top-1 primary':19s} {'macro MCC':19s} {'Brier (lower better)':19s}")
    for n, d in report.items():
        print(f"{n:26s} {cell(d, 'es_f1')} {cell(d, 'top1_primary')} {cell(d, 'macro_mcc')} {cell(d, 'brier')}")
    print("\nEvidence, reliability and cost")
    print(f"{'method':26s} label-F1  gap   grounded  evid.prec  pass^k(top1) same-set  claimed/run  tokens/run")
    for n, d in report.items():
        m, pk = d["metrics"], d["pass_k"]
        gap = (m["label_f1"] - m["es_f1"]) if m["label_f1"] is not None and m["es_f1"] is not None else None
        tok = d["mean_system_tokens"]
        print(f"{n:26s} {fmt(m['label_f1'])}  {fmt(gap)}  {fmt(d['grounded_rate'])}     {fmt(d['evidence_precision'])}"
              f"      {fmt(pk['all_top1_correct'])}       {fmt(pk['identical_sets'])}    {d['mean_claimed']:5.2f}"
              f"     {'-' if not tok else f'{tok / 1e3:.0f}k'}")
    print("\nPer-family MCC")
    print(f"{'method':26s} " + " ".join(f"{SHORT[f]:>6s}" for f in POOL))
    for n, d in report.items():
        print(f"{n:26s} " + " ".join(f"{d['family_mcc'][f]:6.2f}" if f in d["family_mcc"] else "     -" for f in POOL))


if __name__ == "__main__":
    main()
