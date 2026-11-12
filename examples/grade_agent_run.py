"""Grade one agent/baseline run directory (evaluation side; reads hidden references).

    python examples/grade_agent_run.py outputs/agent/0003-smoke-1

Reads result.json (submission) and artifacts.json (evaluator artifacts), applies the Layer A
grounding check and the evidence judge (MECHCAL_JUDGE_* settings), then scores the ranking
against the hidden reference. Writes grade.json next to the run. Never feeds anything back
to the evaluated system.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mechcal.eval.evidence_judge import EvidenceJudge, JudgeClientAdapter
from mechcal.eval.grounding import check_submission
from mechcal.eval.llm_settings import describe, judge_settings
from mechcal.eval.mechanism_ranking import score_mechanism_ranking
from mechcal.eval.submission import ArtifactStore, Submission
from mechcal.runtime.llm import OpenAIJsonClient

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "benchmarks/photomechbench/raw_v2_119/cases"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--no-judge", action="store_true", help="Layer A and ranking metrics only")
    args = parser.parse_args()
    result = json.loads((args.run_dir / "result.json").read_text())
    if result.get("submission") is None:            # no valid answer: scored as an empty answer, not dropped
        case = json.loads((CASES / f"{result['case_id']}.json").read_text())
        ranking = score_mechanism_ranking([], case["hidden_reference"]["reference_mechanisms"])
        grade = {"case_id": result["case_id"], "predicted": [], "no_valid_submission": True,
                 "reference": [(m["label"], m["role"]) for m in case["hidden_reference"]["reference_mechanisms"]],
                 "metrics": {k: (0.0 if k == "es_ndcg_at_3" and v is None else v) for k, v in ranking.items()
                             if k != "details"},
                 "grounding": {"computed_claims": 0, "grounded_computed_claims": 0, "fabricated_claims": 0,
                               "unsourced_claims": 0, "fabrication_rate": 0.0}, "judge": [], "judge_model": None}
        (args.run_dir / "grade.json").write_text(json.dumps(grade, indent=1, default=str))
        print(json.dumps({k: grade[k] for k in ("predicted", "metrics")}, indent=1))
        return
    submission = Submission.model_validate(result["submission"])
    store = ArtifactStore.model_validate_json((args.run_dir / "artifacts.json").read_text())
    case = json.loads((CASES / f"{submission.case_id}.json").read_text())
    smiles = case["public_input"]["molecule"]["structure"]["value"]
    grounding = check_submission(submission, store, smiles)
    judged, scores = [], {}
    if not args.no_judge:
        settings = judge_settings()
        adapter = JudgeClientAdapter(OpenAIJsonClient(settings))
        judge = EvidenceJudge(adapter)
        for pred in submission.mechanism_predictions:
            out = judge.judge(pred, store, smiles)
            judged.append({"label": pred.label, "rank": pred.rank, **out})
            scores[pred.label] = out["score"] if out["score"] is not None else 0.0
    preds = [{"label": p.label, "rank": p.rank} for p in submission.mechanism_predictions]
    ranking = score_mechanism_ranking(preds, case["hidden_reference"]["reference_mechanisms"],
                                      support_scores=scores if scores else None)
    grade = {"case_id": submission.case_id, "predicted": [p["label"] for p in preds],
             "reference": [(m["label"], m["role"]) for m in case["hidden_reference"]["reference_mechanisms"]],
             "metrics": {k: v for k, v in ranking.items() if k != "details"},
             "grounding": {k: grounding.model_dump()[k] for k in ("computed_claims", "grounded_computed_claims",
                                                                  "fabricated_claims", "unsourced_claims", "fabrication_rate")},
             "judge": judged, "judge_model": describe(judge_settings()) if not args.no_judge else None}
    (args.run_dir / "grade.json").write_text(json.dumps(grade, indent=1, default=str))
    print(json.dumps({k: grade[k] for k in ("predicted", "reference", "metrics", "grounding")}, indent=1, default=str))
    for j in judged:
        print(j["rank"], j["label"], "score", j["score"], "judge", j.get("judge_score"), j["status"],
              (j.get("judgement") or {}).get("rationale", "")[:300])


if __name__ == "__main__":
    main()
