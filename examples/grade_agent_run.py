"""Grade one agent/baseline run directory (evaluation side; reads hidden references).

    python examples/grade_agent_run.py outputs/agent/0003-smoke-1

Reads result.json (submission) and artifacts.json (evaluator artifacts), applies the Layer A
grounding check and the evidence judge (MECHCAL_JUDGE_* settings), then computes the per-molecule
metrics (docs/evaluation-metrics-zh.md) against the hidden reference. Writes grade.json next to
the run. A run without a valid answer is graded as an empty answer. Never feeds anything back to
the evaluated system.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mechcal.eval.dataset import case_record
from mechcal.eval.evidence_judge import EvidenceJudge, JudgeClientAdapter
from mechcal.eval.grounding import check_submission
from mechcal.eval.llm_settings import describe, judge_settings, record_judge_usage
from mechcal.eval.molecule_metrics import molecule_metrics, reference_of
from mechcal.eval.submission import ArtifactStore, Submission
from mechcal.runtime.llm import OpenAIJsonClient

GROUNDING_KEYS = ("computed_claims", "grounded_computed_claims", "fabricated_claims", "unsourced_claims", "fabrication_rate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--no-judge", action="store_true", help="grounding and label metrics only (ES-Rec and EP not computed)")
    args = parser.parse_args()
    record_judge_usage(args.run_dir / "judge_telemetry.jsonl")      # judge tokens, apart from the run's own
    result = json.loads((args.run_dir / "result.json").read_text())
    case = case_record(result["case_id"])
    ref = reference_of(case["hidden_reference"])
    smiles = case["public_input"]["molecule"]["structure"]["value"]
    submission = Submission.model_validate(result["submission"]) if result.get("submission") else None
    claimed = [p.label for p in submission.mechanism_predictions] if submission else []
    judged, scores, verdicts, grounding = [], {}, [], dict.fromkeys(GROUNDING_KEYS, 0)
    if submission:
        store = ArtifactStore.model_validate_json((args.run_dir / "artifacts.json").read_text())
        check = check_submission(submission, store, smiles).model_dump()
        grounding = {k: check[k] for k in GROUNDING_KEYS}
        if not args.no_judge:
            judge = EvidenceJudge(JudgeClientAdapter(OpenAIJsonClient(judge_settings())))
            for pred in submission.mechanism_predictions:
                out = judge.judge(pred, store, smiles)
                judged.append({"label": pred.label, "rank": pred.rank, **out})
                scores[pred.label] = out["score"] if out["score"] is not None else 0.0
                verdicts += [v.get("outcome") for v in (out.get("judgement") or {}).get("verifications", [])]
    metrics = molecule_metrics(claimed, ref, scores, verdicts)
    if args.no_judge:
        metrics.update(es_rec=None, ep=None)
    grade = {"case_id": result["case_id"], "claimed": claimed, "no_valid_submission": submission is None,
             "reference": {"supported": sorted(ref.supported), "primary": sorted(ref.primary), "excluded": sorted(ref.excluded)},
             "metrics": metrics, "grounding": grounding, "judge": judged,
             "judge_model": describe(judge_settings()) if not args.no_judge else None}
    (args.run_dir / "grade.json").write_text(json.dumps(grade, indent=1, default=str))
    print(json.dumps({k: grade[k] for k in ("claimed", "reference", "metrics", "grounding")}, indent=1, default=str))
    for j in judged:
        print(j["rank"], j["label"], "score", j["score"], "judge", j.get("judge_score"), j["status"],
              (j.get("judgement") or {}).get("rationale", "")[:300])


if __name__ == "__main__":
    main()
