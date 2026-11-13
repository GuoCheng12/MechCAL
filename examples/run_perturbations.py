"""Evidence-reliability experiment: perturb real submissions and re-score them (evaluation side).

    python examples/run_perturbations.py outputs/agent/dev3a outputs/baselines/react_dev --out outputs/perturbations/c1

For each graded run with a submission, the rank-1 prediction is perturbed by every operator in
mechcal.eval.perturb (tamper a cited value, tamper the written number, swap in another molecule's
artifacts, flip the conclusion, strip artifacts, inject fake citations, plus paraphrase and reorder
controls) and re-scored by Layer A + the evidence judge. A reliable score drops for the damaging
operators and stays for the controls. Because a family usually rests on several claims, damaging
one claim need not change the family's score; the summary therefore also reports, per operator,
the share of changed claims that grounding or the judge flagged (claim_detection_rate). Writes
per-run results and a summary per input directory.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mechcal.eval.evidence_judge import EvidenceJudge, JudgeClientAdapter
from mechcal.eval.llm_settings import describe, judge_settings
from mechcal.eval.perturb import run_suite, summarize_suite
from mechcal.eval.submission import ArtifactStore, Submission
from mechcal.runtime.llm import OpenAIJsonClient

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "benchmarks/photomechbench/raw_v2_119/cases"


def load(root: Path) -> list[tuple[Path, Submission, ArtifactStore, str]]:
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "result.json").is_file()):
        result = json.loads((d / "result.json").read_text())
        if not result.get("submission"):
            continue
        sub = Submission.model_validate(result["submission"])
        case = json.loads((CASES / f"{sub.case_id}.json").read_text())
        out.append((d, sub, ArtifactStore.model_validate_json((d / "artifacts.json").read_text()),
                    case["public_input"]["molecule"]["structure"]["value"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parallel", type=int, default=6)
    args = ap.parse_args()
    judge = EvidenceJudge(JudgeClientAdapter(OpenAIJsonClient(judge_settings())))

    def scorer(submission, store, smiles):          # judge the rank-1 prediction only (cost)
        out = judge.judge(submission.mechanism_predictions[0], store, smiles)
        verdicts = {int(v["claim_id"][1:]) - 1: v.get("outcome") for v in (out.get("judgement") or {}).get("verifications", [])
                    if str(v.get("claim_id", "")).startswith("C") and str(v["claim_id"][1:]).isdigit()}
        return [(out["score"] or 0.0, verdicts)] + [0.0] * (len(submission.mechanism_predictions) - 1)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"judge": describe(judge_settings())}
    for root in args.roots:
        runs = load(root)
        donors = [r[2] for r in runs[1:] + runs[:1]]       # another case's artifacts for swap_molecule
        items = [(sub, store, smiles, donor) for (_, sub, store, smiles), donor in zip(runs, donors)]
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            parts = list(pool.map(lambda it: run_suite([it], scorer, ranks=(1,)), items))
        results = [r for part in parts for r in part]
        rows = [r.__dict__ | {"applicable": r.applicable, "passed": r.passed} for r in results]
        (args.out / f"{root.name}.json").write_text(json.dumps(rows, indent=1, default=str))
        summary[root.name] = summarize_suite(results)
        print(root.name, json.dumps(summary[root.name], indent=1, default=str), flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
