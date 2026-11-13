"""Evidence-reliability experiment: perturb real submissions and re-score them (evaluation side).

    python examples/run_perturbations.py outputs/agent/dev3a outputs/baselines/react_dev --out outputs/perturbations/c1

For each graded run with a submission, the rank-1 prediction is perturbed by every operator in
mechcal.eval.perturb (tamper a cited value, tamper the written number, swap in another molecule's
artifacts, flip the conclusion, strip artifacts, inject fake citations, plus paraphrase and reorder
controls) and re-scored by Layer A + the evidence judge. A reliable score drops for the damaging
operators and stays for the controls. Writes per-run results and a summary per input directory.
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mechcal.agent.schedule import wait_for_off_peak
from mechcal.eval.evidence_judge import EvidenceJudge, JudgeClientAdapter
from mechcal.eval.llm_settings import describe, judge_settings
from mechcal.eval.perturb import run_suite, summarize_suite
from mechcal.eval.submission import ArtifactStore, Submission
from mechcal.runtime.llm import OpenAIJsonClient

ROOT = Path(__file__).resolve().parents[1]
from mechcal.eval.dataset import CASES_DIR as CASES  # noqa: E402  (structure-corrected cases)


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
    ap.add_argument("--schedule", choices=["off-peak", "any"], default="off-peak",
                    help="off-peak (default): judge each run only in DeepSeek off-peak hours; any: immediately")
    ap.add_argument("--peak-margin-minutes", type=float, default=15.0)
    args = ap.parse_args()
    judge = EvidenceJudge(JudgeClientAdapter(OpenAIJsonClient(judge_settings())))

    def scorer(submission, store, smiles):          # judge the rank-1 prediction only (cost)
        first = judge.judge(submission.mechanism_predictions[0], store, smiles)["score"] or 0.0
        return [first] + [0.0] * (len(submission.mechanism_predictions) - 1)

    args.out.mkdir(parents=True, exist_ok=True)
    gate = threading.Lock()
    summary = {"judge": describe(judge_settings())}
    for root in args.roots:
        runs = load(root)
        donors = [r[2] for r in runs[1:] + runs[:1]]       # another case's artifacts for swap_molecule
        items = [(sub, store, smiles, donor) for (_, sub, store, smiles), donor in zip(runs, donors)]
        def one(item):
            if args.schedule == "off-peak":
                with gate:
                    wait_for_off_peak(args.peak_margin_minutes, log=lambda m: print(m, flush=True))
            return run_suite([item], scorer, ranks=(1,))

        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            parts = list(pool.map(one, items))
        results = [r for part in parts for r in part]
        rows = [r.__dict__ | {"applicable": r.applicable, "passed": r.passed} for r in results]
        (args.out / f"{root.name}.json").write_text(json.dumps(rows, indent=1, default=str))
        summary[root.name] = summarize_suite(results)
        print(root.name, json.dumps(summary[root.name], indent=1, default=str), flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
