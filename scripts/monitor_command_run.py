"""Read-only command-loop monitor. Elapsed time freezes on a saved pause or finish."""

import argparse
import json
from pathlib import Path
import time

from mechcal.runtime.command_hypotheses import view
from mechcal.schemas import CaseRun


def snapshot(root, *, now=None):
    now = time.time() if now is None else now
    saved = json.loads((root / "command_checkpoint.json").read_text())
    p = saved["runtime"]["command_protocol"]
    clock = p.get("clock", {})
    summary_path = root.parent.parent / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    elapsed = clock.get("elapsed_seconds", summary.get("elapsed_seconds"))
    if clock and saved["status"] == "running":
        elapsed += max(0, now - clock["updated_at"])
    result = {"status": saved["status"], "phase": p["phase"], "computations": p.get("rounds", 0),
            "judgment_updates": p.get("revision", 0), "elapsed_seconds": elapsed,
            "model_calls": summary.get("model_calls"), "input_tokens": summary.get("input_tokens"),
            "output_tokens": summary.get("output_tokens"),
            "stop_reason": p.get("interruption"),
            "ranking": [{k: r[k] for k in ("label", "confidence", "support")}
                        for r in sorted(p.get("ranking", {}).values(), key=lambda r: r["priority"], reverse=True)[:6]]}
    if p.get("exploration") is not None:
        exploration = view(CaseRun.model_validate(saved))
        result.update(exploration_assessments=exploration["assessment_count"],
                      assessment_required=exploration["assessment_required"],
                      query_turns_remaining=exploration["query_turns_remaining"],
                      exploration_confidence_ranking=exploration["confidence_ranking"][:6],
                      formal_priority_ranking=result.pop("ranking"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.case_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
