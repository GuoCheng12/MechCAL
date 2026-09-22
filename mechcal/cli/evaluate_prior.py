from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from mechcal.eval.common_prior import (
    common_prior_predictions,
    common_prior_support_scores,
)
from mechcal.eval.mechanism_ranking import score_mechanism_ranking


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CommonPrior@3 on the slim mechanism benchmark."
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/common-prior"),
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--exclude-case", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cases = _load_cases(args.case_dir)
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in wanted]
    excluded = set(args.exclude_case)
    cases = [case for case in cases if case["case_id"] not in excluded]
    if args.limit is not None:
        cases = cases[: args.limit]

    output_dir = args.output_dir.expanduser().resolve()
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for case in cases:
        predictions = common_prior_predictions()
        support_scores = common_prior_support_scores(predictions)
        scores = score_mechanism_ranking(
            predictions,
            case["reference_mechanisms"],  # type: ignore[arg-type]
            support_scores=support_scores,
        )
        row = {
            "case_id": case["case_id"],
            "subject_id": "common_prior_at3",
            "mechanism_predictions": predictions,
            "support_scores": support_scores,
            "scores": scores,
        }
        rows.append(row)
        _write_json(eval_dir / f"{case['case_id']}.json", row)
        print(f"{case['case_id']} " + _compact_score_text(scores), flush=True)
    _write_json(
        output_dir / "summary.json",
        {
            "case_count": len(rows),
            "subject_id": "common_prior_at3",
            "averages": _averages(rows),
            "rows": rows,
        },
    )


def _load_cases(case_dir: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for path in sorted(case_dir.expanduser().resolve().glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        hidden_reference = payload.get("hidden_reference") or {}
        reference_mechanisms = hidden_reference.get("reference_mechanisms")
        if not isinstance(reference_mechanisms, list):
            raise ValueError(f"Missing reference_mechanisms in {path}")
        cases.append(
            {
                "case_id": str(payload["case_id"]),
                "reference_mechanisms": reference_mechanisms,
            }
        )
    return cases


def _averages(rows: list[dict[str, object]]) -> dict[str, float | None]:
    metrics = [
        "top1_primary",
        "top1_any",
        "recall_at_3",
        "ndcg_at_3",
        "es_ndcg_at_3",
        "support_at_top1",
        "claim_safety_at_3",
        "unsupported_strong_claim_rate_at_3",
        "credibility_gap_at_3",
    ]
    return {
        metric: _mean(row["scores"].get(metric) for row in rows)  # type: ignore[union-attr]
        for metric in metrics
    }


def _mean(values) -> float | None:  # noqa: ANN001
    numeric = [float(value) for value in values if isinstance(value, int | float)]
    return statistics.fmean(numeric) if numeric else None


def _compact_score_text(scores: dict[str, object]) -> str:
    return (
        f"Top1-primary={scores['top1_primary']:.3f} "
        f"Top1-any={scores['top1_any']:.3f} "
        f"R@3={scores['recall_at_3']:.3f} "
        f"nDCG@3={scores['ndcg_at_3']:.3f} "
        f"ES-nDCG@3={scores['es_ndcg_at_3']:.3f} "
        f"ClaimSafety@3={scores['claim_safety_at_3']:.3f}"
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
