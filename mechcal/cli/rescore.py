from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from mechcal.eval.mechanism_ranking import (
    LLMMechanismEvidenceSupportJudge,
    judge_mechanism_prediction_support,
    score_mechanism_ranking,
)
from mechcal.runtime.llm import OpenAICompatibleSettings
from mechcal.runtime.output_assembler import OutputAssembler
from mechcal.schemas import CaseRun

METRICS = [
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rescore existing MAS mechanism case_run.json files without rerunning MAS."
        )
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Existing supervised MAS output directory containing cases/<case_id>/mas_runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--exclude-case", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--judge-attempts", type=int, default=5)
    args = parser.parse_args()

    settings = OpenAICompatibleSettings.from_env()
    missing = settings.missing_fields()
    if missing:
        raise RuntimeError("Missing LLM settings: " + ", ".join(missing))

    cases = _load_cases(args.case_dir)
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if str(case["case_id"]) in wanted]
    excluded = set(args.exclude_case)
    cases = [case for case in cases if str(case["case_id"]) not in excluded]
    if args.limit is not None:
        cases = cases[: args.limit]

    output_dir = args.output_dir.expanduser().resolve()
    eval_dir = output_dir / "mas_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.source_dir.expanduser().resolve()
    support_judge = LLMMechanismEvidenceSupportJudge(settings=settings)

    rows: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        print(f"[{index}/{len(cases)}] {case_id}", flush=True)
        case_run_path = _source_case_run_path(source_dir, case_id)
        case_run = CaseRun.model_validate_json(case_run_path.read_text(encoding="utf-8"))
        predictions = _prediction_payload(case_run)
        evaluation_status = "scored"
        evaluation_failure = None
        try:
            support = _judge_support_with_retries(
                predictions,
                judge=support_judge,
                label=case_id,
                attempts=args.judge_attempts,
            )
            scores = score_mechanism_ranking(
                predictions,
                case["reference_mechanisms"],  # type: ignore[arg-type]
                support_scores=support["support_scores"],  # type: ignore[arg-type]
            )
        except Exception as exc:  # noqa: BLE001
            evaluation_status = "judge_failed"
            evaluation_failure = {"type": type(exc).__name__, "message": str(exc)}
            support = {"support_scores": {}, "judgements": []}
            scores = _unscored_metrics()
        row = {
            "case_id": case_id,
            "status": case_run.status,
            "evaluation_status": evaluation_status,
            "evaluation_failure": evaluation_failure,
            "round_count": len(case_run.round_ids),
            "reference_mechanisms": case["reference_mechanisms"],
            "mechanism_predictions": predictions,
            "support_judgements": support["judgements"],
            "scores": scores,
            "case_run_path": str(case_run_path),
            "rescore_source_dir": str(source_dir),
        }
        rows.append(row)
        _write_json(eval_dir / f"{case_id}.json", row)
        _write_summary(output_dir / "summary.json", rows)
        print(
            "  "
            + _compact_score_text(scores)
            + f"; status={case_run.status}; evaluation_status={evaluation_status}",
            flush=True,
        )
        if evaluation_failure is not None:
            print(
                f"    evaluation_failure={evaluation_failure['type']}: "
                f"{evaluation_failure['message']}",
                flush=True,
            )
    summary = _write_summary(output_dir / "summary.json", rows)
    print(f"Final summary: {output_dir / 'summary.json'}", flush=True)
    print(
        "Averages: "
        + " ".join(
            f"{metric}={_fmt(summary['averages'].get(metric))}" for metric in METRICS
        ),
        flush=True,
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


def _source_case_run_path(source_dir: Path, case_id: str) -> Path:
    candidates = [
        source_dir / "cases" / case_id / "mas_runs" / case_id / "case_run.json",
        source_dir / "mas_runs" / case_id / "case_run.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No cached case_run.json for "
        f"{case_id}; checked: {', '.join(str(path) for path in candidates)}"
    )


def _prediction_payload(case_run: CaseRun) -> list[dict[str, object]]:
    predictions = _mechanism_predictions(case_run)
    payload: list[dict[str, object]] = []
    for index, prediction in enumerate(predictions, start=1):
        payload.append(
            {
                "prediction_id": f"P{index:03d}",
                "rank": prediction.rank,
                "label": prediction.label,
                "confidence": prediction.confidence,
                "differential_priority": prediction.differential_priority,
                "plausibility_confidence": prediction.plausibility_confidence,
                "claim_status": prediction.claim_status,
                "support_strength": prediction.support_strength,
                "evidence_support": prediction.evidence_support,
                "evidence_refs": list(prediction.evidence_refs),
                "evidence": list(prediction.evidence),
                "evidence_details": [
                    item.model_dump(mode="json") for item in prediction.evidence_details
                ],
                "validation_needed": list(prediction.validation_needed),
                "limitations": list(prediction.limitations),
            }
        )
    return payload


def _mechanism_predictions(case_run: CaseRun):
    if (
        case_run.differential_mechanism_portfolio is not None
        or case_run.mechanism_support_arguments
    ):
        rebuilt = OutputAssembler().build_mechanism_predictions(case_run)
        if rebuilt:
            return rebuilt
    predictions = case_run.mechanism_predictions
    if not predictions and case_run.final_answer is not None:
        predictions = case_run.final_answer.mechanism_predictions
    return predictions


def _judge_support_with_retries(
    predictions: list[dict[str, object]],
    *,
    judge: LLMMechanismEvidenceSupportJudge,
    label: str,
    attempts: int,
) -> dict[str, object]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return judge_mechanism_prediction_support(predictions, judge=judge)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(
                f"  support judge attempt {attempt}/{attempts} failed for {label}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if attempt < attempts:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Support judge failed for {label}: {last_error}") from last_error


def _write_summary(path: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    scored_rows = [
        row for row in rows if row.get("evaluation_status", "scored") == "scored"
    ]
    averages = {
        metric: _mean(
            row["scores"].get(metric)  # type: ignore[union-attr]
            for row in scored_rows
        )
        for metric in METRICS
    }
    summary: dict[str, object] = {
        "case_count": len(rows),
        "scored_case_count": len(scored_rows),
        "unscored_case_count": len(rows) - len(scored_rows),
        "averages": averages,
        "rows": rows,
    }
    _write_json(path, summary)
    return summary


def _mean(values) -> float | None:  # noqa: ANN001
    numeric = [float(value) for value in values if isinstance(value, int | float)]
    return statistics.fmean(numeric) if numeric else None


def _unscored_metrics() -> dict[str, object]:
    metrics = [
        "top1_primary",
        "top1_any",
        "top1_hit",
        "recall_at_3",
        "ndcg_at_3",
        "es_ndcg_at_3",
        "support_at_top1",
        "unsupported_strong_claim_rate_at_3",
        "claim_safety_at_3",
        "ucr_at_3",
        "credibility_gap_at_3",
    ]
    return {metric: None for metric in metrics}


def _compact_score_text(scores: dict[str, object]) -> str:
    if not isinstance(scores.get("top1_primary"), int | float):
        return "unscored"
    return (
        f"Top1-primary={scores['top1_primary']:.3f} "
        f"Top1-any={scores['top1_any']:.3f} "
        f"R@3={scores['recall_at_3']:.3f} "
        f"nDCG@3={scores['ndcg_at_3']:.3f} "
        f"ES-nDCG@3={scores['es_ndcg_at_3']:.3f} "
        f"Support@Top1={scores['support_at_top1']:.3f} "
        f"ClaimSafety@3={scores['claim_safety_at_3']:.3f}"
    )


def _fmt(value: object) -> str:
    return f"{value:.3f}" if isinstance(value, int | float) else "NA"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
