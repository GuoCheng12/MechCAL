from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from mechcal.baselines.chemgraph_mas import (
    BudgetLimits,
    ChemGraphMASSubject,
)
from mechcal.eval.mechanism_baselines import (
    prediction_payload_from_structure_llm_mechanism_json,
)
from mechcal.eval.mechanism_ranking import (
    LLMMechanismEvidenceSupportJudge,
    judge_mechanism_prediction_support,
    score_mechanism_ranking,
)
from mechcal.public_ids import anonymized_case_id
from mechcal.runtime.llm import OpenAICompatibleSettings

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate adapted ChemGraph-MAS on PhotoMechBench."
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/chemgraph"),
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument(
        "--exclude-case",
        action="append",
        default=[],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-model-calls", type=int, required=True)
    parser.add_argument("--max-tool-calls", type=int, required=True)
    parser.add_argument("--max-total-tokens", type=int, required=True)
    parser.add_argument("--max-output-tokens-per-call", type=int, default=2400)
    parser.add_argument("--recursion-limit", type=int, default=80)
    parser.add_argument("--schema-retries", type=int, default=1)
    parser.add_argument(
        "--responses-api",
        action="store_true",
        help=(
            "Use the OpenAI Responses API through LangChain. Sampling controls "
            "are omitted for providers that reject explicit temperature settings."
        ),
    )
    parser.add_argument("--disable-amesp", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    settings = OpenAICompatibleSettings.from_env()
    missing = settings.missing_fields()
    if missing:
        raise RuntimeError("Missing LLM settings: " + ", ".join(missing))
    cases = _load_cases(args.case_dir)
    excluded = set(args.exclude_case)
    cases = [case for case in cases if case["case_id"] not in excluded]
    _assign_public_case_ids(cases)
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in wanted]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise RuntimeError("No benchmark cases selected.")

    output_dir = args.output_dir.expanduser().resolve()
    raw_dir = output_dir / "raw"
    eval_dir = output_dir / "eval"
    raw_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    budget_limits = BudgetLimits(
        max_model_calls=args.max_model_calls,
        max_tool_calls=args.max_tool_calls,
        max_total_tokens=args.max_total_tokens,
        max_output_tokens_per_call=args.max_output_tokens_per_call,
    )
    subject = ChemGraphMASSubject(
        settings=settings,
        budget_limits=budget_limits,
        run_base_dir=output_dir / "chemgraph_runs",
        enable_amesp=not args.disable_amesp,
        recursion_limit=args.recursion_limit,
        schema_retries=args.schema_retries,
        use_responses_api=args.responses_api,
    )
    support_judge = LLMMechanismEvidenceSupportJudge(
        settings=replace(
            settings,
            reasoning_effort=None,
            max_tokens=900,
            temperature=0.0,
            top_p=1.0,
            seed=None,
        )
    )
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        raw_path = raw_dir / f"{case_id}_{subject.subject_id}_raw.json"
        eval_path = eval_dir / f"{case_id}.json"
        print(f"[{index}/{len(cases)}] {case_id}", flush=True)
        if eval_path.exists() and not args.refresh:
            row = json.loads(eval_path.read_text(encoding="utf-8"))
            if _cache_within_budget(row):
                rows.append(row)
                _write_summary(output_dir / "summary.json", rows, budget_limits)
                print("  cached " + _compact_score_text(row["scores"]), flush=True)
                continue
            print("  cached evaluation exceeded budget; regenerating", flush=True)

        raw = _run_or_load_raw(
            subject,
            case,
            raw_path=raw_path,
            refresh=args.refresh,
        )
        predictions = prediction_payload_from_structure_llm_mechanism_json(
            raw["raw_json"]
        )
        support = _judge_support_with_retries(
            predictions,
            judge=support_judge,
            label=case_id,
        )
        scores = score_mechanism_ranking(
            predictions,
            case["reference_mechanisms"],
            support_scores=support["support_scores"],
        )
        row = {
            "case_id": case_id,
            "public_case_id": case["public_case_id"],
            "subject_id": subject.subject_id,
            "subject_kind": subject.subject_kind,
            "model": subject.model,
            "prompt_version": raw["prompt_version"],
            "budget": raw["budget"],
            "tool_policy": raw["tool_policy"],
            "reference_mechanisms": case["reference_mechanisms"],
            "mechanism_predictions": predictions,
            "support_judgements": support["judgements"],
            "scores": scores,
            "raw_output_path": str(raw_path.resolve()),
        }
        rows.append(row)
        _write_json(eval_path, row)
        _write_summary(output_dir / "summary.json", rows, budget_limits)
        print("  " + _compact_score_text(scores), flush=True)

    print(f"Completed {len(rows)} cases: {output_dir}", flush=True)


def _load_cases(case_dir: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in sorted(case_dir.expanduser().resolve().glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        public_input = payload.get("public_input") or {}
        structure = ((public_input.get("molecule") or {}).get("structure") or {})
        hidden_reference = payload.get("hidden_reference") or {}
        reference_mechanisms = hidden_reference.get("reference_mechanisms")
        if not isinstance(reference_mechanisms, list):
            raise ValueError(f"Missing reference_mechanisms in {path}")
        task = str(public_input.get("task") or "").strip()
        cases.append(
            {
                "case_id": str(payload["case_id"]),
                "smiles": str(structure["value"]),
                "task": task
                or "Predict and rank likely AIE/photophysical mechanisms.",
                "reference_mechanisms": reference_mechanisms,
            }
        )
    return cases


def _assign_public_case_ids(cases: list[dict[str, Any]]) -> None:
    for index, case in enumerate(cases, start=1):
        case["public_case_id"] = anonymized_case_id(index)


def _run_or_load_raw(
    subject: ChemGraphMASSubject,
    case: dict[str, Any],
    *,
    raw_path: Path,
    refresh: bool,
    attempts: int = 3,
) -> dict[str, Any]:
    if raw_path.exists() and not refresh:
        cached = json.loads(raw_path.read_text(encoding="utf-8"))
        if _cache_within_budget(cached):
            return cached
        print("  cached raw output exceeded budget; regenerating", flush=True)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            raw = subject.run(
                case_id=str(case["case_id"]),
                public_case_id=str(case["public_case_id"]),
                smiles=str(case["smiles"]),
                task=str(case["task"]),
            )
            _write_json(raw_path, raw)
            return raw
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(
                f"  subject attempt {attempt}/{attempts} failed for "
                f"{case['case_id']}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            if attempt < attempts:
                time.sleep(2 * attempt)
    raise RuntimeError(
        f"Subject failed for {case['case_id']}: {last_error}"
    ) from last_error


def _cache_within_budget(payload: dict[str, Any]) -> bool:
    budget = payload.get("budget")
    usage = budget.get("usage") if isinstance(budget, dict) else None
    if not isinstance(usage, dict):
        return False
    total_tokens = usage.get("total_tokens")
    limits = budget.get("limits")
    max_total_tokens = (
        limits.get("max_total_tokens") if isinstance(limits, dict) else None
    )
    if not isinstance(total_tokens, int | float):
        return False
    if not isinstance(max_total_tokens, int | float):
        return False
    return (
        float(total_tokens) <= float(max_total_tokens)
        and float(usage.get("token_overrun") or 0) == 0.0
    )


def _judge_support_with_retries(
    predictions: list[dict[str, object]],
    *,
    judge: LLMMechanismEvidenceSupportJudge,
    label: str,
    attempts: int = 3,
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


def _write_summary(
    path: Path,
    rows: list[dict[str, Any]],
    budget_limits: BudgetLimits,
) -> None:
    _write_json(
        path,
        {
            "case_count": len(rows),
            "subject_id": rows[0]["subject_id"] if rows else None,
            "model": rows[0]["model"] if rows else None,
            "anonymized_public_case_ids": True,
            "budget_limits": {
                "max_model_calls": budget_limits.max_model_calls,
                "max_tool_calls": budget_limits.max_tool_calls,
                "max_total_tokens": budget_limits.max_total_tokens,
                "max_output_tokens_per_call": (
                    budget_limits.max_output_tokens_per_call
                ),
            },
            "averages": _averages(rows),
            "rows": rows,
        },
    )


def _averages(rows: list[dict[str, Any]]) -> dict[str, float | None]:
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
        metric: _mean(row["scores"].get(metric) for row in rows)
        for metric in metrics
    }


def _mean(values: Any) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, int | float)]
    return statistics.fmean(numeric) if numeric else None


def _compact_score_text(scores: dict[str, Any]) -> str:
    return (
        f"Top1-primary={scores['top1_primary']:.3f} "
        f"R@3={scores['recall_at_3']:.3f} "
        f"ES-nDCG@3={scores['es_ndcg_at_3']:.3f} "
        f"Support@Top1={scores['support_at_top1']:.3f}"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
