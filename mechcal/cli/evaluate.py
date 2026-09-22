from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from pathlib import Path

from mechcal.eval.mechanism_ranking import (
    MECHANISM_EVIDENCE_RUBRIC_PATH,
    MECHANISM_EVIDENCE_SUPPORT_PROMPT_VERSION,
    LLMMechanismEvidenceSupportJudge,
    judge_mechanism_prediction_support,
    score_mechanism_ranking,
)
from mechcal.public_ids import anonymized_case_id
from mechcal.runtime.ablations import ABLATION_MODES, AblationMode
from mechcal.runtime.llm import OpenAICompatibleSettings
from mechcal.runtime.orchestrator import MechCALOrchestrator, OrchestratorConfig
from mechcal.runtime.output_assembler import OutputAssembler
from mechcal.schemas import CaseRun


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MAS on the slim mechanism-ranking benchmark."
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mechcal"),
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--exclude-case", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=30)
    parser.add_argument(
        "--ablation-mode",
        choices=ABLATION_MODES,
        default="full_mechcal",
        help="Run Full MechCAL or one registered component ablation.",
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--public-case-id-map",
        type=Path,
        default=None,
        help=(
            "Optional JSON map from internal case_id to anonymized public case id. "
            "Use this for sharded runs so parallel scheduling does not change "
            "the input metadata seen by MAS agents."
        ),
    )
    parser.add_argument("--no-amesp", action="store_true")
    parser.add_argument(
        "--disable-support-auditor",
        action="store_true",
        help=(
            "Ablation flag: skip the internal PhotophysicsArbiterAgent "
            "SupportAuditor while keeping MechanismCritic and the external "
            "LLM-as-judge support scoring enabled."
        ),
    )
    parser.add_argument(
        "--disable-agenda-coverage-reviewer",
        action="store_true",
        help=(
            "Ablation flag: skip the internal Agenda & Coverage Reviewer "
            "coverage memo while keeping Planner, workers, SupportAuditor, "
            "MechanismCritic, and external scoring enabled."
        ),
    )
    portfolio_group = parser.add_mutually_exclusive_group()
    portfolio_group.add_argument(
        "--enable-incremental-portfolio",
        dest="enable_incremental_portfolio",
        action="store_true",
        help=(
            "Legacy comparison only: expose previous portfolio state to Planner "
            "updates. The current formal MechCAL default is stateless per-round "
            "portfolio synthesis."
        ),
    )
    portfolio_group.add_argument(
        "--disable-incremental-portfolio",
        dest="enable_incremental_portfolio",
        action="store_false",
        help=(
            "Ablation flag: hide previous mechanism portfolio state from Planner "
            "update/calibration calls, forcing stateless per-round ranking from "
            "public evidence while keeping the rest of MechCAL enabled."
        ),
    )
    parser.set_defaults(enable_incremental_portfolio=False)
    parser.add_argument("--enable-support-auditor-web-search", action="store_true")
    parser.add_argument(
        "--fail-on-arbiter-error",
        action="store_true",
        help=(
            "Abort a MAS run when the LLM photophysics verifier times out or "
            "fails. By default the verifier failure is logged and the run "
            "continues without applying a synthetic audit."
        ),
    )
    parser.add_argument(
        "--fail-on-planner-timeout",
        action="store_true",
        help=(
            "Abort a MAS run when a Planner update round has a transient "
            "LLM/API timeout. By default the failure is logged and the run "
            "stops from the last persisted Planner state without a scientific fallback."
        ),
    )
    parser.add_argument(
        "--verifier-failure-stop-threshold",
        type=int,
        default=0,
        help=(
            "Stop a MAS case after this many logged PhotophysicsArbiterAgent "
            "failures. Use 0 to disable this resource guard."
        ),
    )
    args = parser.parse_args()

    settings = OpenAICompatibleSettings.from_env()
    missing = settings.missing_fields()
    if missing:
        raise RuntimeError("Missing LLM settings: " + ", ".join(missing))

    output_dir = args.output_dir.expanduser().resolve()
    run_base_dir = output_dir / "mas_runs"
    eval_dir = output_dir / "mas_eval"
    run_base_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    cases = _load_cases(args.case_dir)
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in wanted]
    excluded = set(args.exclude_case)
    cases = [case for case in cases if case["case_id"] not in excluded]
    if args.limit is not None:
        cases = cases[: args.limit]
    _assign_public_case_ids(cases, map_path=args.public_case_id_map)

    support_judge = LLMMechanismEvidenceSupportJudge(settings=settings)
    _write_json(
        output_dir / "experiment_manifest.json",
        {
            "ablation_mode": args.ablation_mode,
            "case_dir": str(args.case_dir.expanduser().resolve()),
            "selected_case_count": len(cases),
            "max_rounds": args.max_rounds,
            "subject_model": settings.model,
            "support_judge_model": support_judge.model,
            "support_judge_prompt_version": MECHANISM_EVIDENCE_SUPPORT_PROMPT_VERSION,
            "rubric_path": str(MECHANISM_EVIDENCE_RUBRIC_PATH.resolve()),
            "rubric_version": support_judge.rubric.get(
                "mechanism_evidence_rubric_version"
            ),
            "incremental_mechanism_portfolio": args.enable_incremental_portfolio,
            "smiles_only": True,
            "hidden_reference_available_to_subject": False,
        },
    )
    rows: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        print(f"[{index}/{len(cases)}] {case_id}", flush=True)
        case_run = _run_or_load_mas(
            case,
            output_dir=output_dir,
            run_base_dir=run_base_dir,
            max_rounds=args.max_rounds,
            ablation_mode=args.ablation_mode,
            enable_amesp=not args.no_amesp,
            enable_support_auditor=not args.disable_support_auditor,
            enable_agenda_coverage_reviewer=not args.disable_agenda_coverage_reviewer,
            enable_incremental_mechanism_portfolio=args.enable_incremental_portfolio,
            enable_support_auditor_web_search=args.enable_support_auditor_web_search,
            continue_on_arbiter_failure=not args.fail_on_arbiter_error,
            continue_on_planner_update_failure=not args.fail_on_planner_timeout,
            verifier_failure_stop_threshold=args.verifier_failure_stop_threshold,
            refresh=args.refresh,
        )
        predictions = _prediction_payload(case_run)
        evaluation_status = "scored"
        evaluation_failure = None
        if case_run.status == "failed":
            evaluation_status = "mas_failed"
            evaluation_failure = case_run.runtime.get("failure")
            support = {"support_scores": {}, "judgements": []}
            scores = _unscored_metrics()
        else:
            try:
                support = _judge_support_with_retries(
                    predictions,
                    judge=support_judge,
                    label=case_id,
                )
                scores = score_mechanism_ranking(
                    predictions,
                    case["reference_mechanisms"],  # type: ignore[arg-type]
                    support_scores=support["support_scores"],  # type: ignore[arg-type]
                )
            except Exception as exc:  # noqa: BLE001
                evaluation_status = "judge_failed"
                evaluation_failure = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                support = {"support_scores": {}, "judgements": []}
                scores = _unscored_metrics()
        row = {
            "case_id": case_id,
            "status": case_run.status,
            "ablation_mode": (case_run.runtime.get("ablation") or {}).get(
                "mode",
                args.ablation_mode,
            ),
            "evaluation_status": evaluation_status,
            "evaluation_failure": evaluation_failure,
            "round_count": len(case_run.round_ids),
            "verifier_failure_count": len(
                case_run.runtime.get("verifier_failures", [])
            ),
            "planner_failure_count": len(case_run.runtime.get("planner_failures", [])),
            "mechanism_critic_failure_count": len(
                case_run.runtime.get("mechanism_critic_failures", [])
            ),
            "support_auditor_enabled": bool(
                (case_run.runtime.get("support_auditor") or {}).get("enabled", True)
            ),
            "support_auditor_skipped_round_count": len(
                case_run.runtime.get("support_auditor_skipped_rounds", [])
            ),
            "agenda_coverage_reviewer_enabled": bool(
                (case_run.runtime.get("agenda_coverage_reviewer") or {}).get(
                    "enabled",
                    True,
                )
            ),
            "incremental_mechanism_portfolio_enabled": bool(
                (case_run.runtime.get("incremental_mechanism_portfolio") or {}).get(
                    "enabled",
                    True,
                )
            ),
            "runtime_failure": case_run.runtime.get("failure"),
            "worker_call_counts": case_run.runtime.get("worker_call_counts", {}),
            "termination": case_run.runtime.get("termination"),
            "subject_model": settings.model,
            "support_judge_model": support_judge.model,
            "rubric_version": support_judge.rubric.get(
                "mechanism_evidence_rubric_version"
            ),
            "reference_mechanisms": case["reference_mechanisms"],
            "mechanism_predictions": predictions,
            "support_judgements": support["judgements"],
            "scores": scores,
            "case_run_path": str((run_base_dir / case_id / "case_run.json").resolve()),
        }
        rows.append(row)
        _write_json(eval_dir / f"{case_id}.json", row)
        _write_summary(output_dir / "summary.json", rows)
        print(
            "  "
            + _compact_score_text(scores)
            + f"; status={case_run.status}; rounds={len(case_run.round_ids)}"
            + f"; verifier_failures={row['verifier_failure_count']}"
            + f"; planner_failures={row['planner_failure_count']}"
            + f"; mechanism_critic_failures={row['mechanism_critic_failure_count']}"
            + f"; support_auditor_enabled={row['support_auditor_enabled']}"
            + f"; agenda_coverage_enabled={row['agenda_coverage_reviewer_enabled']}"
            + "; incremental_portfolio_enabled="
            + f"{row['incremental_mechanism_portfolio_enabled']}"
            + f"; ablation_mode={row['ablation_mode']}"
            + f"; evaluation_status={evaluation_status}",
            flush=True,
        )
        if evaluation_failure is not None:
            print(
                f"    evaluation_failure={evaluation_failure['type']}: "
                f"{evaluation_failure['message']}",
                flush=True,
            )
        for prediction in predictions[:3]:
            print(
                f"    rank {prediction['rank']}: {prediction['label']} "
                f"conf={prediction['confidence']} refs={prediction['evidence_refs']}",
                flush=True,
            )


def _load_cases(case_dir: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for path in sorted(case_dir.expanduser().resolve().glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        structure = ((payload.get("public_input") or {}).get("molecule") or {}).get(
            "structure"
        ) or {}
        hidden_reference = payload.get("hidden_reference") or {}
        reference_mechanisms = hidden_reference.get("reference_mechanisms")
        if not isinstance(reference_mechanisms, list):
            raise ValueError(f"Missing reference_mechanisms in {path}")
        cases.append(
            {
                "case_id": str(payload["case_id"]),
                "smiles": str(structure["value"]),
                "task": str((payload.get("public_input") or {}).get("task") or ""),
                "reference_mechanisms": reference_mechanisms,
            }
        )
    return cases


def _assign_public_case_ids(
    cases: list[dict[str, object]], *, map_path: Path | None
) -> None:
    if map_path is not None:
        public_case_id_map = _load_public_case_id_map(map_path)
        missing = [
            str(case["case_id"])
            for case in cases
            if str(case["case_id"]) not in public_case_id_map
        ]
        if missing:
            raise ValueError(
                "Public case id map is missing case ids: " + ", ".join(missing)
            )
        for case in cases:
            case["public_case_id"] = public_case_id_map[str(case["case_id"])]
        return
    for index, case in enumerate(cases, start=1):
        case["public_case_id"] = anonymized_case_id(index)


def _load_public_case_id_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("public_case_id_map"), dict):
        payload = payload["public_case_id_map"]
    if not isinstance(payload, dict):
        raise ValueError(f"Public case id map must be a JSON object: {path}")
    public_case_id_map = {str(key): str(value) for key, value in payload.items()}
    if not all(public_case_id_map.values()):
        raise ValueError(f"Public case id map contains empty values: {path}")
    return public_case_id_map


def _run_or_load_mas(
    case: dict[str, object],
    *,
    output_dir: Path,
    run_base_dir: Path,
    max_rounds: int,
    ablation_mode: AblationMode,
    enable_amesp: bool,
    enable_support_auditor: bool,
    enable_agenda_coverage_reviewer: bool,
    enable_incremental_mechanism_portfolio: bool,
    enable_support_auditor_web_search: bool,
    continue_on_arbiter_failure: bool,
    continue_on_planner_update_failure: bool,
    verifier_failure_stop_threshold: int,
    refresh: bool,
) -> CaseRun:
    case_id = str(case["case_id"])
    case_run_path = run_base_dir / case_id / "case_run.json"
    if refresh and (run_base_dir / case_id).exists():
        shutil.rmtree(run_base_dir / case_id)
    if case_run_path.exists() and not refresh:
        case_run = CaseRun.model_validate_json(case_run_path.read_text(encoding="utf-8"))
        if case_run.status in {"finalized", "stopped", "failed"}:
            existing_mode = (case_run.runtime.get("ablation") or {}).get("mode")
            if existing_mode != ablation_mode:
                raise RuntimeError(
                    "Existing case run uses ablation_mode="
                    f"{existing_mode!r}, requested {ablation_mode!r}. "
                    "Use a distinct output directory or --refresh."
                )
            return case_run
    try:
        return MechCALOrchestrator(
            OrchestratorConfig(
                run_base_dir=run_base_dir,
                max_rounds=max_rounds,
                ablation_mode=ablation_mode,
                enable_llm_planner=True,
                enable_llm_worker_planning=False,
                enable_llm_worker_reports=False,
                enable_amesp=enable_amesp,
                enable_legacy_conclusion_ledger=False,
                enable_support_auditor=enable_support_auditor,
                enable_agenda_coverage_reviewer=enable_agenda_coverage_reviewer,
                enable_incremental_mechanism_portfolio=(
                    enable_incremental_mechanism_portfolio
                ),
                enable_support_auditor_web_search=enable_support_auditor_web_search,
                continue_on_arbiter_failure=continue_on_arbiter_failure,
                continue_on_planner_update_failure=continue_on_planner_update_failure,
                verifier_failure_stop_threshold=verifier_failure_stop_threshold,
                defer_review_until_portfolio=True,
            )
        ).run(
            smiles=str(case["smiles"]),
            user_query=str(case["task"]) or "Predict likely AIE mechanisms.",
            case_id=case_id,
            public_case_id=str(case.get("public_case_id") or ""),
        )
    except Exception:
        if case_run_path.exists():
            return CaseRun.model_validate_json(case_run_path.read_text(encoding="utf-8"))
        raise


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
                "low_margin_group": prediction.low_margin_group,
                "margin_to_next": prediction.margin_to_next,
                "ranking_stability_note": prediction.ranking_stability_note,
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


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
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
    averages = {
        metric: _mean(
            row["scores"].get(metric)  # type: ignore[union-attr]
            for row in rows
        )
        for metric in metrics
    }
    _write_json(
        path,
        {
            "case_count": len(rows),
            "scored_case_count": sum(
                1 for row in rows if row.get("evaluation_status", "scored") == "scored"
            ),
            "unscored_case_count": sum(
                1 for row in rows if row.get("evaluation_status", "scored") != "scored"
            ),
            "averages": averages,
            "average_verifier_failure_count": _mean(
                row.get("verifier_failure_count") for row in rows
            ),
            "average_planner_failure_count": _mean(
                row.get("planner_failure_count") for row in rows
            ),
            "average_mechanism_critic_failure_count": _mean(
                row.get("mechanism_critic_failure_count") for row in rows
            ),
            "rows": rows,
        },
    )


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


if __name__ == "__main__":
    main()
