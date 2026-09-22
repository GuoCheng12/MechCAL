from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
from pathlib import Path

from mechcal.eval.mechanism_baselines import (
    MECHANISM_STRUCTURE_LLM_CHEM_R_TAGGED_PROMPT,
    MECHANISM_STRUCTURE_LLM_CHEM_R_TAGGED_PROMPT_VERSION,
    MECHANISM_STRUCTURE_LLM_EVIDENCE_FORMATTED_PROMPT,
    MECHANISM_STRUCTURE_LLM_EVIDENCE_FORMATTED_PROMPT_VERSION,
    StructureLLMMechanismBaseline,
    prediction_payload_from_structure_llm_mechanism_json,
)
from mechcal.eval.mechanism_ranking import (
    LLMMechanismEvidenceSupportJudge,
    judge_mechanism_prediction_support,
    score_mechanism_ranking,
)
from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.public_ids import anonymized_case_id
from mechcal.runtime.llm import OpenAICompatibleSettings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate structure_llm on the slim mechanism-ranking benchmark."
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/direct"),
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--exclude-case", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for generate-only subject calls.",
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--evidence-format",
        action="store_true",
        help=(
            "Run the format-control structure-only baseline that requires "
            "Finding/Warrant/Boundary evidence text but still uses no tools."
        ),
    )
    parser.add_argument("--subject-id", default=None)
    parser.add_argument("--subject-kind", default=None)
    parser.add_argument(
        "--strict-json-schema",
        action="store_true",
        help="Require exactly three schema-valid mechanism predictions.",
    )
    parser.add_argument(
        "--native-think-answer",
        action="store_true",
        help=(
            "Preserve the subject model's <think>/<answer> protocol and require "
            "the final JSON inside <answer>."
        ),
    )
    parser.add_argument(
        "--native-thinking-json",
        action="store_true",
        help=(
            "Keep the subject model's native thinking mode and parse the final "
            "JSON after its closing think tag."
        ),
    )
    parser.add_argument(
        "--intern-api-thinking",
        action="store_true",
        help=(
            "Use the official InternLM API thinking_mode field with "
            "--native-thinking-json."
        ),
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate and cache subject raw JSON without external support judging.",
    )
    parser.add_argument(
        "--require-cached-raw",
        action="store_true",
        help="Fail instead of calling the subject model when a raw cache is missing.",
    )
    parser.add_argument(
        "--score-missing-raw-as-zero",
        action="store_true",
        help=(
            "When cached subject generation is required, score a missing or "
            "invalid raw output as an empty prediction instead of resampling."
        ),
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.workers > 1 and not args.generate_only:
        parser.error("--workers greater than 1 is supported only with --generate-only")

    settings = OpenAICompatibleSettings.from_env()
    missing = settings.missing_fields()
    if missing:
        raise RuntimeError("Missing LLM settings: " + ", ".join(missing))

    cases = _load_cases(args.case_dir)
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in wanted]
    excluded = set(args.exclude_case)
    cases = [case for case in cases if case["case_id"] not in excluded]
    if args.limit is not None:
        cases = cases[: args.limit]
    _assign_public_case_ids(cases)

    output_dir = args.output_dir.expanduser().resolve()
    raw_dir = output_dir / "raw"
    eval_dir = output_dir / "eval"
    raw_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    subject = _make_subject(
        settings=settings,
        evidence_format=args.evidence_format,
        subject_id=args.subject_id,
        subject_kind=args.subject_kind,
        strict_json_schema=args.strict_json_schema,
        native_think_answer=args.native_think_answer,
        native_thinking_json=args.native_thinking_json,
        intern_api_thinking=args.intern_api_thinking,
    )
    support_judge = (
        None if args.generate_only else LLMMechanismEvidenceSupportJudge(settings=settings)
    )
    if args.generate_only and args.workers > 1:
        _generate_raw_cases_parallel(
            subject=subject,
            cases=cases,
            raw_dir=raw_dir,
            workers=args.workers,
            refresh=args.refresh,
            required_predictions=(
                3
                if args.strict_json_schema
                or args.native_think_answer
                or args.native_thinking_json
                else 1
            ),
            require_evidence_headings=(
                args.native_think_answer or args.native_thinking_json
            ),
            require_cached=args.require_cached_raw,
        )
        return
    rows: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        print(f"[{index}/{len(cases)}] {case_id}", flush=True)
        raw_path = raw_dir / f"{case_id}_{subject.subject_id}_raw.json"
        try:
            raw_record = _run_or_load_raw(
                subject,
                case,
                raw_path=raw_path,
                refresh=args.refresh,
                required_predictions=(
                    3
                    if args.strict_json_schema
                    or args.native_think_answer
                    or args.native_thinking_json
                    else 1
                ),
                require_evidence_headings=(
                    args.native_think_answer or args.native_thinking_json
                ),
                require_cached=args.require_cached_raw,
            )
        except RuntimeError as exc:
            if not (
                args.require_cached_raw and args.score_missing_raw_as_zero
            ):
                raise
            row = _format_failure_row(
                case=case,
                subject=subject,
                reason=str(exc),
                raw_path=raw_path,
            )
            rows.append(row)
            _write_json(eval_dir / f"{case_id}.json", row)
            _write_summary(output_dir / "summary.json", rows)
            print("  format failure scored as zero", flush=True)
            continue
        if args.generate_only:
            predictions = prediction_payload_from_structure_llm_mechanism_json(
                _raw_json(raw_record)
            )
            print(
                f"  cached {len(predictions)} schema-valid predictions",
                flush=True,
            )
            continue
        predictions = prediction_payload_from_structure_llm_mechanism_json(
            _raw_json(raw_record)
        )
        assert support_judge is not None
        eval_path = eval_dir / f"{case_id}.json"
        cached_row = _load_cached_eval(
            eval_path,
            refresh=args.refresh,
            case_id=case_id,
            subject=subject,
            judge_model=support_judge.model,
        )
        if cached_row is not None:
            rows.append(cached_row)
            _write_summary(output_dir / "summary.json", rows)
            print(
                "  cached judge result " + _compact_score_text(cached_row["scores"]),
                flush=True,
            )
            continue
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
        row = {
            "case_id": case_id,
            "public_case_id": case["public_case_id"],
            "subject_id": subject.subject_id,
            "subject_kind": subject.subject_kind,
            "model": subject.model,
            "prompt_version": subject.prompt_version,
            "reference_mechanisms": case["reference_mechanisms"],
            "mechanism_predictions": predictions,
            "support_judgements": support["judgements"],
            "scores": scores,
            "raw_output_path": str(
                (raw_dir / f"{case_id}_{subject.subject_id}_raw.json").resolve()
            ),
        }
        rows.append(row)
        _write_json(eval_path, row)
        _write_summary(output_dir / "summary.json", rows)
        print("  " + _compact_score_text(scores), flush=True)
        for prediction in predictions[:3]:
            print(
                f"    rank {prediction['rank']}: {prediction['label']} "
                f"conf={prediction['confidence']}",
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


def _assign_public_case_ids(cases: list[dict[str, object]]) -> None:
    for index, case in enumerate(cases, start=1):
        case["public_case_id"] = anonymized_case_id(index)


def _generate_raw_cases_parallel(
    *,
    subject: StructureLLMMechanismBaseline,
    cases: list[dict[str, object]],
    raw_dir: Path,
    workers: int,
    refresh: bool,
    required_predictions: int,
    require_evidence_headings: bool,
    require_cached: bool,
) -> None:
    def generate(index_and_case: tuple[int, dict[str, object]]) -> str:
        index, case = index_and_case
        case_id = str(case["case_id"])
        print(f"[{index}/{len(cases)}] {case_id}", flush=True)
        raw_record = _run_or_load_raw(
            subject,
            case,
            raw_path=raw_dir / f"{case_id}_{subject.subject_id}_raw.json",
            refresh=refresh,
            required_predictions=required_predictions,
            require_evidence_headings=require_evidence_headings,
            require_cached=require_cached,
        )
        predictions = prediction_payload_from_structure_llm_mechanism_json(
            _raw_json(raw_record)
        )
        print(
            f"  {case_id}: cached {len(predictions)} schema-valid predictions",
            flush=True,
        )
        return case_id

    indexed_cases = list(enumerate(cases, start=1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(generate, item) for item in indexed_cases]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def _format_failure_row(
    *,
    case: dict[str, object],
    subject: StructureLLMMechanismBaseline,
    reason: str,
    raw_path: Path,
) -> dict[str, object]:
    predictions: list[dict[str, object]] = []
    scores = score_mechanism_ranking(
        predictions,
        case["reference_mechanisms"],  # type: ignore[arg-type]
        support_scores=[],
    )
    attempt_paths = sorted(
        str(path.resolve())
        for path in (raw_path.parent / "attempts").glob(
            f"{raw_path.stem}_attempt_*.json"
        )
    )
    return {
        "case_id": str(case["case_id"]),
        "public_case_id": str(case["public_case_id"]),
        "subject_id": subject.subject_id,
        "subject_kind": subject.subject_kind,
        "model": subject.model,
        "prompt_version": subject.prompt_version,
        "reference_mechanisms": case["reference_mechanisms"],
        "mechanism_predictions": predictions,
        "support_judgements": [],
        "scores": scores,
        "raw_output_path": None,
        "format_failure": True,
        "format_failure_reason": reason,
        "format_failure_attempt_paths": attempt_paths,
    }


def _run_or_load_raw(
    subject: StructureLLMMechanismBaseline,
    case: dict[str, object],
    *,
    raw_path: Path,
    refresh: bool,
    required_predictions: int = 1,
    require_evidence_headings: bool = False,
    attempts: int = 3,
    require_cached: bool = False,
) -> dict[str, object]:
    if raw_path.exists() and not refresh:
        cached = json.loads(raw_path.read_text(encoding="utf-8"))
        if _has_mechanism_predictions(
            cached,
            min_count=required_predictions,
            require_evidence_headings=require_evidence_headings,
        ):
            return cached
        print(
            f"  cached structure_llm raw is empty or invalid for {case['case_id']}; "
            "regenerating",
            flush=True,
        )
    if require_cached:
        raise RuntimeError(
            f"Required valid raw cache is missing for {case['case_id']}: {raw_path}"
        )
    last_error: Exception | None = None
    previous_failure = ""
    for attempt in range(1, attempts + 1):
        try:
            attempt_case = dict(case)
            if attempt > 1:
                attempt_case["_bounded_output"] = True
                attempt_case["task"] = (
                    f"{case.get('task', '')}\n"
                    f"Schema retry {attempt} of {attempts}. The previous response "
                    f"was rejected because {previous_failure} "
                    "Select exactly three different labels. Before writing the "
                    "answer, verify that the label set has size three. Write "
                    "case-specific evidence for each selected mechanism."
                )
            raw_record = subject.run(attempt_case)
            attempt_path = (
                raw_path.parent
                / "attempts"
                / f"{raw_path.stem}_attempt_{attempt}.json"
            )
            _write_json(attempt_path, raw_record)
            if not _has_mechanism_predictions(
                raw_record,
                min_count=required_predictions,
                require_evidence_headings=require_evidence_headings,
            ):
                previous_failure = _raw_validation_feedback(
                    raw_record,
                    require_evidence_headings=require_evidence_headings,
                )
                raise ValueError(
                    "structure_llm raw JSON has fewer than "
                    f"{required_predictions} valid mechanism predictions: "
                    f"{previous_failure}"
                )
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(
                f"  raw structure_llm attempt {attempt}/{attempts} failed for "
                f"{case['case_id']}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            if attempt < attempts:
                time.sleep(2 * attempt)
    else:
        raise RuntimeError(
            f"structure_llm raw generation failed for {case['case_id']}: {last_error}"
        ) from last_error
    _write_json(raw_path, raw_record)
    return raw_record


def _raw_validation_feedback(
    raw_record: dict[str, object],
    *,
    require_evidence_headings: bool = False,
) -> str:
    raw_json = raw_record.get("raw_json")
    rows = raw_json.get("mechanism_predictions") if isinstance(raw_json, dict) else None
    labels = [
        str(row.get("label") or "")
        for row in rows
        if isinstance(row, dict) and row.get("label")
    ] if isinstance(rows, list) else []
    if labels and len(set(labels)) < len(labels):
        return (
            f"the labels {labels!r} contain duplicates; each label may appear once."
        )
    if require_evidence_headings and isinstance(rows, list):
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            evidence = row.get("evidence")
            joined = " ".join(
                str(item) for item in evidence
            ) if isinstance(evidence, list) else ""
            missing = [
                heading
                for heading in ("Finding:", "Warrant:", "Boundary:")
                if heading not in joined
            ]
            if missing:
                return (
                    f"prediction {index} evidence is missing "
                    + ", ".join(missing)
                    + "."
                )
    parse_error = raw_record.get("parse_error")
    if parse_error:
        return f"the tagged answer could not be parsed ({parse_error})."
    finish_reason = raw_record.get("finish_reason")
    if finish_reason and finish_reason != "stop":
        return f"generation ended with finish_reason={finish_reason!r}."
    return "the final JSON did not satisfy the required three-prediction schema."


def _load_cached_eval(
    path: Path,
    *,
    refresh: bool,
    case_id: str,
    subject: StructureLLMMechanismBaseline,
    judge_model: str,
) -> dict[str, object] | None:
    if refresh or not path.exists():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(row, dict):
        return None
    if row.get("case_id") != case_id:
        return None
    if row.get("subject_id") != subject.subject_id:
        return None
    if row.get("model") != subject.model:
        return None
    if row.get("prompt_version") != subject.prompt_version:
        return None
    predictions = row.get("mechanism_predictions")
    judgements = row.get("support_judgements")
    scores = row.get("scores")
    if not isinstance(predictions, list) or len(predictions) != 3:
        return None
    if not isinstance(judgements, list) or not judgements:
        return None
    if any(
        not isinstance(item, dict) or item.get("judge_model") != judge_model
        for item in judgements
    ):
        return None
    if not isinstance(scores, dict) or scores.get("es_ndcg_at_3") is None:
        return None
    return row


def _make_subject(
    *,
    settings: OpenAICompatibleSettings,
    evidence_format: bool,
    subject_id: str | None = None,
    subject_kind: str | None = None,
    strict_json_schema: bool = False,
    native_think_answer: bool = False,
    native_thinking_json: bool = False,
    intern_api_thinking: bool = False,
) -> StructureLLMMechanismBaseline:
    if native_think_answer:
        return StructureLLMMechanismBaseline(
            settings=settings,
            subject_id=subject_id or "chem_r_8b_direct",
            subject_kind=subject_kind or "structure_llm",
            prompt_name=MECHANISM_STRUCTURE_LLM_CHEM_R_TAGGED_PROMPT,
            prompt_version=MECHANISM_STRUCTURE_LLM_CHEM_R_TAGGED_PROMPT_VERSION,
            strict_json_schema=True,
            native_think_answer=True,
        )
    if native_thinking_json:
        return StructureLLMMechanismBaseline(
            settings=settings,
            subject_id=subject_id or "native_thinking_structure_llm",
            subject_kind=subject_kind or "structure_llm",
            prompt_name=MECHANISM_STRUCTURE_LLM_EVIDENCE_FORMATTED_PROMPT,
            prompt_version=MECHANISM_STRUCTURE_LLM_EVIDENCE_FORMATTED_PROMPT_VERSION,
            strict_json_schema=True,
            native_thinking_json=True,
            intern_api_thinking=intern_api_thinking,
        )
    if not evidence_format:
        return StructureLLMMechanismBaseline(
            settings=settings,
            subject_id=subject_id,
            subject_kind=subject_kind,
            strict_json_schema=strict_json_schema,
        )
    return StructureLLMMechanismBaseline(
        settings=settings,
        subject_id=subject_id or "evidence_formatted_structure_llm",
        subject_kind=subject_kind or "evidence_formatted_structure_llm",
        prompt_name=MECHANISM_STRUCTURE_LLM_EVIDENCE_FORMATTED_PROMPT,
        prompt_version=MECHANISM_STRUCTURE_LLM_EVIDENCE_FORMATTED_PROMPT_VERSION,
        strict_json_schema=strict_json_schema,
    )


def _has_mechanism_predictions(
    raw_record: dict[str, object],
    *,
    min_count: int = 1,
    require_evidence_headings: bool = False,
) -> bool:
    finish_reason = raw_record.get("finish_reason")
    if finish_reason and finish_reason != "stop":
        return False
    raw_json = raw_record.get("raw_json")
    if not isinstance(raw_json, dict):
        return False
    predictions = raw_json.get("mechanism_predictions")
    if not isinstance(predictions, list):
        return False
    valid_predictions = [item for item in predictions if isinstance(item, dict)]
    if len(valid_predictions) < min_count:
        return False
    if min_count == 3:
        if len(valid_predictions) != 3:
            return False
        labels = [str(item.get("label") or "") for item in valid_predictions]
        if any(label not in MECHANISM_POOL for label in labels):
            return False
        if len(set(labels)) != 3:
            return False
        if [item.get("rank") for item in valid_predictions] != [1, 2, 3]:
            return False
        for item in valid_predictions:
            evidence = item.get("evidence")
            limitations = item.get("limitations")
            if not isinstance(evidence, list) or not any(
                isinstance(text, str) and text.strip() for text in evidence
            ):
                return False
            if not isinstance(limitations, list) or not any(
                isinstance(text, str) and text.strip() for text in limitations
            ):
                return False
            joined_evidence = " ".join(
                text for text in evidence if isinstance(text, str)
            )
            if require_evidence_headings and not all(
                heading in joined_evidence
                for heading in ("Finding:", "Warrant:", "Boundary:")
            ):
                return False
            if (
                "case-specific smiles-level structural observation"
                in joined_evidence.lower()
            ):
                return False
    return True


def _raw_json(raw_record: dict[str, object]) -> dict[str, object]:
    raw_json = raw_record.get("raw_json")
    if not isinstance(raw_json, dict):
        raise ValueError("raw structure_llm record must contain raw_json object.")
    return raw_json


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


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    subject_id = str(rows[0]["subject_id"]) if rows else "structure_llm"
    _write_json(
        path,
        {
            "case_count": len(rows),
            "subject_id": subject_id,
            "anonymized_public_case_ids": True,
            "averages": _averages(rows),
            "rows": rows,
        },
    )


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
        f"Support@Top1={scores['support_at_top1']:.3f} "
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
