from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from mechcal.eval.mechanism_baselines import (
    MechanismBaselineTranscriptExtractor,
    prediction_payload_from_mechanism_extractor_json,
)
from mechcal.eval.mechanism_ranking import (
    LLMMechanismEvidenceSupportJudge,
    judge_mechanism_prediction_support,
    score_mechanism_ranking,
)
from mechcal.eval.schemas import EvalCase, SubjectRawOutput
from mechcal.eval.subjects import (
    EvidenceFormattedReactToolLLMFullBaselineSubject,
    ReactToolLLMFullBaselineSubject,
)
from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.public_ids import anonymized_case_id
from mechcal.runtime.llm import OpenAICompatibleSettings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate react_tool_llm_full on the slim mechanism-ranking benchmark."
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tool-llm"),
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--exclude-case", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--react-max-steps", type=int, default=30)
    parser.add_argument("--decision-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--finalization-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--disable-amesp", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--evidence-format",
        action="store_true",
        help=(
            "Run the format-control ReAct+tools baseline that requires "
            "Finding/Warrant/Boundary evidence text while keeping ReAct transcript-only state."
        ),
    )
    args = parser.parse_args()

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
    extracted_dir = output_dir / "extracted"
    eval_dir = output_dir / "eval"
    react_run_dir = output_dir / "react_runs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    subject = _make_subject(
        settings=settings,
        run_base_dir=react_run_dir,
        max_steps=args.react_max_steps,
        enable_amesp=not args.disable_amesp,
        decision_timeout_seconds=args.decision_timeout_seconds,
        finalization_timeout_seconds=args.finalization_timeout_seconds,
        evidence_format=args.evidence_format,
    )
    extractor = MechanismBaselineTranscriptExtractor(settings=settings)
    support_judge = LLMMechanismEvidenceSupportJudge(settings=settings)
    rows: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        eval_path = eval_dir / f"{case_id}.json"
        raw_path = raw_dir / f"{case_id}_{subject.subject_id}_raw.json"
        extracted_path = (
            extracted_dir / f"{case_id}_{subject.subject_id}_mechanism_extracted.json"
        )
        print(f"[{index}/{len(cases)}] {case_id}", flush=True)
        if eval_path.exists() and not args.refresh:
            row = json.loads(eval_path.read_text(encoding="utf-8"))
            if row.get("subject_id") == subject.subject_id:
                rows.append(row)
                _write_summary(output_dir / "summary.json", rows)
                print("  cached " + _compact_score_text(row["scores"]), flush=True)
                continue
        raw_output = _run_or_load_raw(
            subject,
            case,
            raw_path=raw_path,
            refresh=args.refresh,
        )
        extracted = _extract_or_load(
            extractor,
            raw_output,
            public_case_id=str(case["public_case_id"]),
            extracted_path=extracted_path,
            refresh=args.refresh,
        )
        predictions = prediction_payload_from_mechanism_extractor_json(extracted)
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
            "extractor_id": extractor.extractor_id,
            "extractor_model": extractor.model,
            "extractor_prompt_version": extractor.prompt_version,
            "react_max_steps": args.react_max_steps,
            "react_tool_ids": list(subject.tool_ids),
            "reference_mechanisms": case["reference_mechanisms"],
            "mechanism_predictions": predictions,
            "support_judgements": support["judgements"],
            "scores": scores,
            "raw_output_path": str(raw_path.resolve()),
            "extracted_output_path": str(extracted_path.resolve()),
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
        task = str((payload.get("public_input") or {}).get("task") or "").strip()
        mechanism_task = (
            task
            or "Predict and rank likely AIE/photophysical mechanisms from structure."
        )
        mechanism_task += (
            " Use this closed mechanism pool for the final mechanism diagnosis: "
            + ", ".join(MECHANISM_POOL)
            + "."
        )
        cases.append(
            {
                "case_id": str(payload["case_id"]),
                "smiles": str(structure["value"]),
                "task": mechanism_task,
                "reference_mechanisms": reference_mechanisms,
            }
        )
    return cases


def _assign_public_case_ids(cases: list[dict[str, object]]) -> None:
    for index, case in enumerate(cases, start=1):
        case["public_case_id"] = anonymized_case_id(index)


def _make_subject(
    *,
    settings: OpenAICompatibleSettings,
    run_base_dir: Path,
    max_steps: int,
    enable_amesp: bool,
    decision_timeout_seconds: float,
    finalization_timeout_seconds: float,
    evidence_format: bool,
) -> ReactToolLLMFullBaselineSubject:
    subject_cls = (
        EvidenceFormattedReactToolLLMFullBaselineSubject
        if evidence_format
        else ReactToolLLMFullBaselineSubject
    )
    return subject_cls(
        settings=settings,
        run_base_dir=run_base_dir,
        max_steps=max_steps,
        enable_amesp=enable_amesp,
        decision_timeout_seconds=decision_timeout_seconds,
        finalization_timeout_seconds=finalization_timeout_seconds,
    )


def _run_or_load_raw(
    subject: ReactToolLLMFullBaselineSubject,
    case: dict[str, object],
    *,
    raw_path: Path,
    refresh: bool,
) -> SubjectRawOutput:
    if raw_path.exists() and not refresh:
        return SubjectRawOutput.model_validate(
            json.loads(raw_path.read_text(encoding="utf-8"))
        )
    eval_case = EvalCase(
        case_id=str(case["case_id"]),
        public_case_id=str(case["public_case_id"]),
        smiles=str(case["smiles"]),
        user_query=str(case["task"]),
    )
    raw_output = subject.run(eval_case)
    _write_json(raw_path, raw_output.model_dump(mode="json"))
    return raw_output


def _extract_or_load(
    extractor: MechanismBaselineTranscriptExtractor,
    raw_output: SubjectRawOutput,
    *,
    public_case_id: str,
    extracted_path: Path,
    refresh: bool,
) -> dict[str, object]:
    if extracted_path.exists() and not refresh:
        return json.loads(extracted_path.read_text(encoding="utf-8"))
    extracted = extractor.extract(raw_output, public_case_id=public_case_id)
    _write_json(extracted_path, extracted)
    return extracted


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
    subject_id = rows[0]["subject_id"] if rows else None
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
