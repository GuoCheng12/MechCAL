from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import typer

from mechcal.eval.baseline_adapter import BaselineCommonNarrativeAdapter
from mechcal.eval.metrics import LLMSemanticAlignmentJudge, score_prediction
from mechcal.eval.schemas import (
    EvalCase,
    EvalRunRecord,
    NormalizedPrediction,
    SubjectRawOutput,
)
from mechcal.eval.subjects import (
    EvidenceFormattedReactToolLLMFullBaselineSubject,
    ReactToolLLMBaselineSubject,
    ReactToolLLMFullBaselineSubject,
    StructureLLMBaselineSubject,
    ZeroShotLLMBaselineSubject,
)

app = typer.Typer(help="Run normalized MechCAL baseline evaluation subjects.")

OUTPUT_DIR_OPTION = typer.Option(
    Path("outputs/evaluation"),
    "--output-dir",
    help="Directory for normalized prediction records.",
)
REFERENCE_JSON_OPTION = typer.Option(
    None,
    "--reference-json",
    help=(
        "Optional hidden reference JSON with semantic_evidence_targets and "
        "semantic_diagnosis_targets."
    ),
)


def run_eval_case(
    *,
    case: EvalCase,
    subject: Literal[
        "zero-shot-llm",
        "structure-llm",
        "react-tool-llm",
        "react-tool-llm-full",
        "evidence-formatted-react-tool-llm-full",
    ],
    output_dir: Path,
    use_judge: bool = True,
) -> EvalRunRecord:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_runner = _subject_runner(subject)
    subject_result = subject_runner.run(case)
    prediction = _normalize_subject_result(case, subject_result)
    judge = LLMSemanticAlignmentJudge() if use_judge and case.hidden_reference else None
    metrics = score_prediction(prediction, case.hidden_reference, judge=judge)
    record = EvalRunRecord(
        run_id=f"{case.case_id}:{prediction.subject_id}",
        case=case,
        prediction=prediction,
        metrics=metrics,
    )
    _append_jsonl(output_dir / "normalized_predictions.jsonl", prediction.model_dump(mode="json"))
    _append_jsonl(output_dir / "metrics.jsonl", record.model_dump(mode="json"))
    (output_dir / f"{case.case_id}.json").write_text(
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_summary(output_dir / "summary.json", record)
    return record


@app.command()
def run(
    smiles: str = typer.Option(..., "--smiles", help="Input molecule SMILES."),
    case_id: str = typer.Option("case_0001", "--case-id", help="Stable case id."),
    user_query: str = typer.Option(
        "Assess the likely AIE mechanism for this molecule.",
        "--user-query",
        help="Question passed to the baseline subject.",
    ),
    subject: Literal[
        "zero-shot-llm",
        "structure-llm",
        "react-tool-llm",
        "react-tool-llm-full",
        "evidence-formatted-react-tool-llm-full",
    ] = typer.Option(
        "zero-shot-llm",
        "--subject",
        help="Baseline subject to run.",
    ),
    output_dir: Path = OUTPUT_DIR_OPTION,
    reference_json: Path | None = REFERENCE_JSON_OPTION,
    use_judge: bool = typer.Option(
        True,
        "--use-judge/--no-judge",
        help="Use the LLM semantic judge when hidden reference targets are supplied.",
    ),
) -> None:
    hidden_reference = _load_reference(reference_json)
    record = run_eval_case(
        case=EvalCase(
            case_id=case_id,
            public_case_id="CASE_001",
            smiles=smiles,
            user_query=user_query,
            hidden_reference=hidden_reference,
        ),
        subject=subject,
        output_dir=output_dir,
        use_judge=use_judge,
    )
    typer.echo(json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2))


def _subject_runner(
    subject: Literal[
        "zero-shot-llm",
        "structure-llm",
        "react-tool-llm",
        "react-tool-llm-full",
        "evidence-formatted-react-tool-llm-full",
    ],
):
    if subject == "zero-shot-llm":
        return ZeroShotLLMBaselineSubject()
    if subject == "structure-llm":
        return StructureLLMBaselineSubject()
    if subject == "react-tool-llm":
        return ReactToolLLMBaselineSubject()
    if subject == "react-tool-llm-full":
        return ReactToolLLMFullBaselineSubject()
    if subject == "evidence-formatted-react-tool-llm-full":
        return EvidenceFormattedReactToolLLMFullBaselineSubject()
    raise ValueError(f"Unsupported subject: {subject}")


def _normalize_subject_result(
    case: EvalCase,
    subject_result: SubjectRawOutput | NormalizedPrediction,
) -> NormalizedPrediction:
    if isinstance(subject_result, NormalizedPrediction):
        return subject_result
    adapter_case = _case_without_hidden_reference(case)
    return BaselineCommonNarrativeAdapter().extract(adapter_case, subject_result)


def _case_without_hidden_reference(case: EvalCase) -> EvalCase:
    return case.model_copy(update={"hidden_reference": None})


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_summary(path: Path, record: EvalRunRecord) -> None:
    summary = {
        "run_id": record.run_id,
        "case_id": record.case.case_id,
        "subject_id": record.prediction.subject_id,
        "subject_kind": record.prediction.subject_kind,
        "model": record.prediction.model,
        "prompt_version": record.prediction.prompt_version,
        "extractor_model": record.prediction.extractor_model,
        "extractor_prompt_version": record.prediction.extractor_prompt_version,
        "metric_status": {
            metric.metric_name: {
                "status": metric.status,
                "score": metric.score,
                "precision": metric.precision,
                "recall": metric.recall,
                "f1": metric.f1,
                "predicted_positive_count": metric.predicted_positive_count,
                "reference_target_count": metric.reference_target_count,
                "matched_prediction_count": metric.matched_prediction_count,
                "matched_target_count": metric.matched_target_count,
                "true_positives": metric.true_positives,
                "false_positives": metric.false_positives,
                "false_negatives": metric.false_negatives,
                "target_count": metric.target_count,
            }
            for metric in record.metrics
        },
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_reference(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    parsed = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if isinstance(parsed, dict) and isinstance(parsed.get("hidden_reference"), dict):
        return parsed["hidden_reference"]
    if not isinstance(parsed, dict):
        raise ValueError("reference JSON must contain an object.")
    return parsed


if __name__ == "__main__":
    app()
