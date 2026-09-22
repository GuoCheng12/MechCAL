from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from mechcal.schemas import DiagnosisUnit, EvidenceUnit
from mechcal.schemas.base import JsonDict, MechCALModel, utc_now

EvalSubjectKind = Literal[
    "zero_shot_llm",
    "structure_llm",
    "evidence_formatted_structure_llm",
    "codex_structure_prior_llm",
    "react_tool_llm",
    "react_tool_llm_full",
    "evidence_formatted_react_tool_llm_full",
    "mechcal",
]
MetricName = Literal["EA", "DA"]
MetricStatus = Literal["scored", "reference_missing", "judge_failed"]


class EvalCase(MechCALModel):
    record_type: str = "EvalCase"
    case_id: str
    public_case_id: str | None = None
    smiles: str
    user_query: str = "Assess the likely AIE mechanism for this molecule."
    label: str | None = None
    hidden_reference: JsonDict | None = None

    @field_validator("case_id", "smiles", "user_query", "public_case_id")
    @classmethod
    def _non_empty_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("EvalCase text fields must not be empty.")
        return normalized


class SubjectRawOutput(MechCALModel):
    record_type: str = "SubjectRawOutput"
    raw_output_id: str
    case_id: str
    subject_id: str
    subject_kind: EvalSubjectKind
    model: str
    prompt_version: str
    raw_text: str = ""
    raw_json: JsonDict | None = None
    created_at: str = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)


class NormalizedPrediction(MechCALModel):
    record_type: str = "NormalizedPrediction"
    prediction_id: str
    case_id: str
    subject_id: str
    subject_kind: EvalSubjectKind
    model: str
    prompt_version: str
    extractor_id: str | None = None
    extractor_model: str | None = None
    extractor_prompt_version: str | None = None
    final_summary: str
    evidence_units: list[EvidenceUnit] = Field(default_factory=list)
    diagnosis_units: list[DiagnosisUnit] = Field(default_factory=list)
    raw_output: SubjectRawOutput
    created_at: str = Field(default_factory=utc_now)
    metadata: JsonDict = Field(default_factory=dict)


class AlignmentMetric(MechCALModel):
    record_type: str = "AlignmentMetric"
    metric_name: MetricName
    status: MetricStatus
    score: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    predicted_positive_count: int | None = None
    reference_target_count: int | None = None
    matched_prediction_count: int | None = None
    matched_target_count: int | None = None
    true_positives: int | None = None
    false_positives: int | None = None
    false_negatives: int | None = None
    target_count: int = 0
    rationale: str = ""
    judge_model: str | None = None
    judge_prompt_version: str | None = None
    details: JsonDict = Field(default_factory=dict)


class EvalRunRecord(MechCALModel):
    record_type: str = "EvalRunRecord"
    run_id: str
    case: EvalCase
    prediction: NormalizedPrediction
    metrics: list[AlignmentMetric] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)
