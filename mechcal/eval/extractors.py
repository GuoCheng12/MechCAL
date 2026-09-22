from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from mechcal.eval.prompts import load_eval_prompt
from mechcal.eval.schemas import EvalCase, NormalizedPrediction, SubjectRawOutput
from mechcal.runtime.llm import OpenAICompatibleSettings, OpenAIJsonClient
from mechcal.schemas import DiagnosisUnit, EvidenceUnit
from mechcal.schemas.evidence import (
    EvidenceBasis,
    EvidenceFamily,
    EvidenceRelation,
    EvidenceStatus,
    EvidenceSupport,
)
from mechcal.schemas.mechanisms import DiagnosisStatus

EXTRACTOR_ID = "free_text_to_schema_v1"
EXTRACTOR_PROMPT = "free_text_extractor.md"
EXTRACTOR_PROMPT_VERSION = "2026-06-01"


class ExtractedEvidenceCandidate(BaseModel):
    claim: str
    context: str = "structure-only baseline interpretation"
    basis: EvidenceBasis = "proxy"
    support: EvidenceSupport = "unresolved"
    summary: str
    limits: list[str] = Field(default_factory=list)
    observable: str | None = None
    family: EvidenceFamily = "geometry_precondition"
    relation: EvidenceRelation = "unknown"
    status: EvidenceStatus = "present"
    observable_tags: list[str] = Field(default_factory=list)


class ExtractedDiagnosisCandidate(BaseModel):
    mechanism: str
    context: str = "structure-only baseline interpretation"
    status: DiagnosisStatus = "plausible"
    reasoning_summary: str
    missing_or_unresolved: list[str] = Field(default_factory=list)
    scope_limits: list[str] = Field(default_factory=list)


class ExtractedPredictionPayload(BaseModel):
    final_summary: str
    evidence_units: list[ExtractedEvidenceCandidate] = Field(default_factory=list)
    diagnosis_units: list[ExtractedDiagnosisCandidate] = Field(default_factory=list)


class FreeTextToSchemaExtractor:
    def __init__(
        self,
        *,
        client: OpenAIJsonClient | None = None,
        settings: OpenAICompatibleSettings | None = None,
        max_attempts: int = 2,
    ) -> None:
        self.client = client or OpenAIJsonClient(settings)
        self.max_attempts = max(1, max_attempts)

    @property
    def model(self) -> str:
        return self.client.settings.model

    def extract(self, case: EvalCase, raw_output: SubjectRawOutput) -> NormalizedPrediction:
        feedback: str | None = None
        last_error: Exception | None = None
        for _ in range(self.max_attempts):
            try:
                payload = self.client.complete_json(
                    system_prompt=load_eval_prompt(EXTRACTOR_PROMPT),
                    payload={
                        "case_id": case.case_id,
                        "smiles": case.smiles,
                        "user_query": case.user_query,
                        "subject_id": raw_output.subject_id,
                        "subject_kind": raw_output.subject_kind,
                        "raw_text": raw_output.raw_text,
                        "raw_json": raw_output.raw_json,
                    },
                    schema_feedback=feedback,
                )
                extracted = ExtractedPredictionPayload.model_validate(payload)
                return normalized_prediction_from_payload(
                    case=case,
                    raw_output=raw_output,
                    payload=extracted,
                    extractor_id=EXTRACTOR_ID,
                    extractor_model=self.model,
                    extractor_prompt_version=EXTRACTOR_PROMPT_VERSION,
                )
            except (ValidationError, ValueError) as exc:
                last_error = exc
                feedback = str(exc)
        raise RuntimeError(f"Could not extract normalized prediction: {last_error}") from last_error


def normalized_prediction_from_payload(
    *,
    case: EvalCase,
    raw_output: SubjectRawOutput,
    payload: ExtractedPredictionPayload,
    extractor_id: str | None = None,
    extractor_model: str | None = None,
    extractor_prompt_version: str | None = None,
) -> NormalizedPrediction:
    evidence_units = [
        _evidence_unit_from_candidate(
            case=case,
            raw_output=raw_output,
            candidate=candidate,
            index=index,
        )
        for index, candidate in enumerate(payload.evidence_units, start=1)
    ]
    evidence_refs = [item.evidence_id for item in evidence_units]
    diagnosis_units = [
        _diagnosis_unit_from_candidate(
            candidate=candidate,
            index=index,
            evidence_refs=evidence_refs,
        )
        for index, candidate in enumerate(payload.diagnosis_units, start=1)
    ]
    return NormalizedPrediction(
        prediction_id=f"{case.case_id}:{raw_output.subject_id}:prediction",
        case_id=case.case_id,
        subject_id=raw_output.subject_id,
        subject_kind=raw_output.subject_kind,
        model=raw_output.model,
        prompt_version=raw_output.prompt_version,
        extractor_id=extractor_id,
        extractor_model=extractor_model,
        extractor_prompt_version=extractor_prompt_version,
        final_summary=payload.final_summary.strip(),
        evidence_units=evidence_units,
        diagnosis_units=diagnosis_units,
        raw_output=raw_output,
    )


def payload_from_structure_llm_json(payload: dict[str, Any]) -> ExtractedPredictionPayload:
    return ExtractedPredictionPayload.model_validate(_normalize_structure_llm_payload(payload))


def _normalize_structure_llm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    evidence_units = normalized.get("evidence_units")
    if isinstance(evidence_units, list):
        normalized["evidence_units"] = [
            _normalize_structure_llm_evidence_unit(item) for item in evidence_units
        ]
    diagnosis_units = normalized.get("diagnosis_units")
    if isinstance(diagnosis_units, list):
        normalized["diagnosis_units"] = [
            _normalize_structure_llm_diagnosis_unit(item) for item in diagnosis_units
        ]
    return normalized


def _normalize_structure_llm_evidence_unit(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    normalized = dict(item)
    normalized["basis"] = _normalize_enum_value(
        normalized.get("basis"),
        {
            "structure": "proxy",
            "structural": "proxy",
            "structural_proxy": "proxy",
            "qualitative": "proxy",
        },
    )
    normalized["support"] = _normalize_enum_value(
        normalized.get("support"),
        {
            "support": "supports",
            "supported": "supports",
            "supporting": "supports",
            "challenge": "weakens",
            "challenges": "weakens",
            "challenged": "weakens",
            "weak": "weakens",
            "weakened": "weakens",
            "unknown": "unresolved",
            "uncertain": "unresolved",
            "partial": "unresolved",
            "plausible": "unresolved",
            "not applicable": "not_applicable",
            "n/a": "not_applicable",
        },
    )
    normalized["relation"] = _normalize_enum_value(
        normalized.get("relation"),
        {
            "support": "supports",
            "supported": "supports",
            "weakens": "challenges",
            "weakened": "challenges",
            "challenge": "challenges",
            "unresolved": "unknown",
            "uncertain": "unknown",
            "not_applicable": "unknown",
            "not applicable": "unknown",
            "n/a": "unknown",
        },
    )
    normalized["status"] = _normalize_enum_value(
        normalized.get("status"),
        {
            "unresolved": "partial",
            "unknown": "partial",
            "uncertain": "partial",
            "plausible": "present",
            "supported": "present",
            "supporting": "present",
            "not_applicable": "unsupported",
            "not applicable": "unsupported",
            "n/a": "unsupported",
        },
    )
    return normalized


def _normalize_structure_llm_diagnosis_unit(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    normalized = dict(item)
    normalized["status"] = _normalize_enum_value(
        normalized.get("status"),
        {
            "supported": "proxy_supported",
            "structure_supported": "proxy_supported",
            "structurally_supported": "proxy_supported",
            "partially_supported": "plausible",
            "possible": "plausible",
            "unknown": "underdetermined",
            "unresolved": "underdetermined",
            "uncertain": "underdetermined",
            "not_supported": "underdetermined",
            "unsupported": "underdetermined",
        },
    )
    return normalized


def _normalize_enum_value(value: Any, mapping: dict[str, str]) -> Any:
    if not isinstance(value, str):
        return value
    key = value.strip().lower().replace("-", "_")
    key = " ".join(key.split())
    return mapping.get(key, value)


def _evidence_unit_from_candidate(
    *,
    case: EvalCase,
    raw_output: SubjectRawOutput,
    candidate: ExtractedEvidenceCandidate,
    index: int,
) -> EvidenceUnit:
    return EvidenceUnit(
        evidence_id=f"{case.case_id}:{raw_output.subject_id}:E{index:03d}",
        round_id="baseline",
        source_report_id=raw_output.raw_output_id,
        agent_name=raw_output.subject_id,
        capability_id=f"baseline.{raw_output.subject_kind}",
        claim=candidate.claim,
        context=candidate.context,
        basis=_safe_basis(candidate.basis, raw_output.subject_kind),
        support=candidate.support,
        summary=candidate.summary,
        limits=candidate.limits,
        observable=candidate.observable,
        artifact_refs=[],
        metrics={},
        family=candidate.family,
        relation=candidate.relation,
        status=candidate.status,
        claim_refs=[],
        hypothesis_refs=[],
        observable_tags=candidate.observable_tags,
    )


def _diagnosis_unit_from_candidate(
    *,
    candidate: ExtractedDiagnosisCandidate,
    index: int,
    evidence_refs: list[str],
) -> DiagnosisUnit:
    status = _safe_diagnosis_status(candidate.status, evidence_refs)
    missing_or_unresolved = list(candidate.missing_or_unresolved)
    if status == "underdetermined" and not missing_or_unresolved:
        missing_or_unresolved.append(
            "No direct benchmark reference or direct experimental evidence is available."
        )
    return DiagnosisUnit(
        diagnosis_id=f"D{index:03d}",
        mechanism=candidate.mechanism,
        context=candidate.context,
        status=status,
        evidence_refs=evidence_refs if status in _EVIDENCE_REQUIRED_STATUSES else [],
        missing_or_unresolved=missing_or_unresolved,
        reasoning_summary=candidate.reasoning_summary,
        scope_limits=candidate.scope_limits,
    )


def _safe_basis(basis: EvidenceBasis, subject_kind: str) -> EvidenceBasis:
    if subject_kind == "zero_shot_llm" and basis == "computed":
        return "proxy"
    return basis


_EVIDENCE_REQUIRED_STATUSES = {
    "computed_supported",
    "proxy_supported",
    "weakened",
    "rejected",
}


def _safe_diagnosis_status(
    status: DiagnosisStatus,
    evidence_refs: list[str],
) -> DiagnosisStatus:
    if status in _EVIDENCE_REQUIRED_STATUSES and not evidence_refs:
        return "underdetermined"
    return status
