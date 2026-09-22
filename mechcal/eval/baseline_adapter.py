from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, Field, ValidationError

from mechcal.eval.extractors import (
    ExtractedDiagnosisCandidate,
    ExtractedEvidenceCandidate,
    ExtractedPredictionPayload,
    normalized_prediction_from_payload,
)
from mechcal.eval.prompts import load_eval_prompt
from mechcal.eval.schemas import EvalCase, NormalizedPrediction, SubjectRawOutput
from mechcal.public_ids import normalize_public_case_id
from mechcal.runtime.llm import OpenAICompatibleSettings, OpenAIJsonClient

BASELINE_COMMON_ADAPTER_ID = "baseline_common_narrative_to_schema_v1"
BASELINE_COMMON_ADAPTER_PROMPT = "baseline_common_extractor.md"
BASELINE_COMMON_ADAPTER_PROMPT_VERSION = "2026-06-03"


class GroundedEvidenceCandidate(ExtractedEvidenceCandidate):
    source_ref: str
    source_snippet: str


class GroundedDiagnosisCandidate(ExtractedDiagnosisCandidate):
    source_ref: str
    source_snippet: str


class GroundedBaselinePredictionPayload(BaseModel):
    final_summary: str
    evidence_units: list[GroundedEvidenceCandidate] = Field(default_factory=list)
    diagnosis_units: list[GroundedDiagnosisCandidate] = Field(default_factory=list)


class BaselineCommonNarrativeAdapter:
    """Shared narrative-to-schema adapter for non-MAS baselines."""

    extractor_id = BASELINE_COMMON_ADAPTER_ID

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
        raw_packet = render_baseline_adapter_packet(case, raw_output)
        feedback: str | None = None
        last_error: Exception | None = None
        for _ in range(self.max_attempts):
            try:
                response = self.client.complete_json(
                    system_prompt=load_eval_prompt(BASELINE_COMMON_ADAPTER_PROMPT),
                    payload={
                        "case_id": _prompt_case_id(case),
                        "smiles": case.smiles,
                        "user_query": case.user_query,
                        "subject_id": raw_output.subject_id,
                        "subject_kind": raw_output.subject_kind,
                        "raw_packet": raw_packet,
                    },
                    schema_feedback=feedback,
                )
                grounded = GroundedBaselinePredictionPayload.model_validate(
                    _normalize_grounded_baseline_payload(response)
                )
                grounded, audit = _deduplicate_grounded_payload(grounded, raw_packet)
                _validate_grounding_audit(audit)
                prediction = normalized_prediction_from_payload(
                    case=case,
                    raw_output=raw_output,
                    payload=_ungrounded_payload(grounded),
                    extractor_id=BASELINE_COMMON_ADAPTER_ID,
                    extractor_model=self.model,
                    extractor_prompt_version=BASELINE_COMMON_ADAPTER_PROMPT_VERSION,
                )
                prediction.metadata.update(
                    {
                        "adapter_role": "baseline_common_extraction",
                        "adapter_policy": (
                            "extract_all_distinct_source_grounded_units_without_hard_cap"
                        ),
                        "grounding_audit": audit,
                    }
                )
                return prediction
            except (ValidationError, ValueError) as exc:
                last_error = exc
                feedback = str(exc)
        raise RuntimeError(f"Could not extract baseline common prediction: {last_error}") from (
            last_error
        )


def render_baseline_adapter_packet(case: EvalCase, raw_output: SubjectRawOutput) -> str:
    lines = [
        "CASE:",
        f"case_id: {_prompt_case_id(case)}",
        f"smiles: {case.smiles}",
        f"user_query: {case.user_query}",
        "",
        "SUBJECT:",
        f"subject_id: {raw_output.subject_id}",
        f"subject_kind: {raw_output.subject_kind}",
        f"model: {raw_output.model}",
        f"prompt_version: {raw_output.prompt_version}",
    ]
    if raw_output.raw_text.strip():
        lines.extend(["", "RAW_TEXT:", raw_output.raw_text.strip()])
    raw_json = _adapter_visible_raw_json(raw_output)
    if raw_json is not None:
        lines.extend(
            [
                "",
                "RAW_JSON:",
                json.dumps(raw_json, ensure_ascii=False, indent=2),
            ]
        )
    if raw_output.metadata:
        lines.extend(
            [
                "",
                "RAW_METADATA:",
                json.dumps(raw_output.metadata, ensure_ascii=False, indent=2),
            ]
        )
    return "\n".join(lines)


def _prompt_case_id(case: EvalCase) -> str:
    return normalize_public_case_id(case.public_case_id)


def _adapter_visible_raw_json(raw_output: SubjectRawOutput) -> dict[str, object] | None:
    if raw_output.raw_json is None:
        return None
    if raw_output.subject_kind in {
        "react_tool_llm",
        "react_tool_llm_full",
        "evidence_formatted_react_tool_llm_full",
    }:
        return None
    return raw_output.raw_json


def _normalize_grounded_baseline_payload(payload: object) -> object:
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    evidence_units = normalized.get("evidence_units")
    if isinstance(evidence_units, list):
        normalized["evidence_units"] = [
            _normalize_grounded_evidence_unit(item) for item in evidence_units
        ]
    diagnosis_units = normalized.get("diagnosis_units")
    if isinstance(diagnosis_units, list):
        normalized["diagnosis_units"] = [
            _normalize_grounded_diagnosis_unit(item) for item in diagnosis_units
        ]
    return normalized


def _normalize_grounded_evidence_unit(item: object) -> object:
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


def _normalize_grounded_diagnosis_unit(item: object) -> object:
    if not isinstance(item, dict):
        return item
    normalized = dict(item)
    normalized["status"] = _normalize_enum_value(
        normalized.get("status"),
        {
            "supported": "proxy_supported",
            "proxy_supported": "proxy_supported",
            "proxy support": "proxy_supported",
            "structure_supported": "proxy_supported",
            "structurally_supported": "proxy_supported",
            "partially_supported": "plausible",
            "possible": "plausible",
            "boundary_only": "underdetermined",
            "boundary only": "underdetermined",
            "boundary": "underdetermined",
            "unknown": "underdetermined",
            "unresolved": "underdetermined",
            "uncertain": "underdetermined",
            "not_supported": "underdetermined",
            "unsupported": "underdetermined",
        },
    )
    return normalized


def _normalize_enum_value(value: object, mapping: dict[str, str]) -> object:
    if not isinstance(value, str):
        return value
    key = value.strip().lower().replace("-", "_")
    key = " ".join(key.split())
    return mapping.get(key, value)


def _deduplicate_grounded_payload(
    payload: GroundedBaselinePredictionPayload,
    raw_packet: str,
) -> tuple[GroundedBaselinePredictionPayload, dict[str, object]]:
    evidence_units, removed_evidence = _deduplicate_items(
        payload.evidence_units,
        key_fn=lambda item: (
            _normalized_key(item.claim),
            str(item.basis),
            str(item.support),
        ),
    )
    diagnosis_units, removed_diagnoses = _deduplicate_items(
        payload.diagnosis_units,
        key_fn=lambda item: (_normalized_key(item.mechanism), str(item.status)),
    )
    evidence_units, evidence_repairs = _repair_grounded_snippets(evidence_units, raw_packet)
    diagnosis_units, diagnosis_repairs = _repair_grounded_snippets(
        diagnosis_units,
        raw_packet,
    )
    grounding = [
        *_grounding_items(evidence_units, raw_packet, unit_type="evidence"),
        *_grounding_items(diagnosis_units, raw_packet, unit_type="diagnosis"),
    ]
    return (
        GroundedBaselinePredictionPayload(
            final_summary=payload.final_summary,
            evidence_units=evidence_units,
            diagnosis_units=diagnosis_units,
        ),
        {
            "evidence_unit_count": len(evidence_units),
            "diagnosis_unit_count": len(diagnosis_units),
            "removed_duplicate_evidence_count": len(removed_evidence),
            "removed_duplicate_diagnosis_count": len(removed_diagnoses),
            "snippet_repairs": [*evidence_repairs, *diagnosis_repairs],
            "grounding": grounding,
        },
    )


T = TypeVar("T")


def _deduplicate_items(
    items: list[T],
    *,
    key_fn: Callable[[T], tuple[object, ...]],
) -> tuple[list[T], list[T]]:
    seen: set[tuple[str, ...]] = set()
    kept = []
    removed = []
    for item in items:
        key = tuple(str(part) for part in key_fn(item))
        if key in seen:
            removed.append(item)
            continue
        seen.add(key)
        kept.append(item)
    return kept, removed


def _repair_grounded_snippets(
    items: list[T],
    raw_packet: str,
) -> tuple[list[T], list[dict[str, object]]]:
    repaired_items: list[T] = []
    repairs: list[dict[str, object]] = []
    normalized_packet = _normalize_for_grounding(raw_packet)
    for index, item in enumerate(items, start=1):
        snippet = str(getattr(item, "source_snippet", "")).strip()
        if _normalize_for_grounding(snippet) in normalized_packet:
            repaired_items.append(item)
            continue
        repaired_snippet = _find_locatable_snippet(snippet, normalized_packet)
        if repaired_snippet is None:
            repaired_items.append(item)
            continue
        repaired_items.append(item.model_copy(update={"source_snippet": repaired_snippet}))
        repairs.append(
            {
                "unit_index": index,
                "source_ref": getattr(item, "source_ref", ""),
                "old_source_snippet": snippet,
                "new_source_snippet": repaired_snippet,
            }
        )
    return repaired_items, repairs


def _find_locatable_snippet(snippet: str, normalized_packet: str) -> str | None:
    for candidate in _snippet_repair_candidates(snippet):
        if _normalize_for_grounding(candidate) in normalized_packet:
            return candidate
    return None


def _snippet_repair_candidates(snippet: str) -> list[str]:
    candidates: list[str] = []
    for separator in (";", ". ", ", "):
        if separator not in snippet:
            continue
        candidates.extend(part.strip(" .;,\n\t") for part in snippet.split(separator))
    if " reports " in snippet:
        candidates.append(snippet.split(" reports ", 1)[1].strip(" .;,\n\t"))
    return [
        candidate
        for candidate in candidates
        if 3 <= len(candidate) <= 180 and not candidate.startswith("{")
    ]


def _grounding_items(items: list, raw_packet: str, *, unit_type: str) -> list[dict[str, object]]:
    normalized_packet = _normalize_for_grounding(raw_packet)
    audit: list[dict[str, object]] = []
    for index, item in enumerate(items, start=1):
        snippet = str(item.source_snippet).strip()
        audit.append(
            {
                "unit_type": unit_type,
                "unit_index": index,
                "source_ref": item.source_ref,
                "source_snippet": snippet,
                "source_snippet_found": _normalize_for_grounding(snippet)
                in normalized_packet,
            }
        )
    return audit


def _validate_grounding_audit(audit: dict[str, object]) -> None:
    grounding = audit.get("grounding")
    if not isinstance(grounding, list):
        return
    misses = [item for item in grounding if not item.get("source_snippet_found")]
    if not misses:
        return
    details = []
    for item in misses[:8]:
        details.append(
            "{unit_type}#{unit_index} source_ref={source_ref!r} source_snippet={snippet!r}".format(
                unit_type=item.get("unit_type"),
                unit_index=item.get("unit_index"),
                source_ref=item.get("source_ref"),
                snippet=str(item.get("source_snippet", ""))[:240],
            )
        )
    raise ValueError(
        "Every source_snippet must be a contiguous copied substring from raw_packet. "
        "These snippets were not found after case-insensitive whitespace-normalized "
        f"matching: {'; '.join(details)}"
    )


def _ungrounded_payload(payload: GroundedBaselinePredictionPayload) -> ExtractedPredictionPayload:
    return ExtractedPredictionPayload(
        final_summary=payload.final_summary,
        evidence_units=[
            ExtractedEvidenceCandidate.model_validate(item.model_dump())
            for item in payload.evidence_units
        ],
        diagnosis_units=[
            ExtractedDiagnosisCandidate.model_validate(item.model_dump())
            for item in payload.diagnosis_units
        ],
    )


def _normalized_key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _normalize_for_grounding(text: str) -> str:
    return " ".join(text.lower().split())
