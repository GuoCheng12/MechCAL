from __future__ import annotations

import re
from typing import Any

from mechcal.public_ids import public_case_id_from_metadata
from mechcal.runtime.llm import OpenAIJsonClient
from mechcal.runtime.prompts import load_prompt
from mechcal.schemas import CaseRun, ConclusionLedger, DiagnosisConclusion

FORBIDDEN_CONCLUSION_TEXT = {
    "hidden_reference",
    "semantic_evidence_targets",
    "semantic_diagnosis_targets",
    "reference_evidence_units",
    "reference_diagnosis_units",
    "benchmark reference",
    "target-label schema",
    "ea score",
    "da score",
}
DIAGNOSIS_STATUSES_REQUIRING_REFS = {
    "computed_supported",
    "proxy_supported",
    "weakened",
    "rejected",
}


class ConclusionLedgerAgent:
    """LLM-driven runtime registrar for Planner-grounded scientific conclusions."""

    def __init__(self, *, llm_client: OpenAIJsonClient | None = None) -> None:
        self.llm_client = llm_client or OpenAIJsonClient()

    def build(self, case_run: CaseRun) -> ConclusionLedger:
        round_id = case_run.current_round_id or (
            case_run.round_ids[-1] if case_run.round_ids else "FINAL"
        )
        allowed_evidence_ids = [item.evidence_id for item in case_run.evidence_ledger.items]
        if not self.llm_client.is_configured():
            raise RuntimeError(
                "ConclusionLedgerAgent requires a configured LLM client."
            )

        payload = _ledger_payload(case_run, round_id=round_id)
        last_error: str | None = None
        attempt_errors: list[str] = []
        for attempt in range(2):
            try:
                response = self.llm_client.complete_json(
                    system_prompt=load_prompt("conclusion_ledger.md"),
                    payload=payload,
                    schema_feedback=last_error,
                )
                ledger = ConclusionLedger.model_validate(
                    _normalize_ledger_response(
                        response,
                        case_run=case_run,
                        round_id=round_id,
                    )
                )
                return _sanitize_ledger(
                    ledger,
                    case_run=case_run,
                    round_id=round_id,
                    allowed_evidence_ids=allowed_evidence_ids,
                    source_corpus=_source_corpus(case_run),
                )
            except Exception as exc:  # noqa: BLE001
                last_error = f"Attempt {attempt + 1} failed: {type(exc).__name__}: {exc}"
                attempt_errors.append(last_error)
        raise RuntimeError(
            "ConclusionLedgerAgent failed to produce a valid ledger: "
            + " | ".join(attempt_errors or [str(last_error)])
        )


def _ledger_payload(case_run: CaseRun, *, round_id: str) -> dict[str, Any]:
    return {
        "public_case": {
            "case_label": "current_smiles_case",
            "smiles": case_run.input.smiles,
            "user_query": _clip(case_run.input.user_query, 180),
            "round_id": round_id,
        },
        "planner_state": {
            "current_hypothesis": case_run.portfolio.current,
            "runner_up": case_run.portfolio.runner_up,
            "hypotheses": [
                {
                    "name": item.name,
                    "confidence": item.confidence,
                    "status": item.status,
                    "rationale": _clip(item.rationale, 220),
                    "evidence_refs": item.evidence_refs[:6],
                }
                for item in case_run.portfolio.sorted_hypotheses()[:8]
            ],
        },
        "mechanism_agenda": _compact_mechanism_agenda(case_run),
        "photophysics_review": _compact_photophysics_review(case_run),
        "accepted_evidence": [
            {
                "evidence_id": item.evidence_id,
                "agent_name": item.agent_name,
                "capability_id": item.capability_id,
                "basis": item.basis,
                "support": item.support,
                "family": item.family,
                "status": item.status,
                "claim": _clip(item.claim, 180),
                "summary": _clip(item.summary, 180),
                "limits": [_clip(limit, 120) for limit in item.limits[:2]],
                "observable_tags": item.observable_tags[:5],
            }
            for item in case_run.evidence_ledger.recent_items(limit=12)
        ],
        "allowed_evidence_ids": [
            item.evidence_id for item in case_run.evidence_ledger.items
        ],
        "output_contract": {
            "ledger_id": f"{_public_case_id(case_run)}:{round_id}:conclusion_ledger",
            "case_id": _public_case_id(case_run),
            "round_id": round_id,
            "evidence_conclusions": (
                "list of conclusion-level statements grounded in accepted_evidence"
            ),
            "diagnosis_conclusions": (
                "list of Planner-grounded mechanism assessments; do not add new "
                "mechanisms not present in planner_state or photophysics_review"
            ),
            "pending_questions": "list of unresolved evidence needs",
            "policy_notes": "short notes about provenance and limits",
        },
        "max_output_units": {
            "evidence_conclusions": 5,
            "diagnosis_conclusions": 5,
            "pending_questions": 4,
        },
        "policy": {
            "reference_access": "none",
            "external_answer_access": "none",
            "case_specific_literature_search": "forbidden",
            "final_decision_authority": "Planner only",
            "source_grounding": "Each accepted conclusion must cite allowed_evidence_ids.",
        },
    }


def _compact_mechanism_agenda(case_run: CaseRun) -> dict[str, Any] | None:
    program = case_run.mechanism_program
    if program is None:
        return None
    return {
        "candidate_mechanisms": [
            {
                "label": item.label,
                "mechanism_family": item.mechanism_family,
                "rationale": _clip(item.rationale, 120),
            }
            for item in program.candidate_mechanisms[:5]
        ],
        "scope_boundaries": [_clip(item.statement, 120) for item in program.scope_boundaries[:4]],
    }


def _compact_photophysics_review(case_run: CaseRun) -> dict[str, Any] | None:
    review = case_run.photophysics_review
    if review is None:
        return None
    return {
        "hypothesis_cards": [
            {
                "mechanism": item.mechanism,
                "status": item.status,
                "support_evidence_refs": item.support_evidence_refs[:4],
                "weakening_evidence_refs": item.weakening_evidence_refs[:4],
                "missing_or_unresolved": item.missing_or_unresolved[:3],
                "reasoning_summary": _clip(item.reasoning_summary, 180),
                "scope_limits": item.scope_limits[:2],
            }
            for item in review.hypothesis_cards[:5]
        ],
        "overclaim_warnings": [_clip(item, 120) for item in review.overclaim_warnings[:3]],
    }


def _normalize_ledger_response(
    response: dict[str, Any],
    *,
    case_run: CaseRun,
    round_id: str,
) -> dict[str, Any]:
    payload = _first_mapping(
        response,
        "conclusion_ledger",
        "ledger",
        "scientific_conclusion_ledger",
    ) or response
    if any(term in str(payload).lower() for term in FORBIDDEN_CONCLUSION_TEXT):
        raise ValueError("Conclusion ledger contained forbidden private/source text.")
    public_case_id = _public_case_id(case_run)
    return {
        "ledger_id": f"{public_case_id}:{round_id}:conclusion_ledger",
        "case_id": public_case_id,
        "round_id": round_id,
        "evidence_conclusions": [
            _normalize_evidence_conclusion(item, index=index)
            for index, item in enumerate(
                _first_dict_list(payload, "evidence_conclusions", "evidence", "findings"),
                start=1,
            )
        ],
        "diagnosis_conclusions": [
            _normalize_diagnosis_conclusion(item, index=index)
            for index, item in enumerate(
                _first_dict_list(payload, "diagnosis_conclusions", "diagnoses"),
                start=1,
            )
        ],
        "pending_questions": [
            _normalize_pending_question(item, index=index)
            for index, item in enumerate(
                _first_dict_list(payload, "pending_questions", "unresolved_questions"),
                start=1,
            )
        ],
        "policy_notes": _as_text_list(payload.get("policy_notes")),
    }


def _normalize_evidence_conclusion(item: dict[str, Any], *, index: int) -> dict[str, Any]:
    return {
        "conclusion_id": _first_text(item, "conclusion_id", "id") or f"EC{index:03d}",
        "statement": _first_text(item, "statement", "claim", "conclusion") or "Unspecified.",
        "context": _first_text(item, "context") or "ConclusionLedgerAgent runtime record",
        "mechanism_family": _first_text(item, "mechanism_family", "mechanism"),
        "direction": _normalize_direction(_first_text(item, "direction", "support")),
        "basis": _normalize_basis(_first_text(item, "basis")),
        "source_evidence_refs": _as_text_list(
            item.get("source_evidence_refs") or item.get("evidence_refs")
        ),
        "source_snippets": _as_text_list(item.get("source_snippets")),
        "limits": _as_text_list(item.get("limits") or item.get("scope_limits")),
        "confidence": _normalize_confidence(_first_text(item, "confidence")),
    }


def _normalize_diagnosis_conclusion(item: dict[str, Any], *, index: int) -> dict[str, Any]:
    refs = _as_text_list(item.get("source_evidence_refs") or item.get("evidence_refs"))
    status = _normalize_status(_first_text(item, "status"))
    missing = _as_text_list(item.get("missing_or_unresolved"))
    if status in DIAGNOSIS_STATUSES_REQUIRING_REFS and not refs:
        status = "underdetermined"
        if not missing:
            missing = ["Needs source-grounded runtime evidence before stronger assessment."]
    if status == "underdetermined" and not missing:
        missing = ["Needs additional source-grounded runtime evidence."]
    return {
        "conclusion_id": _first_text(item, "conclusion_id", "diagnosis_id", "id")
        or f"DC{index:03d}",
        "mechanism": _first_text(item, "mechanism") or "Planner mechanism assessment",
        "status": status,
        "statement": _first_text(item, "statement", "conclusion") or (
            _first_text(item, "reasoning_summary") or "Mechanism assessment."
        ),
        "reasoning_summary": _first_text(item, "reasoning_summary", "reasoning")
        or (_first_text(item, "statement", "conclusion") or "Mechanism assessment."),
        "source_evidence_refs": refs,
        "source_snippets": _as_text_list(item.get("source_snippets")),
        "missing_or_unresolved": missing,
        "scope_limits": _as_text_list(item.get("scope_limits") or item.get("limits")),
        "confidence": _normalize_confidence(_first_text(item, "confidence")),
    }


def _normalize_pending_question(item: dict[str, Any], *, index: int) -> dict[str, Any]:
    return {
        "question_id": _first_text(item, "question_id", "id") or f"PQ{index:03d}",
        "question": _first_text(item, "question", "statement") or "Unresolved question.",
        "needed_evidence": _as_text_list(item.get("needed_evidence")),
        "source_evidence_refs": _as_text_list(item.get("source_evidence_refs")),
    }


def _sanitize_ledger(
    ledger: ConclusionLedger,
    *,
    case_run: CaseRun,
    round_id: str,
    allowed_evidence_ids: list[str],
    source_corpus: str,
) -> ConclusionLedger:
    allowed = set(allowed_evidence_ids)
    evidence_conclusions = [
        item.model_copy(
            update={
                "source_evidence_refs": _valid_refs(item.source_evidence_refs, allowed),
                "source_snippets": _grounded_snippets(item.source_snippets, source_corpus),
            }
        )
        for item in ledger.evidence_conclusions
    ]
    evidence_conclusions = [
        item for item in evidence_conclusions if item.source_evidence_refs
    ]
    diagnosis_conclusions: list[DiagnosisConclusion] = []
    for item in ledger.diagnosis_conclusions:
        refs = _valid_refs(item.source_evidence_refs, allowed)
        if item.status in DIAGNOSIS_STATUSES_REQUIRING_REFS and not refs:
            continue
        snippets = _grounded_snippets(item.source_snippets, source_corpus)
        if item.status == "underdetermined" and not item.missing_or_unresolved:
            continue
        diagnosis_conclusions.append(
            item.model_copy(
                update={
                    "source_evidence_refs": refs,
                    "source_snippets": snippets,
                }
            )
        )
    pending_questions = [
        item.model_copy(
            update={"source_evidence_refs": _valid_refs(item.source_evidence_refs, allowed)}
        )
        for item in ledger.pending_questions
    ]
    public_case_id = _public_case_id(case_run)
    return ledger.model_copy(
        update={
            "ledger_id": f"{public_case_id}:{round_id}:conclusion_ledger",
            "case_id": public_case_id,
            "round_id": round_id,
            "evidence_conclusions": evidence_conclusions,
            "diagnosis_conclusions": diagnosis_conclusions,
            "pending_questions": pending_questions,
            "policy_notes": [
                *ledger.policy_notes,
                (
                    "ConclusionLedgerAgent used only public SMILES, Planner state, "
                    "PhotophysicsReview, and allowed evidence refs."
                ),
            ],
        }
    )


def _source_corpus(case_run: CaseRun) -> str:
    parts: list[str] = []
    for item in case_run.evidence_ledger.items:
        parts.extend([item.claim, item.summary, *item.limits])
    review = case_run.photophysics_review
    if review is not None:
        for card in review.hypothesis_cards:
            parts.extend([card.reasoning_summary, *card.missing_or_unresolved, *card.scope_limits])
        for axis in review.coverage_axes:
            parts.append(axis.rationale)
    for hypothesis in case_run.portfolio.hypotheses:
        parts.append(hypothesis.rationale)
    return _normalize_text(" ".join(parts))


def _grounded_snippets(snippets: list[str], source_corpus: str) -> list[str]:
    return [
        snippet
        for snippet in snippets
        if _normalize_text(snippet) and _normalize_text(snippet) in source_corpus
    ]


def _valid_refs(refs: list[str], allowed: set[str]) -> list[str]:
    return [ref for ref in refs if ref in allowed]


def _normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def _first_mapping(payload: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return None


def _first_dict_list(payload: dict[str, Any], *keys: str) -> list[dict[str, Any]]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return [value]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _first_text(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_direction(value: str | None) -> str:
    normalized = (value or "unresolved").strip().lower()
    mapped = {
        "support": "supports",
        "supported": "supports",
        "supporting": "supports",
        "challenge": "weakens",
        "challenges": "weakens",
        "weakened": "weakens",
        "weak": "weakens",
        "unknown": "unresolved",
        "uncertain": "unresolved",
        "not applicable": "not_applicable",
        "n/a": "not_applicable",
    }
    return mapped.get(normalized, normalized if normalized in {
        "supports",
        "weakens",
        "unresolved",
        "not_applicable",
    } else "unresolved")


def _normalize_basis(value: str | None) -> str:
    normalized = (value or "proxy").strip().lower()
    if normalized in {"computed", "proxy", "checklist"}:
        return normalized
    if normalized in {"structure", "structural", "qualitative"}:
        return "proxy"
    return "proxy"


def _normalize_status(value: str | None) -> str:
    normalized = (value or "underdetermined").strip().lower()
    mapped = {
        "supported": "proxy_supported",
        "structure_supported": "proxy_supported",
        "computed": "computed_supported",
        "weakens": "weakened",
        "unknown": "underdetermined",
        "uncertain": "underdetermined",
    }
    normalized = mapped.get(normalized, normalized)
    allowed = {
        "computed_supported",
        "proxy_supported",
        "plausible",
        "weakened",
        "rejected",
        "underdetermined",
    }
    return normalized if normalized in allowed else "underdetermined"


def _normalize_confidence(value: str | None) -> str:
    normalized = (value or "medium").strip().lower()
    if normalized in {"low", "medium", "high"}:
        return normalized
    try:
        numeric = float(normalized)
    except ValueError:
        return "medium"
    if numeric >= 0.75:
        return "high"
    if numeric <= 0.35:
        return "low"
    return "medium"


def _public_case_id(case_run: CaseRun) -> str:
    return public_case_id_from_metadata(case_run.input.metadata)


def _clip(value: str | None, limit: int) -> str:
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
