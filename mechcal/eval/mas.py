from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from mechcal.eval.baseline_adapter import BaselineCommonNarrativeAdapter
from mechcal.eval.extractors import (
    ExtractedPredictionPayload,
    normalized_prediction_from_payload,
)
from mechcal.eval.prompts import load_eval_prompt
from mechcal.eval.schemas import EvalCase, NormalizedPrediction, SubjectRawOutput
from mechcal.public_ids import public_case_id_from_metadata
from mechcal.runtime.llm import OpenAICompatibleSettings, OpenAIJsonClient
from mechcal.schemas import CaseRun, EvidenceUnit
from mechcal.schemas.base import JsonDict
from mechcal.schemas.round import RoundRecord

MECHCAL_PROMPT_VERSION = "mechcal-runtime-2026-06-01"
MAS_NARRATIVE_EXTRACTOR_ID = "mas_narrative_to_schema_v1"
MAS_NARRATIVE_EXTRACTOR_PROMPT = "mas_narrative_extractor.md"
MAS_NARRATIVE_EXTRACTOR_PROMPT_VERSION = "2026-06-01"


def prediction_from_case_run(
    case_run: CaseRun,
    *,
    model: str,
    subject_id: str = "mechcal",
    metadata: JsonDict | None = None,
) -> NormalizedPrediction:
    final_answer = case_run.final_answer
    final_summary = final_answer.answer if final_answer is not None else ""
    diagnosis_units = (
        case_run.diagnosis_units
        or (final_answer.diagnosis_units if final_answer is not None else [])
    )
    raw_output = SubjectRawOutput(
        raw_output_id=f"{case_run.case_id}:{subject_id}:raw",
        case_id=case_run.case_id,
        subject_id=subject_id,
        subject_kind="mechcal",
        model=model,
        prompt_version=MECHCAL_PROMPT_VERSION,
        raw_text=final_summary,
        raw_json=case_run.model_dump(mode="json"),
        metadata=metadata or {},
    )
    return NormalizedPrediction(
        prediction_id=f"{case_run.case_id}:{subject_id}:prediction",
        case_id=case_run.case_id,
        subject_id=subject_id,
        subject_kind="mechcal",
        model=model,
        prompt_version=MECHCAL_PROMPT_VERSION,
        final_summary=final_summary,
        evidence_units=_canonical_evidence_units(case_run),
        diagnosis_units=list(diagnosis_units),
        raw_output=raw_output,
        metadata={
            **(metadata or {}),
            "adapter_role": "normalization_only",
            "canonical_evidence_source": "planner_scientific_evidence",
        },
    )


def strict_common_prediction_from_case_run(
    case_run: CaseRun,
    *,
    case: EvalCase,
    model: str,
    subject_id: str = "mechcal_conclusion_ledger",
    metadata: JsonDict | None = None,
    adapter: BaselineCommonNarrativeAdapter | None = None,
) -> NormalizedPrediction:
    raw_output = SubjectRawOutput(
        raw_output_id=f"{case_run.case_id}:{subject_id}:strict_common_raw",
        case_id=case_run.case_id,
        subject_id=subject_id,
        subject_kind="mechcal",
        model=model,
        prompt_version=MECHCAL_PROMPT_VERSION,
        raw_text=build_mas_strict_source_text(case_run),
        raw_json=None,
        metadata={
            **(metadata or {}),
            "adapter_input": "mas_conclusion_ledger_strict_source_text",
            "reference_access": "none",
        },
    )
    public_case = case.model_copy(update={"hidden_reference": None})
    return (adapter or BaselineCommonNarrativeAdapter()).extract(public_case, raw_output)


def _canonical_evidence_units(case_run: CaseRun) -> list[EvidenceUnit]:
    planner_items = [
        item
        for item in case_run.evidence_ledger.items
        if item.agent_name == "planner"
        and "planner_scientific_evidence" in item.observable_tags
    ]
    return planner_items or list(case_run.evidence_ledger.items)


class MASNarrativeToSchemaExtractor:
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

    def extract(
        self,
        *,
        case_run: CaseRun,
        raw_output: SubjectRawOutput,
        round_records: Sequence[RoundRecord] | None = None,
    ) -> NormalizedPrediction:
        feedback: str | None = None
        last_error: Exception | None = None
        raw_text = build_mas_narrative_text(case_run, round_records=round_records)
        for _ in range(self.max_attempts):
            try:
                response = self.client.complete_json(
                    system_prompt=load_eval_prompt(MAS_NARRATIVE_EXTRACTOR_PROMPT),
                    payload={
                        "case_id": case_run.case_id,
                        "smiles": case_run.input.smiles,
                        "user_query": case_run.input.user_query,
                        "subject_id": raw_output.subject_id,
                        "subject_kind": raw_output.subject_kind,
                        "raw_text": raw_text,
                        "raw_json": raw_output.raw_json,
                    },
                    schema_feedback=feedback,
                )
                payload = ExtractedPredictionPayload.model_validate(response)
                return normalized_prediction_from_payload(
                    case=_eval_case_like(case_run),
                    raw_output=raw_output,
                    payload=payload,
                    extractor_id=MAS_NARRATIVE_EXTRACTOR_ID,
                    extractor_model=self.model,
                    extractor_prompt_version=MAS_NARRATIVE_EXTRACTOR_PROMPT_VERSION,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                feedback = str(exc)
        raise RuntimeError(
            f"Could not extract MAS narrative prediction: {last_error}"
        ) from last_error


def narrative_prediction_from_case_run(
    case_run: CaseRun,
    *,
    model: str,
    subject_id: str = "mechcal",
    metadata: JsonDict | None = None,
    round_records: Sequence[RoundRecord] | None = None,
    round_records_dir: Path | None = None,
    extractor: MASNarrativeToSchemaExtractor | None = None,
) -> NormalizedPrediction:
    loaded_round_records = list(round_records or [])
    if not loaded_round_records and round_records_dir is not None:
        loaded_round_records = load_round_records(round_records_dir)
    raw_text = build_mas_narrative_text(case_run, round_records=loaded_round_records)
    raw_output = SubjectRawOutput(
        raw_output_id=f"{case_run.case_id}:{subject_id}:narrative_raw",
        case_id=case_run.case_id,
        subject_id=subject_id,
        subject_kind="mechcal",
        model=model,
        prompt_version=MECHCAL_PROMPT_VERSION,
        raw_text=raw_text,
        raw_json=case_run.model_dump(mode="json"),
        metadata=metadata or {},
    )
    return (extractor or MASNarrativeToSchemaExtractor()).extract(
        case_run=case_run,
        raw_output=raw_output,
        round_records=loaded_round_records,
    )


def load_round_records(run_dir: Path) -> list[RoundRecord]:
    root = run_dir.expanduser().resolve()
    rounds_dir = root / "rounds"
    if not rounds_dir.exists():
        return []
    records: list[RoundRecord] = []
    for path in sorted(rounds_dir.glob("R*/round.json")):
        records.append(RoundRecord.model_validate(json.loads(path.read_text(encoding="utf-8"))))
    return records


def build_mas_narrative_text(
    case_run: CaseRun,
    *,
    round_records: Sequence[RoundRecord] | None = None,
) -> str:
    lines = [
        f"CASE_ID: {_public_case_id(case_run)}",
        f"SMILES: {case_run.input.smiles}",
        f"USER_QUERY: {case_run.input.user_query}",
        f"STATUS: {case_run.status}",
        "",
        "FINAL_ANSWER:",
        case_run.final_answer.answer if case_run.final_answer is not None else "",
    ]
    if case_run.final_answer is not None:
        lines.extend(
            [
                "",
                "FINAL_LIMITATIONS:",
                *[f"- {item}" for item in case_run.final_answer.limitations],
                "",
                "FINAL_EVIDENCE_REFS:",
                ", ".join(case_run.final_answer.evidence_refs),
                "FINAL_ARTIFACT_REFS:",
                ", ".join(case_run.final_answer.artifact_refs),
            ]
        )
        final_diagnoses = case_run.diagnosis_units or case_run.final_answer.diagnosis_units
        if final_diagnoses:
            lines.extend(["", "FINAL_DIAGNOSIS_UNITS:"])
            for diagnosis in final_diagnoses:
                lines.extend(
                    [
                        (
                            f"- {diagnosis.diagnosis_id}: status={diagnosis.status}; "
                            f"mechanism={diagnosis.mechanism}"
                        ),
                        f"  context: {diagnosis.context}",
                        f"  reasoning: {diagnosis.reasoning_summary}",
                        "  evidence_refs: " + ", ".join(diagnosis.evidence_refs),
                        "  missing_or_unresolved: "
                        + "; ".join(diagnosis.missing_or_unresolved),
                    ]
                )

    prediction_lines = _mechanism_prediction_lines(case_run)
    if prediction_lines:
        lines.extend(["", "MECHANISM_PREDICTIONS:", *prediction_lines])

    conclusion_lines = _conclusion_ledger_lines(case_run)
    if conclusion_lines:
        lines.extend(["", "CONCLUSION_LEDGER:", *conclusion_lines])

    records = list(round_records or [])
    if records:
        lines.extend(["", "ROUND_NARRATIVE:"])
        for record in records:
            lines.extend(_round_record_lines(record))

    if case_run.evidence_ledger.items:
        lines.extend(["", "SUPPORTING_TYPED_EVIDENCE_IDS_DO_NOT_COPY_AS_CLAIMS:"])
        for item in case_run.evidence_ledger.items:
            lines.extend(
                [
                    (
                        f"- {item.evidence_id}: basis={item.basis}; "
                        f"family={item.family}; observable={item.observable}; "
                        f"artifacts={', '.join(item.artifact_refs)}"
                    ),
                    f"  claim: {item.claim}",
                    f"  context: {item.context}",
                    f"  supporting_summary: {item.summary}",
                    f"  basis/support/status: {item.basis}/{item.support}/{item.status}",
                    f"  observable_tags: {', '.join(item.observable_tags)}",
                    f"  metrics: {_truncate(json.dumps(item.metrics, ensure_ascii=False), 1200)}",
                    f"  limits: {'; '.join(item.limits)}",
                ]
            )
    artifact_lines = _artifact_observable_lines(case_run)
    if artifact_lines:
        lines.extend(["", "SUPPORTING_ARTIFACT_OBSERVABLES:", *artifact_lines])
    return "\n".join(lines)


def build_mas_strict_source_text(case_run: CaseRun) -> str:
    lines = [
        f"CASE_ID: {_public_case_id(case_run)}",
        f"SMILES: {case_run.input.smiles}",
        f"USER_QUERY: {case_run.input.user_query}",
        f"STATUS: {case_run.status}",
        "",
    ]
    if case_run.final_answer is not None:
        lines.extend(
            [
                "FINAL_ANSWER:",
                _truncate(case_run.final_answer.answer, 1400),
                "",
            ]
        )
    prediction_lines = _mechanism_prediction_lines(case_run)
    if prediction_lines:
        lines.extend(["MECHANISM_PREDICTIONS:", *prediction_lines, ""])
    conclusion_lines = _conclusion_ledger_lines(case_run)
    if conclusion_lines:
        lines.extend(["CONCLUSION_LEDGER:", *conclusion_lines, ""])
    final_diagnoses = case_run.diagnosis_units
    if final_diagnoses:
        lines.append("FINAL_DIAGNOSIS_UNITS:")
        for diagnosis in final_diagnoses[:10]:
            lines.extend(
                [
                    (
                        f"- {diagnosis.diagnosis_id}: status={diagnosis.status}; "
                        f"mechanism={diagnosis.mechanism}"
                    ),
                    f"  reasoning: {_truncate(diagnosis.reasoning_summary, 360)}",
                    "  evidence_refs: " + ", ".join(diagnosis.evidence_refs),
                    "  missing_or_unresolved: "
                    + "; ".join(diagnosis.missing_or_unresolved[:6]),
                ]
            )
        lines.append("")
    if case_run.evidence_ledger.items:
        lines.append("SOURCE_EVIDENCE:")
        for item in case_run.evidence_ledger.items[:20]:
            lines.extend(
                [
                    (
                        f"- {item.evidence_id}: agent={item.agent_name}; "
                        f"capability={item.capability_id}; basis={item.basis}; "
                        f"support={item.support}; status={item.status}"
                    ),
                    f"  claim: {_truncate(item.claim, 300)}",
                    f"  summary: {_truncate(item.summary, 300)}",
                    f"  limits: {'; '.join(_truncate(limit, 140) for limit in item.limits[:3])}",
                ]
            )
    return "\n".join(lines)


def _public_case_id(case_run: CaseRun) -> str:
    return public_case_id_from_metadata(case_run.input.metadata)


def _mechanism_prediction_lines(case_run: CaseRun) -> list[str]:
    predictions = case_run.mechanism_predictions
    if not predictions and case_run.final_answer is not None:
        predictions = case_run.final_answer.mechanism_predictions
    lines: list[str] = []
    for item in predictions[:5]:
        lines.extend(
            [
                (
                    f"- rank={item.rank}; label={item.label}; "
                    f"confidence={item.confidence}; "
                    f"differential_priority={item.differential_priority}; "
                    f"plausibility_confidence={item.plausibility_confidence}; "
                    f"claim_status={item.claim_status}; "
                    f"support_strength={item.support_strength}; "
                    f"evidence_support={item.evidence_support}"
                ),
                "  evidence_refs: " + ", ".join(item.evidence_refs),
                "  validation_needed: "
                + "; ".join(_truncate(text, 180) for text in item.validation_needed[:4]),
                "  limitations: "
                + "; ".join(_truncate(text, 180) for text in item.limitations[:4]),
            ]
        )
        if item.evidence_details:
            for detail in item.evidence_details[:4]:
                lines.extend(
                    [
                        "  evidence_detail:",
                        f"    finding: {_truncate(detail.finding, 220)}",
                        f"    warrant: {_truncate(detail.warrant, 220)}",
                        f"    boundary: {_truncate(detail.boundary, 220)}",
                        "    evidence_refs: " + ", ".join(detail.evidence_refs),
                    ]
                )
        else:
            lines.append(
                "  evidence: "
                + "; ".join(_truncate(text, 220) for text in item.evidence[:4])
            )
    return lines


def _conclusion_ledger_lines(case_run: CaseRun) -> list[str]:
    ledger = case_run.conclusion_ledger
    if ledger is None:
        return []
    lines = [
        f"ledger_id: {ledger.ledger_id}",
        f"round_id: {ledger.round_id}",
    ]
    if ledger.evidence_conclusions:
        lines.append("evidence_conclusions:")
        for item in ledger.evidence_conclusions:
            lines.extend(
                [
                    (
                        f"- {item.conclusion_id}: direction={item.direction}; "
                        f"basis={item.basis}; mechanism_family={item.mechanism_family or 'none'}"
                    ),
                    f"  statement: {_truncate(item.statement, 360)}",
                    f"  context: {_truncate(item.context, 240)}",
                    "  source_evidence_refs: " + ", ".join(item.source_evidence_refs),
                    f"  limits: {'; '.join(_truncate(limit, 160) for limit in item.limits[:4])}",
                ]
            )
    if ledger.diagnosis_conclusions:
        lines.append("diagnosis_conclusions:")
        for item in ledger.diagnosis_conclusions:
            lines.extend(
                [
                    (
                        f"- {item.conclusion_id}: status={item.status}; "
                        f"mechanism={item.mechanism}"
                    ),
                    f"  statement: {_truncate(item.statement, 360)}",
                    f"  reasoning: {_truncate(item.reasoning_summary, 420)}",
                    "  source_evidence_refs: " + ", ".join(item.source_evidence_refs),
                    "  missing_or_unresolved: " + "; ".join(item.missing_or_unresolved[:6]),
                    "  scope_limits: "
                    + "; ".join(_truncate(limit, 160) for limit in item.scope_limits[:4]),
                ]
            )
    if ledger.pending_questions:
        lines.append("pending_questions:")
        for item in ledger.pending_questions:
            lines.extend(
                [
                    f"- {item.question_id}: {_truncate(item.question, 320)}",
                    "  needed_evidence: " + "; ".join(item.needed_evidence[:6]),
                    "  source_evidence_refs: " + ", ".join(item.source_evidence_refs),
                ]
            )
    if ledger.policy_notes:
        lines.append("policy_notes:")
        lines.extend(f"- {_truncate(item, 220)}" for item in ledger.policy_notes[:6])
    return lines


def _round_record_lines(record: RoundRecord) -> list[str]:
    lines = [f"\n[{record.round_id}]"]
    if record.planner_decision is not None:
        decision = record.planner_decision
        lines.extend(
            [
                f"planner_action: {decision.action}",
                f"planner_current_hypothesis: {decision.current_hypothesis}",
                f"planner_confidence: {decision.confidence}",
                f"planner_diagnosis: {_truncate(decision.diagnosis, 1400)}",
            ]
        )
        if decision.final_answer_draft:
            lines.append(
                f"planner_final_answer_draft: {_truncate(decision.final_answer_draft, 1800)}"
            )
        if decision.unresolved_gaps:
            lines.append("planner_unresolved_gaps: " + "; ".join(decision.unresolved_gaps[:8]))
    for report in record.agent_reports:
        route = None
        if report.operational_note is not None:
            route = report.operational_note.selected_route
        lines.extend(
            [
                f"agent_report: {report.report_id}",
                f"  agent/status/route: {report.agent_name}/{report.status}/{route or 'unknown'}",
                f"  task: {_truncate(report.task_received, 500)}",
                f"  readable_report: {_truncate(report.planner_readable_report, 1400)}",
                "  evidence_refs: " + ", ".join(item.evidence_id for item in report.evidence_units),
                "  artifact_refs: "
                + ", ".join(item.artifact_id for item in report.artifact_updates),
            ]
        )
    return lines


def _truncate(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _artifact_observable_lines(case_run: CaseRun) -> list[str]:
    interesting_kinds = {
        "amesp_baseline_bundle",
        "conformer_bundle",
        "torsion_snapshot_bundle",
        "targeted_localized_orbital_bundle",
        "targeted_natural_orbital_bundle",
    }
    lines: list[str] = []
    for artifact in case_run.artifact_manifest.artifacts:
        if artifact.kind not in interesting_kinds:
            continue
        bundle = artifact.metadata.get("bundle")
        if not isinstance(bundle, dict):
            continue
        parsed = bundle.get("parsed_observables")
        if not isinstance(parsed, dict):
            continue
        lines.append(f"- {artifact.artifact_id}: kind={artifact.kind}; status={artifact.status}")
        descriptors = parsed.get("available_descriptors") or []
        if descriptors:
            lines.append("  available_descriptors: " + ", ".join(str(item) for item in descriptors))
        missing = parsed.get("missing_descriptors") or parsed.get("missing_deliverables") or []
        if missing:
            lines.append("  missing: " + ", ".join(str(item) for item in missing))
        bright = parsed.get("bright_state")
        if isinstance(bright, dict) and bright:
            lines.append(
                "  bright_state: "
                f"S{bright.get('state_index', 'n/a')} "
                f"{bright.get('excitation_energy_ev', 'n/a')} eV "
                f"f={bright.get('oscillator_strength', 'n/a')}"
            )
        summary = parsed.get("route_summary")
        if isinstance(summary, dict) and summary:
            lines.append(
                "  route_summary: "
                + _truncate(json.dumps(summary, ensure_ascii=False), 900)
            )
        records = parsed.get("route_records") or []
        if isinstance(records, list):
            extrema = _oscillator_extrema([item for item in records if isinstance(item, dict)])
            if extrema is not None:
                lines.append(
                    "  oscillator_extrema: "
                    f"min f={extrema[0][0]} at angle={extrema[0][1]}; "
                    f"max f={extrema[1][0]} at angle={extrema[1][1]}"
                )
    return lines


def _oscillator_extrema(
    records: list[dict[str, object]],
) -> tuple[tuple[object, object], tuple[object, object]] | None:
    usable = [
        (
            item.get("first_oscillator_strength"),
            item.get("target_angle_deg", item.get("initial_angle_deg", "n/a")),
        )
        for item in records
        if item.get("first_oscillator_strength") is not None
    ]
    if not usable:
        return None
    return min(usable, key=lambda item: float(item[0])), max(
        usable,
        key=lambda item: float(item[0]),
    )


def _eval_case_like(case_run: CaseRun) -> EvalCase:
    return EvalCase(
        case_id=case_run.case_id,
        public_case_id=_public_case_id(case_run),
        smiles=case_run.input.smiles,
        user_query=case_run.input.user_query,
    )
