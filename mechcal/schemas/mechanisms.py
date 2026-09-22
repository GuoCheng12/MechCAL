from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from mechcal.schemas.base import MechCALModel
from mechcal.schemas.evidence import EvidenceBasis

EvidenceQuestionStatus = Literal[
    "unresolved",
    "computable",
    "checklist_only",
    "covered",
    "blocked",
]
MechanismAgentName = Literal["macro", "microscopic"]
DiagnosisStatus = Literal[
    "computed_supported",
    "proxy_supported",
    "plausible",
    "weakened",
    "rejected",
    "underdetermined",
]
ConclusionConfidence = Literal["low", "medium", "high"]
ConclusionDirection = Literal["supports", "weakens", "unresolved", "not_applicable"]
PhotophysicsAxisStatus = Literal[
    "covered",
    "missing",
    "boundary_only",
    "not_applicable",
]
MechanismSupportLevel = Literal["strong", "partial", "weak", "unsupported"]
MechanismSupportAuditStatus = Literal[
    "not_reviewed",
    "accepted",
    "boundary_added",
    "downgraded",
    "unsupported",
    "invalid_ref",
    "overclaim",
]
MechanismClaimStatus = Literal[
    "supported_claim",
    "partially_supported_candidate",
    "candidate_requires_validation",
    "underdetermined",
]
MechanismPredictionSupportStrength = Literal[
    "strong",
    "partial",
    "weak_or_proxy",
    "unsupported",
]
DifferentialTriggerStatus = Literal[
    "triggered",
    "weak_trigger",
    "not_triggered",
    "contradicted",
]
DifferentialSupportStrength = Literal[
    "unsupported",
    "weak_proxy",
    "partial",
    "strong",
]
DifferentialClaimStatus = Literal[
    "candidate_requires_validation",
    "partially_supported",
    "supported",
    "weakened",
]
MechanismPriorityEffect = Literal[
    "increase",
    "decrease",
    "neutral",
]
MechanismSupportEffect = Literal[
    "supports",
    "weakens",
    "boundary_only",
    "not_applicable",
]
MechanismSupportChange = Literal[
    "unchanged",
    "upgraded",
    "downgraded",
    "weakened",
    "newly_supported",
]
AgendaCoverageStatus = Literal[
    "under_screened",
    "screened",
    "screened_out",
    "needs_follow_up",
]
AgendaUrgency = Literal["high", "medium", "low"]


class EvidenceMechanismAttributionUpdate(MechCALModel):
    record_type: str = "EvidenceMechanismAttributionUpdate"
    label: str
    priority_effect: MechanismPriorityEffect = "neutral"
    support_effect: MechanismSupportEffect = "not_applicable"
    warrant: str
    boundary: str

    @field_validator("label", "warrant", "boundary")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("EvidenceMechanismAttributionUpdate text must not be empty.")
        return normalized


class EvidenceMechanismAttribution(MechCALModel):
    record_type: str = "EvidenceMechanismAttribution"
    evidence_id: str
    updates: list[EvidenceMechanismAttributionUpdate] = Field(default_factory=list)

    @field_validator("evidence_id")
    @classmethod
    def _non_empty_evidence_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("EvidenceMechanismAttribution.evidence_id must not be empty.")
        return normalized


class MechanismPriorityDelta(MechCALModel):
    record_type: str = "MechanismPriorityDelta"
    round_id: str
    label: str
    previous_priority: float = 0.0
    priority_delta: float = 0.0
    new_priority: float = 0.0
    delta_reason: str
    evidence_refs_added: list[str] = Field(default_factory=list)
    support_change: MechanismSupportChange = "unchanged"
    ranking_stability_note: str | None = None

    @field_validator("round_id", "label", "delta_reason")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("MechanismPriorityDelta text fields must not be empty.")
        return normalized

    @field_validator("previous_priority", "new_priority")
    @classmethod
    def _priority_range(cls, value: float) -> float:
        priority = float(value)
        if not 0.0 <= priority <= 1.0:
            raise ValueError("MechanismPriorityDelta priority fields must be 0-1.")
        return priority

    @field_validator("priority_delta")
    @classmethod
    def _delta_range(cls, value: float) -> float:
        delta = float(value)
        if not -1.0 <= delta <= 1.0:
            raise ValueError("MechanismPriorityDelta.priority_delta must be -1 to 1.")
        return delta

    @field_validator("evidence_refs_added")
    @classmethod
    def _non_empty_list_items(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]

    @field_validator("ranking_stability_note")
    @classmethod
    def _optional_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class AgendaCoverageItem(MechCALModel):
    record_type: str = "AgendaCoverageItem"
    label: str
    coverage_status: AgendaCoverageStatus
    why_relevant: str
    missing_evidence: str
    suggested_question: str
    suggested_route: str
    urgency: AgendaUrgency = "medium"
    boundary: str

    @field_validator(
        "label",
        "why_relevant",
        "missing_evidence",
        "suggested_question",
        "suggested_route",
        "boundary",
    )
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("AgendaCoverageItem text fields must not be empty.")
        return normalized


class ScreenedOutMechanism(MechCALModel):
    record_type: str = "ScreenedOutMechanism"
    label: str
    reason: str

    @field_validator("label", "reason")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("ScreenedOutMechanism text fields must not be empty.")
        return normalized


class LowMarginMechanismCompetition(MechCALModel):
    record_type: str = "LowMarginMechanismCompetition"
    labels: list[str]
    reason: str
    suggested_disambiguating_route: str

    @field_validator("labels")
    @classmethod
    def _labels(cls, value: list[str]) -> list[str]:
        labels = [item.strip() for item in value if item.strip()]
        if len(labels) < 2:
            raise ValueError("LowMarginMechanismCompetition requires at least two labels.")
        return labels

    @field_validator("reason", "suggested_disambiguating_route")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("LowMarginMechanismCompetition text fields must not be empty.")
        return normalized


class RecommendedAgendaRoute(MechCALModel):
    record_type: str = "RecommendedAgendaRoute"
    route: str
    targets: list[str] = Field(default_factory=list)
    reason: str
    capability_ids: list[str] = Field(default_factory=list)

    @field_validator("route", "reason")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("RecommendedAgendaRoute text fields must not be empty.")
        return normalized

    @field_validator("targets", "capability_ids")
    @classmethod
    def _non_empty_list_items(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


class MechanismAgendaCoverage(MechCALModel):
    record_type: str = "MechanismAgendaCoverage"
    coverage_id: str
    case_id: str
    round_id: str
    coverage_summary: str
    agenda_items: list[AgendaCoverageItem] = Field(default_factory=list)
    screened_out: list[ScreenedOutMechanism] = Field(default_factory=list)
    low_margin_competitions: list[LowMarginMechanismCompetition] = Field(default_factory=list)
    recommended_next_routes: list[RecommendedAgendaRoute] = Field(default_factory=list)

    @field_validator("coverage_id", "case_id", "round_id", "coverage_summary")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("MechanismAgendaCoverage identity fields are required.")
        return normalized

    @model_validator(mode="after")
    def _has_operational_content(self) -> MechanismAgendaCoverage:
        if not self.agenda_items and not self.recommended_next_routes:
            raise ValueError(
                "MechanismAgendaCoverage must contain agenda_items or recommended_next_routes."
            )
        return self


class DifferentialMechanismPortfolioRow(MechCALModel):
    record_type: str = "DifferentialMechanismPortfolioRow"
    label: str
    trigger_status: DifferentialTriggerStatus = "not_triggered"
    differential_priority: float = 0.0
    support_strength: DifferentialSupportStrength = "unsupported"
    claim_status: DifferentialClaimStatus = "candidate_requires_validation"
    positive_evidence_refs: list[str] = Field(default_factory=list)
    negative_evidence_refs: list[str] = Field(default_factory=list)
    missing_validation: list[str] = Field(default_factory=list)
    rationale: str

    @field_validator("label", "rationale")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("DifferentialMechanismPortfolioRow text must not be empty.")
        return normalized

    @field_validator("differential_priority")
    @classmethod
    def _priority_range(cls, value: float) -> float:
        priority = float(value)
        if not 0.0 <= priority <= 1.0:
            raise ValueError(
                "DifferentialMechanismPortfolioRow.differential_priority must be 0-1."
            )
        return priority

    @field_validator("positive_evidence_refs", "negative_evidence_refs", "missing_validation")
    @classmethod
    def _non_empty_list_items(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]

    @model_validator(mode="after")
    def _supported_claim_requires_evidence(self) -> DifferentialMechanismPortfolioRow:
        if self.claim_status == "supported" and not self.positive_evidence_refs:
            raise ValueError("supported portfolio rows must cite positive_evidence_refs.")
        if self.support_strength == "strong" and not self.positive_evidence_refs:
            raise ValueError("strong support rows must cite positive_evidence_refs.")
        if self.claim_status == "partially_supported" and not self.positive_evidence_refs:
            raise ValueError(
                "partially_supported portfolio rows must cite positive_evidence_refs."
            )
        return self


class DifferentialMechanismPortfolio(MechCALModel):
    record_type: str = "DifferentialMechanismPortfolio"
    portfolio_id: str
    case_id: str
    round_id: str
    rows: list[DifferentialMechanismPortfolioRow] = Field(default_factory=list)
    evidence_attributions: list[EvidenceMechanismAttribution] = Field(default_factory=list)
    policy_notes: list[str] = Field(default_factory=list)

    @field_validator("portfolio_id", "case_id", "round_id")
    @classmethod
    def _non_empty_identity(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("DifferentialMechanismPortfolio identity fields are required.")
        return normalized

    @field_validator("policy_notes")
    @classmethod
    def _non_empty_policy_notes(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]

    @model_validator(mode="after")
    def _unique_labels(self) -> DifferentialMechanismPortfolio:
        labels = [row.label for row in self.rows]
        if len(labels) != len(set(labels)):
            raise ValueError("DifferentialMechanismPortfolio rows must have unique labels.")
        return self


class MechanismSupportArgument(MechCALModel):
    record_type: str = "MechanismSupportArgument"
    support_id: str
    target_label: str
    support_level: MechanismSupportLevel
    finding: str
    warrant: str
    boundary: str
    observation_refs: list[str] = Field(default_factory=list)
    audit_status: MechanismSupportAuditStatus = "not_reviewed"
    audit_notes: list[str] = Field(default_factory=list)

    @field_validator(
        "support_id",
        "target_label",
        "finding",
        "warrant",
        "boundary",
    )
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("MechanismSupportArgument text fields must not be empty.")
        return normalized

    @field_validator("observation_refs", "audit_notes")
    @classmethod
    def _non_empty_list_items(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]

    @model_validator(mode="after")
    def _requires_observation_refs(self) -> MechanismSupportArgument:
        if not self.observation_refs:
            raise ValueError("MechanismSupportArgument must cite observation_refs.")
        return self


class MechanismSupportAudit(MechCALModel):
    record_type: str = "MechanismSupportAudit"
    support_id: str
    audit_status: MechanismSupportAuditStatus
    recommended_support_level: MechanismSupportLevel | None = None
    audit_notes: list[str] = Field(default_factory=list)
    boundary_patch: str | None = None

    @field_validator("support_id")
    @classmethod
    def _non_empty_support_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("MechanismSupportAudit.support_id must not be empty.")
        return normalized

    @field_validator("audit_notes")
    @classmethod
    def _non_empty_audit_notes(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


class MechanismPredictionEvidence(MechCALModel):
    record_type: str = "MechanismPredictionEvidence"
    finding: str
    warrant: str
    boundary: str
    evidence_refs: list[str] = Field(default_factory=list)

    @field_validator("finding", "warrant", "boundary")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("MechanismPredictionEvidence text fields must not be empty.")
        return normalized

    @field_validator("evidence_refs")
    @classmethod
    def _non_empty_list_items(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


class MechanismPrediction(MechCALModel):
    record_type: str = "MechanismPrediction"
    label: str
    rank: int
    confidence: float = 0.0
    differential_priority: float | None = None
    plausibility_confidence: float | None = None
    claim_status: MechanismClaimStatus = "candidate_requires_validation"
    support_strength: MechanismPredictionSupportStrength = "unsupported"
    evidence_support: MechanismPredictionSupportStrength | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    evidence_details: list[MechanismPredictionEvidence] = Field(default_factory=list)
    validation_needed: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    low_margin_group: str | None = None
    margin_to_next: float | None = None
    ranking_stability_note: str | None = None

    @field_validator("label")
    @classmethod
    def _non_empty_label(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("MechanismPrediction.label must not be empty.")
        return normalized

    @field_validator("rank")
    @classmethod
    def _positive_rank(cls, value: int) -> int:
        if value < 1:
            raise ValueError("MechanismPrediction.rank must be positive.")
        return value

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, value: float) -> float:
        confidence = float(value)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("MechanismPrediction.confidence must be between 0.0 and 1.0.")
        return confidence

    @field_validator("plausibility_confidence")
    @classmethod
    def _optional_confidence_range(cls, value: float | None) -> float | None:
        if value is None:
            return None
        confidence = float(value)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "MechanismPrediction.plausibility_confidence must be between 0.0 and 1.0."
            )
        return confidence

    @field_validator("differential_priority")
    @classmethod
    def _optional_priority_range(cls, value: float | None) -> float | None:
        if value is None:
            return None
        priority = float(value)
        if not 0.0 <= priority <= 1.0:
            raise ValueError(
                "MechanismPrediction.differential_priority must be between 0.0 and 1.0."
            )
        return priority

    @field_validator("margin_to_next")
    @classmethod
    def _optional_margin_range(cls, value: float | None) -> float | None:
        if value is None:
            return None
        margin = float(value)
        if not 0.0 <= margin <= 1.0:
            raise ValueError("MechanismPrediction.margin_to_next must be between 0.0 and 1.0.")
        return margin

    @field_validator("low_margin_group", "ranking_stability_note")
    @classmethod
    def _optional_non_empty_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("evidence_refs", "evidence", "validation_needed", "limitations")
    @classmethod
    def _non_empty_list_items(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


class CandidateMechanism(MechCALModel):
    record_type: str = "CandidateMechanism"
    mechanism_id: str
    label: str
    mechanism_family: str
    context: str
    rationale: str = ""

    @field_validator("mechanism_id", "label", "mechanism_family", "context")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("CandidateMechanism text fields must not be empty.")
        return normalized


class EvidenceQuestion(MechCALModel):
    record_type: str = "EvidenceQuestion"
    question_id: str
    mechanism_id: str
    question: str
    observable: str
    expected_basis: EvidenceBasis
    status: EvidenceQuestionStatus = "unresolved"
    acceptable_capability_ids: list[str] = Field(default_factory=list)
    missing_or_unresolved: list[str] = Field(default_factory=list)

    @field_validator("question_id", "mechanism_id", "question", "observable")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("EvidenceQuestion text fields must not be empty.")
        return normalized


class ComputationalRoute(MechCALModel):
    record_type: str = "ComputationalRoute"
    question_id: str
    capability_id: str
    agent_name: MechanismAgentName
    route: str
    expected_basis: EvidenceBasis

    @field_validator("question_id", "capability_id", "route")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("ComputationalRoute text fields must not be empty.")
        return normalized


class ScopeBoundary(MechCALModel):
    record_type: str = "ScopeBoundary"
    boundary_id: str
    statement: str

    @field_validator("boundary_id", "statement")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("ScopeBoundary text fields must not be empty.")
        return normalized


class MechanismProgram(MechCALModel):
    record_type: str = "MechanismProgram"
    program_id: str
    case_id: str
    round_id: str
    candidate_mechanisms: list[CandidateMechanism] = Field(default_factory=list)
    evidence_questions: list[EvidenceQuestion] = Field(default_factory=list)
    computational_routes: list[ComputationalRoute] = Field(default_factory=list)
    scope_boundaries: list[ScopeBoundary] = Field(default_factory=list)

    @field_validator("program_id", "case_id", "round_id")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("MechanismProgram identity fields must not be empty.")
        return normalized

    @model_validator(mode="after")
    def _routes_reference_questions(self) -> MechanismProgram:
        question_ids = {item.question_id for item in self.evidence_questions}
        missing = [
            item.question_id
            for item in self.computational_routes
            if item.question_id not in question_ids
        ]
        if missing:
            raise ValueError(
                "ComputationalRoute.question_id must reference an EvidenceQuestion: "
                + ", ".join(sorted(set(missing)))
            )
        return self


class DiagnosisUnit(MechCALModel):
    record_type: str = "DiagnosisUnit"
    diagnosis_id: str
    mechanism: str
    context: str
    status: DiagnosisStatus
    evidence_refs: list[str] = Field(default_factory=list)
    missing_or_unresolved: list[str] = Field(default_factory=list)
    reasoning_summary: str
    scope_limits: list[str] = Field(default_factory=list)

    @field_validator("diagnosis_id", "mechanism", "context", "reasoning_summary")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("DiagnosisUnit text fields must not be empty.")
        return normalized

    @model_validator(mode="after")
    def _status_evidence_contract(self) -> DiagnosisUnit:
        evidence_required = {
            "computed_supported",
            "proxy_supported",
            "weakened",
            "rejected",
        }
        if self.status in evidence_required and not self.evidence_refs:
            raise ValueError(f"{self.status} DiagnosisUnit must cite evidence_refs.")
        if self.status == "underdetermined" and not self.missing_or_unresolved:
            raise ValueError("underdetermined DiagnosisUnit must list unresolved needs.")
        return self


class PhotophysicsCoverageAxis(MechCALModel):
    record_type: str = "PhotophysicsCoverageAxis"
    axis_id: str
    label: str
    status: PhotophysicsAxisStatus
    evidence_refs: list[str] = Field(default_factory=list)
    rationale: str
    recommended_routes: list[str] = Field(default_factory=list)

    @field_validator("axis_id", "label", "rationale")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("PhotophysicsCoverageAxis text fields must not be empty.")
        return normalized


class MechanismHypothesisCard(MechCALModel):
    record_type: str = "MechanismHypothesisCard"
    hypothesis_id: str
    mechanism: str
    status: DiagnosisStatus
    support_evidence_refs: list[str] = Field(default_factory=list)
    weakening_evidence_refs: list[str] = Field(default_factory=list)
    missing_or_unresolved: list[str] = Field(default_factory=list)
    reasoning_summary: str
    scope_limits: list[str] = Field(default_factory=list)
    priority: float = 0.0

    @field_validator("hypothesis_id", "mechanism", "reasoning_summary")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("MechanismHypothesisCard text fields must not be empty.")
        return normalized

    @model_validator(mode="after")
    def _status_evidence_contract(self) -> MechanismHypothesisCard:
        evidence_required = {
            "computed_supported",
            "proxy_supported",
            "weakened",
            "rejected",
        }
        if (
            self.status in evidence_required
            and not self.support_evidence_refs
            and not self.weakening_evidence_refs
        ):
            raise ValueError(f"{self.status} MechanismHypothesisCard must cite evidence.")
        if self.status == "underdetermined" and not self.missing_or_unresolved:
            raise ValueError(
                "underdetermined MechanismHypothesisCard must list unresolved needs."
            )
        return self


class PhotophysicsReview(MechCALModel):
    record_type: str = "PhotophysicsReview"
    review_id: str
    case_id: str
    round_id: str
    coverage_axes: list[PhotophysicsCoverageAxis] = Field(default_factory=list)
    hypothesis_cards: list[MechanismHypothesisCard] = Field(default_factory=list)
    support_argument_audits: list[MechanismSupportAudit] = Field(default_factory=list)
    recommended_next_routes: list[str] = Field(default_factory=list)
    overclaim_warnings: list[str] = Field(default_factory=list)
    source_evidence_refs: list[str] = Field(default_factory=list)

    @field_validator("review_id", "case_id", "round_id")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("PhotophysicsReview identity fields must not be empty.")
        return normalized


class EvidenceConclusion(MechCALModel):
    record_type: str = "EvidenceConclusion"
    conclusion_id: str
    statement: str
    context: str
    mechanism_family: str | None = None
    direction: ConclusionDirection = "unresolved"
    basis: EvidenceBasis = "proxy"
    source_evidence_refs: list[str] = Field(default_factory=list)
    source_snippets: list[str] = Field(default_factory=list)
    limits: list[str] = Field(default_factory=list)
    confidence: ConclusionConfidence = "medium"

    @field_validator("conclusion_id", "statement", "context")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("EvidenceConclusion text fields must not be empty.")
        return normalized

    @field_validator("source_evidence_refs", "source_snippets", "limits")
    @classmethod
    def _non_empty_list_items(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


class DiagnosisConclusion(MechCALModel):
    record_type: str = "DiagnosisConclusion"
    conclusion_id: str
    mechanism: str
    status: DiagnosisStatus
    statement: str
    reasoning_summary: str
    source_evidence_refs: list[str] = Field(default_factory=list)
    source_snippets: list[str] = Field(default_factory=list)
    missing_or_unresolved: list[str] = Field(default_factory=list)
    scope_limits: list[str] = Field(default_factory=list)
    confidence: ConclusionConfidence = "medium"

    @field_validator("conclusion_id", "mechanism", "statement", "reasoning_summary")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("DiagnosisConclusion text fields must not be empty.")
        return normalized

    @field_validator(
        "source_evidence_refs",
        "source_snippets",
        "missing_or_unresolved",
        "scope_limits",
    )
    @classmethod
    def _non_empty_list_items(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]

    @model_validator(mode="after")
    def _status_evidence_contract(self) -> DiagnosisConclusion:
        evidence_required = {
            "computed_supported",
            "proxy_supported",
            "weakened",
            "rejected",
        }
        if self.status in evidence_required and not self.source_evidence_refs:
            raise ValueError(f"{self.status} DiagnosisConclusion must cite evidence refs.")
        if self.status == "underdetermined" and not self.missing_or_unresolved:
            raise ValueError("underdetermined DiagnosisConclusion must list unresolved needs.")
        return self


class PendingConclusionQuestion(MechCALModel):
    record_type: str = "PendingConclusionQuestion"
    question_id: str
    question: str
    needed_evidence: list[str] = Field(default_factory=list)
    source_evidence_refs: list[str] = Field(default_factory=list)

    @field_validator("question_id", "question")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("PendingConclusionQuestion text fields must not be empty.")
        return normalized

    @field_validator("needed_evidence", "source_evidence_refs")
    @classmethod
    def _non_empty_list_items(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


class ConclusionLedger(MechCALModel):
    record_type: str = "ConclusionLedger"
    ledger_id: str
    case_id: str
    round_id: str
    evidence_conclusions: list[EvidenceConclusion] = Field(default_factory=list)
    diagnosis_conclusions: list[DiagnosisConclusion] = Field(default_factory=list)
    pending_questions: list[PendingConclusionQuestion] = Field(default_factory=list)
    policy_notes: list[str] = Field(default_factory=list)

    @field_validator("ledger_id", "case_id", "round_id")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("ConclusionLedger identity fields must not be empty.")
        return normalized

    @field_validator("policy_notes")
    @classmethod
    def _non_empty_policy_notes(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]
