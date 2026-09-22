from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from mechcal.schemas.base import JsonDict, MechCALModel

EvidenceFamily = Literal[
    "geometry_precondition",
    "state_ordering_brightness",
    "torsion_sensitivity",
    "conformer_sensitivity",
    "charge_localization",
    "raw_artifact_inspection",
]
EvidenceRelation = Literal["supports", "challenges", "mixed", "neutral", "unknown"]
EvidenceStatus = Literal["present", "partial", "failed", "unsupported", "missing"]
EvidenceBasis = Literal["computed", "proxy", "checklist"]
EvidenceSupport = Literal["supports", "weakens", "unresolved", "not_applicable"]
HypothesisStatus = Literal[
    "plausible",
    "pending",
    "screened",
    "blocked",
    "dropped",
    "unknown",
]


class HypothesisEntry(MechCALModel):
    record_type: str = "HypothesisEntry"
    name: str
    confidence: float = 0.0
    differential_priority: float | None = None
    evidence_support: str = "unsupported"
    claim_status: str = "candidate_requires_validation"
    status: HypothesisStatus = "pending"
    rationale: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    validation_needed: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, value: float) -> float:
        confidence = float(value)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("HypothesisEntry.confidence must be between 0.0 and 1.0.")
        return confidence

    @field_validator("differential_priority")
    @classmethod
    def _optional_priority_range(cls, value: float | None) -> float | None:
        if value is None:
            return None
        priority = float(value)
        if not 0.0 <= priority <= 1.0:
            raise ValueError(
                "HypothesisEntry.differential_priority must be between 0.0 and 1.0."
            )
        return priority


class HypothesisPortfolio(MechCALModel):
    record_type: str = "HypothesisPortfolio"
    hypotheses: list[HypothesisEntry] = Field(default_factory=list)
    current: str = "unknown"
    runner_up: str | None = None

    def sorted_hypotheses(self) -> list[HypothesisEntry]:
        return sorted(
            self.hypotheses,
            key=lambda item: (
                item.differential_priority
                if item.differential_priority is not None
                else item.confidence
            ),
            reverse=True,
        )


class EvidenceUnit(MechCALModel):
    record_type: str = "EvidenceUnit"
    evidence_id: str
    round_id: str
    source_report_id: str
    agent_name: str
    capability_id: str
    claim: str
    context: str
    basis: EvidenceBasis
    support: EvidenceSupport = "unresolved"
    summary: str
    limits: list[str] = Field(default_factory=list)
    observable: str | None = None
    artifact_refs: list[str] = Field(default_factory=list)
    metrics: JsonDict = Field(default_factory=dict)

    # Compatibility fields for the existing ClaimLedger reducer. They remain
    # typed so old coverage logic can migrate one call site at a time.
    family: EvidenceFamily
    relation: EvidenceRelation = "unknown"
    status: EvidenceStatus = "present"
    claim_refs: list[str] = Field(default_factory=list)
    hypothesis_refs: list[str] = Field(default_factory=list)
    observable_tags: list[str] = Field(default_factory=list)

    @field_validator(
        "evidence_id",
        "round_id",
        "source_report_id",
        "agent_name",
        "capability_id",
        "claim",
        "context",
        "summary",
    )
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("EvidenceUnit text fields must not be empty.")
        return normalized

    @field_validator("limits")
    @classmethod
    def _non_empty_limits(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


class EvidenceLedger(MechCALModel):
    record_type: str = "EvidenceLedger"
    case_id: str
    items: list[EvidenceUnit] = Field(default_factory=list)

    def with_items(self, items: list[EvidenceUnit]) -> EvidenceLedger:
        by_id = {item.evidence_id: item for item in self.items}
        by_id.update({item.evidence_id: item for item in items})
        return self.model_copy(update={"items": list(by_id.values())})

    def covered_families(self) -> list[EvidenceFamily]:
        families = {
            item.family
            for item in self.items
            if item.status in {"present", "partial"}
        }
        return sorted(families)

    def recent_items(self, limit: int = 6) -> list[EvidenceUnit]:
        return self.items[-max(1, limit):]
