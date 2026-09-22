from __future__ import annotations

from typing import Literal

from pydantic import Field

from mechcal.schemas.base import MechCALModel
from mechcal.schemas.evidence import EvidenceRelation

ClaimCoverageStatus = Literal["uncovered", "covered", "blocked"]
ClaimRole = Literal["context", "proxy", "direct_prerequisite", "external_check"]
HypothesisCoverageStatus = Literal["open", "covered", "blocked"]


class ClaimDefinition(MechCALModel):
    record_type: str = "ClaimDefinition"
    claim_id: str
    hypothesis: str
    topic: str
    role: ClaimRole
    description: str
    covering_routes: list[str] = Field(default_factory=list)
    screening_claim: bool = True


class ClaimAssessment(MechCALModel):
    record_type: str = "ClaimAssessment"
    claim_id: str
    hypothesis: str
    topic: str
    role: ClaimRole
    coverage_status: ClaimCoverageStatus = "uncovered"
    relation: EvidenceRelation = "unknown"
    evidence_refs: list[str] = Field(default_factory=list)
    covered_routes: list[str] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)
    summary: str = ""


class HypothesisCoverageDebt(MechCALModel):
    record_type: str = "HypothesisCoverageDebt"
    hypothesis: str
    status: HypothesisCoverageStatus = "open"
    open_claim_ids: list[str] = Field(default_factory=list)
    blocked_claim_ids: list[str] = Field(default_factory=list)
    covered_claim_ids: list[str] = Field(default_factory=list)


class ClaimLedger(MechCALModel):
    record_type: str = "ClaimLedger"
    case_id: str
    claim_definitions: list[ClaimDefinition] = Field(default_factory=list)
    assessments: list[ClaimAssessment] = Field(default_factory=list)
    hypothesis_debts: list[HypothesisCoverageDebt] = Field(default_factory=list)

    def open_claim_ids(self) -> list[str]:
        return [
            item.claim_id
            for item in self.assessments
            if item.coverage_status == "uncovered"
        ]

    def blocked_claim_ids(self) -> list[str]:
        return [
            item.claim_id
            for item in self.assessments
            if item.coverage_status == "blocked"
        ]

    def coverage_debt_hypotheses(self) -> list[str]:
        return [
            item.hypothesis
            for item in self.hypothesis_debts
            if item.status in {"open", "blocked"}
        ]
