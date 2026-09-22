from __future__ import annotations

from typing import Literal

from pydantic import Field

from mechcal.schemas.artifacts import ArtifactManifest
from mechcal.schemas.base import JsonDict, MechCALModel, utc_now
from mechcal.schemas.claims import ClaimLedger
from mechcal.schemas.evidence import EvidenceLedger, HypothesisPortfolio
from mechcal.schemas.mechanisms import (
    ConclusionLedger,
    DiagnosisUnit,
    DifferentialMechanismPortfolio,
    MechanismAgendaCoverage,
    MechanismPrediction,
    MechanismPriorityDelta,
    MechanismProgram,
    MechanismSupportArgument,
    PhotophysicsReview,
)
from mechcal.schemas.reports import OperationalNote

CaseStatus = Literal["created", "running", "finalized", "stopped", "failed"]


class CaseInput(MechCALModel):
    record_type: str = "CaseInput"
    case_id: str
    smiles: str
    user_query: str
    metadata: JsonDict = Field(default_factory=dict)


class FinalAnswer(MechCALModel):
    record_type: str = "FinalAnswer"
    case_id: str
    current_hypothesis: str
    confidence: float
    answer: str
    evidence_refs: list[str] = Field(default_factory=list)
    round_refs: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    diagnosis_units: list[DiagnosisUnit] = Field(default_factory=list)
    mechanism_predictions: list[MechanismPrediction] = Field(default_factory=list)


class CaseRun(MechCALModel):
    record_type: str = "CaseRun"
    case_id: str
    input: CaseInput
    status: CaseStatus = "created"
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    current_round_id: str | None = None
    round_ids: list[str] = Field(default_factory=list)
    portfolio: HypothesisPortfolio = Field(default_factory=HypothesisPortfolio)
    evidence_ledger: EvidenceLedger
    claim_ledger: ClaimLedger | None = None
    mechanism_program: MechanismProgram | None = None
    mechanism_agenda_coverage: MechanismAgendaCoverage | None = None
    photophysics_review: PhotophysicsReview | None = None
    conclusion_ledger: ConclusionLedger | None = None
    differential_mechanism_portfolio: DifferentialMechanismPortfolio | None = None
    mechanism_priority_deltas: list[MechanismPriorityDelta] = Field(default_factory=list)
    mechanism_support_arguments: list[MechanismSupportArgument] = Field(default_factory=list)
    mechanism_predictions: list[MechanismPrediction] = Field(default_factory=list)
    diagnosis_units: list[DiagnosisUnit] = Field(default_factory=list)
    artifact_manifest: ArtifactManifest
    operational_notes: list[OperationalNote] = Field(default_factory=list)
    final_answer: FinalAnswer | None = None
    runtime: JsonDict = Field(default_factory=dict)

    def touch(self, **updates: object) -> CaseRun:
        return self.model_copy(update={**updates, "updated_at": utc_now()})
