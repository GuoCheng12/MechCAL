from __future__ import annotations

from pydantic import BaseModel, Field

from mechcal.schemas import (
    ArtifactRecord,
    CaseRun,
    ClaimLedger,
    EvidenceUnit,
    OperationalNote,
)


class PlannerContextView(BaseModel):
    case_id: str
    smiles: str
    user_query: str
    round_index: int
    max_rounds: int
    current_hypothesis: str
    confidence: float
    covered_evidence_families: list[str] = Field(default_factory=list)
    claim_ledger: ClaimLedger | None = None
    open_claim_ids: list[str] = Field(default_factory=list)
    blocked_claim_ids: list[str] = Field(default_factory=list)
    coverage_debt_hypotheses: list[str] = Field(default_factory=list)
    recent_evidence: list[EvidenceUnit] = Field(default_factory=list)
    reusable_artifacts: list[ArtifactRecord] = Field(default_factory=list)
    operational_notes: list[OperationalNote] = Field(default_factory=list)
    unresolved_gaps: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def build_planner_context(
    case_run: CaseRun,
    *,
    round_index: int,
    max_rounds: int,
    unresolved_gaps: list[str] | None = None,
) -> PlannerContextView:
    sorted_hypotheses = case_run.portfolio.sorted_hypotheses()
    top = sorted_hypotheses[0] if sorted_hypotheses else None
    claim_ledger = case_run.claim_ledger
    return PlannerContextView(
        case_id=case_run.case_id,
        smiles=case_run.input.smiles,
        user_query=case_run.input.user_query,
        round_index=round_index,
        max_rounds=max_rounds,
        current_hypothesis=case_run.portfolio.current,
        confidence=top.confidence if top is not None else 0.0,
        covered_evidence_families=list(case_run.evidence_ledger.covered_families()),
        claim_ledger=claim_ledger,
        open_claim_ids=claim_ledger.open_claim_ids() if claim_ledger is not None else [],
        blocked_claim_ids=(
            claim_ledger.blocked_claim_ids() if claim_ledger is not None else []
        ),
        coverage_debt_hypotheses=(
            claim_ledger.coverage_debt_hypotheses()
            if claim_ledger is not None
            else []
        ),
        recent_evidence=case_run.evidence_ledger.recent_items(),
        reusable_artifacts=[
            artifact
            for artifact in case_run.artifact_manifest.artifacts
            if artifact.status in {"available", "partial"}
        ],
        operational_notes=list(case_run.operational_notes[-12:]),
        unresolved_gaps=list(unresolved_gaps or []),
    )
