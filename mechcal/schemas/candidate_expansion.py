"""Proposals are investigation records, never scientific evidence or rankings."""

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from mechcal.schemas.base import MechCALModel

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CandidateProposal(MechCALModel):
    record_type: str = "CandidateProposal"
    hypothesis: Text
    related_labels: list[str] = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)
    observation_basis: Text
    distinguishing_observation: Text
    boundary: Text


class CandidateExpansion(MechCALModel):
    record_type: str = "CandidateExpansion"
    proposals: list[CandidateProposal] = Field(default_factory=list, max_length=2)
    rationale: Text


class CandidateResponse(MechCALModel):
    record_type: str = "CandidateResponse"
    candidate_id: Text
    decision: Literal["investigate", "merge", "defer", "reject"]
    reason: Text
    question_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
