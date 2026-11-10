"""Planner-authored reasons; no scientific scores or automatic ordering."""

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from mechcal.schemas.base import MechCALModel

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CandidateBasis(MechCALModel):
    record_type: str = "CandidateBasis"
    label: str
    prior_rationale: Text
    supported_observation: Text
    positive_refs: list[str] = Field(default_factory=list)
    counter_refs: list[str] = Field(default_factory=list)
    inference_gap: Text


class BoundaryComparison(MechCALModel):
    record_type: str = "BoundaryComparison"
    included_label: str
    excluded_label: str
    outcome: Literal["favors_included", "favors_excluded", "unresolved", "coexisting_not_separated"]
    basis: Literal["observations", "prior_only", "mixed", "unresolved"]
    evidence_refs: list[str] = Field(default_factory=list)
    common_standard: Text
    reason_for_order: Text
    remaining_difference: Text


class ConsideredRoute(MechCALModel):
    record_type: str = "ConsideredRoute"
    capability_id: str
    tool_args: dict = Field(default_factory=dict)
    disposition: Literal["blocked", "not_useful", "useful"]
    reason: Text
    preparation_assessment: Text | None = None


class ContinuationAssessment(MechCALModel):
    record_type: str = "ContinuationAssessment"
    scope: Literal["bounded_ranking"]
    remaining_uncertainty: Text
    routes: list[ConsideredRoute] = Field(default_factory=list)
    decision_reason: Text
