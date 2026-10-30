"""Concrete Planner hypotheses and receipts; no prescribed scientific scores."""

from typing import Annotated, Literal

from pydantic import Field, StrictInt, StringConstraints

from mechcal.schemas.base import MechCALModel

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class LeadingComparison(MechCALModel):
    record_type: str = "LeadingComparison"
    leader: Text
    challenger: Text
    leading_hypothesis: Text
    alternative_hypothesis: Text
    basis_for_preference: Text
    preference_status: Literal["evidence_favors_leader", "prior_only", "unresolved"]
    evidence_refs: list[str] = Field(default_factory=list)
    previous_basis_update: Text


class HypothesisTest(MechCALModel):
    record_type: str = "HypothesisTest"
    target_label: Text
    alternative_label: Text
    tested_claim: Text
    relation_to_current_comparison: Text
    # possible_outcomes on InvestigationPlan supplies prospective observations.


class TestAssessment(MechCALModel):
    record_type: str = "TestAssessment"
    matched_outcome_indices: list[StrictInt] = Field(
        default_factory=list,
        description="Zero-based indices into the original plan.possible_outcomes; [] for an unexpected or unavailable observation.",
    )
    observed_result: Text
    basis_effect: Literal["strengthened", "weakened", "unchanged", "unresolved"]
    comparison_effect: Literal["favors_target", "favors_alternative", "non_discriminating", "unavailable"]
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="Scientific observation IDs from this execution only. Use [] when none were delivered; failures belong in operational_refs.",
    )
    operational_refs: list[str] = Field(
        default_factory=list,
        description="Operational-only IDs from this execution explaining failure or missing output, never scientific support or counterevidence.",
    )
    explanation: Text
