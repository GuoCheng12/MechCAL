"""Explanations for a frozen Planner proposal, without writable ranking fields."""

from typing import Literal

from pydantic import Field

from mechcal.schemas.hypothesis_tests import Text
from mechcal.schemas.judgment_commands import CommandModel


class LeaderExplanation(CommandModel):
    challenger: Text
    leading_hypothesis: Text
    alternative_hypothesis: Text
    basis_for_preference: Text
    preference_status: Literal["evidence_favors_leader", "prior_only", "unresolved"]
    evidence_refs: list[str] = Field(default_factory=list)
    previous_basis_update: Text


class BoundaryExplanation(CommandModel):
    target_id: Text
    outcome: Literal["favors_included", "favors_excluded", "unresolved", "coexisting_not_separated"]
    basis: Literal["observations", "prior_only", "mixed", "unresolved"]
    evidence_refs: list[str] = Field(default_factory=list)
    common_standard: Text
    reason_for_order: Text
    remaining_difference: Text


class RankingExplanations(CommandModel):
    action: Literal["explain_ranking"]
    draft_id: Text
    leading: LeaderExplanation | None
    boundaries: list[BoundaryExplanation]
