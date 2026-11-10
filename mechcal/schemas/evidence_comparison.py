"""Planner-authored comparison records, not deterministic scientific scores."""

from typing import Literal

from pydantic import Field

from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.schemas.base import MechCALModel
from mechcal.schemas.boundary_decisions import BoundaryComparison, CandidateBasis
from mechcal.schemas.hypothesis_tests import HypothesisTest, LeadingComparison

# Shared by the live schema and the runtime decision-value check.
MATERIAL_OUTCOME_CHANGES = {"ranking": ("promote", "demote"), "support": ("strengthen", "weaken")}


class OutcomeExpectation(MechCALModel):
    record_type: str = "OutcomeExpectation"
    observation: str = Field(min_length=1)
    judgment_effect: str = Field(min_length=1)
    decision_effect: Literal["ranking", "support", "preparation", "no_change"] | None = None
    affected_labels: list[str] = Field(default_factory=list)
    judgment_change: Literal["promote", "demote", "strengthen", "weaken"] | None = None


class DecisionValue(MechCALModel):
    """Planner's prospective justification, not a computed utility or support score."""

    record_type: str = "DecisionValue"
    delivered_observable: str = Field(min_length=1)
    observation_scope: str = Field(min_length=1)
    unresolved_extension: str = Field(min_length=1)
    why_not_redundant: str = Field(min_length=1)
    downstream_capability_id: str | None = None
    downstream_question: str | None = None
    downstream_outcomes: list[OutcomeExpectation] = Field(default_factory=list)


class ContinuationOpportunity(MechCALModel):
    record_type: str = "ContinuationOpportunity"
    capability_id: str
    decision_value: DecisionValue
    possible_outcomes: list[OutcomeExpectation] = Field(min_length=1)


class InvestigationPlan(MechCALModel):
    record_type: str = "InvestigationPlan"
    dispatch_id: str
    purpose: Literal["discrimination", "preparation", "exploration", "recovery"]
    candidate_labels: list[str] = Field(min_length=1)
    decision_question: str = Field(min_length=1)
    possible_outcomes: list[OutcomeExpectation] = Field(min_length=1)
    usefulness: str = Field(min_length=1)
    decision_value: DecisionValue | None = None
    hypothesis_test: HypothesisTest | None = None


class MechanismEffect(MechCALModel):
    record_type: str = "MechanismEffect"
    label: str
    effect: Literal["supports", "weakens", "non_discriminating", "unavailable"]
    reason: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)
    judgment_response: str | None = Field(
        default=None, min_length=1,
        description="For a Worker-flagged mechanism, explain how this observation changes or leaves "
        "the current judgment: strengthen, weaken, retain, or reconsider an alternative. "
        "Name any revised factual claim or remaining gap. Actual edits belong in hypotheses, "
        "candidate_bases and support arguments; this response does not apply them automatically.",
    )


class ArgumentChange(MechCALModel):
    record_type: str = "ArgumentChange"
    support_id: str
    disposition: Literal["revised", "withdrawn", "superseded"]
    reason: str = Field(min_length=1)
    replacement_ids: list[str] = Field(
        default_factory=list,
        description="revised: [] or [support_id], because the old ID remains current. withdrawn: []. "
        "superseded: distinct current replacement IDs, with the old ID absent from current arguments. "
        "An additional finding does not replace an argument that is still present.",
    )
    evidence_refs: list[str] = Field(default_factory=list)


class RankingUpdate(MechCALModel):
    record_type: str = "RankingUpdate"
    basis: Literal["initial", "new_evidence", "reinterpretation", "unchanged"]
    explanation: str = Field(min_length=1)
    alternatives_considered: list[str] = Field(
        min_length=1,
        description="Exact mechanism-pool labels only; put the comparison reasoning in explanation.",
        json_schema_extra={"items": {"type": "string", "enum": list(MECHANISM_POOL)}},
    )
    evidence_refs: list[str] = Field(default_factory=list)
    leading_comparison: LeadingComparison | None = None
    candidate_bases: list[CandidateBasis] = Field(default_factory=list)
    boundary_comparisons: list[BoundaryComparison] = Field(default_factory=list)
