from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from mechcal.schemas.base import JsonDict, MechCALModel
from mechcal.schemas.contracts import object_variants
from mechcal.schemas.evidence import EvidenceFamily, HypothesisPortfolio
from mechcal.schemas.failures import FailureReport
from mechcal.schemas.evidence_comparison import ContinuationOpportunity
from mechcal.schemas.observation_delivery import ObservableRequirement
from mechcal.schemas.mechanisms import (
    MechanismPriorityDelta,
    MechanismProgram,
    MechanismSupportArgument,
)

AgentName = Literal["macro", "microscopic"]
PlannerAction = Literal["dispatch", "finalize", "stop"]
GateStatus = Literal[
    "allow",
    "retry_schema",
    "retry_policy",
    "max_rounds_stop",
    "finalize",
    "stop",
]


class DispatchRequest(MechCALModel):
    record_type: str = "DispatchRequest"
    dispatch_id: str
    round_id: str
    agent_name: AgentName
    capability_id: str
    task: str = Field(min_length=1, pattern=r"\S")
    objective: str = Field(min_length=1, pattern=r"\S")
    evidence_goal_family: EvidenceFamily
    route: str = Field(min_length=1, pattern=r"\S")
    input_refs: list[str] = Field(default_factory=list)
    constraints: JsonDict = Field(default_factory=dict)
    tool_args: JsonDict = Field(default_factory=dict)
    question_ids: list[str] = Field(default_factory=list)
    expected_observation: str = ""
    required_observables: list[ObservableRequirement] = Field(default_factory=list, max_length=12)
    retry_reason: str = ""
    reuse_result_ids: list[str] = Field(default_factory=list)
    expected_new_information: str = ""

    @field_validator("capability_id", "task", "objective", "route")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("DispatchRequest text fields must not be empty.")
        return normalized


class PlannerAssessment(MechCALModel):
    """Concise decision summary, not a transcript of private reasoning."""

    record_type: str = "PlannerAssessment"
    current_best: str = Field(min_length=1, pattern=r"\S")
    weakened_or_unresolved: str = Field(min_length=1, pattern=r"\S")
    next_useful_step: str = Field(min_length=1, pattern=r"\S")
    evidence_refs: list[str] = Field(default_factory=list)

    @field_validator("current_best", "weakened_or_unresolved", "next_useful_step")
    @classmethod
    def _non_empty_answer(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Each Planner assessment question needs a non-empty answer.")
        return value.strip()


class StoppingAssessment(MechCALModel):
    record_type: str = "StoppingAssessment"
    judgment_state: Literal["adequate_for_bounded_ranking", "unresolved"]
    finish_basis: Literal["judgment_sufficient", "no_useful_available_step"] | None = None
    reason: str = Field(min_length=1, pattern=r"\S")

    @field_validator("reason")
    @classmethod
    def _non_empty_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Stopping assessment needs a reason.")
        return value.strip()

    @model_validator(mode="after")
    def _consistent_status(self) -> StoppingAssessment:
        if self.finish_basis == "judgment_sufficient" and self.judgment_state != "adequate_for_bounded_ranking":
            raise ValueError("judgment_sufficient requires adequate_for_bounded_ranking.")
        return self

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        base = handler.resolve_ref_schema(handler(core_schema))
        return object_variants(base, [
            {"finish_basis": {"type": "string", "const": "judgment_sufficient"},
             "judgment_state": {"type": "string", "const": "adequate_for_bounded_ranking"}},
            {"finish_basis": {"enum": ["no_useful_available_step", None]}},
        ])


class StopReview(MechCALModel):
    """Advisory feedback, never a scientific decision or a dispatch."""

    record_type: str = "StopReview"
    status: Literal["reviewed", "unavailable", "disabled"] = "reviewed"
    challenge: str = ""
    rationale: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)
    capability_ids: list[str] = Field(default_factory=list, max_length=6)
    continuation_opportunity: ContinuationOpportunity | None = None


class PlannerDecision(MechCALModel):
    record_type: str = "PlannerDecision"
    decision_id: str
    round_id: str
    portfolio: HypothesisPortfolio
    current_hypothesis: str
    runner_up_hypothesis: str | None = None
    confidence: float = 0.0
    diagnosis: str
    # Optional when reading old checkpoints; required for new LLM decisions.
    decision_assessment: PlannerAssessment | None = None
    stopping_assessment: StoppingAssessment | None = None
    reviewer_response: str | None = None
    action: PlannerAction
    dispatch_requests: list[DispatchRequest] = Field(default_factory=list, max_length=6)
    unresolved_gaps: list[str] = Field(default_factory=list)
    mechanism_program: MechanismProgram | None = None
    priority_deltas: list[MechanismPriorityDelta] = Field(default_factory=list)
    mechanism_support_arguments: list[MechanismSupportArgument] = Field(default_factory=list)
    final_answer_draft: str | None = None
    rationale: str = ""
    raw_response: JsonDict = Field(default_factory=dict)
    failure: FailureReport = Field(default_factory=FailureReport)

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, value: float) -> float:
        confidence = float(value)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("PlannerDecision.confidence must be between 0.0 and 1.0.")
        return confidence

    @model_validator(mode="after")
    def _action_contract(self) -> PlannerDecision:
        dispatch_ids = [item.dispatch_id for item in self.dispatch_requests]
        if len(dispatch_ids) != len(set(dispatch_ids)):
            raise ValueError("Each tool request requires a distinct dispatch_id.")
        if any(item.round_id != self.round_id for item in self.dispatch_requests):
            raise ValueError("Tool requests must belong to the current decision round.")
        action_dispatch_requirements = {
            "dispatch": bool(self.dispatch_requests),
            "finalize": not self.dispatch_requests and bool((self.final_answer_draft or "").strip()),
            "stop": not self.dispatch_requests,
        }
        if not action_dispatch_requirements[self.action]:
            raise ValueError("PlannerDecision action does not match dispatch/final fields.")

        portfolio_names = {item.name for item in self.portfolio.hypotheses}
        known_current = (
            self.current_hypothesis == "unknown"
            or self.current_hypothesis in portfolio_names
        )
        if not known_current:
            raise ValueError("current_hypothesis must be present in the hypothesis portfolio.")

        return self


class DecisionGateResult(MechCALModel):
    record_type: str = "DecisionGateResult"
    gate_id: str
    round_id: str
    status: GateStatus
    terminal: bool = False
    terminal_reason: str | None = None
    dispatch_requests: list[DispatchRequest] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
