from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from mechcal.schemas.base import JsonDict, MechCALModel
from mechcal.schemas.evidence import EvidenceFamily, HypothesisPortfolio
from mechcal.schemas.failures import FailureReport
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
    task: str
    objective: str
    evidence_goal_family: EvidenceFamily
    route: str
    input_refs: list[str] = Field(default_factory=list)
    constraints: JsonDict = Field(default_factory=dict)

    @field_validator("capability_id", "task", "objective", "route")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("DispatchRequest text fields must not be empty.")
        return normalized


class PlannerDecision(MechCALModel):
    record_type: str = "PlannerDecision"
    decision_id: str
    round_id: str
    portfolio: HypothesisPortfolio
    current_hypothesis: str
    runner_up_hypothesis: str | None = None
    confidence: float = 0.0
    diagnosis: str
    action: PlannerAction
    dispatch_requests: list[DispatchRequest] = Field(default_factory=list)
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
        action_dispatch_requirements = {
            "dispatch": bool(self.dispatch_requests),
            "finalize": not self.dispatch_requests and bool(self.final_answer_draft),
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
