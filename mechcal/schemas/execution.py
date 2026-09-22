from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from mechcal.schemas.base import JsonDict, MechCALModel
from mechcal.schemas.decisions import AgentName

PreconditionStatus = Literal["satisfied", "missing"]
PlanningMode = Literal["llm", "deterministic_fallback"]


class AgentExecutionPlan(MechCALModel):
    record_type: str = "AgentExecutionPlan"
    plan_id: str
    round_id: str
    agent_name: AgentName
    dispatch_id: str
    capability_id: str
    selected_route: str
    tool_steps: list[str] = Field(default_factory=list)
    tool_args: JsonDict = Field(default_factory=dict)
    tool_arg_rationale: str = Field(..., min_length=1, max_length=240)
    parameter_adjustment_hint: str = Field(..., min_length=1, max_length=240)
    artifact_refs: list[str] = Field(default_factory=list)
    precondition_status: PreconditionStatus = "satisfied"
    failure_mode: str | None = None
    planning_mode: PlanningMode = "deterministic_fallback"
    notes: list[str] = Field(default_factory=list)

    @field_validator(
        "plan_id",
        "round_id",
        "dispatch_id",
        "capability_id",
        "selected_route",
    )
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("AgentExecutionPlan text fields must not be empty.")
        return normalized

    @field_validator("tool_arg_rationale", "parameter_adjustment_hint")
    @classmethod
    def _non_empty_operational_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("AgentExecutionPlan operational text must not be empty.")
        return normalized
