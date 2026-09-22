from __future__ import annotations

from pydantic import Field

from mechcal.schemas.base import MechCALModel, utc_now
from mechcal.schemas.decisions import DecisionGateResult, DispatchRequest, PlannerDecision
from mechcal.schemas.reports import AgentReport


class RoundRecord(MechCALModel):
    record_type: str = "RoundRecord"
    case_id: str
    round_id: str
    round_index: int
    created_at: str = Field(default_factory=utc_now)
    planner_decision: PlannerDecision | None = None
    dispatch_requests: list[DispatchRequest] = Field(default_factory=list)
    agent_reports: list[AgentReport] = Field(default_factory=list)
    gate_result: DecisionGateResult | None = None
    notes: list[str] = Field(default_factory=list)
