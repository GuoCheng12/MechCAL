from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, Field, model_validator

from mechcal.schemas.artifacts import ArtifactRecord
from mechcal.schemas.base import JsonDict, MechCALModel
from mechcal.schemas.decisions import AgentName
from mechcal.schemas.evidence import EvidenceUnit
from mechcal.schemas.failures import FailureReport

ToolExecutionStatus = Literal[
    "success",
    "partial",
    "failed",
    "unsupported",
    "precondition_missing",
]


class ToolExecutionResult(MechCALModel):
    record_type: str = "ToolExecutionResult"
    tool_result_id: str
    round_id: str
    agent_name: AgentName
    dispatch_id: str
    capability_id: str
    selected_route: str
    tool_calls: list[str] = Field(default_factory=list)
    status: ToolExecutionStatus
    raw_results: JsonDict = Field(default_factory=dict)
    structured_results: JsonDict = Field(default_factory=dict)
    evidence_units: list[EvidenceUnit] = Field(
        default_factory=list,
        validation_alias=AliasChoices("evidence_units", "evidence_items"),
    )
    artifact_updates: list[ArtifactRecord] = Field(default_factory=list)
    failure: FailureReport = Field(default_factory=FailureReport)
    summary: str = ""

    @property
    def evidence_items(self) -> list[EvidenceUnit]:
        return self.evidence_units

    @model_validator(mode="after")
    def _status_failure_contract(self) -> ToolExecutionResult:
        non_success_statuses = {
            "partial",
            "failed",
            "unsupported",
            "precondition_missing",
        }
        if self.status in non_success_statuses and not self.failure.is_failure:
            raise ValueError("Non-success ToolExecutionResult must carry FailureReport.")
        if self.status == "success" and self.failure.is_failure:
            raise ValueError("Successful ToolExecutionResult cannot carry a failure.")

        mismatched_sources = [
            item.evidence_id
            for item in self.evidence_units
            if item.source_report_id != self.tool_result_id
        ]
        if mismatched_sources:
            raise ValueError(
                "EvidenceUnit.source_report_id must match ToolExecutionResult.tool_result_id."
            )
        if not self.evidence_units:
            raise ValueError("ToolExecutionResult must carry at least one EvidenceUnit.")
        return self
