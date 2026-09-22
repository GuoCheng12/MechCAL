from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator

from mechcal.schemas.artifacts import ArtifactRecord
from mechcal.schemas.base import JsonDict, MechCALModel
from mechcal.schemas.evidence import EvidenceUnit
from mechcal.schemas.failures import FailureReport

AgentReportStatus = Literal[
    "success",
    "partial",
    "failed",
    "unsupported",
    "precondition_missing",
]

NON_PLANNER_AGENT_NAMES = {"macro", "microscopic"}
WORKER_FORBIDDEN_PHRASES = (
    "therefore the mechanism",
    "we conclude",
    "the best hypothesis",
    "next action should",
    "planner should",
    "should call",
    "should treat",
    "final mechanism",
    "最终机制",
    "因此机制",
    "下一步应该",
    "建议下一步",
)
OPERATIONAL_NOTE_FORBIDDEN_PHRASES = (
    *WORKER_FORBIDDEN_PHRASES,
    "confidence",
    "supports mechanism",
    "refutes mechanism",
    "mechanism is",
    "机制是",
    "置信度",
)


class OperationalNote(MechCALModel):
    record_type: str = "OperationalNote"
    note_id: str
    source_report_id: str
    round_id: str
    agent_name: str
    dispatch_id: str
    capability_id: str
    selected_route: str
    tool_args: JsonDict = Field(default_factory=dict)
    args_rationale: str = Field(default="", max_length=240)
    adjustment_hint: str = Field(default="", max_length=240)
    status_note: str = Field(default="", max_length=160)

    @field_validator(
        "note_id",
        "source_report_id",
        "round_id",
        "agent_name",
        "dispatch_id",
        "capability_id",
        "selected_route",
    )
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("OperationalNote identity fields must not be empty.")
        return normalized

    @model_validator(mode="after")
    def _operational_only(self) -> OperationalNote:
        text = " ".join(
            [
                self.args_rationale,
                self.adjustment_hint,
                self.status_note,
            ]
        ).lower()
        violations = [
            phrase for phrase in OPERATIONAL_NOTE_FORBIDDEN_PHRASES if phrase in text
        ]
        if violations:
            joined = ", ".join(violations)
            raise ValueError(
                "OperationalNote must stay operational; forbidden phrase(s): "
                + joined
            )
        return self


class AgentReport(MechCALModel):
    record_type: str = "AgentReport"
    report_id: str
    round_id: str
    agent_name: str
    task_received: str
    tool_calls: list[str] = Field(default_factory=list)
    raw_results: JsonDict = Field(default_factory=dict)
    structured_results: JsonDict = Field(default_factory=dict)
    status: AgentReportStatus
    evidence_units: list[EvidenceUnit] = Field(
        default_factory=list,
        validation_alias=AliasChoices("evidence_units", "evidence_items"),
    )
    artifact_updates: list[ArtifactRecord] = Field(default_factory=list)
    planner_readable_report: str
    operational_note: OperationalNote | None = None
    failure: FailureReport = Field(default_factory=FailureReport)
    policy_notes: list[str] = Field(default_factory=list)

    @property
    def evidence_items(self) -> list[EvidenceUnit]:
        return self.evidence_units

    @model_validator(mode="after")
    def _non_planner_report_is_evidence_only(self) -> AgentReport:
        agent_name = self.agent_name.strip().lower()
        text = self.planner_readable_report.lower()
        if agent_name in NON_PLANNER_AGENT_NAMES:
            violations = [phrase for phrase in WORKER_FORBIDDEN_PHRASES if phrase in text]
            if violations:
                joined = ", ".join(violations)
                raise ValueError(
                    f"Non-Planner report must be evidence-only; forbidden phrase(s): {joined}"
                )

        non_success_statuses = {
            "partial",
            "failed",
            "unsupported",
            "precondition_missing",
        }
        if self.status in non_success_statuses and not self.failure.is_failure:
            raise ValueError("Non-success AgentReport must carry a typed FailureReport.")
        if self.status == "success" and self.failure.is_failure:
            raise ValueError("Successful AgentReport cannot carry a failure.")

        mismatched_sources = [
            item.evidence_id
            for item in self.evidence_units
            if item.source_report_id != self.report_id
        ]
        if mismatched_sources:
            raise ValueError("EvidenceUnit.source_report_id must match AgentReport.report_id.")
        if not self.evidence_units:
            raise ValueError("AgentReport must carry at least one EvidenceUnit.")

        if self.operational_note is not None:
            expected = {
                "source_report_id": self.report_id,
                "round_id": self.round_id,
                "agent_name": self.agent_name,
            }
            note_mismatches = [
                field
                for field, value in expected.items()
                if getattr(self.operational_note, field) != value
            ]
            if note_mismatches:
                raise ValueError(
                    "OperationalNote identity must match AgentReport: "
                    + ", ".join(note_mismatches)
                )

        return self
