"""Worker interpretations are attributed commentary, not new observations."""

from typing import Literal

from pydantic import Field, model_validator

from mechcal.schemas.base import MechCALModel
from mechcal.schemas.contracts import NONBLANK, object_variants
from mechcal.mechanism_pool import MECHANISM_POOL


class MechanismImplication(MechCALModel):
    """A source-linked suggestion, not a rank or a new observation."""

    record_type: str = "MechanismImplication"
    label: str = Field(min_length=1, json_schema_extra={"enum": list(MECHANISM_POOL)})
    reason: str = Field(min_length=1, max_length=1000)


class WorkerFinding(MechCALModel):
    record_type: str = "WorkerFinding"
    finding_id: str = Field(min_length=1)
    observation: str = Field(min_length=1, max_length=1600)
    interpretation: str = Field(min_length=1, max_length=1600)
    molecular_scope: str = Field(min_length=1, max_length=1000)
    conditions: str = Field(min_length=1, max_length=1000)
    limitations: str = Field(min_length=1, max_length=1600)
    evidence_refs: list[str] = Field(min_length=1)
    source_paths: list[str] = Field(
        min_length=1,
        description="JSON pointers into the supplied tool_execution_result, under structured_results or evidence_units.",
    )
    mechanism_implications: list[MechanismImplication] = Field(
        default_factory=list,
        description="Mechanisms whose interpretation this finding may affect, including outside the assigned question. "
        "Use the supplied pool and explain the physical connection. Empty if no connection is justified.",
    )


class WorkerExplanation(MechCALModel):
    record_type: str = "WorkerExplanation"
    question_ids: list[str] = Field(min_length=1)
    answer_to_question: str = Field(min_length=1, max_length=2000)
    findings: list[WorkerFinding] = Field(default_factory=list, max_length=8)
    missing_observations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_findings(self) -> "WorkerExplanation":
        ids = [f.finding_id for f in self.findings]
        if len(ids) != len(set(ids)):
            raise ValueError("Worker finding IDs must be distinct within an execution.")
        return self


class FindingAudit(MechCALModel):
    record_type: str = "FindingAudit"
    finding_id: str
    status: Literal["faithful", "correction_required", "unsupported"]
    reason: str = Field(min_length=1)
    correction: str = ""

    @model_validator(mode="after")
    def _correction_required(self) -> "FindingAudit":
        if self.status == "correction_required" and not self.correction.strip():
            raise ValueError("A required correction must be stated explicitly.")
        return self

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        return object_variants(handler.resolve_ref_schema(handler(core_schema)), [
            {"status": {"type": "string", "const": "correction_required"}, "correction": NONBLANK},
            {"status": {"type": "string", "enum": ["faithful", "unsupported"]}},
        ])
