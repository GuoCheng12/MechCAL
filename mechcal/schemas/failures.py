from __future__ import annotations

from typing import Literal

from pydantic import Field

from mechcal.schemas.base import JsonDict, MechCALModel

FailureKind = Literal[
    "none",
    "agent_schema_invalid",
    "llm_schema_invalid",
    "llm_policy_invalid",
    "tool_args_invalid",
    "runtime_failed",
    "capability_unsupported",
    "precondition_missing",
    "partial_evidence",
    "external_empty",
]


class FailureReport(MechCALModel):
    record_type: str = "FailureReport"
    kind: FailureKind = "none"
    message: str = ""
    recoverable: bool = False
    details: JsonDict = Field(default_factory=dict)

    @property
    def is_failure(self) -> bool:
        return self.kind != "none"
