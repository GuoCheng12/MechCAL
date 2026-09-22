from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from mechcal.schemas.base import JsonDict, MechCALModel
from mechcal.schemas.decisions import AgentName
from mechcal.schemas.evidence import EvidenceFamily

CapabilityCost = Literal["free", "low", "medium", "high"]
ToolBackend = Literal["worker"]


class CapabilityToolBinding(MechCALModel):
    record_type: str = "CapabilityToolBinding"
    backend: ToolBackend
    tool_name: str
    server_name: str | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @field_validator("tool_name")
    @classmethod
    def _non_empty_tool_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("CapabilityToolBinding.tool_name must not be empty.")
        return normalized


class CapabilitySpec(MechCALModel):
    record_type: str = "CapabilitySpec"
    capability_id: str
    owner_agent: AgentName
    evidence_family: EvidenceFamily
    route: str
    description: str
    required_artifact_kinds: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    cost: CapabilityCost = "low"
    failure_modes: list[str] = Field(default_factory=list)
    tool_binding: CapabilityToolBinding
    metadata: JsonDict = Field(default_factory=dict)

    @field_validator(
        "capability_id",
        "route",
        "description",
        mode="after",
    )
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("CapabilitySpec text fields must not be empty.")
        return normalized


class CapabilityCatalog(MechCALModel):
    record_type: str = "CapabilityCatalog"
    capabilities: list[CapabilitySpec] = Field(default_factory=list)
