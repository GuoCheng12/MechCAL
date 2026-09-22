from __future__ import annotations

from typing import Literal

from pydantic import Field

from mechcal.schemas.base import JsonDict, MechCALModel

MicroscopicBundleStatus = Literal["available", "partial"]


class MicroscopicStepRecord(MechCALModel):
    record_type: str = "MicroscopicStepRecord"
    step_id: str
    method_label: str
    aip_path: str
    aop_path: str
    stdout_path: str
    stderr_path: str
    exit_code: int
    terminated_normally: bool
    elapsed_seconds: float


class MicroscopicArtifactBundle(MechCALModel):
    record_type: str = "MicroscopicArtifactBundle"
    bundle_id: str
    capability_name: str
    status: MicroscopicBundleStatus
    source_artifact_ids: list[str] = Field(default_factory=list)
    geometry_source: str
    file_paths: dict[str, str] = Field(default_factory=dict)
    step_records: list[MicroscopicStepRecord] = Field(default_factory=list)
    excited_states: list[JsonDict] = Field(default_factory=list)
    parsed_observables: JsonDict = Field(default_factory=dict)
    missing_deliverables: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
