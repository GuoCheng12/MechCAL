from __future__ import annotations

from typing import Literal

from pydantic import Field

from mechcal.schemas.base import JsonDict, MechCALModel

MicroscopicBundleStatus = Literal["available", "partial"]

MEMBER_SERIES_KINDS = (
    "torsion_snapshot_bundle", "torsion_brightness_coupling_bundle",
    "conformer_bundle", "conformer_state_probe_bundle",
)
GEOMETRY_SOURCE_KINDS = ("relaxed_coordinate_scan_bundle", *MEMBER_SERIES_KINDS)


class AmespMethodProvenance(MechCALModel):
    record_type: str = "AmespMethodProvenance"
    source: Literal["submitted_input", "unknown"] = "unknown"
    requested_method_label: str = "unknown"
    executed_method_label: str = "unknown"
    input_keywords: list[str] = Field(default_factory=list)
    input_blocks: list[JsonDict] = Field(default_factory=list)
    input_sha256: str | None = None
    fallbacks: list[JsonDict] = Field(default_factory=list)


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
    method_provenance: AmespMethodProvenance = Field(default_factory=AmespMethodProvenance)


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

    def method_sources(self) -> list[JsonDict]:
        return [
            {"step_id": step.step_id, "aip_path": step.aip_path,
             "recorded_method_label": step.method_label,
             **step.method_provenance.model_dump(mode="json")}
            for step in self.step_records
        ]
