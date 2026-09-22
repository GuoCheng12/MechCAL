from __future__ import annotations

from typing import Literal

from pydantic import Field

from mechcal.schemas.base import JsonDict, MechCALModel

ArtifactStatus = Literal["available", "partial", "missing", "failed"]


class ArtifactRecord(MechCALModel):
    record_type: str = "ArtifactRecord"
    artifact_id: str
    kind: str
    created_by_round: str
    created_by_agent: str
    status: ArtifactStatus = "available"
    file_paths: dict[str, str] = Field(default_factory=dict)
    observable_tags: list[str] = Field(default_factory=list)
    reusable_for: list[str] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)


class ArtifactManifest(MechCALModel):
    record_type: str = "ArtifactManifest"
    case_id: str
    artifacts: list[ArtifactRecord] = Field(default_factory=list)

    def with_records(self, records: list[ArtifactRecord]) -> ArtifactManifest:
        by_id = {record.artifact_id: record for record in self.artifacts}
        by_id.update({record.artifact_id: record for record in records})
        return self.model_copy(update={"artifacts": list(by_id.values())})

    def artifact_ids(self) -> list[str]:
        return [record.artifact_id for record in self.artifacts]

    def available_for(self, purpose: str) -> list[ArtifactRecord]:
        return [
            record
            for record in self.artifacts
            if record.status in {"available", "partial"} and purpose in record.reusable_for
        ]
