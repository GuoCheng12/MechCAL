from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "2.0.0"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class MechCALModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    record_type: str


JsonDict = dict[str, Any]


class RuntimeRef(BaseModel):
    kind: str
    ref_id: str
    path: str | None = None
    metadata: JsonDict = Field(default_factory=dict)
