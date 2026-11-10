"""Read-only, case-scoped history queries, separate from scientific decisions."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from mechcal.schemas.contracts import object_variants


class StoredMetricFields(BaseModel):
    """A partial input view, not a replacement for the stored EvidenceUnit."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    complete: Literal[False] = False
    directory_path: Literal["/metrics"] = "/metrics"
    omitted_paths: list[str] = Field(max_length=12)
    omitted_field_count: int = Field(ge=1)


class MemoryQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["search_case_records", "read_case_records", "find_compatible_artifacts"]
    query: str = Field(default="", max_length=500)
    kind: Literal["observation", "execution", "judgment", "question", "artifact", "context"] | None = None
    record_ids: list[str] = Field(default_factory=list, max_length=6, json_schema_extra={"uniqueItems": True},
                                description="read_case_records needs at least one ID; start>0 needs exactly one.")
    paths: list[str] = Field(default_factory=list, max_length=8,
                             description="Exact JSON pointers for read_case_records. Read directory paths before guessing.")
    view: Literal["auto", "outline", "full", "brief", "findings"] = Field(
        default="auto", description="findings reads complete stored Worker findings and their review from an execution "
        "record, including conditions, limitations and source links; offset paginates findings. "
        "brief returns exact small fields with conditions and unread bulk paths, "
        "without a directory lookup; offset continues the brief view. In delta mode, auto uses brief for oversized "
        "scientific records. Otherwise auto returns requested fields or a small record; large records become a field "
        "directory in focused mode. outline lists paths (offset paginates). Oversized batch/multi-field reads "
        "return complete fields and deferred paths, not fragments. full permits text pagination only for "
        "one record and at most one selected field.",
    )
    capability_id: str | None = Field(default=None, description="Required for find_compatible_artifacts.")
    tool_args: dict = Field(default_factory=dict)
    offset: int = Field(default=0, ge=0)
    start: int = Field(default=0, ge=0, description="Text offset only with exactly one record. Must be 0 for brief/findings; use offset there. findings also requires paths=[].")
    limit: int = Field(default=6, ge=1, le=6)

    @model_validator(mode="after")
    def query_requirements(self):
        if self.action == "read_case_records" and not self.record_ids:
            raise ValueError("read_case_records requires record_ids")
        if self.action == "find_compatible_artifacts" and not self.capability_id:
            raise ValueError("find_compatible_artifacts requires capability_id")
        if len(set(self.record_ids)) != len(self.record_ids):
            raise ValueError("record_ids must be distinct")
        if self.start and len(self.record_ids) != 1:
            raise ValueError("Reading a fragment requires exactly one record ID")
        if self.start and self.view in {"brief", "findings"}:
            raise ValueError(f"{self.view} uses offset, not start")
        if self.view == "findings" and self.paths:
            raise ValueError("findings uses offset; use auto/full with paths for exact source fields")
        return self

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        return object_variants(handler.resolve_ref_schema(handler(core_schema)), [
            {"action": {"type": "string", "const": "read_case_records"}, "record_ids": {"minItems": 1}},
            {"action": {"type": "string", "const": "find_compatible_artifacts"},
             "capability_id": {"type": "string", "minLength": 1}},
            {"action": {"type": "string", "const": "search_case_records"}},
        ])


class ContextReadPlan(BaseModel):
    """Batch known read needs and proceed without a second ready-only model call."""

    model_config = ConfigDict(extra="forbid")
    action: Literal["prepare_context"]
    memory_queries: list[MemoryQuery] = Field(default_factory=list, max_length=4)
    describe_capability_ids: list[str] = Field(default_factory=list, max_length=6)
    focus_capability_ids: list[str] = Field(default_factory=list, max_length=6)


class ContextNeeded(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["need_context"]
    reason: str = Field(min_length=1, max_length=1000)
