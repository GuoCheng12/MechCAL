"""Small task contracts. Scientific judgments remain Planner-owned."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class FixedParameter(TaskModel):
    name: str = Field(min_length=1)
    value: str | int | float | bool
    capability_id: str | None = Field(default=None, min_length=1,
        description="Apply only to this authorized capability. Null applies to every authorized tool exposing "
                    "the parameter. Shared operational inputs such as source_artifact_id must be scoped to a "
                    "capability in multi-tool tasks; downstream tools may use newly produced task artifacts.")


class DelegateComputation(TaskModel):
    kind: Literal["delegate_computation"] = "delegate_computation"
    owner: Literal["macro", "microscopic"]
    objective: str = Field(min_length=1)
    input_refs: list[str] = Field(min_length=1, max_length=16)
    allowed_capability_ids: list[str] = Field(min_length=1, max_length=12)
    fixed_parameters: list[FixedParameter] = Field(default_factory=list)
    question_id: str | None = None


class CompleteWorkerTask(TaskModel):
    kind: Literal["complete_worker_task"] = "complete_worker_task"
    status: Literal["complete", "partial", "blocked"]
    refs: list[str] = Field(max_length=16,
        description="Exact visible record IDs: results, evidence units or artifacts. The runtime links their provenance.")
    summary: str = Field(min_length=1, description="Concise scientific interpretation, comparison and limits. "
        "The runtime attaches execution facts; do not reconstruct the tool/parameter/source history.")


class ReadWorkerLog(TaskModel):
    kind: Literal["read_worker_log"] = "read_worker_log"
    record_id: str
    file_key: str
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=4096, ge=1, le=8192)


class WorkerTaskState(TaskModel):
    version: Literal[1] = 1
    case_id: str
    task_id: str
    contract: DelegateComputation
    started_at: float
    deadline: float
    model_calls: int = 0
    tool_attempts: int = 0
    loaded_tools: dict[str, str] = Field(default_factory=dict)
    result_ids: list[str] = Field(default_factory=list)
    read_handles: list[dict] = Field(default_factory=list)
    directory_reads: list[dict] = Field(default_factory=list)
    progress_note: dict | None = None
    exchange: dict | None = None
    outcome: dict | None = None
    pending_command: dict | None = None
