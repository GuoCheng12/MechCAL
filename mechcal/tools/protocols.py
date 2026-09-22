from __future__ import annotations

from pathlib import Path
from typing import Protocol

from mechcal.schemas import (
    AgentExecutionPlan,
    ArtifactRecord,
    CaseInput,
    CaseRun,
    DispatchRequest,
)
from mechcal.schemas.failures import FailureReport
from mechcal.schemas.tool_results import ToolExecutionResult


class StructureToolClient(Protocol):
    def prepare(
        self,
        case_input: CaseInput,
        *,
        round_id: str,
        workspace: Path,
    ) -> tuple[ArtifactRecord, FailureReport]:
        ...


WorkerToolPayload = ToolExecutionResult | dict[str, object]


class WorkerToolClient(Protocol):
    def run(
        self,
        request: DispatchRequest,
        case_run: CaseRun,
        *,
        execution_plan: AgentExecutionPlan | None = None,
        schema_feedback: str | None = None,
    ) -> WorkerToolPayload:
        ...
