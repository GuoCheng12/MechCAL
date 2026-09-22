from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mechcal.capabilities import CapabilityRegistry
from mechcal.capabilities.schemas import CapabilityCatalog
from mechcal.schemas import (
    AgentExecutionPlan,
    ArtifactRecord,
    CaseInput,
    CaseRun,
    DispatchRequest,
)
from mechcal.schemas.failures import FailureReport
from mechcal.schemas.tool_results import ToolExecutionResult
from mechcal.tools.protocols import WorkerToolPayload


class McpTransport(Protocol):
    def call_tool(
        self,
        *,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class McpToolSpec:
    server_name: str
    tool_name: str


@dataclass(frozen=True)
class AieMcpToolSpecs:
    capabilities: McpToolSpec = McpToolSpec("aie-capability-mcp", "list_capabilities")
    structure: McpToolSpec = McpToolSpec("aie-structure-mcp", "prepare_structure")
    macro: McpToolSpec = McpToolSpec("aie-macro-mcp", "run_macro")
    microscopic: McpToolSpec = McpToolSpec("aie-micro-mcp", "run_microscopic")

    def discover_capability_registry(self, transport: McpTransport) -> CapabilityRegistry:
        try:
            payload = transport.call_tool(
                server_name=self.capabilities.server_name,
                tool_name=self.capabilities.tool_name,
                arguments={},
            )
            catalog_payload = payload.get("catalog", payload)
            catalog = CapabilityCatalog.model_validate(catalog_payload)
            return CapabilityRegistry(catalog.capabilities)
        except Exception as exc:
            raise RuntimeError(
                "MCP capability discovery failed; refusing to continue with a "
                "local default capability registry."
            ) from exc


class McpStructureTool:
    def __init__(self, transport: McpTransport, spec: McpToolSpec | None = None) -> None:
        self.transport = transport
        self.spec = spec or AieMcpToolSpecs().structure

    def prepare(
        self,
        case_input: CaseInput,
        *,
        round_id: str,
        workspace: Path,
    ) -> tuple[ArtifactRecord, FailureReport]:
        payload = self.transport.call_tool(
            server_name=self.spec.server_name,
            tool_name=self.spec.tool_name,
            arguments={
                "case_input": case_input.model_dump(mode="json"),
                "round_id": round_id,
                "workspace": str(workspace),
            },
        )
        return (
            ArtifactRecord.model_validate(payload["artifact"]),
            FailureReport.model_validate(payload.get("failure", {})),
        )


class McpWorkerTool:
    def __init__(self, transport: McpTransport, spec: McpToolSpec) -> None:
        self.transport = transport
        self.spec = spec

    def run(
        self,
        request: DispatchRequest,
        case_run: CaseRun,
        *,
        execution_plan: AgentExecutionPlan | None = None,
        schema_feedback: str | None = None,
    ) -> WorkerToolPayload:
        payload = self.transport.call_tool(
            server_name=self.spec.server_name,
            tool_name=self.spec.tool_name,
            arguments={
                "dispatch_request": request.model_dump(mode="json"),
                "case_run": case_run.model_dump(mode="json"),
                "execution_plan": (
                    execution_plan.model_dump(mode="json")
                    if execution_plan is not None
                    else None
                ),
                "schema_feedback": schema_feedback,
            },
        )
        return payload.get("tool_result", payload)


def validate_mcp_tool_result(payload: WorkerToolPayload) -> ToolExecutionResult:
    return ToolExecutionResult.model_validate(payload)
