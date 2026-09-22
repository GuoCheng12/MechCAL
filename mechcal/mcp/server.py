from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from mechcal.capabilities import default_capability_registry
from mechcal.schemas import (
    AgentExecutionPlan,
    CaseInput,
    CaseRun,
    DispatchRequest,
    ToolExecutionResult,
)
from mechcal.schemas.failures import FailureReport
from mechcal.tools.local_macro import LocalMacroTool
from mechcal.tools.local_micro import LocalMicroscopicTool
from mechcal.tools.local_structure import LocalStructureTool


def _json_model(model: Any) -> dict[str, Any]:
    return model.model_dump(mode="json") if hasattr(model, "model_dump") else dict(model)


def _worker_payload(
    tool: Any,
    *,
    dispatch_request: dict[str, Any],
    case_run: dict[str, Any],
    execution_plan: dict[str, Any] | None,
    schema_feedback: str | None,
) -> dict[str, Any]:
    result = tool.run(
        DispatchRequest.model_validate(dispatch_request),
        CaseRun.model_validate(case_run),
        execution_plan=(
            AgentExecutionPlan.model_validate(execution_plan)
            if execution_plan is not None
            else None
        ),
        schema_feedback=schema_feedback,
    )
    return {"tool_result": _json_model(ToolExecutionResult.model_validate(result))}


def create_mcp_server() -> FastMCP:
    mcp = FastMCP("MechCAL")
    capability_registry = default_capability_registry()
    structure_tool = LocalStructureTool()
    worker_tools = {
        "macro": LocalMacroTool(),
        "microscopic": LocalMicroscopicTool(enable_amesp=_env_flag("MECHCAL_ENABLE_AMESP")),
    }

    @mcp.tool()
    def prepare_structure(
        case_input: dict[str, Any],
        round_id: str,
        workspace: str,
    ) -> dict[str, Any]:
        """Prepare a molecule structure artifact or return a typed fallback artifact."""
        artifact, failure = structure_tool.prepare(
            CaseInput.model_validate(case_input),
            round_id=round_id,
            workspace=Path(workspace),
        )
        return {
            "artifact": artifact.model_dump(mode="json"),
            "failure": failure.model_dump(mode="json"),
        }

    @mcp.tool()
    def list_capabilities() -> dict[str, Any]:
        """Return the capability catalog exposed by this MCP server."""
        return {"catalog": capability_registry.snapshot().model_dump(mode="json")}

    @mcp.tool()
    def run_macro(
        dispatch_request: dict[str, Any],
        case_run: dict[str, Any],
        execution_plan: dict[str, Any] | None = None,
        schema_feedback: str | None = None,
    ) -> dict[str, Any]:
        """Return macro-scale structural proxy evidence as a ToolExecutionResult."""
        return _worker_payload(
            worker_tools["macro"],
            dispatch_request=dispatch_request,
            case_run=case_run,
            execution_plan=execution_plan,
            schema_feedback=schema_feedback,
        )

    @mcp.tool()
    def run_microscopic(
        dispatch_request: dict[str, Any],
        case_run: dict[str, Any],
        execution_plan: dict[str, Any] | None = None,
        schema_feedback: str | None = None,
    ) -> dict[str, Any]:
        """Return microscopic baseline evidence or a typed runtime status result."""
        return _worker_payload(
            worker_tools["microscopic"],
            dispatch_request=dispatch_request,
            case_run=case_run,
            execution_plan=execution_plan,
            schema_feedback=schema_feedback,
        )

    @mcp.tool()
    def healthcheck() -> dict[str, Any]:
        """Return static MCP server capability metadata."""
        return {
            "server": "MechCAL",
            "tools": [
                "list_capabilities",
                "prepare_structure",
                "run_macro",
                "run_microscopic",
            ],
            "capability_count": len(capability_registry.snapshot().capabilities),
            "failure_contract": FailureReport().model_dump(mode="json"),
        }

    return mcp


def main() -> None:
    create_mcp_server().run(show_banner=False)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    main()
