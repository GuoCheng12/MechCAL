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
from mechcal.schemas.evidence import EvidenceUnit
from mechcal.schemas.tool_arguments import merge_tool_args
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
    registry=None,
    owner: str | None = None,
) -> dict[str, Any]:
    request = DispatchRequest.model_validate(dispatch_request)
    case = CaseRun.model_validate(case_run)
    plan = AgentExecutionPlan.model_validate(execution_plan) if execution_plan is not None else None
    registry = registry or default_capability_registry().with_amesp_backend(_env_flag("MECHCAL_ENABLE_AMESP"))
    owner = owner or request.agent_name
    errors = []
    status, failure_kind, recoverable = "failed", "tool_args_invalid", True
    if request.agent_name != owner:
        errors.append("The request owner does not match the called MCP endpoint.")
    allowed_owners = case.runtime.get("ablation", {}).get("enabled_capability_owners")
    if allowed_owners is not None and owner not in allowed_owners:
        errors.append("This worker is disabled by the case ablation configuration.")
    if plan is not None:
        for field in ("round_id", "dispatch_id", "agent_name", "capability_id"):
            if getattr(plan, field) != getattr(request, field):
                errors.append(f"Execution plan {field} does not match the dispatch.")
        if plan.selected_route != request.route or plan.precondition_status != "satisfied":
            errors.append("Execution plan route or precondition status does not authorize this call.")
    try:
        args = merge_tool_args(request, plan)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        effective = request.model_copy(update={"tool_args": args})
        errors.extend(registry.validate_dispatch(effective, case.artifact_manifest))
        if not errors:
            amesp_enabled = case.runtime.get("tool_environment", {}).get("amesp_enabled", True)
            errors.extend(registry.availability_violations(
                effective, case.artifact_manifest,
                amesp_enabled=amesp_enabled,
            ))
            if errors:
                state = registry.readiness(
                    effective.capability_id, case.artifact_manifest, amesp_enabled=amesp_enabled,
                    source_artifact_id=args.get("source_artifact_id"), tool_args=args,
                )
                if state["status"] == "disabled":
                    status, failure_kind, recoverable = "unsupported", "capability_unsupported", False
                elif state["status"] == "missing_input":
                    status, failure_kind = "precondition_missing", "precondition_missing"
    if errors:
        result_id = f"{request.dispatch_id}:contract_rejected"
        result = ToolExecutionResult(
            tool_result_id=result_id, round_id=request.round_id,
            agent_name=owner, dispatch_id=request.dispatch_id, capability_id=request.capability_id,
            selected_route=request.route, status=status, summary="Request rejected before tool execution.",
            structured_results={"boundary_only": True, "violations": errors},
            evidence_units=[EvidenceUnit(
                evidence_id=f"{result_id}:status", source_report_id=result_id,
                round_id=request.round_id, agent_name=owner,
                capability_id=request.capability_id, claim="Tool request failed validation before execution.",
                context="tool interface validation", basis="checklist", support="unresolved",
                limits=["No calculation was executed. This status record is not scientific evidence."],
                observable=request.route, family=request.evidence_goal_family,
                relation="unknown", status="failed", summary="\n".join(errors),
            )],
            failure=FailureReport(kind=failure_kind, message="\n".join(errors), recoverable=recoverable),
        )
        return {"tool_result": _json_model(result)}
    result = tool.run(
        request, case, execution_plan=plan,
        schema_feedback=schema_feedback,
    )
    return {"tool_result": _json_model(ToolExecutionResult.model_validate(result))}


def create_mcp_server() -> FastMCP:
    mcp = FastMCP("MechCAL")
    capability_registry = default_capability_registry().with_amesp_backend(
        _env_flag("MECHCAL_ENABLE_AMESP")
    )
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
            registry=capability_registry, owner="macro",
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
            registry=capability_registry, owner="microscopic",
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
