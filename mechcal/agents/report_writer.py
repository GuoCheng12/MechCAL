from __future__ import annotations

from typing import Any

from mechcal.capabilities import CapabilityRegistry
from mechcal.public_ids import public_case_id_from_metadata
from mechcal.runtime.llm import OpenAIJsonClient
from mechcal.runtime.prompts import load_prompt
from mechcal.schemas import (
    AgentExecutionPlan,
    AgentReport,
    CaseRun,
    DispatchRequest,
    EvidenceUnit,
    OperationalNote,
    ToolExecutionResult,
)
from mechcal.schemas.failures import FailureReport
from mechcal.tools.protocols import WorkerToolPayload


class WorkerReportWriter:
    def __init__(
        self,
        *,
        capability_registry: CapabilityRegistry,
        llm_client: OpenAIJsonClient | None = None,
        enable_llm: bool = False,
    ) -> None:
        self.capability_registry = capability_registry
        self.llm_client = llm_client
        self.enable_llm = enable_llm

    def tool_result_from_payload(
        self,
        payload: WorkerToolPayload,
        *,
        request: DispatchRequest,
    ) -> ToolExecutionResult:
        if isinstance(payload, dict) and "tool_result" in payload:
            return ToolExecutionResult.model_validate(payload["tool_result"])
        if isinstance(payload, dict) and (
            payload.get("record_type") == "ToolExecutionResult" or "tool_result_id" in payload
        ):
            return ToolExecutionResult.model_validate(payload)
        if isinstance(payload, ToolExecutionResult):
            return payload
        raise ValueError("Worker tool payload must be a ToolExecutionResult.")

    def failed_tool_result(
        self,
        request: DispatchRequest,
        failure: FailureReport,
        *,
        error_type: str,
    ) -> ToolExecutionResult:
        status_by_failure = {
            "capability_unsupported": "unsupported",
            "precondition_missing": "precondition_missing",
            "partial_evidence": "partial",
            "external_empty": "partial",
        }
        evidence_status_by_failure = {
            "capability_unsupported": "unsupported",
            "precondition_missing": "missing",
            "partial_evidence": "partial",
            "external_empty": "partial",
        }
        tool_result_id = f"{request.round_id}:{request.agent_name}:runtime_error"
        evidence = EvidenceUnit(
            evidence_id=f"{tool_result_id}:{request.route}",
            round_id=request.round_id,
            source_report_id=tool_result_id,
            agent_name=request.agent_name,
            capability_id=request.capability_id,
            claim=request.objective,
            context="worker tool failure",
            basis="checklist",
            support="unresolved",
            limits=[
                "The worker failed before returning a valid computed/proxy result."
            ],
            observable=request.route,
            family=request.evidence_goal_family,
            status=evidence_status_by_failure.get(failure.kind, "failed"),  # type: ignore[arg-type]
            observable_tags=[request.route],
            summary=(
                f"{request.agent_name} route {request.route} failed before returning "
                f"a valid tool result: {failure.message}"
            ),
        )
        return ToolExecutionResult(
            tool_result_id=tool_result_id,
            round_id=request.round_id,
            agent_name=request.agent_name,
            dispatch_id=request.dispatch_id,
            capability_id=request.capability_id,
            selected_route=request.route,
            tool_calls=[],
            status=status_by_failure.get(failure.kind, "failed"),  # type: ignore[arg-type]
            raw_results={"error_type": error_type, "failure": failure.model_dump(mode="json")},
            structured_results={},
            evidence_units=[evidence],
            artifact_updates=[],
            failure=failure,
            summary=f"{request.agent_name} failed before returning a valid tool result.",
        )

    def write_report(
        self,
        *,
        agent_name: str,
        request: DispatchRequest,
        case_run: CaseRun,
        execution_plan: AgentExecutionPlan,
        tool_result: ToolExecutionResult,
        schema_feedback: str | None,
    ) -> AgentReport | dict[str, Any]:
        if self.enable_llm:
            if not self._can_use_llm():
                raise RuntimeError(
                    "WorkerReportWriter requires a configured LLM client when "
                    "LLM reports are enabled."
                )
            return self._llm_report(
                agent_name=agent_name,
                request=request,
                case_run=case_run,
                execution_plan=execution_plan,
                tool_result=tool_result,
                schema_feedback=schema_feedback,
            )
        return self.deterministic_report(
            request=request,
            execution_plan=execution_plan,
            tool_result=tool_result,
        )

    def deterministic_report(
        self,
        *,
        request: DispatchRequest,
        execution_plan: AgentExecutionPlan | None = None,
        tool_result: ToolExecutionResult,
        policy_notes: list[str] | None = None,
    ) -> AgentReport:
        return AgentReport(
            report_id=tool_result.tool_result_id,
            round_id=tool_result.round_id,
            agent_name=tool_result.agent_name,
            task_received=request.task,
            tool_calls=list(tool_result.tool_calls),
            raw_results=dict(tool_result.raw_results),
            structured_results=dict(tool_result.structured_results),
            status=tool_result.status,
            evidence_units=list(tool_result.evidence_units),
            artifact_updates=list(tool_result.artifact_updates),
            planner_readable_report=tool_result.summary
            or f"{tool_result.agent_name} returned a typed tool result.",
            operational_note=self._operational_note(
                request=request,
                execution_plan=execution_plan,
                tool_result=tool_result,
            ),
            failure=tool_result.failure,
            policy_notes=policy_notes or [],
        )

    def _can_use_llm(self) -> bool:
        return (
            self.enable_llm
            and self.llm_client is not None
            and self.llm_client.is_configured()
        )

    def _llm_report(
        self,
        *,
        agent_name: str,
        request: DispatchRequest,
        case_run: CaseRun,
        execution_plan: AgentExecutionPlan,
        tool_result: ToolExecutionResult,
        schema_feedback: str | None,
    ) -> dict[str, Any]:
        capability = self.capability_registry.require(request.capability_id)
        payload = {
            "agent_name": agent_name,
            "dispatch_request": request.model_dump(mode="json"),
            "execution_plan": _compact_execution_plan(execution_plan),
            "capability_card": _compact_capability_card(capability),
            "tool_execution_result": _compact_tool_result_for_llm(tool_result),
            "case_context": {
                "case_id": _public_case_id(case_run),
                "current_round_id": case_run.current_round_id,
                "covered_evidence_families": case_run.evidence_ledger.covered_families(),
            },
            "payload_policy": {
                "tool_execution_result_is_compact": True,
                "runtime_preserves_full_tool_result": True,
                "planner_readable_report_only": True,
            },
        }
        assert self.llm_client is not None
        result = self.llm_client.complete_json(
            system_prompt=load_prompt("agent_report.md"),
            payload=payload,
            schema_feedback=schema_feedback,
        )
        return self._normalize_llm_report(
            result,
            request=request,
            execution_plan=execution_plan,
            tool_result=tool_result,
        )

    def _normalize_llm_report(
        self,
        result: dict[str, Any],
        *,
        request: DispatchRequest,
        execution_plan: AgentExecutionPlan,
        tool_result: ToolExecutionResult,
    ) -> dict[str, Any]:
        policy_notes = result.get("policy_notes") or []
        if not isinstance(policy_notes, list):
            policy_notes = [str(policy_notes)]

        planner_readable_report = (
            result.get("planner_readable_report")
            or result.get("summary")
            or tool_result.summary
            or f"{tool_result.agent_name} returned a typed tool result."
        )
        return {
            "report_id": tool_result.tool_result_id,
            "round_id": tool_result.round_id,
            "agent_name": tool_result.agent_name,
            "task_received": request.task,
            "tool_calls": list(tool_result.tool_calls),
            "raw_results": dict(tool_result.raw_results),
            "structured_results": dict(tool_result.structured_results),
            "status": tool_result.status,
            "evidence_units": [
                item.model_dump(mode="json") for item in tool_result.evidence_units
            ],
            "artifact_updates": [
                item.model_dump(mode="json") for item in tool_result.artifact_updates
            ],
            "planner_readable_report": str(planner_readable_report),
            "operational_note": self._operational_note(
                request=request,
                execution_plan=execution_plan,
                tool_result=tool_result,
                llm_note=result.get("operational_note"),
            ).model_dump(mode="json"),
            "failure": (
                tool_result.failure.model_dump(mode="json")
                if tool_result.failure is not None
                else None
            ),
            "policy_notes": [str(item) for item in policy_notes],
        }

    def _operational_note(
        self,
        *,
        request: DispatchRequest,
        execution_plan: AgentExecutionPlan | None,
        tool_result: ToolExecutionResult,
        llm_note: Any = None,
    ) -> OperationalNote:
        note_payload = llm_note if isinstance(llm_note, dict) else {}
        tool_args = (
            dict(execution_plan.tool_args)
            if execution_plan is not None
            else _json_dict(tool_result.structured_results.get("tool_args"))
        )
        args_rationale = _short_note_text(
            (execution_plan.tool_arg_rationale if execution_plan is not None else "")
            or note_payload.get("args_rationale")
            or _default_args_rationale(tool_args)
        )
        adjustment_hint = _short_note_text(
            (
                execution_plan.parameter_adjustment_hint
                if execution_plan is not None
                else ""
            )
            or note_payload.get("adjustment_hint")
            or _default_adjustment_hint(tool_args, tool_result.status)
        )
        status_note = _short_note_text(
            note_payload.get("status_note")
            or f"Tool status {tool_result.status}; typed tool result remains authoritative.",
            limit=160,
        )
        note_fields = {
            "note_id": f"{tool_result.tool_result_id}:operational_note",
            "source_report_id": tool_result.tool_result_id,
            "round_id": tool_result.round_id,
            "agent_name": tool_result.agent_name,
            "dispatch_id": request.dispatch_id,
            "capability_id": request.capability_id,
            "selected_route": request.route,
            "tool_args": tool_args,
            "args_rationale": args_rationale,
            "adjustment_hint": adjustment_hint,
            "status_note": status_note,
        }
        try:
            return OperationalNote(**note_fields)
        except ValueError:
            return OperationalNote(
                **{
                    **note_fields,
                    "args_rationale": _default_args_rationale(tool_args),
                    "adjustment_hint": _default_adjustment_hint(
                        tool_args,
                        tool_result.status,
                    ),
                    "status_note": (
                        f"Tool status {tool_result.status}; typed result is authoritative."
                    ),
                }
        )


def _json_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _public_case_id(case_run: CaseRun) -> str:
    return public_case_id_from_metadata(case_run.input.metadata)


def _compact_execution_plan(plan: AgentExecutionPlan) -> dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "round_id": plan.round_id,
        "agent_name": plan.agent_name,
        "dispatch_id": plan.dispatch_id,
        "capability_id": plan.capability_id,
        "selected_route": plan.selected_route,
        "tool_steps": list(plan.tool_steps),
        "tool_args": _compact_json(plan.tool_args, depth=2),
        "tool_arg_rationale": _short_note_text(plan.tool_arg_rationale),
        "parameter_adjustment_hint": _short_note_text(plan.parameter_adjustment_hint),
        "artifact_refs": list(plan.artifact_refs[:8]),
        "precondition_status": plan.precondition_status,
        "failure_mode": plan.failure_mode,
        "planning_mode": plan.planning_mode,
        "notes": [_short_note_text(item, 160) for item in plan.notes[:6]],
    }


def _compact_capability_card(capability: Any) -> dict[str, Any]:
    return {
        "capability_id": capability.capability_id,
        "owner_agent": capability.owner_agent,
        "evidence_family": capability.evidence_family,
        "route": capability.route,
        "description": _short_note_text(capability.description, 220),
        "required_artifact_kinds": list(capability.required_artifact_kinds),
        "outputs": list(capability.outputs),
        "cost": capability.cost,
        "failure_modes": list(capability.failure_modes),
        "tool_name": capability.tool_binding.tool_name,
        "tool_arg_hints": _compact_json(
            capability.metadata.get("tool_arg_hints", {}),
            depth=2,
        ),
    }


def _compact_tool_result_for_llm(tool_result: ToolExecutionResult) -> dict[str, Any]:
    return {
        "tool_result_id": tool_result.tool_result_id,
        "round_id": tool_result.round_id,
        "agent_name": tool_result.agent_name,
        "dispatch_id": tool_result.dispatch_id,
        "capability_id": tool_result.capability_id,
        "selected_route": tool_result.selected_route,
        "tool_calls": list(tool_result.tool_calls),
        "status": tool_result.status,
        "summary": _short_note_text(tool_result.summary, 360),
        "failure": (
            _compact_failure(tool_result.failure)
            if tool_result.failure is not None
            else None
        ),
        "structured_results_summary": _compact_json(
            tool_result.structured_results,
            depth=3,
        ),
        "raw_result_keys": sorted(str(key) for key in tool_result.raw_results)[:20],
        "evidence_units": [
            {
                "evidence_id": item.evidence_id,
                "claim": _short_note_text(item.claim, 260),
                "context": _short_note_text(item.context, 180),
                "basis": item.basis,
                "support": item.support,
                "observable": item.observable,
                "family": item.family,
                "status": item.status,
                "summary": _short_note_text(item.summary, 260),
                "limits": [_short_note_text(limit, 160) for limit in item.limits[:4]],
                "observable_tags": list(item.observable_tags[:8]),
                "artifact_refs": list(item.artifact_refs[:8]),
            }
            for item in tool_result.evidence_units[:8]
        ],
        "artifact_updates": [
            {
                "artifact_id": item.artifact_id,
                "kind": item.kind,
                "status": item.status,
                "reusable_for": list(item.reusable_for[:8]),
            }
            for item in tool_result.artifact_updates[:8]
        ],
        "runtime_will_preserve_full_payloads": True,
    }


def _compact_failure(failure: FailureReport) -> dict[str, Any]:
    return {
        "kind": failure.kind,
        "message": _short_note_text(failure.message, 260),
        "recoverable": failure.recoverable,
        "details": _compact_json(failure.details, depth=2),
    }


def _compact_json(value: Any, *, depth: int) -> Any:
    if depth <= 0:
        return _compact_value(value)
    if isinstance(value, dict):
        return {
            str(key): _compact_json(item, depth=depth - 1)
            for key, item in list(value.items())[:12]
            if str(key)
            not in {
                "step_records",
                "file_paths",
                "path",
                "paths",
                "workdir",
                "workspace",
                "output_path",
                "output_dir",
            }
        }
    if isinstance(value, list):
        return [_compact_json(item, depth=depth - 1) for item in value[:8]]
    if isinstance(value, str):
        return _short_note_text(value, 240)
    if isinstance(value, int | float | bool) or value is None:
        return value
    return _short_note_text(str(value), 160)


def _short_note_text(value: Any, limit: int = 240) -> str:
    return str(value or "").strip()[:limit]


def _default_args_rationale(tool_args: dict[str, Any]) -> str:
    if not tool_args:
        return "No route-local tool_args were set; capability defaults were used."
    return "Route-local tool_args were set within the capability binding: " + (
        _compact_args(tool_args)
    )


def _default_adjustment_hint(tool_args: dict[str, Any], status: str) -> str:
    if not tool_args:
        return "Repeated use can expose bounded tool_args when capability hints allow it."
    if status in {"partial", "failed", "precondition_missing"}:
        return "Repeated use can vary only these route-local args after typed status review."
    return "Repeated use can vary only these route-local args for depth or focus."


def _compact_args(tool_args: dict[str, Any]) -> str:
    parts = [f"{key}={_compact_value(value)}" for key, value in sorted(tool_args.items())]
    return ", ".join(parts)[:180]


def _compact_value(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ",".join(str(item) for item in value[:4]) + "]"
    return str(value)
