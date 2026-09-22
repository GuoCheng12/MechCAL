from __future__ import annotations

from typing import Any

from mechcal.capabilities import CapabilityRegistry
from mechcal.public_ids import public_case_id_from_metadata
from mechcal.runtime.llm import OpenAIJsonClient
from mechcal.runtime.prompts import load_prompt
from mechcal.schemas import AgentExecutionPlan, CaseRun, DispatchRequest


class WorkerExecutionPlanner:
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

    def plan(
        self,
        *,
        agent_name: str,
        request: DispatchRequest,
        case_run: CaseRun,
        schema_feedback: str | None = None,
    ) -> AgentExecutionPlan | dict[str, Any]:
        if self.enable_llm:
            if not self._can_use_llm():
                raise RuntimeError(
                    "WorkerExecutionPlanner requires a configured LLM client when "
                    "LLM planning is enabled."
                )
            return self._llm_plan(
                agent_name=agent_name,
                request=request,
                case_run=case_run,
                schema_feedback=schema_feedback,
            )
        return self.deterministic_plan(
            agent_name=agent_name,
            request=request,
            case_run=case_run,
            notes=["Deterministic bounded fallback used because LLM planning is disabled."],
        )

    def deterministic_plan(
        self,
        *,
        agent_name: str,
        request: DispatchRequest,
        case_run: CaseRun,
        notes: list[str] | None = None,
    ) -> AgentExecutionPlan:
        capability = self.capability_registry.require(request.capability_id)
        artifact_refs = [
            artifact.artifact_id
            for artifact in case_run.artifact_manifest.artifacts
            if artifact.status in {"available", "partial"}
            and artifact.kind in capability.required_artifact_kinds
        ]
        missing_artifacts = [
            kind
            for kind in capability.required_artifact_kinds
            if not any(
                artifact.kind == kind and artifact.status in {"available", "partial"}
                for artifact in case_run.artifact_manifest.artifacts
            )
        ]
        precondition_status = "missing" if missing_artifacts else "satisfied"
        return AgentExecutionPlan(
            plan_id=f"{request.dispatch_id}:plan",
            round_id=request.round_id,
            agent_name=agent_name,  # type: ignore[arg-type]
            dispatch_id=request.dispatch_id,
            capability_id=request.capability_id,
            selected_route=request.route,
            tool_steps=[capability.tool_binding.tool_name],
            artifact_refs=artifact_refs,
            precondition_status=precondition_status,
            failure_mode="precondition_missing" if missing_artifacts else None,
            planning_mode="deterministic_fallback",
            tool_arg_rationale=_default_plan_rationale(missing_artifacts),
            parameter_adjustment_hint=_default_adjustment_hint(missing_artifacts),
            notes=[
                *(notes or []),
                *[f"Missing required artifact kind: {kind}" for kind in missing_artifacts],
            ],
        )

    def _can_use_llm(self) -> bool:
        return (
            self.enable_llm
            and self.llm_client is not None
            and self.llm_client.is_configured()
        )

    def _llm_plan(
        self,
        *,
        agent_name: str,
        request: DispatchRequest,
        case_run: CaseRun,
        schema_feedback: str | None,
    ) -> dict[str, Any]:
        capability = self.capability_registry.require(request.capability_id)
        payload = {
            "agent_name": agent_name,
            "dispatch_request": request.model_dump(mode="json"),
            "capability_card": {
                "capability_id": capability.capability_id,
                "owner_agent": capability.owner_agent,
                "evidence_family": capability.evidence_family,
                "route": capability.route,
                "description": capability.description,
                "required_artifact_kinds": capability.required_artifact_kinds,
                "outputs": capability.outputs,
                "cost": capability.cost,
                "failure_modes": capability.failure_modes,
                "metadata": capability.metadata,
                "tool_arg_hints": capability.metadata.get("tool_arg_hints", {}),
                "tool_name": capability.tool_binding.tool_name,
                "tool_binding": capability.tool_binding.model_dump(mode="json"),
            },
            "artifact_manifest": _compact_artifact_manifest(case_run),
            "case_context": {
                "case_id": _public_case_id(case_run),
                "current_round_id": case_run.current_round_id,
                "covered_evidence_families": case_run.evidence_ledger.covered_families(),
            },
        }
        assert self.llm_client is not None
        result = self.llm_client.complete_json(
            system_prompt=load_prompt("agent_execution_plan.md"),
            payload=payload,
            schema_feedback=schema_feedback,
        )
        return self._normalize_llm_plan(
            result,
            request=request,
            capability=capability,
            case_run=case_run,
        )

    def _normalize_llm_plan(
        self,
        result: dict[str, Any],
        *,
        request: DispatchRequest,
        capability: Any,
        case_run: CaseRun,
    ) -> dict[str, Any]:
        artifact_refs = _artifact_ref_texts(result.get("artifact_refs"))
        artifact_refs.extend(_artifact_ref_texts(result.get("artifacts_to_use")))
        artifact_refs.extend(
            artifact.artifact_id
            for artifact in case_run.artifact_manifest.artifacts
            if artifact.status in {"available", "partial"}
            and artifact.kind in capability.required_artifact_kinds
        )

        tool_steps = list(result.get("tool_steps") or [])
        tool_steps.extend(
            str(item.get("tool_name"))
            for item in result.get("tool_calls") or []
            if isinstance(item, dict) and item.get("tool_name")
        )
        if not tool_steps:
            tool_steps = [capability.tool_binding.tool_name]
        tool_args = _tool_args_from_result(result)
        available_artifact_ids = {
            artifact.artifact_id
            for artifact in case_run.artifact_manifest.artifacts
            if artifact.status in {"available", "partial"}
        }
        source_artifact_id = tool_args.get("source_artifact_id")
        if source_artifact_id is not None:
            source_artifact_id = str(source_artifact_id).strip()
            if source_artifact_id and source_artifact_id in available_artifact_ids:
                tool_args["source_artifact_id"] = source_artifact_id
                artifact_refs.append(source_artifact_id)
            else:
                tool_args.pop("source_artifact_id", None)
        tool_arg_rationale = _short_text(
            result.get("tool_arg_rationale")
            or result.get("args_rationale")
            or result.get("rationale")
            or "",
            240,
        )
        if not tool_arg_rationale:
            tool_arg_rationale = "Use capability-bound route-local arguments."
        parameter_adjustment_hint = _short_text(
            result.get("parameter_adjustment_hint")
            or result.get("adjustment_hint")
            or "",
            240,
        )
        if not parameter_adjustment_hint:
            parameter_adjustment_hint = (
                "Keep capability-declared defaults unless retry feedback requires changes."
            )

        allowed = {
            "schema_version",
            "record_type",
            "plan_id",
            "round_id",
            "agent_name",
            "dispatch_id",
            "capability_id",
            "selected_route",
            "tool_steps",
            "tool_args",
            "tool_arg_rationale",
            "parameter_adjustment_hint",
            "artifact_refs",
            "precondition_status",
            "failure_mode",
            "planning_mode",
            "notes",
        }
        normalized = {key: value for key, value in result.items() if key in allowed}
        normalized.update(
            {
                "plan_id": result.get("plan_id") or f"{request.dispatch_id}:plan",
                "round_id": request.round_id,
                "agent_name": request.agent_name,
                "dispatch_id": request.dispatch_id,
                "capability_id": request.capability_id,
                "selected_route": (
                    result.get("selected_route")
                    or result.get("route")
                    or request.route
                ),
                "tool_steps": tool_steps,
                "tool_args": tool_args,
                "tool_arg_rationale": tool_arg_rationale,
                "parameter_adjustment_hint": parameter_adjustment_hint,
                "artifact_refs": sorted(set(artifact_refs)),
                "precondition_status": _normalize_precondition_status(
                    result.get("precondition_status")
                ),
                "planning_mode": "llm",
            }
        )
        return normalized


def _public_case_id(case_run: CaseRun) -> str:
    return public_case_id_from_metadata(case_run.input.metadata)


def _compact_artifact_manifest(case_run: CaseRun) -> dict[str, Any]:
    return {
        "case_id": _public_case_id(case_run),
        "artifacts": [
            {
                "artifact_id": artifact.artifact_id,
                "kind": artifact.kind,
                "created_by_round": artifact.created_by_round,
                "created_by_agent": artifact.created_by_agent,
                "status": artifact.status,
                "observable_tags": list(artifact.observable_tags[:8]),
                "reusable_for": list(artifact.reusable_for[:8]),
                "capability_name": artifact.metadata.get("capability_name"),
            }
            for artifact in case_run.artifact_manifest.artifacts
        ],
    }


def _artifact_ref_texts(value: Any) -> list[str]:
    refs = value if isinstance(value, list) else []
    normalized: list[str] = []
    for item in refs:
        if isinstance(item, str) and item.strip():
            normalized.append(item.strip())
        if isinstance(item, dict) and item.get("artifact_id"):
            normalized.append(str(item["artifact_id"]))
    return normalized


def _tool_args_from_result(result: dict[str, Any]) -> dict[str, Any]:
    direct = result.get("tool_args")
    if isinstance(direct, dict):
        return direct
    aliases = [result.get("arguments")]
    aliases.extend(
        item.get("arguments")
        for item in result.get("tool_calls") or []
        if isinstance(item, dict)
    )
    aliases.extend(
        item.get("tool_args")
        for item in result.get("steps") or []
        if isinstance(item, dict)
    )
    return next((item for item in aliases if isinstance(item, dict)), {})


def _normalize_precondition_status(value: Any) -> str:
    normalized = str(value or "satisfied").strip().lower()
    satisfied_aliases = {"available", "complete", "met", "ok", "ready", "satisfied"}
    missing_aliases = {"blocked", "missing", "precondition_missing", "unsatisfied"}
    if normalized in missing_aliases:
        return "missing"
    if normalized in satisfied_aliases:
        return "satisfied"
    return "satisfied"


def _default_plan_rationale(missing_artifacts: list[str]) -> str:
    if missing_artifacts:
        return "No route-local args selected because required artifacts are missing."
    return "Capability-bound defaults are sufficient for this route-local execution."


def _default_adjustment_hint(missing_artifacts: list[str]) -> str:
    if missing_artifacts:
        return "Satisfy the missing artifact precondition before retrying this route."
    return "Repeat runs may vary only capability-declared route-local arguments."


def _short_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text[:limit]
