from __future__ import annotations

from pydantic import ValidationError

from mechcal.agents.execution_planner import WorkerExecutionPlanner
from mechcal.schemas.tool_arguments import merge_tool_args
from mechcal.agents.report_writer import WorkerReportWriter
from mechcal.capabilities import CapabilityRegistry, default_capability_registry
from mechcal.runtime.agent_runtime import BaseAgentRuntime
from mechcal.runtime.llm import OpenAIJsonClient
from mechcal.runtime.retry import SchemaRetryError, SchemaRetryRunner
from mechcal.schemas import (
    AgentExecutionPlan,
    AgentReport,
    CaseRun,
    DispatchRequest,
    ToolExecutionResult,
)
from mechcal.schemas.failures import FailureReport
from mechcal.tools.protocols import WorkerToolClient


class WorkerAgent:
    agent_name: str

    def __init__(
        self,
        tool_client: WorkerToolClient,
        runtime: BaseAgentRuntime[AgentReport] | None = None,
        capability_registry: CapabilityRegistry | None = None,
        execution_runtime: BaseAgentRuntime[AgentExecutionPlan] | None = None,
        llm_client: OpenAIJsonClient | None = None,
        enable_llm_planning: bool = False,
        enable_llm_reports: bool = False,
    ) -> None:
        self.tool_client = tool_client
        self.capability_registry = capability_registry or default_capability_registry()
        self.execution_planner = WorkerExecutionPlanner(
            capability_registry=self.capability_registry,
            llm_client=llm_client,
            enable_llm=enable_llm_planning,
        )
        self.execution_runtime = execution_runtime or BaseAgentRuntime(
            agent_name=f"{self.agent_name}:execution_plan",
            response_model=AgentExecutionPlan,
        )
        self.report_writer = WorkerReportWriter(
            capability_registry=self.capability_registry,
            llm_client=llm_client,
            enable_llm=enable_llm_reports,
        )
        self.runtime = runtime or BaseAgentRuntime(
            agent_name=self.agent_name,
            response_model=AgentReport,
        )

    def plan(self, request: DispatchRequest, case_run: CaseRun) -> AgentExecutionPlan:
        try:
            return self.execution_runtime.run_schema_retry(
                lambda feedback: self._produce_execution_plan(
                    request,
                    case_run,
                    schema_feedback=feedback,
                )
            )
        except SchemaRetryError as exc:
            if self.execution_planner.enable_llm:
                details = " | ".join(str(error) for error in exc.errors[-2:])
                raise RuntimeError(
                    f"{self.agent_name} LLM execution planning failed schema validation."
                    + (f" {details}" if details else "")
                ) from exc
            return self.execution_planner.deterministic_plan(
                agent_name=self.agent_name,
                request=request,
                case_run=case_run,
                notes=[
                    "Execution planning failed schema validation; deterministic "
                    "bounded planning is available only when LLM planning is disabled.",
                    *list(exc.errors),
                ],
            )

    def _produce_execution_plan(
        self,
        request: DispatchRequest,
        case_run: CaseRun,
        *,
        schema_feedback: str | None,
    ) -> AgentExecutionPlan:
        plan = AgentExecutionPlan.model_validate(
            self.execution_planner.plan(
                agent_name=self.agent_name,
                request=request,
                case_run=case_run,
                schema_feedback=schema_feedback,
            )
        )
        self._validate_execution_plan(plan, request)
        return plan

    def _validate_execution_plan(
        self,
        plan: AgentExecutionPlan,
        request: DispatchRequest,
    ) -> None:
        expected = {
            "agent_name": self.agent_name,
            "dispatch_id": request.dispatch_id,
            "capability_id": request.capability_id,
            "selected_route": request.route,
        }
        mismatches = [
            f"{field} must be {value!r}, got {getattr(plan, field)!r}."
            for field, value in expected.items()
            if getattr(plan, field) != value
        ]
        if mismatches:
            raise ValueError("Invalid AgentExecutionPlan: " + " ".join(mismatches))
        capability = self.capability_registry.require(request.capability_id)
        unknown_steps = [
            step for step in plan.tool_steps if step != capability.tool_binding.tool_name
        ]
        if unknown_steps:
            raise ValueError(
                "AgentExecutionPlan used tool step(s) outside the capability binding: "
                + ", ".join(unknown_steps)
            )
        forbidden_tool_arg_keys = {
            "agent_name",
            "capability_id",
            "final_answer",
            "hypothesis",
            "mechanism",
            "route",
            "selected_route",
        }
        forbidden = sorted(forbidden_tool_arg_keys & set(plan.tool_args))
        if forbidden:
            raise ValueError(
                "AgentExecutionPlan.tool_args contains control fields outside "
                "the worker boundary: " + ", ".join(forbidden)
            )
        merge_tool_args(request, plan)
        source_artifact_id = plan.tool_args.get("source_artifact_id")
        if source_artifact_id is not None and str(source_artifact_id) not in plan.artifact_refs:
            raise ValueError(
                "AgentExecutionPlan.tool_args.source_artifact_id must refer to an "
                "artifact selected in AgentExecutionPlan.artifact_refs."
            )
        # Route-local numeric arguments are optional; when an LLM omits them,
        # the public tool defaults remain authoritative.

    def run(
        self,
        request: DispatchRequest,
        case_run: CaseRun,
        *,
        execution_plan: AgentExecutionPlan | None = None,
    ) -> AgentReport:
        _, report = self.run_with_trace(
            request,
            case_run,
            execution_plan=execution_plan,
        )
        return report

    def run_with_trace(
        self,
        request: DispatchRequest,
        case_run: CaseRun,
        *,
        execution_plan: AgentExecutionPlan | None = None,
    ) -> tuple[ToolExecutionResult, AgentReport]:
        plan = execution_plan or self.plan(request, case_run)
        tool_result = self.execute_tool(request, case_run, execution_plan=plan)
        return tool_result, self.report_result(request, case_run, execution_plan=plan, tool_result=tool_result)

    def execute_tool(
        self, request: DispatchRequest, case_run: CaseRun, *, execution_plan: AgentExecutionPlan,
    ) -> ToolExecutionResult:
        plan = execution_plan
        self._validate_execution_plan(plan, request)
        tool_result = self._run_tool(request, case_run, execution_plan=plan)
        if case_run.runtime.get("question_loop") or case_run.runtime.get("command_protocol"):
            # Legacy Macro IDs are round-scoped. New question links require one
            # identity per execution, even for repeated same-route requests.
            import hashlib

            identity = hashlib.sha256(
                f"{case_run.case_id}|{request.dispatch_id}|{tool_result.tool_result_id}".encode(),
            ).hexdigest()[:20]
            source_id = f"execution:{identity}"
            evidence = [item.model_copy(update={
                "evidence_id": f"{source_id}:E{index:03d}", "source_report_id": source_id,
            }) for index, item in enumerate(tool_result.evidence_units, 1)]
            tool_result = ToolExecutionResult.model_validate({
                **tool_result.model_dump(mode="json"), "tool_result_id": source_id,
                "evidence_units": [e.model_dump(mode="json") for e in evidence],
                "raw_results": {**tool_result.raw_results,
                                "original_tool_result_id": tool_result.tool_result_id,
                                "original_evidence_ids": [e.evidence_id for e in tool_result.evidence_units]},
            })
        return tool_result

    def report_result(
        self, request: DispatchRequest, case_run: CaseRun, *, execution_plan: AgentExecutionPlan,
        tool_result: ToolExecutionResult,
        event_sink=None,
    ) -> AgentReport:
        plan = execution_plan
        self._validate_execution_plan(plan, request)
        self._validate_tool_result(tool_result, request)
        try:
            runner = SchemaRetryRunner(
                AgentReport, policy=self.runtime.retry_runner.policy, event_sink=event_sink,
                validators=self.runtime.retry_runner.validators,
                strict_errors=bool(case_run.runtime.get("question_loop")),
                repair_context=bool(case_run.runtime.get("question_loop")),
            )
            report = runner.run(
                lambda feedback: self._produce_agent_report(
                    request,
                    case_run,
                    execution_plan=plan,
                    tool_result=tool_result,
                    schema_feedback=feedback,
                    event_sink=event_sink,
                )
            )
            return report
        except SchemaRetryError as exc:
            if self.report_writer.enable_llm:
                raise RuntimeError(
                    f"{self.agent_name} LLM report writing failed schema validation."
                ) from exc
            report = self.report_writer.deterministic_report(
                request=request,
                execution_plan=plan,
                tool_result=tool_result,
                policy_notes=[
                    "Report writing failed schema validation; deterministic "
                    "ToolExecutionResult projection is available only when LLM "
                    "reports are disabled.",
                    *list(exc.errors),
                ],
            )
            return report
        except Exception as exc:
            if self.report_writer.enable_llm:
                raise RuntimeError(
                    f"{self.agent_name} LLM report writing failed at runtime: {exc}"
                ) from exc
            report = self.report_writer.deterministic_report(
                request=request,
                execution_plan=plan,
                tool_result=tool_result,
                policy_notes=[
                    "Report writing failed at runtime; deterministic "
                    "ToolExecutionResult projection is available only when LLM "
                    "reports are disabled.",
                    f"{type(exc).__name__}: {exc}",
                ],
            )
            return report

    def _run_tool(
        self,
        request: DispatchRequest,
        case_run: CaseRun,
        *,
        execution_plan: AgentExecutionPlan,
    ) -> ToolExecutionResult:
        try:
            payload = self.tool_client.run(
                request,
                case_run,
                execution_plan=execution_plan,
                schema_feedback=None,
            )
            tool_result = self.report_writer.tool_result_from_payload(
                payload,
                request=request,
            )
            self._validate_tool_result(tool_result, request)
            return tool_result
        except PermissionError:
            raise
        except (ValidationError, ValueError) as exc:
            failure = FailureReport(
                kind="agent_schema_invalid",
                message=(
                    f"{self.agent_name} tool returned payload that could not be "
                    "converted into ToolExecutionResult."
                ),
                recoverable=True,
                details={"attempt_errors": [str(exc)]},
            )
            return self.report_writer.failed_tool_result(
                request,
                failure,
                error_type=type(exc).__name__,
            )
        except Exception as exc:
            status = getattr(exc, "status_code", getattr(getattr(exc, "response", None), "status_code", None))
            if status in {401, 403}:
                raise
            failure = FailureReport(
                kind="runtime_failed",
                message=str(exc),
                recoverable=True,
                details={"error_type": type(exc).__name__},
            )
            return self.report_writer.failed_tool_result(
                request,
                failure,
                error_type=type(exc).__name__,
            )

    def _produce_agent_report(
        self,
        request: DispatchRequest,
        case_run: CaseRun,
        *,
        execution_plan: AgentExecutionPlan,
        tool_result: ToolExecutionResult,
        schema_feedback: str | None,
        event_sink=None,
    ) -> AgentReport:
        report = AgentReport.model_validate(
            self.report_writer.write_report(
                agent_name=self.agent_name,
                request=request,
                case_run=case_run,
                execution_plan=execution_plan,
                tool_result=tool_result,
                schema_feedback=schema_feedback,
                event_sink=event_sink,
            )
        )
        self._validate_agent_report(report, request)
        return report

    def _validate_tool_result(
        self,
        tool_result: ToolExecutionResult,
        request: DispatchRequest,
    ) -> None:
        expected = {
            "round_id": request.round_id,
            "agent_name": self.agent_name,
            "dispatch_id": request.dispatch_id,
            "capability_id": request.capability_id,
            "selected_route": request.route,
        }
        mismatches = [
            f"{field} must be {value!r}, got {getattr(tool_result, field)!r}."
            for field, value in expected.items()
            if getattr(tool_result, field) != value
        ]
        if mismatches:
            raise ValueError("Invalid ToolExecutionResult: " + " ".join(mismatches))

    def _validate_agent_report(
        self,
        report: AgentReport,
        request: DispatchRequest,
    ) -> None:
        expected = {
            "round_id": request.round_id,
            "agent_name": self.agent_name,
            "task_received": request.task,
        }
        mismatches = [
            f"{field} must be {value!r}, got {getattr(report, field)!r}."
            for field, value in expected.items()
            if getattr(report, field) != value
        ]
        if mismatches:
            raise ValueError("Invalid AgentReport: " + " ".join(mismatches))
