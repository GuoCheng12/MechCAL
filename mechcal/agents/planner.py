from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any

from mechcal.agents.evidence_context import (
    calculation_history,
    current_support_arguments,
    evidence_index,
    relevant_evidence,
)
from mechcal.agents.evidence_references import EvidenceReferences
from mechcal.agents.planner_references import PlannerReferences
from mechcal.capabilities import CapabilityRegistry, default_capability_registry
from mechcal.capabilities.discovery import DiscoveryEventSink, ToolDiscoverySession
from mechcal.capabilities.evidence_guide import (
    compact_ability_evidence_guide,
)
from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.public_ids import public_case_id_from_metadata
from mechcal.runtime.agent_runtime import BaseAgentRuntime
from mechcal.runtime.context import PlannerContextView
from mechcal.runtime.llm import OpenAIJsonClient
from mechcal.runtime.mechanism_program import build_diagnosis_units
from mechcal.runtime.prompts import load_prompt
from mechcal.runtime.retry import SchemaRetryError, SchemaRetryRunner, RejectedResponseError, complete_recorded_json
from mechcal.schemas import (
    CaseRun,
    DiagnosisUnit,
    FinalAnswer,
    HypothesisEntry,
    MechanismPriorityDelta,
    MechanismSupportArgument,
    PlannerDecision,
)
from mechcal.schemas.failures import FailureReport
from mechcal.schemas.decisions import PlannerAssessment, StoppingAssessment

LOW_MARGIN_THRESHOLD = 0.05
MAX_DISPATCHES_PER_ROUND = 6


class PlannerAgent:
    """Schema-first Planner that schedules bounded evidence collection."""

    def __init__(
        self,
        runtime: BaseAgentRuntime[PlannerDecision] | None = None,
        capability_registry: CapabilityRegistry | None = None,
        llm_client: OpenAIJsonClient | None = None,
        enable_llm_decisions: bool = False,
        tool_loading_mode: str = "eager",
    ) -> None:
        if tool_loading_mode not in {"eager", "on_demand"}:
            raise ValueError("tool_loading_mode must be eager or on_demand")
        self.tool_loading_mode = tool_loading_mode
        self.capability_registry = capability_registry or default_capability_registry()
        self.llm_client = llm_client
        self.enable_llm_decisions = enable_llm_decisions
        self.runtime = runtime or BaseAgentRuntime(
            agent_name="planner",
            response_model=PlannerDecision,
        )

    def plan_initial(
        self, case_run: CaseRun, *, round_id: str,
        discovery_event_sink: DiscoveryEventSink | None = None,
        request_event_sink=None,
        memory_store=None,
    ) -> PlannerDecision:
        if case_run.runtime.get("question_loop"):
            return self._question_action(case_run, round_id, discovery_event_sink, request_event_sink, memory_store)
        return self._validated_decision(
            lambda feedback, discovery: self._build_initial_decision(case_run, round_id, feedback, discovery),
            case_run=case_run,
            round_id=round_id,
            stage="planner_initial",
            discovery_event_sink=discovery_event_sink,
        )

    def _question_action(self, case, round_id, discovery_event_sink, request_event_sink, memory_store=None):
        from mechcal.runtime.questions import decision_from_action, public_payload, validate_decision
        from mechcal.schemas.question_loop import QuestionAction

        if not self._can_use_llm():
            raise RuntimeError("Question action requires a configured Planner.")
        discovery = ToolDiscoverySession(self.capability_registry, case, round_id, discovery_event_sink) \
            if self.tool_loading_mode == "on_demand" else None
        payload = public_payload(case, self.capability_registry, stage="planner_question_act")
        payload.update(round_id=round_id, output_contract=QuestionAction.model_json_schema())
        refs = EvidenceReferences.from_ids([*[e.evidence_id for e in case.evidence_ledger.items], "input:smiles"])
        memory = self._memory_session(case, memory_store, payload)
        if memory is not None:
            payload = memory.payload
            if discovery is None:
                payload["capability_cards"] = self.capability_registry.cards()
        memory_refs = PlannerReferences(case)

        def produce(feedback):
            if memory is not None:
                return memory.complete(
                    self.llm_client, memory_refs, payload=payload,
                    system_prompt=load_prompt("planner_question_act.md")
                    + ("\n" + load_prompt("planner_tool_discovery.md") if discovery else ""),
                    schema_feedback=feedback, event_sink=request_event_sink, discovery=discovery,
                )
            if discovery is not None:
                response, _ = self._discovery_response(discovery, payload, refs, feedback,
                                                       "planner_question_act.md", request_event_sink)
                try:
                    return refs.decode(response)
                except ValueError as exc:
                    raise RejectedResponseError(str(exc), response) from exc
            return complete_recorded_json(
                self.llm_client, event_sink=request_event_sink,
                system_prompt=load_prompt("planner_question_act.md"),
                payload={**payload, "capability_cards": self.capability_registry.cards()}, schema_feedback=feedback,
            )

        def validate(action):
            current = case if discovery is None else case.touch(runtime={
                **case.runtime, "tool_discovery": discovery.state.model_dump(mode="json"),
            })
            validate_decision(decision_from_action(action, current, round_id), current, self.capability_registry)

        if memory is not None and case.runtime.get("planner_strict_outputs"):
            from mechcal.agents.command_repair import CommandRepairProducer

            produce = CommandRepairProducer(
                complete=lambda **kwargs: memory.complete(self.llm_client, memory_refs,
                                                          discovery=discovery, **kwargs),
                refs=memory_refs, model=QuestionAction, payload=payload,
                prompt=load_prompt("planner_question_act.md")
                       + ("\n" + load_prompt("planner_tool_discovery.md") if discovery else ""),
                event_sink=request_event_sink,
            )

        action = SchemaRetryRunner(
            QuestionAction, policy=self.runtime.retry_runner.policy, validators=[validate],
            event_sink=request_event_sink, repair_context=True,
            separate_validation_budget=True,
        ).run(produce)
        decision = decision_from_action(action, case, round_id)
        if discovery is not None:
            decision.raw_response.update(
                tool_discovery_state=discovery.state.model_dump(mode="json"),
                tool_discovery_budget=discovery.budget_snapshot(),
            )
            discovery.emit({"kind": "tool_discovery_decision", "action": decision.action})
        return decision

    def assimilate(
        self, case_run: CaseRun, *,
        validation_event_sink: Callable[[dict[str, Any]], None] | None = None,
        request_event_sink=None,
        memory_store=None,
    ):
        if case_run.runtime.get("incremental_judgment"):
            from mechcal.agents.judgment_commands import run_judgment_command
            return run_judgment_command(
                self, case_run, validation_event_sink=validation_event_sink,
                request_event_sink=request_event_sink, memory_store=memory_store,
            )
        from mechcal.runtime.questions import public_payload, state, validate_assimilation
        from mechcal.runtime.retry import SchemaRetryRunner
        from mechcal.runtime.result_contract import constrain_disposition_schema
        from mechcal.schemas.question_loop import QuestionAssimilation

        if not self._can_use_llm():
            raise RuntimeError("Question assimilation requires a configured Planner.")
        payload = public_payload(case_run, self.capability_registry, stage="planner_assimilate")
        pending_ids = [r["result_id"] for r in payload["pending_results"]]
        payload["assimilation_context"] = {
            "purpose": "respond_to_argument_review" if payload["argument_review_pending"]
            else "assimilate_results" if pending_ids else "update_investigation",
            "required_disposition_result_ids": pending_ids,
            "result_review_is_current": bool(pending_ids) and payload["result_review"] is not None,
        }
        payload["output_contract"] = QuestionAssimilation.model_json_schema()
        constrain_disposition_schema(payload["output_contract"], payload["disposition_requirements"])
        from mechcal.runtime.incremental_judgment import add_incremental_context, planner_prompt
        add_incremental_context(case_run, payload)
        memory = self._memory_session(case_run, memory_store, payload)
        if memory is not None:
            payload = memory.payload
        references = PlannerReferences(case_run)
        def produce(feedback):
            if memory is not None:
                return memory.complete(
                    self.llm_client, references, payload=payload, event_sink=request_event_sink,
                    system_prompt=planner_prompt(case_run, "planner_assimilate.md"), schema_feedback=feedback,
                )
            return references.complete(
                self.llm_client, event_sink=request_event_sink,
                system_prompt=planner_prompt(case_run, "planner_assimilate.md"), payload=payload, schema_feedback=feedback,
            )
        if case_run.runtime.get("comparative_evidence", False):
            from mechcal.runtime.field_repair import FieldRepairProducer
            from mechcal.agents.observation_view import repair_payload
            produce = FieldRepairProducer(produce, lambda correction, feedback: references.complete(
                self.llm_client, event_sink=request_event_sink,
                system_prompt=load_prompt("planner_field_repair.md"), payload=repair_payload(payload, correction),
                schema_feedback="Correct only allowed_paths; issues and the rejected candidate are in the payload.",
            ), event_sink=request_event_sink, response_model=QuestionAssimilation)
        return SchemaRetryRunner(
            QuestionAssimilation, policy=self.runtime.retry_runner.policy,
            normalized_feedback=case_run.runtime.get("decision_economy", False),
            validators=[lambda item: validate_assimilation(item, case_run)],
            event_sink=request_event_sink, repair_context=True,
            on_validation_error=(lambda event: validation_event_sink({
                **event, "timestamp": time.time(), "stage": "planner_assimilate",
                "case_id": case_run.case_id,
                "round_id": f"R{state(case_run).get('round_index', 1):03d}",
                "revision": payload["revision"] + 1,
            })) if validation_event_sink is not None else None,
        ).run(produce)

    def revise_final_arguments(self, case_run: CaseRun, *, validation_event_sink=None, request_event_sink=None,
                               memory_store=None):
        if case_run.runtime.get("incremental_judgment"):
            from mechcal.agents.judgment_commands import run_judgment_command
            return run_judgment_command(
                self, case_run, final=True, validation_event_sink=validation_event_sink,
                request_event_sink=request_event_sink, memory_store=memory_store,
            )
        from mechcal.runtime.questions import apply_final_revision, public_payload, state
        from mechcal.runtime.retry import SchemaRetryRunner
        from mechcal.schemas.question_loop import FinalArgumentRevision

        if not self._can_use_llm():
            raise RuntimeError("Final revision requires a configured Planner.")
        payload = public_payload(case_run, self.capability_registry, stage="planner_final_revision")
        if not case_run.runtime.get("evidence_decision_contracts"):
            payload.pop("capability_catalog", None)
        payload.update(
            final_internal_audit=state(case_run)["final_audit"],
            previous_final_decision=state(case_run)["decision"],
            output_contract=FinalArgumentRevision.model_json_schema(),
        )
        from mechcal.runtime.incremental_judgment import add_incremental_context, planner_prompt
        add_incremental_context(case_run, payload)
        memory = self._memory_session(case_run, memory_store, payload)
        if memory is not None:
            payload = memory.payload
        references = PlannerReferences(case_run)
        def validate(item):
            from mechcal.runtime.retry import RequestValidationError
            try:
                apply_final_revision(case_run, item, self.capability_registry)
            except RequestValidationError as exc:
                issues = [{**issue, "repair_paths": [[issue.get("response_field", "update"), *path] for path in issue["repair_paths"]]}
                          if issue.get("repair_paths") else issue for issue in exc.issues]
                raise RequestValidationError(issues) from exc

        def produce(feedback):
            if memory is not None:
                return memory.complete(
                    self.llm_client, references, payload=payload, event_sink=request_event_sink,
                    system_prompt=planner_prompt(case_run, "planner_final_revision.md"), schema_feedback=feedback,
                )
            return references.complete(
                self.llm_client, event_sink=request_event_sink,
                system_prompt=planner_prompt(case_run, "planner_final_revision.md"), payload=payload, schema_feedback=feedback,
            )

        if case_run.runtime.get("comparative_evidence", False):
            from mechcal.runtime.field_repair import FieldRepairProducer
            from mechcal.agents.observation_view import repair_payload
            produce = FieldRepairProducer(produce, lambda correction, feedback: references.complete(
                self.llm_client, event_sink=request_event_sink,
                system_prompt=load_prompt("planner_field_repair.md"), payload=repair_payload(payload, correction),
                schema_feedback="Correct only allowed_paths; issues and the rejected candidate are in the payload.",
            ), event_sink=request_event_sink, response_model=FinalArgumentRevision)
        return SchemaRetryRunner(
            FinalArgumentRevision, policy=self.runtime.retry_runner.policy,
            normalized_feedback=case_run.runtime.get("decision_economy", False),
            validators=[validate],
            on_validation_error=validation_event_sink,
            event_sink=request_event_sink, repair_context=True,
        ).run(produce)

    def _memory_session(self, case, store, payload):
        if not case.runtime.get("planner_case_memory"):
            return None
        if store is None:
            raise ValueError("Planner memory requires an explicit case RunStore")
        from mechcal.runtime.planner_memory import PlannerMemorySession
        return PlannerMemorySession(case, store, self.capability_registry, payload["stage"], payload)

    def _build_initial_decision(
        self,
        case_run: CaseRun,
        round_id: str,
        schema_feedback: str | None,
        discovery: ToolDiscoverySession | None = None,
    ) -> PlannerDecision:
        if not self._can_use_llm():
            raise RuntimeError(
                "PlannerAgent requires enable_llm_decisions=True and a configured "
                "LLM client."
            )
        return self._llm_decision(
            case_run=case_run,
            round_id=round_id,
            stage="planner_initial",
            schema_feedback=schema_feedback,
            discovery=discovery,
        )

    def plan_update(
        self,
        context: PlannerContextView,
        case_run: CaseRun,
        *,
        round_id: str,
        discovery_event_sink: DiscoveryEventSink | None = None,
    ) -> PlannerDecision:
        return self._validated_decision(
            lambda feedback, discovery: self._build_update_decision(
                context,
                case_run,
                round_id,
                feedback,
                discovery,
            ),
            case_run=case_run,
            round_id=round_id,
            stage="planner_update",
            discovery_event_sink=discovery_event_sink,
        )

    def _build_update_decision(
        self,
        context: PlannerContextView,
        case_run: CaseRun,
        round_id: str,
        schema_feedback: str | None,
        discovery: ToolDiscoverySession | None = None,
    ) -> PlannerDecision:
        if not self._can_use_llm():
            raise RuntimeError(
                "PlannerAgent requires enable_llm_decisions=True and a configured "
                "LLM client."
            )
        return self._llm_decision(
            case_run=case_run,
            round_id=round_id,
            stage="planner_update",
            schema_feedback=schema_feedback,
            context=context.model_dump(mode="json"),
            discovery=discovery,
        )

    def calibrate_final_portfolio(
        self,
        case_run: CaseRun,
        *,
        round_id: str,
        discovery_event_sink: DiscoveryEventSink | None = None,
    ) -> PlannerDecision:
        return self._validated_decision(
            lambda feedback, discovery: self._build_final_calibration_decision(
                case_run,
                round_id,
                feedback,
                discovery,
            ),
            case_run=case_run,
            round_id=round_id,
            stage="planner_final_calibration",
            discovery_event_sink=discovery_event_sink,
        )

    def _build_final_calibration_decision(
        self,
        case_run: CaseRun,
        round_id: str,
        schema_feedback: str | None,
        discovery: ToolDiscoverySession | None = None,
    ) -> PlannerDecision:
        if not self._can_use_llm():
            raise RuntimeError(
                "PlannerAgent requires enable_llm_decisions=True and a configured "
                "LLM client."
            )
        return self._llm_decision(
            case_run=case_run,
            round_id=round_id,
            stage="planner_final_calibration",
            schema_feedback=schema_feedback,
            discovery=discovery,
            context={
                "final_calibration": True,
                "round_index": len(case_run.round_ids),
            },
        )

    def build_final_answer(self, decision: PlannerDecision, case_run: CaseRun) -> FinalAnswer:
        evidence_refs = [item.evidence_id for item in case_run.evidence_ledger.items]
        diagnosis_units = case_run.diagnosis_units or build_diagnosis_units(case_run)
        answer = _append_diagnosis_summary(
            decision.final_answer_draft
            or "The MechCAL run completed with a conservative unresolved mechanism assessment.",
            diagnosis_units,
        )
        return FinalAnswer(
            case_id=case_run.case_id,
            current_hypothesis=decision.current_hypothesis,
            confidence=decision.confidence,
            answer=answer,
            evidence_refs=evidence_refs,
            round_refs=list(case_run.round_ids),
            artifact_refs=case_run.artifact_manifest.artifact_ids(),
            limitations=self._final_limitations(case_run),
            diagnosis_units=diagnosis_units,
            mechanism_predictions=case_run.mechanism_predictions,
        )

    def respond_to_stop_review(
        self, case_run: CaseRun, *, round_id: str,
        discovery_event_sink: DiscoveryEventSink | None = None,
    ) -> PlannerDecision:
        return self._validated_decision(
            lambda feedback, discovery: self._llm_decision(
                case_run=case_run, round_id=round_id, stage="planner_stop_review_response",
                schema_feedback=feedback, discovery=discovery,
            ),
            case_run=case_run, round_id=round_id, stage="planner_stop_review_response",
            discovery_event_sink=discovery_event_sink,
        )

    def synthesize_at_limit(
        self, case_run: CaseRun, *, round_id: str,
        discovery_event_sink: DiscoveryEventSink | None = None,
    ) -> PlannerDecision:
        return self._validated_decision(
            lambda feedback, discovery: self._llm_decision(
                case_run=case_run, round_id=round_id, stage="planner_final_synthesis",
                schema_feedback=feedback, discovery=discovery,
            ),
            case_run=case_run, round_id=round_id, stage="planner_final_synthesis",
            discovery_event_sink=discovery_event_sink,
        )

    def _final_limitations(self, case_run: CaseRun) -> list[str]:
        limitations = [
            (
                "Planner synthesis is based only on public runtime evidence collected "
                "during this run."
            ),
        ]
        if case_run.runtime.get("question_loop", {}).get("final_revision_complete"):
            limitations.append(
                "Planner revised the answer after the internal final audit using existing evidence. "
                "Changed arguments have not received a second internal audit."
            )
        if case_run.claim_ledger is not None:
            open_claims = case_run.claim_ledger.open_claim_ids()
            blocked_claims = case_run.claim_ledger.blocked_claim_ids()
            if open_claims:
                limitations.append(
                    "Claim coverage remains open: " + ", ".join(open_claims) + "."
                )
            if blocked_claims:
                limitations.append(
                    "Claim coverage has typed blocked routes: "
                    + ", ".join(blocked_claims)
                    + "."
                )
        return limitations

    def _validated_decision(
        self,
        producer: Callable[[str | None, ToolDiscoverySession | None], Any],
        *,
        case_run: CaseRun,
        round_id: str,
        stage: str,
        discovery_event_sink: DiscoveryEventSink | None = None,
    ) -> PlannerDecision:
        discovery = ToolDiscoverySession(
            self.capability_registry, case_run, round_id, discovery_event_sink,
        ) if self.tool_loading_mode == "on_demand" else None

        def produce(feedback: str | None) -> PlannerDecision:
            result = self._policy_checked_decision(producer(feedback, discovery), case_run=case_run)
            if discovery is not None:
                discovery.validate_requests(result.dispatch_requests)
            return result

        try:
            decision = self.runtime.run_schema_retry(produce)
        except SchemaRetryError as exc:
            failure = self.runtime.schema_failure(exc, kind="llm_schema_invalid")
            decision = self._stop_decision(case_run, round_id, stage, failure)
        if discovery is not None:
            decision = decision.model_copy(update={"raw_response": {
                **decision.raw_response,
                "tool_discovery_state": discovery.state.model_dump(mode="json"),
                "tool_discovery_budget": discovery.budget_snapshot(),
            }})
            discovery.emit({"kind": "tool_discovery_decision", "action": decision.action})
        return decision

    def _policy_checked_decision(
        self,
        payload: Any,
        *,
        case_run: CaseRun,
    ) -> PlannerDecision:
        decision = PlannerDecision.model_validate(payload)
        if case_run.runtime.get("question_loop") and not decision.failure.is_failure:
            from mechcal.runtime.questions import validate_decision

            validate_decision(decision, case_run, self.capability_registry)
        self._validate_planner_policy(decision, case_run)
        return decision

    def _validate_planner_policy(
        self,
        decision: PlannerDecision,
        case_run: CaseRun,
    ) -> None:
        limits = case_run.runtime.get("run_limits", {})
        if limits.get("stopping_protocol") == "planner_stop_v1":
            if decision.stopping_assessment is None:
                raise ValueError("New decisions require stopping_assessment.")
            if decision.action == "dispatch" and not limits.get("scientific_tools_allowed", False):
                raise ValueError("Scientific tools are prohibited during final synthesis.")
            if (decision.action == "finalize" and limits.get("scientific_tools_allowed")
                    and decision.stopping_assessment.finish_basis is None):
                raise ValueError("Voluntary finalization requires a finish_basis.")
            if (case_run.runtime.get("stop_review_feedback") is not None
                    and not (decision.reviewer_response or "").strip()):
                raise ValueError("Respond explicitly to the stop review, including unavailable reviews.")
        if decision.action == "finalize":
            labels = [item.name for item in decision.portfolio.hypotheses]
            if len(labels) < 3 or len(labels) != len(set(labels)) or not set(labels).issubset(MECHANISM_POOL):
                raise ValueError("Finalization requires at least three distinct mechanism_pool labels.")
        if (
            decision.action != "stop"
            and _portfolio_is_placeholder(decision.portfolio.model_dump(mode="json"))
            and _has_hypothesis_context(case_run)
            and not _is_initial_evidence_routing(decision, case_run)
        ):
            raise ValueError(
                "PlannerDecision portfolio.hypotheses must include at least one "
                "concrete mechanism hypothesis once runtime evidence or a mechanism "
                "agenda is available; a placeholder 'unknown' portfolio is only "
                "allowed before hypothesis context exists."
            )
        violations = [
            violation
            for request in decision.dispatch_requests
            for violation in self.capability_registry.validate_dispatch(
                request,
                case_run.artifact_manifest,
            )
        ]
        if violations:
            raise ValueError("PlannerDecision violates capability policy: " + " ".join(violations))
        allowed_evidence_ids = {item.evidence_id for item in case_run.evidence_ledger.items} | {"input:smiles"}
        if decision.decision_assessment is not None:
            invalid_assessment_refs = set(decision.decision_assessment.evidence_refs) - allowed_evidence_ids
            if invalid_assessment_refs:
                raise ValueError(
                    "Planner assessment must cite existing EvidenceLedger IDs only: "
                    + ", ".join(sorted(invalid_assessment_refs))
                )
        invalid_hypothesis_refs = {
            ref
            for hypothesis in decision.portfolio.hypotheses
            for ref in hypothesis.evidence_refs
            if ref not in allowed_evidence_ids
        }
        if invalid_hypothesis_refs:
            raise ValueError(
                "PlannerDecision hypotheses must cite existing EvidenceLedger IDs only: "
                + ", ".join(sorted(invalid_hypothesis_refs))
            )
        invalid_support_refs = [
            ref
            for argument in decision.mechanism_support_arguments
            for ref in argument.observation_refs
            if ref not in allowed_evidence_ids
        ]
        if invalid_support_refs:
            raise ValueError(
                "PlannerDecision mechanism_support_arguments must cite existing "
                "EvidenceLedger IDs only: "
                + ", ".join(sorted(set(invalid_support_refs))[:5])
            )
        invalid_delta_refs = [
            ref
            for delta in decision.priority_deltas
            for ref in delta.evidence_refs_added
            if ref not in allowed_evidence_ids
        ]
        if invalid_delta_refs:
            raise ValueError(
                "PlannerDecision priority_deltas must cite existing EvidenceLedger "
                "IDs only: "
                + ", ".join(sorted(set(invalid_delta_refs))[:5])
            )
        if len(decision.dispatch_requests) > MAX_DISPATCHES_PER_ROUND:
            raise ValueError(f"Select at most {MAX_DISPATCHES_PER_ROUND} tool actions per round.")

    def _stop_decision(
        self,
        case_run: CaseRun,
        round_id: str,
        stage: str,
        failure: FailureReport,
    ) -> PlannerDecision:
        return PlannerDecision(
            decision_id=f"{round_id}:{stage}:schema_failure",
            round_id=round_id,
            portfolio=case_run.portfolio,
            current_hypothesis=case_run.portfolio.current,
            confidence=0.0,
            diagnosis=f"Planner schema validation failed during {stage}.",
            action="stop",
            unresolved_gaps=["Planner did not produce a valid typed decision."],
            rationale="Schema retry budget was exhausted.",
            raw_response={"failure": failure.model_dump(mode="json")},
            failure=failure,
        )

    def _can_use_llm(self) -> bool:
        return (
            self.enable_llm_decisions
            and self.llm_client is not None
            and self.llm_client.is_configured()
        )

    def _llm_decision(
        self,
        *,
        case_run: CaseRun,
        round_id: str,
        stage: str,
        schema_feedback: str | None,
        context: dict[str, object] | None = None,
        discovery: ToolDiscoverySession | None = None,
    ) -> PlannerDecision:
        assert self.llm_client is not None
        if case_run.runtime.get("question_loop"):
            from mechcal.runtime.questions import decision_from_action, public_payload, validate_action
            from mechcal.schemas.question_loop import QuestionAction

            payload = public_payload(case_run, self.capability_registry, stage="planner_question_act")
            payload["round_id"] = round_id
            payload["output_contract"] = QuestionAction.model_json_schema()
            references = EvidenceReferences.from_ids([
                *[e.evidence_id for e in case_run.evidence_ledger.items], "input:smiles",
            ])
            if discovery is not None:
                response, _ = self._discovery_response(
                    discovery, payload, references, schema_feedback, "planner_question_act.md",
                )
                response = references.decode(response)
            else:
                payload["capability_cards"] = self.capability_registry.cards()
                response = self.llm_client.complete_json(
                    system_prompt=load_prompt("planner_question_act.md"), payload=payload,
                    schema_feedback=schema_feedback,
                )
            action = QuestionAction.model_validate(response)
            validate_action(action, case_run, self.capability_registry)
            return decision_from_action(action, case_run, round_id)
        payload = _compact_planner_payload(
            case_run=case_run,
            round_id=round_id,
            stage=stage,
            context=context or {},
            capability_registry=self.capability_registry,
        )
        prompt_name = "planner_decision_compact.md"
        payload_shape = "planner_assessment_loop_v6"
        references = EvidenceReferences.from_ids(
            [item.evidence_id for item in case_run.evidence_ledger.items]
        )
        if discovery is None:
            wire_response = self.llm_client.complete_json(
                system_prompt=load_prompt(prompt_name),
                payload=references.encode(payload),
                schema_feedback=references.encode(schema_feedback),
            )
        else:
            wire_response, payload = self._discovery_response(
                discovery, payload, references, schema_feedback, prompt_name,
            )
        response = references.decode(wire_response)
        # Never invent the assessment when the model omits it. Let schema retry
        # request a corrected decision, while old saved decisions remain readable.
        PlannerAssessment.model_validate(_unwrap_planner_response(response).get("decision_assessment"))
        if case_run.runtime.get("run_limits", {}).get("stopping_protocol") == "planner_stop_v1":
            StoppingAssessment.model_validate(_unwrap_planner_response(response).get("stopping_assessment"))
        response = _normalize_planner_llm_response(
            response,
            round_id=round_id,
            existing_portfolio=case_run.portfolio.model_dump(mode="json"),
            previous_portfolio_hypotheses=[
                item.model_dump(mode="json") for item in _full_portfolio_hypotheses(case_run)
            ],
            allowed_evidence_ids=[item.evidence_id for item in case_run.evidence_ledger.items],
            capability_registry=self.capability_registry,
            stage=stage,
            prompt_payload=payload,
        )
        if not isinstance(response.get("raw_response"), dict):
            response["raw_response"] = {}
        response["raw_response"]["prompt_name"] = prompt_name
        response["raw_response"]["payload_shape"] = payload_shape
        response["raw_response"]["evidence_reference_map"] = references.short_to_full
        response["raw_response"]["wire_model_response"] = wire_response
        return PlannerDecision.model_validate(response)

    def _discovery_response(
        self, discovery: ToolDiscoverySession, payload: dict[str, Any],
        references: EvidenceReferences, schema_feedback: str | None, prompt_name: str,
        request_event_sink=None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        assert self.llm_client is not None
        system_prompt = load_prompt(prompt_name) + "\n" + load_prompt("planner_tool_discovery.md")
        while True:
            payload = discovery.augment_payload(payload)
            discovery.emit({"kind": "tool_discovery_waiting", "planner_stage": payload.get("stage")})
            started = time.monotonic()
            kwargs = {
                "system_prompt": system_prompt, "payload": references.encode(payload),
                "schema_feedback": references.encode(schema_feedback),
            }
            try:
                usage = None
                if request_event_sink is not None:
                    def record(event):
                        nonlocal usage
                        usage = event.get("usage")
                        request_event_sink(event)
                    response = complete_recorded_json(self.llm_client, event_sink=record, **kwargs)
                elif isinstance(self.llm_client, OpenAIJsonClient):
                    completion = self.llm_client.complete_json_result(**kwargs)
                    response, usage = completion.data, completion.usage
                else:
                    response = self.llm_client.complete_json(**kwargs)
            except Exception as exc:
                discovery.emit({
                    "kind": "tool_discovery_model_error", "error_type": type(exc).__name__,
                    "planner_stage": payload.get("stage"),
                    "latency_seconds": time.monotonic() - started,
                })
                raise
            action = _unwrap_planner_response(response).get("action")
            discovery.emit({
                "kind": "tool_discovery_model_response", "action": action, "usage": usage,
                "planner_stage": payload.get("stage"),
                "latency_seconds": time.monotonic() - started,
                "schema_feedback_present": schema_feedback is not None,
            })
            if action != "describe_tools":
                return response, payload
            try:
                discovery.query(_unwrap_planner_response(response))
            except ValueError as exc:
                raise RejectedResponseError(str(exc), response) from exc

    def _raw_response(self, prompt_name: str, schema_feedback: str | None) -> dict[str, object]:
        raw_response: dict[str, object] = {
            "prompt_name": prompt_name,
            "capability_cards": self.capability_registry.cards(),
        }
        if schema_feedback is not None:
            raw_response["schema_feedback"] = schema_feedback
        return raw_response


def _compact_planner_payload(
    *,
    case_run: CaseRun,
    round_id: str,
    stage: str,
    context: dict[str, object],
    capability_registry: CapabilityRegistry,
) -> dict[str, object]:
    portfolio_hypotheses = _full_portfolio_hypotheses(case_run)
    support_arguments = current_support_arguments(case_run, limit=len(MECHANISM_POOL))
    new_round_evidence = [
        _compact_evidence_for_planner(item)
        for item in _new_round_evidence_for_planner(case_run)
    ]
    compact_update = stage == "planner_update"
    ranking_init_update = _should_use_initial_evidence_ranking_payload(
        case_run=case_run,
        stage=stage,
        new_round_evidence=new_round_evidence,
    )
    return {
        "stage": stage,
        "round_id": round_id,
        "public_case": {
            "case_id": _public_case_id(case_run),
            "smiles": case_run.input.smiles,
            "user_query": case_run.input.user_query,
            "status": case_run.status,
            "round_count": len(case_run.round_ids),
        },
        "mechanism_pool": list(MECHANISM_POOL),
        "ablation_constraints": _ablation_constraints(case_run),
        "previous_mechanism_portfolio_state": _compact_previous_portfolio_state(
            case_run,
            portfolio_hypotheses,
        ),
        "new_round_evidence": new_round_evidence,
        "evidence_index": evidence_index(case_run),
        "relevant_evidence": [
            _compact_evidence_for_planner(item)
            for item in relevant_evidence(case_run, support_arguments)
            if item.evidence_id not in {row["evidence_id"] for row in new_round_evidence}
        ],
        "calculation_history": calculation_history(case_run),
        "previous_decision_assessment": case_run.runtime.get("last_planner_assessment"),
        "proposed_stop": case_run.runtime.get("proposed_stop"),
        "stop_review_feedback": case_run.runtime.get("stop_review_feedback"),
        "executed_capability_ids": _executed_capability_ids_for_case_run(case_run),
        "incremental_update_contract": _compact_incremental_update_contract(),
        "current_portfolio": "same_as_previous_mechanism_portfolio_state",
        "mechanism_triage_contract": _compact_mechanism_triage_contract(),
        "differential_ranking_guidance": _compact_differential_ranking_guidance(),
        "mechanism_route_priority_map": None
        if compact_update
        else _compact_mechanism_route_priority_map(),
        "final_calibration_contract": _compact_final_calibration_contract()
        if stage == "planner_final_calibration"
        else None,
        "recent_evidence": "same_as_new_round_evidence",
        "covered_evidence_families": case_run.evidence_ledger.covered_families(),
        "current_support_arguments": [
            _compact_support_argument_for_planner(item)
            for item in support_arguments
        ],
        "mechanism_evidence_coverage_table": _ultra_compact_mechanism_evidence_hits(
            case_run
        )
        if compact_update and not ranking_init_update
        else _compact_mechanism_evidence_hits(
            case_run
        )
        if not ranking_init_update
        else [],
        "priority_hygiene_summary": _priority_hygiene_summary(
            case_run,
            compact=compact_update,
        ),
        "differential_mechanism_portfolio": _compact_differential_portfolio_for_planner(
            case_run
        ),
        "agenda_coverage_memo": _minimal_agenda_coverage_for_planner(
            case_run.mechanism_agenda_coverage
        )
        if ranking_init_update
        else _compact_agenda_coverage_for_planner(
            case_run.mechanism_agenda_coverage
        ),
        "artifact_manifest": [
            {
                "artifact_id": artifact.artifact_id,
                "kind": artifact.kind,
                "status": artifact.status,
                "created_by_agent": artifact.created_by_agent,
                "reusable_for": artifact.reusable_for,
                "capability_name": artifact.metadata.get("capability_name"),
            }
            for artifact in case_run.artifact_manifest.artifacts
        ],
        "last_tool_feedback": case_run.runtime.get("last_tool_feedback", []),
        "structure_preparation": case_run.runtime.get("structure_preparation"),
        "run_limits": case_run.runtime.get("run_limits", {}),
        "artifact_available_capability_ids": sorted(
            _artifact_available_capability_ids(capability_registry, case_run)
        ),
        "claim_context": {"open_claim_ids": [], "blocked_claim_ids": []}
        if compact_update
        else _compact_claim_context(case_run),
        "mechanism_agenda": None
        if compact_update
        else _compact_mechanism_agenda_for_planner(case_run.mechanism_program),
        "photophysics_review": _compact_photophysics_review_for_planner(
            case_run.photophysics_review
        ),
        "planner_context": _compact_context_for_planner(
            context,
            compact=compact_update,
        ),
        "capability_cards": _compact_capability_cards(
            capability_registry,
            case_run=case_run,
            stage=stage,
        ),
        "ability_evidence_guide": []
        if ranking_init_update
        else _minimal_planner_ability_guide(
            capability_registry,
            case_run=case_run,
            stage=stage,
        )
        if compact_update
        else _compact_planner_ability_guide(
            capability_registry,
            case_run=case_run,
            stage=stage,
        ),
        "output_contract": _planner_output_contract(
            stopping=case_run.runtime.get("run_limits", {}).get("stopping_protocol") == "planner_stop_v1",
        ),
        "policy": {
            "private_answer_key_access": "none",
            "external_answer_access": "none",
            "case_specific_literature_search": "forbidden",
            "workers_collect_evidence_only": True,
            "planner_is_only_final_scientific_reasoner": True,
            "planner_is_only_mechanism_ranking_owner": True,
            "planner_owns_tool_selection_and_finalization": True,
            "reviewer_routes_are_advice_only": True,
            "final_mechanism_predictions_from_portfolio": True,
            "assessment_before_tool_selection": True,
        },
    }


def _should_use_initial_evidence_ranking_payload(
    *,
    case_run: CaseRun,
    stage: str,
    new_round_evidence: object,
) -> bool:
    stateless_runtime = case_run.runtime.get("incremental_mechanism_portfolio")
    ablation_runtime = case_run.runtime.get("ablation")
    stateless_first_update = (
        isinstance(stateless_runtime, dict)
        and stateless_runtime.get("enabled") is False
        and stateless_runtime.get("prior_planner_ranking_present") is False
        and isinstance(ablation_runtime, dict)
        and ablation_runtime.get("mode")
        in {"mechcal_wo_macro", "mechcal_wo_microscopic"}
    )
    if not _has_bootstrap_initial_coverage(case_run) and not stateless_first_update:
        return False
    if not _portfolio_has_full_mechanism_pool(case_run, _full_portfolio_hypotheses(case_run)):
        return False
    return (
        stage == "planner_update"
        and isinstance(new_round_evidence, list)
        and bool(new_round_evidence)
        and _portfolio_is_initial_pool_state(
            case_run,
            _full_portfolio_hypotheses(case_run),
        )
    )


def _initial_evidence_ranking_payload(
    *,
    case_run: CaseRun,
    round_id: str,
    context: dict[str, object],
    capability_registry: CapabilityRegistry,
) -> dict[str, object]:
    evidence_items = _new_round_evidence_for_planner(case_run)
    evidence = [_initial_ranking_evidence_item(item) for item in evidence_items]
    return {
        "stage": "planner_initial_evidence_ranking",
        "round_id": round_id,
        "public_case": {
            "case_id": _public_case_id(case_run),
            "smiles": case_run.input.smiles,
            "round_count": len(case_run.round_ids),
        },
        "mechanism_pool": list(MECHANISM_POOL),
        "ablation_constraints": _ablation_constraints(case_run),
        "previous_mechanism_portfolio_state": _compact_previous_portfolio_state(
            case_run,
            _full_portfolio_hypotheses(case_run),
        ),
        "evidence": evidence,
        "agenda": _minimal_agenda_coverage_for_planner(
            case_run.mechanism_agenda_coverage
        ),
        "dispatch_capability_ids_available": _initial_followup_capability_ids(
            case_run,
            capability_registry=capability_registry,
        ),
        "planner_context": {
            "round_index": context.get("round_index"),
            "max_rounds": context.get("max_rounds"),
        },
        "output_contract": {
            "portfolio": "{current, runner_up, hypotheses: []}",
            "priority_deltas": (
                "3-6 items; labels from mechanism_pool; evidence_refs_added must "
                "use evidence ids shown in evidence."
            ),
            "mechanism_support_arguments": (
                "0-4 source-grounded finding/warrant/boundary items."
            ),
            "action": "dispatch or finalize; selected_capability_ids optional.",
        },
        "policy": {
            "private_answer_key_access": "none",
            "external_answer_access": "none",
            "case_specific_literature_search": "forbidden",
            "rank_by_differential_priority_not_support_strength": True,
            "proxy_boundaries_required": True,
        },
    }


def _initial_verdict_payload(
    *,
    case_run: CaseRun,
    round_id: str,
    context: dict[str, object],
    capability_registry: CapabilityRegistry,
) -> dict[str, object]:
    evidence = [
        _initial_verdict_evidence_item(item)
        for item in _new_round_evidence_for_planner(case_run)
    ]
    return {
        "stage": "planner_initial_verdict",
        "round_id": round_id,
        "public_case": {
            "case_label": "current_smiles_case",
            "input_type": "SMILES-only molecule",
        },
        "mechanism_pool": ",".join(MECHANISM_POOL),
        "ablation_constraints": _ablation_constraints(case_run),
        "evidence": evidence,
        "coverage_under_screened": _under_screened_labels(case_run),
        "next_capability_ids_available": _initial_followup_capability_ids(
            case_run,
            capability_registry=capability_registry,
        ),
        "priority_hygiene_summary": _priority_hygiene_summary(case_run),
        "planner_context": {
            "round_index": context.get("round_index"),
            "max_rounds": context.get("max_rounds"),
        },
        "policy": {
            "private_answer_key_access": "none",
            "external_answer_access": "none",
            "case_specific_literature_search": "forbidden",
            "rank_by_differential_priority_not_support_strength": True,
        },
    }


def _initial_verdict_evidence_item(item) -> dict[str, object]:
    profile = _ability_profile_for_capability(item.capability_id)
    payload = {
        "id": item.evidence_id,
        "screens": profile.get("screens", []),
        "tier": profile.get("evidence_tier", "unknown"),
        "status": item.status,
        "metrics": _initial_ranking_metrics(item.metrics),
    }
    return {
        **payload,
        "screens": _positive_screens_for_payload_item(payload),
    }


def _payload_screen_is_positive_for_label(
    label: str,
    item: dict[str, object],
) -> bool:
    """Return whether a route-level screen is positive for this mechanism.

    Capability guides describe which mechanisms a route can screen. The metrics
    decide whether the current molecule actually triggered that screen.
    """

    capability_id = _payload_capability_id(item)
    metrics = _metrics_from_payload_item(item)
    if capability_id in {
        "macro.screen_donor_acceptor_layout",
        "macro.screen_donor_acceptor_architecture",
    }:
        if label == "ICT_TICT_CT":
            return bool(
                metrics.get("donor_acceptor_proxy")
                or metrics.get("donor_acceptor_partition_proxy")
            )
        if label == "PET_ET":
            return bool(metrics.get("pet_receptor_like_proxy"))
        return False
    if capability_id == "macro.screen_polar_binding_site_prior":
        if label == "HOST_GUEST_INTERACTION":
            return bool(
                metrics.get("host_guest_followup_trigger")
                or metrics.get("polar_binding_site_proxy")
            )
        if label == "PET_ET":
            return bool(
                metrics.get("pet_receptor_like_proxy")
                or metrics.get("polar_binding_site_proxy")
                or metrics.get("donor_acceptor_proxy")
            )
        return False
    if capability_id == "macro.screen_metal_triplet_prior":
        return label == "TRIPLET_METAL_ENERGY_TRANSFER" and bool(
            metrics.get("metal_or_lanthanide_prior")
            or metrics.get("heavy_atom_triplet_prior")
            or metrics.get("sulfur_phosphorus_triplet_prior")
        )
    if capability_id in {
        "macro.screen_rotor_torsion_topology",
        "macro.screen_rotor_rim_prior",
    }:
        if label != "RIM_RIR_RIV":
            return False
        flexible = max(
            _number_like(metrics.get("rotatable_bond_count")) or 0.0,
            _number_like(metrics.get("torsion_candidate_count")) or 0.0,
        )
        return flexible > 0.0
    if capability_id in {
        "microscopic.run_torsion_brightness_coupling_scan",
        "microscopic.run_torsion_snapshots",
    }:
        flexible = max(
            _number_like(metrics.get("rotatable_bond_count")) or 0.0,
            _number_like(metrics.get("torsion_candidate_count")) or 0.0,
            _number_like(metrics.get("state_count")) or 0.0,
        )
        if label == "RIM_RIR_RIV":
            return flexible > 0.0
        if label == "RACI_CI_ACCESS":
            return _has_raci_torsion_proxy(metrics)
        return False
    if capability_id == "macro.screen_aggregation_prone_scaffold":
        if label in {"PACKING_HOST_MATRIX_CONFINEMENT", "RIM_RIR_RIV"}:
            return bool(metrics.get("aggregation_prone_proxy"))
        if label == "AGGREGATE_EXCITON_EXCIMER":
            return bool(
                metrics.get("aggregate_exciton_proxy")
                or metrics.get("excimer_geometry_proxy")
                or metrics.get("spectral_new_band_proxy")
            )
        return False
    if capability_id == "macro.screen_esipt_structural_motif":
        return label == "ESIPT_PT" and bool(
            metrics.get("esipt_motif_proxy")
            or metrics.get("proton_transfer_pair_proxy_count")
        )
    if capability_id == "macro.run_solid_state_emission_proxy":
        if label == "RIM_RIR_RIV":
            return (_number_like(metrics.get("rotor_burden_proxy")) or 0.0) > 0.0
        if label in {"PACKING_HOST_MATRIX_CONFINEMENT", "AGGREGATE_EXCITON_EXCIMER"}:
            return bool(metrics.get("aggregation_prone_proxy"))
        return False
    if capability_id in {
        "macro.run_dimer_packing_proxy",
        "macro.run_aggregate_contact_proxy",
    }:
        if label == "PACKING_HOST_MATRIX_CONFINEMENT":
            return bool(
                metrics.get("hydrophobic_contact_proxy")
                or metrics.get("polar_disruption_proxy")
                or (_number_like(metrics.get("dimer_contact_score_proxy")) or 0.0) >= 0.55
            )
        if label == "AGGREGATE_EXCITON_EXCIMER":
            return bool(
                metrics.get("aggregate_exciton_proxy")
                or metrics.get("excimer_geometry_proxy")
                or metrics.get("spectral_new_band_proxy")
            )
        return False
    if capability_id == "microscopic.run_bright_dark_state_ordering":
        state_count = _number_like(metrics.get("state_count")) or 0.0
        bright_state_index = _number_like(metrics.get("bright_state_index")) or 0.0
        if label == "SOKR_ANTI_KASHA":
            return state_count > 1.0 and bright_state_index > 1.0
        if label == "RADIATIVE_RATE_STATE_BALANCE":
            return (
                state_count > 1.0
                or bright_state_index > 1.0
                or _oscillator_strength_from_metrics(metrics) is not None
            )
        return False
    if capability_id == "microscopic.run_baseline_bundle":
        return label == "RADIATIVE_RATE_STATE_BALANCE" and (
            (_number_like(metrics.get("state_count")) or 0.0) > 1.0
            or _oscillator_strength_from_metrics(metrics) is not None
        )
    if capability_id in {
        "microscopic.run_frontier_orbital_partition",
        "microscopic.extract_ct_descriptors_from_bundle",
    }:
        if label == "ICT_TICT_CT":
            return True
        if label == "PET_ET":
            return bool(
                metrics.get("pet_receptor_like_proxy")
                or metrics.get("redox_alignment_proxy")
                or metrics.get("electron_transfer_pathway_proxy")
            )
        return False
    return True


def _has_balanced_donor_acceptor_metrics(metrics: dict[str, object]) -> bool:
    donors = metrics.get("donor_atom_symbols")
    acceptors = metrics.get("acceptor_atom_symbols")
    if isinstance(donors, list) and isinstance(acceptors, list):
        return bool(donors) and bool(acceptors)
    donor_count = _number_like(metrics.get("donor_atom_count")) or 0.0
    acceptor_count = _number_like(metrics.get("acceptor_atom_count")) or 0.0
    if donor_count > 0.0 or acceptor_count > 0.0:
        return donor_count > 0.0 and acceptor_count > 0.0
    return False


def _has_raci_torsion_proxy(metrics: dict[str, object]) -> bool:
    state_count = _number_like(metrics.get("state_count")) or 0.0
    if state_count <= 0.0:
        return False
    signal_keys = (
        "oscillator_strength_range",
        "bright_oscillator_strength_delta",
        "first_excitation_energy_shift_ev",
        "energy_range_hartree",
        "oscillator_strength",
    )
    if any(_nonzero_metric(metrics.get(key)) for key in signal_keys):
        return True
    bright_state = metrics.get("bright_state")
    if isinstance(bright_state, dict) and bright_state:
        return True
    parsed_states = metrics.get("parsed_states")
    return isinstance(parsed_states, list) and bool(parsed_states)


def _is_material_followup_capability(capability_id: str) -> bool:
    return capability_id in {
        "macro.run_solid_state_emission_proxy",
        "macro.run_dimer_packing_proxy",
        "macro.run_aggregate_contact_proxy",
    }


def _nonzero_metric(value: object) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, int | float):
        return float(value) != 0.0
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _executed_capability_ids_from_payload(payload: dict[str, object]) -> set[str]:
    executed: set[str] = set(_text_list(payload.get("executed_capability_ids"), limit=40))
    for key in (
        "new_round_evidence",
        "mechanism_evidence_coverage_table",
        "priority_hygiene_summary",
    ):
        for item in _as_dict_list(payload.get(key)):
            capability_id = _short_text(item.get("capability_id"), 160)
            if capability_id:
                executed.add(capability_id)
            for hit in _as_dict_list(item.get("route_hits")):
                hit_capability_id = _short_text(hit.get("capability_id"), 160)
                if hit_capability_id:
                    executed.add(hit_capability_id)
            for nested_capability_id in _text_list(item.get("capability_ids"), limit=8):
                executed.add(nested_capability_id)
    return executed


def _oscillator_strength_from_metrics(metrics: dict[str, object]) -> float | None:
    return _number_like(metrics.get("oscillator_strength"))


def _metrics_from_payload_item(item: dict[str, object]) -> dict[str, object]:
    metrics = item.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    key_metrics = item.get("key_metrics")
    if isinstance(key_metrics, dict):
        return key_metrics
    return {}


def _number_like(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _payload_capability_id(item: dict[str, object]) -> str:
    return _short_text(item.get("capability_id"), 180)


def _initial_ranking_evidence_item(item) -> dict[str, object]:
    profile = _ability_profile_for_capability(item.capability_id)
    payload = {
        "evidence_id": item.evidence_id,
        "capability_id": item.capability_id,
        "screens": profile.get("screens", []),
        "evidence_tier": profile.get("evidence_tier", "unknown"),
        "status": item.status,
        "basis": item.basis,
        "finding": _short_text(item.claim or item.summary, 180),
        "key_metrics": _initial_ranking_metrics(item.metrics),
        "boundary": _short_text(
            item.limits[0] if item.limits else "Proxy evidence only.",
            120,
        ),
    }
    return {
        **payload,
        "screens": _positive_screens_for_payload_item(payload),
    }


def _initial_ranking_metrics(metrics: object) -> dict[str, object]:
    if not isinstance(metrics, dict):
        return {}
    keep_keys = {
        "oscillator_strength",
        "state_count",
        "hbd_count",
        "hba_count",
        "tautomerizable_subgraph_proxy",
        "proton_transfer_pair_proxy_count",
        "closest_proton_transfer_topological_distance",
        "esipt_motif_proxy",
        "donor_acceptor_proxy",
        "donor_acceptor_partition_proxy",
        "donor_atom_symbols",
        "acceptor_atom_symbols",
        "polar_atom_symbols",
        "hetero_atom_count",
        "conjugation_proxy",
        "polar_binding_site_proxy",
        "carbonyl_like_site_count",
        "host_guest_followup_trigger",
        "pet_receptor_like_proxy",
        "metal_or_lanthanide_prior",
        "heavy_atom_triplet_prior",
        "sulfur_phosphorus_triplet_prior",
        "triplet_metal_followup_trigger",
        "rotatable_bond_count",
        "torsion_candidate_count",
        "rotatable_bond_inventory",
        "flexibility_proxy",
        "aggregation_prone_proxy",
        "aromatic_ring_count",
        "mol_logp",
    }
    compact: dict[str, object] = {}
    if isinstance(metrics.get("computed_observables"), dict):
        compact["computed_observables"] = metrics["computed_observables"]
    for key in keep_keys:
        if key in metrics and isinstance(metrics[key], str | int | float | bool):
            compact[key] = metrics[key]
    pairs = metrics.get("proton_transfer_pair_proxies")
    if isinstance(pairs, list):
        compact["proton_transfer_pairs"] = [
            {
                "pair_type": item.get("pair_type"),
                "topological_distance": item.get("topological_distance"),
            }
            for item in pairs[:2]
            if isinstance(item, dict)
        ]
    return compact


def _evidence_by_candidate_for_initial_ranking(
    evidence: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for item in evidence:
        for label in item.get("screens", []):
            if not isinstance(label, str) or label not in MECHANISM_POOL:
                continue
            row = grouped.setdefault(
                label,
                {
                    "label": label,
                    "evidence_refs": [],
                    "evidence_tiers": [],
                    "key_findings": [],
                },
            )
            row["evidence_refs"].append(item["evidence_id"])  # type: ignore[union-attr]
            row["evidence_tiers"].append(item["evidence_tier"])  # type: ignore[union-attr]
            row["key_findings"].append(item["finding"])  # type: ignore[union-attr]
    return [
        {
            "label": label,
            "evidence_refs": list(dict.fromkeys(row["evidence_refs"]))[:4],
            "evidence_tiers": list(dict.fromkeys(row["evidence_tiers"]))[:3],
            "key_findings": [_short_text(item, 120) for item in row["key_findings"][:2]],
        }
        for label, row in grouped.items()
    ]


def _initial_followup_capability_ids(
    case_run: CaseRun,
    *,
    capability_registry: CapabilityRegistry,
) -> list[str]:
    executed = {
        item.capability_id for item in case_run.evidence_ledger.items if item.capability_id
    }
    preferred = [
        "macro.screen_intramolecular_hbond_preorganization",
        "microscopic.run_bright_dark_state_ordering",
        "microscopic.run_frontier_orbital_partition",
        "microscopic.extract_ct_descriptors_from_bundle",
        "microscopic.run_torsion_brightness_coupling_scan",
        "macro.run_solid_state_emission_proxy",
        "macro.run_dimer_packing_proxy",
        "macro.run_aggregate_contact_proxy",
        "macro.run_crystal_restriction_checklist",
    ]
    available = _artifact_available_capability_ids(capability_registry, case_run)
    return [
        capability_id
        for capability_id in preferred
        if capability_id in available and capability_id not in executed
    ][:8]


def _executed_capability_ids_for_case_run(case_run: CaseRun) -> list[str]:
    return list(
        dict.fromkeys(
            str(item.capability_id)
            for item in case_run.evidence_ledger.items
            if str(item.capability_id or "").strip()
        )
    )


def _under_screened_labels(case_run: CaseRun) -> list[str]:
    coverage = case_run.mechanism_agenda_coverage
    if coverage is None:
        return []
    return [
        item.label
        for item in coverage.agenda_items[:8]
        if item.coverage_status in {"under_screened", "needs_follow_up"}
    ]




def _mechanism_triage_contract() -> dict[str, object]:
    return {
        "required_scope": (
            "Before finalizing, triage every label in mechanism_pool at least "
            "conceptually. The final portfolio may contain only the most important "
            "candidates, but omitted mechanisms should be omitted because they are "
            "not triggered, lower priority, or counterevidenced, not because they "
            "are harder to prove."
        ),
        "triage_status_values": [
            "triggered",
            "needs_screening",
            "unlikely",
            "not_triggered",
        ],
        "triage_fields_to_consider": [
            "structural_trigger",
            "explanatory_relevance",
            "mechanistic_specificity",
            "counterevidence",
            "validation_urgency",
            "available_or_missing_evidence_routes",
        ],
        "portfolio_fields": {
            "confidence": (
                "Use as the differential-priority score for candidate ranking."
            ),
            "differential_priority": (
                "Same 0-1 priority score as confidence; include it explicitly "
                "when possible."
            ),
            "evidence_support": (
                "Current support strength only: strong, partial, weak_or_proxy, "
                "or unsupported."
            ),
            "claim_status": (
                "Use candidate_requires_validation or underdetermined when the "
                "candidate is important but not proven."
            ),
            "validation_needed": (
                "Concrete next evidence needed before making a strong claim."
            ),
        },
    }


def _compact_mechanism_triage_contract() -> dict[str, object]:
    return {
        "goal": "Reconsider all supplied mechanism_pool labels at every decision.",
        "statuses": ["triggered", "needs_screening", "unlikely", "not_triggered"],
        "keep_hard_mechanisms": (
            "Do not omit high-level mechanisms only because current tools provide "
            "proxy evidence; keep them as validation-needed candidates when relevant."
        ),
    }


def _incremental_update_contract() -> dict[str, object]:
    return {
        "state_update_rule": (
            "Treat previous_mechanism_portfolio_state as the authoritative "
            "portfolio state. Update it incrementally with new_round_evidence, "
            "support audits, and reviewer feedback. Do not rebuild the mechanism "
            "ranking from scratch each round."
        ),
        "must_preserve": [
            "Keep every mechanism_pool label in portfolio.hypotheses.",
            "Do not drop lower-ranked mechanisms just because they are outside Top-3.",
            (
                "Carry forward evidence_refs and validation_needed unless new "
                "evidence or audit feedback changes them."
            ),
        ],
        "allowed_updates": [
            "differential_priority",
            "evidence_support",
            "claim_status",
            "status",
            "rationale",
            "evidence_refs",
            "validation_needed",
            "current",
            "runner_up",
        ],
        "priority_delta_contract": {
            "required_field": "priority_deltas",
            "shape": (
                "list of {label, previous_priority, priority_delta, new_priority, "
                "delta_reason, evidence_refs_added, support_change}"
            ),
            "rules": [
                (
                    "Base each delta on previous_mechanism_portfolio_state; if no "
                    "new evidence or audit feedback changed a mechanism, keep "
                    "priority_delta close to 0."
                ),
                (
                    "If abs(priority_delta) > 0.15, delta_reason must cite the "
                    "new observation, counterevidence, or audit feedback that "
                    "justifies the change."
                ),
                (
                    "Do not use priority_deltas to hide a new free-form ranking; "
                    "they are an auditable state transition log."
                ),
            ],
            "support_change_values": [
                "unchanged",
                "upgraded",
                "downgraded",
                "weakened",
                "newly_supported",
            ],
        },
        "low_margin_rule": {
            "threshold": LOW_MARGIN_THRESHOLD,
            "rule": (
                "When adjacent candidates are separated by less than the threshold, "
                "preserve the previous relative order unless new evidence or audit "
                "feedback clearly supports a reversal. Report the competition as "
                "low-margin uncertainty; do not alter evidence_support."
            ),
        },
        "route_selection_rule": (
            "Select the next evidence route by asking which new observation would "
            "most reduce uncertainty among close candidates in the current portfolio."
        ),
    }


def _compact_incremental_update_contract() -> dict[str, object]:
    return {
        "state_update_rule": (
            "Return a complete candidate ranking from current evidence. "
            "The hypotheses list, not priority deltas or runtime rules, determines ranking."
        ),
        "priority_deltas_required": False,
        "low_margin_rule": "Explain close alternatives without forced carry-forward ordering.",
        "route_selection_rule": (
            "First assess current best explanations, weakened or unresolved alternatives, "
            "and the value of a next check. Reviewer urgency is advice, not a requirement."
        ),
    }


def _differential_ranking_guidance() -> dict[str, object]:
    return {
        "ranking_goal": (
            "Rank evidence-aware differential mechanism candidates. Do not rank "
            "only the mechanisms with the strongest current evidence."
        ),
        "separate_scores": {
            "differential_priority": (
                "How important the mechanism is to keep in the Top-K differential "
                "diagnosis under current uncertainty."
            ),
            "evidence_support": (
                "How strongly the collected runtime evidence supports the claim."
            ),
        },
        "candidate_policy": [
            (
                "A mechanism may have high differential priority and only weak/proxy "
                "evidence if it is structurally or photophysically important and "
                "requires validation outside the current evidence."
            ),
            (
                "Keep weak-but-important candidates as candidate_requires_validation "
                "instead of turning them into supported claims."
            ),
            (
                "Do not let broad, easy-to-proxy mechanisms automatically outrank "
                "more specific microscopic pathways when the observations make the "
                "specific pathway important."
            ),
            (
                "Do not assign dominant priority to RIM_RIR_RIV from rotatable bonds "
                "or generic flexibility alone. Treat broad RIM as a layer or "
                "secondary candidate unless runtime evidence makes motion restriction "
                "central."
            ),
            (
                "Avoid identical priority scores among top candidates. Break ties "
                "by runtime observations, explanatory specificity, and diagnostic "
                "importance rather than mechanism-pool order."
            ),
        ],
        "disambiguation_axes": [
            (
                "RIM_RIR_RIV is a broad motion-restriction class; RACI_CI_ACCESS is "
                "a more specific motion-coupled nonradiative pathway. If torsional "
                "or flapping coordinates affect excited-state brightness or dark-state "
                "risk, consider both, with explicit CI-search boundaries."
            ),
            (
                "PACKING_HOST_MATRIX_CONFINEMENT concerns environmental restriction "
                "or rigidification. AGGREGATE_EXCITON_EXCIMER needs aggregate-state "
                "electronic evidence for a supported claim, but aggregate/contact "
                "proxies may still make it a high-priority validation-needed candidate."
            ),
            (
                "ICT_TICT_CT requires charge-transfer character beyond a bare D-A "
                "motif. PET_ET requires an electron-transfer quenching or turn-on "
                "pathway, not merely donor/acceptor language. Weak D-A or orbital "
                "proxies should not outrank aggregate, packing, RACI, or radiative "
                "candidates that have more case-specific observations."
            ),
            (
                "RADIATIVE_RATE_STATE_BALANCE and SOKR_ANTI_KASHA can be important "
                "when state ordering, oscillator strength, bright/dark balance, or "
                "higher-state emission proxies are central, but they need explicit "
                "boundaries when only low-cost proxies exist."
            ),
        ],
        "route_selection_goal": (
            "Choose routes that discriminate competing mechanisms, not merely routes "
            "that provide easy support for already common mechanisms."
        ),
    }


def _compact_differential_ranking_guidance() -> dict[str, object]:
    return {
        "ranking_goal": "Evidence-aware differential mechanism candidate ranking.",
        "separate_scores": {
            "differential_priority": "candidate importance for Top-K retrieval",
            "evidence_support": "current evidence support strength",
        },
        "candidate_policy": (
            "A mechanism may rank high with weak/proxy support if it is important "
            "and marked candidate_requires_validation."
        ),
        "overclaim_policy": (
            "Do not convert structural/proxy observations into supported wet-lab, "
            "CI, lifetime, PLQY, or high-level computation claims."
        ),
    }


def _final_calibration_contract() -> dict[str, object]:
    return {
        "purpose": (
            "This is the final Planner-owned calibration pass before mechanism "
            "predictions are assembled. Do not dispatch new work here. Re-rank the "
            "existing full-pool portfolio from public runtime evidence, support "
            "arguments, reviewer feedback, and explicit evidence boundaries."
        ),
        "required_action": "finalize",
        "dispatch_rule": "selected_capability_ids and dispatch_requests must be empty.",
        "calibration_checks": [
            (
                "Review all mechanism_pool labels once more. Do not let broad "
                "easy-to-proxy labels crowd out a specific high-importance "
                "candidate just because the specific label only has proxy support."
            ),
            (
                "For top candidates, write Finding-Warrant-Boundary support "
                "arguments with existing EvidenceLedger refs whenever possible."
            ),
            (
                "If a candidate is important but only structurally/proxy supported, "
                "rank it by differential priority while marking evidence_support "
                "weak_or_proxy and claim_status candidate_requires_validation."
            ),
            (
                "If RIM, packing, or aggregate candidates are present only from "
                "generic structural priors, keep their support bounded and compare "
                "them against ESIPT, PET, radiative-rate, SOKR, triplet/metal, "
                "host-guest, and RACI candidates triggered by runtime observations."
            ),
            (
                "If state-ordering, oscillator-strength, frontier-orbital, charge, "
                "metal/heavy-atom, host/guest, proton-transfer, or torsion-brightness "
                "observations exist, explicitly decide whether the corresponding "
                "hard mechanism should enter Top-3 as a validation-needed candidate."
            ),
        ],
        "top3_output_rule": (
            "The final mechanism_predictions will be sorted from your calibrated "
            "portfolio. No later module can add a missing mechanism or change the "
            "ranking."
        ),
    }


def _compact_final_calibration_contract() -> dict[str, object]:
    return {
        "required_action": "finalize",
        "dispatch_rule": "selected_capability_ids and dispatch_requests must be empty.",
        "purpose": (
            "Calibrate final portfolio from runtime evidence and boundaries; no new "
            "tool work."
        ),
    }


def _mechanism_route_priority_map() -> dict[str, object]:
    return {
        "purpose": (
            "Generic mapping from public runtime observations to mechanism "
            "differential-priority updates. It is a workflow guide, not a private "
            "answer key and not a fixed ranking."
        ),
        "rules": [
            (
                "When a route directly screens a mechanism-specific question, "
                "raise or lower differential_priority based on the observation even "
                "if support_strength remains weak_or_proxy."
            ),
            (
                "Do not wait for direct wet-lab or high-level computation before "
                "keeping a mechanism in Top-3 if the low-cost route makes it a key "
                "differential candidate."
            ),
            (
                "Do not inflate broad labels from generic structural priors when a "
                "more specific route provides a competing explanation."
            ),
            (
                "Use evidence_tier when comparing close candidates: computed_direct_proxy "
                "or mechanism_specific_structural_trigger should usually outweigh a "
                "generic structural_trigger or electronic_weak_trigger unless the "
                "latter has stronger case-specific runtime observations."
            ),
        ],
        "evidence_tier_order": [
            "computed_direct_proxy",
            "mechanism_specific_structural_trigger",
            "structural_trigger",
            "electronic_weak_trigger",
            "artifact_weak_trigger",
            "boundary_only",
        ],
        "route_to_mechanisms": {
            "macro.screen_esipt_structural_motif": ["ESIPT_PT"],
            "macro.screen_intramolecular_hbond_preorganization": ["ESIPT_PT"],
            "macro.screen_metal_triplet_prior": ["TRIPLET_METAL_ENERGY_TRANSFER"],
            "macro.screen_rotor_torsion_topology": [
                "RIM_RIR_RIV",
                "ICT_TICT_CT",
            ],
            "macro.screen_rotor_rim_prior": ["RIM_RIR_RIV"],
            "macro.screen_donor_acceptor_layout": ["ICT_TICT_CT", "PET_ET"],
            "macro.screen_donor_acceptor_architecture": ["ICT_TICT_CT", "PET_ET"],
            "macro.screen_polar_binding_site_prior": [
                "HOST_GUEST_INTERACTION",
                "PET_ET",
            ],
            "macro.screen_aggregation_prone_scaffold": [
                "AGGREGATE_EXCITON_EXCIMER",
                "PACKING_HOST_MATRIX_CONFINEMENT",
                "RIM_RIR_RIV",
            ],
            "macro.screen_pi_stacking_prone_geometry": [
                "AGGREGATE_EXCITON_EXCIMER",
                "PACKING_HOST_MATRIX_CONFINEMENT",
            ],
            "microscopic.run_frontier_orbital_partition": ["ICT_TICT_CT", "PET_ET"],
            "microscopic.run_charge_population_panel": ["ICT_TICT_CT", "PET_ET"],
            "microscopic.run_baseline_bundle": [
                "RADIATIVE_RATE_STATE_BALANCE",
            ],
            "microscopic.run_bright_dark_state_ordering": [
                "RADIATIVE_RATE_STATE_BALANCE",
                "SOKR_ANTI_KASHA",
            ],
            "microscopic.run_targeted_transition_dipole_analysis": [
                "RADIATIVE_RATE_STATE_BALANCE",
                "SOKR_ANTI_KASHA",
            ],
            "microscopic.run_torsion_brightness_coupling_scan": [
                "RACI_CI_ACCESS",
                "RIM_RIR_RIV",
            ],
            "microscopic.run_torsion_snapshots": ["RACI_CI_ACCESS", "RIM_RIR_RIV"],
            "macro.run_crystal_restriction_checklist": [
                "PACKING_HOST_MATRIX_CONFINEMENT",
                "RIM_RIR_RIV",
            ],
            "macro.run_dimer_packing_proxy": [
                "PACKING_HOST_MATRIX_CONFINEMENT",
                "AGGREGATE_EXCITON_EXCIMER",
            ],
            "macro.run_aggregate_contact_proxy": [
                "PACKING_HOST_MATRIX_CONFINEMENT",
                "AGGREGATE_EXCITON_EXCIMER",
            ],
            "macro.run_solid_state_emission_proxy": [
                "PACKING_HOST_MATRIX_CONFINEMENT",
                "AGGREGATE_EXCITON_EXCIMER",
                "RIM_RIR_RIV",
            ],
            "microscopic.extract_ct_descriptors_from_bundle": [
                "ICT_TICT_CT",
                "PET_ET",
            ],
        },
    }


def _compact_mechanism_route_priority_map() -> dict[str, object]:
    route_map = _mechanism_route_priority_map()["route_to_mechanisms"]
    if not isinstance(route_map, dict):
        return {"route_to_mechanisms": {}}
    compact_routes = {
        key: route_map[key]
        for key in sorted(route_map)
    }
    return {
        "purpose": "Public route-to-mechanism coverage guide, not a fixed ranking.",
        "route_to_mechanisms": compact_routes,
        "evidence_tier_order": [
            "computed_direct_proxy",
            "mechanism_specific_structural_trigger",
            "structural_trigger",
            "electronic_weak_trigger",
            "boundary_only",
        ],
    }


def _full_portfolio_hypotheses(case_run: CaseRun) -> list[HypothesisEntry]:
    ordered_existing = [
        item
        for item in case_run.portfolio.sorted_hypotheses()
        if item.name != "unknown"
    ]
    existing_names = {item.name for item in ordered_existing}
    missing_pool = [
        HypothesisEntry(
            name=label,
            confidence=0.0,
            differential_priority=0.0,
            evidence_support="unsupported",
            claim_status="underdetermined",
            status="pending",
            rationale="Not yet prioritized by Planner.",
        )
        for label in MECHANISM_POOL
        if label not in existing_names
    ]
    return [*ordered_existing, *missing_pool]


def _compact_hypothesis_for_planner(item: HypothesisEntry) -> dict[str, object]:
    return {
        "name": item.name,
        "confidence": item.confidence,
        "differential_priority": item.differential_priority,
        "evidence_support": item.evidence_support,
        "claim_status": item.claim_status,
        "status": item.status,
        "rationale": _short_text(item.rationale, 220),
        "evidence_refs": item.evidence_refs[:8],
        "validation_needed": item.validation_needed[:6],
    }


def _compact_previous_portfolio_state(
    case_run: CaseRun,
    portfolio_hypotheses: list[HypothesisEntry],
) -> dict[str, object]:
    if _portfolio_is_initial_pool_state(case_run, portfolio_hypotheses):
        return {
            "current": case_run.portfolio.current,
            "runner_up": case_run.portfolio.runner_up,
            "hypotheses": [],
            "initial_state_note": (
                "All mechanism_pool labels start at priority 0 with unsupported "
                "evidence; use priority_deltas to initialize ranking from new evidence."
            ),
        }
    return {
        "current": case_run.portfolio.current,
        "runner_up": case_run.portfolio.runner_up,
        "hypotheses": [
            _compact_hypothesis_for_planner(item)
            for item in portfolio_hypotheses[:8]
        ],
    }


def _portfolio_is_initial_pool_state(
    case_run: CaseRun,
    portfolio_hypotheses: list[HypothesisEntry],
) -> bool:
    if case_run.portfolio.current not in {"", "unknown"}:
        return False
    if case_run.portfolio.runner_up not in {None, "", "unknown"}:
        return False
    concrete = [item for item in portfolio_hypotheses if item.name != "unknown"]
    if not concrete:
        return True
    return all(
        not item.evidence_refs
        and not item.validation_needed
        and (item.differential_priority is None or item.differential_priority == 0.0)
        and item.confidence == 0.0
        and item.evidence_support == "unsupported"
        for item in concrete
    )


def _new_round_evidence_for_planner(case_run: CaseRun) -> list[Any]:
    current_round_id = case_run.current_round_id
    if current_round_id:
        items = [
            item
            for item in case_run.evidence_ledger.items
            if item.round_id == current_round_id
        ]
        if items:
            return items
    return case_run.evidence_ledger.recent_items(limit=8)


def _public_case_id(case_run: CaseRun) -> str:
    return public_case_id_from_metadata(case_run.input.metadata)


def _compact_evidence_for_planner(item) -> dict[str, object]:
    capability_profile = _ability_profile_for_capability(item.capability_id)
    payload = {
        "evidence_id": item.evidence_id,
        "agent_name": item.agent_name,
        "capability_id": item.capability_id,
        "screens": capability_profile.get("screens", []),
        "evidence_tier": capability_profile.get("evidence_tier", "unknown"),
        "support_ceiling": capability_profile.get("support_ceiling", "weak_or_proxy"),
        "family": item.family,
        "status": item.status,
        "support": item.support,
        "claim": _short_text(item.claim, 120),
        "summary": _short_text(item.summary, 100),
        "limits": list(item.limits),
        "artifact_refs": list(item.artifact_refs),
        "observable": item.observable,
        "observable_tags": item.observable_tags[:3],
        "metrics": _compact_evidence_metrics(item.metrics),
    }
    return {
        **payload,
        "screens": _positive_screens_for_payload_item(payload),
    }


def _positive_screens_for_payload_item(item: dict[str, object]) -> list[str]:
    screens = []
    for raw_label in item.get("screens", []) if isinstance(item.get("screens"), list) else []:
        label = _normalize_mechanism_label(raw_label)
        if label in MECHANISM_POOL and _payload_screen_is_positive_for_label(label, item):
            screens.append(label)
    return list(dict.fromkeys(screens))


def _ability_profile_for_capability(capability_id: str) -> dict[str, object]:
    for item in compact_ability_evidence_guide(default_capability_registry()):
        if str(item.get("capability_id")) == capability_id:
            return {
                "screens": item.get("screens", []),
                "evidence_tier": item.get("evidence_tier", "unknown"),
                "support_ceiling": item.get("support_ceiling", "weak_or_proxy"),
            }
    return {"screens": [], "evidence_tier": "unknown", "support_ceiling": "weak_or_proxy"}


def _portfolio_has_full_mechanism_pool(
    case_run: CaseRun,
    portfolio_hypotheses: list[HypothesisEntry],
) -> bool:
    labels = {item.name for item in portfolio_hypotheses if item.name != "unknown"}
    if labels:
        return set(MECHANISM_POOL).issubset(labels)
    return case_run.portfolio.current in {"", "unknown"}


def _has_bootstrap_initial_coverage(case_run: CaseRun) -> bool:
    coverage = case_run.mechanism_agenda_coverage
    if coverage is None:
        return False
    return coverage.coverage_id.endswith(":agenda_coverage_bootstrap")


def _compact_evidence_metrics(metrics: object) -> dict[str, object]:
    if not isinstance(metrics, dict):
        return {}
    compact: dict[str, object] = {}
    for key, value in metrics.items():
        key_text = str(key)
        if key_text == "computed_observables" and isinstance(value, dict):
            compact[key_text] = value
            continue
        if key_text in {
            "smiles",
            "prepared_file_paths",
            "structure_source",
            "scaffold_proxy",
        }:
            continue
        if isinstance(value, bool | int | float) or value is None:
            compact[key_text] = value
            continue
        if isinstance(value, str):
            compact[key_text] = _short_text(value, 80)
            continue
        if isinstance(value, list):
            compact[key_text] = [
                safe_item
                for item in value[:2]
                if (safe_item := _safe_metric_list_item(item)) is not None
            ]
            continue
        if isinstance(value, dict):
            compact[key_text] = {
                str(inner_key): (
                    _short_text(inner_value, 80)
                    if isinstance(inner_value, str)
                    else inner_value
                )
                for inner_key, inner_value in list(value.items())[:4]
                if isinstance(inner_value, str | int | float | bool) or inner_value is None
            }
    return compact


def _safe_metric_list_item(value: object) -> object | None:
    if isinstance(value, str):
        return _short_text(value, 80)
    if isinstance(value, int | float | bool):
        return value
    if isinstance(value, dict):
        safe_dict = {
            str(inner_key): (
                _short_text(inner_value, 48)
                if isinstance(inner_value, str)
                else inner_value
            )
            for inner_key, inner_value in list(value.items())[:4]
            if isinstance(inner_value, str | int | float | bool) or inner_value is None
        }
        return safe_dict or None
    return None


def _compact_support_argument_for_planner(
    item: MechanismSupportArgument,
) -> dict[str, object]:
    return {
        "support_id": item.support_id,
        "target_label": item.target_label,
        "support_level": item.support_level,
        "finding": _short_text(item.finding, 180),
        "warrant": _short_text(item.warrant, 220),
        "boundary": _short_text(item.boundary, 220),
        "observation_refs": list(item.observation_refs),
        "audit_status": item.audit_status,
        "audit_notes": [_short_text(note, 160) for note in item.audit_notes[:3]],
    }


def _mechanism_evidence_coverage_table(case_run: CaseRun) -> list[dict[str, object]]:
    route_map = _mechanism_route_priority_map()["route_to_mechanisms"]
    if not isinstance(route_map, dict):
        return []
    evidence_tier_by_capability = _evidence_tier_by_capability()
    rows: dict[str, dict[str, object]] = {
        label: {
            "label": label,
            "route_hits": [],
            "evidence_refs": [],
            "positive_or_proxy_signal_count": 0,
            "failed_or_unsupported_route_count": 0,
            "latest_signal_summary": "",
        }
        for label in MECHANISM_POOL
    }
    for item in case_run.evidence_ledger.items:
        labels = route_map.get(item.capability_id)
        if not isinstance(labels, list):
            continue
        route_hit = {
            "evidence_id": item.evidence_id,
            "capability_id": item.capability_id,
            "round_id": item.round_id,
            "status": item.status,
            "support": item.support,
            "basis": item.basis,
            "signal_type": _evidence_signal_type(item),
            "evidence_tier": evidence_tier_by_capability.get(
                item.capability_id,
                "unknown",
            ),
            "summary": _short_text(item.summary or item.claim, 180),
            "claim": _short_text(item.claim, 180),
            "limits": [_short_text(limit, 120) for limit in item.limits[:2]],
            "observable_tags": item.observable_tags[:6],
        }
        payload_item = {
            "capability_id": item.capability_id,
            "metrics": item.metrics,
        }
        for label in labels:
            if label not in rows:
                continue
            if not _payload_screen_is_positive_for_label(label, payload_item):
                continue
            row = rows[label]
            route_hits = row["route_hits"]
            refs = row["evidence_refs"]
            if isinstance(route_hits, list):
                route_hits.append(route_hit)
                row["route_hits"] = route_hits[-6:]
            if isinstance(refs, list):
                refs.append(item.evidence_id)
                row["evidence_refs"] = list(dict.fromkeys(refs))[-8:]
            if _evidence_signal_type(item) == "positive_or_proxy_observable":
                row["positive_or_proxy_signal_count"] = (
                    int(row["positive_or_proxy_signal_count"]) + 1
                )
            if _evidence_signal_type(item) in {
                "missing_or_failed",
                "checklist_boundary_only",
            }:
                row["failed_or_unsupported_route_count"] = (
                    int(row["failed_or_unsupported_route_count"]) + 1
                )
            row["latest_signal_summary"] = _short_text(item.claim or item.summary, 220)
    return [
        row
        for row in rows.values()
        if row["route_hits"]
    ][: len(MECHANISM_POOL)]


def _compact_mechanism_evidence_hits(case_run: CaseRun) -> list[dict[str, object]]:
    rows = _mechanism_evidence_coverage_table(case_run)
    return [
        {
            "label": row["label"],
            "evidence_refs": row["evidence_refs"],
            "positive_or_proxy_signal_count": row["positive_or_proxy_signal_count"],
            "failed_or_unsupported_route_count": row["failed_or_unsupported_route_count"],
            "latest_signal_summary": _short_text(row["latest_signal_summary"], 120),
            "capability_ids": list(
                dict.fromkeys(
                    str(hit.get("capability_id"))
                    for hit in row.get("route_hits", [])
                    if isinstance(hit, dict) and hit.get("capability_id")
                )
            )[:4],
        }
        for row in rows
    ]


def _ultra_compact_mechanism_evidence_hits(case_run: CaseRun) -> list[dict[str, object]]:
    rows = _mechanism_evidence_coverage_table(case_run)
    return [
        {
            "label": row["label"],
            "evidence_refs": row["evidence_refs"][:4],
            "signal_count": row["positive_or_proxy_signal_count"],
            "capability_ids": list(
                dict.fromkeys(
                    str(hit.get("capability_id"))
                    for hit in row.get("route_hits", [])
                    if isinstance(hit, dict) and hit.get("capability_id")
                )
            )[:3],
        }
        for row in rows
    ]


def _priority_hygiene_summary(
    case_run: CaseRun,
    *,
    compact: bool = False,
) -> list[dict[str, object]]:
    """Summarize evidence directness for Planner calibration without ranking."""

    evidence_by_id = {item.evidence_id: item for item in case_run.evidence_ledger.items}
    route_map = _mechanism_route_priority_map()["route_to_mechanisms"]
    if not isinstance(route_map, dict):
        return []
    evidence_tier_by_capability = _evidence_tier_by_capability()
    tiers_by_label: dict[str, list[dict[str, object]]] = {label: [] for label in MECHANISM_POOL}
    for item in case_run.evidence_ledger.items:
        labels = route_map.get(item.capability_id)
        if not isinstance(labels, list):
            continue
        payload_item = {
            "capability_id": item.capability_id,
            "key_metrics": _compact_evidence_metrics(item.metrics),
            "status": item.status,
            "support": item.support,
            "basis": item.basis,
            "claim": item.claim,
            "summary": item.summary,
        }
        for label in labels:
            if label not in tiers_by_label:
                continue
            signal_type = _evidence_signal_type(item)
            if signal_type not in {
                "missing_or_failed",
                "checklist_boundary_only",
            } and not _payload_screen_is_positive_for_label(
                label,
                payload_item,
            ) and not _is_generic_aggregate_calibration_hit(label, item.capability_id):
                continue
            tiers_by_label[label].append(
                {
                    "evidence_id": item.evidence_id,
                    "capability_id": item.capability_id,
                    "evidence_tier": evidence_tier_by_capability.get(
                        item.capability_id,
                        "unknown",
                    ),
                    "signal_type": _evidence_signal_type(item),
                    "status": item.status,
                    "support": item.support,
                    "key_metrics": _compact_evidence_metrics(item.metrics),
                }
            )
    summary: list[dict[str, object]] = []
    portfolio_items = list(case_run.portfolio.sorted_hypotheses()) or [
        HypothesisEntry(name=label) for label in MECHANISM_POOL
    ]
    for item in portfolio_items:
        label = item.name
        if label not in tiers_by_label or label == "unknown":
            continue
        route_hits = tiers_by_label[label]
        explicit_refs = [ref for ref in item.evidence_refs if ref in evidence_by_id]
        critic_refs = _critic_positive_refs_for_label(case_run, label)
        directness = _directness_summary(route_hits)
        summary.append(
            {
                "label": label,
                "current_priority": _priority_for_payload(item.model_dump(mode="json")),
                "evidence_refs": list(
                    dict.fromkeys(
                        [
                            *explicit_refs,
                            *[
                                str(hit.get("evidence_id"))
                                for hit in route_hits
                                if str(hit.get("evidence_id") or "").strip()
                            ],
                            *critic_refs,
                        ]
                    )
                )[:6],
                "best_evidence_tier": directness["best_evidence_tier"],
                "computed_direct_proxy_count": directness["computed_direct_proxy_count"],
                "mechanism_specific_structural_count": directness[
                    "mechanism_specific_structural_count"
                ],
                "critic_positive_ref_count": len(critic_refs),
                "structural_or_electronic_weak_count": directness[
                    "structural_or_electronic_weak_count"
                ],
                "boundary_or_failed_count": directness["boundary_or_failed_count"],
                "calibration_note": _short_text(
                    _priority_hygiene_note(label, directness),
                    260,
                ),
                "route_hits": _compact_priority_route_hits(route_hits[:4]),
            }
        )
    return summary[: len(MECHANISM_POOL)]


def _is_generic_aggregate_calibration_hit(label: str, capability_id: object) -> bool:
    return label == "AGGREGATE_EXCITON_EXCIMER" and str(capability_id or "") in {
        "macro.screen_aggregation_prone_scaffold",
        "macro.run_solid_state_emission_proxy",
        "macro.run_dimer_packing_proxy",
        "macro.run_aggregate_contact_proxy",
    }


def _compact_priority_route_hits(
    route_hits: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "evidence_id": item.get("evidence_id"),
            "capability_id": item.get("capability_id"),
            "evidence_tier": item.get("evidence_tier"),
            "signal_type": item.get("signal_type"),
            "status": item.get("status"),
            "support": item.get("support"),
            "key_metrics": _compact_priority_route_hit_metrics(item),
        }
        for item in route_hits
    ]


def _compact_priority_route_hit_metrics(
    route_hit: dict[str, object],
) -> dict[str, object]:
    metrics = route_hit.get("key_metrics")
    if not isinstance(metrics, dict):
        return {}
    keep = {
        "aggregation_prone_proxy",
        "aromatic_ring_count",
        "esipt_motif_proxy",
        "hbd_count",
        "heavy_atom_triplet_prior",
        "host_guest_followup_trigger",
        "metal_or_lanthanide_prior",
        "oscillator_strength",
        "pet_receptor_like_proxy",
        "polar_binding_site_proxy",
        "carbonyl_like_site_count",
        "proton_transfer_pair_proxy_count",
        "rotatable_bond_count",
        "rotatable_bond_inventory",
        "state_count",
        "sulfur_phosphorus_triplet_prior",
        "triplet_metal_followup_trigger",
        "torsion_candidate_count",
    }
    return {key: value for key, value in metrics.items() if key in keep}


def _critic_positive_refs_for_label(case_run: CaseRun, label: str) -> list[str]:
    portfolio = case_run.differential_mechanism_portfolio
    if portfolio is None:
        return []
    refs: list[str] = []
    for attribution in portfolio.evidence_attributions:
        for update in attribution.updates:
            if update.label != label:
                continue
            if update.priority_effect == "decrease" or update.support_effect == "weakens":
                continue
            if update.priority_effect == "increase" or update.support_effect == "supports":
                refs.append(attribution.evidence_id)
    return list(dict.fromkeys(refs))


def _directness_summary(route_hits: list[dict[str, object]]) -> dict[str, object]:
    tier_order = {
        "computed_direct_proxy": 5,
        "mechanism_specific_structural_trigger": 4,
        "structural_trigger": 3,
        "electronic_weak_trigger": 2,
        "artifact_weak_trigger": 1,
        "boundary_only": 0,
        "unknown": 0,
    }
    tiers = [str(hit.get("evidence_tier") or "unknown") for hit in route_hits]
    best = max(tiers, key=lambda tier: tier_order.get(tier, 0), default="unknown")
    signal_types = [str(hit.get("signal_type") or "") for hit in route_hits]
    positive_tiers = [
        tier
        for tier, signal in zip(tiers, signal_types, strict=False)
        if signal not in {"missing_or_failed", "checklist_boundary_only"}
    ]
    return {
        "best_evidence_tier": best,
        "computed_direct_proxy_count": positive_tiers.count("computed_direct_proxy"),
        "mechanism_specific_structural_count": positive_tiers.count(
            "mechanism_specific_structural_trigger"
        ),
        "structural_or_electronic_weak_count": sum(
            tier in {"structural_trigger", "electronic_weak_trigger", "artifact_weak_trigger"}
            for tier in positive_tiers
        ),
        "boundary_or_failed_count": sum(
            tier == "boundary_only"
            or signal in {"missing_or_failed", "checklist_boundary_only"}
            for tier, signal in zip(tiers, signal_types, strict=False)
        ),
    }


def _priority_hygiene_note(label: str, directness: dict[str, object]) -> str:
    if label == "AGGREGATE_EXCITON_EXCIMER" and int(
        directness["structural_or_electronic_weak_count"]
    ) > 0 and int(directness["computed_direct_proxy_count"]) == 0:
        return (
            "AGGREGATE_EXCITON_EXCIMER currently appears supported only by "
            "structural/contact/solid-state proxies. Without aggregate "
            "excited-state evidence, prefer PACKING_HOST_MATRIX_CONFINEMENT over "
            "the aggregate-exciton/excimer label when both rely on the same proxy."
        )
    if label == "PACKING_HOST_MATRIX_CONFINEMENT" and int(
        directness["structural_or_electronic_weak_count"]
    ) > 0:
        return (
            "PACKING_HOST_MATRIX_CONFINEMENT can be the more appropriate "
            "differential candidate when structural, solid-state, contact, or "
            "confinement proxies exist but aggregate excited-state evidence is "
            "missing."
        )
    if int(directness["computed_direct_proxy_count"]) > 0:
        return (
            f"{label} has low-cost computed proxy evidence; compare it against "
            "candidates supported only by generic structural or electronic weak triggers."
        )
    if int(directness["mechanism_specific_structural_count"]) > 0:
        return (
            f"{label} has a mechanism-specific structural trigger; keep support "
            "bounded but do not ignore it during differential ranking."
        )
    if int(directness["boundary_or_failed_count"]) > 0 and int(
        directness["structural_or_electronic_weak_count"]
    ) == 0:
        return (
            f"{label} currently has only boundary or failed-route evidence; it "
            "should not gain priority from that alone."
        )
    if int(directness["structural_or_electronic_weak_count"]) > 0:
        return (
            f"{label} is currently supported only by generic structural/electronic "
            "weak triggers; avoid letting this alone dominate higher-directness "
            "runtime observations."
        )
    return f"{label} has no mechanism-specific runtime evidence yet."


def _evidence_tier_by_capability() -> dict[str, str]:
    return {
        str(item.get("capability_id")): str(item.get("evidence_tier") or "unknown")
        for item in compact_ability_evidence_guide(default_capability_registry())
    }


def _evidence_signal_type(item) -> str:
    if item.status in {"failed", "unsupported", "missing"}:
        return "missing_or_failed"
    if _is_material_followup_capability(str(item.capability_id or "")):
        return "positive_or_proxy_observable"
    text = f"{item.claim} {item.summary}".lower()
    if item.basis == "checklist" or any(
        term in text
        for term in (
            "records which",
            "checklist",
        )
    ):
        return "checklist_boundary_only"
    return "positive_or_proxy_observable"


def _compact_differential_portfolio_for_planner(
    case_run: CaseRun,
) -> dict[str, object] | None:
    portfolio = case_run.differential_mechanism_portfolio
    if portfolio is None:
        return None
    return {
        "round_id": portfolio.round_id,
        "rows": [
            {
                "label": row.label,
                "trigger_status": row.trigger_status,
                "differential_priority": row.differential_priority,
                "support_strength": row.support_strength,
                "claim_status": row.claim_status,
                "positive_evidence_refs": row.positive_evidence_refs[:5],
                "negative_evidence_refs": row.negative_evidence_refs[:5],
                "missing_validation": row.missing_validation[:4],
                "rationale": _short_text(row.rationale, 180),
            }
            for row in sorted(
                portfolio.rows,
                key=lambda item: item.differential_priority,
                reverse=True,
            )
        ],
    }


def _compact_agenda_coverage_for_planner(coverage) -> dict[str, object] | None:
    if coverage is None:
        return None
    return {
        "round_id": coverage.round_id,
        "coverage_summary": _short_text(coverage.coverage_summary, 120),
        "decision_challenges": coverage.decision_challenges,
        "agenda_items": [
            {
                "label": item.label,
                "coverage_status": item.coverage_status,
                "why_relevant": item.why_relevant,
                "missing_evidence": item.missing_evidence,
                "suggested_question": item.suggested_question,
                "boundary": item.boundary,
                "suggested_route": _short_text(item.suggested_route, 160),
                "urgency": item.urgency,
            }
            for item in coverage.agenda_items[:6]
        ],
        "screened_out": [
            {
                "label": item.label,
                "reason": _short_text(item.reason, 180),
            }
            for item in coverage.screened_out[:4]
        ],
        "low_margin_competitions": [
            {
                "labels": item.labels[:4],
                "reason": item.reason,
                "suggested_disambiguating_route": _short_text(
                    item.suggested_disambiguating_route,
                    180,
                ),
            }
            for item in coverage.low_margin_competitions[:3]
        ],
        "recommended_next_routes": [
            {
                "route": _short_text(item.route, 180),
                "targets": item.targets[:4],
                "reason": item.reason,
                "capability_ids": item.capability_ids[:4],
            }
            for item in coverage.recommended_next_routes[:6]
        ],
    }


def _minimal_agenda_coverage_for_planner(coverage) -> dict[str, object] | None:
    if coverage is None:
        return None
    recommended_next_routes = [
        {
            "capability_ids": route.capability_ids[:2],
            "targets": route.targets[:3],
        }
        for route in coverage.recommended_next_routes[:6]
    ]
    return {
        "round_id": coverage.round_id,
        "coverage_summary": _short_text(coverage.coverage_summary, 80),
        "decision_challenges": coverage.decision_challenges,
        "recommended_next_capability_ids": list(
            dict.fromkeys(
                capability_id
                for route in coverage.recommended_next_routes[:6]
                for capability_id in route.capability_ids[:2]
            )
        )[:8],
        "recommended_next_routes": recommended_next_routes,
        "agenda_items": [
            {
                "label": item.label,
                "coverage_status": item.coverage_status,
                "urgency": item.urgency,
            }
            for item in coverage.agenda_items[:8]
        ],
        "under_screened_labels": [
            item.label
            for item in coverage.agenda_items[:8]
            if item.coverage_status in {"under_screened", "needs_follow_up"}
        ],
    }


def _compact_claim_context(case_run: CaseRun) -> dict[str, object]:
    ledger = case_run.claim_ledger
    if ledger is None:
        return {"open_claim_ids": [], "blocked_claim_ids": []}
    return {
        "open_claim_ids": ledger.open_claim_ids()[:12],
        "blocked_claim_ids": ledger.blocked_claim_ids()[:12],
        "coverage_debt_hypotheses": [
            {
                "hypothesis": item.hypothesis,
                "open_claim_ids": item.open_claim_ids[:8],
                "blocked_claim_ids": item.blocked_claim_ids[:8],
            }
            for item in ledger.hypothesis_debts[:8]
            if item.status in {"open", "blocked"}
        ],
    }


def _compact_mechanism_agenda_for_planner(program) -> dict[str, object] | None:
    if program is None:
        return None
    return {
        "candidate_mechanisms": [
            {
                "mechanism_id": item.mechanism_id,
                "label": item.label,
                "mechanism_family": item.mechanism_family,
            }
            for item in program.candidate_mechanisms[:6]
        ],
        "evidence_questions": [
            {
                "question_id": item.question_id,
                "mechanism_id": item.mechanism_id,
                "observable": item.observable,
                "status": item.status,
                "acceptable_capability_ids": item.acceptable_capability_ids[:4],
            }
            for item in program.evidence_questions[:8]
        ],
        "computational_routes": [
            {
                "question_id": item.question_id,
                "capability_id": item.capability_id,
                "agent_name": item.agent_name,
                "route": item.route,
            }
            for item in program.computational_routes[:10]
        ],
        "scope_boundaries": [
            _short_text(item.statement, 220)
            for item in program.scope_boundaries[:6]
        ],
    }


def _compact_photophysics_review_for_planner(review) -> dict[str, object] | None:
    if review is None:
        return None
    return {
        "coverage_axes": [
            {
                "label": item.label,
                "status": item.status,
                "evidence_refs": item.evidence_refs[:6],
                "rationale": _short_text(item.rationale, 240),
                "recommended_routes": item.recommended_routes[:4],
            }
            for item in review.coverage_axes[:8]
        ],
        "hypothesis_cards": [
            {
                "mechanism": item.mechanism,
                "status": item.status,
                "support_evidence_refs": item.support_evidence_refs[:6],
                "weakening_evidence_refs": item.weakening_evidence_refs[:6],
                "reasoning_summary": _short_text(item.reasoning_summary, 260),
                "missing_or_unresolved": item.missing_or_unresolved[:4],
            }
            for item in review.hypothesis_cards[:8]
        ],
        "recommended_next_routes": review.recommended_next_routes[:6],
        "overclaim_warnings": [
            _short_text(item, 180) for item in review.overclaim_warnings[:4]
        ],
    }


def _compact_context_for_planner(
    context: dict[str, object],
    *,
    compact: bool = False,
) -> dict[str, object]:
    if compact:
        return {
            "round_index": context.get("round_index"),
            "max_rounds": context.get("max_rounds"),
            "open_claim_count": len(list(context.get("open_claim_ids") or [])),
            "blocked_claim_count": len(list(context.get("blocked_claim_ids") or [])),
            "operational_notes": list(context.get("operational_notes") or [])[-6:],
        }
    return {
        "round_index": context.get("round_index"),
        "max_rounds": context.get("max_rounds"),
        "open_claim_ids": list(context.get("open_claim_ids") or [])[:12],
        "blocked_claim_ids": list(context.get("blocked_claim_ids") or [])[:12],
        "operational_notes": list(context.get("operational_notes") or [])[-6:],
    }


def _ablation_constraints(case_run: CaseRun) -> dict[str, object]:
    raw = case_run.runtime.get("ablation")
    if not isinstance(raw, dict):
        return {}
    disabled_owners = [
        str(item)
        for item in raw.get("disabled_capability_owners", [])
        if str(item).strip()
    ]
    return {
        "mode": str(raw.get("mode") or "full_mechcal"),
        "disabled_capability_owners": disabled_owners,
        "available_capability_owners": [
            str(item)
            for item in raw.get("enabled_capability_owners", [])
            if str(item).strip()
        ],
        "no_substitution": bool(disabled_owners),
        "missing_domain_evidence_policy": (
            "Preserve unavailable-domain observations as evidence gaps or "
            "validation needs. Do not ask another worker to simulate the "
            "disabled capability owner."
        ),
    }


def _compact_capability_cards(
    capability_registry: CapabilityRegistry,
    *,
    case_run: CaseRun,
    stage: str,
) -> list[dict[str, object]]:
    return [
        {
            "capability_id": item["capability_id"],
            "owner_agent": item["owner_agent"],
            "evidence_family": item["evidence_family"],
            "route": item["route"],
            "description": item["description"],
            "required_artifact_kinds": item["required_artifact_kinds"],
            "source_artifact_kinds": item.get("source_artifact_kinds", []),
            "tool_arg_schema": item["tool_arg_schema"],
        }
        for item in capability_registry.cards()
        if str(item.get("capability_id", "")) in _planner_visible_capability_ids(
            capability_registry,
            case_run=case_run,
            stage=stage,
        )
    ]


def _compact_planner_ability_guide(
    capability_registry: CapabilityRegistry,
    *,
    case_run: CaseRun,
    stage: str,
) -> list[dict[str, object]]:
    visible = _planner_visible_capability_ids(
        capability_registry,
        case_run=case_run,
        stage=stage,
    )
    return [
        {
            "capability_id": item.get("capability_id"),
            "screens": item.get("screens", []),
            "evidence_tier": item.get("evidence_tier"),
            "suggested_claim_strength": item.get("suggested_claim_strength"),
        }
        for item in compact_ability_evidence_guide(capability_registry)
        if str(item.get("capability_id", "")) in visible
    ]


def _minimal_planner_ability_guide(
    capability_registry: CapabilityRegistry,
    *,
    case_run: CaseRun,
    stage: str,
) -> list[dict[str, object]]:
    visible = _planner_visible_capability_ids(
        capability_registry,
        case_run=case_run,
        stage=stage,
    )
    return [
        {
            "capability_id": item.get("capability_id"),
            "screens": item.get("screens", []),
            "evidence_tier": item.get("evidence_tier"),
        }
        for item in compact_ability_evidence_guide(capability_registry)
        if str(item.get("capability_id", "")) in visible
    ]


def _planner_visible_capability_ids(
    capability_registry: CapabilityRegistry,
    *,
    case_run: CaseRun,
    stage: str,
) -> set[str]:
    return _artifact_available_capability_ids(capability_registry, case_run)


def _artifact_available_capability_ids(
    capability_registry: CapabilityRegistry,
    case_run: CaseRun,
) -> set[str]:
    available_kinds = {
        artifact.kind
        for artifact in case_run.artifact_manifest.artifacts
        if artifact.status in {"available", "partial"}
    }
    ids: set[str] = set()
    for card in capability_registry.cards():
        required = {
            str(kind)
            for kind in card.get("required_artifact_kinds", [])
            if str(kind).strip()
        }
        alternatives = set(card.get("source_artifact_kinds", []))
        if required.issubset(available_kinds) and (not alternatives or alternatives & available_kinds):
            ids.add(str(card.get("capability_id", "")))
    return ids




def _planner_output_contract(*, stopping: bool = False) -> dict[str, object]:
    contract = {
        "mode": "complete_planner_owned_ranking",
        "required_keys": [
            "decision_id",
            "round_id",
            "portfolio",
            "current_hypothesis",
            "confidence",
            "diagnosis",
            "decision_assessment",
            "action",
            "selected_capability_ids",
            "unresolved_gaps",
            "priority_deltas",
            "mechanism_support_arguments",
            "final_answer_draft",
            "rationale",
            "raw_response",
        ],
        "portfolio_update": "Use {current, runner_up, hypotheses: []} in updates.",
        "decision_assessment": {
            "current_best": "Best current explanations across the supplied pool and why.",
            "weakened_or_unresolved": "What weakened, or remains indistinguishable, and why.",
            "next_useful_step": "What outcomes would change the judgment, or why finish.",
            "evidence_refs": ["existing evidence_id; empty before observations"],
        },
        "delta_fields": [
            "label",
            "previous_priority",
            "priority_delta",
            "new_priority",
            "delta_reason",
            "evidence_refs_added",
            "support_change",
        ],
        "support_argument_fields": [
            "support_id",
            "target_label",
            "support_level",
            "finding",
            "warrant",
            "boundary",
            "observation_refs",
        ],
    }
    if stopping:
        contract["required_keys"].append("stopping_assessment")
        contract["stopping_assessment"] = {
            "judgment_state": "adequate_for_bounded_ranking|unresolved",
            "finish_basis": "judgment_sufficient|no_useful_available_step|null",
            "reason": "Brief scientific reason; the runtime records any actual limit.",
        }
        contract["reviewer_response"] = "Required when stop_review_feedback is supplied; otherwise null."
    return contract


def _normalize_planner_llm_response(
    response: dict[str, object],
    *,
    round_id: str,
    existing_portfolio: dict[str, object],
    previous_portfolio_hypotheses: list[dict[str, object]],
    allowed_evidence_ids: list[str],
    capability_registry: CapabilityRegistry,
    stage: str = "planner_update",
    prompt_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    payload = _unwrap_planner_response(response)
    if not payload:
        raise ValueError("Planner LLM response must not be an empty JSON object.")
    if payload.get("round_id", round_id) != round_id:
        raise ValueError("Planner round_id must match the current round.")
    portfolio = _extract_portfolio_payload(payload, {})
    if _is_delta_only_portfolio_shell(portfolio):
        raise ValueError("Return a complete portfolio.hypotheses list, not a delta-only update.")
    portfolio = _sort_portfolio_by_priority(_normalize_portfolio_payload(portfolio))
    priority_deltas = _normalize_priority_deltas(
        payload,
        round_id=round_id,
        previous_hypotheses=previous_portfolio_hypotheses,
        portfolio=portfolio,
        allowed_evidence_ids=allowed_evidence_ids,
    )
    low_margin_notes = _low_margin_notes_for_portfolio(portfolio)
    raw_current = _first_present_text(
        payload,
        (
            "current_hypothesis",
            "primary_hypothesis",
            "leading_hypothesis",
            "current_mechanism",
            "primary_mechanism",
        ),
    )
    current_candidate = raw_current if raw_current.lower() != "unknown" else ""
    top_from_planner = _top_hypothesis_name(portfolio)
    if top_from_planner and top_from_planner != current_candidate:
        current_candidate = _top_hypothesis_name(portfolio) or current_candidate
    current = _short_text(
        current_candidate or portfolio.get("current") or "unknown",
        120,
    )
    portfolio["current"] = current
    runner_up = (
        _second_hypothesis_name(portfolio)
        if top_from_planner == current
        else (
            _first_present_text(
                payload,
                (
                    "runner_up_hypothesis",
                    "runner_up",
                    "secondary_hypothesis",
                    "secondary_mechanism",
                ),
            )
            or portfolio.get("runner_up")
        )
    )
    runner_up = None if _is_nullish(runner_up) else _short_text(runner_up, 120)
    portfolio["runner_up"] = runner_up

    action = str(payload.get("action") or "").strip().lower()
    if action not in {"dispatch", "finalize", "stop"}:
        raise ValueError("Planner action must be dispatch, finalize, or stop.")
    if stage == "planner_final_calibration" and action != "finalize":
        raise ValueError("Final calibration requires an explicit finalize decision.")
    raw_dispatches = payload.get("dispatch_requests", [])
    if not isinstance(raw_dispatches, list) or any(not isinstance(item, dict) for item in raw_dispatches):
        raise ValueError("dispatch_requests must be a list of objects.")
    dispatch_requests = [
        _normalize_dispatch_payload(item, round_id=round_id)
        for item in raw_dispatches
    ]
    if not dispatch_requests:
        dispatch_requests = _dispatches_from_capability_ids(
            _selected_capability_ids(payload),
            round_id=round_id,
            capability_registry=capability_registry,
        )
    if len(dispatch_requests) > MAX_DISPATCHES_PER_ROUND:
        raise ValueError(f"Select at most {MAX_DISPATCHES_PER_ROUND} tool actions per round.")
    if action != "dispatch" and dispatch_requests:
        raise ValueError("Only dispatch decisions may request tools.")
    final_answer_draft = payload.get("final_answer_draft")
    if _is_nullish(final_answer_draft):
        final_answer_draft = None
    if action == "finalize" and not str(final_answer_draft or "").strip():
        raise ValueError("Finalization requires the Planner's non-empty final_answer_draft.")

    return {
        "decision_id": _short_text(
            payload.get("decision_id") or f"{round_id}:planner_llm",
            160,
        ),
        "round_id": _short_text(payload.get("round_id") or round_id, 80),
        "portfolio": portfolio,
        "current_hypothesis": current,
        "runner_up_hypothesis": runner_up,
        "confidence": _float_between(payload.get("confidence"), default=0.0),
        "diagnosis": _short_text(payload.get("diagnosis"), 1200)
        or "Planner LLM returned no diagnosis text.",
        "decision_assessment": payload.get("decision_assessment"),
        "stopping_assessment": payload.get("stopping_assessment"),
        "reviewer_response": payload.get("reviewer_response"),
        "action": action,
        "dispatch_requests": dispatch_requests,
        "unresolved_gaps": _text_list(payload.get("unresolved_gaps"), limit=12),
        "priority_deltas": priority_deltas,
        "mechanism_support_arguments": _normalize_support_arguments(
            payload,
            round_id=round_id,
            allowed_evidence_ids=allowed_evidence_ids,
        ),
        "final_answer_draft": final_answer_draft if action == "finalize" else None,
        "rationale": _short_text(payload.get("rationale"), 600),
        "raw_response": {
            **(
                payload.get("raw_response")
                if isinstance(payload.get("raw_response"), dict)
                else {}
            ),
            "low_margin_competitions": low_margin_notes,
            "priority_delta_count": len(priority_deltas),
            "planner_stage": stage,
            "model_response": response,
        },
    }


def _unwrap_planner_response(response: dict[str, object]) -> dict[str, object]:
    for key in ("planner_decision", "decision", "response"):
        value = response.get(key)
        if isinstance(value, dict):
            return value
    return response


def _extract_portfolio_payload(
    payload: dict[str, object],
    existing_portfolio: dict[str, object],
) -> dict[str, object]:
    for key in (
        "portfolio",
        "hypothesis_portfolio",
        "mechanism_portfolio",
    ):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    for key in (
        "hypotheses",
        "candidate_hypotheses",
        "hypothesis_candidates",
        "mechanism_hypotheses",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return {
                "current": _first_present_text(
                    payload,
                    (
                        "current_hypothesis",
                        "primary_hypothesis",
                        "leading_hypothesis",
                        "current_mechanism",
                        "primary_mechanism",
                    ),
                ),
                "runner_up": _first_present_text(
                    payload,
                    (
                        "runner_up_hypothesis",
                        "runner_up",
                        "secondary_hypothesis",
                        "secondary_mechanism",
                    ),
                ),
                "hypotheses": value,
            }
    return existing_portfolio


def _normalize_portfolio_payload(portfolio: dict[str, object]) -> dict[str, object]:
    hypothesis_items = (
        _as_dict_list(portfolio.get("hypotheses"))
        or _as_dict_list(portfolio.get("candidate_hypotheses"))
        or _as_dict_list(portfolio.get("hypothesis_candidates"))
        or _as_dict_list(portfolio.get("mechanism_hypotheses"))
        or _as_dict_list(portfolio.get("items"))
        or _as_dict_list(portfolio.get("entries"))
    )
    hypotheses = []
    for item in hypothesis_items:
        name = _short_text(
            item.get("name")
            or item.get("hypothesis")
            or item.get("hypothesis_name")
            or item.get("mechanism")
            or item.get("mechanism_name")
            or item.get("label")
            or "unknown",
            120,
        )
        status = str(item.get("status") or "pending").strip().lower()
        if status not in {"plausible", "pending", "screened", "blocked", "dropped", "unknown"}:
            status = "pending"
        confidence = _float_between(item.get("confidence"), default=0.0)
        differential_priority = _validated_planner_priority(item)
        hypotheses.append(
            {
                "name": name or "unknown",
                "confidence": confidence,
                "differential_priority": differential_priority,
                "evidence_support": _normalize_prediction_support_strength(
                    item.get("evidence_support")
                    or item.get("support_strength")
                    or item.get("support_level")
                ),
                "claim_status": _normalize_claim_status(item.get("claim_status")),
                "status": status,
                "rationale": _short_text(
                    item.get("rationale")
                    or item.get("reasoning")
                    or item.get("reasoning_summary"),
                    360,
                ),
                "evidence_refs": _text_list(item.get("evidence_refs"), limit=8),
                "validation_needed": _text_list(
                    item.get("validation_needed")
                    or item.get("needed_validation")
                    or item.get("missing_validation"),
                    limit=6,
                ),
            }
        )
    if not hypotheses:
        hypotheses = [
            {
                "name": "unknown",
                "confidence": 0.2,
                "differential_priority": 0.2,
                "evidence_support": "unsupported",
                "claim_status": "underdetermined",
                "status": "pending",
                "rationale": "No valid hypothesis list was returned.",
                "evidence_refs": [],
                "validation_needed": [],
            }
        ]
    current = _short_text(
        portfolio.get("current")
        or portfolio.get("primary_hypothesis")
        or portfolio.get("leading_hypothesis")
        or portfolio.get("current_mechanism")
        or portfolio.get("primary_mechanism")
        or hypotheses[0]["name"],
        120,
    )
    names = {item["name"] for item in hypotheses}
    if current not in names and current != "unknown":
        hypotheses.insert(
            0,
            {
                "name": current,
                "confidence": 0.0,
                "differential_priority": 0.0,
                "evidence_support": "unsupported",
                "claim_status": "candidate_requires_validation",
                "status": "pending",
                "rationale": "Current hypothesis supplied by Planner LLM.",
                "evidence_refs": [],
                "validation_needed": [],
            },
        )
    return {
        "current": current if current in names or current == "unknown" else hypotheses[0]["name"],
        "runner_up": None
        if _is_nullish(
            portfolio.get("runner_up")
            or portfolio.get("runner_up_hypothesis")
            or portfolio.get("secondary_hypothesis")
            or portfolio.get("secondary_mechanism")
        )
        else _short_text(
            portfolio.get("runner_up")
            or portfolio.get("runner_up_hypothesis")
            or portfolio.get("secondary_hypothesis")
            or portfolio.get("secondary_mechanism"),
            120,
        ),
        "hypotheses": hypotheses,
    }


def _is_delta_only_portfolio_shell(portfolio: dict[str, object]) -> bool:
    if not isinstance(portfolio, dict):
        return False
    hypothesis_keys = (
        "hypotheses",
        "candidate_hypotheses",
        "hypothesis_candidates",
        "mechanism_hypotheses",
        "items",
        "entries",
    )
    for key in hypothesis_keys:
        value = portfolio.get(key)
        if isinstance(value, list) and value:
            return False
    return any(
        not _is_nullish(portfolio.get(key))
        for key in (
            "current",
            "primary_hypothesis",
            "leading_hypothesis",
            "current_mechanism",
            "primary_mechanism",
            "runner_up",
            "runner_up_hypothesis",
            "secondary_hypothesis",
            "secondary_mechanism",
        )
    )


def _normalize_priority_deltas(
    payload: dict[str, object],
    *,
    round_id: str,
    previous_hypotheses: list[dict[str, object]],
    portfolio: dict[str, object],
    allowed_evidence_ids: list[str],
) -> list[dict[str, object]]:
    previous_by_label = {
        _hypothesis_name(item): _normalize_hypothesis_payload(item)
        for item in previous_hypotheses
        if _hypothesis_name(item)
    }
    current_by_label = {
        _hypothesis_name(item): _normalize_hypothesis_payload(item)
        for item in _as_dict_list(portfolio.get("hypotheses"))
        if _hypothesis_name(item)
    }
    raw_items: list[dict[str, object]] = []
    for key in ("priority_deltas", "portfolio_updates", "mechanism_priority_deltas"):
        raw_items.extend(_as_dict_list(payload.get(key)))
    allowed_refs = set(allowed_evidence_ids)
    deltas_by_label: dict[str, dict[str, object]] = {}
    for item in raw_items[: len(MECHANISM_POOL) + 12]:
        label = _short_text(
            item.get("label")
            or item.get("name")
            or item.get("mechanism")
            or item.get("target_label"),
            120,
        )
        if not label or label.lower() == "unknown":
            continue
        previous_priority = _priority_for_payload(previous_by_label.get(label, {}))
        current_item = current_by_label.get(label, {})
        raw_new_priority = (
            item.get("new_priority")
            or item.get("updated_priority")
            or current_item.get("differential_priority")
            or current_item.get("confidence")
        )
        new_priority = _float_between(raw_new_priority, default=previous_priority)
        raw_delta = item.get("priority_delta") or item.get("delta") or item.get("priority_change")
        priority_delta = _signed_float_between(
            raw_delta,
            default=round(new_priority - previous_priority, 6),
        )
        if item.get("new_priority") is None and item.get("updated_priority") is None:
            new_priority = _float_between(
                previous_priority + priority_delta,
                default=previous_priority,
            )
        refs = [
            ref
            for ref in (
                _text_list(item.get("evidence_refs_added"), limit=8)
                or _text_list(item.get("new_evidence_refs"), limit=8)
                or _text_list(item.get("evidence_refs"), limit=8)
            )
            if ref in allowed_refs
        ]
        if not refs:
            refs = _new_refs_for_label(
                previous_by_label.get(label, {}),
                current_item,
                allowed_refs,
            )
        deltas_by_label[label] = {
            "round_id": round_id,
            "label": label,
            "previous_priority": round(previous_priority, 6),
            "priority_delta": round(new_priority - previous_priority, 6)
            if raw_delta is None
            else round(priority_delta, 6),
            "new_priority": round(new_priority, 6),
            "delta_reason": _short_text(
                item.get("delta_reason")
                or item.get("reason")
                or item.get("rationale")
                or _inferred_delta_reason(
                    label,
                    previous_priority=previous_priority,
                    new_priority=new_priority,
                ),
                700,
            ),
            "evidence_refs_added": refs,
            "support_change": _normalize_support_change(item.get("support_change")),
        }
    for label, current_item in current_by_label.items():
        if label.lower() == "unknown":
            continue
        previous_item = previous_by_label.get(label, {})
        previous_priority = _priority_for_payload(previous_item)
        new_priority = _priority_for_payload(current_item)
        refs = _new_refs_for_label(previous_item, current_item, allowed_refs)
        support_change = _inferred_support_change(previous_item, current_item)
        changed = (
            abs(new_priority - previous_priority) > 0.0005
            or refs
            or support_change != "unchanged"
        )
        if label in deltas_by_label or not changed:
            continue
        deltas_by_label[label] = {
            "round_id": round_id,
            "label": label,
            "previous_priority": round(previous_priority, 6),
            "priority_delta": round(new_priority - previous_priority, 6),
            "new_priority": round(new_priority, 6),
            "delta_reason": _inferred_delta_reason(
                label,
                previous_priority=previous_priority,
                new_priority=new_priority,
            ),
            "evidence_refs_added": refs,
            "support_change": support_change,
        }
    return [
        MechanismPriorityDelta.model_validate(item).model_dump(mode="json")
        for item in sorted(
            deltas_by_label.values(),
            key=lambda item: abs(float(item["priority_delta"])),
            reverse=True,
        )
    ]


def _sort_portfolio_by_priority(portfolio: dict[str, object]) -> dict[str, object]:
    hypotheses = [
        _normalize_hypothesis_payload(item)
        for item in _as_dict_list(portfolio.get("hypotheses"))
        if _hypothesis_name(item)
    ]
    hypotheses.sort(
        key=lambda item: (
            _priority_for_payload(item),
            _hypothesis_name(item) == str(portfolio.get("current") or ""),
        ),
        reverse=True,
    )
    if not hypotheses:
        return portfolio
    return {
        **portfolio,
        "current": _hypothesis_name(hypotheses[0]),
        "runner_up": _hypothesis_name(hypotheses[1]) if len(hypotheses) > 1 else None,
        "hypotheses": hypotheses,
    }


def _has_higher_directness_signal(directness: dict[str, object] | None) -> bool:
    if not isinstance(directness, dict):
        return False
    return (
        int(directness.get("computed_direct_proxy_count") or 0) > 0
        or int(directness.get("mechanism_specific_structural_count") or 0) > 0
        or int(directness.get("critic_positive_ref_count") or 0) > 0
    )


def _low_margin_notes_for_portfolio(
    portfolio: dict[str, object],
) -> list[dict[str, object]]:
    hypotheses = [
        _normalize_hypothesis_payload(item)
        for item in _as_dict_list(portfolio.get("hypotheses"))
        if _hypothesis_name(item)
    ]
    notes: list[dict[str, object]] = []
    for upper, lower in zip(hypotheses, hypotheses[1:], strict=False):
        upper_label = _hypothesis_name(upper)
        lower_label = _hypothesis_name(lower)
        margin = _priority_for_payload(upper) - _priority_for_payload(lower)
        if 0.0 <= margin < LOW_MARGIN_THRESHOLD:
            notes.append(
                _low_margin_note(
                    upper_label,
                    lower_label,
                    margin,
                    "final_calibration_low_margin_reported",
                )
            )
    return notes


def _top_hypothesis_name(portfolio: dict[str, object]) -> str:
    hypotheses = _as_dict_list(portfolio.get("hypotheses"))
    if not hypotheses:
        return ""
    return _hypothesis_name(hypotheses[0])


def _second_hypothesis_name(portfolio: dict[str, object]) -> str:
    hypotheses = _as_dict_list(portfolio.get("hypotheses"))
    if len(hypotheses) < 2:
        return ""
    return _hypothesis_name(hypotheses[1])


def _priority_for_payload(item: dict[str, object]) -> float:
    if "differential_priority" in item and item.get("differential_priority") is not None:
        return _float_between(item.get("differential_priority"), default=0.0)
    return _float_between(item.get("confidence"), default=0.0)


def _signed_float_between(value: object, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return max(min(default, 1.0), -1.0)
    return max(min(number, 1.0), -1.0)


def _inferred_delta_reason(
    label: str,
    *,
    previous_priority: float,
    new_priority: float,
) -> str:
    delta = new_priority - previous_priority
    if abs(delta) <= 0.0005:
        return f"{label} priority was carried forward; no material state change was inferred."
    direction = "increased" if delta > 0 else "decreased"
    return (
        f"{label} priority {direction} from {previous_priority:.3f} to "
        f"{new_priority:.3f} based on the Planner returned portfolio state."
    )


def _new_refs_for_label(
    previous_item: dict[str, object],
    current_item: dict[str, object],
    allowed_refs: set[str],
) -> list[str]:
    previous_refs = set(_text_list(previous_item.get("evidence_refs"), limit=12))
    current_refs = _text_list(current_item.get("evidence_refs"), limit=12)
    return [ref for ref in current_refs if ref in allowed_refs and ref not in previous_refs][:8]


def _normalize_support_change(value: object) -> str:
    text = str(value or "unchanged").strip().lower()
    aliases = {
        "upgrade": "upgraded",
        "downgrade": "downgraded",
        "weaken": "weakened",
        "new": "newly_supported",
        "new_support": "newly_supported",
        "same": "unchanged",
        "none": "unchanged",
    }
    normalized = aliases.get(text, text)
    allowed = {
        "unchanged",
        "upgraded",
        "downgraded",
        "weakened",
        "newly_supported",
    }
    return normalized if normalized in allowed else "unchanged"


def _inferred_support_change(
    previous_item: dict[str, object],
    current_item: dict[str, object],
) -> str:
    previous_rank = _support_strength_rank(previous_item.get("evidence_support"))
    current_rank = _support_strength_rank(current_item.get("evidence_support"))
    if previous_rank <= 0 < current_rank:
        return "newly_supported"
    if current_rank > previous_rank:
        return "upgraded"
    if current_rank < previous_rank:
        return "downgraded"
    return "unchanged"


def _support_strength_rank(value: object) -> int:
    return {
        "strong": 3,
        "partial": 2,
        "weak_or_proxy": 1,
        "weak": 1,
        "unsupported": 0,
    }.get(str(value or "unsupported").strip().lower(), 0)


def _low_margin_note(
    first_label: str,
    second_label: str,
    margin: float,
    action: str,
) -> dict[str, object]:
    return {
        "labels": [first_label, second_label],
        "margin": round(max(margin, 0.0), 6),
        "threshold": LOW_MARGIN_THRESHOLD,
        "action": action,
    }


def _normalize_hypothesis_payload(item: dict[str, object]) -> dict[str, object]:
    name = _hypothesis_name(item) or "unknown"
    confidence = _float_between(item.get("confidence"), default=0.0)
    differential_priority = _validated_planner_priority(item)
    evidence_support = _normalize_prediction_support_strength(
        item.get("evidence_support")
        or item.get("support_strength")
        or item.get("support_level")
    )
    evidence_refs = _text_list(item.get("evidence_refs"), limit=8)
    return {
        "name": name,
        "confidence": confidence,
        "differential_priority": differential_priority,
        "evidence_support": evidence_support,
        "claim_status": _normalize_claim_status(item.get("claim_status")),
        "status": _normalize_hypothesis_status(item.get("status")),
        "rationale": _short_text(
            item.get("rationale")
            or item.get("reasoning")
            or item.get("reasoning_summary"),
            360,
        ),
        "evidence_refs": evidence_refs,
        "validation_needed": _text_list(
            item.get("validation_needed")
            or item.get("needed_validation")
            or item.get("missing_validation"),
            limit=6,
        ),
    }


def _validated_planner_priority(item: dict[str, object]) -> float:
    value = next(
        (item[key] for key in ("differential_priority", "priority", "differential_score", "confidence")
         if item.get(key) is not None),
        0.0,
    )
    try:
        priority = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Planner priority must be a number between 0 and 1.") from exc
    if not 0.0 <= priority <= 1.0:
        raise ValueError("Planner priority must be between 0 and 1; scores are not clipped.")
    return priority


def _hypothesis_name(item: dict[str, object]) -> str:
    return _short_text(
        item.get("name")
        or item.get("hypothesis")
        or item.get("hypothesis_name")
        or item.get("mechanism")
        or item.get("mechanism_name")
        or item.get("label"),
        120,
    )


def _normalize_hypothesis_status(value: object) -> str:
    status = str(value or "pending").strip().lower()
    allowed = {"plausible", "pending", "screened", "blocked", "dropped", "unknown"}
    return status if status in allowed else "pending"


def _normalize_support_arguments(
    payload: dict[str, object],
    *,
    round_id: str,
    allowed_evidence_ids: list[str],
) -> list[dict[str, object]]:
    raw_items: list[dict[str, object]] = []
    for key in (
        "mechanism_support_arguments",
        "support_arguments",
        "support_synthesis",
        "mechanism_support_synthesis",
    ):
        raw_items.extend(_as_dict_list(payload.get(key)))
    allowed = set(allowed_evidence_ids)
    arguments: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_items[:12], start=1):
        target_label = _short_text(
            item.get("target_label")
            or item.get("mechanism")
            or item.get("label")
            or item.get("name"),
            120,
        )
        refs = [
            ref
            for ref in (
                _text_list(item.get("observation_refs"), limit=8)
                or _text_list(item.get("source_evidence_refs"), limit=8)
                or _text_list(item.get("evidence_refs"), limit=8)
            )
            if ref in allowed
        ]
        finding = _short_text(item.get("finding") or item.get("observation"), 700)
        warrant = _short_text(item.get("warrant") or item.get("reasoning"), 900)
        boundary = _short_text(
            item.get("boundary")
            or item.get("limitations")
            or item.get("scope_limit"),
            700,
        )
        if not target_label or not refs or not finding or not warrant or not boundary:
            continue
        support_id = _short_text(
            item.get("support_id") or f"{round_id}:support:{index:03d}",
            180,
        )
        if not support_id.startswith(f"{round_id}:"):
            support_id = f"{round_id}:{support_id}"
        if support_id in seen_ids:
            support_id = f"{round_id}:support:{index:03d}"
        seen_ids.add(support_id)
        arguments.append(
            {
                "support_id": support_id,
                "target_label": target_label,
                "support_level": _normalize_support_level(item.get("support_level")),
                "finding": finding,
                "warrant": warrant,
                "boundary": boundary,
                "observation_refs": refs,
                "audit_status": "not_reviewed",
                "audit_notes": [],
            }
        )
    return arguments


def _normalize_support_level(value: object) -> str:
    text = str(value or "partial").strip().lower()
    aliases = {
        "proxy": "partial",
        "proxy_supported": "partial",
        "computed_supported": "strong",
        "plausible": "weak",
        "underdetermined": "weak",
        "none": "unsupported",
        "not_supported": "unsupported",
    }
    normalized = aliases.get(text, text)
    allowed = {"strong", "partial", "weak", "unsupported"}
    return normalized if normalized in allowed else "partial"


def _normalize_mechanism_label(value: object) -> str:
    text = str(value or "").strip()
    if text in MECHANISM_POOL:
        return text
    token = (
        text.upper()
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )
    aliases = {
        "RIM": "RIM_RIR_RIV",
        "RIR": "RIM_RIR_RIV",
        "RIV": "RIM_RIR_RIV",
        "RESTRICTION_OF_INTRAMOLECULAR_MOTION": "RIM_RIR_RIV",
        "PACKING": "PACKING_HOST_MATRIX_CONFINEMENT",
        "PACKING_CONFINEMENT": "PACKING_HOST_MATRIX_CONFINEMENT",
        "HOST_MATRIX_CONFINEMENT": "PACKING_HOST_MATRIX_CONFINEMENT",
        "HOST_GUEST": "HOST_GUEST_INTERACTION",
        "HOST_GUEST_BINDING": "HOST_GUEST_INTERACTION",
        "ICT": "ICT_TICT_CT",
        "TICT": "ICT_TICT_CT",
        "CT": "ICT_TICT_CT",
        "CHARGE_TRANSFER": "ICT_TICT_CT",
        "ESIPT": "ESIPT_PT",
        "PT": "ESIPT_PT",
        "PROTON_TRANSFER": "ESIPT_PT",
        "PET": "PET_ET",
        "ET": "PET_ET",
        "AGGREGATE": "AGGREGATE_EXCITON_EXCIMER",
        "EXCIMER": "AGGREGATE_EXCITON_EXCIMER",
        "EXCITON": "AGGREGATE_EXCITON_EXCIMER",
        "RADIATIVE_RATE": "RADIATIVE_RATE_STATE_BALANCE",
        "STATE_BALANCE": "RADIATIVE_RATE_STATE_BALANCE",
        "BRIGHT_DARK_STATE_BALANCE": "RADIATIVE_RATE_STATE_BALANCE",
        "OSCILLATOR_STRENGTH": "RADIATIVE_RATE_STATE_BALANCE",
        "TRIPLET": "TRIPLET_METAL_ENERGY_TRANSFER",
        "METAL": "TRIPLET_METAL_ENERGY_TRANSFER",
        "METAL_ENERGY_TRANSFER": "TRIPLET_METAL_ENERGY_TRANSFER",
        "RACI": "RACI_CI_ACCESS",
        "CI": "RACI_CI_ACCESS",
        "CI_ACCESS": "RACI_CI_ACCESS",
        "CONICAL_INTERSECTION_ACCESS": "RACI_CI_ACCESS",
        "SOKR": "SOKR_ANTI_KASHA",
        "ANTI_KASHA": "SOKR_ANTI_KASHA",
    }
    return aliases.get(token, text)


def _normalize_prediction_support_strength(value: object) -> str:
    text = str(value or "unsupported").strip().lower()
    aliases = {
        "weak": "weak_or_proxy",
        "proxy": "weak_or_proxy",
        "proxy_supported": "weak_or_proxy",
        "computed_supported": "strong",
        "not_supported": "unsupported",
        "none": "unsupported",
    }
    normalized = aliases.get(text, text)
    allowed = {"strong", "partial", "weak_or_proxy", "unsupported"}
    return normalized if normalized in allowed else "unsupported"


def _normalize_claim_status(value: object) -> str:
    text = str(value or "candidate_requires_validation").strip().lower()
    aliases = {
        "supported": "supported_claim",
        "strong": "supported_claim",
        "partial": "partially_supported_candidate",
        "proxy_supported": "partially_supported_candidate",
        "weak": "candidate_requires_validation",
        "weak_or_proxy": "candidate_requires_validation",
        "weak_proxy_candidate": "candidate_requires_validation",
        "needs_validation": "candidate_requires_validation",
    }
    normalized = aliases.get(text, text)
    allowed = {
        "supported_claim",
        "partially_supported_candidate",
        "candidate_requires_validation",
        "underdetermined",
    }
    return normalized if normalized in allowed else "candidate_requires_validation"


def _portfolio_is_placeholder(portfolio: dict[str, object]) -> bool:
    hypotheses = _as_dict_list(portfolio.get("hypotheses"))
    if not hypotheses:
        return True
    concrete_names = [
        _short_text(
            item.get("name")
            or item.get("hypothesis")
            or item.get("mechanism")
            or item.get("label"),
            120,
        )
        for item in hypotheses
    ]
    return not any(name and name.lower() != "unknown" for name in concrete_names)


def _is_initial_evidence_routing(
    decision: PlannerDecision,
    case_run: CaseRun,
) -> bool:
    return (
        decision.round_id == "R001"
        and decision.action == "dispatch"
        and bool(decision.dispatch_requests)
        and not case_run.evidence_ledger.items
    )


def _has_hypothesis_context(case_run: CaseRun) -> bool:
    if case_run.mechanism_agenda_coverage is not None:
        return True
    if case_run.mechanism_program is not None and case_run.mechanism_program.candidate_mechanisms:
        return True
    if case_run.photophysics_review is not None and case_run.photophysics_review.hypothesis_cards:
        return True
    return bool(case_run.evidence_ledger.items)




def _normalize_dispatch_payload(
    item: dict[str, object],
    *,
    round_id: str,
) -> dict[str, object]:
    capability_id = _short_text(item.get("capability_id"), 160)
    agent_name = _short_text(item.get("agent_name"), 40)
    route = _short_text(item.get("route"), 120)
    return {
        "dispatch_id": _short_text(
            item.get("dispatch_id")
            or f"{round_id}:dispatch:{agent_name or 'macro'}:{route or 'route'}",
            220,
        ),
        "round_id": _short_text(item.get("round_id") or round_id, 80),
        "agent_name": agent_name,
        "capability_id": capability_id,
        "task": _short_text(item.get("task"), 500) or "Collect bounded evidence.",
        "objective": _short_text(item.get("objective"), 500)
        or _short_text(item.get("task"), 500)
        or "Collect bounded evidence.",
        "evidence_goal_family": _short_text(item.get("evidence_goal_family"), 80),
        "route": route,
        "input_refs": _text_list(item.get("input_refs"), limit=8),
        "constraints": item.get("constraints") if isinstance(item.get("constraints"), dict) else {},
        "tool_args": item.get("tool_args", {}),
        "required_observables": item.get("required_observables", []),
    }


def _selected_capability_ids(payload: dict[str, object]) -> list[str]:
    selected: list[str] = []
    for key in (
        "selected_capability_ids",
        "dispatch_capability_ids",
        "next_capability_ids",
    ):
        values = payload.get(key, [])
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(f"{key} must be a list of capability IDs.")
        selected.extend(value.strip() for value in values)
    selected.extend(
        _short_text(item.get("capability_id"), 160)
        for item in _as_dict_list(payload.get("selected_capabilities"))
    )
    return [item for item in dict.fromkeys(selected) if item]






def _available_capability_ids_from_payload(payload_context: dict[str, object]) -> set[str]:
    if "artifact_available_capability_ids" in payload_context:
        return set(payload_context["artifact_available_capability_ids"])
    ids: set[str] = set()
    for key in ("capability_cards", "ability_evidence_guide"):
        for item in _as_dict_list(payload_context.get(key)):
            capability_id = _short_text(item.get("capability_id"), 160)
            if capability_id:
                ids.add(capability_id)
    return ids








def _dispatches_from_capability_ids(
    capability_ids: list[str],
    *,
    round_id: str,
    capability_registry: CapabilityRegistry,
) -> list[dict[str, object]]:
    dispatches: list[dict[str, object]] = []
    for capability_id in capability_ids:
        capability = capability_registry.get(capability_id)
        if capability is None:
            raise ValueError(f"Unknown capability_id: {capability_id}")
        dispatches.append(
            {
                "dispatch_id": (
                    f"{round_id}:dispatch:{capability.owner_agent}:{capability.route}"
                ),
                "round_id": round_id,
                "agent_name": capability.owner_agent,
                "capability_id": capability.capability_id,
                "task": (
                    "Collect bounded evidence using capability "
                    f"{capability.capability_id}: {capability.description}"
                ),
                "objective": capability.description,
                "evidence_goal_family": capability.evidence_family,
                "route": capability.route,
                "input_refs": [],
                "constraints": {},
            }
        )
    return dispatches








def _as_dict_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _text_list(value: object, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        value = [] if value is None else [value]
    return [_short_text(item, 240) for item in value[:limit] if _short_text(item, 240)]


def _first_present_text(payload: dict[str, object], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = _short_text(payload.get(key), 120)
        if value and value.lower() not in {"none", "null"}:
            return value
    return ""


def _float_between(value: object, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(number, 0.0), 1.0)


def _is_nullish(value: object) -> bool:
    return value is None or str(value).strip().lower() in {"", "none", "null"}


def _short_text(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _append_diagnosis_summary(answer: str, diagnoses: list[DiagnosisUnit]) -> str:
    if not diagnoses:
        return answer
    lines = [answer.rstrip(), "", "MECHANISM_DIAGNOSIS:"]
    for diagnosis in diagnoses:
        missing = "; ".join(diagnosis.missing_or_unresolved[:2])
        refs = ", ".join(diagnosis.evidence_refs[:4]) or "none"
        lines.append(
            "- "
            f"{diagnosis.mechanism} [{diagnosis.status}; refs={refs}]: "
            f"{diagnosis.reasoning_summary}"
            + (f" Missing/unresolved: {missing}." if missing else "")
        )
    return "\n".join(lines)
