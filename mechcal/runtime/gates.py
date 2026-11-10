from __future__ import annotations

from dataclasses import dataclass

from mechcal.capabilities import CapabilityRegistry, default_capability_registry
from mechcal.capabilities.discovery import LoadedToolState, discovery_dispatch_violations
from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.schemas import CaseRun, DecisionGateResult, DispatchRequest, PlannerDecision


@dataclass(frozen=True)
class GatePolicy:
    max_rounds: int = 4


class DeterministicGate:
    def __init__(
        self,
        policy: GatePolicy | None = None,
        capability_registry: CapabilityRegistry | None = None,
    ) -> None:
        self.policy = policy or GatePolicy()
        self.capability_registry = capability_registry or default_capability_registry()

    def apply(self, decision: PlannerDecision, case_run: CaseRun) -> DecisionGateResult:
        if decision.failure.is_failure:
            return DecisionGateResult(
                gate_id=f"{decision.round_id}:gate:invalid_decision",
                round_id=decision.round_id,
                status="retry_schema",
                terminal=True,
                terminal_reason="planner_decision_invalid",
                violations=[decision.failure.message],
            )
        if case_run.runtime.get("question_loop"):
            from mechcal.runtime.questions import validate_decision

            try:
                validate_decision(decision, case_run, self.capability_registry)
            except ValueError as exc:
                return self._policy_retry(decision.round_id, [str(exc)])
        round_index = int(decision.round_id.removeprefix("R"))
        limits = case_run.runtime.get("run_limits", {})
        new_stopping = limits.get("stopping_protocol") == "planner_stop_v1"
        at_limit = ((limits.get("max_exploration_rounds") is not None
                     and round_index > limits["max_exploration_rounds"])
                    or limits.get("scientific_tools_allowed") is False) if new_stopping else False
        if new_stopping:
            if decision.stopping_assessment is None:
                return self._policy_retry(decision.round_id, ["Missing stopping_assessment."])
            if decision.action == "finalize" and not at_limit and decision.stopping_assessment.finish_basis is None:
                return self._policy_retry(decision.round_id, ["Voluntary finalization needs a finish_basis."])
        if decision.action == "finalize":
            violations = self._finalize_violations(decision, case_run)
            if violations:
                return self._policy_retry(decision.round_id, violations)
            return DecisionGateResult(
                gate_id=f"{decision.round_id}:gate:finalize",
                round_id=decision.round_id,
                status="finalize",
                terminal=True,
                terminal_reason=("exploration_limit_reached" if at_limit else
                                 decision.stopping_assessment.finish_basis) if new_stopping
                else "planner_finalize_allowed",
            )

        if decision.action == "stop":
            return DecisionGateResult(
                gate_id=f"{decision.round_id}:gate:stop",
                round_id=decision.round_id,
                status="stop",
                terminal=True,
                terminal_reason="exploration_limit_reached" if at_limit else "planner_stop",
            )

        if at_limit or (not new_stopping and round_index >= self.policy.max_rounds):
            return DecisionGateResult(
                gate_id=f"{decision.round_id}:gate:max_rounds",
                round_id=decision.round_id,
                status="max_rounds_stop",
                terminal=True,
                terminal_reason="exploration_limit_reached" if new_stopping else "max_rounds_reached",
                violations=["planner_requested_more_work_after_round_budget"],
            )

        violations = self._dispatch_violations(decision.dispatch_requests, case_run)
        if violations:
            return self._policy_retry(decision.round_id, violations)

        return DecisionGateResult(
            gate_id=f"{decision.round_id}:gate:allow",
            round_id=decision.round_id,
            status="allow",
            dispatch_requests=list(decision.dispatch_requests),
        )

    def _dispatch_violations(
        self,
        dispatch_requests: list[DispatchRequest],
        case_run: CaseRun,
    ) -> list[str]:
        violations = [
            violation
            for request in dispatch_requests
            for violation in self.capability_registry.validate_dispatch(
                request,
                case_run.artifact_manifest,
            )
        ]
        if case_run.runtime.get("planner_tool_loading") == "on_demand":
            state = LoadedToolState.model_validate(case_run.runtime.get(
                "tool_discovery", {"case_id": case_run.case_id},
            ))
            violations.extend(discovery_dispatch_violations(
                self.capability_registry, case_run, state, dispatch_requests,
            ))
        return violations

    def _finalize_violations(
        self,
        decision: PlannerDecision,
        case_run: CaseRun,
    ) -> list[str]:
        labels = [item.name for item in decision.portfolio.hypotheses]
        evidence_ids = {item.evidence_id for item in case_run.evidence_ledger.items} | {"input:smiles"}
        checks = {
            "Planner finalize requires a non-empty final_answer_draft.": bool(
                (decision.final_answer_draft or "").strip()
            ),
            "Planner finalize requires at least three distinct mechanism_pool labels.":
                len(labels) >= 3 and len(labels) == len(set(labels)) and set(labels).issubset(MECHANISM_POOL),
            "Planner evidence references must exist.": all(
                ref in evidence_ids
                for item in decision.portfolio.hypotheses
                for ref in item.evidence_refs
            ) and all(
                ref in evidence_ids
                for argument in decision.mechanism_support_arguments
                for ref in argument.observation_refs
            ),
        }
        return [message for message, passed in checks.items() if not passed]

    def _policy_retry(self, round_id: str, violations: list[str]) -> DecisionGateResult:
        return DecisionGateResult(
            gate_id=f"{round_id}:gate:capability_policy",
            round_id=round_id,
            status="retry_policy",
            terminal=True,
            terminal_reason="capability_policy_violation",
            violations=violations,
        )
