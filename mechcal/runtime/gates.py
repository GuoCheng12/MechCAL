from __future__ import annotations

from dataclasses import dataclass

from mechcal.capabilities import CapabilityRegistry, default_capability_registry
from mechcal.schemas import CaseRun, DecisionGateResult, DispatchRequest, PlannerDecision

GENERIC_INITIAL_CAPABILITY_IDS = {
    "macro.screen_aggregation_prone_scaffold",
    "macro.screen_donor_acceptor_architecture",
    "macro.screen_donor_acceptor_layout",
    "macro.screen_esipt_structural_motif",
    "macro.screen_metal_triplet_prior",
    "macro.screen_polar_binding_site_prior",
    "macro.screen_rotor_torsion_topology",
    "microscopic.run_baseline_bundle",
}

DISCRIMINATING_FOLLOWUP_CAPABILITY_IDS = {
    "macro.run_aggregate_contact_proxy",
    "macro.run_crystal_restriction_checklist",
    "macro.run_dimer_packing_proxy",
    "macro.run_solid_state_emission_proxy",
    "microscopic.extract_ct_descriptors_from_bundle",
    "microscopic.run_bright_dark_state_ordering",
    "microscopic.run_charge_population_panel",
    "microscopic.run_frontier_orbital_partition",
    "microscopic.run_solvation_polarity_proxy",
    "microscopic.run_torsion_brightness_coupling_scan",
}


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
        round_index = int(decision.round_id.removeprefix("R"))
        if decision.action == "finalize":
            violations = self._finalize_violations(decision, case_run)
            if violations:
                return self._policy_retry(decision.round_id, violations)
            return DecisionGateResult(
                gate_id=f"{decision.round_id}:gate:finalize",
                round_id=decision.round_id,
                status="finalize",
                terminal=True,
                terminal_reason="planner_finalize_allowed",
            )

        if round_index >= self.policy.max_rounds:
            return DecisionGateResult(
                gate_id=f"{decision.round_id}:gate:max_rounds",
                round_id=decision.round_id,
                status="max_rounds_stop",
                terminal=True,
                terminal_reason="max_rounds_reached",
                violations=["planner_requested_more_work_after_round_budget"],
            )

        if decision.action == "stop":
            return DecisionGateResult(
                gate_id=f"{decision.round_id}:gate:stop",
                round_id=decision.round_id,
                status="stop",
                terminal=True,
                terminal_reason="planner_stop",
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
        return [
            violation
            for request in dispatch_requests
            for violation in self.capability_registry.validate_dispatch(
                request,
                case_run.artifact_manifest,
            )
        ]

    def _finalize_violations(
        self,
        decision: PlannerDecision,
        case_run: CaseRun,
    ) -> list[str]:
        evidence = case_run.evidence_ledger.items
        executed_capabilities = {item.capability_id for item in evidence if item.capability_id}
        available_owners = {
            str(item["owner_agent"])
            for item in self.capability_registry.cards()
        }
        checks = {
            "Planner finalize requires a non-empty final_answer_draft.": bool(
                (decision.final_answer_draft or "").strip()
            ),
            "Planner finalize requires a Macro structural_prior evidence attempt.": any(
                item.agent_name == "macro"
                and item.family == "geometry_precondition"
                and "structural_prior" in item.observable_tags
                for item in evidence
            )
            or "macro" not in available_owners,
            "Planner finalize requires a Microscopic baseline route attempt.": any(
                item.agent_name == "microscopic"
                and "run_baseline_bundle" in item.observable_tags
                for item in evidence
            )
            or "microscopic" not in available_owners,
            (
                "Planner finalize requires at least one discriminating follow-up "
                "route beyond generic initial priors."
            ): bool(executed_capabilities & DISCRIMINATING_FOLLOWUP_CAPABILITY_IDS),
        }
        violations = [message for message, passed in checks.items() if not passed]
        if _top3_are_generic_prior_only(case_run, executed_capabilities):
            violations.append(
                "Planner finalize blocked because Top-3 are still supported only "
                "by generic initial-prior routes; dispatch a disambiguating route "
                "or lower unsupported generic candidates before finalizing."
            )
        return violations

    def _policy_retry(self, round_id: str, violations: list[str]) -> DecisionGateResult:
        return DecisionGateResult(
            gate_id=f"{round_id}:gate:capability_policy",
            round_id=round_id,
            status="retry_policy",
            terminal=True,
            terminal_reason="capability_policy_violation",
            violations=violations,
        )


def _top3_are_generic_prior_only(
    case_run: CaseRun,
    executed_capabilities: set[str],
) -> bool:
    if executed_capabilities & DISCRIMINATING_FOLLOWUP_CAPABILITY_IDS:
        return False
    top3 = [
        item
        for item in case_run.portfolio.sorted_hypotheses()[:3]
        if item.name and item.name != "unknown"
    ]
    if len(top3) < 3:
        return False
    if not all(item.evidence_refs for item in top3[:2]):
        return True
    refs_by_id = {item.evidence_id: item for item in case_run.evidence_ledger.items}
    for hypothesis in top3:
        for ref in hypothesis.evidence_refs:
            evidence = refs_by_id.get(ref)
            if evidence and evidence.capability_id not in GENERIC_INITIAL_CAPABILITY_IDS:
                return False
    return True
