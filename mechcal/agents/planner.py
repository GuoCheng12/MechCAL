from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from mechcal.capabilities import CapabilityRegistry, default_capability_registry
from mechcal.capabilities.evidence_guide import (
    compact_ability_evidence_guide,
)
from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.public_ids import public_case_id_from_metadata
from mechcal.runtime.agent_runtime import BaseAgentRuntime
from mechcal.runtime.context import PlannerContextView
from mechcal.runtime.llm import OpenAICompatibleSettings, OpenAIJsonClient
from mechcal.runtime.mechanism_program import build_diagnosis_units
from mechcal.runtime.prompts import load_prompt
from mechcal.runtime.retry import SchemaRetryError
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

LOW_MARGIN_THRESHOLD = 0.05
LOW_MARGIN_STABLE_SEPARATION = 0.001
UNSUPPORTED_PRIOR_PRIORITY_CEILING = 0.2
NO_REF_WEAK_PRIORITY_CEILING = 0.12
MAX_DISPATCHES_PER_ROUND = 6
COMPUTED_DIRECT_PRIORITY_FLOOR = 0.24
MECHANISM_SPECIFIC_STRUCTURAL_PRIORITY_FLOOR = 0.23
ARTIFACT_BACKED_CT_PRIORITY_FLOOR = 0.32
WEAK_ARTIFACT_BACKED_CT_PRIORITY_FLOOR = 0.24
GENERIC_STRUCTURAL_PRIORITY_CEILING = 0.19
GENERIC_PACKING_PROXY_PRIORITY_CEILING = 0.22
WEAK_CT_PROXY_PRIORITY_CEILING = 0.20
STANDALONE_OSCILLATOR_PRIORITY_CEILING = 0.18
MODERATE_STANDALONE_OSCILLATOR_PRIORITY = 0.24
STRONG_STANDALONE_OSCILLATOR_PRIORITY = 0.29
COUPLED_OSCILLATOR_PRIORITY_FLOOR = 0.24
PUBLIC_ABSORPTION_COMPUTED_PRIORITY = 0.30
PUBLIC_ABSORPTION_MECHANISM_SPECIFIC_PRIORITY = 0.31
PUBLIC_ABSORPTION_ELECTRONIC_PRIORITY = 0.24
PUBLIC_ABSORPTION_STRUCTURAL_PRIORITY = 0.21
PUBLIC_ABSORPTION_MECHANISM_SPECIFIC_CEILING_WITH_COMPUTED = 0.31


class PlannerAgent:
    """Schema-first Planner that schedules bounded evidence collection."""

    def __init__(
        self,
        runtime: BaseAgentRuntime[PlannerDecision] | None = None,
        capability_registry: CapabilityRegistry | None = None,
        llm_client: OpenAIJsonClient | None = None,
        enable_llm_decisions: bool = False,
    ) -> None:
        self.capability_registry = capability_registry or default_capability_registry()
        self.llm_client = llm_client
        self.enable_llm_decisions = enable_llm_decisions
        self.runtime = runtime or BaseAgentRuntime(
            agent_name="planner",
            response_model=PlannerDecision,
        )

    def plan_initial(self, case_run: CaseRun, *, round_id: str) -> PlannerDecision:
        return self._validated_decision(
            lambda feedback: self._build_initial_decision(case_run, round_id, feedback),
            case_run=case_run,
            round_id=round_id,
            stage="planner_initial",
        )

    def _build_initial_decision(
        self,
        case_run: CaseRun,
        round_id: str,
        schema_feedback: str | None,
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
        )

    def plan_update(
        self,
        context: PlannerContextView,
        case_run: CaseRun,
        *,
        round_id: str,
    ) -> PlannerDecision:
        return self._validated_decision(
            lambda feedback: self._build_update_decision(
                context,
                case_run,
                round_id,
                feedback,
            ),
            case_run=case_run,
            round_id=round_id,
            stage="planner_update",
        )

    def _build_update_decision(
        self,
        context: PlannerContextView,
        case_run: CaseRun,
        round_id: str,
        schema_feedback: str | None,
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
        )

    def calibrate_final_portfolio(
        self,
        case_run: CaseRun,
        *,
        round_id: str,
    ) -> PlannerDecision:
        return self._validated_decision(
            lambda feedback: self._build_final_calibration_decision(
                case_run,
                round_id,
                feedback,
            ),
            case_run=case_run,
            round_id=round_id,
            stage="planner_final_calibration",
        )

    def _build_final_calibration_decision(
        self,
        case_run: CaseRun,
        round_id: str,
        schema_feedback: str | None,
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

    def _final_limitations(self, case_run: CaseRun) -> list[str]:
        limitations = [
            (
                "Planner synthesis is based only on public runtime evidence collected "
                "during this run."
            ),
        ]
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
        producer: Callable[[str | None], Any],
        *,
        case_run: CaseRun,
        round_id: str,
        stage: str,
    ) -> PlannerDecision:
        try:
            decision = self.runtime.run_schema_retry(
                lambda feedback: self._policy_checked_decision(
                    producer(feedback),
                    case_run=case_run,
                )
            )
        except SchemaRetryError as exc:
            failure = self.runtime.schema_failure(exc, kind="llm_schema_invalid")
            decision = self._stop_decision(case_run, round_id, stage, failure)
        return decision

    def _policy_checked_decision(
        self,
        payload: Any,
        *,
        case_run: CaseRun,
    ) -> PlannerDecision:
        decision = PlannerDecision.model_validate(payload)
        self._validate_planner_policy(decision, case_run)
        return decision

    def _validate_planner_policy(
        self,
        decision: PlannerDecision,
        case_run: CaseRun,
    ) -> None:
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
        allowed_evidence_ids = {item.evidence_id for item in case_run.evidence_ledger.items}
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
        if _ignored_high_urgency_under_screened_coverage(
            decision,
            case_run,
            capability_registry=self.capability_registry,
        ):
            rationale_text = " ".join(
                [
                    decision.rationale,
                    decision.diagnosis,
                    " ".join(decision.unresolved_gaps),
                ]
            ).lower()
            if not any(term in rationale_text for term in ("coverage", "agenda", "under-screen")):
                raise ValueError(
                    "PlannerDecision ignored high-urgency under-screened agenda "
                    "coverage without an explicit reason."
                )

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
    ) -> PlannerDecision:
        assert self.llm_client is not None
        payload = _compact_planner_payload(
            case_run=case_run,
            round_id=round_id,
            stage=stage,
            context=context or {},
            capability_registry=self.capability_registry,
        )
        initial_coverage_decision = _initial_coverage_dispatch_decision(
            case_run=case_run,
            round_id=round_id,
            stage=stage,
            payload=payload,
            capability_registry=self.capability_registry,
        )
        if initial_coverage_decision is not None:
            return initial_coverage_decision
        prompt_name = "planner_decision_compact.md"
        payload_shape = "compact_incremental_planner_payload_v3"
        if _should_use_initial_evidence_ranking_payload(
            case_run=case_run,
            stage=stage,
            new_round_evidence=payload.get("new_round_evidence"),
        ):
            payload = _initial_evidence_ranking_payload(
                case_run=case_run,
                round_id=round_id,
                context=context or {},
                capability_registry=self.capability_registry,
            )
            prompt_name = "planner_initial_evidence_ranking.md"
            payload_shape = "initial_evidence_ranking_payload_v1"
        use_update_reducer = (
            payload_shape == "compact_incremental_planner_payload_v3"
            and _use_public_update_evidence_reducer_as_primary()
            and _has_positive_public_update_signal(payload)
        )
        if use_update_reducer:
            response = _public_update_evidence_response(
                case_run=case_run,
                round_id=round_id,
                payload=payload,
            )
        elif (
            payload_shape
            in {"initial_verdict_payload_v1", "initial_evidence_ranking_payload_v1"}
            and _use_public_initial_evidence_reducer()
        ):
            response = _public_initial_verdict_response(payload)
        else:
            client = self.llm_client
            if payload_shape in {
                "initial_verdict_payload_v1",
                "initial_evidence_ranking_payload_v1",
            }:
                client = _initial_verdict_llm_client(self.llm_client)
            response = client.complete_json(
                system_prompt=load_prompt(prompt_name),
                payload=payload,
                schema_feedback=schema_feedback,
            )
        if payload_shape in {
            "initial_verdict_payload_v1",
            "initial_evidence_ranking_payload_v1",
        }:
            response = _planner_decision_from_initial_verdict(
                response,
                case_run=case_run,
                round_id=round_id,
                payload=payload,
            )
        try:
            response = _normalize_planner_llm_response(
                response,
                round_id=round_id,
                existing_portfolio=case_run.portfolio.model_dump(mode="json"),
                previous_portfolio_hypotheses=[
                    item.model_dump(mode="json")
                    for item in _full_portfolio_hypotheses(case_run)
                ],
                allowed_evidence_ids=[
                    item.evidence_id for item in case_run.evidence_ledger.items
                ],
                capability_registry=self.capability_registry,
                stage=stage,
                prompt_payload=payload,
            )
        except ValueError:
            if (
                payload_shape != "compact_incremental_planner_payload_v3"
                or not _has_positive_public_update_signal(payload)
            ):
                raise
            response = _normalize_planner_llm_response(
                _public_update_evidence_response(
                    case_run=case_run,
                    round_id=round_id,
                    payload=payload,
                ),
                round_id=round_id,
                existing_portfolio=case_run.portfolio.model_dump(mode="json"),
                previous_portfolio_hypotheses=[
                    item.model_dump(mode="json")
                    for item in _full_portfolio_hypotheses(case_run)
                ],
                allowed_evidence_ids=[
                    item.evidence_id for item in case_run.evidence_ledger.items
                ],
                capability_registry=self.capability_registry,
                stage=stage,
                prompt_payload=payload,
            )
        if not isinstance(response.get("raw_response"), dict):
            response["raw_response"] = {}
        response["raw_response"]["prompt_name"] = prompt_name
        response["raw_response"]["payload_shape"] = payload_shape
        return PlannerDecision.model_validate(response)

    def _raw_response(self, prompt_name: str, schema_feedback: str | None) -> dict[str, object]:
        raw_response: dict[str, object] = {
            "prompt_name": prompt_name,
            "capability_cards": self.capability_registry.cards(),
        }
        if schema_feedback is not None:
            raw_response["schema_feedback"] = schema_feedback
        return raw_response


def _initial_verdict_llm_client(client: OpenAIJsonClient) -> OpenAIJsonClient:
    if not hasattr(client, "settings"):
        return client
    settings = client.settings
    verdict_settings = OpenAICompatibleSettings(
        base_url=settings.base_url,
        model=settings.model,
        api_key=settings.api_key,
        reasoning_effort=settings.reasoning_effort,
        timeout_seconds=max(settings.timeout_seconds, 180.0),
        max_retries=settings.max_retries,
        max_tokens=1600 if settings.max_tokens is None else min(settings.max_tokens, 1600),
        trust_env=settings.trust_env,
        temperature=settings.temperature,
        top_p=settings.top_p,
        seed=settings.seed,
    )
    return OpenAIJsonClient(settings=verdict_settings)


def _use_public_initial_evidence_reducer() -> bool:
    mode = os.environ.get("MECHCAL_INITIAL_VERDICT_MODE", "reducer").strip().lower()
    return mode not in {"llm", "agent", "llm_agent"}


def _use_public_update_evidence_reducer_as_primary() -> bool:
    mode = os.environ.get("MECHCAL_UPDATE_REDUCER_MODE", "fallback").strip().lower()
    return mode in {"reducer", "primary", "on", "true", "1"}


def _compact_planner_payload(
    *,
    case_run: CaseRun,
    round_id: str,
    stage: str,
    context: dict[str, object],
    capability_registry: CapabilityRegistry,
) -> dict[str, object]:
    portfolio_hypotheses = _full_portfolio_hypotheses(case_run)
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
            for item in case_run.mechanism_support_arguments[-6:]
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
        "artifact_manifest": [] if compact_update else [
            {
                "artifact_id": artifact.artifact_id,
                "kind": artifact.kind,
                "status": artifact.status,
                "created_by_agent": artifact.created_by_agent,
                "reusable_for": artifact.reusable_for,
                "capability_name": artifact.metadata.get("capability_name"),
            }
            for artifact in case_run.artifact_manifest.artifacts[-20:]
        ],
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
        "output_contract": _planner_output_contract(),
        "policy": {
            "private_answer_key_access": "none",
            "external_answer_access": "none",
            "case_specific_literature_search": "forbidden",
            "workers_collect_evidence_only": True,
            "planner_is_only_final_scientific_reasoner": True,
            "planner_is_only_mechanism_ranking_owner": True,
            "final_mechanism_predictions_from_portfolio": True,
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


def _public_initial_verdict_response(payload: dict[str, object]) -> dict[str, object]:
    candidates: dict[str, dict[str, object]] = {}
    for item in _as_dict_list(payload.get("evidence")):
        evidence_id = _payload_evidence_id(item)
        if not evidence_id:
            continue
        for raw_label in item.get("screens", []) if isinstance(item.get("screens"), list) else []:
            label = _normalize_mechanism_label(raw_label)
            if label not in MECHANISM_POOL:
                continue
            priority = _evidence_item_priority_for_label(label, item)
            if priority <= 0.0:
                continue
            existing = candidates.get(label)
            if existing is None:
                candidates[label] = {
                    "label": label,
                    "priority": priority,
                    "support": "weak_or_proxy",
                    "status": "candidate_requires_validation",
                    "evidence_refs": [evidence_id],
                    "reason": _public_initial_reason(label, item),
                }
                continue
            existing["priority"] = max(float(existing["priority"]), priority)
            refs = _text_list(existing.get("evidence_refs"), limit=6)
            refs.append(evidence_id)
            existing["evidence_refs"] = list(dict.fromkeys(refs))[:4]
    candidate_items = list(candidates.values())
    if not candidate_items:
        for label in MECHANISM_POOL[:3]:
            candidate_items.append(
                {
                    "label": label,
                    "priority": 0.05,
                    "support": "unsupported",
                    "status": "underdetermined",
                    "evidence_refs": [],
                    "reason": "No positive public runtime evidence was available yet.",
                }
            )
    candidate_items.sort(key=lambda item: float(item["priority"]), reverse=True)
    next_ids = _evidence_gap_followup_capability_ids(
        {
            "ranked_candidates": candidate_items[:6],
            "coverage_under_screened": payload.get("coverage_under_screened"),
            "agenda": payload.get("agenda"),
        },
        payload,
    )
    return {
        "ranked_candidates": candidate_items[:6],
        "action": "dispatch" if next_ids else "finalize",
        "next_capability_ids": next_ids,
        "raw_response": {"source": "public_initial_evidence_reducer"},
    }


def _has_positive_public_update_signal(payload: dict[str, object]) -> bool:
    for item in _as_dict_list(payload.get("new_round_evidence")):
        if _payload_evidence_signal_type(item) in {
            "missing_or_failed",
            "checklist_boundary_only",
        }:
            continue
        if _payload_evidence_tier(item) not in {
            "computed_direct_proxy",
            "mechanism_specific_structural_trigger",
            "structural_trigger",
            "electronic_weak_trigger",
            "artifact_weak_trigger",
        }:
            continue
        if _payload_item_screen_labels(item):
            return True
    return False


def _public_update_evidence_response(
    *,
    case_run: CaseRun,
    round_id: str,
    payload: dict[str, object],
) -> dict[str, object]:
    previous_hypotheses = [
        item.model_dump(mode="json") for item in _full_portfolio_hypotheses(case_run)
    ]
    previous_by_label = {
        _hypothesis_name(item): _normalize_hypothesis_payload(item)
        for item in previous_hypotheses
        if _hypothesis_name(item)
    }
    candidates = {
        label: dict(item)
        for label, item in previous_by_label.items()
        if label in MECHANISM_POOL
    }
    evidence_refs_by_label: dict[str, list[str]] = {label: [] for label in MECHANISM_POOL}
    reasons_by_label: dict[str, list[str]] = {label: [] for label in MECHANISM_POOL}
    support_by_label: dict[str, str] = {}
    for item in _as_dict_list(payload.get("new_round_evidence")):
        evidence_id = _payload_evidence_id(item)
        if not evidence_id:
            continue
        if _payload_evidence_signal_type(item) in {
            "missing_or_failed",
            "checklist_boundary_only",
        }:
            continue
        for label in _payload_item_screen_labels(item):
            priority = _public_update_priority_for_label(label, item, payload=payload)
            if priority <= 0.0:
                continue
            previous = candidates.get(label) or {
                "name": label,
                "confidence": 0.0,
                "differential_priority": 0.0,
                "evidence_support": "unsupported",
                "claim_status": "underdetermined",
                "status": "pending",
                "rationale": "Not yet prioritized by Planner.",
                "evidence_refs": [],
                "validation_needed": [],
            }
            previous_priority = _priority_for_payload(previous)
            new_priority = max(previous_priority, priority)
            evidence_refs = _merged_text_list(
                previous.get("evidence_refs"),
                [evidence_id],
                limit=8,
            )
            candidates[label] = {
                **previous,
                "name": label,
                "confidence": new_priority,
                "differential_priority": new_priority,
                "evidence_support": _stronger_prediction_support_strength(
                    previous.get("evidence_support"),
                    "weak_or_proxy",
                ),
                "claim_status": _stronger_claim_status(
                    previous.get("claim_status"),
                    "candidate_requires_validation",
                ),
                "status": "plausible",
                "rationale": _append_short_note(
                    previous.get("rationale"),
                    _public_update_reason(label, item),
                ),
                "evidence_refs": evidence_refs,
                "validation_needed": _merged_text_list(
                    previous.get("validation_needed"),
                    [
                        (
                            "Validate this proxy-supported mechanism with direct "
                            "photophysical, experimental, or higher-level "
                            "computational evidence."
                        )
                    ],
                    limit=6,
                ),
            }
            evidence_refs_by_label[label].append(evidence_id)
            reasons_by_label[label].append(_public_update_reason(label, item))
            support_by_label[label] = "newly_supported"
    ordered_candidates = _sort_update_candidates(candidates, previous_hypotheses)
    current = _hypothesis_name(ordered_candidates[0]) if ordered_candidates else "unknown"
    runner_up = (
        _hypothesis_name(ordered_candidates[1]) if len(ordered_candidates) > 1 else None
    )
    priority_deltas = _public_update_priority_deltas(
        ordered_candidates,
        previous_by_label=previous_by_label,
        round_id=round_id,
        evidence_refs_by_label=evidence_refs_by_label,
        reasons_by_label=reasons_by_label,
        support_by_label=support_by_label,
    )
    support_arguments = _public_update_support_arguments(
        ordered_candidates,
        round_id=round_id,
        evidence_refs_by_label=evidence_refs_by_label,
        reasons_by_label=reasons_by_label,
    )
    selected = _public_update_next_capability_ids(payload, evidence_refs_by_label)
    action = "dispatch" if selected else "finalize"
    return {
        "decision_id": f"{round_id}:planner_public_update_reducer",
        "round_id": round_id,
        "portfolio": {
            "current": current,
            "runner_up": runner_up,
            "hypotheses": ordered_candidates,
        },
        "current_hypothesis": current,
        "runner_up_hypothesis": runner_up,
        "confidence": _priority_for_payload(ordered_candidates[0])
        if ordered_candidates
        else 0.0,
        "diagnosis": (
            "Planner portfolio updated from public runtime evidence. Ranking uses "
            "differential priority, while mechanism claims remain bounded to "
            "weak/proxy evidence unless direct validation is available."
        ),
        "action": action,
        "selected_capability_ids": selected,
        "unresolved_gaps": [
            "Mechanism candidates remain structure/tool-proxy supported; direct "
            "wet-lab or high-level excited-state validation was not used."
        ],
        "priority_deltas": priority_deltas,
        "mechanism_support_arguments": support_arguments,
        "final_answer_draft": (
            "Planner finalized a source-grounded mechanism portfolio from public "
            "runtime evidence with proxy boundaries."
            if action == "finalize"
            else None
        ),
        "rationale": (
            "Used the public capability evidence guide to absorb current-round "
            "evidence into Planner-owned portfolio deltas; no private reference or "
            "case-specific source was accessed."
        ),
        "raw_response": {"source": "public_update_evidence_reducer"},
    }


def _payload_item_screen_labels(item: dict[str, object]) -> list[str]:
    labels = []
    for raw_label in item.get("screens", []) if isinstance(item.get("screens"), list) else []:
        label = _normalize_mechanism_label(raw_label)
        if label in MECHANISM_POOL and _payload_screen_is_positive_for_label(label, item):
            labels.append(label)
    if not labels:
        route_map = _mechanism_route_priority_map()["route_to_mechanisms"]
        if isinstance(route_map, dict):
            for raw_label in route_map.get(_payload_capability_id(item), []):
                label = _normalize_mechanism_label(raw_label)
                if label in MECHANISM_POOL and _payload_screen_is_positive_for_label(label, item):
                    labels.append(label)
    return list(dict.fromkeys(labels))


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


def _payload_evidence_signal_type(item: dict[str, object]) -> str:
    status = str(item.get("status") or "").strip().lower()
    if status in {"failed", "unsupported", "missing"}:
        return "missing_or_failed"
    capability_id = _payload_capability_id(item)
    if _is_material_followup_capability(capability_id):
        return "positive_or_proxy_observable"
    text = " ".join(
        [
            str(item.get("claim") or ""),
            str(item.get("summary") or ""),
            str(item.get("finding") or ""),
            " ".join(_text_list(item.get("limits"), limit=3)),
            str(item.get("boundary") or ""),
        ]
    ).lower()
    if str(item.get("basis") or "").strip().lower() == "checklist" or any(
        term in text
        for term in (
            "records which",
            "checklist",
        )
    ):
        return "checklist_boundary_only"
    return "positive_or_proxy_observable"


def _public_update_priority_for_label(
    label: str,
    item: dict[str, object],
    *,
    payload: dict[str, object],
) -> float:
    tier = _payload_evidence_tier(item)
    capability_id = _payload_capability_id(item)
    base = _evidence_item_priority_for_label(label, item)
    route_priority = _route_label_priority_override(label, item, payload=payload)
    if route_priority is not None:
        base = max(base, route_priority)
    elif tier == "computed_direct_proxy":
        base = max(base, PUBLIC_ABSORPTION_COMPUTED_PRIORITY)
    elif tier == "mechanism_specific_structural_trigger":
        base = max(base, PUBLIC_ABSORPTION_MECHANISM_SPECIFIC_PRIORITY)
    elif tier == "electronic_weak_trigger":
        base = max(base, PUBLIC_ABSORPTION_ELECTRONIC_PRIORITY)
    elif tier == "structural_trigger":
        base = max(base, PUBLIC_ABSORPTION_STRUCTURAL_PRIORITY)
    if tier == "mechanism_specific_structural_trigger" and _payload_has_computed_context(payload):
        base = min(base, PUBLIC_ABSORPTION_MECHANISM_SPECIFIC_CEILING_WITH_COMPUTED)
    if (
        label == "AGGREGATE_EXCITON_EXCIMER"
        and tier in {"structural_trigger", "mechanism_specific_structural_trigger"}
        and not _is_material_followup_capability(capability_id)
    ):
        base = min(base, 0.2)
    if label == "ICT_TICT_CT" and _is_weak_ct_proxy_item(item):
        base = min(base, WEAK_CT_PROXY_PRIORITY_CEILING)
    if capability_id in {
        "microscopic.run_frontier_orbital_partition",
        "microscopic.extract_ct_descriptors_from_bundle",
    } and not _has_informative_ct_metrics(item):
        base = min(base, 0.2)
    return base


def _route_label_priority_override(
    label: str,
    item: dict[str, object],
    *,
    payload: dict[str, object],
) -> float | None:
    capability_id = _payload_capability_id(item)
    metrics = _metrics_from_payload_item(item)
    if capability_id == "microscopic.run_torsion_brightness_coupling_scan":
        if label == "RIM_RIR_RIV":
            return PUBLIC_ABSORPTION_COMPUTED_PRIORITY
        if label == "RACI_CI_ACCESS":
            return 0.27 if _has_raci_torsion_proxy(metrics) else 0.16
        if label == "ICT_TICT_CT":
            return 0.21
    if capability_id == "microscopic.run_bright_dark_state_ordering":
        if label == "RADIATIVE_RATE_STATE_BALANCE":
            state_count = _number_like(metrics.get("state_count")) or 0.0
            bright_state_index = _number_like(metrics.get("bright_state_index")) or 0.0
            if state_count > 1.0 or bright_state_index > 1.0:
                return PUBLIC_ABSORPTION_COMPUTED_PRIORITY
            return _standalone_oscillator_priority(metrics)
        if label == "SOKR_ANTI_KASHA":
            state_count = _number_like(metrics.get("state_count")) or 0.0
            bright_state_index = _number_like(metrics.get("bright_state_index")) or 0.0
            return 0.24 if state_count > 1 or bright_state_index > 1 else 0.16
    if capability_id in {
        "microscopic.run_frontier_orbital_partition",
        "microscopic.extract_ct_descriptors_from_bundle",
    }:
        if label == "PET_ET" and not _has_pet_specific_metrics(item):
            return 0.18
        if _has_informative_ct_metrics(item):
            return ARTIFACT_BACKED_CT_PRIORITY_FLOOR
        if label == "ICT_TICT_CT" and _payload_has_esipt_context(payload):
            return 0.16
        return 0.19
    if capability_id == "macro.run_solid_state_emission_proxy" and label in {
        "PACKING_HOST_MATRIX_CONFINEMENT",
        "AGGREGATE_EXCITON_EXCIMER",
        "RIM_RIR_RIV",
    }:
        if label == "RIM_RIR_RIV":
            return max(
                _rim_priority_from_metrics(metrics, computed=False),
                _rim_priority_from_solid_state_proxy(metrics),
            )
        if label == "PACKING_HOST_MATRIX_CONFINEMENT":
            return (
                GENERIC_PACKING_PROXY_PRIORITY_CEILING
                if metrics.get("aggregation_prone_proxy")
                else 0.16
            )
        return 0.18
    if capability_id == "macro.screen_polar_binding_site_prior" and label in {
        "HOST_GUEST_INTERACTION",
        "PET_ET",
    }:
        return _polar_screening_priority(label, metrics)
    if (
        capability_id == "macro.screen_metal_triplet_prior"
        and label == "TRIPLET_METAL_ENERGY_TRANSFER"
    ):
        return _evidence_item_priority_for_label(label, item)
    return None


def _has_informative_ct_metrics(item: dict[str, object]) -> bool:
    metrics = _metrics_from_payload_item(item)
    informative_keys = {
        "frontier_fragment_separation",
        "homo_lumo_fragment_shift",
        "ct_descriptor_proxy",
        "charge_transfer_proxy",
        "electron_hole_separation_proxy",
    }
    if any(_nonzero_metric(metrics.get(key)) for key in informative_keys):
        return True
    capability_id = _payload_capability_id(item)
    if capability_id in {
        "microscopic.run_frontier_orbital_partition",
        "microscopic.extract_ct_descriptors_from_bundle",
    }:
        return False
    return False


def _is_weak_ct_proxy_item(item: dict[str, object]) -> bool:
    capability_id = _payload_capability_id(item)
    if capability_id in {
        "macro.screen_donor_acceptor_layout",
        "macro.screen_donor_acceptor_architecture",
    }:
        return True
    if capability_id in {
        "microscopic.run_frontier_orbital_partition",
        "microscopic.extract_ct_descriptors_from_bundle",
    }:
        return not _has_informative_ct_metrics(item)
    return False


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


def _has_pet_specific_metrics(item: dict[str, object]) -> bool:
    metrics = _metrics_from_payload_item(item)
    return bool(
        metrics.get("pet_receptor_like_proxy")
        or metrics.get("redox_alignment_proxy")
        or metrics.get("electron_transfer_pathway_proxy")
        or metrics.get("quencher_receptor_orbital_proxy")
    )


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


def _standalone_oscillator_priority(metrics: dict[str, object]) -> float:
    oscillator = _oscillator_strength_from_metrics(metrics)
    if oscillator is None:
        return STANDALONE_OSCILLATOR_PRIORITY_CEILING
    if oscillator >= 0.20:
        return STRONG_STANDALONE_OSCILLATOR_PRIORITY
    if oscillator >= 0.10:
        return 0.17
    if oscillator >= 0.02:
        return STANDALONE_OSCILLATOR_PRIORITY_CEILING
    if oscillator >= 0.005:
        return 0.17
    return STANDALONE_OSCILLATOR_PRIORITY_CEILING


def _polar_screening_priority(label: str, metrics: dict[str, object]) -> float:
    formal_charge = abs(_number_like(metrics.get("formal_charge")) or 0.0)
    hbd = _number_like(metrics.get("hbd_count")) or 0.0
    hba = _number_like(metrics.get("hba_count")) or 0.0
    carbonyl_sites = _number_like(metrics.get("carbonyl_like_site_count")) or 0.0
    strong_binding_prior = formal_charge > 0.0 or hbd > 0.0 or hba >= 4.0
    receptor_prior = bool(metrics.get("pet_receptor_like_proxy"))
    if label == "HOST_GUEST_INTERACTION":
        if strong_binding_prior:
            return 0.25
        if metrics.get("host_guest_followup_trigger") or metrics.get(
            "polar_binding_site_proxy"
        ):
            return 0.19
        return 0.0
    if label == "PET_ET":
        if receptor_prior and strong_binding_prior:
            return 0.22
        if receptor_prior or metrics.get("polar_binding_site_proxy") or carbonyl_sites >= 2.0:
            return 0.18
    return 0.0


def _rim_priority_from_solid_state_proxy(metrics: dict[str, object]) -> float:
    if not metrics.get("aggregation_prone_proxy"):
        return 0.0
    rotor_burden = _number_like(metrics.get("rotor_burden_proxy")) or 0.0
    if rotor_burden >= 2.0:
        return 0.25
    if rotor_burden >= 1.0:
        return 0.23
    return 0.0


def _rim_priority_from_metrics(
    metrics: dict[str, object],
    *,
    computed: bool,
) -> float:
    rotor_count = _number_like(metrics.get("rotatable_bond_count")) or 0.0
    torsion_count = _number_like(metrics.get("torsion_candidate_count")) or 0.0
    branch_points = _number_like(metrics.get("branch_point_count")) or 0.0
    flexibility = _number_like(metrics.get("flexibility_proxy")) or 0.0
    flexible = max(rotor_count, torsion_count)
    if flexible >= 1 and (flexibility >= 8.0 or branch_points >= 8.0):
        return 0.26 if computed else 0.25
    if flexible >= 4:
        return PUBLIC_ABSORPTION_COMPUTED_PRIORITY if computed else 0.28
    if flexible >= 2:
        return 0.24 if computed else 0.21
    if flexible >= 1:
        return 0.20 if computed else 0.17
    return 0.12


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


def _payload_has_computed_context(payload: dict[str, object]) -> bool:
    for item in _as_dict_list(payload.get("priority_hygiene_summary")):
        if int(item.get("computed_direct_proxy_count") or 0) > 0:
            return True
    for item in _as_dict_list(payload.get("new_round_evidence")):
        if _payload_evidence_tier(item) == "computed_direct_proxy":
            return True
    return False


def _public_update_reason(label: str, item: dict[str, object]) -> str:
    finding = _short_text(
        item.get("finding") or item.get("claim") or item.get("summary"),
        180,
    )
    tier = _payload_evidence_tier(item)
    metric_summary = _public_runtime_metric_summary(item)
    if finding:
        if metric_summary:
            finding = _short_text(f"{finding} Runtime details: {metric_summary}.", 260)
        return (
            f"Public {tier} evidence from {_payload_evidence_id(item)} screens "
            f"{label}: {finding}"
        )
    return f"Public {tier} evidence from {_payload_evidence_id(item)} screens {label}."


def _public_runtime_metric_summary(item: dict[str, object]) -> str:
    metrics = _metrics_from_payload_item(item)
    keep = [
        "oscillator_strength",
        "bright_state_index",
        "state_count",
        "missing_deliverable_count",
        "rotatable_bond_count",
        "torsion_candidate_count",
        "aggregation_prone_proxy",
        "aromatic_ring_count",
        "mol_logp",
        "donor_acceptor_proxy",
        "donor_acceptor_partition_proxy",
        "donor_atom_symbols",
        "acceptor_atom_symbols",
        "polar_atom_symbols",
        "polar_binding_site_proxy",
        "carbonyl_like_site_count",
        "host_guest_followup_trigger",
        "pet_receptor_like_proxy",
        "metal_or_lanthanide_prior",
        "heavy_atom_triplet_prior",
        "sulfur_phosphorus_triplet_prior",
        "hbd_count",
        "hba_count",
        "proton_transfer_pair_proxy_count",
    ]
    pairs = []
    for key in keep:
        value = metrics.get(key)
        if value is None or value == "":
            continue
        pairs.append(f"{key}={value}")
        if len(pairs) >= 5:
            break
    return "; ".join(pairs)


def _sort_update_candidates(
    candidates: dict[str, dict[str, object]],
    previous_hypotheses: list[dict[str, object]],
) -> list[dict[str, object]]:
    previous_order = {
        _hypothesis_name(item): index
        for index, item in enumerate(previous_hypotheses)
        if _hypothesis_name(item)
    }
    return [
        _normalize_hypothesis_payload(candidates[label])
        for label in sorted(
            [label for label in candidates if label and label != "unknown"],
            key=lambda name: (
                _priority_for_payload(candidates[name]),
                -previous_order.get(name, len(previous_order)),
            ),
            reverse=True,
        )
    ]


def _public_update_priority_deltas(
    ordered_candidates: list[dict[str, object]],
    *,
    previous_by_label: dict[str, dict[str, object]],
    round_id: str,
    evidence_refs_by_label: dict[str, list[str]],
    reasons_by_label: dict[str, list[str]],
    support_by_label: dict[str, str],
) -> list[dict[str, object]]:
    deltas: list[dict[str, object]] = []
    for item in ordered_candidates:
        label = _hypothesis_name(item)
        if not label:
            continue
        previous_priority = _priority_for_payload(previous_by_label.get(label, {}))
        new_priority = _priority_for_payload(item)
        refs = list(dict.fromkeys(evidence_refs_by_label.get(label, [])))[:4]
        changed = abs(new_priority - previous_priority) > 0.0005 or refs
        if not changed:
            continue
        reason = " ".join(reasons_by_label.get(label, [])[:2]) or _inferred_delta_reason(
            label,
            previous_priority=previous_priority,
            new_priority=new_priority,
        )
        deltas.append(
            {
                "round_id": round_id,
                "label": label,
                "previous_priority": round(previous_priority, 6),
                "priority_delta": round(new_priority - previous_priority, 6),
                "new_priority": round(new_priority, 6),
                "delta_reason": _short_text(reason, 700),
                "evidence_refs_added": refs,
                "support_change": support_by_label.get(label, "unchanged"),
            }
        )
    return deltas


def _public_update_support_arguments(
    ordered_candidates: list[dict[str, object]],
    *,
    round_id: str,
    evidence_refs_by_label: dict[str, list[str]],
    reasons_by_label: dict[str, list[str]],
) -> list[dict[str, object]]:
    arguments: list[dict[str, object]] = []
    for item in ordered_candidates:
        label = _hypothesis_name(item)
        refs = list(dict.fromkeys(evidence_refs_by_label.get(label, [])))[:3]
        if not label or not refs:
            continue
        finding = " ".join(reasons_by_label.get(label, [])[:2])
        arguments.append(
            {
                "support_id": f"{round_id}:support:{len(arguments) + 1:03d}",
                "target_label": label,
                "support_level": "weak",
                "finding": _short_text(
                    finding or f"Runtime evidence screens {label}.",
                    700,
                ),
                "warrant": _initial_verdict_warrant(label),
                "boundary": (
                    "This is source-grounded proxy support from public runtime "
                    "evidence. It does not claim source-paper, wet-lab, lifetime, "
                    "PLQY, conical-intersection, nonadiabatic, or definitive "
                    "excited-state validation."
                ),
                "observation_refs": refs,
            }
        )
        if len(arguments) >= 6:
            break
    return arguments


def _public_update_next_capability_ids(
    payload: dict[str, object],
    evidence_refs_by_label: dict[str, list[str]],
) -> list[str]:
    labels = {label for label, refs in evidence_refs_by_label.items() if refs}
    candidates = _next_discriminating_followup_capability_ids(
        {
            **payload,
            "priority_hygiene_summary": [
                item
                for item in _as_dict_list(payload.get("priority_hygiene_summary"))
                if _normalize_mechanism_label(item.get("label")) in labels
            ],
        }
    )
    executed = _executed_capability_ids_from_payload(payload)
    selected = [capability_id for capability_id in candidates if capability_id not in executed]
    return selected[:3]


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


def _stronger_prediction_support_strength(first: object, second: object) -> str:
    first_text = _normalize_prediction_support_strength(first)
    second_text = _normalize_prediction_support_strength(second)
    if _support_strength_rank(first_text) >= _support_strength_rank(second_text):
        return first_text
    return second_text


def _stronger_claim_status(first: object, second: object) -> str:
    first_text = _normalize_claim_status(first)
    second_text = _normalize_claim_status(second)
    rank = {
        "supported_claim": 3,
        "partially_supported_candidate": 2,
        "candidate_requires_validation": 1,
        "underdetermined": 0,
    }
    return first_text if rank[first_text] >= rank[second_text] else second_text


def _public_initial_reason(label: str, item: dict[str, object]) -> str:
    metrics = _metrics_from_payload_item(item)
    tier = _payload_evidence_tier(item)
    if label == "ESIPT_PT" and metrics.get("esipt_motif_proxy") is True:
        return "Public ESIPT donor/acceptor motif proxy triggers an ESIPT validation candidate."
    if label == "RADIATIVE_RATE_STATE_BALANCE":
        oscillator = _oscillator_strength(item)
        if oscillator is not None:
            return (
                "Low-cost oscillator-strength/state-ordering proxy triggers a "
                "radiative/state-balance validation candidate."
            )
    if label == "ICT_TICT_CT" and metrics.get("donor_acceptor_proxy"):
        return "Public donor-acceptor structural proxy triggers an ICT/TICT/CT candidate."
    if label == "PET_ET" and (
        metrics.get("pet_receptor_like_proxy") or metrics.get("donor_acceptor_proxy")
    ):
        return (
            "Public polar/receptor or donor-acceptor proxy triggers a PET/ET "
            "screening candidate."
        )
    if label == "RIM_RIR_RIV":
        return "Public rotor/torsion topology proxy triggers a RIM/RIR/RIV validation candidate."
    if label == "RACI_CI_ACCESS":
        return (
            "Public torsion-brightness observables trigger a RACI/CI-access "
            "validation candidate."
        )
    if label == "PACKING_HOST_MATRIX_CONFINEMENT":
        return (
            "Public aggregation or solid-state structural proxy triggers a "
            "packing/confinement candidate."
        )
    if label == "AGGREGATE_EXCITON_EXCIMER":
        return "Public aggregation-prone scaffold proxy triggers an aggregate/excimer candidate."
    if label == "TRIPLET_METAL_ENERGY_TRANSFER":
        return (
            "Public metal/heavy-atom/triplet structural proxy triggers a "
            "triplet/metal candidate."
        )
    return f"Public runtime evidence tier {tier} triggered this differential candidate."


def _planner_decision_from_initial_verdict(
    response: dict[str, object],
    *,
    case_run: CaseRun,
    round_id: str,
    payload: dict[str, object],
) -> dict[str, object]:
    evidence_ids = {
        _payload_evidence_id(item)
        for item in _as_dict_list(payload.get("evidence"))
        if _payload_evidence_id(item)
    }
    raw_candidates = (
        _as_dict_list(response.get("ranked_candidates"))
        or _as_dict_list(response.get("mechanism_predictions"))
        or _as_dict_list(response.get("predictions"))
        or _as_dict_list(response.get("candidates"))
        or _as_dict_list(_nested_dict(response, "verdict").get("ranked_candidates"))
        or _as_dict_list(_nested_dict(response, "verdict").get("mechanism_predictions"))
        or _as_dict_list(_nested_dict(response, "verdict").get("predictions"))
        or _as_dict_list(_nested_dict(response, "verdict").get("candidates"))
        or _as_dict_list(_nested_dict(response, "verdict").get("differential_priority"))
    )
    candidates = []
    for index, item in enumerate(raw_candidates[:5], start=1):
        label = _normalize_mechanism_label(
            item.get("label")
            or item.get("mechanism")
            or item.get("mechanism_label")
            or item.get("name")
        )
        if label not in MECHANISM_POOL:
            continue
        refs = [
            ref
            for ref in _text_list(item.get("evidence_refs"), limit=6)
            if ref in evidence_ids
        ]
        if not refs:
            refs = _default_refs_for_verdict_label(label, payload)[:3]
        candidates.append(
            {
                "label": label,
                "priority": _initial_verdict_priority(
                    item,
                    rank=index,
                    label=label,
                    refs=refs,
                    payload=payload,
                ),
                "support": _normalize_prediction_support_strength(
                    item.get("support")
                    or item.get("support_strength")
                    or item.get("evidence_support")
                    or "weak_or_proxy"
                ),
                "status": _normalize_claim_status(
                    item.get("status")
                    or item.get("claim_status")
                    or "candidate_requires_validation"
                ),
                "evidence_refs": refs,
                "reason": _short_text(item.get("reason"), 260)
                or "Initial verdict ranked this mechanism from runtime evidence.",
            }
        )
    candidates = _backfill_initial_verdict_candidates(candidates, payload)
    if not candidates:
        raise ValueError(
            "Initial mechanism verdict must return 3 to 5 ranked_candidates with "
            "valid mechanism_pool labels and evidence_refs from the payload; an "
            "empty JSON object or empty candidate list is not a valid PlannerDecision."
        )
    candidates.sort(key=lambda item: float(item["priority"]), reverse=True)
    current = str(candidates[0]["label"])
    runner_up = str(candidates[1]["label"]) if len(candidates) > 1 else None
    hypothesis_items = [
        {
            "name": item["label"],
            "confidence": item["priority"],
            "differential_priority": item["priority"],
            "evidence_support": item["support"],
            "claim_status": item["status"],
            "status": "pending",
            "rationale": item["reason"],
            "evidence_refs": item["evidence_refs"],
            "validation_needed": [
                "Validate this proxy-supported mechanism with direct photophysical, "
                "experimental, or higher-level computational evidence."
            ],
        }
        for item in candidates
    ]
    priority_deltas = [
        {
            "round_id": round_id,
            "label": item["label"],
            "previous_priority": 0.0,
            "priority_delta": item["priority"],
            "new_priority": item["priority"],
            "delta_reason": item["reason"],
            "evidence_refs_added": item["evidence_refs"],
            "support_change": "newly_supported"
            if item["support"] in {"strong", "partial", "weak_or_proxy"}
            else "unchanged",
        }
        for item in candidates
    ]
    support_arguments = [
        {
            "support_id": f"{round_id}:support:{index:03d}",
            "target_label": item["label"],
            "support_level": "weak"
            if item["support"] == "weak_or_proxy"
            else item["support"],
            "finding": item["reason"],
            "warrant": _initial_verdict_warrant(str(item["label"])),
            "boundary": (
                "Support remains bounded by available low-cost proxy evidence; no "
                "private answer key, source paper, wet-lab result, CI search, "
                "lifetime, or PLQY evidence was used."
            ),
            "observation_refs": item["evidence_refs"],
        }
        for index, item in enumerate(candidates[:4], start=1)
        if item["evidence_refs"]
    ]
    selected = _ranked_initial_followup_capability_ids(
        [
            capability_id
            for capability_id in _initial_verdict_selected_capability_ids(response, payload)
            if capability_id in _initial_available_followup_capability_ids(payload)
        ],
        candidates=candidates,
    )
    action = str(response.get("action") or "finalize").strip().lower()
    if action not in {"dispatch", "finalize"}:
        action = "finalize"
    if selected:
        action = "dispatch"
    return {
        "decision_id": f"{round_id}:planner_initial_verdict",
        "round_id": round_id,
        "portfolio": {
            "current": current,
            "runner_up": runner_up,
            "hypotheses": hypothesis_items,
        },
        "current_hypothesis": current,
        "runner_up_hypothesis": runner_up,
        "confidence": float(candidates[0]["priority"]),
        "diagnosis": (
            "Initial mechanism verdict ranked candidates from first-round public "
            "runtime evidence while preserving proxy boundaries and using agenda "
            "coverage to select remaining under-screened follow-up routes."
        ),
        "action": action,
        "selected_capability_ids": selected if action == "dispatch" else [],
        "unresolved_gaps": [
            "Further experimental or higher-level computational validation is still needed."
        ],
        "priority_deltas": priority_deltas,
        "mechanism_support_arguments": support_arguments,
        "final_answer_draft": (
            "Initial Planner verdict completed from public runtime evidence."
            if action == "finalize"
            else None
        ),
        "rationale": (
            "Converted compact initial verdict into a PlannerDecision; agenda "
            "coverage remains the source for under-screened route follow-up."
        ),
        "raw_response": {"initial_verdict_raw": response},
    }


def _nested_dict(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _initial_verdict_selected_capability_ids(
    response: dict[str, object],
    payload: dict[str, object],
) -> list[str]:
    selected: list[str] = []
    selected.extend(
        _text_list(response.get("next_capability_ids"), limit=MAX_DISPATCHES_PER_ROUND)
    )
    selected.extend(
        _text_list(
            response.get("selected_capability_ids"),
            limit=MAX_DISPATCHES_PER_ROUND,
        )
    )
    plan = _nested_dict(response, "plan")
    selected.extend(_text_list(plan.get("actions"), limit=6))
    selected.extend(_text_list(plan.get("capability_ids"), limit=6))
    verdict = _nested_dict(response, "verdict")
    selected.extend(_text_list(verdict.get("next_capability_ids"), limit=MAX_DISPATCHES_PER_ROUND))
    selected.extend(
        _text_list(verdict.get("recommended_capability_ids"), limit=MAX_DISPATCHES_PER_ROUND)
    )
    if not selected:
        selected.extend(_evidence_gap_followup_capability_ids(response, payload))
    return list(dict.fromkeys(selected))


def _ranked_initial_followup_capability_ids(
    capability_ids: list[str],
    *,
    candidates: list[dict[str, object]],
) -> list[str]:
    top_labels = [str(item["label"]) for item in candidates[:3] if item.get("label")]
    preferred: list[str] = []
    if set(top_labels) & {"AGGREGATE_EXCITON_EXCIMER", "PACKING_HOST_MATRIX_CONFINEMENT"}:
        preferred.extend(
            [
                "macro.run_solid_state_emission_proxy",
                "macro.run_dimer_packing_proxy",
                "macro.run_aggregate_contact_proxy",
            ]
        )
    if set(top_labels) & {"RIM_RIR_RIV", "RACI_CI_ACCESS"}:
        preferred.append("microscopic.run_torsion_brightness_coupling_scan")
    if set(top_labels) & {"RADIATIVE_RATE_STATE_BALANCE", "SOKR_ANTI_KASHA"}:
        preferred.append("microscopic.run_bright_dark_state_ordering")
    if set(top_labels) & {"ICT_TICT_CT", "PET_ET"}:
        preferred.extend(
            [
                "microscopic.run_frontier_orbital_partition",
                "microscopic.extract_ct_descriptors_from_bundle",
            ]
        )
    ordered = [
        capability_id
        for capability_id in [*preferred, *capability_ids]
        if capability_id in capability_ids
    ]
    return list(dict.fromkeys(ordered))[:MAX_DISPATCHES_PER_ROUND]


def _evidence_gap_followup_capability_ids(
    response: dict[str, object],
    payload: dict[str, object],
) -> list[str]:
    """Choose generic follow-up routes when initial verdict leaves proxy gaps.

    This inspects only public runtime evidence and mechanism-pool labels already
    mentioned by the model or coverage memo. It is a route-selection guard, not a
    mechanism answer prior.
    """

    available = set(
        _text_list(payload.get("next_capability_ids_available"), limit=30)
        or _text_list(payload.get("dispatch_capability_ids_available"), limit=30)
    )
    labels = _initial_verdict_mentioned_labels(response, payload)
    desired: list[str] = []
    agenda = payload.get("agenda")
    if isinstance(agenda, dict):
        for capability_id in _text_list(
            agenda.get("recommended_next_capability_ids"),
            limit=8,
        ):
            desired.append(capability_id)
        for item in _as_dict_list(agenda.get("recommended_next_routes")):
            desired.extend(_text_list(item.get("capability_ids"), limit=4))
    if labels & {"RADIATIVE_RATE_STATE_BALANCE", "SOKR_ANTI_KASHA"}:
        desired.append("microscopic.run_bright_dark_state_ordering")
    if labels & {"ICT_TICT_CT", "PET_ET"}:
        desired.extend(
            [
                "microscopic.run_frontier_orbital_partition",
                "microscopic.extract_ct_descriptors_from_bundle",
            ]
        )
    if labels & {"RIM_RIR_RIV", "RACI_CI_ACCESS"}:
        desired.append("microscopic.run_torsion_brightness_coupling_scan")
    if labels & {"AGGREGATE_EXCITON_EXCIMER", "PACKING_HOST_MATRIX_CONFINEMENT"}:
        desired.append("macro.run_solid_state_emission_proxy")
    return [capability_id for capability_id in desired if capability_id in available]


def _initial_verdict_mentioned_labels(
    response: dict[str, object],
    payload: dict[str, object],
) -> set[str]:
    labels: set[str] = set()
    candidate_sources = [
        response,
        _nested_dict(response, "verdict"),
    ]
    for source in candidate_sources:
        for key in (
            "ranked_candidates",
            "mechanism_predictions",
            "predictions",
            "candidates",
            "differential_priority",
        ):
            for item in _as_dict_list(source.get(key)):
                label = _normalize_mechanism_label(
                    item.get("label")
                    or item.get("mechanism")
                    or item.get("mechanism_label")
                    or item.get("name")
                )
                if label in MECHANISM_POOL:
                    labels.add(label)
    labels.update(
        label
        for label in _text_list(payload.get("coverage_under_screened"), limit=12)
        if label in MECHANISM_POOL
    )
    agenda = payload.get("agenda")
    if isinstance(agenda, dict):
        labels.update(
            label
            for label in _text_list(agenda.get("under_screened_labels"), limit=12)
            if label in MECHANISM_POOL
        )
        for item in _as_dict_list(agenda.get("agenda_items")):
            label = _normalize_mechanism_label(item.get("label"))
            if label in MECHANISM_POOL:
                labels.add(label)
        for item in _as_dict_list(agenda.get("recommended_next_routes")):
            labels.update(
                label
                for label in (
                    _normalize_mechanism_label(value)
                    for value in _text_list(item.get("targets"), limit=6)
                )
                if label in MECHANISM_POOL
            )
    for item in _as_dict_list(payload.get("priority_hygiene_summary")):
        label = _normalize_mechanism_label(item.get("label"))
        if label in MECHANISM_POOL:
            labels.add(label)
    return labels


def _initial_available_followup_capability_ids(payload: dict[str, object]) -> set[str]:
    return set(
        _text_list(payload.get("next_capability_ids_available"), limit=30)
        or _text_list(payload.get("dispatch_capability_ids_available"), limit=30)
    )


def _initial_verdict_priority(
    item: dict[str, object],
    *,
    rank: int,
    label: str,
    refs: list[str],
    payload: dict[str, object],
) -> float:
    for key in (
        "priority",
        "differential_priority",
        "confidence",
        "plausibility_confidence",
    ):
        if item.get(key) is not None:
            return _float_between(item.get(key), default=0.0)
    evidence_priority = _evidence_default_priority_for_label(
        label,
        refs=refs,
        payload=payload,
    )
    if evidence_priority > 0.0:
        return evidence_priority
    rank_defaults = {1: 0.34, 2: 0.29, 3: 0.24, 4: 0.2, 5: 0.16}
    return rank_defaults.get(rank, 0.12)


def _backfill_initial_verdict_candidates(
    candidates: list[dict[str, object]],
    payload: dict[str, object],
) -> list[dict[str, object]]:
    by_label = {str(item["label"]): item for item in candidates}
    for label in MECHANISM_POOL:
        if label in by_label:
            continue
        refs = _default_refs_for_verdict_label(label, payload)[:3]
        priority = _evidence_default_priority_for_label(label, refs=refs, payload=payload)
        if priority < 0.2:
            continue
        by_label[label] = {
            "label": label,
            "priority": priority,
            "support": "weak_or_proxy",
            "status": "candidate_requires_validation",
            "evidence_refs": refs,
            "reason": "Public runtime evidence triggered this differential candidate.",
        }
    return list(by_label.values())


def _evidence_default_priority_for_label(
    label: str,
    *,
    refs: list[str],
    payload: dict[str, object],
) -> float:
    evidence_by_id = {
        _short_text(item.get("id"), 220): item
        for item in _as_dict_list(payload.get("evidence"))
        if _short_text(item.get("id"), 220)
    }
    priorities = [
        _evidence_item_priority_for_label(label, item)
        for ref in refs
        if (item := evidence_by_id.get(ref)) is not None
    ]
    if label == "RADIATIVE_RATE_STATE_BALANCE" and _payload_has_esipt_context(payload):
        oscillator_values = [
            value
            for ref in refs
            if (item := evidence_by_id.get(ref)) is not None
            for value in [_oscillator_strength(item)]
            if value is not None
        ]
        if oscillator_values and max(oscillator_values) >= 0.02:
            priorities.append(COUPLED_OSCILLATOR_PRIORITY_FLOOR)
    return max(priorities, default=0.0)


def _evidence_item_priority_for_label(label: str, item: dict[str, object]) -> float:
    metrics = _metrics_from_payload_item(item)
    if label == "ESIPT_PT" and metrics.get("esipt_motif_proxy") is True:
        return 0.31
    if label == "RADIATIVE_RATE_STATE_BALANCE":
        return _standalone_oscillator_priority(metrics)
    if label == "AGGREGATE_EXCITON_EXCIMER" and metrics.get("aggregation_prone_proxy") is True:
        ring_count = _number_like(metrics.get("aromatic_ring_count")) or 0.0
        logp = _number_like(metrics.get("mol_logp")) or 0.0
        if (
            metrics.get("aggregate_exciton_proxy")
            or metrics.get("excimer_geometry_proxy")
            or metrics.get("spectral_new_band_proxy")
        ):
            return 0.22 if ring_count >= 3 or logp >= 3.0 else 0.18
        return 0.18
    if label == "RIM_RIR_RIV":
        return _rim_priority_from_metrics(metrics, computed=False)
    if label == "ICT_TICT_CT" and metrics.get("donor_acceptor_proxy"):
        conjugation = _number_like(metrics.get("conjugation_proxy")) or 0.0
        hetero_count = _number_like(metrics.get("hetero_atom_count")) or 0.0
        if conjugation >= 8.0 or hetero_count >= 4:
            return 0.24
        return 0.21
    if label == "HOST_GUEST_INTERACTION":
        return _polar_screening_priority(label, metrics)
    if label == "PET_ET":
        priority = _polar_screening_priority(label, metrics)
        if priority > 0.0:
            return priority
        if metrics.get("donor_acceptor_proxy"):
            return 0.21
    if label == "TRIPLET_METAL_ENERGY_TRANSFER":
        if metrics.get("metal_or_lanthanide_prior"):
            return 0.3
        if metrics.get("heavy_atom_triplet_prior"):
            return 0.28
        if metrics.get("sulfur_phosphorus_triplet_prior"):
            return 0.23
    tier = _payload_evidence_tier(item)
    return {
        "computed_direct_proxy": 0.2,
        "mechanism_specific_structural_trigger": 0.2,
        "structural_trigger": 0.16,
        "electronic_weak_trigger": 0.16,
        "artifact_weak_trigger": 0.14,
    }.get(tier, 0.0)


def _payload_has_esipt_context(payload: dict[str, object]) -> bool:
    for item in _as_dict_list(payload.get("evidence")):
        metrics = _metrics_from_payload_item(item)
        screens = item.get("screens")
        screen_labels = {str(value) for value in screens} if isinstance(screens, list) else set()
        if metrics.get("esipt_motif_proxy") is True:
            return True
        if "ESIPT_PT" in screen_labels and (
            metrics.get("hbd_count") or metrics.get("proton_transfer_pair_proxy_count")
        ):
            return True
    for item in _as_dict_list(payload.get("priority_hygiene_summary")):
        if str(item.get("label") or "") != "ESIPT_PT":
            continue
        for hit in _as_dict_list(item.get("route_hits")):
            metrics = _metrics_from_payload_item(hit)
            if metrics.get("esipt_motif_proxy") is True:
                return True
            if metrics.get("hbd_count") or metrics.get("proton_transfer_pair_proxy_count"):
                return True
    return False


def _oscillator_strength(item: dict[str, object]) -> float | None:
    metrics = _metrics_from_payload_item(item)
    return _oscillator_strength_from_metrics(metrics)


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


def _initial_verdict_warrant(label: str) -> str:
    warrants = {
        "RIM_RIR_RIV": (
            "Rotor/torsion and aggregation or rigidification proxies are relevant "
            "because RIM/RIR/RIV candidates require motion-coupled nonradiative "
            "channels that may be restricted in condensed environments."
        ),
        "PACKING_HOST_MATRIX_CONFINEMENT": (
            "Packing, solid-state, contact, or confinement proxies are relevant "
            "because this candidate concerns environmental restriction rather "
            "than an isolated-molecule proof."
        ),
        "HOST_GUEST_INTERACTION": (
            "Polar, pore, binding-site, or guest-accessibility proxies are relevant "
            "screening evidence for host-guest interaction candidates."
        ),
        "ICT_TICT_CT": (
            "Donor-acceptor layout, frontier-orbital, torsional, or charge-locality "
            "proxies are relevant because ICT/TICT/CT candidates require electronic "
            "redistribution or twisted charge-transfer pathways."
        ),
        "ESIPT_PT": (
            "A proton donor-acceptor motif or intramolecular H-bond proxy is a "
            "mechanism-specific structural trigger for an ESIPT/PT validation "
            "candidate."
        ),
        "PET_ET": (
            "Receptor-like polar motifs, donor/acceptor layout, or frontier-orbital "
            "alignment proxies are relevant screening evidence for PET/ET candidates."
        ),
        "AGGREGATE_EXCITON_EXCIMER": (
            "Aromatic surface, aggregation-prone scaffold, contact, or dimer proxy "
            "evidence can justify an aggregate/excimer differential candidate, "
            "while remaining weaker than aggregate-state spectral proof."
        ),
        "RADIATIVE_RATE_STATE_BALANCE": (
            "Low-cost state-ordering, oscillator-strength, or bright/dark-state "
            "observables are relevant proxies for radiative-rate or state-balance "
            "candidates."
        ),
        "TRIPLET_METAL_ENERGY_TRANSFER": (
            "Metal, heavy-atom, triplet, or energy-transfer structural priors are "
            "screening evidence for triplet/metal energy-transfer candidates."
        ),
        "RACI_CI_ACCESS": (
            "Torsional or flapping coordinates can motivate a RACI/CI-access "
            "candidate when they may couple molecular motion to nonradiative decay."
        ),
        "SOKR_ANTI_KASHA": (
            "Bright/dark ordering or higher-state oscillator-strength proxies are "
            "screening evidence for SOKR/anti-Kasha candidates."
        ),
    }
    return warrants.get(
        label,
        "The cited public runtime evidence is relevant only as bounded proxy support.",
    )


def _number_like(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _default_refs_for_verdict_label(
    label: str,
    payload: dict[str, object],
) -> list[str]:
    refs = []
    for item in _as_dict_list(payload.get("evidence")):
        raw_screens = item.get("screens")
        screens = [str(value) for value in raw_screens] if isinstance(raw_screens, list) else []
        evidence_id = _payload_evidence_id(item)
        if (
            label in screens
            and evidence_id
            and _payload_screen_is_positive_for_label(label, item)
        ):
            refs.append(evidence_id)
    return refs


def _payload_evidence_id(item: dict[str, object]) -> str:
    return _short_text(item.get("id") or item.get("evidence_id"), 220)


def _payload_capability_id(item: dict[str, object]) -> str:
    return _short_text(item.get("capability_id"), 180)


def _payload_evidence_tier(item: dict[str, object]) -> str:
    return str(item.get("tier") or item.get("evidence_tier") or "unknown")


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


def _initial_coverage_dispatch_decision(
    *,
    case_run: CaseRun,
    round_id: str,
    stage: str,
    payload: dict[str, object],
    capability_registry: CapabilityRegistry,
) -> PlannerDecision | None:
    if stage != "planner_initial" or case_run.evidence_ledger.items:
        return None
    dispatch_requests = _dispatches_from_coverage_memo(
        payload_context=payload,
        round_id=round_id,
        capability_registry=capability_registry,
    )
    if not dispatch_requests:
        return None
    return PlannerDecision(
        decision_id=f"{round_id}:planner_initial_coverage_dispatch",
        round_id=round_id,
        portfolio=case_run.portfolio,
        current_hypothesis=case_run.portfolio.current,
        runner_up_hypothesis=case_run.portfolio.runner_up,
        confidence=0.0,
        diagnosis=(
            "Initial Planner step routed balanced public evidence collection from "
            "the agenda coverage memo; no mechanism ranking was made before evidence."
        ),
        action="dispatch",
        dispatch_requests=dispatch_requests,
        unresolved_gaps=[
            "Mechanism portfolio ranking is deferred until first-round evidence is collected."
        ],
        rationale="Initial evidence routing only; Planner ranking remains evidence-gated.",
        raw_response={
            "prompt_name": "planner_initial_coverage_dispatch",
            "payload_shape": "coverage_route_dispatch_without_mechanism_ranking",
        },
    )


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
        "goal": "Triage all mechanism_pool labels conceptually before finalizing.",
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
            "Update previous_mechanism_portfolio_state by delta; do not rebuild "
            "ranking from scratch."
        ),
        "priority_deltas_required": True,
        "large_delta_rule": "If abs(delta)>0.15, cite new evidence/review feedback.",
        "low_margin_rule": (
            f"If adjacent priority gap < {LOW_MARGIN_THRESHOLD}, preserve previous "
            "order unless new evidence clearly supports reversal."
        ),
        "route_selection_rule": (
            "Prefer high-urgency agenda coverage and routes that disambiguate close "
            "candidates."
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
        if key in _default_planner_capability_ids()
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
        "limits": [_short_text(limit, 80) for limit in item.limits[:1]],
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
        "observation_refs": item.observation_refs[:6],
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
        "agenda_items": [
            {
                "label": item.label,
                "coverage_status": item.coverage_status,
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
            "operational_note_count": len(list(context.get("operational_notes") or [])),
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
    available = _artifact_available_capability_ids(capability_registry, case_run)
    visible = _default_planner_capability_ids() & available
    coverage = case_run.mechanism_agenda_coverage
    if coverage is not None:
        for route in coverage.recommended_next_routes:
            visible.update(
                capability_id
                for capability_id in route.capability_ids
                if capability_id in available
            )
    if stage == "planner_final_calibration":
        used_capabilities = {
            item.capability_id
            for item in case_run.evidence_ledger.items
            if item.capability_id in available
        }
        visible.update(used_capabilities)
    return visible


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
        if required.issubset(available_kinds):
            ids.add(str(card.get("capability_id", "")))
    return ids


def _default_planner_capability_ids() -> set[str]:
    return {
        "microscopic.run_baseline_bundle",
        "macro.screen_esipt_structural_motif",
        "macro.screen_polar_binding_site_prior",
        "macro.screen_donor_acceptor_layout",
        "macro.screen_donor_acceptor_architecture",
        "macro.screen_rotor_torsion_topology",
        "macro.screen_aggregation_prone_scaffold",
        "macro.screen_metal_triplet_prior",
        "microscopic.run_frontier_orbital_partition",
        "microscopic.run_torsion_brightness_coupling_scan",
        "microscopic.run_bright_dark_state_ordering",
        "macro.run_crystal_restriction_checklist",
        "macro.run_dimer_packing_proxy",
        "macro.run_aggregate_contact_proxy",
        "macro.run_solid_state_emission_proxy",
    }


def _planner_output_contract() -> dict[str, object]:
    return {
        "mode": "delta_first_planner_decision",
        "required_keys": [
            "decision_id",
            "round_id",
            "portfolio",
            "current_hypothesis",
            "confidence",
            "diagnosis",
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
    portfolio = _extract_portfolio_payload(payload, existing_portfolio)
    delta_only_portfolio = _is_delta_only_portfolio_shell(portfolio)
    portfolio = (
        {
            **existing_portfolio,
            "current": _short_text(portfolio.get("current"), 120)
            or existing_portfolio.get("current")
            or "unknown",
            "runner_up": (
                None
                if _is_nullish(portfolio.get("runner_up"))
                else _short_text(portfolio.get("runner_up"), 120)
            ),
        }
        if delta_only_portfolio
        else _normalize_portfolio_payload(portfolio)
    )
    final_calibration = stage == "planner_final_calibration"
    if final_calibration:
        portfolio = _merge_final_calibration_portfolio_payload(
            portfolio,
            previous_portfolio_hypotheses,
        )
    else:
        portfolio = _merge_incremental_portfolio_payload(
            portfolio,
            previous_portfolio_hypotheses,
        )
    priority_deltas = _normalize_priority_deltas(
        payload,
        round_id=round_id,
        previous_hypotheses=previous_portfolio_hypotheses,
        portfolio=portfolio,
        allowed_evidence_ids=allowed_evidence_ids,
    )
    if final_calibration:
        portfolio = _sort_portfolio_by_priority(portfolio)
        low_margin_notes = _low_margin_notes_for_portfolio(portfolio)
    else:
        portfolio = _apply_priority_deltas_to_portfolio(
            portfolio,
            previous_portfolio_hypotheses,
            priority_deltas,
        )
    portfolio, priority_deltas = _apply_evidence_tier_priority_floor(
        portfolio,
        priority_deltas,
        round_id=round_id,
        prompt_payload=prompt_payload or {},
    )
    if final_calibration:
        low_margin_notes = _low_margin_notes_for_portfolio(portfolio)
    else:
        portfolio, priority_deltas, low_margin_notes = _apply_low_margin_carry_forward(
            portfolio,
            round_id=round_id,
            previous_portfolio_hypotheses=previous_portfolio_hypotheses,
            priority_deltas=priority_deltas,
        )
    carried_forward_low_margin = any(
        item.get("action") == "previous_order_carried_forward"
        for item in low_margin_notes
    )
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
    top_after_reducer = _top_hypothesis_name(portfolio)
    if carried_forward_low_margin or (
        top_after_reducer and top_after_reducer != current_candidate
    ):
        current_candidate = _top_hypothesis_name(portfolio) or current_candidate
    current = _short_text(
        current_candidate or portfolio.get("current") or "unknown",
        120,
    )
    portfolio["current"] = current
    runner_up = (
        _second_hypothesis_name(portfolio)
        if carried_forward_low_margin or top_after_reducer == current
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

    action = str(payload.get("action") or "stop").strip().lower()
    if action not in {"dispatch", "finalize", "stop"}:
        action = "stop"
    if stage == "planner_final_calibration":
        action = "finalize"
    dispatch_requests = [
        _normalize_dispatch_payload(item, round_id=round_id)
        for item in _as_dict_list(payload.get("dispatch_requests"))
    ]
    if not dispatch_requests:
        dispatch_requests = _dispatches_from_capability_ids(
            _selected_capability_ids(payload),
            round_id=round_id,
            capability_registry=capability_registry,
        )
    dispatch_requests = _merge_coverage_dispatches(
        dispatch_requests,
        payload_context=prompt_payload or {},
        round_id=round_id,
        capability_registry=capability_registry,
    )
    if action in {"finalize", "stop"} and stage != "planner_final_calibration":
        coverage_dispatches = _dispatches_from_coverage_memo(
            payload_context=prompt_payload or {},
            round_id=round_id,
            capability_registry=capability_registry,
        )
        if coverage_dispatches:
            dispatch_requests = _merge_dispatch_lists(
                coverage_dispatches,
                dispatch_requests,
            )
            action = "dispatch"
    if (
        action == "finalize"
        and stage != "planner_final_calibration"
        and _needs_discriminating_followup(prompt_payload or {})
    ):
        followup_dispatches = _dispatches_from_capability_ids(
            _next_discriminating_followup_capability_ids(prompt_payload or {}),
            round_id=round_id,
            capability_registry=capability_registry,
        )
        if followup_dispatches:
            dispatch_requests = _merge_dispatch_lists(
                dispatch_requests,
                followup_dispatches,
            )
            action = "dispatch"
    if stage == "planner_initial":
        coverage_dispatches = _dispatches_from_coverage_memo(
            payload_context=prompt_payload or {},
            round_id=round_id,
            capability_registry=capability_registry,
        )
        if coverage_dispatches:
            dispatch_requests = coverage_dispatches
            action = "dispatch"
    if (
        action == "stop"
        and stage != "planner_final_calibration"
        and not dispatch_requests
        and not _portfolio_is_placeholder(portfolio)
    ):
        # Exhausting the capabilities available in the current workflow is a
        # bounded scientific conclusion, not a runtime failure. Typed/schema
        # failures bypass this normalizer and remain true stop decisions.
        action = "finalize"
    final_answer_draft = payload.get("final_answer_draft")
    if _is_nullish(final_answer_draft):
        final_answer_draft = None
    if action == "finalize" and final_answer_draft is None:
        final_answer_draft = (
            "Planner synthesis completed from public runtime evidence. "
            "Mechanism candidates are ranked by differential priority and claims "
            "remain bounded by their evidence support."
        )

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
        "action": action,
        "dispatch_requests": dispatch_requests if action == "dispatch" else [],
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
        differential_priority = _float_between(
            item.get("differential_priority")
            or item.get("priority")
            or item.get("differential_score"),
            default=confidence,
        )
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


def _merge_incremental_portfolio_payload(
    portfolio: dict[str, object],
    previous_hypotheses: list[dict[str, object]],
) -> dict[str, object]:
    if not previous_hypotheses:
        return portfolio
    if _portfolio_is_placeholder(portfolio):
        return portfolio
    merged = {
        _hypothesis_name(item): _normalize_hypothesis_payload(item)
        for item in previous_hypotheses
        if _hypothesis_name(item)
    }
    for item in _as_dict_list(portfolio.get("hypotheses")):
        name = _hypothesis_name(item)
        if not name:
            continue
        previous = merged.get(name, {})
        merged[name] = {
            **previous,
            **item,
            "evidence_refs": _merged_text_list(
                previous.get("evidence_refs"),
                item.get("evidence_refs"),
                limit=8,
            ),
            "validation_needed": _merged_text_list(
                previous.get("validation_needed"),
                item.get("validation_needed"),
                limit=6,
            ),
        }
    previous_order = {
        _hypothesis_name(item): index
        for index, item in enumerate(previous_hypotheses)
        if _hypothesis_name(item)
    }
    ordered_names = sorted(
        [name for name in merged if name != "unknown"],
        key=lambda name: (
            _float_between(
                merged[name].get("differential_priority"),
                default=_float_between(merged[name].get("confidence"), default=0.0),
            ),
            -previous_order.get(name, len(previous_order)),
        ),
        reverse=True,
    )
    hypotheses = [_normalize_hypothesis_payload(merged[name]) for name in ordered_names]
    if not hypotheses:
        hypotheses = list(_as_dict_list(portfolio.get("hypotheses")))
    current = _short_text(portfolio.get("current"), 120)
    names = {str(item.get("name")) for item in hypotheses}
    if current not in names and hypotheses:
        current = str(
            max(
                hypotheses,
                key=lambda item: _float_between(
                    item.get("differential_priority"),
                    default=_float_between(item.get("confidence"), default=0.0),
                ),
            ).get("name")
        )
    return {**portfolio, "current": current or "unknown", "hypotheses": hypotheses}


def _merge_final_calibration_portfolio_payload(
    portfolio: dict[str, object],
    previous_hypotheses: list[dict[str, object]],
) -> dict[str, object]:
    if not previous_hypotheses or _portfolio_is_placeholder(portfolio):
        return portfolio
    previous = {
        _hypothesis_name(item): _normalize_hypothesis_payload(item)
        for item in previous_hypotheses
        if _hypothesis_name(item)
    }
    calibrated = {
        _hypothesis_name(item): _normalize_hypothesis_payload(item)
        for item in _as_dict_list(portfolio.get("hypotheses"))
        if _hypothesis_name(item)
    }
    merged: dict[str, dict[str, object]] = {}
    for label in [label for label in MECHANISM_POOL if label in previous or label in calibrated]:
        old = previous.get(label, {})
        new = calibrated.get(label, {})
        if new:
            merged[label] = {
                **old,
                **new,
                "evidence_refs": _merged_text_list(
                    old.get("evidence_refs"),
                    new.get("evidence_refs"),
                    limit=8,
                ),
                "validation_needed": _merged_text_list(
                    old.get("validation_needed"),
                    new.get("validation_needed"),
                    limit=6,
                ),
            }
            continue
        merged[label] = old
    for label, item in calibrated.items():
        if label not in merged and label != "unknown":
            merged[label] = item
    return _sort_portfolio_by_priority(
        {
            **portfolio,
            "hypotheses": [_normalize_hypothesis_payload(item) for item in merged.values()],
        }
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


def _apply_priority_deltas_to_portfolio(
    portfolio: dict[str, object],
    previous_hypotheses: list[dict[str, object]],
    priority_deltas: list[dict[str, object]],
) -> dict[str, object]:
    if not previous_hypotheses:
        return portfolio
    if _portfolio_is_placeholder(portfolio) and not priority_deltas:
        return portfolio
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
    delta_by_label = {
        _short_text(item.get("label"), 120): item
        for item in priority_deltas
        if _short_text(item.get("label"), 120)
    }
    merged: dict[str, dict[str, object]] = {}
    for label, previous in previous_by_label.items():
        current = current_by_label.get(label, {})
        delta = delta_by_label.get(label)
        new_priority = (
            _float_between(delta.get("new_priority"), default=_priority_for_payload(previous))
            if delta
            else _priority_for_payload(current or previous)
        )
        delta_refs = _text_list(delta.get("evidence_refs_added"), limit=8) if delta else []
        merged[label] = {
            **previous,
            **current,
            "confidence": new_priority,
            "differential_priority": new_priority,
            "evidence_refs": _merged_text_list(
                _merged_text_list(
                    previous.get("evidence_refs"),
                    current.get("evidence_refs"),
                    limit=8,
                ),
                delta_refs,
                limit=8,
            ),
            "validation_needed": _merged_text_list(
                previous.get("validation_needed"),
                current.get("validation_needed"),
                limit=6,
            ),
        }
    for label, current in current_by_label.items():
        if label not in merged:
            merged[label] = current
    return _portfolio_from_merged_items(portfolio, merged, previous_hypotheses)


def _apply_low_margin_carry_forward(
    portfolio: dict[str, object],
    *,
    round_id: str,
    previous_portfolio_hypotheses: list[dict[str, object]],
    priority_deltas: list[dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    if not _previous_order_is_scientific_state(previous_portfolio_hypotheses):
        return portfolio, priority_deltas, []
    previous_order = {
        _hypothesis_name(item): index
        for index, item in enumerate(previous_portfolio_hypotheses)
        if _hypothesis_name(item)
    }
    delta_by_label = {
        _short_text(item.get("label"), 120): item
        for item in priority_deltas
        if _short_text(item.get("label"), 120)
    }
    items = {
        _hypothesis_name(item): _normalize_hypothesis_payload(item)
        for item in _as_dict_list(portfolio.get("hypotheses"))
        if _hypothesis_name(item)
    }
    notes: list[dict[str, object]] = []
    changed = True
    while changed:
        changed = False
        ordered = sorted(
            items.values(),
            key=lambda item: (
                _priority_for_payload(item),
                -previous_order.get(_hypothesis_name(item), len(previous_order)),
            ),
            reverse=True,
        )
        for upper, lower in zip(ordered, ordered[1:], strict=False):
            upper_label = _hypothesis_name(upper)
            lower_label = _hypothesis_name(lower)
            if upper_label not in previous_order or lower_label not in previous_order:
                continue
            upper_priority = _priority_for_payload(upper)
            lower_priority = _priority_for_payload(lower)
            margin = upper_priority - lower_priority
            if not 0.0 <= margin < LOW_MARGIN_THRESHOLD:
                continue
            if previous_order[lower_label] >= previous_order[upper_label]:
                notes.append(
                    _low_margin_note(upper_label, lower_label, margin, "previous_order_kept")
                )
                continue
            if _delta_supports_reversal(
                promoted_label=upper_label,
                demoted_label=lower_label,
                delta_by_label=delta_by_label,
            ):
                notes.append(
                    _low_margin_note(upper_label, lower_label, margin, "reversal_supported")
                )
                continue
            if _delta_is_priority_demotion(delta_by_label.get(lower_label)):
                notes.append(
                    _low_margin_note(
                        upper_label,
                        lower_label,
                        margin,
                        "evidence_cap_preserved",
                    )
                )
                continue
            adjusted_priority = min(1.0, upper_priority + LOW_MARGIN_STABLE_SEPARATION)
            items[lower_label] = {
                **lower,
                "confidence": adjusted_priority,
                "differential_priority": adjusted_priority,
                "rationale": _append_stability_rationale(
                    lower.get("rationale"),
                    upper_label,
                    margin,
                ),
            }
            delta_by_label[lower_label] = _stability_adjusted_delta(
                delta_by_label.get(lower_label),
                label=lower_label,
                round_id=round_id,
                previous_hypotheses=previous_portfolio_hypotheses,
                new_priority=adjusted_priority,
                competitor_label=upper_label,
                margin=margin,
            )
            notes.append(
                _low_margin_note(
                    lower_label,
                    upper_label,
                    margin,
                    "previous_order_carried_forward",
                )
            )
            changed = True
            break
    updated_deltas = [
        MechanismPriorityDelta.model_validate(item).model_dump(mode="json")
        for item in delta_by_label.values()
    ]
    return (
        _portfolio_from_merged_items(portfolio, items, previous_portfolio_hypotheses),
        sorted(
            updated_deltas,
            key=lambda item: abs(float(item["priority_delta"])),
            reverse=True,
        ),
        notes,
    )


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


def _apply_evidence_tier_priority_floor(
    portfolio: dict[str, object],
    priority_deltas: list[dict[str, object]],
    *,
    round_id: str,
    prompt_payload: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    hygiene = prompt_payload.get("priority_hygiene_summary")
    if not isinstance(hygiene, list):
        return portfolio, priority_deltas
    directness_by_label = {
        _short_text(item.get("label"), 120): item
        for item in hygiene
        if isinstance(item, dict) and _short_text(item.get("label"), 120)
    }
    if not directness_by_label:
        return portfolio, priority_deltas
    hypotheses = [
        _normalize_hypothesis_payload(item)
        for item in _as_dict_list(portfolio.get("hypotheses"))
        if _hypothesis_name(item)
    ]
    if not hypotheses:
        return portfolio, priority_deltas
    hypotheses, boundary_deltas = _cap_boundary_only_hypotheses(
        hypotheses,
        directness_by_label=directness_by_label,
        round_id=round_id,
    )
    hypotheses, generic_deltas = _cap_generic_structural_hypotheses(
        hypotheses,
        directness_by_label=directness_by_label,
        round_id=round_id,
    )
    hypotheses, standalone_oscillator_deltas = _cap_standalone_oscillator_hypotheses(
        hypotheses,
        directness_by_label=directness_by_label,
        round_id=round_id,
        prompt_payload=prompt_payload,
    )
    hypotheses, absent_signal_deltas = _cap_absent_positive_signal_hypotheses(
        hypotheses,
        directness_by_label=directness_by_label,
        round_id=round_id,
    )
    hypotheses, generic_packing_deltas = _cap_generic_packing_proxy_hypotheses(
        hypotheses,
        directness_by_label=directness_by_label,
        round_id=round_id,
    )
    delta_by_label = {
        _short_text(delta.get("label"), 120): delta
        for delta in [
            *priority_deltas,
            *boundary_deltas,
            *generic_deltas,
            *standalone_oscillator_deltas,
            *absent_signal_deltas,
            *generic_packing_deltas,
        ]
        if _short_text(delta.get("label"), 120)
    }
    updated = bool(
        boundary_deltas
        or generic_deltas
        or standalone_oscillator_deltas
        or absent_signal_deltas
        or generic_packing_deltas
    )
    updated_hypotheses: list[dict[str, object]] = []
    for item in hypotheses:
        label = _hypothesis_name(item)
        directness = directness_by_label.get(label)
        if not _has_priority_floor_signal(directness):
            updated_hypotheses.append(item)
            continue
        if _is_screened_or_contradicted_hypothesis(item):
            updated_hypotheses.append(item)
            continue
        floor = _priority_floor_for_directness(directness, prompt_payload=prompt_payload)
        previous_priority = _priority_for_payload(item)
        if previous_priority >= floor or floor <= 0.0:
            updated_hypotheses.append(item)
            continue
        new_priority = round(floor, 6)
        updated_item = {
            **item,
            "confidence": new_priority,
            "differential_priority": new_priority,
            "rationale": _append_short_note(
                item.get("rationale"),
                (
                    "Priority floor applied because public runtime evidence includes "
                    "a computed or mechanism-specific structural proxy; support "
                    "remains bounded as weak/proxy."
                ),
            ),
        }
        updated_hypotheses.append(updated_item)
        evidence_refs = [
            str(hit.get("evidence_id"))
            for hit in _as_dict_list(
                directness.get("route_hits") if isinstance(directness, dict) else None
            )
            if str(hit.get("evidence_id") or "").strip()
        ]
        evidence_refs = list(
            dict.fromkeys(
                [
                    *evidence_refs,
                    *(
                        _text_list(directness.get("evidence_refs"), limit=8)
                        if isinstance(directness, dict)
                        else []
                    ),
                ]
            )
        )
        delta_by_label[label] = {
            **delta_by_label.get(label, {}),
            "round_id": round_id,
            "label": label,
            "previous_priority": previous_priority,
            "priority_delta": round(new_priority - previous_priority, 6),
            "new_priority": new_priority,
            "delta_reason": (
                "Applied generic evidence-tier calibration: public runtime "
                "evidence gives this mechanism a bounded minimum differential "
                "priority without turning proxy support into a strong claim."
            ),
            "evidence_refs_added": evidence_refs[:4],
            "support_change": "unchanged",
            "ranking_stability_note": (
                "Evidence-tier floor affects candidate priority only; it does not "
                "upgrade evidence support or create a final diagnosis."
            ),
        }
        updated = True
    if not updated:
        return portfolio, priority_deltas
    return (
        _sort_portfolio_by_priority({**portfolio, "hypotheses": updated_hypotheses}),
        list(delta_by_label.values()),
    )


def _priority_floor_for_directness(
    directness: dict[str, object] | None,
    *,
    prompt_payload: dict[str, object],
) -> float:
    if not isinstance(directness, dict):
        return 0.0
    if _has_only_boundary_or_failed_signal(directness):
        return 0.0
    if int(directness.get("computed_direct_proxy_count") or 0) > 0:
        oscillator_values = []
        only_baseline_bundle = True
        for hit in _as_dict_list(directness.get("route_hits")):
            if str(hit.get("signal_type") or "") in {
                "missing_or_failed",
                "checklist_boundary_only",
            }:
                continue
            if str(hit.get("capability_id") or "") != "microscopic.run_baseline_bundle":
                only_baseline_bundle = False
            key_metrics = hit.get("key_metrics")
            if not isinstance(key_metrics, dict):
                continue
            value = _number_like(key_metrics.get("oscillator_strength"))
            if value is not None:
                oscillator_values.append(value)
        if only_baseline_bundle and _payload_has_esipt_context(prompt_payload):
            return COUPLED_OSCILLATOR_PRIORITY_FLOOR
        if only_baseline_bundle:
            return _priority_from_oscillator_values(oscillator_values)
        if oscillator_values and max(oscillator_values) < 0.02:
            return MODERATE_STANDALONE_OSCILLATOR_PRIORITY
        return COMPUTED_DIRECT_PRIORITY_FLOOR
    artifact_ct_floor = _artifact_backed_ct_priority_floor(directness, prompt_payload)
    if artifact_ct_floor > 0.0:
        return artifact_ct_floor
    if int(directness.get("mechanism_specific_structural_count") or 0) > 0:
        return MECHANISM_SPECIFIC_STRUCTURAL_PRIORITY_FLOOR
    return 0.0


def _cap_standalone_oscillator_hypotheses(
    hypotheses: list[dict[str, object]],
    *,
    directness_by_label: dict[str, dict[str, object]],
    round_id: str,
    prompt_payload: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if _payload_has_esipt_context(prompt_payload):
        return hypotheses, []
    updated: list[dict[str, object]] = []
    deltas: list[dict[str, object]] = []
    for item in hypotheses:
        label = _hypothesis_name(item)
        directness = directness_by_label.get(label)
        if label != "RADIATIVE_RATE_STATE_BALANCE" or not _is_standalone_baseline_bundle(
            directness
        ):
            updated.append(item)
            continue
        previous_priority = _priority_for_payload(item)
        oscillator_values = _oscillator_values_from_directness(directness)
        cap = _priority_from_oscillator_values(oscillator_values)
        new_priority = min(previous_priority, cap)
        if new_priority == previous_priority:
            updated.append(item)
            continue
        updated.append(
            {
                **item,
                "confidence": new_priority,
                "differential_priority": new_priority,
                "rationale": _append_short_note(
                    item.get("rationale"),
                    (
                        "Priority capped because baseline oscillator strength alone "
                        "is a bounded radiative-channel proxy, not direct evidence "
                        "that radiative-rate/state-balance is the dominant mechanism."
                    ),
                ),
            }
        )
        evidence_refs = [
            str(hit.get("evidence_id"))
            for hit in _as_dict_list(
                directness.get("route_hits") if isinstance(directness, dict) else None
            )
            if str(hit.get("evidence_id") or "").strip()
        ]
        deltas.append(
            {
                "round_id": round_id,
                "label": label,
                "previous_priority": previous_priority,
                "priority_delta": round(new_priority - previous_priority, 6),
                "new_priority": new_priority,
                "delta_reason": (
                    "Capped standalone oscillator-strength proxy: without coupled "
                    "mechanism-specific context or a dedicated bright/dark route, "
                    "it should not outrank stronger structural mechanism triggers."
                ),
                "evidence_refs_added": evidence_refs[:4],
                "support_change": "unchanged",
                "ranking_stability_note": (
                    "Standalone oscillator cap affects candidate priority only; "
                    "it does not downgrade the recorded evidence support score."
                ),
            }
        )
    return updated, deltas


def _cap_absent_positive_signal_hypotheses(
    hypotheses: list[dict[str, object]],
    *,
    directness_by_label: dict[str, dict[str, object]],
    round_id: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not directness_by_label:
        return hypotheses, []
    updated: list[dict[str, object]] = []
    deltas: list[dict[str, object]] = []
    for item in hypotheses:
        label = _hypothesis_name(item)
        if _is_screened_or_contradicted_hypothesis(item):
            updated.append(item)
            continue
        directness = directness_by_label.get(label)
        if (
            label == "AGGREGATE_EXCITON_EXCIMER"
            and _as_dict_list(
                directness.get("route_hits") if isinstance(directness, dict) else None
            )
            and not _has_aggregate_exciton_directness(directness)
        ):
            previous_priority = _priority_for_payload(item)
            new_priority = min(
                previous_priority,
                min(UNSUPPORTED_PRIOR_PRIORITY_CEILING, GENERIC_STRUCTURAL_PRIORITY_CEILING),
            )
            if new_priority < previous_priority:
                updated.append(
                    {
                        **item,
                        "confidence": new_priority,
                        "differential_priority": new_priority,
                        "rationale": _append_short_note(
                            item.get("rationale"),
                            (
                                "Priority capped because aggregate/exciton claims "
                                "need aggregate-state electronic, spectral, or "
                                "dimer excited-state evidence; generic contact or "
                                "hydrophobic aggregation proxies are treated as "
                                "packing/confinement evidence instead."
                            ),
                        ),
                    }
                )
                deltas.append(
                    {
                        "round_id": round_id,
                        "label": label,
                        "previous_priority": previous_priority,
                        "priority_delta": round(new_priority - previous_priority, 6),
                        "new_priority": new_priority,
                        "delta_reason": (
                            "Aggregate/exciton priority capped because no "
                            "aggregate-state electronic, spectral, or dimer "
                            "excited-state proxy was present."
                        ),
                        "evidence_refs_added": _text_list(item.get("evidence_refs"), limit=4),
                        "support_change": "weakened",
                        "ranking_stability_note": (
                            "Generic aggregation/contact evidence remains available "
                            "for packing/confinement but not as aggregate-exciton "
                            "support."
                        ),
                    }
                )
                continue
            updated.append(item)
            continue
        if _has_positive_directness(directness_by_label.get(label)):
            updated.append(item)
            continue
        previous_priority = _priority_for_payload(item)
        if previous_priority <= NO_REF_WEAK_PRIORITY_CEILING:
            updated.append(item)
            continue
        evidence_refs = _text_list(item.get("evidence_refs"), limit=8)
        if evidence_refs:
            updated.append(item)
            continue
        if not evidence_refs:
            cap = NO_REF_WEAK_PRIORITY_CEILING
        else:
            cap = UNSUPPORTED_PRIOR_PRIORITY_CEILING
        new_priority = min(previous_priority, cap)
        if new_priority == previous_priority:
            updated.append(item)
            continue
        updated.append(
            {
                **item,
                "confidence": new_priority,
                "differential_priority": new_priority,
                "rationale": _append_short_note(
                    item.get("rationale"),
                    (
                        "Priority capped because the current public evidence "
                        "coverage table contains no positive route hit for this "
                        "mechanism after metric-level screen filtering."
                    ),
                ),
            }
        )
        deltas.append(
            {
                "round_id": round_id,
                "label": label,
                "previous_priority": previous_priority,
                "priority_delta": round(new_priority - previous_priority, 6),
                "new_priority": new_priority,
                "delta_reason": (
                    "No positive public runtime screen remains for this mechanism "
                    "after applying metric-level evidence filtering; retained only "
                    "as a weak candidate."
                ),
                "evidence_refs_added": evidence_refs[:4],
                "support_change": "weakened",
                "ranking_stability_note": (
                    "This cap prevents stale incremental portfolio priority from "
                    "surviving after all positive route hits have been filtered out."
                ),
            }
        )
    return updated, deltas


def _has_positive_directness(directness: dict[str, object] | None) -> bool:
    if not isinstance(directness, dict):
        return False
    return any(
        int(directness.get(key) or 0) > 0
        for key in (
            "computed_direct_proxy_count",
            "mechanism_specific_structural_count",
            "structural_or_electronic_weak_count",
            "critic_positive_ref_count",
        )
    )


def _oscillator_values_from_directness(
    directness: dict[str, object] | None,
) -> list[float]:
    if not isinstance(directness, dict):
        return []
    values: list[float] = []
    for hit in _as_dict_list(directness.get("route_hits")):
        if str(hit.get("signal_type") or "") in {
            "missing_or_failed",
            "checklist_boundary_only",
        }:
            continue
        metrics = _metrics_from_payload_item(hit)
        value = _oscillator_strength_from_metrics(metrics)
        if value is not None:
            values.append(value)
    return values


def _priority_from_oscillator_values(values: list[float]) -> float:
    if not values:
        return STANDALONE_OSCILLATOR_PRIORITY_CEILING
    return _standalone_oscillator_priority({"oscillator_strength": max(values)})


def _has_aggregate_exciton_directness(directness: dict[str, object] | None) -> bool:
    if not isinstance(directness, dict):
        return False
    for hit in _as_dict_list(directness.get("route_hits")):
        metrics = _metrics_from_payload_item(hit)
        if (
            metrics.get("aggregate_exciton_proxy")
            or metrics.get("excimer_geometry_proxy")
            or metrics.get("spectral_new_band_proxy")
        ):
            return True
    return False


def _has_weak_polar_only_directness(
    label: str,
    directness: dict[str, object] | None,
) -> bool:
    if label not in {"HOST_GUEST_INTERACTION", "PET_ET"} or not isinstance(
        directness, dict
    ):
        return False
    positive_hits = [
        hit
        for hit in _as_dict_list(directness.get("route_hits"))
        if str(hit.get("signal_type") or "")
        not in {"missing_or_failed", "checklist_boundary_only"}
    ]
    if not positive_hits:
        return False
    for hit in positive_hits:
        if str(hit.get("capability_id") or "") != "macro.screen_polar_binding_site_prior":
            return False
        priority = _polar_screening_priority(label, _metrics_from_payload_item(hit))
        if priority > GENERIC_STRUCTURAL_PRIORITY_CEILING:
            return False
    return True


def _has_sulfur_only_triplet_directness(directness: dict[str, object] | None) -> bool:
    if not isinstance(directness, dict):
        return False
    positive_hits = [
        hit
        for hit in _as_dict_list(directness.get("route_hits"))
        if str(hit.get("signal_type") or "")
        not in {"missing_or_failed", "checklist_boundary_only"}
    ]
    if not positive_hits:
        return False
    for hit in positive_hits:
        if str(hit.get("capability_id") or "") != "macro.screen_metal_triplet_prior":
            return False
        metrics = _metrics_from_payload_item(hit)
        if metrics.get("metal_or_lanthanide_prior") or metrics.get(
            "heavy_atom_triplet_prior"
        ):
            return False
        if not metrics.get("sulfur_phosphorus_triplet_prior"):
            return False
    return True


def _has_artifact_backed_ct_signal(directness: dict[str, object] | None) -> bool:
    return _artifact_backed_ct_priority_floor(directness, {}) > 0.0


def _artifact_backed_ct_priority_floor(
    directness: dict[str, object] | None,
    prompt_payload: dict[str, object],
) -> float:
    if not isinstance(directness, dict):
        return 0.0
    if str(directness.get("label") or "") != "ICT_TICT_CT":
        return 0.0
    has_donor_acceptor = False
    has_ct_artifact = False
    has_informative_ct = False
    donor_acceptor_strength = 0.0
    oscillator_values: list[float] = []
    for hit in _as_dict_list(directness.get("route_hits")):
        capability_id = str(hit.get("capability_id") or "")
        metrics = _metrics_from_payload_item(hit)
        if capability_id == "macro.screen_donor_acceptor_layout" and (
            metrics.get("donor_acceptor_proxy")
            or metrics.get("donor_acceptor_partition_proxy")
            or str(hit.get("evidence_id") or "").strip()
        ):
            has_donor_acceptor = True
            donor_acceptor_strength = max(
                donor_acceptor_strength,
                _number_like(metrics.get("donor_acceptor_proxy")) or 0.0,
            )
        if capability_id == "microscopic.extract_ct_descriptors_from_bundle" and (
            (_number_like(metrics.get("state_count")) or 0.0) > 0.0
        ):
            has_ct_artifact = True
            value = _number_like(metrics.get("oscillator_strength"))
            if value is not None:
                oscillator_values.append(value)
            has_informative_ct = has_informative_ct or _has_informative_ct_metrics(hit)
    if not (has_donor_acceptor and has_ct_artifact):
        return 0.0
    if has_informative_ct:
        return ARTIFACT_BACKED_CT_PRIORITY_FLOOR
    max_oscillator = max(oscillator_values, default=0.0)
    has_competing_esipt = _payload_has_esipt_context(prompt_payload)
    has_competing_environment = _payload_has_environment_context(prompt_payload)
    if max_oscillator >= 0.25 and (
        donor_acceptor_strength >= 2.0
        or not (has_competing_esipt or has_competing_environment)
    ):
        return ARTIFACT_BACKED_CT_PRIORITY_FLOOR
    return WEAK_ARTIFACT_BACKED_CT_PRIORITY_FLOOR


def _payload_has_environment_context(payload: dict[str, object]) -> bool:
    for item in [
        *_as_dict_list(payload.get("evidence")),
        *_as_dict_list(payload.get("new_round_evidence")),
    ]:
        metrics = _metrics_from_payload_item(item)
        if (
            metrics.get("aggregation_prone_proxy")
            or metrics.get("host_guest_followup_trigger")
            or metrics.get("polar_binding_site_proxy")
            or metrics.get("metal_or_lanthanide_prior")
            or metrics.get("heavy_atom_triplet_prior")
        ):
            return True
    for item in _as_dict_list(payload.get("priority_hygiene_summary")):
        for hit in _as_dict_list(item.get("route_hits")):
            metrics = _metrics_from_payload_item(hit)
            if (
                metrics.get("aggregation_prone_proxy")
                or metrics.get("host_guest_followup_trigger")
                or metrics.get("polar_binding_site_proxy")
                or metrics.get("metal_or_lanthanide_prior")
                or metrics.get("heavy_atom_triplet_prior")
            ):
                return True
    return False


def _is_standalone_baseline_bundle(directness: dict[str, object] | None) -> bool:
    if not isinstance(directness, dict):
        return False
    route_hits = _as_dict_list(directness.get("route_hits"))
    positive_hits = [
        hit
        for hit in route_hits
        if str(hit.get("signal_type") or "")
        not in {"missing_or_failed", "checklist_boundary_only"}
    ]
    return bool(positive_hits) and all(
        str(hit.get("capability_id") or "") == "microscopic.run_baseline_bundle"
        for hit in positive_hits
    )


def _is_screened_or_contradicted_hypothesis(item: dict[str, object]) -> bool:
    status = str(item.get("status") or "").strip().lower()
    trigger_status = str(item.get("trigger_status") or "").strip().lower()
    claim_status = str(item.get("claim_status") or "").strip().lower()
    return (
        status in {"screened", "screened_out", "contradicted", "dropped"}
        or trigger_status in {"screened_out", "contradicted"}
        or claim_status in {"screened_out", "contradicted", "weakened"}
    )


def _cap_boundary_only_hypotheses(
    hypotheses: list[dict[str, object]],
    *,
    directness_by_label: dict[str, dict[str, object]],
    round_id: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    updated: list[dict[str, object]] = []
    deltas: list[dict[str, object]] = []
    for item in hypotheses:
        label = _hypothesis_name(item)
        directness = directness_by_label.get(label)
        if not _has_only_boundary_or_failed_signal(directness):
            updated.append(item)
            continue
        previous_priority = _priority_for_payload(item)
        new_priority = min(previous_priority, 0.05)
        if new_priority == previous_priority:
            updated.append(item)
            continue
        updated.append(
            {
                **item,
                "confidence": new_priority,
                "differential_priority": new_priority,
                "rationale": _append_short_note(
                    item.get("rationale"),
                    (
                        "Priority capped because cited runtime evidence is only "
                        "failed or boundary-status evidence, not positive support."
                    ),
                ),
            }
        )
        deltas.append(
            {
                "round_id": round_id,
                "label": label,
                "previous_priority": previous_priority,
                "priority_delta": round(new_priority - previous_priority, 6),
                "new_priority": new_priority,
                "delta_reason": (
                    "Boundary-only or failed runtime route cannot support a high "
                    "differential priority."
                ),
                "evidence_refs_added": _text_list(
                    directness.get("evidence_refs") if isinstance(directness, dict) else [],
                    limit=4,
                ),
                "support_change": "weakened",
                "ranking_stability_note": (
                    "Failed or boundary-only evidence records missing support; it "
                    "does not count as positive mechanism evidence."
                ),
            }
        )
    return updated, deltas


def _cap_generic_structural_hypotheses(
    hypotheses: list[dict[str, object]],
    *,
    directness_by_label: dict[str, dict[str, object]],
    round_id: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not any(
        _has_priority_floor_signal(directness)
        for directness in directness_by_label.values()
        if isinstance(directness, dict)
    ):
        return hypotheses, []
    updated: list[dict[str, object]] = []
    deltas: list[dict[str, object]] = []
    for item in hypotheses:
        label = _hypothesis_name(item)
        directness = directness_by_label.get(label)
        if not _is_generic_or_unresolved_candidate(item, directness):
            updated.append(item)
            continue
        if _is_screened_or_contradicted_hypothesis(item):
            updated.append(item)
            continue
        previous_priority = _priority_for_payload(item)
        cap = GENERIC_STRUCTURAL_PRIORITY_CEILING
        if _has_weak_polar_only_directness(label, directness):
            cap = 0.19
        new_priority = min(previous_priority, cap)
        if new_priority == previous_priority:
            updated.append(item)
            continue
        support_change = (
            "weakened"
            if label == "AGGREGATE_EXCITON_EXCIMER"
            and not _has_aggregate_exciton_directness(directness)
            else "unchanged"
        )
        updated.append(
            {
                **item,
                "confidence": new_priority,
                "differential_priority": new_priority,
                "rationale": _append_short_note(
                    item.get("rationale"),
                    (
                        "Priority capped because the cited evidence is a generic "
                        "structural/electronic screening trigger while other "
                        "candidates have computed or mechanism-specific proxy "
                        "evidence."
                    ),
                ),
            }
        )
        deltas.append(
            {
                "round_id": round_id,
                "label": label,
                "previous_priority": previous_priority,
                "priority_delta": round(new_priority - previous_priority, 6),
                "new_priority": new_priority,
                "delta_reason": (
                    "Applied generic evidence-tier calibration: broad structural "
                    "or electronic screening triggers should not dominate "
                    "computed or mechanism-specific proxy evidence in "
                    "differential ranking."
                ),
                "evidence_refs_added": _text_list(
                    directness.get("evidence_refs") if isinstance(directness, dict) else [],
                    limit=4,
                ),
                "support_change": support_change,
                "ranking_stability_note": (
                    "Generic-prior cap affects candidate priority only; it does "
                    "not weaken the cited evidence or change support scoring."
                ),
            }
        )
    return updated, deltas


def _is_generic_or_unresolved_candidate(
    item: dict[str, object],
    directness: dict[str, object] | None,
) -> bool:
    label = _hypothesis_name(item)
    if _priority_for_payload(item) <= 0.0:
        return False
    if not isinstance(directness, dict):
        return not _text_list(item.get("evidence_refs"), limit=1)
    item_refs = set(_text_list(item.get("evidence_refs"), limit=12))
    if _has_weak_polar_only_directness(label, directness):
        return True
    if item_refs:
        for hit in _as_dict_list(directness.get("route_hits")):
            evidence_id = _short_text(hit.get("evidence_id"), 220)
            if evidence_id not in item_refs:
                continue
            if str(hit.get("evidence_tier") or "") in {
                "computed_direct_proxy",
                "mechanism_specific_structural_trigger",
            }:
                return False
            if _is_material_followup_capability(str(hit.get("capability_id") or "")):
                return False
            if label == "TRIPLET_METAL_ENERGY_TRANSFER" and _has_sulfur_only_triplet_directness(
                directness
            ):
                return False
            if _is_mechanism_specific_structural_route(
                label,
                str(hit.get("capability_id") or ""),
            ):
                return False
    return (
        int(directness.get("computed_direct_proxy_count") or 0) == 0
        and int(directness.get("mechanism_specific_structural_count") or 0) == 0
        and (
            int(directness.get("structural_or_electronic_weak_count") or 0) > 0
            or not _text_list(item.get("evidence_refs"), limit=1)
        )
    )


def _is_mechanism_specific_structural_route(label: str, capability_id: str) -> bool:
    return (
        label,
        capability_id,
    ) in {
        ("RIM_RIR_RIV", "macro.screen_rotor_torsion_topology"),
        ("RIM_RIR_RIV", "macro.screen_rotor_rim_prior"),
        ("HOST_GUEST_INTERACTION", "macro.screen_polar_binding_site_prior"),
        ("PET_ET", "macro.screen_polar_binding_site_prior"),
        ("TRIPLET_METAL_ENERGY_TRANSFER", "macro.screen_metal_triplet_prior"),
    }


def _cap_generic_packing_proxy_hypotheses(
    hypotheses: list[dict[str, object]],
    *,
    directness_by_label: dict[str, dict[str, object]],
    round_id: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    updated: list[dict[str, object]] = []
    deltas: list[dict[str, object]] = []
    for item in hypotheses:
        label = _hypothesis_name(item)
        directness = directness_by_label.get(label)
        if label != "PACKING_HOST_MATRIX_CONFINEMENT" or not _is_generic_packing_directness(
            directness
        ):
            updated.append(item)
            continue
        if _is_screened_or_contradicted_hypothesis(item):
            updated.append(item)
            continue
        previous_priority = _priority_for_payload(item)
        new_priority = min(previous_priority, GENERIC_PACKING_PROXY_PRIORITY_CEILING)
        if new_priority == previous_priority:
            updated.append(item)
            continue
        updated.append(
            {
                **item,
                "confidence": new_priority,
                "differential_priority": new_priority,
                "rationale": _append_short_note(
                    item.get("rationale"),
                    (
                        "Priority capped because generic aggregation or solid-state "
                        "screening is a broad packing/confinement proxy; it should "
                        "not dominate mechanism-specific structural or computed "
                        "proxies without direct packing/framework evidence."
                    ),
                ),
            }
        )
        deltas.append(
            {
                "round_id": round_id,
                "label": label,
                "previous_priority": previous_priority,
                "priority_delta": round(new_priority - previous_priority, 6),
                "new_priority": new_priority,
                "delta_reason": (
                    "Capped generic packing/confinement proxy: aggregation or "
                    "solid-state propensity alone remains useful for screening, "
                    "but direct packing/framework/dimer evidence is needed for a "
                    "higher differential priority."
                ),
                "evidence_refs_added": _text_list(item.get("evidence_refs"), limit=4),
                "support_change": "unchanged",
                "ranking_stability_note": (
                    "This cap affects candidate ranking only; it does not change "
                    "the evidence support score."
                ),
            }
        )
    return updated, deltas


def _is_generic_packing_directness(directness: dict[str, object] | None) -> bool:
    if not isinstance(directness, dict):
        return False
    positive_hits = [
        hit
        for hit in _as_dict_list(directness.get("route_hits"))
        if str(hit.get("signal_type") or "")
        not in {"missing_or_failed", "checklist_boundary_only"}
    ]
    if not positive_hits:
        return False
    checked = False
    for hit in positive_hits:
        capability_id = str(hit.get("capability_id") or "")
        metrics = _metrics_from_payload_item(hit)
        if capability_id in {
            "macro.run_dimer_packing_proxy",
            "macro.run_aggregate_contact_proxy",
        }:
            return False
        if capability_id in {
            "macro.screen_aggregation_prone_scaffold",
            "macro.run_solid_state_emission_proxy",
        } and metrics.get("aggregation_prone_proxy"):
            checked = True
            continue
        return False
    return checked


def _has_higher_directness_signal(directness: dict[str, object] | None) -> bool:
    if not isinstance(directness, dict):
        return False
    return (
        int(directness.get("computed_direct_proxy_count") or 0) > 0
        or int(directness.get("mechanism_specific_structural_count") or 0) > 0
        or int(directness.get("critic_positive_ref_count") or 0) > 0
    )


def _has_priority_floor_signal(
    directness: dict[str, object] | None,
) -> bool:
    if not isinstance(directness, dict):
        return False
    if _has_only_boundary_or_failed_signal(directness):
        return False
    return (
        int(directness.get("computed_direct_proxy_count") or 0) > 0
        or int(directness.get("mechanism_specific_structural_count") or 0) > 0
        or _has_artifact_backed_ct_signal(directness)
    )


def _has_only_boundary_or_failed_signal(
    directness: dict[str, object] | None,
) -> bool:
    if not isinstance(directness, dict):
        return False
    return (
        int(directness.get("computed_direct_proxy_count") or 0) == 0
        and int(directness.get("mechanism_specific_structural_count") or 0) == 0
        and int(directness.get("structural_or_electronic_weak_count") or 0) == 0
        and int(directness.get("boundary_or_failed_count") or 0) > 0
    )


def _append_short_note(value: object, note: str) -> str:
    text = _short_text(value, 280)
    if not text:
        return _short_text(note, 360)
    if note in text:
        return text
    return _short_text(f"{text} {note}", 360)


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


def _portfolio_from_merged_items(
    portfolio: dict[str, object],
    merged: dict[str, dict[str, object]],
    previous_hypotheses: list[dict[str, object]],
) -> dict[str, object]:
    previous_order = {
        _hypothesis_name(item): index
        for index, item in enumerate(previous_hypotheses)
        if _hypothesis_name(item)
    }
    ordered_names = sorted(
        [name for name in merged if name != "unknown"],
        key=lambda name: (
            _priority_for_payload(merged[name]),
            -previous_order.get(name, len(previous_order)),
        ),
        reverse=True,
    )
    hypotheses = [_normalize_hypothesis_payload(merged[name]) for name in ordered_names]
    names = {str(item.get("name")) for item in hypotheses}
    current = _short_text(portfolio.get("current"), 120)
    if current not in names and hypotheses:
        current = str(hypotheses[0].get("name"))
    runner_up = _short_text(portfolio.get("runner_up"), 120)
    if runner_up not in names:
        runner_up = str(hypotheses[1].get("name")) if len(hypotheses) > 1 else ""
    return {
        **portfolio,
        "current": current or "unknown",
        "runner_up": runner_up or None,
        "hypotheses": hypotheses,
    }


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


def _previous_order_is_scientific_state(previous_hypotheses: list[dict[str, object]]) -> bool:
    for item in previous_hypotheses:
        if _hypothesis_name(item) == "unknown":
            continue
        if _priority_for_payload(item) > 0.0005:
            return True
        if _text_list(item.get("evidence_refs"), limit=1):
            return True
        status = _normalize_hypothesis_status(item.get("status"))
        if status not in {"pending", "unknown"}:
            return True
    return False


def _delta_supports_reversal(
    *,
    promoted_label: str,
    demoted_label: str,
    delta_by_label: dict[str, dict[str, object]],
) -> bool:
    promoted = delta_by_label.get(promoted_label, {})
    demoted = delta_by_label.get(demoted_label, {})
    promoted_delta = _signed_float_between(promoted.get("priority_delta"), default=0.0)
    demoted_delta = _signed_float_between(demoted.get("priority_delta"), default=0.0)
    promoted_support = _normalize_support_change(promoted.get("support_change"))
    demoted_support = _normalize_support_change(demoted.get("support_change"))
    promoted_refs = _text_list(promoted.get("evidence_refs_added"), limit=1)
    if promoted_support in {"upgraded", "newly_supported"}:
        return True
    if demoted_support in {"downgraded", "weakened"} and promoted_delta >= 0.0:
        return True
    if promoted_refs and promoted_delta > 0.0:
        return True
    return bool(promoted_refs) and promoted_delta - demoted_delta >= LOW_MARGIN_THRESHOLD


def _delta_is_priority_demotion(delta: dict[str, object] | None) -> bool:
    if not isinstance(delta, dict):
        return False
    return _signed_float_between(delta.get("priority_delta"), default=0.0) < -0.0005


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


def _append_stability_rationale(
    rationale: object,
    competitor_label: str,
    margin: float,
) -> str:
    base = _short_text(rationale, 300)
    note = (
        f"Low-margin carry-forward retained previous relative order against "
        f"{competitor_label} (margin {margin:.3f}) because no clear new evidence "
        "supported reversal."
    )
    if not base:
        return note
    if "Low-margin carry-forward" in base:
        return base
    return _short_text(f"{base} {note}", 360)


def _stability_adjusted_delta(
    existing_delta: dict[str, object] | None,
    *,
    label: str,
    round_id: str,
    previous_hypotheses: list[dict[str, object]],
    new_priority: float,
    competitor_label: str,
    margin: float,
) -> dict[str, object]:
    previous_by_label = {
        _hypothesis_name(item): _normalize_hypothesis_payload(item)
        for item in previous_hypotheses
        if _hypothesis_name(item)
    }
    previous_priority = _priority_for_payload(previous_by_label.get(label, {}))
    base = dict(existing_delta or {})
    note = (
        f"Low-margin carry-forward retained previous relative order against "
        f"{competitor_label}; observed margin was {margin:.3f}."
    )
    return {
        "round_id": _short_text(base.get("round_id"), 80) or round_id,
        "label": label,
        "previous_priority": round(previous_priority, 6),
        "priority_delta": round(new_priority - previous_priority, 6),
        "new_priority": round(new_priority, 6),
        "delta_reason": _short_text(
            base.get("delta_reason")
            or _inferred_delta_reason(
                label,
                previous_priority=previous_priority,
                new_priority=new_priority,
            ),
            700,
        ),
        "evidence_refs_added": _text_list(base.get("evidence_refs_added"), limit=8),
        "support_change": _normalize_support_change(base.get("support_change")),
        "ranking_stability_note": note,
    }


def _normalize_hypothesis_payload(item: dict[str, object]) -> dict[str, object]:
    name = _hypothesis_name(item) or "unknown"
    confidence = _float_between(item.get("confidence"), default=0.0)
    differential_priority = _float_between(
        item.get("differential_priority")
        or item.get("priority")
        or item.get("differential_score"),
        default=confidence,
    )
    evidence_support = _normalize_prediction_support_strength(
        item.get("evidence_support")
        or item.get("support_strength")
        or item.get("support_level")
    )
    evidence_refs = _text_list(item.get("evidence_refs"), limit=8)
    if evidence_support in {"unsupported", "weak_or_proxy"} and not evidence_refs:
        ceiling = (
            NO_REF_WEAK_PRIORITY_CEILING
            if evidence_support == "weak_or_proxy"
            else UNSUPPORTED_PRIOR_PRIORITY_CEILING
        )
        confidence = min(confidence, ceiling)
        differential_priority = min(differential_priority, ceiling)
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


def _merged_text_list(
    first: object,
    second: object,
    *,
    limit: int,
) -> list[str]:
    return list(dict.fromkeys([*_text_list(first, limit=limit), *_text_list(second, limit=limit)]))[
        :limit
    ]


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


def _ignored_high_urgency_under_screened_coverage(
    decision: PlannerDecision,
    case_run: CaseRun,
    *,
    capability_registry: CapabilityRegistry,
) -> bool:
    coverage = case_run.mechanism_agenda_coverage
    if coverage is None or decision.action != "dispatch":
        return False
    executed_capability_ids = {
        item.capability_id
        for item in case_run.evidence_ledger.items
        if item.capability_id
    }
    urgent_labels = set()
    for item in coverage.agenda_items:
        if (
            item.urgency != "high"
            or item.coverage_status not in {"under_screened", "needs_follow_up"}
        ):
            continue
        capability_id = _capability_id_for_route(
            item.suggested_route,
            capability_registry=capability_registry,
        )
        # A disabled worker route remains an evidence gap in an ablation, but it
        # cannot be a mandatory dispatch target for the remaining workers.
        if capability_id is not None and capability_id not in executed_capability_ids:
            urgent_labels.add(item.label)
    if not urgent_labels:
        return False
    selected_capability_ids = {request.capability_id for request in decision.dispatch_requests}
    selected_text = " ".join(
        [
            *(request.capability_id for request in decision.dispatch_requests),
            *(request.route for request in decision.dispatch_requests),
            *(request.task for request in decision.dispatch_requests),
            *(request.objective for request in decision.dispatch_requests),
        ]
    ).lower()
    for route in coverage.recommended_next_routes:
        target_hit = not route.targets or any(label in urgent_labels for label in route.targets)
        capability_hit = bool(set(route.capability_ids) & selected_capability_ids)
        route_hit = bool(route.route and route.route.lower() in selected_text)
        if target_hit and (capability_hit or route_hit):
            return False
    for item in coverage.agenda_items:
        if item.label not in urgent_labels:
            continue
        route = item.suggested_route.lower()
        if route and route in selected_text:
            return False
    return True


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
    }


def _selected_capability_ids(payload: dict[str, object]) -> list[str]:
    selected: list[str] = []
    for key in (
        "selected_capability_ids",
        "dispatch_capability_ids",
        "next_capability_ids",
    ):
        selected.extend(_text_list(payload.get(key), limit=6))
    selected.extend(
        _short_text(item.get("capability_id"), 160)
        for item in _as_dict_list(payload.get("selected_capabilities"))
    )
    return [item for item in dict.fromkeys(selected) if item]


def _needs_discriminating_followup(payload_context: dict[str, object]) -> bool:
    if not payload_context:
        return False
    available = set(_text_list(payload_context.get("next_capability_ids_available"), limit=30))
    if not available:
        available = _available_capability_ids_from_payload(payload_context)
    if not available:
        return False
    if _has_executed_discriminating_followup(payload_context):
        return False
    if _next_discriminating_followup_capability_ids(payload_context):
        return True
    return False


def _next_discriminating_followup_capability_ids(
    payload_context: dict[str, object],
) -> list[str]:
    available = set(_text_list(payload_context.get("next_capability_ids_available"), limit=30))
    if not available:
        available = _available_capability_ids_from_payload(payload_context)
    labels = _labels_needing_discriminating_followup(payload_context)
    desired: list[str] = []
    if labels & {"RADIATIVE_RATE_STATE_BALANCE", "SOKR_ANTI_KASHA"}:
        desired.append("microscopic.run_bright_dark_state_ordering")
    if labels & {"ICT_TICT_CT", "PET_ET"}:
        desired.extend(
            [
                "microscopic.run_frontier_orbital_partition",
                "microscopic.extract_ct_descriptors_from_bundle",
            ]
        )
    if labels & {"RIM_RIR_RIV", "RACI_CI_ACCESS"}:
        desired.append("microscopic.run_torsion_brightness_coupling_scan")
    if labels & {"AGGREGATE_EXCITON_EXCIMER", "PACKING_HOST_MATRIX_CONFINEMENT"}:
        desired.extend(
            [
                "macro.run_solid_state_emission_proxy",
                "macro.run_dimer_packing_proxy",
                "macro.run_aggregate_contact_proxy",
                "macro.run_crystal_restriction_checklist",
            ]
        )
    return [
        capability_id
        for capability_id in dict.fromkeys(desired)
        if capability_id in available
    ][:3]


def _available_capability_ids_from_payload(payload_context: dict[str, object]) -> set[str]:
    ids: set[str] = set()
    for key in ("capability_cards", "ability_evidence_guide"):
        for item in _as_dict_list(payload_context.get(key)):
            capability_id = _short_text(item.get("capability_id"), 160)
            if capability_id:
                ids.add(capability_id)
    return ids


def _has_executed_discriminating_followup(payload_context: dict[str, object]) -> bool:
    executed = set()
    for key in (
        "new_round_evidence",
        "mechanism_evidence_coverage_table",
        "priority_hygiene_summary",
    ):
        for item in _as_dict_list(payload_context.get(key)):
            capability_id = _short_text(item.get("capability_id"), 160)
            if capability_id:
                executed.add(capability_id)
            for hit in _as_dict_list(item.get("route_hits")):
                hit_capability_id = _short_text(hit.get("capability_id"), 160)
                if hit_capability_id:
                    executed.add(hit_capability_id)
            for capability_id in _text_list(item.get("capability_ids"), limit=8):
                executed.add(capability_id)
    discriminating = {
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
    return bool(executed & discriminating)


def _labels_needing_discriminating_followup(
    payload_context: dict[str, object],
) -> set[str]:
    labels: set[str] = set()
    for item in _as_dict_list(payload_context.get("priority_hygiene_summary")):
        label = _normalize_mechanism_label(item.get("label"))
        if label in MECHANISM_POOL:
            labels.add(label)
    memo = payload_context.get("agenda_coverage_memo")
    if isinstance(memo, dict):
        for item in _as_dict_list(memo.get("agenda_items")):
            if str(item.get("urgency") or "").lower() != "high":
                continue
            label = _normalize_mechanism_label(item.get("label"))
            if label in MECHANISM_POOL:
                labels.add(label)
        for item in _as_dict_list(memo.get("recommended_next_routes")):
            targets = _text_list(item.get("targets"), limit=6)
            labels.update(
                label
                for label in (_normalize_mechanism_label(value) for value in targets)
                if label in MECHANISM_POOL
            )
    if not labels:
        labels.update(MECHANISM_POOL)
    return labels


def _merge_dispatch_lists(
    first: list[dict[str, object]],
    second: list[dict[str, object]],
) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    seen: set[str] = set()
    for request in [*first, *second]:
        capability_id = _short_text(request.get("capability_id"), 160)
        if not capability_id or capability_id in seen:
            continue
        merged.append(request)
        seen.add(capability_id)
        if len(merged) >= MAX_DISPATCHES_PER_ROUND:
            break
    return merged


def _dispatches_from_capability_ids(
    capability_ids: list[str],
    *,
    round_id: str,
    capability_registry: CapabilityRegistry,
) -> list[dict[str, object]]:
    dispatches: list[dict[str, object]] = []
    for capability_id in capability_ids[:MAX_DISPATCHES_PER_ROUND]:
        capability = capability_registry.get(capability_id)
        if capability is None:
            continue
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


def _dispatches_from_coverage_memo(
    *,
    payload_context: dict[str, object],
    round_id: str,
    capability_registry: CapabilityRegistry,
) -> list[dict[str, object]]:
    memo = payload_context.get("agenda_coverage_memo")
    if not isinstance(memo, dict):
        return []
    capability_ids: list[str] = []
    # High-urgency under-screened items must survive the per-round dispatch
    # limit; general recommendations are appended after those mandatory routes.
    for item in _as_dict_list(memo.get("agenda_items")):
        if str(item.get("urgency") or "").strip().lower() != "high":
            continue
        if str(item.get("coverage_status") or "").strip().lower() not in {
            "under_screened",
            "needs_follow_up",
        }:
            continue
        resolved = _capability_id_for_route(
            item.get("suggested_route"),
            capability_registry=capability_registry,
        )
        if resolved:
            capability_ids.append(resolved)
    for item in _as_dict_list(memo.get("recommended_next_routes")):
        capability_ids.extend(_text_list(item.get("capability_ids"), limit=3))
        if not _text_list(item.get("capability_ids"), limit=3):
            resolved = _capability_id_for_route(
                item.get("route"),
                capability_registry=capability_registry,
            )
            if resolved:
                capability_ids.append(resolved)
    executed = _executed_capability_ids_from_payload(payload_context)
    return _dispatches_from_capability_ids(
        [
            item
            for item in dict.fromkeys(capability_ids)
            if item and item not in executed
        ],
        round_id=round_id,
        capability_registry=capability_registry,
    )


def _capability_id_for_route(
    value: object,
    *,
    capability_registry: CapabilityRegistry,
) -> str | None:
    route = _short_text(value, 160)
    if not route:
        return None
    if capability_registry.get(route) is not None:
        return route
    matches = [
        str(card["capability_id"])
        for card in capability_registry.cards()
        if route
        in {
            str(card.get("route") or ""),
            str(card.get("capability_id") or "").rsplit(".", maxsplit=1)[-1],
        }
    ]
    return matches[0] if len(matches) == 1 else None


def _merge_coverage_dispatches(
    dispatch_requests: list[dict[str, object]],
    *,
    payload_context: dict[str, object],
    round_id: str,
    capability_registry: CapabilityRegistry,
) -> list[dict[str, object]]:
    coverage_dispatches = _dispatches_from_coverage_memo(
        payload_context=payload_context,
        round_id=round_id,
        capability_registry=capability_registry,
    )
    if not coverage_dispatches:
        return dispatch_requests
    if not dispatch_requests:
        return coverage_dispatches[:MAX_DISPATCHES_PER_ROUND]
    merged: list[dict[str, object]] = []
    seen: set[str] = set()
    prioritized_requests = [
        *coverage_dispatches[:1],
        *dispatch_requests,
        *coverage_dispatches[1:],
    ]
    for request in prioritized_requests:
        capability_id = _short_text(request.get("capability_id"), 160)
        if not capability_id or capability_id in seen:
            continue
        merged.append(request)
        seen.add(capability_id)
        if len(merged) >= MAX_DISPATCHES_PER_ROUND:
            break
    return merged


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
