from __future__ import annotations

from typing import Any

from mechcal.capabilities import CapabilityRegistry, default_capability_registry
from mechcal.capabilities.evidence_guide import compact_ability_evidence_guide
from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.public_ids import public_case_id_from_metadata
from mechcal.runtime.llm import OpenAIJsonClient
from mechcal.runtime.prompts import load_prompt
from mechcal.runtime.retry import SchemaRetryError
from mechcal.schemas import (
    CaseRun,
    MechanismAgendaCoverage,
    MechanismProgram,
)

FORBIDDEN_AGENDA_TEXT = {
    "hidden_reference",
    "semantic_evidence_targets",
    "semantic_diagnosis_targets",
    "reference_evidence_units",
    "reference_diagnosis_units",
    "final diagnosis",
    "final_diagnosis",
    "target-label schema",
    "benchmark reference",
    "reference mechanisms",
    "top-3",
    "top 3",
    "top_3",
    "mechanism_predictions",
}


class MechanismAgendaAgent:
    """Agenda and coverage reviewer for mechanism evidence gaps.

    The agent proposes questions, coverage states, and routes only. It does not
    produce evidence, diagnosis units, evaluation payloads, final scientific
    conclusions, priority scores, or ranked mechanism predictions.
    """

    def __init__(
        self,
        *,
        capability_registry: CapabilityRegistry | None = None,
        llm_client: OpenAIJsonClient | None = None,
    ) -> None:
        self.capability_registry = capability_registry or default_capability_registry()
        self.llm_client = llm_client or OpenAIJsonClient()

    def propose(self, case_run: CaseRun, *, round_id: str) -> MechanismProgram:
        if not self.llm_client.is_configured():
            raise RuntimeError(
                "MechanismAgendaAgent requires a configured LLM client."
            )

        prompt = load_prompt("mechanism_agenda.md")
        payload = _agenda_payload(case_run, self.capability_registry, round_id)
        feedback: str | None = None
        errors: list[str] = []
        for _ in range(3):
            try:
                response = self.llm_client.complete_json(
                    system_prompt=prompt,
                    payload=payload,
                    schema_feedback=feedback,
                )
                return _sanitize_program(
                    response,
                    case_run=case_run,
                    round_id=round_id,
                    capability_registry=self.capability_registry,
                )
            except Exception as exc:  # noqa: BLE001
                feedback = (
                    "Return one valid JSON object matching output_contract. "
                    f"Previous error: {type(exc).__name__}: {exc}"
                )
                errors.append(feedback)
        try:
            raise SchemaRetryError(errors)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"MechanismAgendaAgent failed to produce a valid agenda: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def review_coverage(
        self,
        case_run: CaseRun,
        *,
        round_id: str,
        previous_planner_action: str | None = None,
    ) -> MechanismAgendaCoverage:
        if _should_bootstrap_initial_coverage(case_run=case_run, round_id=round_id):
            return _bootstrap_initial_coverage(
                case_run=case_run,
                round_id=round_id,
                capability_registry=self.capability_registry,
            )

        if not self.llm_client.is_configured():
            raise RuntimeError(
                "MechanismAgendaAgent requires a configured LLM client."
            )

        prompt = load_prompt("mechanism_agenda_coverage.md")
        payload = _coverage_payload(
            case_run,
            self.capability_registry,
            round_id,
            previous_planner_action=previous_planner_action,
        )
        feedback: str | None = None
        errors: list[str] = []
        for _ in range(1):
            try:
                response = self.llm_client.complete_json(
                    system_prompt=prompt,
                    payload=payload,
                    schema_feedback=feedback,
                )
                return _sanitize_coverage(
                    response,
                    case_run=case_run,
                    round_id=round_id,
                    capability_registry=self.capability_registry,
                )
            except Exception as exc:  # noqa: BLE001
                feedback = (
                    "Return one valid JSON object matching output_contract. "
                    f"Previous error: {type(exc).__name__}: {exc}"
                )
                errors.append(feedback)
        try:
            raise SchemaRetryError(errors)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"MechanismAgendaAgent failed to produce a valid coverage memo: "
                f"{type(exc).__name__}: {exc}"
            ) from exc


def _agenda_payload(
    case_run: CaseRun,
    capability_registry: CapabilityRegistry,
    round_id: str,
) -> dict[str, Any]:
    return {
        "public_case": {
            "case_label": "current_smiles_case",
            "smiles": case_run.input.smiles,
            "user_query": _clip(case_run.input.user_query, 180),
            "round_id": round_id,
        },
        "recent_evidence": [
            {
                "evidence_id": item.evidence_id,
                "capability_id": item.capability_id,
                "family": item.family,
                "support": item.support,
                "status": item.status,
                "claim": _clip(item.claim, 180),
                "observable_tags": list(item.observable_tags[:5]),
            }
            for item in case_run.evidence_ledger.recent_items(limit=8)
        ],
        "previous_agenda": _compact_previous_agenda(case_run.mechanism_program),
        "mechanism_pool": list(MECHANISM_POOL),
        "ablation_constraints": _ablation_constraints(case_run),
        "capability_cards": [
            {
                "capability_id": item["capability_id"],
                "owner_agent": item["owner_agent"],
                "evidence_family": item["evidence_family"],
                "route": item["route"],
                "description": _clip(str(item["description"]), 160),
                "required_artifact_kinds": list(item["required_artifact_kinds"]),
            }
            for item in capability_registry.cards()
        ],
        "ability_evidence_guide": _coverage_ability_guide(capability_registry),
        "policy": {
            "private_answer_key_access": "none",
            "external_answer_access": "none",
            "case_specific_search": "forbidden",
            "final_decision_authority": "Planner only",
            "planner_authority": "Planner decides final hypothesis and final answer.",
        },
    }


def _coverage_payload(
    case_run: CaseRun,
    capability_registry: CapabilityRegistry,
    round_id: str,
    *,
    previous_planner_action: str | None,
) -> dict[str, Any]:
    return {
        "task": "agenda_coverage_review",
        "public_case": {
            "case_label": "current_smiles_case",
            "smiles": case_run.input.smiles,
            "user_query": _clip(case_run.input.user_query, 180),
            "round_id": round_id,
        },
        "mechanism_pool": list(MECHANISM_POOL),
        "ablation_constraints": _ablation_constraints(case_run),
        "current_evidence_ledger_summary": [
            {
                "evidence_id": item.evidence_id,
                "capability_id": item.capability_id,
                "family": item.family,
                "support": item.support,
                "status": item.status,
                "claim": _clip(item.claim, 100),
                "observable_tags": list(item.observable_tags[:3]),
            }
            for item in case_run.evidence_ledger.recent_items(limit=6)
        ],
        "current_mechanism_portfolio": _compact_current_portfolio(case_run),
        "previous_agenda_coverage": _compact_previous_coverage(
            case_run.mechanism_agenda_coverage
        ),
        "previous_planner_route_or_action": previous_planner_action,
        "support_audit_feedback": _compact_support_audit(case_run),
        "photophysics_review_feedback": _compact_photophysics_review(case_run),
        "capability_cards": [
            {
                "capability_id": item["capability_id"],
                "route": item["route"],
                "required_artifact_kinds": list(item["required_artifact_kinds"]),
            }
            for item in _coverage_capability_cards(capability_registry)
        ],
        "ability_evidence_guide": _coverage_ability_guide(capability_registry),
        "output_contract": {
            "coverage_summary": "one sentence, non-diagnostic coverage memo",
            "limits": (
                "Return at most 3 agenda_items, 2 screened_out, "
                "1 low_margin_competition, and 4 recommended_next_routes."
            ),
            "agenda_items": [
                {
                    "label": "exact mechanism_pool label",
                    "coverage_status": (
                        "under_screened|screened|screened_out|needs_follow_up"
                    ),
                    "why_relevant": "one short phrase",
                    "missing_evidence": "one short phrase",
                    "suggested_question": "short question for Planner",
                    "suggested_route": "route/capability family to consider",
                    "urgency": "high|medium|low",
                    "boundary": "one short boundary; not diagnosis or ranking",
                }
            ],
            "screened_out": [
                {
                    "label": "exact mechanism_pool label",
                    "reason": "public-evidence reason for temporary screen-out",
                }
            ],
            "low_margin_competitions": [
                {
                    "labels": ["exact mechanism_pool label", "exact mechanism_pool label"],
                    "reason": "why these remain close or underdetermined",
                    "suggested_disambiguating_route": "route/capability family",
                }
            ],
            "recommended_next_routes": [
                {
                    "route": "route/capability family",
                    "targets": ["exact mechanism_pool label"],
                    "reason": "why this route improves coverage or disambiguation",
                    "capability_ids": ["capability.id.from.capability_cards"],
                }
            ],
        },
        "policy": {
            "private_answer_key_access": "none",
            "external_answer_access": "none",
            "case_specific_search": "forbidden",
            "final_decision_authority": "Planner only",
            "ranking_authority": "Planner only",
            "may_modify_portfolio_priority": False,
            "may_output_top3": False,
            "may_output_final_diagnosis": False,
        },
    }


def _compact_previous_agenda(program: MechanismProgram | None) -> dict[str, Any] | None:
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
                "acceptable_capability_ids": list(item.acceptable_capability_ids[:4]),
            }
            for item in program.evidence_questions[:8]
        ],
    }


def _coverage_ability_guide(
    capability_registry: CapabilityRegistry,
) -> list[dict[str, object]]:
    allowed = _coverage_capability_ids(capability_registry)
    return [
        {
            "capability_id": item["capability_id"],
            "screens": item.get("screens", []),
            "evidence_tier": item.get("evidence_tier"),
            "suggested_claim_strength": item.get("suggested_claim_strength"),
        }
        for item in compact_ability_evidence_guide(capability_registry)
        if str(item.get("capability_id", "")) in allowed
    ]


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
            "Mark unavailable-domain evidence as unresolved. Do not recommend "
            "that an available worker imitate a disabled worker."
        ),
    }


def _coverage_capability_cards(
    capability_registry: CapabilityRegistry,
) -> list[dict[str, object]]:
    allowed = _coverage_capability_ids(capability_registry)
    return [
        item
        for item in capability_registry.cards()
        if str(item.get("capability_id", "")) in allowed
    ]


def _coverage_capability_ids(capability_registry: CapabilityRegistry) -> set[str]:
    wanted = {
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
        "macro.run_dimer_packing_proxy",
        "macro.run_aggregate_contact_proxy",
        "macro.run_crystal_restriction_checklist",
        "macro.run_solid_state_emission_proxy",
    }
    available = {
        str(item.get("capability_id", ""))
        for item in capability_registry.cards()
    }
    return wanted & available


def _compact_current_portfolio(case_run: CaseRun) -> dict[str, Any]:
    return {
        "current": case_run.portfolio.current,
        "runner_up": case_run.portfolio.runner_up,
        "hypotheses": [
            {
                "name": item.name,
                "differential_priority": item.differential_priority,
                "evidence_support": item.evidence_support,
                "claim_status": item.claim_status,
                "status": item.status,
                "rationale": _clip(item.rationale, 180),
                "evidence_refs": item.evidence_refs[:6],
                "validation_needed": item.validation_needed[:4],
            }
            for item in case_run.portfolio.sorted_hypotheses()[: len(MECHANISM_POOL)]
        ],
    }


def _compact_previous_coverage(
    coverage: MechanismAgendaCoverage | None,
) -> dict[str, Any] | None:
    if coverage is None:
        return None
    return {
        "round_id": coverage.round_id,
        "coverage_summary": _clip(coverage.coverage_summary, 220),
        "agenda_items": [
            {
                "label": item.label,
                "coverage_status": item.coverage_status,
                "suggested_route": item.suggested_route,
                "urgency": item.urgency,
            }
            for item in coverage.agenda_items[:4]
        ],
        "low_margin_competitions": [
            {
                "labels": item.labels,
                "suggested_disambiguating_route": item.suggested_disambiguating_route,
            }
            for item in coverage.low_margin_competitions[:2]
        ],
        "recommended_next_routes": [
            {
                "route": item.route,
                "targets": item.targets[:4],
                "capability_ids": item.capability_ids[:4],
            }
            for item in coverage.recommended_next_routes[:3]
        ],
    }


def _compact_support_audit(case_run: CaseRun) -> list[dict[str, Any]]:
    return [
        {
            "support_id": item.support_id,
            "target_label": item.target_label,
            "support_level": item.support_level,
            "audit_status": item.audit_status,
            "audit_notes": [_clip(note, 160) for note in item.audit_notes[:3]],
            "boundary": _clip(item.boundary, 180),
        }
        for item in case_run.mechanism_support_arguments[-10:]
    ]


def _compact_photophysics_review(case_run: CaseRun) -> dict[str, Any] | None:
    review = case_run.photophysics_review
    if review is None:
        return None
    return {
        "round_id": review.round_id,
        "recommended_next_routes": review.recommended_next_routes[:6],
        "overclaim_warnings": [_clip(item, 180) for item in review.overclaim_warnings[:4]],
        "coverage_axes": [
            {
                "label": item.label,
                "status": item.status,
                "rationale": _clip(item.rationale, 180),
                "recommended_routes": item.recommended_routes[:4],
            }
            for item in review.coverage_axes[:8]
        ],
    }


def _sanitize_coverage(
    response: dict[str, Any],
    *,
    case_run: CaseRun,
    round_id: str,
    capability_registry: CapabilityRegistry,
) -> MechanismAgendaCoverage:
    payload = _first_mapping(
        response,
        "mechanism_agenda_coverage",
        "coverage_memo",
        "agenda_coverage",
        "coverage_review",
    ) or response
    if _has_forbidden_coverage_content(payload):
        raise ValueError("Mechanism coverage memo contained forbidden final/evaluation text.")

    known_labels = set(MECHANISM_POOL)
    known_capabilities = {
        item["capability_id"]
        for item in capability_registry.cards()
        if isinstance(item.get("capability_id"), str)
    }
    public_case_id = _public_case_id(case_run)
    agenda_items = []
    for index, raw_item in enumerate(
        _first_dict_list(payload, "agenda_items", "coverage_items", "items")[:3],
        start=1,
    ):
        item = _normalize_coverage_item(raw_item, index=index, known_labels=known_labels)
        if item is not None:
            agenda_items.append(item)

    screened_out = []
    for raw_item in _first_dict_list(payload, "screened_out", "screened_out_mechanisms")[
        :2
    ]:
        item = _normalize_screened_out_item(raw_item, known_labels=known_labels)
        if item is not None:
            screened_out.append(item)

    low_margin_competitions = []
    for raw_item in _first_dict_list(
        payload,
        "low_margin_competitions",
        "low_margin_candidates",
    )[:1]:
        item = _normalize_low_margin_item(raw_item, known_labels=known_labels)
        if item is not None:
            low_margin_competitions.append(item)

    recommended_next_routes = []
    for raw_item in _first_dict_list(
        payload,
        "recommended_next_routes",
        "recommended_routes",
        "route_recommendations",
    )[:4]:
        item = _normalize_recommended_route_item(
            raw_item,
            known_labels=known_labels,
            known_capabilities=known_capabilities,
        )
        if item is not None:
            recommended_next_routes.append(item)
    recommended_next_routes = _augment_coverage_routes(
        recommended_next_routes,
        case_run=case_run,
        round_id=round_id,
        known_capabilities=known_capabilities,
    )[:4]
    if not agenda_items and not recommended_next_routes:
        agenda_items = [_coverage_no_action_item(case_run=case_run, round_id=round_id)]

    coverage = {
        "coverage_id": f"{public_case_id}:{round_id}:agenda_coverage",
        "case_id": public_case_id,
        "round_id": round_id,
        "coverage_summary": _clip(
            _first_text(payload, "coverage_summary", "summary", "memo")
            or "Coverage reviewer identified mechanism-screening gaps from public evidence.",
            220,
        ),
        "agenda_items": agenda_items,
        "screened_out": screened_out,
        "low_margin_competitions": low_margin_competitions,
        "recommended_next_routes": recommended_next_routes,
    }
    return MechanismAgendaCoverage.model_validate(coverage)


def _coverage_no_action_item(
    *,
    case_run: CaseRun,
    round_id: str,
) -> dict[str, str]:
    label = _first_current_portfolio_label(case_run) or "RADIATIVE_RATE_STATE_BALANCE"
    return {
        "label": label,
        "coverage_status": "screened",
        "why_relevant": "No new high-urgency coverage gap was returned for this round.",
        "missing_evidence": (
            "Planner should rely on the existing source-grounded ledger unless it "
            "needs a disambiguating route."
        ),
        "suggested_question": (
            "Is existing evidence sufficient to keep the current portfolio stable?"
        ),
        "suggested_route": "no_new_route_recommended",
        "urgency": "low",
        "boundary": (
            f"This {round_id} coverage memo is a non-ranking operational note, "
            "not a mechanism conclusion."
        ),
    }


def _first_current_portfolio_label(case_run: CaseRun) -> str | None:
    for item in case_run.portfolio.sorted_hypotheses():
        if item.name in MECHANISM_POOL:
            return item.name
    return None


def _should_bootstrap_initial_coverage(*, case_run: CaseRun, round_id: str) -> bool:
    if round_id != "R001":
        return False
    return not case_run.evidence_ledger.items


def _bootstrap_initial_coverage(
    *,
    case_run: CaseRun,
    round_id: str,
    capability_registry: CapabilityRegistry,
) -> MechanismAgendaCoverage:
    known_capabilities = {
        item["capability_id"]
        for item in capability_registry.cards()
        if isinstance(item.get("capability_id"), str)
    }
    recommended_next_routes = _augment_coverage_routes(
        [],
        case_run=case_run,
        round_id=round_id,
        known_capabilities=known_capabilities,
    )
    route_targets = {
        label
        for route in recommended_next_routes
        for label in route.get("targets", [])
        if label in MECHANISM_POOL
    }
    agenda_items = [
        _bootstrap_coverage_item(label, recommended_next_routes)
        for label in _bootstrap_agenda_labels(route_targets)
    ]
    screened_out = _bootstrap_screened_out_items(case_run)
    coverage = {
        "coverage_id": f"{_public_case_id(case_run)}:{round_id}:agenda_coverage_bootstrap",
        "case_id": _public_case_id(case_run),
        "round_id": round_id,
        "coverage_summary": (
            "Initial public-structure coverage bootstrap selected generic evidence "
            "routes before any mechanism ranking."
        ),
        "agenda_items": agenda_items[:4],
        "screened_out": screened_out[:2],
        "low_margin_competitions": [],
        "recommended_next_routes": recommended_next_routes[:6],
    }
    return MechanismAgendaCoverage.model_validate(coverage)


def _bootstrap_agenda_labels(route_targets: set[str]) -> list[str]:
    preferred_order = [
        "RADIATIVE_RATE_STATE_BALANCE",
        "ESIPT_PT",
        "ICT_TICT_CT",
        "RIM_RIR_RIV",
        "RACI_CI_ACCESS",
        "PACKING_HOST_MATRIX_CONFINEMENT",
        "AGGREGATE_EXCITON_EXCIMER",
        "PET_ET",
        "HOST_GUEST_INTERACTION",
        "TRIPLET_METAL_ENERGY_TRANSFER",
    ]
    labels = [label for label in preferred_order if label in route_targets]
    return labels or ["RADIATIVE_RATE_STATE_BALANCE"]


def _bootstrap_coverage_item(
    label: str,
    recommended_next_routes: list[dict[str, Any]],
) -> dict[str, Any]:
    matching_route = next(
        (
            route
            for route in recommended_next_routes
            if label in route.get("targets", [])
        ),
        {},
    )
    suggested_route = str(matching_route.get("route") or "public_structure_screen")
    return {
        "label": label,
        "coverage_status": "under_screened",
        "why_relevant": (
            f"{label} has not yet been screened with public runtime evidence."
        ),
        "missing_evidence": (
            "No source-grounded observation has been collected in this run yet."
        ),
        "suggested_question": (
            "Which public structure-derived observation can bound this mechanism?"
        ),
        "suggested_route": suggested_route,
        "urgency": "high" if label == "RADIATIVE_RATE_STATE_BALANCE" else "medium",
        "boundary": (
            "This is an initial coverage route recommendation, not a diagnosis, "
            "mechanism ranking, or support claim."
        ),
    }


def _bootstrap_screened_out_items(case_run: CaseRun) -> list[dict[str, str]]:
    screened_out: list[dict[str, str]] = []
    if not _has_triplet_metal_structural_trigger(case_run):
        screened_out.append(
            {
                "label": "TRIPLET_METAL_ENERGY_TRANSFER",
                "reason": (
                    "No public metal, lanthanide, heavy-atom, sulfur, or phosphorus "
                    "motif was detected for initial triplet/metal route coverage."
                ),
            }
        )
    if not _has_esipt_structural_trigger(case_run):
        screened_out.append(
            {
                "label": "ESIPT_PT",
                "reason": (
                    "No public proton donor/acceptor structural trigger was detected "
                    "for initial ESIPT route coverage."
                ),
            }
        )
    return screened_out


def _augment_coverage_routes(
    routes: list[dict[str, Any]],
    *,
    case_run: CaseRun,
    round_id: str,
    known_capabilities: set[str],
) -> list[dict[str, Any]]:
    if round_id != "R001":
        return _augment_followup_coverage_routes(
            routes,
            case_run=case_run,
            known_capabilities=known_capabilities,
        )
    has_prepared_structure = any(
        artifact.kind == "prepared_structure" and artifact.status in {"available", "partial"}
        for artifact in case_run.artifact_manifest.artifacts
    )
    has_smiles_structure_context = any(
        artifact.kind == "smiles_structure_context"
        and artifact.status in {"available", "partial"}
        for artifact in case_run.artifact_manifest.artifacts
    )
    if not has_prepared_structure and not has_smiles_structure_context:
        return routes
    prioritized: list[tuple[int, dict[str, Any]]] = []
    if has_prepared_structure and "microscopic.run_baseline_bundle" in known_capabilities:
        prioritized.append(
            (
                100,
                {
                    "route": "microscopic.run_baseline_bundle",
                    "targets": [
                        "RADIATIVE_RATE_STATE_BALANCE",
                        "SOKR_ANTI_KASHA",
                    ],
                    "reason": (
                        "Generic early coverage of low-cost state-ordering and brightness "
                        "observables; this is not a mechanism ranking."
                    ),
                    "capability_ids": ["microscopic.run_baseline_bundle"],
                },
            )
        )
    if _has_esipt_structural_trigger(case_run) and (
        "macro.screen_esipt_structural_motif" in known_capabilities
    ):
        prioritized.append(
            (
                98,
                {
                    "route": "macro.screen_esipt_structural_motif",
                    "targets": ["ESIPT_PT"],
                    "reason": (
                        "Generic early coverage of proton donor/acceptor structural "
                        "motifs; this is not proton-transfer evidence."
                    ),
                    "capability_ids": ["macro.screen_esipt_structural_motif"],
                },
            )
        )
    if _has_triplet_metal_structural_trigger(case_run) and (
        "macro.screen_metal_triplet_prior" in known_capabilities
    ):
        prioritized.append(
            (
                95,
                {
                    "route": "macro.screen_metal_triplet_prior",
                    "targets": ["TRIPLET_METAL_ENERGY_TRANSFER"],
                    "reason": (
                        "Generic early coverage of metal, heavy-atom, sulfur, or "
                        "phosphorus triplet priors; this is not phosphorescence, RTP, "
                        "TADF, or energy-transfer evidence."
                    ),
                    "capability_ids": ["macro.screen_metal_triplet_prior"],
                },
            )
        )
    if _has_polar_binding_site_trigger(case_run) and (
        "macro.screen_polar_binding_site_prior" in known_capabilities
    ):
        prioritized.append(
            (
                90,
                {
                    "route": "macro.screen_polar_binding_site_prior",
                    "targets": ["HOST_GUEST_INTERACTION", "PET_ET"],
                    "reason": (
                        "Generic early coverage of polar, charged, carbonyl, or "
                        "H-bonding interaction priors; this is not binding, redox, "
                        "guest-uptake, or PET evidence."
                    ),
                    "capability_ids": ["macro.screen_polar_binding_site_prior"],
                },
            )
        )
    d_a_capability = _preferred_donor_acceptor_capability(known_capabilities)
    if d_a_capability is not None:
        prioritized.append(
            (
                80,
                {
                    "route": d_a_capability,
                    "targets": ["ICT_TICT_CT", "PET_ET"],
                    "reason": (
                        "Generic early coverage of donor-acceptor and charge-transfer "
                        "structural priors; this is not CT, TICT, PET, or redox evidence."
                    ),
                    "capability_ids": [d_a_capability],
                },
            )
        )
    if "macro.screen_rotor_torsion_topology" in known_capabilities:
        prioritized.append(
            (
                70,
                {
                    "route": "macro.screen_rotor_torsion_topology",
                    "targets": ["RIM_RIR_RIV", "RACI_CI_ACCESS", "ICT_TICT_CT"],
                    "reason": (
                        "Generic early coverage of rotor and torsion topology for "
                        "motion-coupled differential screening; this is not direct "
                        "RIM, RACI, or TICT evidence."
                    ),
                    "capability_ids": ["macro.screen_rotor_torsion_topology"],
                },
            )
        )
    if "macro.screen_aggregation_prone_scaffold" in known_capabilities:
        prioritized.append(
            (
                60,
                {
                    "route": "macro.screen_aggregation_prone_scaffold",
                    "targets": [
                        "PACKING_HOST_MATRIX_CONFINEMENT",
                        "AGGREGATE_EXCITON_EXCIMER",
                        "RIM_RIR_RIV",
                    ],
                    "reason": (
                        "Generic early coverage of aggregation, packing, and "
                        "solid-state restriction structural proxies; this is not a "
                        "mechanism ranking."
                    ),
                    "capability_ids": ["macro.screen_aggregation_prone_scaffold"],
                },
            )
        )
    augmented = [
        route for _, route in sorted(prioritized, key=lambda item: item[0], reverse=True)
    ]
    d_a_capability_ids = {
        "macro.screen_donor_acceptor_architecture",
        "macro.screen_donor_acceptor_layout",
    }
    preferred_routes: list[dict[str, Any]] = []
    fallback_routes: list[dict[str, Any]] = []
    already_added = {
        capability_id
        for route in augmented
        if isinstance(route.get("capability_ids"), list)
        for capability_id in route["capability_ids"]
    }
    for route in routes:
        capability_ids = route.get("capability_ids", [])
        if not isinstance(capability_ids, list):
            continue
        if any(capability_id in already_added for capability_id in capability_ids):
            continue
        if any(capability_id in d_a_capability_ids for capability_id in capability_ids):
            fallback_routes.append(route)
            continue
        preferred_routes.append(route)
    for route in [*preferred_routes, *fallback_routes]:
        augmented.append(route)
        if len(augmented) >= 6:
            break
    return _dedupe_recommended_routes(augmented)[:6]


def _augment_followup_coverage_routes(
    routes: list[dict[str, Any]],
    *,
    case_run: CaseRun,
    known_capabilities: set[str],
) -> list[dict[str, Any]]:
    executed = _executed_capability_ids(case_run)
    prioritized: list[tuple[int, dict[str, Any]]] = []
    if _needs_packing_followup(case_run, executed):
        for priority, capability_id, reason in (
            (
                100,
                "macro.run_dimer_packing_proxy",
                (
                    "Follow-up packing/contact coverage because packing or aggregate "
                    "candidates remain proxy-level."
                ),
            ),
            (
                95,
                "macro.run_aggregate_contact_proxy",
                (
                    "Follow-up aggregate-contact coverage to distinguish "
                    "packing/confinement from generic aggregation."
                ),
            ),
            (
                90,
                "macro.run_crystal_restriction_checklist",
                (
                    "Boundary checklist for missing crystal/solid-state evidence "
                    "before finalizing packing/confinement claims."
                ),
            ),
        ):
            if capability_id in known_capabilities and capability_id not in executed:
                prioritized.append(
                    (
                        priority,
                        {
                            "route": capability_id,
                            "targets": [
                                "PACKING_HOST_MATRIX_CONFINEMENT",
                                "AGGREGATE_EXCITON_EXCIMER",
                                "RIM_RIR_RIV",
                            ],
                            "reason": reason,
                            "capability_ids": [capability_id],
                        },
                    )
                )
    if _needs_state_ordering_followup(case_run, executed):
        capability_id = "microscopic.run_bright_dark_state_ordering"
        if capability_id in known_capabilities and capability_id not in executed:
            prioritized.append(
                (
                    80,
                    {
                        "route": capability_id,
                        "targets": [
                            "RADIATIVE_RATE_STATE_BALANCE",
                            "SOKR_ANTI_KASHA",
                        ],
                        "reason": (
                            "Follow-up bright/dark ordering coverage for state-balance "
                            "and anti-Kasha screening without claiming higher-state emission."
                        ),
                        "capability_ids": [capability_id],
                    },
                )
            )
    augmented = [
        route for _, route in sorted(prioritized, key=lambda item: item[0], reverse=True)
    ]
    for route in routes:
        capability_ids = route.get("capability_ids", [])
        if not isinstance(capability_ids, list):
            continue
        if any(str(capability_id) in executed for capability_id in capability_ids):
            continue
        augmented.append(route)
    return _dedupe_recommended_routes(augmented)[:6]


def _executed_capability_ids(case_run: CaseRun) -> set[str]:
    return {
        str(item.capability_id)
        for item in case_run.evidence_ledger.items
        if str(item.capability_id or "").strip()
    }


def _needs_packing_followup(case_run: CaseRun, executed: set[str]) -> bool:
    packing_followups = {
        "macro.run_dimer_packing_proxy",
        "macro.run_aggregate_contact_proxy",
        "macro.run_crystal_restriction_checklist",
    }
    if packing_followups <= executed:
        return False
    if {
        "macro.screen_aggregation_prone_scaffold",
        "macro.run_solid_state_emission_proxy",
    } & executed:
        return True
    labels = {
        item.name
        for item in case_run.portfolio.sorted_hypotheses()[:6]
        if item.name and item.name != "unknown"
    }
    return bool(
        labels
        & {
            "PACKING_HOST_MATRIX_CONFINEMENT",
            "AGGREGATE_EXCITON_EXCIMER",
            "RIM_RIR_RIV",
        }
    )


def _needs_state_ordering_followup(case_run: CaseRun, executed: set[str]) -> bool:
    if "microscopic.run_bright_dark_state_ordering" in executed:
        return False
    labels = {
        item.name
        for item in case_run.portfolio.sorted_hypotheses()[:5]
        if item.name and item.name != "unknown"
    }
    if labels & {"RADIATIVE_RATE_STATE_BALANCE", "SOKR_ANTI_KASHA"}:
        return True
    return "microscopic.run_baseline_bundle" in executed


def _preferred_donor_acceptor_capability(
    known_capabilities: set[str],
) -> str | None:
    for capability_id in (
        "macro.screen_donor_acceptor_layout",
        "macro.screen_donor_acceptor_architecture",
    ):
        if capability_id in known_capabilities:
            return capability_id
    return None


def _dedupe_recommended_routes(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for route in routes:
        capability_ids = route.get("capability_ids", [])
        if not isinstance(capability_ids, list):
            capability_ids = []
        key = tuple(str(item) for item in capability_ids if str(item).strip()) or (
            str(route.get("route") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(route)
    return deduped


def _has_esipt_structural_trigger(case_run: CaseRun) -> bool:
    smiles = case_run.input.smiles
    try:
        from rdkit import Chem
        from rdkit.Chem import Lipinski
    except ModuleNotFoundError:
        lower = smiles.lower()
        return "o" in lower and "n" in lower
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        lower = smiles.lower()
        return "o" in lower and "n" in lower
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    return hbd > 0 and hba > 0


def _has_triplet_metal_structural_trigger(case_run: CaseRun) -> bool:
    smiles = case_run.input.smiles
    fallback_tokens = (
        "BR",
        "[I",
        "SE",
        "TE",
        "PT",
        "IR",
        "RU",
        "OS",
        "RE",
        "EU",
        "TB",
        "GD",
        "[S",
        "[P",
    )
    try:
        from rdkit import Chem
    except ModuleNotFoundError:
        upper = smiles.upper()
        return any(token in upper for token in fallback_tokens)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        upper = smiles.upper()
        return any(token in upper for token in fallback_tokens)
    symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    triplet_symbols = {
        "S",
        "P",
        "Br",
        "I",
        "Se",
        "Te",
        "At",
        "Au",
        "Pt",
        "Ir",
        "Ru",
        "Os",
        "Re",
    }
    lanthanides = {
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
    }
    first_row_and_heavy_transition_metals = {
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Pd",
        "Ag",
        "Cd",
        "Hf",
        "Ta",
        "W",
        "Hg",
    }
    return bool(
        symbols
        & (triplet_symbols | lanthanides | first_row_and_heavy_transition_metals)
    )


def _has_polar_binding_site_trigger(case_run: CaseRun) -> bool:
    smiles = case_run.input.smiles
    try:
        from rdkit import Chem
        from rdkit.Chem import Lipinski
    except ModuleNotFoundError:
        return any(token in smiles for token in ("+", "-", "=O", "N", "O"))
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return any(token in smiles for token in ("+", "-", "=O", "N", "O"))
    formal_charge = int(Chem.GetFormalCharge(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    hetero_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() not in {"C", "H"})
    carbonyl = mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3]=[OX1]"))
    return bool(formal_charge or hbd or hba >= 2 or hetero_atoms >= 3 or carbonyl)


def _normalize_coverage_item(
    item: dict[str, Any],
    *,
    index: int,
    known_labels: set[str],
) -> dict[str, Any] | None:
    label = _mechanism_label(item, known_labels)
    if not label:
        return None
    status = _first_text(item, "coverage_status", "status") or "under_screened"
    if status not in {"under_screened", "screened", "screened_out", "needs_follow_up"}:
        status = "under_screened"
    urgency = _first_text(item, "urgency", "priority") or "medium"
    if urgency not in {"high", "medium", "low"}:
        urgency = "medium"
    return {
        "label": label,
        "coverage_status": status,
        "why_relevant": _clip(
            _first_text(item, "why_relevant", "relevance", "reason")
            or f"{label} remains part of the public mechanism pool coverage review.",
            360,
        ),
        "missing_evidence": _clip(
            _first_text(item, "missing_evidence", "missing", "gap")
            or "Mechanism-specific evidence remains incomplete.",
            300,
        ),
        "suggested_question": _clip(
            _first_text(item, "suggested_question", "question")
            or "What public runtime observation would screen or bound this mechanism?",
            360,
        ),
        "suggested_route": _clip(
            _first_text(item, "suggested_route", "route")
            or f"coverage_follow_up_{index:03d}",
            180,
        ),
        "urgency": urgency,
        "boundary": _clip(
            _first_text(item, "boundary", "scope_boundary")
            or "This coverage item is not a final diagnosis or ranking decision.",
            360,
        ),
    }


def _normalize_screened_out_item(
    item: dict[str, Any],
    *,
    known_labels: set[str],
) -> dict[str, Any] | None:
    label = _mechanism_label(item, known_labels)
    if not label:
        return None
    return {
        "label": label,
        "reason": _clip(
            _first_text(item, "reason", "rationale")
            or "Current public evidence does not trigger this mechanism strongly.",
            360,
        ),
    }


def _normalize_low_margin_item(
    item: dict[str, Any],
    *,
    known_labels: set[str],
) -> dict[str, Any] | None:
    labels = [
        label
        for label in (
            _label_from_value(value, known_labels)
            for value in (item.get("labels") if isinstance(item.get("labels"), list) else [])
        )
        if label
    ]
    for key in ("label", "mechanism", "target"):
        label = _label_from_value(item.get(key), known_labels)
        if label and label not in labels:
            labels.append(label)
    if len(labels) < 2:
        return None
    return {
        "labels": labels[:4],
        "reason": _clip(
            _first_text(item, "reason", "rationale")
            or "These candidates are close under current proxy-level evidence.",
            360,
        ),
        "suggested_disambiguating_route": _clip(
            _first_text(
                item,
                "suggested_disambiguating_route",
                "suggested_route",
                "route",
            )
            or "Collect a route that distinguishes the competing mechanisms.",
            220,
        ),
    }


def _normalize_recommended_route_item(
    item: dict[str, Any],
    *,
    known_labels: set[str],
    known_capabilities: set[str],
) -> dict[str, Any] | None:
    route = _first_text(item, "route", "suggested_route", "capability_family")
    targets = [
        label
        for label in (
            _label_from_value(value, known_labels)
            for value in (item.get("targets") if isinstance(item.get("targets"), list) else [])
        )
        if label
    ]
    label = _mechanism_label(item, known_labels)
    if label and label not in targets:
        targets.append(label)
    capability_ids = [
        capability_id
        for capability_id in _as_text_list(item.get("capability_ids"))
        if capability_id in known_capabilities
    ]
    if not route and capability_ids:
        route = capability_ids[0]
    if not route:
        return None
    return {
        "route": _clip(route, 220),
        "targets": targets[:4],
        "reason": _clip(
            _first_text(item, "reason", "rationale")
            or "This route could improve mechanism coverage or disambiguation.",
            360,
        ),
        "capability_ids": capability_ids[:4],
    }


def _mechanism_label(item: dict[str, Any], known_labels: set[str]) -> str:
    for key in ("label", "mechanism", "target_label", "name"):
        label = _label_from_value(item.get(key), known_labels)
        if label:
            return label
    return ""


def _has_forbidden_agenda_content(payload: Any) -> bool:
    text_blob = str(payload).lower()
    allowed_negations = (
        "not a final diagnosis",
        "not final diagnosis",
        "no final diagnosis",
        "not a final scientific conclusion",
        "not final scientific conclusion",
    )
    for phrase in allowed_negations:
        text_blob = text_blob.replace(phrase, "")
    return any(term in text_blob for term in FORBIDDEN_AGENDA_TEXT)


def _has_forbidden_coverage_content(payload: Any) -> bool:
    text_blob = str(payload).lower()
    allowed_negations = (
        "not a final diagnosis",
        "not final diagnosis",
        "no final diagnosis",
        "not a final scientific conclusion",
        "not final scientific conclusion",
    )
    for phrase in allowed_negations:
        text_blob = text_blob.replace(phrase, "")
    forbidden_terms = {
        "hidden_reference",
        "semantic_evidence_targets",
        "semantic_diagnosis_targets",
        "reference_evidence_units",
        "reference_diagnosis_units",
        "target-label schema",
        "benchmark reference",
        "reference mechanisms",
        "final_diagnosis",
        "mechanism_predictions",
    }
    if any(term in text_blob for term in forbidden_terms):
        return True
    top_tokens = ("top-3", "top 3")
    rank_tokens = ("rank 1", "rank1", "ranked top", "top-ranked")
    return any(term in text_blob for term in top_tokens + rank_tokens)


def program_from_coverage(
    coverage: MechanismAgendaCoverage,
    *,
    capability_registry: CapabilityRegistry,
) -> MechanismProgram:
    known_capabilities = {
        item["capability_id"]: item
        for item in capability_registry.cards()
        if isinstance(item.get("capability_id"), str)
    }
    candidate_mechanisms = []
    evidence_questions = []
    routes = []
    for index, item in enumerate(coverage.agenda_items[:8], start=1):
        mechanism_id = _slug(item.label, fallback=f"mechanism_{index:03d}")
        question_id = f"Q{index:03d}"
        candidate_mechanisms.append(
            {
                "mechanism_id": mechanism_id,
                "label": item.label,
                "mechanism_family": item.label,
                "context": "Agenda coverage reviewer item, not a ranked prediction.",
                "rationale": item.why_relevant,
            }
        )
        acceptable = [
            capability_id
            for route_item in coverage.recommended_next_routes
            if item.label in route_item.targets
            for capability_id in route_item.capability_ids
            if capability_id in known_capabilities
        ][:4]
        if not acceptable:
            acceptable = [
                capability_id
                for capability_id, card in known_capabilities.items()
                if _route_matches_coverage_item(card, item)
            ][:4]
        evidence_questions.append(
            {
                "question_id": question_id,
                "mechanism_id": mechanism_id,
                "question": item.suggested_question,
                "observable": _slug(item.suggested_route, fallback=f"coverage_{index:03d}")[:80],
                "expected_basis": "proxy",
                "status": "unresolved"
                if item.coverage_status in {"under_screened", "needs_follow_up"}
                else "covered",
                "acceptable_capability_ids": acceptable,
            }
        )
        for capability_id in acceptable:
            card = known_capabilities[capability_id]
            routes.append(
                {
                    "question_id": question_id,
                    "capability_id": capability_id,
                    "agent_name": card["owner_agent"],
                    "route": card["route"],
                    "expected_basis": "proxy",
                }
            )
    if not candidate_mechanisms:
        candidate_mechanisms.append(
            {
                "mechanism_id": "coverage_review",
                "label": "Coverage review",
                "mechanism_family": "coverage_review",
                "context": "Agenda coverage reviewer item, not a ranked prediction.",
                "rationale": coverage.coverage_summary,
            }
        )
        evidence_questions.append(
            {
                "question_id": "Q001",
                "mechanism_id": "coverage_review",
                "question": "What public runtime evidence would improve mechanism coverage?",
                "observable": "mechanism_coverage_gap",
                "expected_basis": "proxy",
                "status": "unresolved",
                "acceptable_capability_ids": list(known_capabilities)[:4],
            }
        )
    return MechanismProgram.model_validate(
        {
            "program_id": coverage.coverage_id.replace(":agenda_coverage", ":mechanism_agenda"),
            "case_id": coverage.case_id,
            "round_id": coverage.round_id,
            "candidate_mechanisms": candidate_mechanisms,
            "evidence_questions": evidence_questions,
            "computational_routes": routes[:12],
            "scope_boundaries": [
                {
                    "boundary_id": "coverage_not_ranking",
                    "statement": (
                        "Agenda coverage items are route suggestions and coverage "
                        "boundaries, not final mechanism rankings."
                    ),
                }
            ],
        }
    )


def _route_matches_coverage_item(
    card: dict[str, Any],
    item: Any,
) -> bool:
    haystack = _slug(
        " ".join(
            str(card.get(key, ""))
            for key in ("capability_id", "evidence_family", "route", "description")
        ),
        fallback="capability",
    )
    needle = _slug(
        " ".join([str(item.label), str(item.suggested_route), str(item.suggested_question)]),
        fallback="coverage",
    )
    return any(token and token in haystack for token in needle.split("_") if len(token) > 3)


def _label_from_value(value: Any, known_labels: set[str]) -> str:
    text = str(value or "").strip()
    if text in known_labels:
        return text
    normalized = text.lower()
    for label in known_labels:
        if normalized == label.lower():
            return label
    return ""


def _sanitize_program(
    response: dict[str, Any],
    *,
    case_run: CaseRun,
    round_id: str,
    capability_registry: CapabilityRegistry,
) -> MechanismProgram:
    raw_program = _first_mapping(
        response,
        "mechanism_program",
        "proposed_agenda",
        "agenda",
        "mechanism_agenda",
    )
    payload = raw_program if raw_program is not None else response
    if _has_forbidden_agenda_content(payload):
        raise ValueError("Mechanism agenda contained forbidden final/evaluation text.")

    known_capabilities = {
        item["capability_id"]: item
        for item in capability_registry.cards()
        if isinstance(item.get("capability_id"), str)
    }
    candidate_items = _first_dict_list(
        payload,
        "candidate_mechanisms",
        "candidate_mechanism_families",
        "mechanism_families",
        "candidate_families",
        "mechanisms",
    )
    question_items = _first_dict_list(
        payload,
        "evidence_questions",
        "questions",
        "discriminating_questions",
        "evidence_agenda",
    )
    route_items = _first_dict_list(
        payload,
        "computational_routes",
        "recommended_routes",
        "route_recommendations",
    )
    boundary_items = _first_boundary_list(
        payload,
        "scope_boundaries",
        "boundaries",
        "scope_limits",
    )
    public_case_id = _public_case_id(case_run)
    candidate_mechanisms = [
        _normalize_candidate_item(item, index=index)
        for index, item in enumerate(candidate_items[:10], start=1)
    ]
    candidate_mechanisms = [
        item
        for item in candidate_mechanisms
        if item.get("mechanism_id")
        and item.get("label")
        and item.get("mechanism_family")
        and item.get("rationale")
    ]
    sanitized: dict[str, Any] = {
        "program_id": f"{public_case_id}:{round_id}:mechanism_agenda",
        "case_id": public_case_id,
        "round_id": round_id,
        "candidate_mechanisms": candidate_mechanisms,
    }
    if not sanitized["candidate_mechanisms"] and question_items:
        sanitized["candidate_mechanisms"] = [
            {
                "mechanism_id": "llm_open_mechanism_agenda",
                "label": "LLM-proposed open mechanism agenda",
                "mechanism_family": "open_mechanism_agenda",
                "context": "SMILES and supplied evidence agenda",
                "rationale": (
                    "The LLM supplied evidence questions without a separate "
                    "candidate-mechanism list; this is an operational agenda "
                    "container, not a scientific diagnosis."
                ),
            }
        ]
    candidate_ids = [
        str(item.get("mechanism_id"))
        for item in sanitized["candidate_mechanisms"]
        if str(item.get("mechanism_id", "")).strip()
    ]

    questions = []
    question_ids: set[str] = set()
    nested_route_items: list[dict[str, Any]] = []
    for index, raw_item in enumerate(question_items[:10], start=1):
        item = _normalize_question_item(raw_item, index=index)
        if not item.get("mechanism_id") and candidate_ids:
            item["mechanism_id"] = candidate_ids[min(index - 1, len(candidate_ids) - 1)]
        acceptable = [
            capability_id
            for capability_id in _capability_id_list(item.get("acceptable_capability_ids"))
            if capability_id in known_capabilities
        ][:4]
        item["acceptable_capability_ids"] = acceptable
        nested_route_items.extend(
            {
                **route_item,
                "question_id": item["question_id"],
            }
            for route_item in _as_dicts(raw_item.get("computational_routes"))
        )
        if not (
            item.get("question_id")
            and item.get("mechanism_id")
            and item.get("question")
            and item.get("observable")
        ):
            continue
        questions.append(item)
        if isinstance(item.get("question_id"), str):
            question_ids.add(item["question_id"])
    if not questions and candidate_ids:
        questions = [
            _repair_question_from_candidate(
                candidate,
                index=index,
                known_capabilities=known_capabilities,
            )
            for index, candidate in enumerate(
                sanitized["candidate_mechanisms"][:4], start=1
            )
        ]
        question_ids = {str(item["question_id"]) for item in questions}
    sanitized["evidence_questions"] = questions

    routes = []
    for item in [*route_items, *nested_route_items][:16]:
        item = _normalize_route_item(item)
        capability_id = str(item.get("capability_id", "")).strip()
        question_id = str(item.get("question_id", "")).strip()
        card = known_capabilities.get(capability_id)
        if card is None or question_id not in question_ids:
            continue
        item["agent_name"] = card["owner_agent"]
        item["route"] = card["route"]
        item["expected_basis"] = item.get("expected_basis") or "computed"
        routes.append(item)
    sanitized["computational_routes"] = routes
    sanitized["scope_boundaries"] = [
        item
        for item in (
            _normalize_boundary_item(item)
            for item in boundary_items[:8]
        )
        if item.get("boundary_id") and item.get("statement")
    ]

    program = MechanismProgram.model_validate(sanitized)
    if not program.candidate_mechanisms or not program.evidence_questions:
        raise ValueError("Mechanism agenda must contain candidates and questions.")
    return program


def _repair_question_from_candidate(
    candidate: dict[str, Any],
    *,
    index: int,
    known_capabilities: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    mechanism_id = str(candidate["mechanism_id"])
    mechanism_family = str(candidate["mechanism_family"])
    acceptable = [
        capability_id
        for capability_id, card in known_capabilities.items()
        if _capability_matches_family(card, mechanism_family)
    ][:4]
    if not acceptable:
        acceptable = list(known_capabilities)[:4]
    return {
        "question_id": f"Q{index:03d}",
        "mechanism_id": mechanism_id,
        "question": (
            "What source-grounded observations would help screen this "
            "candidate mechanism without making a terminal mechanism claim?"
        ),
        "observable": "candidate_mechanism_screening_observations",
        "expected_basis": "proxy",
        "status": "unresolved",
        "acceptable_capability_ids": acceptable,
    }


def _capability_matches_family(card: dict[str, Any], mechanism_family: str) -> bool:
    family_tokens = {
        token
        for token in _slug(mechanism_family, fallback="mechanism").split("_")
        if len(token) > 2
    }
    haystack = _slug(
        " ".join(
            str(card.get(key, ""))
            for key in ("capability_id", "evidence_family", "route", "description")
        ),
        fallback="capability",
    )
    return any(token in haystack for token in family_tokens)


def _as_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _public_case_id(case_run: CaseRun) -> str:
    return public_case_id_from_metadata(case_run.input.metadata)


def _as_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        value = [] if value is None else [value]
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_question_item(item: dict[str, Any], *, index: int) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    normalized["question_id"] = _first_text(
        item,
        "question_id",
        "evidence_question_id",
        "question_key",
        "id",
    ) or f"Q{index:03d}"
    normalized["mechanism_id"] = _first_text(
        item,
        "mechanism_id",
        "candidate_mechanism_id",
        "candidate_id",
        "family_id",
        "mechanism_family_id",
    ) or _first_text_from_list(item.get("relevant_families"))
    normalized["question"] = _first_text(
        item,
        "question",
        "evidence_question",
        "prompt",
        "text",
    )
    question_text = normalized["question"]
    normalized["observable"] = _first_text(
        item,
        "observable",
        "observable_axis",
        "target_observable",
    ) or _slug(question_text or normalized["question_id"], fallback=f"observable_{index}")[:80]
    expected_basis = _first_text(item, "expected_basis", "basis") or "proxy"
    if expected_basis not in {"computed", "proxy", "checklist"}:
        expected_basis = "proxy"
    normalized["expected_basis"] = expected_basis
    status = _first_text(item, "status") or "unresolved"
    if status not in {"unresolved", "computable", "checklist_only", "covered", "blocked"}:
        status = "unresolved"
    normalized["status"] = status
    acceptable = []
    for key in (
        "acceptable_capability_ids",
        "acceptable_capabilities",
        "capability_ids",
        "available_capability_ids",
        "recommended_capability_ids",
    ):
        acceptable.extend(_as_text_list(item.get(key)))
    for route_item in _as_dicts(item.get("computational_routes")):
        acceptable.extend(_capability_id_list(route_item.get("capability_id")))
    normalized["acceptable_capability_ids"] = acceptable
    return normalized


def _normalize_candidate_item(item: dict[str, Any], *, index: int) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    label = _first_text(
        item,
        "label",
        "mechanism",
        "name",
        "family_label",
        "family_id",
        "mechanism_label",
    )
    normalized["mechanism_id"] = _first_text(
        item,
        "mechanism_id",
        "candidate_mechanism_id",
        "candidate_id",
        "family_id",
        "mechanism_family_id",
        "id",
    ) or _slug(label, fallback=f"mechanism_{index:03d}")
    normalized["label"] = label or f"Mechanism family {index}"
    normalized["mechanism_family"] = _first_text(
        item,
        "mechanism_family",
        "family",
        "mechanism_type",
        "family_label",
        "family_id",
    ) or normalized["mechanism_id"]
    normalized["context"] = (
        _first_text(item, "context", "agenda_context")
        or "SMILES and supplied evidence agenda"
    )
    normalized["rationale"] = _first_text(
        item,
        "rationale",
        "reasoning",
        "description",
    )
    return normalized


def _normalize_route_item(item: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    normalized["question_id"] = _first_text(
        item,
        "question_id",
        "evidence_question_id",
        "question_key",
        "id",
    )
    normalized["capability_id"] = _first_text(
        item,
        "capability_id",
        "capability",
        "recommended_capability_id",
        "selected_capability_id",
    )
    expected_basis = _first_text(item, "expected_basis", "basis") or "computed"
    if expected_basis not in {"computed", "proxy", "checklist"}:
        expected_basis = "computed"
    normalized["expected_basis"] = expected_basis
    return normalized


def _normalize_boundary_item(item: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    normalized["boundary_id"] = _first_text(item, "boundary_id", "id") or "boundary"
    normalized["statement"] = _first_text(
        item,
        "statement",
        "scope_boundary",
        "boundary",
        "text",
    )
    return normalized


def _first_text(mapping: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _first_mapping(mapping: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, dict):
            return value
    return None


def _first_text_from_list(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            text = str(item).strip()
            if text:
                return text
    return ""


def _first_dict_list(mapping: dict[str, Any], *keys: str) -> list[dict[str, Any]]:
    for key in keys:
        value = mapping.get(key)
        items = _as_dicts(value)
        if items:
            return items
    return []


def _first_boundary_list(mapping: dict[str, Any], *keys: str) -> list[dict[str, Any]]:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, list):
            boundaries: list[dict[str, Any]] = []
            for index, item in enumerate(value, start=1):
                if isinstance(item, dict):
                    boundaries.append(item)
                elif str(item).strip():
                    boundaries.append(
                        {
                            "boundary_id": f"B{index:03d}",
                            "statement": str(item).strip(),
                        }
                    )
            if boundaries:
                return boundaries
    return []


def _capability_id_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        value = [] if value is None else [value]
    result: list[str] = []
    for item in value:
        if isinstance(item, dict):
            for key in ("capability_id", "id", "name"):
                text = str(item.get(key) or "").strip()
                if text:
                    result.append(text)
                    break
        else:
            text = str(item).strip()
            if text:
                result.append(text)
    return result


def _slug(value: str, *, fallback: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or fallback


def _clip(value: str | None, limit: int) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
