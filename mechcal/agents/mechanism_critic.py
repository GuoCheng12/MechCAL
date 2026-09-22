from __future__ import annotations

from typing import Any

from mechcal.capabilities import default_capability_registry
from mechcal.capabilities.evidence_guide import compact_ability_evidence_guide
from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.public_ids import public_case_id_from_metadata
from mechcal.runtime.llm import OpenAIJsonClient
from mechcal.runtime.prompts import load_prompt
from mechcal.schemas import (
    AgentReport,
    CaseRun,
    DifferentialMechanismPortfolio,
    EvidenceUnit,
    MechanismSupportArgument,
)
from mechcal.schemas.mechanisms import (
    DifferentialMechanismPortfolioRow,
    EvidenceMechanismAttribution,
    EvidenceMechanismAttributionUpdate,
)

DEFAULT_CRITIC_EVIDENCE_LIMIT = 8
DEFAULT_CRITIC_SUPPORT_LIMIT = 6
COMPACT_CRITIC_CONTRACT = (
    "Return JSON with keys portfolio_id, case_id, round_id, rows, "
    "evidence_attributions, policy_notes. rows must contain exactly one item for "
    "every mechanism_pool label. Each row needs label, trigger_status "
    "(triggered|weak_trigger|not_triggered|contradicted), differential_priority "
    "0-1, support_strength (unsupported|weak_proxy|partial|strong), claim_status "
    "(candidate_requires_validation|partially_supported|supported|weakened), "
    "positive_evidence_refs, negative_evidence_refs, missing_validation, rationale. "
    "evidence_attributions items need evidence_id and updates. Each update needs "
    "label, priority_effect (increase|decrease|neutral), support_effect "
    "(supports|weakens|boundary_only|not_applicable), warrant, boundary. "
    "Use only labels from mechanism_pool and IDs from allowed_evidence_ids."
)


class MechanismCriticAgent:
    """LLM-driven reviewer for differential mechanism portfolio state."""

    def __init__(
        self,
        *,
        llm_client: OpenAIJsonClient | None = None,
        max_attempts: int = 3,
    ) -> None:
        self.llm_client = llm_client or OpenAIJsonClient()
        self.max_attempts = max(1, max_attempts)

    def review(self, case_run: CaseRun) -> DifferentialMechanismPortfolio:
        if not self.llm_client.is_configured():
            raise RuntimeError("MechanismCriticAgent requires a configured LLM client.")
        round_id = case_run.current_round_id or (
            case_run.round_ids[-1] if case_run.round_ids else "FINAL"
        )
        payload = _critic_payload(case_run, round_id=round_id)
        last_error: str | None = None
        for attempt in range(self.max_attempts):
            try:
                response = self.llm_client.complete_json(
                    system_prompt=load_prompt("mechanism_critic.md"),
                    payload=payload,
                    schema_feedback=last_error,
                )
                return _sanitize_portfolio_response(
                    response,
                    case_run=case_run,
                    round_id=round_id,
                    allowed_evidence_ids=set(payload["allowed_evidence_ids"]),
                )
            except Exception as exc:  # noqa: BLE001
                last_error = (
                    f"Attempt {attempt + 1} failed: {type(exc).__name__}: {exc}"
                )
        raise RuntimeError(
            f"MechanismCriticAgent failed to produce a valid portfolio: {last_error}"
        )

    def synthesize_parallel_worker_summary(
        self,
        case_run: CaseRun,
        *,
        worker_reports: list[AgentReport],
    ) -> DifferentialMechanismPortfolio:
        if not self.llm_client.is_configured():
            raise RuntimeError(
                "Parallel-worker summary synthesis requires a configured LLM client."
            )
        round_id = case_run.current_round_id or (
            case_run.round_ids[-1] if case_run.round_ids else "R001"
        )
        payload = _parallel_worker_summary_payload(
            case_run,
            round_id=round_id,
            worker_reports=worker_reports,
        )
        last_error: str | None = None
        for attempt in range(self.max_attempts):
            try:
                response = self.llm_client.complete_json(
                    system_prompt=load_prompt("parallel_worker_summary_synthesis.md"),
                    payload=payload,
                    schema_feedback=last_error,
                )
                return _sanitize_portfolio_response(
                    response,
                    case_run=case_run,
                    round_id=round_id,
                    allowed_evidence_ids=set(payload["allowed_evidence_ids"]),
                    calibrate_evidence_tier_priority=False,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = (
                    f"Attempt {attempt + 1} failed: {type(exc).__name__}: {exc}"
                )
        raise RuntimeError(
            "Parallel-worker summary synthesis failed to produce a valid portfolio: "
            f"{last_error}"
        )


def _critic_payload(case_run: CaseRun, *, round_id: str) -> dict[str, Any]:
    evidence_items = _select_evidence(case_run.evidence_ledger.items)
    evidence_ids = [item.evidence_id for item in evidence_items]
    return {
        "public_case": {
            "case_label": "current_smiles_case",
            "smiles": case_run.input.smiles,
            "user_query": _clip(case_run.input.user_query, 160),
            "round_id": round_id,
        },
        "mechanism_pool": list(MECHANISM_POOL),
        "generic_route_hints": _generic_route_hints(),
        "mechanism_evidence_coverage_table": _mechanism_evidence_coverage_table(
            evidence_items
        ),
        "planner_state": {
            "current_hypothesis": case_run.portfolio.current,
            "hypotheses": [
                {
                    "name": item.name,
                    "confidence": item.confidence,
                    "differential_priority": item.differential_priority,
                    "evidence_support": item.evidence_support,
                    "claim_status": item.claim_status,
                    "status": item.status,
                    "rationale": _clip(item.rationale, 120),
                    "evidence_refs": list(item.evidence_refs[:4]),
                    "validation_needed": list(item.validation_needed[:3]),
                }
                for item in case_run.portfolio.sorted_hypotheses()[:8]
            ],
        },
        "previous_differential_portfolio": _compact_previous_portfolio(case_run),
        "mechanism_agenda": _compact_mechanism_agenda(case_run),
        "photophysics_review": _compact_photophysics_review(case_run),
        "mechanism_support_arguments": [
            _compact_support_argument(item)
            for item in case_run.mechanism_support_arguments[-DEFAULT_CRITIC_SUPPORT_LIMIT:]
        ],
        "evidence_ledger": [_safe_evidence_payload(item) for item in evidence_items],
        "allowed_evidence_ids": evidence_ids,
        "output_contract": COMPACT_CRITIC_CONTRACT,
        "policy": {
            "private_answer_key_access": "none",
            "external_answer_access": "none",
            "case_specific_search": "forbidden",
            "final_decision_authority": "Planner only",
        },
    }


def _parallel_worker_summary_payload(
    case_run: CaseRun,
    *,
    round_id: str,
    worker_reports: list[AgentReport],
) -> dict[str, Any]:
    payload = _critic_payload(case_run, round_id=round_id)
    for key in (
        "planner_state",
        "previous_differential_portfolio",
        "mechanism_agenda",
        "photophysics_review",
        "mechanism_support_arguments",
    ):
        payload.pop(key, None)
    payload["task"] = "parallel_worker_summary_one_shot_synthesis"
    payload["output_contract"] = (
        "Return JSON with keys rows, evidence_attributions, policy_notes. rows "
        "must contain the 3-6 most relevant candidates selected from "
        "mechanism_pool. Each row needs label, trigger_status, "
        "differential_priority as a decimal from 0 to 1, "
        "support_strength, claim_status, positive_evidence_refs, "
        "negative_evidence_refs, missing_validation, rationale. Keep rationale "
        "under 18 words and missing_validation to at most one short item. "
        "evidence_attributions may be an empty list. policy_notes may be empty."
    )
    payload["worker_reports"] = [
        {
            "agent_name": report.agent_name,
            "round_id": report.round_id,
            "status": report.status,
            "planner_readable_report": _clip(report.planner_readable_report, 320),
            "policy_notes": [_clip(item, 120) for item in report.policy_notes[:3]],
            "evidence_ids": [
                item.evidence_id
                for item in report.evidence_units
                if item.evidence_id in payload["allowed_evidence_ids"]
            ],
        }
        for report in worker_reports
    ]
    payload["policy"] = {
        "private_answer_key_access": "none",
        "external_answer_access": "none",
        "case_specific_search": "forbidden",
        "planner_present": False,
        "coverage_reviewer_present": False,
        "support_auditor_present": False,
        "synthesis_scope": "one shot from the two supplied worker reports only",
    }
    return payload


def _select_evidence(
    items: list[EvidenceUnit],
    *,
    limit: int = DEFAULT_CRITIC_EVIDENCE_LIMIT,
) -> list[EvidenceUnit]:
    if len(items) <= limit:
        return list(items)
    selected: list[EvidenceUnit] = []
    selected_ids: set[str] = set()
    seen_capability: set[str] = set()
    for item in reversed(items):
        capability_key = item.capability_id or item.agent_name or item.family
        if capability_key in seen_capability:
            continue
        selected.append(item)
        selected_ids.add(item.evidence_id)
        seen_capability.add(capability_key)
        if len(selected) >= limit // 2:
            break
    for item in reversed(items):
        if item.evidence_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item.evidence_id)
        if len(selected) >= limit:
            break
    return sorted(selected, key=lambda item: (item.round_id, item.evidence_id))


def _safe_evidence_payload(item: EvidenceUnit) -> dict[str, Any]:
    metrics = {
        str(key): _safe_metric_value(value)
        for key, value in item.metrics.items()
        if key not in {"smiles", "prepared_file_paths"}
    }
    return {
        "evidence_id": item.evidence_id,
        "round_id": item.round_id,
        "agent_name": item.agent_name,
        "capability_id": item.capability_id,
        "claim": _clip(item.claim, 160),
        "summary": _clip(item.summary, 120),
        "basis": item.basis,
        "support": item.support,
        "family": item.family,
        "status": item.status,
        "observable_tags": list(item.observable_tags[:5]),
        "limits": [_clip(limit, 80) for limit in item.limits[:2]],
        "metrics": dict(list(metrics.items())[:8]),
    }


def _mechanism_evidence_coverage_table(
    evidence_items: list[EvidenceUnit],
) -> list[dict[str, Any]]:
    route_map = _route_to_mechanisms()
    evidence_tier_by_capability = _evidence_tier_by_capability()
    rows: dict[str, dict[str, Any]] = {
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
    for item in evidence_items:
        labels = route_map.get(item.capability_id, [])
        for label in labels:
            row = rows.get(label)
            if row is None:
                continue
            row["route_hits"].append(
                {
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
                    "claim": _clip(item.claim, 180),
                    "observable_tags": list(item.observable_tags[:6]),
                }
            )
            row["evidence_refs"] = list(
                dict.fromkeys([*row["evidence_refs"], item.evidence_id])
            )[-8:]
            if _evidence_signal_type(item) == "positive_or_proxy_observable":
                row["positive_or_proxy_signal_count"] += 1
            if _evidence_signal_type(item) in {
                "missing_or_failed",
                "checklist_boundary_only",
            }:
                row["failed_or_unsupported_route_count"] += 1
            row["latest_signal_summary"] = _clip(item.claim or item.summary, 220)
    return [row for row in rows.values() if row["route_hits"]]


def _evidence_tier_by_capability() -> dict[str, str]:
    return {
        str(item.get("capability_id")): str(item.get("evidence_tier") or "unknown")
        for item in compact_ability_evidence_guide(default_capability_registry())
    }


def _evidence_signal_type(item: EvidenceUnit) -> str:
    if item.status in {"failed", "unsupported", "missing"}:
        return "missing_or_failed"
    text = f"{item.claim} {item.summary}".lower()
    if item.basis == "checklist" or any(
        term in text
        for term in (
            "records which",
            "checklist",
            "absence",
            "absent",
            "not present",
            "missing",
        )
    ):
        return "checklist_boundary_only"
    return "positive_or_proxy_observable"


def _route_to_mechanisms() -> dict[str, list[str]]:
    return {
        "macro.screen_esipt_structural_motif": ["ESIPT_PT"],
        "macro.screen_intramolecular_hbond_preorganization": ["ESIPT_PT"],
        "macro.screen_metal_triplet_prior": ["TRIPLET_METAL_ENERGY_TRANSFER"],
        "macro.screen_rotor_torsion_topology": [
            "RIM_RIR_RIV",
            "RACI_CI_ACCESS",
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
        "microscopic.extract_ct_descriptors_from_bundle": ["ICT_TICT_CT", "PET_ET"],
    }


def _compact_support_argument(item: MechanismSupportArgument) -> dict[str, Any]:
    return {
        "support_id": item.support_id,
        "target_label": item.target_label,
        "support_level": item.support_level,
        "finding": _clip(item.finding, 180),
        "warrant": _clip(item.warrant, 220),
        "boundary": _clip(item.boundary, 180),
        "observation_refs": list(item.observation_refs[:5]),
        "audit_status": item.audit_status,
        "audit_notes": list(item.audit_notes[:2]),
    }


def _compact_previous_portfolio(case_run: CaseRun) -> list[dict[str, Any]]:
    portfolio = case_run.differential_mechanism_portfolio
    if portfolio is None:
        return []
    return [
        {
            "label": row.label,
            "trigger_status": row.trigger_status,
            "differential_priority": row.differential_priority,
            "support_strength": row.support_strength,
            "claim_status": row.claim_status,
            "positive_evidence_refs": list(row.positive_evidence_refs[:5]),
            "negative_evidence_refs": list(row.negative_evidence_refs[:5]),
            "missing_validation": list(row.missing_validation[:4]),
            "rationale": _clip(row.rationale, 180),
        }
        for row in portfolio.rows
    ]


def _compact_mechanism_agenda(case_run: CaseRun) -> dict[str, Any] | None:
    program = case_run.mechanism_program
    if program is None:
        return None
    return {
        "candidate_mechanisms": [
            {
                "label": item.label,
                "mechanism_family": item.mechanism_family,
                "rationale": _clip(item.rationale, 100),
            }
            for item in program.candidate_mechanisms[:8]
        ],
        "evidence_questions": [
            {
                "question_id": item.question_id,
                "mechanism_id": item.mechanism_id,
                "observable": _clip(item.observable, 120),
                "status": item.status,
                "acceptable_capability_ids": list(item.acceptable_capability_ids[:4]),
            }
            for item in program.evidence_questions[:8]
        ],
    }


def _compact_photophysics_review(case_run: CaseRun) -> dict[str, Any] | None:
    review = case_run.photophysics_review
    if review is None:
        return None
    return {
        "coverage_axes": [
            {
                "label": item.label,
                "status": item.status,
                "evidence_refs": list(item.evidence_refs[:4]),
                "rationale": _clip(item.rationale, 140),
            }
            for item in review.coverage_axes[:5]
        ],
        "hypothesis_cards": [
            {
                "mechanism": item.mechanism,
                "status": item.status,
                "support_evidence_refs": list(item.support_evidence_refs[:4]),
                "weakening_evidence_refs": list(item.weakening_evidence_refs[:4]),
                "missing_or_unresolved": list(item.missing_or_unresolved[:4]),
                "reasoning_summary": _clip(item.reasoning_summary, 140),
                "priority": item.priority,
            }
            for item in review.hypothesis_cards[:5]
        ],
        "overclaim_warnings": list(review.overclaim_warnings[:4]),
        "source_evidence_refs": list(review.source_evidence_refs[:6]),
    }


def _sanitize_portfolio_response(
    response: dict[str, Any],
    *,
    case_run: CaseRun,
    round_id: str,
    allowed_evidence_ids: set[str],
    calibrate_evidence_tier_priority: bool = True,
) -> DifferentialMechanismPortfolio:
    payload = _first_mapping(response, "differential_mechanism_portfolio", "portfolio")
    raw = payload if payload is not None else response
    raw_rows = _first_dict_list(raw, "rows", "portfolio_rows", "mechanism_rows")
    rows = [
        _normalize_row(item, allowed_evidence_ids=allowed_evidence_ids)
        for item in raw_rows
    ]
    if calibrate_evidence_tier_priority:
        rows = _apply_evidence_tier_priority_calibration(
            rows,
            evidence_by_id={
                item.evidence_id: item
                for item in case_run.evidence_ledger.items
                if item.evidence_id in allowed_evidence_ids
            },
        )
    labels = {row["label"] for row in rows}
    missing = [label for label in MECHANISM_POOL if label not in labels]
    extra = [label for label in labels if label not in MECHANISM_POOL]
    if extra:
        raise ValueError(
            "MechanismCritic rows must cover mechanism_pool exactly; "
            f"missing={missing}, extra={extra}"
        )
    if missing:
        rows.extend(
            _carry_forward_missing_rows(
                missing,
                case_run=case_run,
                allowed_evidence_ids=allowed_evidence_ids,
            )
        )
    attributions = [
        _normalize_attribution(item, allowed_evidence_ids=allowed_evidence_ids)
        for item in _first_dict_list(raw, "evidence_attributions", "attributions")
    ]
    public_case_id = public_case_id_from_metadata(case_run.input.metadata)
    return DifferentialMechanismPortfolio(
        portfolio_id=f"{public_case_id}:{round_id}:differential_mechanism_portfolio",
        case_id=public_case_id,
        round_id=round_id,
        rows=[DifferentialMechanismPortfolioRow.model_validate(row) for row in rows],
        evidence_attributions=[
            EvidenceMechanismAttribution.model_validate(item)
            for item in attributions
        ],
        policy_notes=_as_text_list(raw.get("policy_notes"))[:4],
    )


def _carry_forward_missing_rows(
    missing_labels: list[str],
    *,
    case_run: CaseRun,
    allowed_evidence_ids: set[str],
) -> list[dict[str, Any]]:
    previous_by_label = {
        row.label: row
        for row in (
            case_run.differential_mechanism_portfolio.rows
            if case_run.differential_mechanism_portfolio is not None
            else []
        )
    }
    carried: list[dict[str, Any]] = []
    for label in missing_labels:
        previous = previous_by_label.get(label)
        if previous is not None:
            carried.append(
                {
                    "label": label,
                    "trigger_status": previous.trigger_status,
                    "differential_priority": previous.differential_priority,
                    "support_strength": previous.support_strength,
                    "claim_status": previous.claim_status,
                    "positive_evidence_refs": [
                        ref
                        for ref in previous.positive_evidence_refs
                        if ref in allowed_evidence_ids
                    ],
                    "negative_evidence_refs": [
                        ref
                        for ref in previous.negative_evidence_refs
                        if ref in allowed_evidence_ids
                    ],
                    "missing_validation": list(previous.missing_validation[:5]),
                    "rationale": (
                        previous.rationale
                        or "MechanismCritic omitted this row; previous state was retained."
                    ),
                }
            )
            continue
        carried.append(
            {
                "label": label,
                "trigger_status": "not_triggered",
                "differential_priority": 0.0,
                "support_strength": "unsupported",
                "claim_status": "candidate_requires_validation",
                "positive_evidence_refs": [],
                "negative_evidence_refs": [],
                "missing_validation": [
                    "No source-grounded runtime evidence was assigned to this mechanism."
                ],
                "rationale": (
                    "MechanismCritic omitted this mechanism-pool row; sanitizer "
                    "retained a neutral unsupported boundary without adding "
                    "scientific evidence."
                ),
            }
        )
    return carried


def _normalize_row(
    item: dict[str, Any],
    *,
    allowed_evidence_ids: set[str],
) -> dict[str, Any]:
    row = _strip_mapping_keys(item)
    label = _normalize_label(row.get("label") or row.get("mechanism") or row.get("name"))
    positive_refs = _filter_refs(row.get("positive_evidence_refs"), allowed_evidence_ids)
    if not positive_refs:
        positive_refs = _filter_refs(row.get("evidence_refs"), allowed_evidence_ids)
    negative_refs = _filter_refs(row.get("negative_evidence_refs"), allowed_evidence_ids)
    support_strength = _normalize_support_strength(row.get("support_strength"))
    claim_status = _normalize_claim_status(row.get("claim_status"))
    if not positive_refs and support_strength in {"partial", "strong"}:
        support_strength = "weak_proxy"
    if not positive_refs and claim_status in {"supported", "partially_supported"}:
        claim_status = "candidate_requires_validation"
    if negative_refs and not positive_refs and row.get("claim_status") == "weakened":
        claim_status = "weakened"
    return {
        "label": label,
        "trigger_status": _normalize_trigger_status(row.get("trigger_status")),
        "differential_priority": _bounded_float(
            row.get("differential_priority")
            or row.get("priority")
            or row.get("plausibility_confidence")
            or row.get("confidence")
        ),
        "support_strength": support_strength,
        "claim_status": claim_status,
        "positive_evidence_refs": positive_refs,
        "negative_evidence_refs": negative_refs,
        "missing_validation": _as_text_list(
            row.get("missing_validation") or row.get("validation_needed")
        )[:5],
        "rationale": _clip(row.get("rationale"), 320)
        or "MechanismCritic supplied no rationale beyond current runtime evidence.",
    }


def _apply_evidence_tier_priority_calibration(
    rows: list[dict[str, Any]],
    *,
    evidence_by_id: dict[str, EvidenceUnit],
) -> list[dict[str, Any]]:
    evidence_tier_by_capability = _evidence_tier_by_capability()
    updated: list[dict[str, Any]] = []
    for row in rows:
        support_strength = str(row.get("support_strength") or "unsupported")
        trigger_status = str(row.get("trigger_status") or "not_triggered")
        positive_refs = [
            ref
            for ref in _as_text_list(row.get("positive_evidence_refs"))
            if ref in evidence_by_id
        ]
        if not positive_refs or support_strength == "unsupported":
            current_priority = _bounded_float(row.get("differential_priority"))
            capped_priority = min(current_priority, 0.05)
            if capped_priority == current_priority:
                updated.append(row)
            else:
                updated.append({**row, "differential_priority": capped_priority})
            continue
        if trigger_status not in {"triggered", "weak_trigger"}:
            updated.append(row)
            continue
        tiers = [
            evidence_tier_by_capability.get(
                evidence_by_id[ref].capability_id,
                "unknown",
            )
            for ref in positive_refs
        ]
        capability_ids = {
            evidence_by_id[ref].capability_id
            for ref in positive_refs
        }
        floor = max(
            (_priority_floor_for_evidence_tier(tier) for tier in tiers),
            default=0.0,
        )
        if floor <= 0.0:
            updated.append(row)
            continue
        current_priority = _bounded_float(row.get("differential_priority"))
        cap = max((_priority_cap_for_evidence_tier(tier) for tier in tiers), default=1.0)
        evidence_bonus = min(0.04, 0.02 * max(len(positive_refs) - 1, 0))
        route_diversity_bonus = min(0.03, 0.015 * max(len(capability_ids) - 1, 0))
        calibrated_priority = min(
            max(current_priority, floor) + evidence_bonus + route_diversity_bonus,
            cap,
        )
        if calibrated_priority == current_priority:
            updated.append(row)
            continue
        updated.append({**row, "differential_priority": calibrated_priority})
    return updated


def _priority_floor_for_evidence_tier(tier: str) -> float:
    return {
        "computed_direct_proxy": 0.22,
        "mechanism_specific_structural_trigger": 0.18,
        "structural_trigger": 0.12,
        "electronic_weak_trigger": 0.10,
        "artifact_weak_trigger": 0.08,
    }.get(tier, 0.0)


def _priority_cap_for_evidence_tier(tier: str) -> float:
    return {
        "computed_direct_proxy": 0.30,
        "mechanism_specific_structural_trigger": 0.24,
        "structural_trigger": 0.18,
        "electronic_weak_trigger": 0.16,
        "artifact_weak_trigger": 0.12,
    }.get(tier, 1.0)


def _normalize_attribution(
    item: dict[str, Any],
    *,
    allowed_evidence_ids: set[str],
) -> dict[str, Any]:
    raw = _strip_mapping_keys(item)
    evidence_id = str(raw.get("evidence_id") or "").strip()
    if evidence_id not in allowed_evidence_ids:
        raise ValueError(f"MechanismCritic cited unavailable evidence_id: {evidence_id}")
    updates = [
        EvidenceMechanismAttributionUpdate.model_validate(
            {
                "label": _normalize_label(update.get("label") or update.get("mechanism")),
                "priority_effect": _normalize_priority_effect(
                    update.get("priority_effect")
                ),
                "support_effect": _normalize_support_effect(update.get("support_effect")),
                "warrant": _clip(update.get("warrant"), 260)
                or "Runtime evidence was assigned to this mechanism.",
                "boundary": _clip(update.get("boundary"), 220)
                or "Boundary was not specified by the critic.",
            }
        ).model_dump(mode="json")
        for update in _as_dicts(raw.get("updates"))
        if _normalize_label(update.get("label") or update.get("mechanism"))
    ]
    return {"evidence_id": evidence_id, "updates": updates}


def _normalize_label(value: Any) -> str:
    text = str(value or "").strip()
    aliases = {
        "RIM": "RIM_RIR_RIV",
        "RIR": "RIM_RIR_RIV",
        "RIV": "RIM_RIR_RIV",
        "PACKING": "PACKING_HOST_MATRIX_CONFINEMENT",
        "HOST_GUEST": "HOST_GUEST_INTERACTION",
        "ICT": "ICT_TICT_CT",
        "TICT": "ICT_TICT_CT",
        "CT": "ICT_TICT_CT",
        "ESIPT": "ESIPT_PT",
        "PT": "ESIPT_PT",
        "PET": "PET_ET",
        "ET": "PET_ET",
        "AGGREGATE": "AGGREGATE_EXCITON_EXCIMER",
        "EXCIMER": "AGGREGATE_EXCITON_EXCIMER",
        "RADIATIVE_RATE": "RADIATIVE_RATE_STATE_BALANCE",
        "TRIPLET": "TRIPLET_METAL_ENERGY_TRANSFER",
        "METAL": "TRIPLET_METAL_ENERGY_TRANSFER",
        "RACI": "RACI_CI_ACCESS",
        "CI": "RACI_CI_ACCESS",
        "SOKR": "SOKR_ANTI_KASHA",
        "ANTI_KASHA": "SOKR_ANTI_KASHA",
    }
    return aliases.get(text.upper(), text)


def _normalize_trigger_status(value: Any) -> str:
    text = str(value or "not_triggered").strip().lower()
    aliases = {
        "trigger": "triggered",
        "present": "triggered",
        "weak": "weak_trigger",
        "possible": "weak_trigger",
        "plausible": "weak_trigger",
        "candidate": "weak_trigger",
        "absent": "not_triggered",
        "unsupported": "not_triggered",
        "countered": "contradicted",
        "contrary": "contradicted",
    }
    normalized = aliases.get(text, text)
    return normalized if normalized in {
        "triggered",
        "weak_trigger",
        "not_triggered",
        "contradicted",
    } else "not_triggered"


def _normalize_support_strength(value: Any) -> str:
    text = str(value or "unsupported").strip().lower()
    aliases = {
        "weak": "weak_proxy",
        "proxy": "weak_proxy",
        "weak_or_proxy": "weak_proxy",
        "proxy_only": "weak_proxy",
        "computed_supported": "strong",
        "supported": "strong",
        "not_supported": "unsupported",
    }
    normalized = aliases.get(text, text)
    return normalized if normalized in {
        "unsupported",
        "weak_proxy",
        "partial",
        "strong",
    } else "unsupported"


def _normalize_claim_status(value: Any) -> str:
    text = str(value or "candidate_requires_validation").strip().lower()
    aliases = {
        "candidate": "candidate_requires_validation",
        "weak_candidate": "candidate_requires_validation",
        "underdetermined": "candidate_requires_validation",
        "partial": "partially_supported",
        "partially_supported_candidate": "partially_supported",
        "supported_claim": "supported",
        "rejected": "weakened",
        "contradicted": "weakened",
    }
    normalized = aliases.get(text, text)
    return normalized if normalized in {
        "candidate_requires_validation",
        "partially_supported",
        "supported",
        "weakened",
    } else "candidate_requires_validation"


def _normalize_priority_effect(value: Any) -> str:
    text = str(value or "neutral").strip().lower()
    aliases = {"up": "increase", "positive": "increase", "down": "decrease", "negative": "decrease"}
    normalized = aliases.get(text, text)
    return normalized if normalized in {"increase", "decrease", "neutral"} else "neutral"


def _normalize_support_effect(value: Any) -> str:
    text = str(value or "not_applicable").strip().lower()
    aliases = {
        "support": "supports",
        "weaken": "weakens",
        "boundary": "boundary_only",
        "none": "not_applicable",
    }
    normalized = aliases.get(text, text)
    return normalized if normalized in {
        "supports",
        "weakens",
        "boundary_only",
        "not_applicable",
    } else "not_applicable"


def _filter_refs(value: Any, allowed_evidence_ids: set[str]) -> list[str]:
    return [
        ref
        for ref in _as_text_list(value)
        if ref in allowed_evidence_ids
    ]


def _bounded_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if 1.0 < number <= 100.0:
        number /= 100.0
    return min(max(number, 0.0), 1.0)


def _first_mapping(payload: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return None


def _first_dict_list(payload: dict[str, Any], *keys: str) -> list[dict[str, Any]]:
    for key in keys:
        values = _as_dicts(payload.get(key))
        if values:
            return values
    return []


def _as_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in raw_items if str(item).strip()]


def _strip_mapping_keys(payload: dict[str, Any]) -> dict[str, Any]:
    return {str(key).strip(): value for key, value in payload.items()}


def _safe_metric_value(value: Any) -> Any:
    if isinstance(value, str):
        return _clip(value, 100)
    if isinstance(value, int | float | bool) or value is None:
        return value
    if isinstance(value, list):
        return [_safe_metric_value(item) for item in value[:5]]
    if isinstance(value, dict):
        return {
            str(key): _safe_metric_value(item)
            for key, item in list(value.items())[:6]
        }
    return _clip(str(value), 100)


def _clip(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _generic_route_hints() -> dict[str, list[str]]:
    return {
        "RIM_RIR_RIV": [
            "rotor/torsion topology, conformer sensitivity, torsion brightness proxy",
            "missing: viscosity/pressure/aggregate restriction or direct nonradiative evidence",
        ],
        "PACKING_HOST_MATRIX_CONFINEMENT": [
            "crystal restriction checklist, dimer/aggregate contact proxy, solid-state proxy",
            "missing: packing, matrix, pressure, framework, or morphology observation",
        ],
        "HOST_GUEST_INTERACTION": [
            "host/pore/accessibility or guest-specific interaction evidence when available",
            "missing: uptake, binding, in-situ spectra, pore/guest occupancy",
        ],
        "ICT_TICT_CT": [
            "donor-acceptor architecture, CT descriptors, charge redistribution, polarity proxy",
            "missing: excited-state CT/NTO/electron-hole separation or solvatochromism",
        ],
        "ESIPT_PT": [
            "proton donor/acceptor and intramolecular H-bond preorganization proxies",
            "missing: FPT/RPT barrier, dual emission, isotope/time-resolved evidence",
        ],
        "PET_ET": [
            "receptor/fluorophore ET pathway proxies and frontier/charge alignment evidence",
            "missing: redox alignment, binding-state comparison, ET quenching evidence",
        ],
        "AGGREGATE_EXCITON_EXCIMER": [
            (
                "pi-stacking/dimer contact proxies may raise candidate priority, "
                "but support remains weak until aggregate excited-state evidence appears"
            ),
            (
                "missing: new aggregate band, exciton coupling, dimer/excimer "
                "excited-state calculation"
            ),
        ],
        "RADIATIVE_RATE_STATE_BALANCE": [
            "bright/dark state ordering, oscillator strength, transition-dipole proxies",
            "missing: lifetime/QY/radiative-rate or state population evidence",
        ],
        "TRIPLET_METAL_ENERGY_TRANSFER": [
            (
                "metal/heavy-atom, sulfur/phosphorus triplet priors, or "
                "triplet/lanthanide energy-transfer cues when observed"
            ),
            "missing: triplet energy, RTP/TADF/lanthanide lines, time-resolved evidence",
        ],
        "RACI_CI_ACCESS": [
            "torsion/flapping brightness-coupling or nonradiative-risk proxies",
            "missing: explicit S1/S0 crossing, conical-intersection search, nonadiabatic dynamics",
        ],
        "SOKR_ANTI_KASHA": [
            "higher-state brightness, S1 dark/Sn bright, state ordering proxy",
            "missing: anti-Kasha emission, excitation dependence, time-resolved state assignment",
        ],
    }
