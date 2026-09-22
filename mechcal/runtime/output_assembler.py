from __future__ import annotations

from mechcal.capabilities import default_capability_registry
from mechcal.capabilities.evidence_guide import compact_ability_evidence_guide
from mechcal.schemas import (
    CaseRun,
    MechanismPrediction,
    MechanismPredictionEvidence,
    MechanismSupportArgument,
)

LOW_MARGIN_THRESHOLD = 0.05
WEAK_CT_PROXY_PRIORITY_CEILING = 0.20

_ABILITY_GUIDE_BY_CAPABILITY = {
    str(item["capability_id"]): item
    for item in compact_ability_evidence_guide(default_capability_registry())
}
_LOW_MARGIN_ORDERING_TIERS = {
    "computed_direct_proxy",
    "mechanism_specific_structural_trigger",
    "electronic_weak_trigger",
}
PlannerPredictionCandidate = tuple[
    object,
    list[MechanismSupportArgument],
    list[str],
    list[MechanismPredictionEvidence],
    list[str],
]


class OutputAssembler:
    """Deterministic packager for Planner-owned mechanism rankings.

    This module does not add mechanisms, change confidence, or judge evidence
    strength. It packages Planner portfolio entries into final predictions and
    applies source-grounding safety ordering so unsupported/no-ref candidates do
    not outrank source-grounded candidates in the exported Top-K.
    """

    def build_mechanism_predictions(
        self,
        case_run: CaseRun,
        *,
        top_k: int = 3,
    ) -> list[MechanismPrediction]:
        evidence_by_id = {
            item.evidence_id: item
            for item in case_run.evidence_ledger.items
        }
        support_by_label = _support_arguments_by_label(case_run)
        attribution_by_label = _attributions_by_label(case_run)
        predictions: list[MechanismPrediction] = []
        seen_labels: set[str] = set()
        candidates: list[PlannerPredictionCandidate] = []
        for hypothesis in case_run.portfolio.hypotheses:
            label = hypothesis.name.strip()
            if not label or label.lower() == "unknown" or label in seen_labels:
                continue
            differential_priority = (
                hypothesis.differential_priority
                if hypothesis.differential_priority is not None
                else hypothesis.confidence
            )
            support_arguments = support_by_label.get(label, [])
            refs = [
                ref
                for argument in support_arguments
                for ref in argument.observation_refs
                if ref in evidence_by_id
            ]
            refs = list(dict.fromkeys(refs))
            evidence_text = [
                _support_argument_text(argument)
                for argument in support_arguments
            ]
            evidence_details = [
                detail
                for argument in support_arguments
                for detail in _support_argument_details(
                    argument,
                    label=label,
                    evidence_by_id=evidence_by_id,
                )
            ]
            evidence_details = _ordered_evidence_details_for_label(label, evidence_details)
            if evidence_details:
                evidence_text = [
                    _prediction_evidence_text(item) for item in evidence_details
                ]
            extra_refs = [
                ref
                for ref in hypothesis.evidence_refs
                if ref in evidence_by_id and ref not in refs
            ]
            if extra_refs:
                extra_details = _row_evidence_details(
                    label,
                    extra_refs,
                    evidence_by_id=evidence_by_id,
                    attribution_by_label=attribution_by_label,
                    fallback_boundary=(
                        hypothesis.validation_needed[0]
                        if hypothesis.validation_needed
                        else (
                            "Evidence is bounded by public runtime observations "
                            "and remains incomplete."
                        )
                    ),
                    fallback_warrant=hypothesis.rationale
                    or "The cited runtime evidence is a bounded proxy for this mechanism.",
                )
                evidence_details.extend(extra_details)
                evidence_details = _ordered_evidence_details_for_label(
                    label,
                    evidence_details,
                )
                refs = list(dict.fromkeys([*refs, *extra_refs]))[:6]
                evidence_text = [
                    _prediction_evidence_text(item) for item in evidence_details
                ]
            screened_refs = (
                _screened_evidence_refs_for_label(
                    label,
                    evidence_by_id=evidence_by_id,
                    excluded_refs=refs,
                    limit=max(3 - len(evidence_details), 0),
                )
                if refs or evidence_details
                else []
            )
            if screened_refs:
                screened_details = _row_evidence_details(
                    label,
                    screened_refs,
                    evidence_by_id=evidence_by_id,
                    attribution_by_label=attribution_by_label,
                    fallback_boundary=(
                        hypothesis.validation_needed[0]
                        if hypothesis.validation_needed
                        else (
                            "Evidence is bounded by public runtime observations "
                            "and remains incomplete."
                        )
                    ),
                    fallback_warrant=hypothesis.rationale
                    or "The cited runtime evidence is a bounded proxy for this mechanism.",
                )
                evidence_details.extend(screened_details)
                evidence_details = _ordered_evidence_details_for_label(
                    label,
                    evidence_details,
                )
                refs = list(dict.fromkeys([*refs, *screened_refs]))[:6]
                evidence_text = [
                    _prediction_evidence_text(item) for item in evidence_details
                ]
            if not evidence_details:
                refs = _refs_from_attributions(
                    label,
                    attribution_by_label=attribution_by_label,
                    evidence_by_id=evidence_by_id,
                )
                evidence_details = _row_evidence_details(
                    label,
                    refs,
                    evidence_by_id=evidence_by_id,
                    attribution_by_label=attribution_by_label,
                    fallback_boundary=(
                        hypothesis.validation_needed[0]
                        if hypothesis.validation_needed
                        else (
                            "Evidence is bounded by public runtime observations "
                            "and remains incomplete."
                        )
                    ),
                    fallback_warrant=hypothesis.rationale,
                )
                evidence_details = _ordered_evidence_details_for_label(
                    label,
                    evidence_details,
                )
                evidence_text = [
                    _prediction_evidence_text(item) for item in evidence_details
                ]
            candidates.append(
                (hypothesis, support_arguments, refs, evidence_details, evidence_text)
            )
            seen_labels.add(label)

        ordered_candidates = _source_grounded_planner_candidates(
            candidates,
            current=case_run.portfolio.current,
            runner_up=case_run.portfolio.runner_up,
            evidence_by_id=evidence_by_id,
        )
        for hypothesis, support_arguments, refs, evidence_details, evidence_text in (
            ordered_candidates
        ):
            label = hypothesis.name.strip()
            differential_priority = (
                hypothesis.differential_priority
                if hypothesis.differential_priority is not None
                else hypothesis.confidence
            )
            differential_priority = _adjusted_candidate_priority(
                hypothesis,
                refs=refs,
                evidence_by_id=evidence_by_id,
            )
            limitations = _prediction_limitations(case_run, support_arguments)
            support_strength = _support_strength(support_arguments)
            if not support_arguments and evidence_details:
                support_strength = "weak_or_proxy"
            claim_status = _claim_status(support_arguments)
            if not support_arguments and evidence_details:
                claim_status = "candidate_requires_validation"
            predictions.append(
                MechanismPrediction(
                    label=label,
                    rank=len(predictions) + 1,
                    confidence=differential_priority,
                    differential_priority=differential_priority,
                    plausibility_confidence=differential_priority,
                    claim_status=claim_status,
                    support_strength=support_strength,
                    evidence_support=support_strength,
                    evidence_refs=refs,
                    evidence=evidence_text,
                    evidence_details=evidence_details,
                    validation_needed=_validation_needed(
                        support_arguments,
                        hypothesis_validation_needed=hypothesis.validation_needed,
                    ),
                    limitations=limitations,
                )
            )
            seen_labels.add(label)
            if len(predictions) >= top_k:
                break
        return _annotate_low_margin_predictions(predictions)


def _source_grounded_planner_candidates(
    candidates: list[PlannerPredictionCandidate],
    *,
    current: str,
    runner_up: str | None,
    evidence_by_id,
) -> list[PlannerPredictionCandidate]:
    return [
        item
        for _, item in sorted(
            enumerate(candidates),
            key=lambda indexed: _planner_candidate_output_rank(
                indexed[1],
                index=indexed[0],
                current=current,
                runner_up=runner_up,
                evidence_by_id=evidence_by_id,
            ),
            reverse=True,
        )
    ]


def _planner_candidate_output_rank(
    candidate: PlannerPredictionCandidate,
    *,
    index: int,
    current: str,
    runner_up: str | None,
    evidence_by_id,
) -> tuple[int, float, int]:
    del current, runner_up
    hypothesis, _, refs, evidence_details, _ = candidate
    has_source_grounding = bool(refs or evidence_details)
    priority = _adjusted_candidate_priority(
        hypothesis,
        refs=refs,
        evidence_by_id=evidence_by_id,
    )
    return (
        1 if has_source_grounding else 0,
        priority,
        -index,
    )


def _planner_ordered_hypotheses(case_run: CaseRun):
    current = case_run.portfolio.current
    runner_up = case_run.portfolio.runner_up
    indexed = [
        (index, item)
        for index, item in enumerate(case_run.portfolio.hypotheses)
    ]
    return [
        item
        for _, item in sorted(
            indexed,
            key=lambda pair: (
                _hypothesis_priority(pair[1]),
                pair[1].name == current,
                bool(runner_up) and pair[1].name == runner_up,
                -pair[0],
            ),
            reverse=True,
        )
    ]


def _has_reviewed_differential_evidence(case_run: CaseRun) -> bool:
    portfolio = case_run.differential_mechanism_portfolio
    if portfolio is None:
        return False
    evidence_ids = {item.evidence_id for item in case_run.evidence_ledger.items}
    return any(
        ref in evidence_ids
        for row in portfolio.rows
        if row.support_strength != "unsupported"
        for ref in row.positive_evidence_refs
    )


def _hypothesis_priority(hypothesis) -> float:  # noqa: ANN001
    if hypothesis.differential_priority is not None:
        return hypothesis.differential_priority
    return hypothesis.confidence


def _adjusted_candidate_priority(
    hypothesis,  # noqa: ANN001
    *,
    refs: list[str],
    evidence_by_id,
) -> float:
    label = hypothesis.name.strip()
    priority = _hypothesis_priority(hypothesis)
    rim_floor = _rim_ref_priority_floor(label, refs, evidence_by_id)
    if rim_floor is not None:
        priority = max(priority, rim_floor)
    polar_cap = _weak_polar_only_ref_cap(label, refs, evidence_by_id)
    if polar_cap is not None:
        return min(priority, polar_cap)
    weak_ct_cap = _weak_ct_ref_cap(label, refs, evidence_by_id)
    if weak_ct_cap is not None:
        return min(priority, weak_ct_cap)
    radiative_cap = _radiative_single_state_ref_cap(label, refs, evidence_by_id)
    if radiative_cap is not None:
        return min(priority, radiative_cap)
    if label == "PACKING_HOST_MATRIX_CONFINEMENT" and _generic_packing_only_refs(
        refs,
        evidence_by_id,
    ):
        return min(priority, 0.22)
    return priority


def _build_from_differential_portfolio(
    case_run: CaseRun,
    *,
    top_k: int,
) -> list[MechanismPrediction]:
    portfolio = case_run.differential_mechanism_portfolio
    if portfolio is None:
        return []
    evidence_by_id = {item.evidence_id: item for item in case_run.evidence_ledger.items}
    attribution_by_label = _attributions_by_label(case_run)
    predictions: list[MechanismPrediction] = []
    for row in _low_margin_evidence_ordered_rows(portfolio.rows, evidence_by_id):
        if row.trigger_status == "contradicted" and row.differential_priority <= 0.0:
            continue
        refs = [ref for ref in row.positive_evidence_refs if ref in evidence_by_id]
        support_strength = _row_support_strength(row.support_strength, has_refs=bool(refs))
        claim_status = _row_claim_status(
            row.claim_status,
            support_strength=support_strength,
            has_refs=bool(refs),
        )
        evidence_details = _row_evidence_details(
            row.label,
            refs,
            evidence_by_id=evidence_by_id,
            attribution_by_label=attribution_by_label,
            fallback_boundary=_row_boundary(row),
            fallback_warrant=row.rationale,
        )
        evidence_text = [
            _prediction_evidence_text(item) for item in evidence_details
        ] or [_row_no_ref_evidence_text(row)]
        predictions.append(
            MechanismPrediction(
                label=row.label,
                rank=len(predictions) + 1,
                confidence=row.differential_priority,
                differential_priority=row.differential_priority,
                plausibility_confidence=row.differential_priority,
                claim_status=claim_status,
                support_strength=support_strength,
                evidence_support=support_strength,
                evidence_refs=refs,
                evidence=evidence_text,
                evidence_details=evidence_details,
                validation_needed=list(row.missing_validation[:4]) or [
                    "Additional source-grounded validation is required before a stronger claim."
                ],
                limitations=_row_limitations(row),
            )
        )
        if len(predictions) >= top_k:
            break
    return predictions


def _low_margin_evidence_ordered_rows(rows, evidence_by_id):  # noqa: ANN001
    priority_ordered = sorted(
        rows,
        key=lambda item: item.differential_priority,
        reverse=True,
    )
    ordered = []
    index = 0
    while index < len(priority_ordered):
        group = [priority_ordered[index]]
        index += 1
        while (
            index < len(priority_ordered)
            and abs(
                group[0].differential_priority
                - priority_ordered[index].differential_priority
            )
            < LOW_MARGIN_THRESHOLD
        ):
            group.append(priority_ordered[index])
            index += 1
        ordered.extend(
            sorted(
                group,
                key=lambda item: _portfolio_row_low_margin_rank(item, evidence_by_id),
                reverse=True,
            )
        )
    return ordered


def _portfolio_row_low_margin_rank(row, evidence_by_id) -> tuple[int, int, float, float]:  # noqa: ANN001
    refs = [ref for ref in row.positive_evidence_refs if ref in evidence_by_id]
    support_strength = _row_support_strength(row.support_strength, has_refs=bool(refs))
    return (
        _row_support_strength_rank(support_strength),
        _row_strong_evidence_count(refs, evidence_by_id),
        _row_strong_evidence_specificity(refs, evidence_by_id),
        row.differential_priority,
    )


def _portfolio_row_output_rank(row, evidence_by_id) -> tuple[float, int, int, float]:  # noqa: ANN001
    refs = [ref for ref in row.positive_evidence_refs if ref in evidence_by_id]
    support_strength = _row_support_strength(row.support_strength, has_refs=bool(refs))
    return (
        row.differential_priority,
        _row_support_strength_rank(support_strength),
        len(refs),
        _row_evidence_specificity(refs, evidence_by_id),
    )


def _row_support_strength_rank(value: str) -> int:
    return {
        "strong": 3,
        "partial": 2,
        "weak_or_proxy": 1,
        "unsupported": 0,
    }.get(value, 0)


def _row_evidence_specificity(refs: list[str], evidence_by_id) -> float:  # noqa: ANN001
    score = 0.0
    for ref in refs:
        evidence = evidence_by_id.get(ref)
        if evidence is None:
            continue
        guide = _ABILITY_GUIDE_BY_CAPABILITY.get(evidence.capability_id)
        if guide is None:
            continue
        screened = [
            label
            for label in guide.get("screens", [])
            if isinstance(label, str) and label
        ]
        if not screened:
            continue
        score += 1.0 / len(screened)
    return score


def _row_strong_evidence_count(refs: list[str], evidence_by_id) -> int:  # noqa: ANN001
    return sum(
        1
        for ref in refs
        if _evidence_tier(ref, evidence_by_id) in _LOW_MARGIN_ORDERING_TIERS
    )


def _row_strong_evidence_specificity(refs: list[str], evidence_by_id) -> float:  # noqa: ANN001
    return _row_evidence_specificity(
        [
            ref
            for ref in refs
            if _evidence_tier(ref, evidence_by_id) in _LOW_MARGIN_ORDERING_TIERS
        ],
        evidence_by_id,
    )


def _screened_evidence_refs_for_label(
    label: str,
    *,
    evidence_by_id,
    excluded_refs: list[str],
    limit: int,
) -> list[str]:
    if limit <= 0:
        return []
    excluded = set(excluded_refs)
    candidates: list[tuple[int, int, float, str]] = []
    for index, (evidence_id, evidence) in enumerate(evidence_by_id.items()):
        if evidence_id in excluded:
            continue
        guide = _ABILITY_GUIDE_BY_CAPABILITY.get(evidence.capability_id)
        if guide is None:
            continue
        screens = [
            item for item in guide.get("screens", []) if isinstance(item, str) and item
        ]
        if label not in screens:
            continue
        if not _evidence_screen_is_positive_for_label(label, evidence):
            continue
        tier = str(guide.get("evidence_tier") or "")
        if tier == "boundary_only":
            continue
        if str(evidence.status or "").lower() in {"failed", "unsupported", "missing"}:
            continue
        candidates.append(
            (
                _evidence_tier_rank(tier),
                -index,
                1.0 / max(len(screens), 1),
                evidence_id,
            )
        )
    return [
        evidence_id
        for _, _, _, evidence_id in sorted(candidates, reverse=True)[:limit]
    ]


def _evidence_tier_rank(tier: str) -> int:
    return {
        "computed_direct_proxy": 4,
        "mechanism_specific_structural_trigger": 3,
        "electronic_weak_trigger": 2,
        "structural_trigger": 1,
        "artifact_weak_trigger": 1,
    }.get(tier, 0)


def _number_like(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _rim_ref_priority_floor(
    label: str,
    refs: list[str],
    evidence_by_id,  # noqa: ANN001
) -> float | None:
    if label != "RIM_RIR_RIV" or not refs:
        return None
    priorities: list[float] = []
    for ref in refs:
        evidence = evidence_by_id.get(ref)
        if evidence is None:
            continue
        capability_id = str(evidence.capability_id or "")
        metrics = evidence.metrics if isinstance(evidence.metrics, dict) else {}
        if capability_id in {
            "macro.screen_rotor_torsion_topology",
            "macro.screen_rotor_rim_prior",
        }:
            priorities.append(_rim_priority_from_metrics(metrics, computed=False))
        elif capability_id in {
            "microscopic.run_torsion_brightness_coupling_scan",
            "microscopic.run_torsion_snapshots",
        }:
            priorities.append(_rim_priority_from_metrics(metrics, computed=True))
        elif capability_id == "macro.run_solid_state_emission_proxy":
            priorities.append(_rim_priority_from_solid_state_proxy(metrics))
    priorities = [priority for priority in priorities if priority > 0.0]
    return max(priorities) if priorities else None


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
        return 0.30 if computed else 0.28
    if flexible >= 2:
        return 0.24 if computed else 0.21
    if flexible >= 1:
        return 0.20 if computed else 0.17
    return 0.12


def _evidence_screen_is_positive_for_label(label: str, evidence) -> bool:  # noqa: ANN001
    capability_id = str(evidence.capability_id or "")
    metrics = evidence.metrics if isinstance(evidence.metrics, dict) else {}
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
        flexible = max(
            _number_like(metrics.get("rotatable_bond_count")) or 0.0,
            _number_like(metrics.get("torsion_candidate_count")) or 0.0,
        )
        return label == "RIM_RIR_RIV" and flexible > 0.0
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
    if capability_id == "macro.run_solid_state_emission_proxy":
        if label == "RIM_RIR_RIV":
            return (_number_like(metrics.get("rotor_burden_proxy")) or 0.0) > 0.0
        if label == "PACKING_HOST_MATRIX_CONFINEMENT":
            return bool(metrics.get("aggregation_prone_proxy"))
        if label == "AGGREGATE_EXCITON_EXCIMER":
            return bool(
                metrics.get("aggregate_exciton_proxy")
                or metrics.get("excimer_geometry_proxy")
                or metrics.get("spectral_new_band_proxy")
            )
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
    if capability_id in {
        "microscopic.run_frontier_orbital_partition",
        "microscopic.extract_ct_descriptors_from_bundle",
    }:
        if label == "ICT_TICT_CT":
            return _ct_route_has_support_signal(metrics)
        if label == "PET_ET":
            return bool(
                metrics.get("pet_receptor_like_proxy")
                or metrics.get("redox_alignment_proxy")
                or metrics.get("electron_transfer_pathway_proxy")
            )
        return False
    if capability_id == "macro.screen_esipt_structural_motif":
        return label == "ESIPT_PT" and bool(
            metrics.get("esipt_motif_proxy")
            or metrics.get("proton_transfer_pair_proxy_count")
        )
    if capability_id == "microscopic.run_bright_dark_state_ordering":
        state_count = _number_like(metrics.get("state_count")) or 0.0
        bright_state_index = _number_like(metrics.get("bright_state_index")) or 0.0
        if label == "SOKR_ANTI_KASHA":
            return state_count > 1.0 and bright_state_index > 1.0
        return label == "RADIATIVE_RATE_STATE_BALANCE" and (
            state_count > 1.0
            or bright_state_index > 1.0
            or _number_like(metrics.get("oscillator_strength")) is not None
        )
    if capability_id == "microscopic.run_baseline_bundle":
        return label == "RADIATIVE_RATE_STATE_BALANCE" and (
            (_number_like(metrics.get("state_count")) or 0.0) > 1.0
            or _number_like(metrics.get("oscillator_strength")) is not None
        )
    return True


def _ct_route_has_support_signal(metrics: dict[str, object]) -> bool:
    if (_number_like(metrics.get("state_count")) or 0.0) <= 0.0:
        return False
    informative_keys = (
        "frontier_fragment_separation",
        "homo_lumo_fragment_shift",
        "ct_descriptor_proxy",
        "charge_transfer_proxy",
        "electron_hole_separation_proxy",
        "oscillator_strength",
    )
    return any(
        bool(metrics.get(key))
        if not isinstance(metrics.get(key), int | float)
        else (_number_like(metrics.get(key)) or 0.0) > 0.0
        for key in informative_keys
    )


def _weak_ct_ref_cap(
    label: str,
    refs: list[str],
    evidence_by_id,  # noqa: ANN001
) -> float | None:
    if label != "ICT_TICT_CT" or not refs:
        return None
    checked = False
    for ref in refs:
        evidence = evidence_by_id.get(ref)
        if evidence is None:
            continue
        capability_id = str(evidence.capability_id or "")
        metrics = evidence.metrics if isinstance(evidence.metrics, dict) else {}
        if capability_id in {
            "macro.screen_donor_acceptor_layout",
            "macro.screen_donor_acceptor_architecture",
        }:
            checked = True
            continue
        if capability_id in {
            "microscopic.run_frontier_orbital_partition",
            "microscopic.extract_ct_descriptors_from_bundle",
        }:
            if _ct_route_has_support_signal(metrics):
                return None
            checked = True
            continue
        return None
    return WEAK_CT_PROXY_PRIORITY_CEILING if checked else None


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


def _weak_polar_only_ref_cap(
    label: str,
    refs: list[str],
    evidence_by_id,  # noqa: ANN001
) -> float | None:
    if not refs:
        return None
    priorities: list[float] = []
    for ref in refs:
        evidence = evidence_by_id.get(ref)
        if evidence is None:
            continue
        if evidence.capability_id != "macro.screen_polar_binding_site_prior":
            return None
        priority = _polar_screening_priority(label, evidence.metrics)
        if priority <= 0.0:
            return None
        if priority > 0.19:
            return None
        priorities.append(priority)
    return max(priorities) if priorities else None


def _radiative_single_state_ref_cap(
    label: str,
    refs: list[str],
    evidence_by_id,  # noqa: ANN001
) -> float | None:
    if label != "RADIATIVE_RATE_STATE_BALANCE" or not refs:
        return None
    caps: list[float] = []
    checked = False
    for ref in refs:
        evidence = evidence_by_id.get(ref)
        if evidence is None:
            continue
        if evidence.capability_id not in {
            "microscopic.run_baseline_bundle",
            "microscopic.run_bright_dark_state_ordering",
        }:
            return None
        metrics = evidence.metrics if isinstance(evidence.metrics, dict) else {}
        state_count = _number_like(metrics.get("state_count")) or 0.0
        bright_state_index = _number_like(metrics.get("bright_state_index")) or 0.0
        if state_count > 1.0 or bright_state_index > 1.0:
            return None
        checked = True
        caps.append(_standalone_oscillator_priority(metrics))
    return max(caps) if checked and caps else None


def _generic_packing_only_refs(refs: list[str], evidence_by_id) -> bool:  # noqa: ANN001
    if not refs:
        return False
    checked = False
    for ref in refs:
        evidence = evidence_by_id.get(ref)
        if evidence is None:
            continue
        capability_id = str(evidence.capability_id or "")
        metrics = evidence.metrics if isinstance(evidence.metrics, dict) else {}
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


def _standalone_oscillator_priority(metrics: dict[str, object]) -> float:
    oscillator = _number_like(metrics.get("oscillator_strength"))
    if oscillator is None:
        return 0.18
    if oscillator >= 0.20:
        return 0.29
    if oscillator >= 0.10:
        return 0.17
    if oscillator >= 0.02:
        return 0.18
    if oscillator >= 0.005:
        return 0.17
    return 0.18


def _nonzero_metric(value: object) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, int | float):
        return float(value) != 0.0
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _evidence_tier(ref: str, evidence_by_id) -> str:  # noqa: ANN001
    evidence = evidence_by_id.get(ref)
    if evidence is None:
        return ""
    guide = _ABILITY_GUIDE_BY_CAPABILITY.get(evidence.capability_id)
    if guide is None:
        return ""
    return str(guide.get("evidence_tier") or "")


def _refs_from_attributions(
    label: str,
    *,
    attribution_by_label: dict[str, list[tuple[str, str, str]]],
    evidence_by_id,
) -> list[str]:
    refs = [
        evidence_id
        for evidence_id, _, _ in attribution_by_label.get(label, [])
        if evidence_id in evidence_by_id
    ]
    return list(dict.fromkeys(refs))[:3]


def _annotate_low_margin_predictions(
    predictions: list[MechanismPrediction],
) -> list[MechanismPrediction]:
    if len(predictions) < 2:
        return predictions
    updates: dict[int, dict[str, object]] = {}
    group_index = 1
    for index, (current, next_item) in enumerate(
        zip(predictions, predictions[1:], strict=False)
    ):
        current_priority = _prediction_priority(current)
        next_priority = _prediction_priority(next_item)
        margin = max(current_priority - next_priority, 0.0)
        updates.setdefault(index, {})["margin_to_next"] = round(margin, 6)
        if margin >= LOW_MARGIN_THRESHOLD:
            continue
        group_id = f"LMG{group_index:03d}"
        group_index += 1
        note = (
            f"{current.label} and {next_item.label} are low-margin competing "
            f"candidates under current runtime evidence (margin {margin:.3f})."
        )
        for prediction_index in (index, index + 1):
            existing = updates.setdefault(prediction_index, {})
            existing["low_margin_group"] = existing.get("low_margin_group") or group_id
            existing["ranking_stability_note"] = note
            limitations = list(predictions[prediction_index].limitations)
            if note not in limitations:
                limitations.append(note)
            existing["limitations"] = limitations[:6]
    return [
        prediction.model_copy(update=updates.get(index, {}))
        for index, prediction in enumerate(predictions)
    ]


def _prediction_priority(prediction: MechanismPrediction) -> float:
    if prediction.differential_priority is not None:
        return prediction.differential_priority
    if prediction.plausibility_confidence is not None:
        return prediction.plausibility_confidence
    return prediction.confidence


def _attributions_by_label(
    case_run: CaseRun,
) -> dict[str, list[tuple[str, str, str]]]:
    portfolio = case_run.differential_mechanism_portfolio
    if portfolio is None:
        return {}
    grouped: dict[str, list[tuple[str, str, str]]] = {}
    for attribution in portfolio.evidence_attributions:
        for update in attribution.updates:
            grouped.setdefault(update.label, []).append(
                (attribution.evidence_id, update.warrant, update.boundary)
            )
    return grouped


def _row_evidence_details(
    label: str,
    refs: list[str],
    *,
    evidence_by_id,
    attribution_by_label: dict[str, list[tuple[str, str, str]]],
    fallback_boundary: str,
    fallback_warrant: str,
) -> list[MechanismPredictionEvidence]:
    details: list[MechanismPredictionEvidence] = []
    updates_by_ref = {
        evidence_id: (warrant, boundary)
        for evidence_id, warrant, boundary in attribution_by_label.get(label, [])
    }
    for ref in refs[:3]:
        evidence = evidence_by_id.get(ref)
        if evidence is None:
            continue
        warrant, boundary = updates_by_ref.get(ref, (fallback_warrant, fallback_boundary))
        finding = _enhanced_finding_text(label, evidence)
        details.append(
            MechanismPredictionEvidence(
                finding=finding,
                warrant=warrant,
                boundary=boundary,
                evidence_refs=[ref],
            )
        )
    return details


def _ordered_evidence_details_for_label(
    label: str,
    details: list[MechanismPredictionEvidence],
) -> list[MechanismPredictionEvidence]:
    indexed = list(enumerate(details))
    return [
        detail
        for _, detail in sorted(
            indexed,
            key=lambda item: (
                _evidence_detail_label_rank(label, item[1]),
                -item[0],
            ),
            reverse=True,
        )
    ]


def _evidence_detail_label_rank(
    label: str,
    detail: MechanismPredictionEvidence,
) -> int:
    refs = " ".join(detail.evidence_refs)
    text = f"{detail.finding} {detail.warrant}".lower()
    if label == "ICT_TICT_CT":
        if "screen_donor_acceptor" in refs or "donor/acceptor" in text:
            return 5
        if "extract_ct_descriptors" in refs or "ct descriptor" in text:
            return 4
        if "frontier_orbital" in refs:
            return 3
        if "torsion" in refs:
            return 1
    if label == "PACKING_HOST_MATRIX_CONFINEMENT":
        if "run_dimer_packing_proxy" in refs or "run_aggregate_contact_proxy" in refs:
            return 5
        if "solid_state" in refs:
            return 4
    if label == "AGGREGATE_EXCITON_EXCIMER":
        if any(term in text for term in ("excimer", "exciton", "new band")):
            return 5
        if "screen_aggregation" in refs:
            return 1
    return 2


def _enhanced_finding_text(label: str, evidence) -> str:  # noqa: ANN001
    claim = str(evidence.claim or "").strip()
    metrics = evidence.metrics if isinstance(evidence.metrics, dict) else {}
    capability_id = str(evidence.capability_id or "")
    additions: list[str] = []
    if capability_id == "macro.screen_esipt_structural_motif":
        pairs = metrics.get("proton_transfer_pair_proxies")
        if isinstance(pairs, list) and pairs:
            first = pairs[0]
            if isinstance(first, dict):
                additions.append(
                    "nearest proton-transfer pair proxy="
                    f"{first.get('pair_type', 'donor-H...acceptor')} "
                    f"({first.get('donor_atom', 'donor')} to "
                    f"{first.get('acceptor_atom', 'acceptor')}, "
                    f"{first.get('topological_distance', 'unknown')} bonds)"
                )
        additions.extend(
            _metric_fragments(
                metrics,
                (
                    "hbd_count",
                    "hba_count",
                    "tautomerizable_subgraph_proxy",
                    "proton_transfer_pair_proxy_count",
                ),
            )
        )
    elif capability_id in {
        "macro.screen_rotor_torsion_topology",
        "macro.screen_rotor_rim_prior",
    }:
        additions.extend(
            _metric_fragments(
                metrics,
                (
                    "rotatable_bond_count",
                    "torsion_candidate_count",
                    "branch_point_count",
                    "flexibility_proxy",
                    "rotatable_bond_inventory",
                ),
            )
        )
    elif capability_id in {
        "macro.screen_donor_acceptor_layout",
        "macro.screen_donor_acceptor_architecture",
    }:
        additions.extend(
            _metric_fragments(
                metrics,
                (
                    "donor_acceptor_proxy",
                    "donor_acceptor_partition_proxy",
                    "hetero_atom_count",
                    "hbd_count",
                    "hba_count",
                    "conjugation_proxy",
                    "formal_charge",
                    "donor_atom_symbols",
                    "acceptor_atom_symbols",
                    "polar_atom_symbols",
                ),
            )
        )
    elif capability_id == "macro.screen_polar_binding_site_prior":
        additions.extend(
            _metric_fragments(
                metrics,
                (
                    "formal_charge",
                    "hbd_count",
                    "hba_count",
                    "carbonyl_like_proxy",
                    "carbonyl_like_site_count",
                    "polar_binding_site_proxy",
                    "host_guest_followup_trigger",
                    "pet_receptor_like_proxy",
                    "donor_atom_symbols",
                    "acceptor_atom_symbols",
                    "polar_atom_symbols",
                ),
            )
        )
    elif capability_id in {
        "macro.run_dimer_packing_proxy",
        "macro.run_aggregate_contact_proxy",
        "macro.run_solid_state_emission_proxy",
    }:
        additions.extend(
            _metric_fragments(
                metrics,
                (
                    "dimer_contact_score_proxy",
                    "aromatic_fraction_proxy",
                    "aromatic_atom_count",
                    "aromatic_ring_count",
                    "mol_logp",
                    "rotatable_bond_count",
                    "hydrophobic_contact_proxy",
                    "polar_disruption_proxy",
                    "aggregation_prone_proxy",
                    "solid_state_emission_claim_status",
                    "scaffold_proxy",
                ),
            )
        )
    elif capability_id == "macro.screen_metal_triplet_prior":
        additions.extend(
            _metric_fragments(
                metrics,
                (
                    "metal_or_lanthanide_prior",
                    "heavy_atom_triplet_prior",
                    "sulfur_phosphorus_triplet_prior",
                    "triplet_metal_followup_trigger",
                ),
            )
        )
        for key in (
            "metal_atom_symbols",
            "heavy_atom_symbols",
            "triplet_relevant_hetero_symbols",
        ):
            value = metrics.get(key)
            if isinstance(value, list) and value:
                additions.append(f"{key}={','.join(str(item) for item in value[:6])}")
    elif capability_id in {
        "microscopic.run_baseline_bundle",
        "microscopic.run_bright_dark_state_ordering",
        "microscopic.run_targeted_transition_dipole_analysis",
        "microscopic.extract_ct_descriptors_from_bundle",
        "microscopic.run_frontier_orbital_partition",
    }:
        additions.extend(
            _metric_fragments(
                metrics,
                (
                    "state_count",
                    "oscillator_strength",
                    "bright_state_index",
                    "frontier_fragment_separation",
                    "donor_acceptor_partition_proxy",
                    "ct_descriptor_proxy",
                    "transition_dipole_available",
                    "missing_deliverable_count",
                ),
            )
        )
    if not additions:
        return claim
    detail = "; ".join(additions[:8])
    if detail in claim:
        return claim
    return f"{claim} Runtime details: {detail}."


def _metric_fragments(metrics: dict[str, object], keys: tuple[str, ...]) -> list[str]:
    fragments: list[str] = []
    for key in keys:
        if key not in metrics:
            continue
        value = metrics.get(key)
        if value is None:
            continue
        if isinstance(value, float):
            fragments.append(f"{key}={value:.3f}")
        else:
            fragments.append(f"{key}={value}")
    return fragments


def _row_support_strength(value: str, *, has_refs: bool) -> str:
    if not has_refs and value in {"partial", "strong"}:
        return "weak_or_proxy"
    return {
        "strong": "strong",
        "partial": "partial",
        "weak_proxy": "weak_or_proxy",
        "unsupported": "unsupported",
    }.get(value, "unsupported")


def _row_claim_status(value: str, *, support_strength: str, has_refs: bool) -> str:
    if not has_refs or support_strength in {"unsupported", "weak_or_proxy"}:
        return "candidate_requires_validation"
    if value == "supported" and support_strength == "strong":
        return "supported_claim"
    if value in {"supported", "partially_supported"}:
        return "partially_supported_candidate"
    return "candidate_requires_validation"


def _row_boundary(row) -> str:
    return (
        "; ".join(row.missing_validation[:2])
        or "Evidence is bounded by public runtime observations and remains incomplete."
    )


def _row_limitations(row) -> list[str]:
    limitations = list(row.missing_validation[:4])
    if row.negative_evidence_refs:
        limitations.append(
            "MechanismCritic recorded counterevidence refs: "
            + ", ".join(row.negative_evidence_refs[:4])
        )
    return limitations[:5]


def _row_no_ref_evidence_text(row) -> str:
    return (
        f"Candidate rationale: {row.rationale} Boundary: "
        f"{_row_boundary(row)} No positive runtime evidence reference was assigned."
    )


def _prediction_evidence_text(detail: MechanismPredictionEvidence) -> str:
    return (
        f"Finding: {detail.finding} "
        f"Warrant: {detail.warrant} "
        f"Boundary: {detail.boundary}"
    )


def _support_arguments_by_label(
    case_run: CaseRun,
) -> dict[str, list[MechanismSupportArgument]]:
    grouped: dict[str, list[MechanismSupportArgument]] = {}
    for argument in case_run.mechanism_support_arguments:
        if argument.support_level == "unsupported":
            continue
        if argument.audit_status in {"unsupported", "invalid_ref", "overclaim"}:
            continue
        grouped.setdefault(argument.target_label, []).append(argument)
    return {
        label: _select_final_support_arguments(arguments)
        for label, arguments in grouped.items()
    }


def _select_final_support_arguments(
    arguments: list[MechanismSupportArgument],
    *,
    limit: int = 2,
) -> list[MechanismSupportArgument]:
    return sorted(
        arguments,
        key=lambda argument: (
            _support_level_rank(argument.support_level),
            _round_rank(argument.support_id),
            argument.support_id,
        ),
        reverse=True,
    )[:limit]


def _support_level_rank(level: str) -> int:
    return {
        "strong": 3,
        "partial": 2,
        "weak": 1,
        "unsupported": 0,
    }.get(level, 0)


def _round_rank(support_id: str) -> int:
    prefix = support_id.split(":", 1)[0]
    if not prefix.startswith("R"):
        return 0
    try:
        return int(prefix[1:])
    except ValueError:
        return 0


def _support_argument_text(argument: MechanismSupportArgument) -> str:
    return (
        f"Finding: {argument.finding} "
        f"Warrant: {argument.warrant} "
        f"Boundary: {argument.boundary}"
    )


def _support_argument_details(
    argument: MechanismSupportArgument,
    *,
    label: str,
    evidence_by_id,
) -> list[MechanismPredictionEvidence]:
    details: list[MechanismPredictionEvidence] = []
    for ref in argument.observation_refs[:3]:
        evidence = evidence_by_id.get(ref)
        if evidence is None:
            continue
        details.append(
            MechanismPredictionEvidence(
                finding=_enhanced_finding_text(label, evidence),
                warrant=argument.warrant,
                boundary=argument.boundary,
                evidence_refs=[ref],
            )
        )
    if details:
        return details
    return [
        MechanismPredictionEvidence(
            finding=argument.finding,
            warrant=argument.warrant,
            boundary=argument.boundary,
            evidence_refs=list(argument.observation_refs),
        )
    ]


def _support_strength(
    support_arguments: list[MechanismSupportArgument],
) -> str:
    strongest = max(
        (_support_level_rank(argument.support_level) for argument in support_arguments),
        default=0,
    )
    if strongest >= 3:
        return "strong"
    if strongest == 2:
        return "partial"
    if strongest == 1:
        return "weak_or_proxy"
    return "unsupported"


def _claim_status(
    support_arguments: list[MechanismSupportArgument],
) -> str:
    support_strength = _support_strength(support_arguments)
    if support_strength == "strong":
        return "supported_claim"
    if support_strength == "partial":
        return "partially_supported_candidate"
    if support_strength == "weak_or_proxy":
        return "candidate_requires_validation"
    return "underdetermined"


def _validation_needed(
    support_arguments: list[MechanismSupportArgument],
    *,
    hypothesis_validation_needed: list[str],
) -> list[str]:
    if not support_arguments:
        return (
            list(dict.fromkeys(hypothesis_validation_needed))[:3]
            or ["Collect source-grounded evidence before claiming this mechanism."]
        )
    return list(
        dict.fromkeys(
            [argument.boundary for argument in support_arguments]
            + hypothesis_validation_needed
        )
    )[:3]


def _prediction_limitations(
    case_run: CaseRun,
    support_arguments: list[MechanismSupportArgument],
) -> list[str]:
    limitations: list[str] = []
    for argument in support_arguments:
        limitations.append(argument.boundary)
        limitations.extend(argument.audit_notes[:2])
    review = case_run.photophysics_review
    if review is not None:
        limitations.extend(review.overclaim_warnings[:2])
    return list(dict.fromkeys(limitations))[:5]
