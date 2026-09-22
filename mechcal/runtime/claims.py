from __future__ import annotations

from dataclasses import dataclass

from mechcal.schemas import EvidenceLedger, EvidenceUnit
from mechcal.schemas.claims import (
    ClaimAssessment,
    ClaimDefinition,
    ClaimLedger,
    HypothesisCoverageDebt,
)


@dataclass(frozen=True)
class ClaimDispatchRoute:
    claim_id: str
    agent_name: str
    route: str
    evidence_family: str
    task: str
    artifact_kind: str | None = None


CLAIM_DEFINITIONS: tuple[ClaimDefinition, ...] = (
    ClaimDefinition(
        claim_id="shared_structural_prior_checked",
        hypothesis="shared",
        topic="structural_prior",
        role="context",
        description="At least one bounded structural-prior proxy was collected.",
        covering_routes=["macro.macro_structure_scan"],
        screening_claim=False,
    ),
    ClaimDefinition(
        claim_id="neutral_aromatic_le_signature_screened",
        hypothesis="neutral aromatic",
        topic="neutral_aromatic_le_signature",
        role="direct_prerequisite",
        description="Low-cost state ordering/brightness evidence screened the LE baseline.",
        covering_routes=["microscopic.run_baseline_bundle"],
    ),
    ClaimDefinition(
        claim_id="esipt_structural_motif_screened",
        hypothesis="ESIPT proton-transfer photophysics",
        topic="esipt_structural_motif",
        role="proxy",
        description=(
            "A bounded structural motif route screened whether donor/acceptor topology "
            "is compatible with ESIPT."
        ),
        covering_routes=["macro.screen_esipt_structural_motif"],
    ),
    ClaimDefinition(
        claim_id="solid_state_emission_proxy_screened",
        hypothesis="solid-state restricted-rotation AIE",
        topic="solid_state_emission_proxy",
        role="proxy",
        description=(
            "A bounded structural proxy route screened solid-state emission plausibility "
            "without claiming measured PL."
        ),
        covering_routes=["macro.run_solid_state_emission_proxy"],
    ),
    ClaimDefinition(
        claim_id="solid_state_restriction_checklist_screened",
        hypothesis="solid-state restricted-rotation AIE",
        topic="crystal_or_solid_restriction_boundary",
        role="external_check",
        description=(
            "A checklist route recorded the direct crystal/solid-state evidence needed "
            "before asserting a measured restriction mechanism."
        ),
        covering_routes=["macro.run_crystal_restriction_checklist"],
    ),
    ClaimDefinition(
        claim_id="conformer_sensitivity_screened",
        hypothesis="conformational heterogeneity",
        topic="conformer_sensitivity",
        role="direct_prerequisite",
        description="A bounded conformer route screened conformer-dependent observables.",
        covering_routes=[
            "microscopic.run_conformer_bundle",
            "microscopic.run_conformer_state_probe",
        ],
    ),
    ClaimDefinition(
        claim_id="rim_torsion_sensitivity_screened",
        hypothesis="RIM/torsional restriction",
        topic="torsion_sensitivity",
        role="direct_prerequisite",
        description="A bounded torsion route screened torsion-sensitive observables.",
        covering_routes=["microscopic.run_torsion_snapshots"],
    ),
    ClaimDefinition(
        claim_id="ict_localized_orbital_screened",
        hypothesis="ICT/TICT",
        topic="localized_orbital_availability",
        role="direct_prerequisite",
        description="Localized-orbital artifacts were checked for charge-localization support.",
        covering_routes=["microscopic.run_targeted_localized_orbital_analysis"],
    ),
    ClaimDefinition(
        claim_id="ict_natural_orbital_screened",
        hypothesis="ICT/TICT",
        topic="natural_orbital_availability",
        role="direct_prerequisite",
        description="Natural-orbital artifacts were checked for charge-localization support.",
        covering_routes=["microscopic.run_targeted_natural_orbital_analysis"],
    ),
    ClaimDefinition(
        claim_id="high_level_relaxation_boundary_screened",
        hypothesis="ESIPT-coupled torsional/TICT decay",
        topic="nonadiabatic_relaxation_boundary",
        role="external_check",
        description=(
            "The run explicitly recorded whether validated CI/TSH/nonadiabatic relaxation "
            "analysis is available in this tool scope."
        ),
        covering_routes=["microscopic.unsupported_excited_state_relaxation"],
    ),
)

CLAIM_DISPATCH_PLAN: tuple[ClaimDispatchRoute, ...] = (
    ClaimDispatchRoute(
        claim_id="esipt_structural_motif_screened",
        agent_name="macro",
        route="screen_esipt_structural_motif",
        evidence_family="geometry_precondition",
        task=(
            "Screen ESIPT-compatible donor/acceptor and tautomerizable motif proxies "
            "without claiming proton-transfer dynamics."
        ),
    ),
    ClaimDispatchRoute(
        claim_id="solid_state_emission_proxy_screened",
        agent_name="macro",
        route="run_solid_state_emission_proxy",
        evidence_family="geometry_precondition",
        task=(
            "Screen solid-state emission plausibility from structural proxies while "
            "preserving proxy-only limits."
        ),
    ),
    ClaimDispatchRoute(
        claim_id="solid_state_restriction_checklist_screened",
        agent_name="macro",
        route="run_crystal_restriction_checklist",
        evidence_family="geometry_precondition",
        task=(
            "Record the crystal/solid-state restriction evidence checklist so the "
            "final synthesis does not overclaim direct solid-state measurements."
        ),
    ),
    ClaimDispatchRoute(
        claim_id="conformer_sensitivity_screened",
        agent_name="microscopic",
        route="run_conformer_bundle",
        evidence_family="conformer_sensitivity",
        artifact_kind="conformer_bundle",
        task="Screen conformer sensitivity with a bounded multi-conformer bundle.",
    ),
    ClaimDispatchRoute(
        claim_id="rim_torsion_sensitivity_screened",
        agent_name="microscopic",
        route="run_torsion_snapshots",
        evidence_family="torsion_sensitivity",
        artifact_kind="torsion_snapshot_bundle",
        task="Screen torsion sensitivity with a bounded rotated snapshot series.",
    ),
    ClaimDispatchRoute(
        claim_id="ict_localized_orbital_screened",
        agent_name="microscopic",
        route="run_targeted_localized_orbital_analysis",
        evidence_family="charge_localization",
        artifact_kind="targeted_localized_orbital_bundle",
        task="Check localized-orbital artifacts for bounded charge-localization evidence.",
    ),
    ClaimDispatchRoute(
        claim_id="ict_natural_orbital_screened",
        agent_name="microscopic",
        route="run_targeted_natural_orbital_analysis",
        evidence_family="charge_localization",
        artifact_kind="targeted_natural_orbital_bundle",
        task="Check natural-orbital artifacts for bounded charge-localization evidence.",
    ),
    ClaimDispatchRoute(
        claim_id="high_level_relaxation_boundary_screened",
        agent_name="microscopic",
        route="unsupported_excited_state_relaxation",
        evidence_family="state_ordering_brightness",
        task=(
            "Record the capability boundary for conical-intersection, TSH, and nonadiabatic "
            "relaxation analysis instead of fabricating unavailable evidence."
        ),
    ),
)

_TAG_TO_CLAIMS: dict[str, tuple[str, ...]] = {
    "macro_structure_scan_proxy": ("shared_structural_prior_checked",),
    "structural_prior": ("shared_structural_prior_checked",),
    "amesp_s0_s1_baseline": ("neutral_aromatic_le_signature_screened",),
    "run_baseline_bundle": ("neutral_aromatic_le_signature_screened",),
    "screen_esipt_structural_motif_proxy": ("esipt_structural_motif_screened",),
    "screen_esipt_structural_motif": ("esipt_structural_motif_screened",),
    "esipt_motif_proxy": ("esipt_structural_motif_screened",),
    "solid_state_emission_proxy": ("solid_state_emission_proxy_screened",),
    "run_solid_state_emission_proxy": ("solid_state_emission_proxy_screened",),
    "crystal_restriction_checklist": (
        "solid_state_restriction_checklist_screened",
    ),
    "run_crystal_restriction_checklist": (
        "solid_state_restriction_checklist_screened",
    ),
    "conformer_sensitivity": ("conformer_sensitivity_screened",),
    "run_conformer_bundle": ("conformer_sensitivity_screened",),
    "run_conformer_state_probe": ("conformer_sensitivity_screened",),
    "torsion_sensitivity": ("rim_torsion_sensitivity_screened",),
    "run_torsion_snapshots": ("rim_torsion_sensitivity_screened",),
    "localized_orbital_analysis": ("ict_localized_orbital_screened",),
    "run_targeted_localized_orbital_analysis": ("ict_localized_orbital_screened",),
    "natural_orbital_analysis": ("ict_natural_orbital_screened",),
    "run_targeted_natural_orbital_analysis": ("ict_natural_orbital_screened",),
    "unsupported_excited_state_relaxation": (
        "high_level_relaxation_boundary_screened",
    ),
}

_COVERED_STATUSES = {"present", "partial"}
_BLOCKED_STATUSES = {"failed", "unsupported", "missing"}
_RELATION_PRIORITY = {
    "supports": 4,
    "challenges": 4,
    "mixed": 3,
    "neutral": 2,
    "unknown": 1,
}


def empty_claim_ledger(case_id: str) -> ClaimLedger:
    return reduce_claim_ledger(case_id, EvidenceLedger(case_id=case_id))


def annotate_evidence_units(items: list[EvidenceUnit]) -> list[EvidenceUnit]:
    return [
        item.model_copy(
            update={"claim_refs": sorted({*item.claim_refs, *claim_refs_for_evidence(item)})}
        )
        for item in items
    ]


def annotate_evidence_items(items: list[EvidenceUnit]) -> list[EvidenceUnit]:
    return annotate_evidence_units(items)


def claim_refs_for_evidence(item: EvidenceUnit) -> list[str]:
    refs: set[str] = set()
    for tag in item.observable_tags:
        refs.update(_TAG_TO_CLAIMS.get(tag, ()))
    refs.update(_TAG_TO_CLAIMS.get(item.family, ()))
    return sorted(refs)


def reduce_claim_ledger(case_id: str, evidence_ledger: EvidenceLedger) -> ClaimLedger:
    items = annotate_evidence_units(evidence_ledger.items)
    assessments = [_assess_claim(definition, items) for definition in CLAIM_DEFINITIONS]
    return ClaimLedger(
        case_id=case_id,
        claim_definitions=[item.model_copy(deep=True) for item in CLAIM_DEFINITIONS],
        assessments=assessments,
        hypothesis_debts=_hypothesis_debts(assessments),
    )


def dispatch_route_for_claim(claim_id: str) -> ClaimDispatchRoute | None:
    return next((item for item in CLAIM_DISPATCH_PLAN if item.claim_id == claim_id), None)


def _assess_claim(
    definition: ClaimDefinition,
    evidence_items: list[EvidenceUnit],
) -> ClaimAssessment:
    claim_items = [item for item in evidence_items if definition.claim_id in item.claim_refs]
    covered_items = [item for item in claim_items if _covers_claim(definition, item)]
    blocked_items = [item for item in claim_items if item.status in _BLOCKED_STATUSES]
    coverage_status = _coverage_status(bool(covered_items), bool(blocked_items))
    evidence_refs = [item.evidence_id for item in claim_items]
    return ClaimAssessment(
        claim_id=definition.claim_id,
        hypothesis=definition.hypothesis,
        topic=definition.topic,
        role=definition.role,
        coverage_status=coverage_status,
        relation=_dominant_relation(claim_items),
        evidence_refs=evidence_refs,
        covered_routes=_covered_routes(definition, claim_items),
        blocked_reasons=[
            item.summary for item in blocked_items if item.summary.strip()
        ],
        summary=_claim_summary(definition, coverage_status, evidence_refs),
    )


def _coverage_status(has_covered: bool, has_blocked: bool) -> str:
    if has_covered:
        return "covered"
    if has_blocked:
        return "blocked"
    return "uncovered"


def _covers_claim(definition: ClaimDefinition, item: EvidenceUnit) -> bool:
    if item.status not in _COVERED_STATUSES:
        return False
    if item.basis != "checklist":
        return True
    return definition.role in {"context", "external_check"} and item.support == "unresolved"


def _dominant_relation(items: list[EvidenceUnit]) -> str:
    if not items:
        return "unknown"
    return max(items, key=lambda item: _RELATION_PRIORITY[item.relation]).relation


def _covered_routes(
    definition: ClaimDefinition,
    evidence_items: list[EvidenceUnit],
) -> list[str]:
    tags = {tag for item in evidence_items for tag in item.observable_tags}
    routes = [
        route
        for route in definition.covering_routes
        if route.split(".", maxsplit=1)[-1] in tags
    ]
    return sorted(routes)


def _claim_summary(
    definition: ClaimDefinition,
    coverage_status: str,
    evidence_refs: list[str],
) -> str:
    if coverage_status == "covered":
        return f"Claim covered by {len(evidence_refs)} typed evidence unit(s)."
    if coverage_status == "blocked":
        return "Claim route was attempted but returned typed non-success evidence."
    return f"Claim has not yet been screened: {definition.description}"


def _hypothesis_debts(
    assessments: list[ClaimAssessment],
) -> list[HypothesisCoverageDebt]:
    hypotheses = sorted(
        {
            item.hypothesis
            for item in assessments
            if item.hypothesis != "shared"
        }
    )
    return [_hypothesis_debt(hypothesis, assessments) for hypothesis in hypotheses]


def _hypothesis_debt(
    hypothesis: str,
    assessments: list[ClaimAssessment],
) -> HypothesisCoverageDebt:
    related = [item for item in assessments if item.hypothesis == hypothesis]
    open_claim_ids = [
        item.claim_id for item in related if item.coverage_status == "uncovered"
    ]
    blocked_claim_ids = [
        item.claim_id for item in related if item.coverage_status == "blocked"
    ]
    covered_claim_ids = [
        item.claim_id for item in related if item.coverage_status == "covered"
    ]
    status = "covered"
    if open_claim_ids:
        status = "open"
    elif blocked_claim_ids:
        status = "blocked"
    return HypothesisCoverageDebt(
        hypothesis=hypothesis,
        status=status,
        open_claim_ids=open_claim_ids,
        blocked_claim_ids=blocked_claim_ids,
        covered_claim_ids=covered_claim_ids,
    )
