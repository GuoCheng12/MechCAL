from __future__ import annotations

from mechcal.capabilities import CapabilityRegistry
from mechcal.mechanism_pool import MECHANISM_POOL

MECHANISM_LABELS = set(MECHANISM_POOL)


_ABILITY_EVIDENCE_GUIDE: dict[str, dict[str, object]] = {
    "microscopic.analyze_hole_electron_from_bundle": {
        "observables": ["state-resolved hole/electron distributions", "centroid distance", "spatial overlap", "heavy-atom contributions"],
        "useful_for_screening": ["ICT_TICT_CT", "PET_ET"],
        "support_boundary": "Calculated unrelaxed vertical excitation at one geometry. Spatial redistribution and overlap do not establish TICT, PET kinetics or the emitting state. Atomic contributions refer to the calculation's indexed geometry.",
        "suggested_claim_strength": "weak_or_proxy",
        "evidence_tier": "computed_direct_proxy",
    },
    "microscopic.optimize_ground_state_geometry": {
        "observables": ["ground-state optimized geometry", "convergence status"],
        "useful_for_screening": [],
        "support_boundary": "Input preparation only; not evidence for an emission mechanism or a frequency-verified minimum.",
        "suggested_claim_strength": "unsupported",
        "evidence_tier": "geometry_precondition",
    },
    "macro.screen_rotor_torsion_topology": {
        "observables": ["rotatable bonds", "torsion topology", "motion-prone groups"],
        "useful_for_screening": ["RIM_RIR_RIV", "RACI_CI_ACCESS", "ICT_TICT_CT"],
        "support_boundary": (
            "SMILES/geometry topology is only a structural trigger. It does not "
            "prove motion restriction, conical-intersection access, or TICT."
        ),
        "suggested_claim_strength": "weak_or_proxy",
        "evidence_tier": "structural_trigger",
    },
    "macro.screen_rotor_rim_prior": {
        "observables": ["rotor count", "aryl rotor motifs", "flexible torsion motifs"],
        "useful_for_screening": ["RIM_RIR_RIV"],
        "support_boundary": (
            "Rotor priors can motivate RIM follow-up but are not aggregation, "
            "viscosity, packing, or photoluminescence evidence."
        ),
        "suggested_claim_strength": "weak_or_proxy",
        "evidence_tier": "structural_trigger",
    },
    "macro.screen_donor_acceptor_architecture": {
        "observables": ["donor/acceptor motifs", "push-pull topology"],
        "useful_for_screening": ["ICT_TICT_CT", "PET_ET"],
        "support_boundary": (
            "A donor-acceptor motif alone does not prove CT, TICT, PET, or "
            "electron-transfer quenching."
        ),
        "suggested_claim_strength": "weak_or_proxy",
        "evidence_tier": "mechanism_specific_structural_trigger",
    },
    "macro.screen_donor_acceptor_layout": {
        "observables": ["donor/acceptor layout", "push-pull topology"],
        "useful_for_screening": ["ICT_TICT_CT", "PET_ET"],
        "support_boundary": (
            "A donor-acceptor layout is a structural trigger only. It does not "
            "prove excited-state CT, TICT, PET, redox alignment, or quenching."
        ),
        "suggested_claim_strength": "weak_or_proxy",
        "evidence_tier": "mechanism_specific_structural_trigger",
    },
    "macro.screen_polar_binding_site_prior": {
        "observables": [
            "formal charge",
            "H-bond donor/acceptor sites",
            "carbonyl-like interaction sites",
            "polar receptor-like motifs",
        ],
        "useful_for_screening": ["HOST_GUEST_INTERACTION", "PET_ET"],
        "support_boundary": (
            "Polar or charged sites are only host-guest/PET screening triggers. "
            "They are not binding constants, guest uptake, redox alignment, "
            "electron-transfer quenching, or sensing evidence."
        ),
        "suggested_claim_strength": "weak_or_proxy",
        "evidence_tier": "structural_trigger",
    },
    "macro.screen_metal_triplet_prior": {
        "observables": [
            "metal atoms",
            "lanthanide or transition-metal motif",
            "heavy-atom triplet prior",
            "sulfur or phosphorus triplet-relevant motif",
        ],
        "useful_for_screening": [
            "TRIPLET_METAL_ENERGY_TRANSFER",
        ],
        "support_boundary": (
            "Metal, lanthanide, heavy-atom, sulfur, or phosphorus structure is only "
            "a trigger for triplet/metal energy-transfer follow-up. It is not RTP, "
            "TADF, lanthanide emission, triplet-energy, lifetime, or energy-transfer "
            "evidence."
        ),
        "suggested_claim_strength": "weak_or_proxy",
        "evidence_tier": "mechanism_specific_structural_trigger",
    },
    "macro.screen_esipt_structural_motif": {
        "observables": ["proton donor/acceptor motifs", "intramolecular H-bond priors"],
        "useful_for_screening": ["ESIPT_PT"],
        "support_boundary": (
            "A donor/acceptor motif is not excited-state proton-transfer evidence "
            "without geometry, barrier, or spectral support."
        ),
        "suggested_claim_strength": "weak_or_proxy",
        "evidence_tier": "mechanism_specific_structural_trigger",
    },
    "macro.screen_aggregation_prone_scaffold": {
        "observables": ["aromatic surface", "hydrophobicity proxy", "aggregation-prone scaffold"],
        "useful_for_screening": [
            "AGGREGATE_EXCITON_EXCIMER",
            "PACKING_HOST_MATRIX_CONFINEMENT",
            "RIM_RIR_RIV",
        ],
        "support_boundary": (
            "Aggregation-prone structure is not proof of aggregate excitons, "
            "excimers, packing confinement, or solid-state emission."
        ),
        "suggested_claim_strength": "weak_or_proxy",
        "evidence_tier": "structural_trigger",
    },
    "macro.screen_pi_stacking_prone_geometry": {
        "observables": ["planarity proxy", "pi-stacking-prone geometry"],
        "useful_for_screening": [
            "AGGREGATE_EXCITON_EXCIMER",
            "PACKING_HOST_MATRIX_CONFINEMENT",
        ],
        "support_boundary": (
            "Planarity or pi-stacking propensity needs packing or spectral evidence "
            "before claiming excimer/exciton or confinement mechanisms."
        ),
        "suggested_claim_strength": "weak_or_proxy",
        "evidence_tier": "structural_trigger",
    },
    "macro.run_dimer_packing_proxy": {
        "observables": ["aromatic fraction", "logP", "rotatable-bond count", "derived heuristic"],
        "useful_for_screening": [
            "PACKING_HOST_MATRIX_CONFINEMENT",
            "AGGREGATE_EXCITON_EXCIMER",
        ],
        "support_boundary": (
            "No dimer or contact geometry was computed. This recombines existing monomer "
            "descriptors and adds no independent packing, confinement or excimer evidence."
        ),
        "suggested_claim_strength": "weak_or_proxy",
        "evidence_tier": "structural_trigger",
    },
    "macro.run_aggregate_contact_proxy": {
        "observables": ["logP threshold", "heteroatom-count threshold", "monomer descriptors"],
        "useful_for_screening": [
            "PACKING_HOST_MATRIX_CONFINEMENT",
            "AGGREGATE_EXCITON_EXCIMER",
        ],
        "support_boundary": (
            "No aggregate or contact geometry is computed. These are thresholds on "
            "monomer descriptors, not independent packing or aggregate observations."
        ),
        "suggested_claim_strength": "weak_or_proxy",
        "evidence_tier": "structural_trigger",
    },
    "macro.run_crystal_restriction_checklist": {
        "observables": ["missing packing evidence", "missing solid-state PL evidence"],
        "useful_for_screening": [
            "PACKING_HOST_MATRIX_CONFINEMENT",
            "RIM_RIR_RIV",
        ],
        "support_boundary": (
            "The checklist records what is absent in a SMILES-only run. It should "
            "bound claims rather than increase support."
        ),
        "suggested_claim_strength": "unsupported",
        "evidence_tier": "boundary_only",
    },
    "macro.run_solid_state_emission_proxy": {
        "observables": [
            "aromatic fraction proxy",
            "hydrophobicity proxy",
            "solid-state emission plausibility proxy",
        ],
        "useful_for_screening": [
            "PACKING_HOST_MATRIX_CONFINEMENT",
            "AGGREGATE_EXCITON_EXCIMER",
            "RIM_RIR_RIV",
        ],
        "support_boundary": (
            "Solid-state emission proxies are structural only; they are not "
            "packing measurements, aggregate spectra, or photoluminescence data."
        ),
        "suggested_claim_strength": "weak_or_proxy",
        "evidence_tier": "structural_trigger",
    },
    "microscopic.run_baseline_bundle": {
        "observables": ["S0/S1 low-cost baseline", "state ordering", "oscillator strength"],
        "useful_for_screening": [
            "RADIATIVE_RATE_STATE_BALANCE",
        ],
        "support_boundary": (
            "Low-cost vertical states and oscillator strengths are proxies. They "
            "do not prove quantum yield, lifetime, anti-Kasha emission, or rates."
        ),
        "suggested_claim_strength": "weak_or_proxy",
        "evidence_tier": "computed_direct_proxy",
    },
    "microscopic.run_bright_dark_state_ordering": {
        "observables": ["bright/dark state ordering", "oscillator strength distribution"],
        "useful_for_screening": ["RADIATIVE_RATE_STATE_BALANCE", "SOKR_ANTI_KASHA"],
        "support_boundary": (
            "Bright/dark ordering is a state-balance proxy, not direct evidence of "
            "higher-state emission or radiative-rate changes."
        ),
        "suggested_claim_strength": "weak_or_proxy",
        "evidence_tier": "computed_direct_proxy",
    },
    "microscopic.run_torsion_brightness_coupling_scan": {
        "observables": [
            "torsion snapshots",
            "geometry-dependent excitation energy",
            "oscillator-strength range",
        ],
        "useful_for_screening": ["RIM_RIR_RIV", "RACI_CI_ACCESS", "ICT_TICT_CT"],
        "support_boundary": (
            "Torsion-brightness coupling can motivate motion-coupled candidates. "
            "It is not conical-intersection search or nonadiabatic dynamics."
        ),
        "suggested_claim_strength": "weak_or_proxy",
        "evidence_tier": "computed_direct_proxy",
    },
    "microscopic.run_conformer_state_probe": {
        "observables": [
            "conformer-dependent state ordering",
            "conformer-dependent oscillator strength",
        ],
        "useful_for_screening": ["RIM_RIR_RIV", "RADIATIVE_RATE_STATE_BALANCE", "ICT_TICT_CT"],
        "support_boundary": (
            "Conformer sensitivity is a low-cost proxy and does not establish "
            "environmental restriction or excited-state relaxation."
        ),
        "suggested_claim_strength": "weak_or_proxy",
        "evidence_tier": "computed_direct_proxy",
    },
    "microscopic.run_frontier_orbital_partition": {
        "observables": ["HOMO/LUMO energies", "frontier orbital fragment weights"],
        "useful_for_screening": ["ICT_TICT_CT", "PET_ET"],
        "support_boundary": (
            "Frontier orbital partition is not NTO, electron-hole separation, "
            "redox measurement, or analyte-bound PET evidence."
        ),
        "suggested_claim_strength": "weak_or_proxy",
        "evidence_tier": "electronic_weak_trigger",
    },
    "microscopic.run_charge_population_panel": {
        "observables": ["Mulliken charges", "Lowdin charges", "Hirshfeld charges", "CM5 charges"],
        "useful_for_screening": ["ICT_TICT_CT", "PET_ET"],
        "support_boundary": (
            "Ground-state charge populations can only bound charge-localization "
            "questions; they do not prove excited-state CT or PET."
        ),
        "suggested_claim_strength": "weak_or_proxy",
        "evidence_tier": "electronic_weak_trigger",
    },
    "microscopic.run_solvation_polarity_proxy": {
        "observables": ["implicit-solvent state shift", "implicit-solvent oscillator shift"],
        "useful_for_screening": ["ICT_TICT_CT"],
        "support_boundary": (
            "Fixed-geometry implicit-solvent shifts are not experimental "
            "solvatochromism and do not by themselves prove TICT."
        ),
        "suggested_claim_strength": "weak_or_proxy",
        "evidence_tier": "electronic_weak_trigger",
    },
    "microscopic.run_targeted_transition_dipole_analysis": {
        "observables": [
            "transition-dipole output availability",
            "state ordering",
            "oscillator strength",
        ],
        "useful_for_screening": ["RADIATIVE_RATE_STATE_BALANCE", "SOKR_ANTI_KASHA"],
        "support_boundary": (
            "Current parsing confirms transition-dipole sections but does not "
            "turn them into direct lifetime, rate, or anti-Kasha evidence."
        ),
        "suggested_claim_strength": "weak_or_proxy",
        "evidence_tier": "computed_direct_proxy",
    },
    "microscopic.run_ris_state_characterization": {
        "observables": ["RIS/TDA state ordering cross-check", "oscillator strength"],
        "useful_for_screening": ["RADIATIVE_RATE_STATE_BALANCE", "SOKR_ANTI_KASHA"],
        "support_boundary": (
            "RIS state characterization is a low-cost cross-check and not direct "
            "higher-state emission or rate evidence."
        ),
        "suggested_claim_strength": "weak_or_proxy",
        "evidence_tier": "computed_direct_proxy",
    },
    "microscopic.run_targeted_localized_orbital_analysis": {
        "observables": ["localized orbital artifact availability", "MO file availability"],
        "useful_for_screening": ["ICT_TICT_CT", "PET_ET"],
        "support_boundary": (
            "Localized orbital availability is an artifact-level proxy, not a "
            "direct CT, PET, or NTO analysis."
        ),
        "suggested_claim_strength": "weak_or_proxy",
        "evidence_tier": "artifact_weak_trigger",
    },
    "microscopic.run_targeted_natural_orbital_analysis": {
        "observables": ["natural orbital artifact availability", "MO file availability"],
        "useful_for_screening": ["ICT_TICT_CT", "PET_ET"],
        "support_boundary": (
            "Natural orbital availability is an artifact-level proxy unless a "
            "validated parser extracts mechanism-specific quantities."
        ),
        "suggested_claim_strength": "weak_or_proxy",
        "evidence_tier": "artifact_weak_trigger",
    },
    "microscopic.extract_ct_descriptors_from_bundle": {
        "observables": ["artifact-backed CT surrogates", "available descriptor inventory"],
        "useful_for_screening": ["ICT_TICT_CT", "PET_ET"],
        "support_boundary": (
            "Extracted CT surrogates from a low-cost bundle are still not NTO, "
            "electron-hole separation, redox alignment, or PET quenching evidence."
        ),
        "suggested_claim_strength": "weak_or_proxy",
        "evidence_tier": "electronic_weak_trigger",
    },
    "microscopic.select_computed_geometry": {
        "observables": ["explicit geometry selection and provenance"],
        "useful_for_screening": [],
        "support_boundary": "Reuses a converged S0 scan geometry without adding evidence or assigning a mechanism. Explicit source and scan member required. Excited-state optimized geometries are suspended.",
        "suggested_claim_strength": "unsupported", "evidence_tier": "boundary_only",
    },
    "microscopic.run_relaxed_coordinate_scan": {
        "observables": ["constrained optimized structures", "point energies", "endpoint singlet states", "projected force convergence"],
        "useful_for_screening": ["RIM_RIR_RIV", "ICT_TICT_CT", "ESIPT_PT", "RADIATIVE_RATE_STATE_BALANCE"],
        "support_boundary": "S0 constrained local optimizations with vertical excited-state comparisons, not reaction paths or rate predictions. S1 force routes are suspended. Geometry changes and energy trends alone do not establish mechanisms.",
        "suggested_claim_strength": "weak_or_proxy", "evidence_tier": "computed_direct_proxy",
    },
    "microscopic.run_excited_state_relaxation": {
        "observables": ["typed unsupported boundary"],
        "useful_for_screening": [],
        "support_boundary": (
            "Suspended after installed Amesp aTB1/TDA S1/S2 force finite-difference failures. "
            "Previously converged runs are research records, not validated optimized geometries."
        ),
        "suggested_claim_strength": "unsupported",
        "evidence_tier": "boundary_only",
    },
    "microscopic.unsupported_excited_state_relaxation": {
        "observables": ["typed unsupported boundary"],
        "useful_for_screening": ["RACI_CI_ACCESS"],
        "support_boundary": (
            "Legacy boundary route: CI search and nonadiabatic dynamics remain outside current scope. "
            "run_excited_state_relaxation is also suspended pending force validation."
        ),
        "suggested_claim_strength": "unsupported",
        "evidence_tier": "boundary_only",
    },
}


def ability_evidence_guide(
    capability_registry: CapabilityRegistry,
) -> list[dict[str, object]]:
    """Return public, generic guidance connecting capabilities to evidence use.

    The guide is deliberately case-independent. It tells agents how a capability
    can bound a mechanism question and what it must not be overclaimed to prove.
    It does not contain source-paper facts, reference mechanisms, rankings, or
    benchmark-specific decisions.
    """

    cards = {str(item["capability_id"]): item for item in capability_registry.cards()}
    guide = []
    for capability_id, item in _ABILITY_EVIDENCE_GUIDE.items():
        card = cards.get(capability_id)
        if card is None:
            continue
        guide.append(
            {
                "capability_id": capability_id,
                "route": card["route"],
                "owner_agent": card["owner_agent"],
                "evidence_family": card["evidence_family"],
                "required_artifact_kinds": list(card["required_artifact_kinds"]),
                "observables": list(item["observables"]),  # type: ignore[arg-type]
                "useful_for_screening": [
                    label
                    for label in item["useful_for_screening"]  # type: ignore[index]
                    if label in MECHANISM_LABELS
                ],
                "support_boundary": item["support_boundary"],
                "suggested_claim_strength": item["suggested_claim_strength"],
                "evidence_tier": item["evidence_tier"],
            }
        )
    return guide


def compact_ability_evidence_guide(
    capability_registry: CapabilityRegistry,
) -> list[dict[str, object]]:
    """Return a compact runtime guide for LLM payloads.

    The full guide above is useful for auditability, but it is too verbose for
    every Planner/Agenda call. Runtime agents only need the public capability
    id, the mechanism questions it can screen, and the maximum support boundary.
    """

    cards = {str(item["capability_id"]): item for item in capability_registry.cards()}
    guide = []
    for capability_id, item in _ABILITY_EVIDENCE_GUIDE.items():
        card = cards.get(capability_id)
        if card is None:
            continue
        guide.append(
            {
                "capability_id": capability_id,
                "route": card["route"],
                "screens": [
                    label
                    for label in item["useful_for_screening"]  # type: ignore[index]
                    if label in MECHANISM_LABELS
                ],
                "support_ceiling": item["suggested_claim_strength"],
                "evidence_tier": item["evidence_tier"],
                "boundary": _compact_boundary(capability_id),
            }
        )
    return guide


def _compact_boundary(capability_id: str) -> str:
    if capability_id in {"microscopic.optimize_ground_state_geometry",
                         "macro.run_dimer_packing_proxy", "macro.run_aggregate_contact_proxy"}:
        return str(_ABILITY_EVIDENCE_GUIDE[capability_id]["support_boundary"])
    if capability_id == "microscopic.unsupported_excited_state_relaxation":
        return (
            "Legacy placeholder; explicit CI/nonadiabatic dynamics are outside current scope. "
            "run_excited_state_relaxation is also suspended pending force validation."
        )
    if capability_id == "microscopic.run_excited_state_relaxation":
        return "Suspended after excited-state force checks; no character tracking or validated optimized geometry."
    if capability_id.startswith("macro."):
        return "Structural/proxy screening only; not direct photophysical or wet-lab evidence."
    if capability_id in {
        "microscopic.run_baseline_bundle",
        "microscopic.run_bright_dark_state_ordering",
        "microscopic.run_targeted_transition_dipole_analysis",
        "microscopic.run_ris_state_characterization",
    }:
        return (
            "Low-cost state/rate proxy only; not lifetime, QY, anti-Kasha, "
            "or direct rate evidence."
        )
    if capability_id in {
        "microscopic.run_frontier_orbital_partition",
        "microscopic.run_charge_population_panel",
        "microscopic.run_solvation_polarity_proxy",
        "microscopic.run_targeted_localized_orbital_analysis",
        "microscopic.run_targeted_natural_orbital_analysis",
    }:
        return "Electronic proxy only; not direct NTO, PET, TICT, redox, or solvatochromic proof."
    if capability_id in {
        "microscopic.run_torsion_brightness_coupling_scan",
        "microscopic.run_conformer_state_probe",
    }:
        return "Motion-coupled proxy only; not direct RIM, CI, or nonadiabatic proof."
    return "Proxy screening only; keep claim status bounded."
