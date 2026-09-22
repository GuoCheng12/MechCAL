from __future__ import annotations

from mechcal.capabilities.registry import CapabilityRegistry
from mechcal.capabilities.schemas import CapabilitySpec, CapabilityToolBinding

MACRO_CAPABILITY_CARDS = [
    {
        "name": "macro_structure_scan",
        "description": (
            "Collect low-cost structural_prior proxy descriptors; this is not "
            "direct mechanism evidence."
        ),
        "outputs": ["macro_geometry_proxy_report"],
    },
    {
        "name": "screen_donor_acceptor_layout",
        "description": (
            "Screen donor/acceptor layout as a bounded structural_prior proxy; "
            "this is not direct mechanism evidence."
        ),
        "outputs": ["screen_donor_acceptor_layout_proxy_report"],
    },
    {
        "name": "screen_rotor_torsion_topology",
        "description": (
            "Screen rotor/torsion topology as a bounded structural_prior proxy; "
            "this is not direct mechanism evidence."
        ),
        "outputs": ["screen_rotor_torsion_topology_proxy_report"],
    },
    {
        "name": "screen_planarity_compactness",
        "description": (
            "Screen planarity/compactness as a bounded structural_prior proxy; "
            "this is not direct mechanism evidence."
        ),
        "outputs": ["screen_planarity_compactness_proxy_report"],
    },
    {
        "name": "screen_intramolecular_hbond_preorganization",
        "description": (
            "Screen intramolecular H-bond preorganization as a bounded "
            "structural_prior proxy; this is not direct mechanism evidence."
        ),
        "outputs": ["screen_intramolecular_hbond_preorganization_proxy_report"],
    },
    {
        "name": "screen_conformer_geometry_proxy",
        "description": (
            "Screen conformer geometry as a bounded structural_prior proxy; "
            "this is not direct mechanism evidence."
        ),
        "outputs": ["screen_conformer_geometry_proxy_report"],
    },
    {
        "name": "screen_neutral_aromatic_structure",
        "description": (
            "Screen neutral aromatic structure as a bounded structural_prior "
            "proxy; this is not direct mechanism evidence."
        ),
        "outputs": ["screen_neutral_aromatic_structure_proxy_report"],
    },
    {
        "name": "screen_donor_acceptor_architecture",
        "description": (
            "Screen donor-acceptor architecture as a bounded photophysics "
            "structural_prior proxy; this is not direct mechanism evidence."
        ),
        "outputs": ["screen_donor_acceptor_architecture_proxy_report"],
    },
    {
        "name": "screen_cationic_targeting_prior",
        "description": (
            "Screen formal charge and cationic-center priors as bounded targeting "
            "proxies; this is not biological localization evidence."
        ),
        "outputs": ["screen_cationic_targeting_prior_proxy_report"],
    },
    {
        "name": "screen_lipophilicity_proxy",
        "description": (
            "Screen lipophilicity descriptors as bounded partitioning proxies; "
            "this is not an experimental logP measurement."
        ),
        "outputs": ["screen_lipophilicity_proxy_report"],
    },
    {
        "name": "screen_polar_binding_site_prior",
        "description": (
            "Screen polar, charged, carbonyl, and H-bonding sites as bounded "
            "host-guest/PET interaction priors; this is not binding, sensing, "
            "redox, or guest-uptake evidence."
        ),
        "outputs": ["screen_polar_binding_site_prior_proxy_report"],
    },
    {
        "name": "screen_metal_triplet_prior",
        "description": (
            "Screen metal, lanthanide, heavy-atom, and triplet-relevant structural "
            "priors; this is not phosphorescence, RTP, TADF, or energy-transfer "
            "evidence."
        ),
        "outputs": ["screen_metal_triplet_prior_proxy_report"],
    },
    {
        "name": "screen_rotor_rim_prior",
        "description": (
            "Screen rotor/RIM structural priors as bounded topology proxies; "
            "this is not viscosity or AIE evidence."
        ),
        "outputs": ["screen_rotor_rim_prior_proxy_report"],
    },
    {
        "name": "screen_esipt_structural_motif",
        "description": (
            "Screen ESIPT-compatible donor/acceptor motifs as a bounded "
            "structural proxy; this is not proton-transfer evidence."
        ),
        "outputs": ["screen_esipt_structural_motif_proxy_report"],
    },
    {
        "name": "screen_aggregation_prone_scaffold",
        "description": (
            "Screen aromaticity, conjugation, and lipophilicity proxies for "
            "aggregation-prone scaffolds; this is not DLS or solid-state PL evidence."
        ),
        "outputs": ["screen_aggregation_prone_scaffold_proxy_report"],
    },
    {
        "name": "screen_pi_stacking_prone_geometry",
        "description": (
            "Screen planar aromatic geometry proxies for pi-stacking propensity; "
            "this is not crystal packing evidence."
        ),
        "outputs": ["screen_pi_stacking_prone_geometry_proxy_report"],
    },
    {
        "name": "run_dimer_packing_proxy",
        "description": (
            "Estimate a lightweight dimer packing/contact proxy from single-molecule "
            "geometry; this is not molecular dynamics or crystal prediction."
        ),
        "outputs": ["dimer_packing_proxy_report"],
    },
    {
        "name": "run_aggregate_contact_proxy",
        "description": (
            "Screen aggregate contact priors from shape, aromaticity, and polarity "
            "descriptors; this is not an aggregate simulation."
        ),
        "outputs": ["aggregate_contact_proxy_report"],
    },
    {
        "name": "run_crystal_restriction_checklist",
        "description": (
            "Return a bounded checklist for crystal/solid-state restriction evidence; "
            "this does not claim crystal measurements."
        ),
        "outputs": ["crystal_restriction_checklist_report"],
    },
    {
        "name": "run_solid_state_emission_proxy",
        "description": (
            "Screen solid-state emission plausibility from structural proxies and "
            "known evidence gaps; this is not solid-state PL evidence."
        ),
        "outputs": ["solid_state_emission_proxy_report"],
    },
    {
        "name": "run_water_fraction_aggregation_checklist",
        "description": (
            "Return a checklist for water-fraction aggregation evidence such as PL "
            "and DLS; this does not fabricate wet-lab results."
        ),
        "outputs": ["water_fraction_aggregation_checklist_report"],
    },
]

MICROSCOPIC_CAPABILITY_CARDS = [
    {
        "name": "list_rotatable_dihedrals",
        "family": "geometry_precondition",
        "description": "List reusable rotatable dihedral candidates from prepared structure.",
        "required": ["prepared_structure"],
        "outputs": ["rotatable_dihedral_inventory"],
        "cost": "low",
    },
    {
        "name": "list_available_conformers",
        "family": "conformer_sensitivity",
        "description": "List reusable conformer descriptors from prepared structure metadata.",
        "required": ["prepared_structure"],
        "outputs": ["conformer_inventory"],
        "cost": "low",
    },
    {
        "name": "list_artifact_bundles",
        "family": "raw_artifact_inspection",
        "description": "List reusable microscopic artifact bundle descriptors.",
        "required": ["amesp_baseline_bundle"],
        "outputs": ["artifact_bundle_inventory"],
        "cost": "low",
    },
    {
        "name": "list_artifact_bundle_members",
        "family": "raw_artifact_inspection",
        "description": "List stable members inside a reusable microscopic artifact bundle.",
        "required": ["amesp_baseline_bundle"],
        "outputs": ["artifact_bundle_member_inventory"],
        "cost": "low",
    },
    {
        "name": "run_baseline_bundle",
        "family": "state_ordering_brightness",
        "description": (
            "Run a low-cost S0/S1 microscopic baseline bundle with typed "
            "fallback status."
        ),
        "required": ["prepared_structure"],
        "outputs": ["amesp_baseline_bundle", "state_ordering_brightness"],
        "cost": "medium",
    },
    {
        "name": "run_bright_dark_state_ordering",
        "family": "state_ordering_brightness",
        "description": "Run low-cost excited-state ordering and bright/dark state scan.",
        "required": ["prepared_structure"],
        "outputs": ["bright_dark_state_ordering_bundle"],
        "cost": "medium",
    },
    {
        "name": "run_frontier_orbital_partition",
        "family": "charge_localization",
        "description": "Run low-cost frontier orbital partition from Amesp text output.",
        "required": ["prepared_structure"],
        "outputs": ["frontier_orbital_partition_bundle"],
        "cost": "medium",
    },
    {
        "name": "run_charge_population_panel",
        "family": "charge_localization",
        "description": "Run bounded Mulliken/Lowdin/Hirshfeld/CM5 charge panel.",
        "required": ["prepared_structure"],
        "outputs": ["charge_population_panel_bundle"],
        "cost": "medium",
    },
    {
        "name": "run_solvation_polarity_proxy",
        "family": "state_ordering_brightness",
        "description": "Run fixed-geometry implicit-solvent excited-state polarity proxy.",
        "required": ["prepared_structure"],
        "outputs": ["solvation_polarity_proxy_bundle"],
        "cost": "medium",
    },
    {
        "name": "run_conformer_bundle",
        "family": "conformer_sensitivity",
        "description": "Run bounded conformer bundle characterization from prepared structure.",
        "required": ["prepared_structure"],
        "outputs": ["conformer_bundle"],
        "cost": "medium",
    },
    {
        "name": "run_conformer_state_probe",
        "family": "conformer_sensitivity",
        "description": "Run bounded fixed-geometry conformer state probe.",
        "required": ["prepared_structure"],
        "outputs": ["conformer_state_probe_bundle"],
        "cost": "medium",
    },
    {
        "name": "run_torsion_snapshots",
        "family": "torsion_sensitivity",
        "description": "Run bounded torsion snapshot characterization from prepared structure.",
        "required": ["prepared_structure"],
        "outputs": ["torsion_snapshot_bundle"],
        "cost": "medium",
    },
    {
        "name": "run_torsion_brightness_coupling_scan",
        "family": "torsion_sensitivity",
        "description": "Run torsion snapshot brightness-coupling proxy from prepared structure.",
        "required": ["prepared_structure"],
        "outputs": ["torsion_brightness_coupling_bundle"],
        "cost": "medium",
    },
    {
        "name": "extract_ct_descriptors_from_bundle",
        "family": "charge_localization",
        "description": (
            "Inspect an existing microscopic bundle for bounded CT descriptor "
            "surrogates."
        ),
        "required": ["amesp_baseline_bundle"],
        "outputs": ["ct_descriptor_bundle", "artifact_backed_ct_surrogates"],
        "cost": "low",
    },
    {
        "name": "run_targeted_state_characterization",
        "family": "state_ordering_brightness",
        "description": (
            "Run bounded fixed-geometry targeted state characterization from "
            "a reusable bundle."
        ),
        "required": ["amesp_baseline_bundle"],
        "outputs": ["targeted_state_characterization_bundle"],
        "cost": "medium",
    },
    {
        "name": "run_targeted_charge_analysis",
        "family": "charge_localization",
        "description": (
            "Run bounded fixed-geometry targeted charge analysis from a "
            "reusable bundle."
        ),
        "required": ["amesp_baseline_bundle"],
        "outputs": ["targeted_charge_analysis_bundle"],
        "cost": "medium",
    },
    {
        "name": "run_targeted_density_population_analysis",
        "family": "charge_localization",
        "description": (
            "Run bounded fixed-geometry density/population analysis from a "
            "reusable bundle."
        ),
        "required": ["amesp_baseline_bundle"],
        "outputs": ["targeted_density_population_bundle"],
        "cost": "medium",
    },
    {
        "name": "run_targeted_transition_dipole_analysis",
        "family": "state_ordering_brightness",
        "description": (
            "Run bounded fixed-geometry transition-dipole analysis from a "
            "reusable bundle."
        ),
        "required": ["amesp_baseline_bundle"],
        "outputs": ["targeted_transition_dipole_bundle"],
        "cost": "medium",
    },
    {
        "name": "run_targeted_charge_redistribution_analysis",
        "family": "charge_localization",
        "description": "Run bounded charge-redistribution analysis from a reusable bundle.",
        "required": ["amesp_baseline_bundle"],
        "outputs": ["targeted_charge_redistribution_bundle"],
        "cost": "high",
    },
    {
        "name": "run_ris_state_characterization",
        "family": "state_ordering_brightness",
        "description": (
            "Run bounded fixed-geometry RIS state characterization from a "
            "reusable bundle."
        ),
        "required": ["amesp_baseline_bundle"],
        "outputs": ["ris_state_characterization_bundle"],
        "cost": "medium",
    },
    {
        "name": "run_targeted_localized_orbital_analysis",
        "family": "charge_localization",
        "description": (
            "Run bounded localized-orbital availability analysis from a "
            "reusable bundle."
        ),
        "required": ["amesp_baseline_bundle"],
        "outputs": ["targeted_localized_orbital_bundle"],
        "cost": "high",
    },
    {
        "name": "run_targeted_natural_orbital_analysis",
        "family": "charge_localization",
        "description": "Run bounded natural-orbital availability analysis from a reusable bundle.",
        "required": ["amesp_baseline_bundle"],
        "outputs": ["targeted_natural_orbital_bundle"],
        "cost": "high",
    },
    {
        "name": "parse_snapshot_outputs",
        "family": "state_ordering_brightness",
        "description": "Parse reusable microscopic snapshot or baseline bundle outputs.",
        "required": ["amesp_baseline_bundle"],
        "outputs": ["parsed_snapshot_bundle"],
        "cost": "low",
    },
    {
        "name": "extract_torsion_candidates_from_bundle",
        "family": "torsion_sensitivity",
        "description": "Extract torsion candidates from reusable bundle geometry.",
        "required": ["amesp_baseline_bundle"],
        "outputs": ["torsion_candidate_bundle"],
        "cost": "low",
    },
    {
        "name": "extract_geometry_descriptors_from_bundle",
        "family": "geometry_precondition",
        "description": "Extract bounded geometry descriptors from reusable bundle coordinates.",
        "required": ["amesp_baseline_bundle"],
        "outputs": ["geometry_descriptor_bundle"],
        "cost": "low",
    },
    {
        "name": "inspect_raw_artifact_bundle",
        "family": "raw_artifact_inspection",
        "description": "Inspect raw files and extractable observables in a reusable bundle.",
        "required": ["amesp_baseline_bundle"],
        "outputs": ["raw_artifact_inventory"],
        "cost": "low",
    },
    {
        "name": "unsupported_excited_state_relaxation",
        "family": "state_ordering_brightness",
        "description": "Return a typed unsupported result for excited-state relaxation requests.",
        "required": [],
        "outputs": [],
        "cost": "low",
    },
]


def default_capabilities() -> list[CapabilitySpec]:
    return [
        *_macro_capabilities(),
        *_microscopic_capabilities(),
    ]


def default_capability_registry() -> CapabilityRegistry:
    return CapabilityRegistry(default_capabilities())


def _macro_capabilities() -> list[CapabilitySpec]:
    return [
        CapabilitySpec(
            capability_id=f"macro.{card['name']}",
            owner_agent="macro",
            evidence_family="geometry_precondition",
            route=card["name"],
            description=card["description"],
            required_artifact_kinds=[],
            outputs=card["outputs"],
            cost="low",
            failure_modes=["precondition_missing", "runtime_failed"],
            tool_binding=CapabilityToolBinding(
                backend="worker",
                tool_name="run_macro",
            ),
            metadata={
                "mechanism_directness": "not_direct_mechanism_evidence",
                "tool_arg_hints": {
                    "focus_tags": "optional list[str] of route-local descriptors to emphasize"
                },
            },
        )
        for card in MACRO_CAPABILITY_CARDS
    ]


def _microscopic_capabilities() -> list[CapabilitySpec]:
    return [
        CapabilitySpec(
            capability_id=f"microscopic.{card['name']}",
            owner_agent="microscopic",
            evidence_family=card["family"],  # type: ignore[arg-type]
            route=card["name"],
            description=card["description"],
            required_artifact_kinds=card["required"],
            outputs=card["outputs"],
            cost=card["cost"],  # type: ignore[arg-type]
            failure_modes=[
                "capability_unsupported",
                "precondition_missing",
                "runtime_failed",
            ],
            tool_binding=CapabilityToolBinding(
                backend="worker",
                tool_name="run_microscopic",
            ),
            metadata={
                "legacy_capacity_name": card["name"],
                "tool_arg_hints": _microscopic_tool_arg_hints(card["name"]),
            },
        )
        for card in MICROSCOPIC_CAPABILITY_CARDS
    ]


def _microscopic_tool_arg_hints(name: str) -> dict[str, str]:
    hints = {
        "source_artifact_id": (
            "optional artifact id from artifact_refs when multiple candidates exist"
        )
    }
    if name in {
        "run_conformer_bundle",
        "run_conformer_state_probe",
        "run_torsion_snapshots",
        "run_torsion_brightness_coupling_scan",
    }:
        hints["max_members"] = "optional int 2-6 for generated conformer/snapshot members"
    if name in {
        "run_baseline_bundle",
        "run_bright_dark_state_ordering",
        "run_conformer_bundle",
        "run_conformer_state_probe",
        "run_torsion_snapshots",
        "run_torsion_brightness_coupling_scan",
        "run_targeted_state_characterization",
        "run_targeted_transition_dipole_analysis",
        "run_targeted_charge_redistribution_analysis",
        "run_ris_state_characterization",
        "run_targeted_localized_orbital_analysis",
        "run_targeted_natural_orbital_analysis",
        "run_solvation_polarity_proxy",
    }:
        hints["s1_nstates"] = "optional int 1-10 for vertical excited-state calculation"
        hints["td_tout"] = "optional int 1-10 for transition output depth"
    if name == "run_solvation_polarity_proxy":
        hints["solvents"] = "optional list[str] from water/toluene/acetonitrile/methanol"
    if name == "run_charge_population_panel":
        hints["charge_schemes"] = "optional list[str] from mulliken/lowdin/hirshfeld/cm5"
    return hints
