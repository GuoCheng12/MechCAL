from __future__ import annotations

from mechcal.capabilities.registry import CapabilityRegistry
from mechcal.capabilities.schemas import CapabilitySpec, CapabilityToolBinding
from mechcal.schemas.tool_arguments import argument_model
from mechcal.chem.validated_scopes import SPATIAL_SCOPE, SCAN_SCOPE

CONFORMER_FORCE_FIELD_LIMIT = (
    "run_conformer_bundle and run_conformer_state_probe share the RDKit conformer "
    "generator and require MMFF94 or UFF parameters even when a prepared geometry "
    "exists. Switching between these routes does not bypass a parameter failure."
)

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
            "Compute a heuristic from single-molecule aromatic fraction, logP and rotatable-bond "
            "count only. No dimer, contact distances, geometry calculation or new independent "
            "observation. Reuses descriptors from scaffold/rotor screens."
        ),
        "outputs": ["dimer_packing_proxy_report"],
    },
    {
        "name": "run_aggregate_contact_proxy",
        "description": (
            "Screen monomer logP and heteroatom-count thresholds, without constructing "
            "an aggregate or computing contacts. No independent packing observations."
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
        "name": "optimize_ground_state_geometry", "family": "geometry_precondition",
        "description": "Recover an unoptimized initial_structure using Cartesian ASE BFGS and Amesp aTB1 ground-state forces. Default limits: 240 steps and 240 seconds. Publishes prepared_structure only when max atomic force is at most 0.02 eV/angstrom and basic geometry checks pass. No excited-state or mechanistic evidence; not a frequency-verified minimum.",
        "required": ["initial_structure"], "outputs": ["prepared_structure"], "cost": "medium",
    },
    {
        "name": "run_relaxed_coordinate_scan", "family": "conformer_sensitivity",
        "description": "Relax other coordinates at explicitly selected bond lengths or dihedrals with validated Amesp S0 forces. Returns per-point convergence, energies, vertical endpoint states and reusable geometries. Not a reaction path or mechanism verdict. " + SCAN_SCOPE,
        "required": ["prepared_structure"], "outputs": ["relaxed_coordinate_scan_bundle"], "cost": "high",
    },
    {
        "name": "select_computed_geometry", "family": "geometry_precondition",
        "description": "Explicitly select a converged S0 relaxed-scan member and preserve its atom mapping for further tools. Requires source_artifact_id and member_id. Excited-state optimized geometries are suspended. Does not run a calculation.",
        "required": ["relaxed_coordinate_scan_bundle"], "outputs": ["prepared_structure"], "cost": "low",
    },
    {
        "name": "run_excited_state_relaxation",
        "family": "state_ordering_brightness",
        "description": (
            "Suspended: installed Amesp aTB1/TDA excited-state forces failed finite-difference validation. "
            "Returns an unsupported boundary; use S0 relaxed scans and vertical state comparisons instead."
        ),
        "required": ["prepared_structure"],
        "outputs": ["excited_state_relaxation_bundle"],
        "cost": "high",
    },
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
        "description": "Generate RDKit force-field conformers and characterize their electronic states.",
        "limitations": [CONFORMER_FORCE_FIELD_LIMIT],
        "required": ["prepared_structure"],
        "outputs": ["conformer_bundle"],
        "cost": "medium",
    },
    {
        "name": "run_conformer_state_probe",
        "family": "conformer_sensitivity",
        "description": "Generate RDKit force-field conformers and probe their vertical electronic states.",
        "limitations": [CONFORMER_FORCE_FIELD_LIMIT],
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
        "name": "analyze_hole_electron_from_bundle",
        "family": "charge_localization",
        "description": "Postprocess matched canonical TDA-aTB1 orbitals and coefficients into state-resolved hole/electron distributions, centroid distance and overlap; no new quantum job.",
        "required": ["amesp_baseline_bundle"],
        "outputs": ["hole_electron_bundle"],
        "cost": "low",
        "limitations": ["Optional Multiwfn and Amesp m2m required. " + SPATIAL_SCOPE + " Requires td_tout >= 4 and matching successful job files. Atom indices refer to calculation geometry. Optional fragments sum heavy-atom contributions, not transfer probabilities. Au-containing monomers and modeled dimers are supported; this is not SOC or triplet analysis. Unrelaxed vertical excitation is not emitting-state, TICT or PET-kinetic validation."],
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
        "description": "Measure selected distances, angles or dihedrals and expose the XYZ atom map and global geometry descriptors.",
        "required": ["amesp_baseline_bundle"],
        "outputs": ["geometry_descriptor_bundle"],
        "cost": "low",
    },
    {
        "name": "inspect_raw_artifact_bundle",
        "family": "raw_artifact_inspection",
        "description": "List raw file paths and extractable observables in a bundle, without reading file contents.",
        "required": ["amesp_baseline_bundle"],
        "outputs": ["raw_artifact_inventory"],
        "cost": "low",
    },
    {
        "name": "unsupported_excited_state_relaxation",
        "family": "state_ordering_brightness",
        "description": "CI/nonadiabatic dynamics are unsupported; run_excited_state_relaxation is also suspended pending force validation.",
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
    import os
    capabilities = default_capabilities()
    if os.environ.get("MECHCAL_QC_CONFIG"):
        from mechcal.capabilities.external_qc import external_capabilities
        from mechcal.tools.qc_process import QCConfig
        capabilities.extend(external_capabilities(QCConfig.from_environment()))
    return CapabilityRegistry(capabilities)


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
                "tool_arg_schema": argument_model(card["name"], "macro").model_json_schema(),
                "tool_arg_hints": {
                    "focus_tags": "optional list[str] of route-local descriptors to emphasize"
                },
            },
        )
        for card in MACRO_CAPABILITY_CARDS
    ]


def _microscopic_capabilities() -> list[CapabilitySpec]:
    from mechcal.tools.amesp_microscopic import (
        microscopic_capability_policy, microscopic_delivery_contract, microscopic_execution_dependencies,
    )

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
            **microscopic_capability_policy(card["name"]),
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
                "tool_arg_schema": argument_model(card["name"]).model_json_schema(),
                "tool_arg_hints": _microscopic_tool_arg_hints(card["name"]),
                "limitations": card.get("limitations", []),
                "execution_dependencies": microscopic_execution_dependencies(card["name"]),
                "delivery_contract": microscopic_delivery_contract(card["name"]),
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
        hints["td_tout"] = "optional int 1-10; transition coefficient printing threshold 10^-N, not a state count"
    if name == "run_solvation_polarity_proxy":
        hints["solvents"] = "optional 2-4 distinct solvents from the enum in tool_arg_schema"
    if name == "run_charge_population_panel":
        hints["charge_schemes"] = "default mulliken/lowdin; hirshfeld/cm5 may be requested but require valid charge tables"
    if name in {"run_torsion_snapshots", "run_torsion_brightness_coupling_scan"}:
        hints.update(dihedral_id="ID returned by list_rotatable_dihedrals for the same source artifact",
                     dihedral_atoms="alternative to ID: four zero-based atom indices in prepared structure order",
                     angles="2-6 distinct absolute angles in [-180,180]; rigid snapshots, not relaxed scan")
    if name in {"run_conformer_bundle", "run_conformer_state_probe"}:
        hints["conformer_ids"] = "2-6 IDs from list_available_conformers for the same source artifact"
    if name == "run_frontier_orbital_partition":
        hints.update(donor_atoms="explicit zero-based electronic donor fragment atom indices",
                     acceptor_atoms="explicit zero-based electronic acceptor fragment; disjoint from donor")
    if name == "run_excited_state_relaxation":
        hints.update(
            state_root="all excited-state relaxation suspended after derivative validation",
            s1_nstates="state window 1-10, default 3",
            td_tout="transition coefficient printing threshold 10^-N, default 3; not the number of states",
            max_opt_cycles="optimization iterations 1-200, default 60; exhaustion returns incomplete evidence",
        )
    if name == "run_relaxed_coordinate_scan":
        hints.update(state_root="0 only in production; excited-state force routes are suspended",
                     atom_indices="explicit zero-based bonded pair or four-atom dihedral",
                     values="2-6 distinct bond lengths (angstrom) or dihedrals (degrees)",
                     max_wall_seconds="whole-route wall limit, 5-1800 seconds; default 300")
    if name == "select_computed_geometry":
        hints.update(source_artifact_id="required S0 relaxed-scan bundle ID",
                     member_id="required converged scan member, for example scan:01")
    if name in {"run_baseline_bundle", "run_bright_dark_state_ordering"}:
        hints["source_artifact_id"] += (
            "; explicitly selected computed geometries are kept fixed for single-point and vertical-state "
            "calculations; ordinary prepared structures undergo S0 optimization"
        )
    if name == "analyze_hole_electron_from_bundle":
        hints.update(state_root="1-based singlet state present in the selected calculation, default 1",
                     spatial_grid="medium or high, default high; fixed cross-term threshold 0.001",
                     max_wall_seconds="10-300 seconds for all conversion and analysis steps; default 300")
    return hints
