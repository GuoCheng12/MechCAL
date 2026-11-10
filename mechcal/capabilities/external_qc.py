"""Opt-in additions to the single capability registry, not a second registry."""

from mechcal.capabilities.schemas import CapabilitySpec, CapabilityToolBinding
from mechcal.tools.qc_process import QCConfig

ROUTES = {
    "estimate_soc_amesp": ("amesp_soc", "state_ordering_brightness", "Compute B3LYP/TDA/def2 vertical spin states and Amesp effective-charge SOC estimates at a fixed neutral closed-shell geometry, including scalar-ECP Au. Approximate SOC matrix, not an all-electron relativistic reference, ISC rate or measured triplet population."),
    "scan_cis_excitation_coordinate": ("pyscf", "torsion_sensitivity", "Compare 2-4 fixed proton-bond or torsion coordinates with RHF/6-31G/CIS and transition-density state tracking. Connected neutral CHNO, 2-40 atoms. Return S0 and singlet energies, strengths and ambiguity flags; not relaxed barriers, CI or rates."),
    "sample_conformers_crest": ("crest", "conformer_sensitivity", "Sample conformers with CREST 3.0.2/GFN2-xTB and return atom-mapped geometries. max_members limits output, not search; optional md_length_ps controls trajectory length. Automatic searches can exceed several minutes. Unconstrained and non-exhaustive; no freezing or independent energy model."),
    "compare_contact_torsion_response": ("xtb", "torsion_sensitivity", "Compare a bonded torsion or explicit off-axis fragment azimuth around a single/dative axis in a modeled dimer and isolated frozen fragments using GFN2-xTB. Separate monomer strain and contact interaction changes; rigid local responses, not relaxed barriers, measured packing or fluorescence rates."),
    "compare_spin_states_pyscf": ("pyscf_tda", "state_ordering_brightness", "Compare fixed-geometry singlet/triplet states with DF-PBE or DF-PBE0/TDA, def2 basis and scalar Au ECP; return oscillator strengths and Lowdin hole/electron populations. Supports closed-shell charged structures and elements through Au in the declared scope. Does NOT compute SOC, ISC or triplet yield."),
    "evaluate_spin_orbit_couplings": ("pysoc", "state_ordering_brightness", "Compute spin-orbit coupling magnitudes between S0-S3 and T1-T3 at one unchanged geometry with TD-DFTB/mio-1-1 and PySOC ZEFF; neutral closed-shell CHNO only. Compare with spin energies; not ISC rates, triplet yields or a mechanism verdict."),
    "estimate_vertical_ip_ea": ("xtb", "charge_localization", "Estimate vertical ionization potential and electron affinity with IPEA-xTB; not a PET rate or excited-state transfer free energy."),
    "analyze_ground_state_vibrations": ("xtb", "torsion_sensitivity", "Optimize a ground-state structure and identify vibrational frequencies and atom-resolved motions; not a nonradiative decay calculation."),
    "search_dimer_contacts": ("xtb", "geometry_precondition", "Search up to four optimized dimer poses with aISS/GFN2-xTB; return contacts, interaction energies and registered prepared geometries for explicit downstream selection. Modeled dimers, not observed packing."),
    "scan_intermolecular_distance": ("xtb", "conformer_sensitivity", "Scan an explicit atom-pair separation in a modeled dimer by rigidly translating one fragment; return GFN2 energies, contacts and reusable fixed geometries. Not a relaxed path, crystal or excited-state calculation."),
    "evaluate_excited_state_gradient": ("dftb", "state_ordering_brightness", "Evaluate a selected singlet energy and Cartesian forces with TD-DFTB as preparation for geometry/path analysis; CHNO parameter scope only."),
    "compare_singlet_triplet_states": ("dftb", "state_ordering_brightness", "Compare vertical singlet and triplet energies and singlet oscillator strengths at one fixed geometry with TD-DFTB/mio-1-1; neutral closed-shell CHNO only, not SOC, ISC rates or triplet yield."),
    "evaluate_cis_s1_gradient": ("pyscf", "torsion_sensitivity", "Compare matched S0/S1 local gradients and force changes at one unchanged geometry with PySCF 2.14.0 RHF/6-31G/CIS; connected neutral closed-shell CHNO, 2-40 atoms. Qualitative excitation-energy sensitivity, not relaxation, state tracking or decay rates."),
    "scan_excited_state_coordinate": ("dftb", "torsion_sensitivity", "Relax a short bond or torsion scan on a selected singlet TD-DFTB root; disabled until derivative validation; not dynamics or a conical intersection search."),
}


# Operational guidance belongs to the same registry as the interface. These are
# measured scope limitations, not mechanism scores or automatic scientific choices.
RUNTIME_GUIDANCE = {
    "estimate_soc_amesp": "High-cost experimental route. Small structures can finish in minutes, but a full Au complex exceeded 25 minutes in acceptance testing. Supported elements do not imply practical completion. Request explicitly; do not automatically repeat a timed-out job unchanged.",
    "compare_spin_states_pyscf": "Potentially expensive. Larger 50-60 atom structures returned no excited states within a 180 s probe. Stability validation can dominate runtime; completed states with an incomplete/unstable reference remain provisional. No automatic replacement for SOC.",
    "sample_conformers_crest": "Potentially expensive. max_members truncates returned results only. Inspect or set md_length_ps explicitly for a short exploratory search; shortened sampling trades coverage for time. Only checked final-ensemble geometries are reusable.",
    "compare_contact_torsion_response": "Low-cost local probe once a modeled dimer exists. Use off-axis anchors and fragment_azimuth for near-linear single/dative coordination axes. Compare a small local angular window, isolated strain, another contact pose and a far-separation control; do not infer a barrier from a rigid clash.",
    "search_dimer_contacts": "Variable-cost geometry search. A candidate can exhaust max_opt_cycles without converging; a failed pose is not a reusable optimized structure or negative packing evidence. max_candidates and max_opt_cycles are explicit controls, not success guarantees.",
}


def external_capabilities(config: QCConfig):
    result = []
    for route, (backend, family, description) in ROUTES.items():
        reason = config.backend_reason(backend)
        if route == "scan_excited_state_coordinate" and not config.allow_excited_scan:
            reason = "Excited-state scans await derivative and state-continuity validation."
        result.append(CapabilitySpec(
            capability_id=f"microscopic.{route}", owner_agent="microscopic",
            evidence_family=family, route=route, description=description,
            cost="high" if backend in {"amesp_soc", "pyscf_tda", "crest"} else
                "medium" if route in {"search_dimer_contacts", "scan_cis_excitation_coordinate"} else "low",
            required_artifact_kinds=["prepared_structure"], outputs=[f"{route}_bundle"] +
                (["prepared_structure"] if route in {"search_dimer_contacts", "scan_intermolecular_distance", "sample_conformers_crest", "compare_contact_torsion_response"} else []),
            backend_enabled=reason is None, disabled_reason=reason,
            requires_amesp=backend == "amesp_soc",
            tool_binding=CapabilityToolBinding(backend="worker", tool_name="run_microscopic"),
            metadata={"external_backend": backend,
                      **({"spin_parameter_config": {"parameter_dir": config.parameter_dir,
                          "parameter_name": config.parameter_name, "parameter_revision": config.parameter_revision}}
                         if route in {"compare_singlet_triplet_states", "evaluate_spin_orbit_couplings"} else {}),
                      "limitations": [description.split(";", 1)[-1].strip()] +
                          ([RUNTIME_GUIDANCE[route]] if route in RUNTIME_GUIDANCE else []),
                      "tool_arg_hints": {"source_artifact_id": "Registered prepared_structure; zero-based atom indices preserve its XYZ/SDF order.",
                          **({"compute_effort": RUNTIME_GUIDANCE[route]} if route in RUNTIME_GUIDANCE else {})}},
        ))
    return result
