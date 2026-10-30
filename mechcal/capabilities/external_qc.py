"""Opt-in additions to the single capability registry, not a second registry."""

from mechcal.capabilities.schemas import CapabilitySpec, CapabilityToolBinding
from mechcal.tools.qc_process import QCConfig

ROUTES = {
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


def external_capabilities(config: QCConfig):
    result = []
    for route, (backend, family, description) in ROUTES.items():
        reason = config.backend_reason(backend)
        if route == "scan_excited_state_coordinate" and not config.allow_excited_scan:
            reason = "Excited-state scans await derivative and state-continuity validation."
        result.append(CapabilitySpec(
            capability_id=f"microscopic.{route}", owner_agent="microscopic",
            evidence_family=family, route=route, description=description,
            required_artifact_kinds=["prepared_structure"], outputs=[f"{route}_bundle"] +
                (["prepared_structure"] if route in {"search_dimer_contacts", "scan_intermolecular_distance"} else []),
            backend_enabled=reason is None, disabled_reason=reason,
            tool_binding=CapabilityToolBinding(backend="worker", tool_name="run_microscopic"),
            metadata={"external_backend": backend,
                      **({"spin_parameter_config": {"parameter_dir": config.parameter_dir,
                          "parameter_name": config.parameter_name, "parameter_revision": config.parameter_revision}}
                         if route in {"compare_singlet_triplet_states", "evaluate_spin_orbit_couplings"} else {}),
                      "limitations": [description.split(";", 1)[-1].strip()],
                      "tool_arg_hints": {"source_artifact_id": "Registered prepared_structure; zero-based atom indices preserve its XYZ/SDF order."}},
        ))
    return result
