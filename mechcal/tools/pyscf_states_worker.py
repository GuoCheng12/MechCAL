"""Isolated, spin-free TDA state analysis; never substitutes gaps for SOC."""

import json
from pathlib import Path
import sys
import time

import numpy as np

ELEMENTS = {"H", "C", "N", "O", "F", "S", "Cl", "Br", "B", "Si", "Au", "Fe"}


def populations(mol, overlap, coefficients, weights):
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    if eigenvalues.min() < 1e-9:
        raise ValueError("Near linearly dependent AO basis; population analysis not valid.")
    root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    density = np.sum((root @ coefficients)**2 * weights[None, :], axis=1)
    result = [float(density[start:end].sum()) for _, _, start, end in mol.aoslice_by_atom()]
    if not np.isclose(sum(result), 1., atol=1e-6):
        raise ValueError("Hole/electron populations do not sum to one.")
    return result


def calculate(request, output):
    from pyscf import gto, dft, lib, tdscf, __version__
    if __version__ != "2.14.0":
        raise ValueError("This route is validated against PySCF 2.14.0.")
    symbols = request["symbols"]
    if not set(symbols) <= ELEMENTS or not 2 <= len(symbols) <= 150:
        raise ValueError("Unsupported element or atom count.")
    lib.num_threads(request["threads"])
    started = time.monotonic()
    ecp = {s: "def2-svp" for s in set(symbols) if s == "Au"}
    mol = gto.M(atom=list(zip(symbols, request["coordinates_angstrom"])), unit="Angstrom",
        basis=request["basis"], ecp=ecp, charge=request["charge"], spin=0,
        max_memory=2000, verbose=4, output=str(output.with_suffix(".log")))
    mf = dft.RKS(mol).density_fit()
    mf.xc = request["functional"]
    mf.grids.level = request["grid_level"]
    mf.conv_tol = 1e-8
    mf.conv_tol_grad = 1e-5
    mf.max_cycle = 80
    mf.chkfile = str(output.with_suffix(".chk"))
    if request["scf_solver"] == "newton":
        mf = mf.newton()
    mf.kernel()
    if not mf.converged:
        raise ValueError("SCF did not converge; no state energies accepted.")
    scf_seconds = time.monotonic()-started
    # Keep the same DF reference for response and stability, including Newton SCF.
    if hasattr(mf, "undo_soscf"):
        mf = mf.undo_soscf()
    report = dict(schema_version="pyscf_spin_states_v1", request=request, software_version=__version__,
        method=f"DF-{request['functional'].upper()}/TDA/{request['basis']}, grid {request['grid_level']}, gas phase",
        scf_converged=True, scf_internal_stable=None, scf_external_stable=None,
        stability_check_status="not_completed", validation_failures=[],
        ground_energy_hartree=float(mf.e_tot), nao=mol.nao_nr(), scf_seconds=scf_seconds,
        ecp_core_electrons_by_atom=[int(mol.atom_nelec_core(i)) for i in range(mol.natm)],
        scalar_ecp=ecp, states=[], spin_manifold_failures=[],
        limitations=["Spin-free TDA; scalar ECP does not compute SOC, ISC, triplet yield or lifetime.",
            "PBE can underestimate charge-transfer energies; compare sensitivity to PBE0 and basis before interpretation.",
            "Lowdin populations describe this basis-dependent excitation, not observed charge transfer.",
            "S/T indices order energies at this geometry; the same index at another geometry need not identify the same state.",
            "Restricted-reference stability is reported, not a proof of the actual ground-state spin."])
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    overlap = mf.get_ovlp()
    for singlet in (True, False):
        label = "S" if singlet else "T"
        try:
            td = tdscf.TDA(mf)
            td.singlet = singlet
            td.nstates = request["nstates_per_spin"]
            td.conv_tol = 1e-6
            td.max_cycle = 100
            td.kernel()
            if len(td.e) != td.nstates:
                raise ValueError(f"{label} requested root count not returned.")
            vind, _ = td.gen_vind(mf)
            strengths = td.oscillator_strength() if singlet else [None]*len(td.e)
            manifold = []
            for i, (energy, xy, oscillator) in enumerate(zip(td.e, td.xy, strengths)):
                if not td.converged[i] or energy <= 0:
                    report["spin_manifold_failures"].append(dict(manifold=f"{label}{i+1}",
                        reason="State did not converge or has nonpositive energy.", failure_code="not_converged"))
                    continue
                x = np.asarray(xy[0]).ravel()
                residual = float(np.linalg.norm(vind(x[None, :])[0]-energy*x)/np.linalg.norm(x))
                if residual > 2e-5:
                    report["spin_manifold_failures"].append(dict(manifold=f"{label}{i+1}",
                        reason=f"State residual exceeds tolerance: {residual}", failure_code="not_converged"))
                    continue
                weights, nto = td.get_nto(state=i+1, verbose=0)
                nocc = int(np.count_nonzero(mf.mo_occ > 0))
                weights = np.asarray(weights)
                weights = weights/weights.sum()
                occupied = nto[:, :len(weights)]
                virtual = nto[:, nocc:nocc+len(weights)]
                hole = populations(mol, overlap, occupied, weights)
                electron = populations(mol, overlap, virtual, weights)
                fragments = request.get("fragment_atom_indices", [])
                manifold.append(dict(state=f"{label}{i+1}", excitation_ev=float(energy*27.211386245988),
                    oscillator_strength=None if oscillator is None else float(oscillator),
                    td_residual_norm=residual, nto_weights=weights.tolist(),
                    hole_population_by_atom=hole, electron_population_by_atom=electron,
                    fragment_populations=[dict(atom_indices=group, hole=sum(hole[j] for j in group),
                        electron=sum(electron[j] for j in group)) for group in fragments]))
            report["states"].extend(manifold)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            report["spin_manifold_failures"].append(dict(manifold=label, reason=str(exc), failure_code="not_converged"))
        output.write_text(json.dumps(report, indent=2, allow_nan=False))
    # Preserve completed states if this potentially expensive validation times out.
    stability_started = time.monotonic()
    try:
        _, _, internal_stable, external_stable = mf.stability(internal=True, external=True, return_status=True)
        report.update(scf_internal_stable=bool(internal_stable), scf_external_stable=bool(external_stable),
            stability_check_status="completed")
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        report["validation_failures"].append(dict(reason=f"Reference stability check failed: {exc}",failure_code="numerical_validation_failed"))
        report["stability_check_status"]="failed"
    report["stability_seconds"]=time.monotonic()-stability_started
    report["total_seconds"] = time.monotonic()-started
    report["interpretation"] = f"Computed {len(report['states'])} vertical spin-free states at the supplied geometry; no SOC calculation was performed."
    if report["stability_check_status"] != "completed":
        report["interpretation"] += " Reference stability was not established; retained states are provisional."
    elif not report["scf_internal_stable"] or not report["scf_external_stable"]:
        report["interpretation"] += " The restricted reference is unstable in a tested channel; these energies require caution and are not an established ground-spin assignment."
    output.write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    source, output = map(Path, sys.argv[1:])
    try:
        calculate(json.loads(source.read_text()), output)
    except Exception as exc:
        output.with_suffix(".failure.json").write_text(json.dumps({"error": type(exc).__name__, "reason": str(exc)}))
        raise
