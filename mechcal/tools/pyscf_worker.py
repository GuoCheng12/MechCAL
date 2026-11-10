"""Standalone isolated-interpreter worker for the explicitly bounded CIS trial."""

import json
import math
from pathlib import Path
import sys
import time
import traceback

VERSION = "2.14.0"
MAX_ATOMS = 40
ELEMENTS = frozenset({"C", "H", "N", "O"})
METHOD = "RHF/6-31G/CIS, conventional integrals, three singlets, energy-ordered S1"
SCHEMA = "pyscf_cis_s1_v2"


def validate_request(request):
    if set(request) != {"symbols", "coordinates_angstrom", "cpu_threads"}:
        raise ValueError("Unexpected worker request fields.")
    symbols, xyz = request["symbols"], request["coordinates_angstrom"]
    if not isinstance(symbols, list) or not 2 <= len(symbols) <= MAX_ATOMS:
        raise ValueError("CIS trial requires 2 to 40 atoms.")
    if any(not isinstance(s, str) or s not in ELEMENTS for s in symbols):
        raise ValueError("CIS trial supports CHNO only.")
    if not isinstance(xyz, list) or len(xyz) != len(symbols):
        raise ValueError("Coordinate and atom counts disagree.")
    if any(not isinstance(row, list) or len(row) != 3 or
           any(type(v) not in (int, float) or not math.isfinite(v) for v in row) for row in xyz):
        raise ValueError("Expected finite Cartesian coordinates in angstrom.")
    if type(request["cpu_threads"]) is not int or not 1 <= request["cpu_threads"] <= 8:
        raise ValueError("Expected 1 to 8 CPU threads.")


def compute(request, log_path):
    import numpy as np
    import pyscf
    from pyscf import gto, lib, scf

    validate_request(request)
    if pyscf.__version__ != VERSION:
        raise ValueError(f"This validated worker requires PySCF {VERSION}.")
    lib.num_threads(request["cpu_threads"])
    mol = gto.M(atom=list(zip(request["symbols"], request["coordinates_angstrom"])),
        basis="6-31g", charge=0, spin=0, unit="Angstrom", symmetry=False,
        max_memory=1000, verbose=4, output=str(log_path))
    mf = scf.RHF(mol)
    mf.conv_tol, mf.max_cycle = 1e-10, 150
    started = time.monotonic()
    mf.kernel()
    scf_seconds = time.monotonic() - started
    if not mf.converged:
        raise RuntimeError("SCF did not converge.")
    started = time.monotonic()
    ground_gradient = mf.nuc_grad_method().kernel()
    ground_gradient_seconds = time.monotonic() - started
    if ground_gradient.shape != (len(request["symbols"]), 3) or not np.isfinite(ground_gradient).all():
        raise RuntimeError("Invalid S0 gradient.")
    td = mf.TDA()
    td.nstates, td.max_cycle, td.conv_tol, td.lindep = 3, 150, 1e-5, 1e-12
    started = time.monotonic()
    td.kernel()
    td_seconds = time.monotonic() - started
    if len(td.e) != 3 or not np.asarray(td.converged).all():
        raise RuntimeError("Three converged singlet roots are required.")
    vind, _ = td.gen_vind(mf)
    vectors = np.asarray([x.ravel() * np.sqrt(2) for x, y in td.xy])
    residuals = np.linalg.norm(vind(vectors) - np.asarray(td.e)[:, None] * vectors, axis=1)
    if (not np.isfinite(td.e).all() or min(td.e) <= 0 or
            not np.isfinite(residuals).all() or max(residuals) > 1.01e-5):
        raise RuntimeError("Independent excitation residual/energy check failed.")
    transitions = []
    for x, y in td.xy:
        weights = np.asarray(x) ** 2
        i = np.unravel_index(weights.argmax(), weights.shape)
        transitions.append(dict(occupied=int(i[0]), virtual=int(i[1]),
            largest_normalized_weight=float(weights[i] / weights.sum())))
    started = time.monotonic()
    grad = td.nuc_grad_method()
    grad.cphf_max_cycle, grad.cphf_conv_tol = 50, 1e-9
    gradient = grad.kernel(state=1)
    gradient_seconds = time.monotonic() - started
    if gradient.shape != (len(request["symbols"]), 3) or not np.isfinite(gradient).all():
        raise RuntimeError("Invalid S1 gradient.")
    oscillator = np.asarray(td.oscillator_strength())
    if oscillator.shape != (3,) or not np.isfinite(oscillator).all() or min(oscillator) < 0:
        raise RuntimeError("Invalid oscillator strengths.")
    return dict(schema_version=SCHEMA, status="success", version=pyscf.__version__,
        units={"energy": "hartree", "gradient": "hartree/bohr", "coordinates": "angstrom"},
        method=METHOD, request=request, scf_converged=True, td_converged=np.asarray(td.converged).tolist(),
        ground_energy_hartree=float(mf.e_tot), excitations_hartree=np.asarray(td.e).tolist(),
        s1_total_energy_hartree=float(mf.e_tot + td.e[0]),
        gradient_hartree_per_bohr=gradient.tolist(), oscillator_strengths=oscillator.tolist(),
        ground_gradient_hartree_per_bohr=ground_gradient.tolist(),
        ground_gradient_seconds=ground_gradient_seconds,
        dominant_transitions=transitions, td_residual_norms=residuals.tolist(),
        scf_seconds=scf_seconds, td_seconds=td_seconds, gradient_seconds=gradient_seconds,
        memory_setting_mb=1000, nao=mol.nao_nr())


def main():
    source, target = map(Path, sys.argv[1:])
    try:
        result = compute(json.loads(source.read_text()), target.with_suffix(".log"))
    except Exception as exc:
        result = dict(schema_version=SCHEMA, status="failed", error_type=type(exc).__name__, error=str(exc))
        target.with_suffix(".failure.txt").write_text(traceback.format_exc())
    target.write_text(json.dumps(result, indent=2, allow_nan=False))
    if result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
