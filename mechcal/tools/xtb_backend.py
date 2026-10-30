"""Small xTB routes with explicit observation boundaries."""

from pathlib import Path
import re
import time
import numpy as np
from ase.io import read, write
from ase.units import Hartree
from mechcal.tools.qc_process import QCError

NUMBER = r"[-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?"


def number(text, pattern):
    match = re.search(pattern, text)
    if not match:
        raise QCError("parse_failed", f"Required output is absent: {pattern}")
    value = float(match.group(1).replace("D", "E"))
    if not np.isfinite(value):
        raise QCError("parse_failed", "Non-finite computed value.")
    return value


def single(job, atoms, directory, options):
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / "input.xyz", atoms)
    text = job.execute([job.config.xtb_binary, "input.xyz", "--chrg", "0", "--uhf", "0", *options], directory)
    if "normal termination of xtb" not in text or "SCC not converged" in text:
        raise QCError("not_converged", f"xTB did not finish normally: {directory}")
    return text


def optimized(job, atoms, directory=None, max_cycles=150):
    directory = directory or job.workdir / "optimize"
    text = single(job, atoms, directory, ["--gfn", "2", "--opt", "tight", "--cycles", str(max_cycles)])
    if "GEOMETRY OPTIMIZATION CONVERGED" not in text:
        raise QCError("not_converged", "GFN2 geometry optimization did not converge.")
    result = read(directory / "xtbopt.xyz")
    if result.get_chemical_symbols() != atoms.get_chemical_symbols():
        raise QCError("parse_failed", "Atom ordering changed during optimization.")
    return result


def ip_ea(job, atoms):
    geometry = optimized(job, atoms)
    text = single(job, geometry, job.workdir / "ipea", ["--vipea"])
    ip = number(text, rf"delta SCC IP \(eV\):\s*({NUMBER})")
    ea = number(text, rf"delta SCC EA \(eV\):\s*({NUMBER})")
    return {"vertical_ip_ev": ip, "vertical_ea_ev": ea,
            "computed_geometry_xyz": str(job.workdir / "optimize" / "xtbopt.xyz"),
            "method": "IPEA-xTB on GFN2-xTB optimized neutral gas-phase geometry",
            "definitions": "IP=E(N-1)-E(N); EA=E(N)-E(N+1), including xTB empirical corrections",
            "interpretation": f"Electron removal costs {ip:.3f} eV; electron attachment has EA {ea:.3f} eV under the stated sign convention.",
            "limitations": ["Whole-molecule vertical energies, not donor/acceptor fragment energies, solution redox potentials, PET free energy or a transfer rate."]}


def parse_modes(path: Path, natoms: int):
    lines = path.read_text().splitlines()
    modes = []
    for i, line in enumerate(lines):
        if "Frequencies --" not in line:
            continue
        frequencies = [float(v) for v in line.split("--", 1)[1].split()]
        start = next((j + 1 for j in range(i + 1, min(i + 15, len(lines)))
                      if "Atom" in lines[j] and "AN" in lines[j]), None)
        if start is None:
            raise QCError("parse_failed", "Missing normal-mode displacement table.")
        displacement = []
        for line in lines[start:start + natoms]:
            fields = line.split()
            if len(fields) != 2 + 3 * len(frequencies):
                raise QCError("parse_failed", "Incomplete normal-mode atom table.")
            displacement.append([float(v) for v in fields[2:]])
        for k, frequency in enumerate(frequencies):
            vectors = np.array(displacement)[:, 3*k:3*k+3]
            top = np.argsort(np.linalg.norm(vectors, axis=1))[-4:][::-1]
            modes.append({"mode_index": len(modes) + 1, "frequency_cm_minus1": frequency,
                          "displacements": vectors.tolist(), "largest_motion_atom_indices": top.tolist()})
    if not modes or not all(np.isfinite(m["frequency_cm_minus1"]) and
                             np.isfinite(m["displacements"]).all() for m in modes):
        raise QCError("parse_failed", "No finite vibrational modes parsed.")
    return modes


def vibrations(job, atoms):
    geometry = optimized(job, atoms)
    directory = job.workdir / "hessian"
    single(job, geometry, directory, ["--gfn", "2", "--hess"])
    modes = parse_modes(directory / "g98.out", len(atoms))
    imaginary = [m["mode_index"] for m in modes if m["frequency_cm_minus1"] < -20]
    low = [m for m in modes if 20 < m["frequency_cm_minus1"] < 500]
    return {"method": "GFN2-xTB gas-phase numerical Hessian after tight optimization",
            "computed_geometry_xyz": str(job.workdir / "optimize" / "xtbopt.xyz"),
            "normal_modes": low[:6], "mode_vector_selection": "Up to six lowest modes in 20-500 cm^-1; full vectors are in raw g98.out.",
            "all_frequencies_cm_minus1": [m["frequency_cm_minus1"] for m in modes],
            "raw_normal_modes_file": str(directory / "g98.out"),
            "low_frequency_mode_indices": [m["mode_index"] for m in low],
            "imaginary_mode_indices_below_minus20": imaginary,
            "geometry_stationarity": "imaginary_modes_present" if imaginary else "no_significant_imaginary_modes",
            "interpretation": f"{len(low)} modes in 20-500 cm^-1 identify concrete low-frequency atomic motions; {len(imaginary)} modes below -20 cm^-1.",
            "limitations": ["Ground-state modes are not evidence of nonradiative rates or restriction-induced fluorescence.",
                            "Near-zero translations/rotations are not interpreted as internal modes."]}


def contact_observables(job, combined, split, directory):
    """Unrestrained single-point energies of the complex and its frozen fragments."""
    energies = []
    for name, geometry in [("complex", combined), ("a", combined[:split]), ("b", combined[split:])]:
        output = single(job, geometry, directory / name, ["--gfn", "2"])
        energies.append(number(output, rf"TOTAL ENERGY\s+({NUMBER})\s+Eh"))
    distances = np.linalg.norm(combined.positions[:split, None] - combined.positions[None, split:], axis=2)
    contacts = np.argwhere(distances < 3.5)
    symbols = combined.get_chemical_symbols()
    return {"interaction_energy_ev": (energies[0]-energies[1]-energies[2])*Hartree,
            "energy_definition": "E(complex)-E(fragment A at complex geometry)-E(fragment B at complex geometry)",
            "fragment_energies_hartree": energies, "minimum_interfragment_distance_angstrom": float(distances.min()),
            "energy_component_order": ["complex", "fragment_a", "fragment_b"],
            "atom_index_convention": "Zero-based indices in the combined geometry; fragment-local indices are also zero-based.",
            "contacts_under_3_5_angstrom": [{"atom_a": int(a), "atom_b": int(b+split),
                "element_a": symbols[a], "element_b": symbols[b+split],
                "fragment_a": 0, "fragment_b": 1, "fragment_atom_a": int(a), "fragment_atom_b": int(b),
                "distance_angstrom": float(distances[a,b])} for a,b in contacts],
            "fragment_atom_indices": [list(range(split)), list(range(split, len(combined)))]}


def dimer(job, atoms, partner, max_candidates=1, max_opt_cycles=150):
    candidates = []
    attempts = []
    seen_distances = []
    for index in range(max_candidates):
        member = job.workdir / f"candidate_{index+1:02d}"
        attempt = {"search_index": index+1, "partner_rotation_degrees": index*137.5}
        if time_remaining(job) <= 0:
            attempts.append({**attempt, "status": "not_attempted", "reason": "Whole-route deadline exhausted."})
            continue
        directory = member / "dock"
        directory.mkdir(parents=True)
        rotated = partner.copy()
        rotated.rotate(index*137.5, [1, 2, 3], center="COM")
        write(directory / "a.xyz", atoms)
        write(directory / "b.xyz", rotated)
        try:
            # Native aISS final optimization segfaults in this build. Standalone GFN2
            # optimization is the established route; no failed native result is reused.
            text = job.execute([job.config.xtb_binary, "dock", "a.xyz", "b.xyz", "--fast", "--nfinal", "0",
                "--maxgen", "2", "--maxparent", "28", "--chrg1", "0", "--chrg2", "0",
                "--uhf1", "0", "--uhf2", "0"], directory)
            if "normal termination of xtb" not in text:
                raise QCError("not_converged", "aISS did not finish normally.")
            path = directory / "best_after_gen.xyz"
            if not path.is_file() or not path.stat().st_size:
                raise QCError("parse_failed", "aISS produced no post-genetic candidate.")
            combined = read(path)
            if combined.get_chemical_symbols() != atoms.get_chemical_symbols() + partner.get_chemical_symbols():
                raise QCError("parse_failed", "Dimer atom order differs from supplied fragments.")
            combined = optimized(job, combined, member / "optimize", max_opt_cycles)
            distances = combined.get_all_distances()
            if any(np.max(np.abs(distances-old)) < .05 for old in seen_distances):
                attempts.append({**attempt, "status": "duplicate", "reason": "Same indexed interatomic distances within 0.05 angstrom."})
                continue
            observation = contact_observables(job, combined, len(atoms), member / "energies")
            seen_distances.append(distances)
            candidates.append({"member_id": f"dimer:{index+1:02d}", "status": "success",
                "xyz": str(member / "optimize" / "xtbopt.xyz"), "geometry_converged": True, **observation})
            attempts.append({**attempt, "status": "success"})
        except QCError as exc:
            attempts.append({**attempt, "status": "failed", "failure_code": exc.code, "reason": str(exc)})
    return {"method": "aISS candidate search, GFN2-xTB same-geometry interaction energies, gas phase",
            "requested_candidates": max_candidates, "search_attempts": attempts, "candidates": candidates,
            "interpretation": f"Returned {len(candidates)} modeled dimer arrangement(s) with explicit contacts and interaction energies.",
            "limitations": ["Candidates are not observed crystal packing, a binding free energy or an aggregate excited state.",
                            "Fast stochastic search is not exhaustive; no reproducibility claim for pose ordering."]}


def scan_contacts(job, atoms, args, fragments):
    a, b = args.atom_indices
    moving = fragments[1]
    direction = atoms.positions[b] - atoms.positions[a]
    initial_distance = float(np.linalg.norm(direction))
    if initial_distance < 1e-6:
        raise QCError("invalid_args", "Selected atoms coincide.")
    direction /= initial_distance
    rows = []
    for index, distance in enumerate(args.distances_angstrom):
        row = {"member_id": f"contact:{index+1:02d}", "requested_distance_angstrom": distance}
        if time_remaining(job) <= 0:
            rows.append({**row, "status": "not_attempted", "reason": "Whole-route deadline exhausted."})
            continue
        geometry = atoms.copy()
        geometry.positions[moving] += (distance - initial_distance) * direction
        directory = job.workdir / f"point_{index+1:02d}"
        directory.mkdir()
        xyz = directory / "geometry.xyz"
        write(xyz, geometry)
        cross = np.linalg.norm(geometry.positions[fragments[0], None] - geometry.positions[None, moving], axis=2)
        if cross.min() < .7:
            rows.append({**row, "status": "failed", "reason": "Rigid placement creates an atomic overlap below 0.7 angstrom."})
            continue
        try:
            rows.append({**row, "status": "success", "xyz": str(xyz), "geometry_converged": False,
                "actual_distance_angstrom": float(np.linalg.norm(geometry.positions[b]-geometry.positions[a])),
                **contact_observables(job, geometry, len(fragments[0]), directory / "energies")})
        except QCError as exc:
            rows.append({**row, "status": "failed", "failure_code": exc.code, "reason": str(exc)})
    success = sum(r["status"] == "success" for r in rows)
    return {"method": "GFN2-xTB rigid-fragment distance scan, gas phase",
        "atom_indices": args.atom_indices, "initial_distance_angstrom": initial_distance,
        "points": rows, "interpretation": f"{success}/{len(rows)} rigid contact snapshots yielded same-geometry interaction energies.",
        "limitations": ["Internal fragment coordinates and relative orientation are fixed, not relaxed.",
            "Distance is an imposed test condition, not an observed contact or equilibrium distance.",
            "No excited-state, binding free-energy, solid-state or mechanism assignment is computed."]}


def time_remaining(job):
    return job.deadline - time.monotonic()
