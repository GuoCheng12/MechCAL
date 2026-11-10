"""Matched isolated/contact torsion probes, with strain separated from interaction."""

from pathlib import Path
import numpy as np
from ase.io import write
from ase.units import Hartree
from rdkit import Chem
from mechcal.tools.qc_process import QCError
from mechcal.tools.xtb_backend import contact_observables, time_remaining


def moving_side(mol, indices, allowed, coordinate_mode="bonded_dihedral"):
    if coordinate_mode not in {"bonded_dihedral", "fragment_azimuth"}:
        raise ValueError("Unknown contact coordinate mode.")
    if mol is None or len(indices) != 4 or len(set(indices)) != 4 or any(i < 0 or i >= mol.GetNumAtoms() for i in indices):
        raise ValueError("Select four distinct valid atom indices.")
    a, b, c, d = indices
    if not set(indices) <= set(allowed):
        raise ValueError("Select a coordinate entirely within fragment A.")
    if coordinate_mode == "bonded_dihedral" and not all(mol.GetBondBetweenAtoms(i, j) for i, j in zip(indices, indices[1:])):
        raise ValueError("Select a bonded torsion, or explicitly use fragment_azimuth with off-axis anchors.")
    bond = mol.GetBondBetweenAtoms(b, c)
    kinds = {Chem.BondType.SINGLE} if coordinate_mode == "bonded_dihedral" else {Chem.BondType.SINGLE, Chem.BondType.DATIVE}
    if bond is None or bond.IsInRing() or bond.GetBondType() not in kinds:
        raise ValueError("Select an acyclic single central bond, or a dative bond in fragment_azimuth mode.")
    visited, pending = {c}, [c]
    while pending:
        i = pending.pop()
        for neighbor in mol.GetAtomWithIdx(i).GetNeighbors():
            j = neighbor.GetIdx()
            if {i, j} == {b, c} or j in visited:
                continue
            visited.add(j)
            pending.append(j)
    if b in visited or a in visited or d not in visited or not visited <= set(allowed):
        raise ValueError("Cannot isolate the requested torsion side without breaking the fragment.")
    # Anchors must belong to the two sides of this bond, not a disconnected spectator.
    left, pending = {b}, [b]
    while pending:
        i = pending.pop()
        for neighbor in mol.GetAtomWithIdx(i).GetNeighbors():
            j = neighbor.GetIdx()
            if {i, j} == {b, c} or j in left:
                continue
            left.add(j)
            pending.append(j)
    if a not in left or not left <= set(allowed):
        raise ValueError("Reference A must belong to the opposite graph side of the selected axis.")
    return sorted(visited)


def coordinate_preflight(mol, atoms, args, allowed):
    moving = moving_side(mol, args.atom_indices, allowed, args.coordinate_mode)
    a, b, c, d = args.atom_indices
    axis = atoms.positions[c] - atoms.positions[b]
    length = np.linalg.norm(axis)
    if not np.isfinite(atoms.positions).all() or length < 1e-8:
        raise ValueError("Coordinate axis is nonfinite or has zero length.")
    unit = axis / length
    vectors = [atoms.positions[a] - atoms.positions[b], atoms.positions[d] - atoms.positions[c]]
    levers = [float(np.linalg.norm(v - np.dot(v, unit)*unit)) for v in vectors]
    sines = [lever/max(float(np.linalg.norm(v)), 1e-12) for lever, v in zip(levers, vectors)]
    if min(sines) < .05:
        raise ValueError("Near-collinear reference atoms make this angle ill-conditioned. Use fragment_azimuth with explicit off-axis anchors on opposite graph sides.")
    return moving, dict(reference_lever_arms_angstrom=levers, reference_axis_sines=sines,
        minimum_reference_axis_sine=.05, axis_atom_indices=[b, c])


def evaluate(job, atoms, args, source, fragments):
    mol = Chem.MolFromMolFile(source.file_paths["sdf"], removeHs=False)
    moving, conditioning = coordinate_preflight(mol, atoms, args, fragments[0])
    rows = []
    for index, angle in enumerate(args.angles_degrees):
        row = dict(member_id=f"torsion:{index+1:02d}", requested_angle_degrees=angle)
        try:
            if time_remaining(job) <= 0:
                raise QCError("timeout", "Whole-route time exhausted.")
            geometry = atoms.copy()
            geometry.set_dihedral(*args.atom_indices, angle, indices=moving)
            bond_change = max((abs(geometry.get_distance(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                - atoms.get_distance(b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()), default=0.)
            fixed = sorted(set(range(len(atoms))) - set(moving))
            angle_error = abs((geometry.get_dihedral(*args.atom_indices)-angle+180)%360-180)
            if bond_change > 1e-7 or not np.array_equal(geometry.positions[fixed], atoms.positions[fixed]) or angle_error > 1e-6:
                raise QCError("invalid_args", "Rigid-coordinate geometry integrity check failed; no energy calculation performed.")
            distances = geometry.get_all_distances()
            np.fill_diagonal(distances, np.inf)
            if distances.min() < .65:
                raise QCError("invalid_args", "Imposed torsion creates an atomic overlap below 0.65 angstrom.")
            directory = job.workdir/f"torsion_{index+1:02d}"
            directory.mkdir()
            xyz = directory/"geometry.xyz"
            write(xyz, geometry)
            row.update(contact_observables(job, geometry, len(fragments[0]), directory/"energies"))
            row.update(status="success", xyz=str(xyz), geometry_converged=None,
                geometry_optimization_status="not_requested", electronic_status="converged",
                maximum_bond_length_change_angstrom=float(bond_change),
                actual_angle_degrees=float(geometry.get_dihedral(*args.atom_indices)))
        except (QCError, ValueError, ZeroDivisionError) as exc:
            row.update(status="failed", failure_code=getattr(exc, "code", "invalid_args"), reason=str(exc))
        rows.append(row)
    good = [p for p in rows if p["status"] == "success"]
    if good:
        reference = good[0]
        for row in good:
            differences = (np.asarray(row["fragment_energies_hartree"])-reference["fragment_energies_hartree"])*Hartree
            row.update(relative_complex_energy_ev=float(differences[0]),
                relative_isolated_a_energy_ev=float(differences[1]), relative_isolated_b_energy_ev=float(differences[2]),
                contact_contribution_change_ev=row["interaction_energy_ev"]-reference["interaction_energy_ev"],
                reference_member_id=reference["member_id"])
    return dict(method="GFN2-xTB matched fixed-coordinate torsion response, gas phase", points=rows,
        atom_indices=args.atom_indices, coordinate_mode=args.coordinate_mode,
        coordinate_conditioning=conditioning, moved_atom_indices=moving,
        comparison_definition="Relative complex energy = isolated A strain + isolated B strain + change in contact interaction. Reference is the first successful point.",
        interpretation=f"{len(good)}/{len(rows)} matched torsions computed; isolated-fragment and contact contributions are reported separately.",
        limitations=["A modeled two-molecule arrangement, not measured or predicted crystal packing.",
            "Imposed torsion with no relaxation, not an activation barrier or nonradiative rate.",
            "Repeat for other starting contact poses and far-separation controls before generalizing."])
