"""ASE-constrained optimization with real Amesp energy/force evaluations."""

from copy import copy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixInternals
from ase.optimize import BFGS
from ase.units import Hartree, Bohr
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from mechcal.schemas.tool_arguments import RelaxedScanArgs
from mechcal.tools.amesp_baseline import (
    AmespBaselineError,
    AmespBaselineRunner,
    AmespStepResult,
    _parse_final_energy,
    _write_xyz,
)
from mechcal.tools.amesp_relaxation import endpoint_states


def parse_forces(text: str, symbols: list[str]) -> tuple[float, np.ndarray]:
    marker = "Cartesian Force (Hartrees/Bohr):"
    if marker not in text:
        raise AmespBaselineError(
            "parse_failed", "Missing Cartesian force table (not a gradient table)."
        )
    rows = []
    elements = []
    for line in text.rsplit(marker, 1)[1].splitlines():
        parts = line.split()
        if len(parts) == 4 and re.fullmatch(r"[A-Z][a-z]?", parts[0]):
            try:
                rows.append([float(x.replace("D", "E")) for x in parts[1:]])
            except ValueError as exc:
                raise AmespBaselineError("parse_failed", "Malformed force row.") from exc
            elements.append(parts[0])
        elif rows:
            break
    energy = _parse_final_energy(text)
    force = np.asarray(rows)
    if elements != symbols or force.shape != (len(symbols), 3) or not np.isfinite(force).all():
        raise AmespBaselineError(
            "parse_failed", "Force table atom order/count or numeric values are invalid."
        )
    if energy is None or not np.isfinite(energy):
        raise AmespBaselineError("parse_failed", "Missing finite final energy.")
    return energy * Hartree, force * Hartree / Bohr


class AmespForceCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, runner, directory, charge, multiplicity, args, deadline):
        super().__init__()
        self.runner = runner
        self.workdir = directory
        self.charge = charge
        self.multiplicity = multiplicity
        self.args = args
        self.deadline = deadline
        self.records: list[AmespStepResult] = []
        self.last_text = ""

    def execute(self, name, atoms, *, vertical=False):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise AmespBaselineError("wall_limit", "Geometry optimization exhausted its wall-time limit.")
        runner = copy(self.runner)
        runner.timeout_seconds = min(runner.timeout_seconds, remaining)
        # No retry configuration change inside a constrained optimizer evaluation.
        runner.npara = 1
        runner.maxcore_mb = min(runner.maxcore_mb, 1000)
        keywords = ["aTB1", *([] if vertical else ["force"])]
        blocks = [("scf", ["maxcyc 500", "vshift 500", "conver 10"])]
        if (self.args is not None and self.args.state_root) or vertical:
            if self.args is None:
                raise ValueError("Vertical calculations require explicit excited-state settings.")
            keywords.append("tda")
            blocks += [
                ("atb", ["excdip on"]),
                (
                    "posthf",
                    [
                        f"nstates {self.args.s1_nstates}",
                        f"root {max(1, self.args.state_root)}",
                        "tdspin s",
                        f"tout {self.args.td_tout}",
                        "conver 10",
                    ],
                ),
            ]
        step = runner._run_step(
            step_id=name,
            label=name,
            workdir=self.workdir,
            keywords=keywords,
            block_lines=blocks,
            charge=self.charge,
            multiplicity=self.multiplicity,
            symbols=atoms.get_chemical_symbols(),
            coordinates=atoms.positions.tolist(),
        )
        self.records.append(step)
        self.last_text = step.aop_path.read_text(errors="replace")
        return self.last_text

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        text = self.execute(f"force_{len(self.records):04d}", atoms)
        energy, forces = parse_forces(text, atoms.get_chemical_symbols())
        self.results = {"energy": energy, "forces": forces}


@dataclass
class ScanOutput:
    observations: dict = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)
    steps: list[AmespStepResult] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def validate_coordinate(mol, args):
    ids = args.atom_indices
    if any(i >= mol.GetNumAtoms() for i in ids):
        raise ValueError("Coordinate atom index is outside the source molecule.")
    if any(mol.GetBondBetweenAtoms(i, j) is None for i, j in zip(ids, ids[1:])):
        raise ValueError("Selected coordinate atoms must form a bonded path.")
    if args.coordinate == "dihedral" and mol.GetBondBetweenAtoms(ids[1], ids[2]).IsInRing():
        raise ValueError("The initial dihedral implementation does not rotate ring bonds.")


def actual_coordinate(atoms, args):
    if args.coordinate == "bond":
        return float(atoms.get_distance(*args.atom_indices))
    return float((atoms.get_dihedral(*args.atom_indices) + 180) % 360 - 180)


def coordinate_error(actual, target, coordinate):
    return (
        abs(actual - target) if coordinate == "bond" else abs((actual - target + 180) % 360 - 180)
    )


def run_scan(
    runner: AmespBaselineRunner,
    *,
    mol,
    coordinates,
    charge,
    multiplicity,
    args: RelaxedScanArgs,
    workdir: Path,
) -> ScanOutput:
    validate_coordinate(mol, args)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    xyz = np.asarray(coordinates, dtype=float)
    if xyz.shape != (len(symbols), 3) or not np.isfinite(xyz).all():
        raise ValueError("Source coordinates must be finite and match the indexed topology.")
    if not runner.amesp_bin.is_file():
        raise AmespBaselineError(
            "amesp_binary_missing", f"Amesp binary not found: {runner.amesp_bin}"
        )
    output = ScanOutput(
        observations={
            "coordinate": args.coordinate,
            "atom_indices": args.atom_indices,
            "atom_index_base": 0,
            "coordinate_unit": "angstrom" if args.coordinate == "bond" else "degree",
            "method": "aTB1" if args.state_root == 0 else "aTB1/TDA",
            "state_root": args.state_root,
            "environment": "gas_phase",
            "symbols": symbols,
            "state_tracking": "fixed_ordinal_root_not_character_tracked",
            "optimizer": "ASE BFGS + FixInternals",
            "backend_npara": 1,
            "backend_maxcore_mb": min(runner.maxcore_mb, 1000),
            "scf_conver": 10,
            "minimum_verified_by_frequencies": False,
            "route_records": [],
            "available_descriptors": [],
            "missing_descriptors": [],
        }
    )
    with runner.amesp_bin.open("rb") as f:
        output.observations["backend_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
    deadline = time.monotonic() + args.max_wall_seconds
    for index, value in enumerate(args.values, 1):
        member_id = f"scan:{index:02d}"
        directory = workdir / f"point_{index:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        row = dict(
            member_id=member_id, target_value=value, status="failed", geometry_converged=False
        )
        atoms = Atoms(symbols, positions=coordinates)
        calculator = AmespForceCalculator(runner, directory, charge, multiplicity, args, deadline)
        try:
            if args.coordinate == "bond":
                atoms.set_distance(*args.atom_indices, value, fix=0.5)
                constraint = FixInternals(bonds=[[value, args.atom_indices]])
            else:
                snapshot = Chem.Mol(mol)
                conf = snapshot.GetConformer()
                for i, xyz in enumerate(coordinates):
                    conf.SetAtomPosition(i, xyz)
                rdMolTransforms.SetDihedralDeg(conf, *args.atom_indices, value)
                atoms.positions[:] = conf.GetPositions()
                constraint = FixInternals(dihedrals_deg=[[value, args.atom_indices]])
            _write_xyz(
                directory / "initial.xyz",
                label=member_id,
                symbols=symbols,
                coordinates=atoms.positions.tolist(),
            )
            row["initial_xyz"] = str(directory / "initial.xyz")
            atoms.set_constraint(constraint)
            atoms.calc = calculator
            optimizer = BFGS(atoms, logfile=str(directory / "optimizer.log"), maxstep=0.1)
            converged = optimizer.run(fmax=args.fmax_ev_per_angstrom, steps=args.max_opt_cycles)
            actual = actual_coordinate(atoms, args)
            error = coordinate_error(actual, value, args.coordinate)
            projected_force = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
            row.update(
                actual_value=actual,
                coordinate_error=error,
                optimizer_steps=optimizer.nsteps,
                projected_fmax_ev_per_angstrom=projected_force,
                final_energy_hartree=float(atoms.get_potential_energy() / Hartree),
            )
            tolerance = 0.001 if args.coordinate == "bond" else 0.05
            if not converged or projected_force > args.fmax_ev_per_angstrom:
                raise AmespBaselineError(
                    "optimization_not_converged", "Projected force tolerance not reached."
                )
            if error > tolerance:
                raise AmespBaselineError(
                    "constraint_failed",
                    "Optimized geometry does not satisfy the requested coordinate.",
                )
            path = directory / "converged.xyz"
            _write_xyz(path, label=member_id, symbols=symbols, coordinates=atoms.positions.tolist())
            row.update(geometry_converged=True, xyz_path=str(path), status="partial")
            output.paths[f"{member_id}_xyz"] = str(path)
            output.steps.append(calculator.records[-1])
            final_text = calculator.execute("endpoint_vertical", atoms, vertical=True)
            row["excited_states"] = endpoint_states(final_text, args.s1_nstates)
            output.steps.append(calculator.records[-1])
            row["status"] = "success"
        except (AmespBaselineError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            row["failure"] = {"code": getattr(exc, "code", type(exc).__name__), "message": str(exc)}
            if isinstance(exc, AmespBaselineError):
                row["failure"]["files"] = exc.file_paths
            output.missing.append(f"{member_id}:complete_relaxed_point")
        finally:
            evaluation_path = directory / "evaluations.json"
            evaluation_path.write_text(
                json.dumps([s.to_json() for s in calculator.records], indent=2)
            )
            row["evaluation_manifest"] = str(evaluation_path)
            row["evaluation_count"] = len(calculator.records)
            output.observations["route_records"].append(row)
            (workdir / "progress.json").write_text(json.dumps(output.observations, indent=2))
    successful = sum(r["status"] == "success" for r in output.observations["route_records"])
    output.observations["successful_member_count"] = successful
    output.observations["boundary_only"] = not any(
        r["geometry_converged"] for r in output.observations["route_records"]
    )
    output.observations["available_descriptors"] = (
        ["relaxed_coordinate_points"] if successful else []
    )
    output.observations["missing_descriptors"] = output.missing
    output.paths["scan_progress"] = str(workdir / "progress.json")
    (workdir / "progress.json").write_text(json.dumps(output.observations, indent=2))
    return output
