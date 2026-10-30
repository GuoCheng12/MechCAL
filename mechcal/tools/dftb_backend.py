"""CHNO closed-shell TD-DFTB trial; energies/forces retain raw output files."""

import re
import numpy as np
from ase.io import write
from ase.units import Hartree, Bohr
from mechcal.tools.qc_process import QCError
from mechcal.tools.xtb_backend import NUMBER, number


def evaluate(job, atoms, directory, root=1, nstates=6):
    from pathlib import Path
    elements = sorted(set(atoms.get_chemical_symbols()))
    if not set(elements) <= {"C", "H", "N", "O"}:
        raise QCError("unsupported", "Initial TD-DFTB validation is restricted to neutral CHNO singlets.")
    for a in elements:
        for b in elements:
            if not (Path(job.config.parameter_dir) / f"{a}-{b}.skf").is_file():
                raise QCError("missing_input", f"Missing parameter pair {a}-{b}.")
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / "input.gen", atoms, format="gen")
    angular = "\n".join(f'{element} = "{("s" if element == "H" else "p")}"' for element in elements)
    prefix = str(Path(job.config.parameter_dir).resolve()) + "/"
    hsd = f'''Geometry = GenFormat {{
 <<< "input.gen"
}}
Driver = {{}}
Hamiltonian = DFTB {{
 SCC = Yes
 SCCTolerance = 1e-10
 MaxSCCIterations = 250
 Charge = 0
 MaxAngularMomentum = {{ {angular} }}
 SlaterKosterFiles = Type2FileNames {{
  Prefix = "{prefix}"
  Separator = "-"
  Suffix = ".skf"
 }}
}}
ExcitedState = Casida {{
 NrOfExcitations = {nstates}
 StateOfInterest = {root}
 Symmetry = Singlet
 Diagonaliser = Arpack {{}}
}}
Analysis {{ PrintForces = Yes }}
Options {{ WriteResultsTag = Yes }}
ParserOptions {{ ParserVersion = 14 }}
'''
    (directory / "dftb_in.hsd").write_text(hsd)
    try:
        output = job.execute([job.config.dftb_binary], directory)
    except QCError as exc:
        log = directory / "process.log"
        if log.exists() and "SCC is NOT converged" in log.read_text():
            raise QCError("not_converged", "DFTB SCC did not converge; inspect " + str(log)) from exc
        raise
    if "SCC is NOT converged" in output:
        raise QCError("not_converged", "DFTB SCC did not converge.")
    detail = (directory / "detailed.out").read_text()
    ground = number(detail, rf"Total energy:\s*({NUMBER})\s*H")
    states = []
    for line in (directory / "EXC.DAT").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[3] == "->":
            states.append({"state": len(states)+1, "excitation_ev": float(fields[0]),
                           "oscillator_strength": float(fields[1]), "occupied_orbital": int(fields[2]),
                           "virtual_orbital": int(fields[4]), "dominant_weight": float(fields[5])})
    if len(states) < root:
        raise QCError("parse_failed", "Selected excitation absent from EXC.DAT.")
    lines = (directory / "results.tag").read_text().splitlines()
    start = next((i+1 for i,line in enumerate(lines) if line.split(":")[0].strip() == "forces"), None)
    if start is None:
        raise QCError("parse_failed", "Excited-state force table missing.")
    forces = np.array([[float(v) for v in line.split()] for line in lines[start:start+len(atoms)]])
    if forces.shape != (len(atoms),3) or not np.isfinite(forces).all():
        raise QCError("parse_failed", "Invalid excited-state forces.")
    # EXC.DAT rounds excitation energies to 0.001 eV, too coarse for derivatives.
    excitation = number(detail, rf"Excitation Energy:\s*({NUMBER})\s*H")
    energy = (ground + excitation)*Hartree
    if not np.isfinite(energy):
        raise QCError("parse_failed", "Invalid selected-state energy.")
    return {"ground_energy_hartree": ground, "selected_excitation_ev": excitation*Hartree,
            "selected_total_energy_ev": energy,
            "state_root": root, "states": states, "forces_ev_per_angstrom": (forces*Hartree/Bohr).tolist(),
            "method": f"SCC-TD-DFTB/{job.config.parameter_name}, gas-phase neutral singlet",
            "parameter_revision": job.config.parameter_revision,
            "interpretation": f"Evaluated singlet root {root} at the supplied geometry; forces describe the local slope of that root.",
            "limitations": ["Initial CHNO-only parameter scope. Conventional TD-DFTB can misorder charge-transfer states.",
                            "A root number is not a conserved state character; no emission kinetics, CI or nonradiative rate is inferred."]}


def scan(job, atoms, args):
    from ase.calculators.calculator import Calculator, all_changes
    from ase.constraints import FixInternals
    from ase.optimize import BFGS
    if not job.config.allow_excited_scan:
        raise QCError("unsupported", "Excited-state scan is disabled pending validation.")
    if max(args.atom_indices) >= len(atoms):
        raise QCError("missing_input", "Coordinate index outside selected structure.")

    class ExcitedCalculator(Calculator):
        implemented_properties = ["energy", "forces"]
        def __init__(self, pointdir):
            super().__init__()
            self.pointdir, self.calls, self.observations = pointdir, 0, []

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.calls += 1
            data = evaluate(job, atoms, self.pointdir / f"eval_{self.calls:03}", args.state_root, args.nstates)
            self.observations.append(data)
            self.results = {"energy": data["selected_total_energy_ev"],
                            "forces": np.array(data["forces_ev_per_angstrom"])}

    points = []
    for i, value in enumerate(args.values):
        current = atoms.copy()
        indices = args.atom_indices
        if args.coordinate == "bond":
            current.set_distance(*indices, value)
            current.set_constraint(FixInternals(bonds=[[value, indices]]))
        else:
            current.set_dihedral(*indices, value)
            current.set_constraint(FixInternals(dihedrals_deg=[[value, indices]]))
        directory = job.workdir / f"point_{i:02}"
        directory.mkdir()
        calculator = ExcitedCalculator(directory)
        current.calc = calculator
        try:
            converged = BFGS(current, logfile=str(directory / "optimizer.log"), maxstep=0.05).run(
                fmax=args.fmax_ev_per_angstrom, steps=args.max_opt_cycles)
        except QCError as exc:
            if not points:
                raise
            points.append({"coordinate_value": value, "converged": False,
                           "failure_code": exc.code, "failure_message": str(exc),
                           "force_calls": calculator.calls})
            break
        write(directory / "geometry.xyz", current)
        last = calculator.observations[-1]
        signatures = {(o["states"][args.state_root-1]["occupied_orbital"],
                       o["states"][args.state_root-1]["virtual_orbital"]) for o in calculator.observations}
        points.append({"coordinate_value": value, "converged": bool(converged),
                       "energy_ev": last["selected_total_energy_ev"], "states": last["states"],
                       "xyz": str(directory / "geometry.xyz"),
                       "dominant_transition_changed_during_relaxation": len(signatures)>1,
                       "force_calls": calculator.calls})
    return {"points": points, "coordinate": args.coordinate, "atom_indices": args.atom_indices,
            "coordinate_unit": "angstrom" if args.coordinate == "bond" else "degree",
            "state_root": args.state_root, "method": f"SCC-TD-DFTB/{job.config.parameter_name}",
            "interpretation": "Fixed ordinal-root constrained relaxations. Compare only converged points; dominant-transition flags are not full state tracking.",
            "limitations": ["No guaranteed diabatic state tracking; do not infer a reaction barrier across changed state character.",
                            "No dynamics, CI search or kinetic prediction."]}
