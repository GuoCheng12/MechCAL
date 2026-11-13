"""Fixed-geometry TD-DFTB spin manifolds, not a spin-orbit or kinetic model."""

import hashlib
import re
from pathlib import Path

import numpy as np
from ase.io import write

from mechcal.tools.qc_process import QCConfig, QCError, QCJob


def spin_parameters(config: QCConfig, symbols: list[str]) -> tuple[dict, dict]:
    elements = sorted(set(symbols))
    if not elements or not set(elements) <= {"C", "H", "N", "O"}:
        raise ValueError("Spin comparison requires CHNO parameters.")
    if config.parameter_name != "mio-1-1" or not config.parameter_revision:
        raise ValueError("Spin comparison requires identified mio-1-1 parameters and their revision.")
    root = Path(config.parameter_dir).resolve()
    path = root.parent / "extras" / "spinw.txt"
    try:
        raw = path.read_bytes()
        sections = re.split(r"(?m)^([A-Z][a-z]?):\s*$", raw.decode())
        if sections[0].strip():
            raise ValueError("Unexpected spin parameter header.")
        matrices = {}
        for i in range(1, len(sections), 2):
            element = sections[i]
            if element in matrices:
                raise ValueError("Duplicate spin parameter element.")
            matrices[element] = [float(v) for v in sections[i + 1].split()]
        selected = {}
        for element in elements:
            values = matrices.get(element, [])
            size = 1 if element == "H" else 2
            if len(values) != size * size or not np.isfinite(values).all():
                raise ValueError(f"Incomplete or nonfinite spin matrix for {element}.")
            matrix = np.array(values).reshape(size, size)
            if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0):
                raise ValueError(f"Nonsymmetric spin matrix for {element}.")
            selected[element] = values
        hashes = {}
        for a in elements:
            for b in elements:
                pair = root / f"{a}-{b}.skf"
                data = pair.read_bytes()
                if not data.strip():
                    raise ValueError(f"Empty parameter pair {a}-{b}.")
                hashes[pair.name] = hashlib.sha256(data).hexdigest()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read spin/Slater-Koster parameters: {exc}") from exc
    return selected, {"name": config.parameter_name, "revision": config.parameter_revision,
        "spin_constants_file": str(path), "spin_constants_sha256": hashlib.sha256(raw).hexdigest(),
        "slater_koster_sha256": hashes, "shell_resolved_spin": True}


# DFTB+'s Casida/Arpack solver can stop at its iteration limit when few roots are requested; its own
# remedy is to solve for more roots. The lowest requested roots are kept, so results are unchanged.
MORE_ROOTS_MARK = "Increase the number of excited states to solve for"
RETRY_ROOTS = 12


def needs_more_roots(directory: Path) -> bool:
    log = directory / "process.log"
    return log.exists() and MORE_ROOTS_MARK in log.read_text(errors="replace")


def parse_states(text: str, nstates: int, solved: int | None = None) -> list[dict]:
    """Singlet and triplet roots from EXC.DAT; ``solved`` roots per manifold were computed, the lowest ``nstates`` kept."""
    states = []
    try:
        if not any('w [eV]' in line and 'Osc.Str.' in line and 'Sym.' in line for line in text.splitlines()):
            raise ValueError("Missing eV/spin header in EXC.DAT.")
        for line in text.splitlines():
            fields = line.split()
            if "->" not in fields:
                continue
            if len(fields) != 8 or fields[3] != "->" or fields[-1] not in {"S", "T"}:
                raise ValueError("Malformed spin-resolved excitation row.")
            energy, oscillator, weight, ks_energy = map(float, [fields[0], fields[1], fields[5], fields[6]])
            occupied, virtual = int(fields[2]), int(fields[4])
            if (not np.isfinite([energy, oscillator, weight, ks_energy]).all()
                    or energy <= 0 or oscillator < 0 or not 0 <= weight <= 1
                    or occupied < 1 or virtual <= occupied or ks_energy <= 0):
                raise ValueError("Invalid excitation or transition value.")
            states.append(dict(spin=fields[-1], excitation_ev=energy, oscillator_strength=oscillator,
                occupied_orbital=occupied, virtual_orbital=virtual, dominant_weight=weight,
                ks_transition_ev=ks_energy))
        result = []
        for spin in ("S", "T"):
            manifold = sorted([s for s in states if s["spin"] == spin], key=lambda s: s["excitation_ev"])
            if len(manifold) != (solved or nstates):
                raise ValueError(f"Expected {solved or nstates} {spin} states, found {len(manifold)}.")
            result.extend(dict(state=f"{spin}{i}", **state) for i, state in enumerate(manifold[:nstates], 1))
        return result
    except ValueError as exc:
        raise QCError("parse_failed", str(exc)) from exc


def compare_states(job: QCJob, atoms, nstates: int) -> dict:
    constants, provenance = spin_parameters(job.config, atoms.get_chemical_symbols())
    elements = sorted(constants)
    directory = job.workdir / "spin_comparison"
    directory.mkdir()
    write(directory / "input.gen", atoms, format="gen")
    angular = "\n".join(f'{e} = "{("s" if e == "H" else "p")}"' for e in elements)
    spin = "\n".join(e + " = { " + " ".join(map(str, constants[e])) + " }" for e in elements)
    prefix = str(Path(job.config.parameter_dir).resolve()) + "/"
    if any(c in prefix for c in ('"', '\n', '\r')):
        raise QCError("invalid_args", "Parameter path contains unsupported HSD delimiters.")
    def write_input(roots: int) -> None:
        (directory / "dftb_in.hsd").write_text(f'''Geometry = GenFormat {{
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
 SpinConstants = {{ ShellResolvedSpin = Yes
 {spin}
 }}
}}
ExcitedState = Casida {{
 NrOfExcitations = {roots}
 Symmetry = Both
 Diagonaliser = Arpack {{}}
}}
Options {{ WriteResultsTag = Yes }}
ParserOptions {{ ParserVersion = 14 }}
''')
    for roots in (nstates, max(RETRY_ROOTS, nstates + 6)):
        write_input(roots)
        try:
            output = job.execute([job.config.dftb_binary], directory)
            break
        except QCError as exc:
            log = directory / "process.log"
            if log.exists() and "SCC is NOT converged" in log.read_text():
                raise QCError("not_converged", "DFTB SCC did not converge.") from exc
            if roots == nstates and needs_more_roots(directory):
                continue                              # the solver's own remedy: solve for more roots
            raise
    if "SCC is NOT converged" in output:
        raise QCError("not_converged", "DFTB SCC did not converge.")
    try:
        states = parse_states((directory / "EXC.DAT").read_text(), nstates, solved=roots)
    except OSError as exc:
        raise QCError("parse_failed", "Missing spin-resolved excitation table.") from exc
    singlets, triplets = states[:nstates], states[nstates:]
    gaps = [{"singlet": s["state"], "triplet": t["state"],
             "singlet_minus_triplet_ev": s["excitation_ev"] - t["excitation_ev"]}
            for s in singlets for t in triplets]
    gap = gaps[0]["singlet_minus_triplet_ev"]
    return {"states": states, "spin_gaps": gaps, "s1_minus_t1_ev": gap, "roots_solved_per_spin": roots,
        "energy_print_resolution_ev": 0.001, "oscillator_strength_unit": "dimensionless",
        "parameter_provenance": provenance, "geometry_scope": "unchanged selected source geometry",
        "method": "SCC-TD-DFTB/mio-1-1, gas-phase neutral closed-shell reference, vertical S/T roots",
        "interpretation": f"At this geometry S1 is {singlets[0]['excitation_ev']:.3f} eV and T1 is {triplets[0]['excitation_ev']:.3f} eV above S0 (S1 minus T1: {gap:.3f} eV). This compares spin-state energies, not transition rates.",
        "limitations": ["No SOC, ISC/RISC rates, triplet yields, relaxed triplet energy or emission lifetime.",
            "CHNO scope; approximate TD-DFTB can misorder charge-transfer states.",
            "Root labels are energy order within each spin manifold, not tracked state identities across geometries.",
            "Near-degenerate roots at the 0.001 eV output precision cannot be reliably distinguished."]}
