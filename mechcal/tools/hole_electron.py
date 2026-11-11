"""Optional, read-only TDA-aTB1 postprocessing through Amesp m2m and Multiwfn.

Multiwfn manual 3.21.A.3 defines the plain-text coefficient interchange.
No electronic calculation, coefficient normalization, or mechanism scoring here.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from mechcal.tools.amesp_baseline import _parse_final_geometry
from mechcal.tools.amesp_transitions import NUMBER, parse_transitions
from mechcal.chem.validated_scopes import SPATIAL_ELEMENTS, SPATIAL_MAX_ATOMS, SPATIAL_SCOPE

BOUNDARY = (
    "One fixed-geometry TDA-aTB1 vertical excitation; unrelaxed hole/electron densities. "
    "Atomic contributions use Mulliken partitioning, not ground-state charges. "
    "Spatial redistribution alone does not establish TICT, PET kinetics, an emitting state, "
    "or condensed-phase behavior. Numerical consistency is not experimental validation."
)


class SpatialAnalysisError(RuntimeError):
    def __init__(self, code, message, *, details=None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


def backend_paths():
    paths = [Path(os.environ.get(name, "")) for name in
             ("MECHCAL_MULTIWFN_BIN", "MECHCAL_AMESP_M2M_BIN")]
    if any(not p.is_file() or not os.access(p, os.X_OK) for p in paths):
        raise SpatialAnalysisError("capability_unsupported",
            "Configure executable MECHCAL_MULTIWFN_BIN and MECHCAL_AMESP_M2M_BIN.")
    if not (paths[0].parent / "settings.ini").is_file():
        raise SpatialAnalysisError("capability_unsupported", "Multiwfn settings.ini is missing.")
    return paths


def backend_unavailable():
    try:
        backend_paths()
    except SpatialAnalysisError as exc:
        return str(exc)
    return None


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_data(bundle, state_root=1):
    paths = {key: Path(bundle.file_paths.get(f"s1_{key}", "")) for key in ("aip", "aop")}
    paths["mo"] = paths["aip"].with_suffix(".mo")
    if any(not p.is_file() for p in paths.values()) or paths["aop"] != paths["aip"].with_suffix(".aop"):
        raise SpatialAnalysisError("precondition_missing", "Need matching s1 .aip/.aop/.mo from one job.")
    steps = [s for s in bundle.step_records if Path(s.aip_path) == paths["aip"] and Path(s.aop_path) == paths["aop"]]
    if len(steps) != 1 or not steps[0].terminated_normally or steps[0].exit_code != 0:
        raise SpatialAnalysisError("precondition_missing", "A successful matching calculation record is required.")
    provenance = steps[0].method_provenance
    inp, output = paths["aip"].read_text(), paths["aop"].read_text()
    if provenance.input_sha256 != fingerprint(paths["aip"]):
        raise SpatialAnalysisError("precondition_missing", "Calculation input does not match recorded provenance.")
    keywords = re.findall(r"(?m)^!\s*(.*)$", inp)
    if (len(keywords) != 1 or keywords[0].lower().split() != ["atb1", "tda"]
            or [v.lower() for v in provenance.input_keywords] != ["atb1", "tda"]):
        raise SpatialAnalysisError("capability_unsupported", "Initial spatial route supports canonical TDA-aTB1 only.")
    blocks = re.findall(r"(?mi)^>\s*(\w+)", inp)
    forbidden = r"(?mi)^\s*(?:natorb|lmo|tdspin\s+(?!s\s*$))\b"
    geom = re.search(r"(?mi)^>xyz\s+(-?\d+)\s+1\s*\n(.*?)^end\s*$", inp, re.S)
    if not geom or set(blocks) - {"ope", "atb", "scf", "posthf", "xyz"} or re.search(forbidden, inp):
        raise SpatialAnalysisError("capability_unsupported", "Requires a closed-shell gas-phase canonical singlet job.")
    try:
        rows = [r.split() for r in geom.group(2).strip().splitlines()]
        symbols = [r[0] for r in rows]
        coordinates = [[float(v) for v in r[1:]] for r in rows]
        assert all(len(v) == 3 and all(math.isfinite(x) for x in v) for v in coordinates)
    except (ValueError, AssertionError, IndexError) as exc:
        raise SpatialAnalysisError("parse_failed", "Invalid submitted geometry.") from exc
    if not symbols or len(symbols) > SPATIAL_MAX_ATOMS or set(symbols) - SPATIAL_ELEMENTS:
        raise SpatialAnalysisError("capability_unsupported", "Supported element scope: " + SPATIAL_SCOPE)
    osymbols, ocoords = _parse_final_geometry(output)
    if ("Normal termination of Amesp!" not in output or symbols != osymbols
            or any(math.dist(a, b) > 1e-5 for a, b in zip(coordinates, ocoords, strict=True))):
        raise SpatialAnalysisError("precondition_missing", "Output geometry/termination does not match input.")
    states = [s for s in parse_transitions(output)["dominant_transitions"] if s["state_index"] == state_root]
    if len(states) != 1:
        raise SpatialAnalysisError("precondition_missing", "Requested state has no complete transition table.")
    coeff = states[0]["components"]
    norm = sum(c["coefficient"] ** 2 for c in coeff)
    precision = re.search(r"(?mi)^\s*tout\s+(\d+)\s*$", inp)
    if not precision or int(precision[1]) < 4 or abs(norm - .5) > 1e-4:
        raise SpatialAnalysisError("precondition_missing",
            "Insufficient transition coefficients; explicitly request a new calculation with td_tout >= 4. No automatic rerun.")
    energy = re.search(rf"State\s+{state_root}\s*:\s*E\s*=\s*({NUMBER}) eV", output)
    dipoles = [d for d in parse_transitions(output)["ground_to_excited_transition_dipoles"] if d["to_state"] == state_root]
    if not energy or len(dipoles) != 1:
        raise SpatialAnalysisError("precondition_missing", "Energy and transition dipole are required for cross-validation.")
    return paths, output, symbols, coordinates, coeff, norm, float(energy[1]), dipoles[0]["vector_au"]


def validate_molden(text, output, symbols, coordinates, coeff):
    block = re.search(r"\[Atoms\]\s+AU\s*\n(.*?)\[", text, re.S)
    if not block:
        raise SpatialAnalysisError("parse_failed", "Expected m2m atomic-unit geometry.")
    rows = [r.split() for r in block[1].strip().splitlines()]
    if [r[0] for r in rows] != symbols or any(
        math.dist([float(x) * .529177210903 for x in r[3:6]], xyz) > 1e-5
        for r, xyz in zip(rows, coordinates, strict=True)
    ):
        raise SpatialAnalysisError("precondition_missing", "Orbital file geometry/order mismatch.")
    energies = [float(x) for x in re.findall(rf"Ene=\s*({NUMBER})", text)]
    occ = [float(x) for x in re.findall(rf"Occup=\s*({NUMBER})", text)]
    electrons = re.search(r"Alpha electrons:\s*(\d+)\s+Beta electrons:\s*(\d+)", output)
    basis = re.search(r"Number of basis functions:\s*(\d+)", output)
    if (not electrons or not basis or len(energies) != int(basis[1])
            or any(not math.isfinite(v) for v in energies + occ)
            or any(v not in (0, 2) for v in occ)
            or int(electrons[1]) != int(electrons[2])
            or abs(sum(occ) - int(electrons[1]) - int(electrons[2])) > 1e-6):
        raise SpatialAnalysisError("precondition_missing", "Valence electron count, basis size or orbital occupations do not match.")
    table = []
    for label in ("occ", "vir"):
        match = re.search(rf"{label} orbital:\s+\d+\s+\d+\s*\n(.*?)(?:\n\s*\n)", output, re.S)
        if not match:
            raise SpatialAnalysisError("parse_failed", "Canonical orbital energy table unavailable.")
        table.extend(float(x) for x in match[1].split())
    if (len(table) != len(energies) or len(occ) != len(energies)
            or any(abs(a - b) > 2e-6 for a, b in zip(table, energies, strict=True))
            or any(c["from_orbital"] < 1 or c["to_orbital"] > len(occ)
                   or c["to_orbital"] < 1 or c["from_orbital"] > len(occ)
                   or occ[c["from_orbital"] - 1] != 2 or occ[c["to_orbital"] - 1] != 0 for c in coeff)):
        raise SpatialAnalysisError("precondition_missing", "Orbital energies, occupations or transition indices do not match.")


def parse_spatial(log, dipole):
    def scalar(label):
        m = re.search(re.escape(label) + rf"\s*({NUMBER})", log)
        if not m or not math.isfinite(float(m[1])):
            raise SpatialAnalysisError("parse_failed", f"Missing/invalid {label}")
        return float(m[1])

    def vector(label):
        m = re.search(re.escape(label) + rf"\s*({NUMBER})\s+({NUMBER})\s+({NUMBER})", log)
        if not m:
            raise SpatialAnalysisError("parse_failed", f"Missing {label}")
        result = [float(v) for v in m.groups()]
        if not all(math.isfinite(v) for v in result):
            raise SpatialAnalysisError("parse_failed", "Non-finite spatial vector.")
        return result

    hole, electron = scalar("Integral of hole:"), scalar("Integral of electron:")
    transition = scalar("Integral of transition density:")
    computed = vector("Transition dipole moment in X/Y/Z:")
    dipole_error = min(math.dist(computed, dipole), math.dist(computed, [-v for v in dipole]))
    hc, ec = vector("Centroid of hole in X/Y/Z:"), vector("Centroid of electron in X/Y/Z:")
    distance, overlap = scalar("D index:"), scalar("Sr index (integral of Sr function):")
    checks = [
        ("hole_normalization_error", abs(hole - 1), 0, .005),
        ("electron_normalization_error", abs(electron - 1), 0, .005),
        ("transition_density_integral_absolute", abs(transition), 0, .005),
        ("transition_dipole_error_up_to_sign", dipole_error, 0,
         max(.005, math.sqrt(sum(v*v for v in dipole)) * .01)),
        ("spatial_overlap_sr", overlap, 0, 1.001),
        ("centroid_distance_angstrom", distance, 0, None),
        ("centroid_distance_error_angstrom", abs(math.dist(hc, ec) - distance), 0, .002),
    ]
    failed = [{"check": name, "observed": value, "minimum": low, "maximum": high}
              for name, value, low, high in checks if value < low or (high is not None and value > high)]
    if failed:
        raise SpatialAnalysisError("numerical_validation_failed",
            "Spatial numerical checks failed: " + ", ".join(c["check"] for c in failed) + "; no usable evidence.",
            details={"failed_checks": failed})
    return {"hole_centroid_angstrom": hc, "electron_centroid_angstrom": ec,
            "centroid_distance_angstrom": distance, "spatial_overlap_sr": overlap,
            "t_index_angstrom": scalar("t index:"),
            "numerical_checks": {"hole_integral": hole, "electron_integral": electron,
                "transition_density_integral": transition, "transition_dipole_error_up_to_sign": dipole_error}}


def fragment_contributions(atoms, symbols, fragments):
    if not fragments:
        return []
    seen = set()
    result = []
    for name, indices in fragments.items():
        if (not name.strip() or not indices or any(type(i) is not int or not 0 <= i < len(symbols) for i in indices)
                or len(indices) != len(set(indices)) or seen.intersection(indices)):
            raise SpatialAnalysisError("precondition_missing", "Fragment groups must have names and disjoint valid zero-based atom indices.")
        seen.update(indices)
        selected = [a for a in atoms if a["atom_index"] in indices]
        result.append({"name": name, "atom_indices": indices,
            "hole_percent": sum(a["hole_percent"] for a in selected),
            "electron_percent": sum(a["electron_percent"] for a in selected),
            "omitted_hydrogen_indices": [i for i in indices if symbols[i] == "H"],
            "scope": "Sum of reported heavy-atom Mulliken contributions; hydrogens excluded, not renormalized or an electron-transfer probability."})
    return result


def run_spatial(bundle, out, state_root=1, spatial_grid="high", max_wall_seconds=300, fragments=None):
    try:
        configured_limit = int(os.environ.get("MECHCAL_SPATIAL_MAX_WALL_SECONDS", "1800"))
        if not 10 <= configured_limit <= 1800:
            raise ValueError("outside 10-1800")
    except ValueError as exc:
        raise SpatialAnalysisError("capability_unsupported", "Invalid MECHCAL_SPATIAL_MAX_WALL_SECONDS configuration.") from exc
    if not 10 <= max_wall_seconds <= configured_limit:
        raise SpatialAnalysisError("invalid_args", "Requested postprocessing time exceeds the permitted range.",
            details={"stage": "preflight", "requested_wall_seconds": max_wall_seconds,
                     "effective_max_wall_seconds": configured_limit})
    multiwfn, m2m = backend_paths()
    paths, output, symbols, xyz, coeff, norm, energy, dipole = source_data(bundle, state_root)
    fragment_contributions([], symbols, fragments)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    completed_stages = []
    hashes = {k: fingerprint(p) for k, p in paths.items()}
    for key, path in paths.items():
        shutil.copy2(path, out / f"input.{key}")
    settings = (multiwfn.parent / "settings.ini").read_text()
    for key, value in (("nthreads", "1"), ("cfgcrossthres", "0.001")):
        settings, count = re.subn(rf"(?m)^(\s*{key}=\s*)\S+", rf"\g<1>{value}", settings)
        if count != 1:
            raise SpatialAnalysisError("capability_unsupported", f"Unsupported Multiwfn settings: {key}")
    (out / "settings.ini").write_text(settings)
    env = dict(os.environ, OMP_STACKSIZE="512M", OMP_NUM_THREADS="1", Multiwfnpath=str(multiwfn.parent))

    def call(executable, args, commands, name):
        (out / f"{name}.stdin").write_text(commands)
        remaining = max_wall_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise SpatialAnalysisError("timeout", "Spatial analysis wall limit exhausted.",
                details={"stage": name, "max_wall_seconds": max_wall_seconds,
                         "completed_stages": list(completed_stages),
                         "elapsed_seconds": time.monotonic() - started})
        try:
            with (out / f"{name}.log").open("w") as log:
                result = subprocess.run([str(executable), *args], cwd=out, env=env, input=commands,
                    text=True, stdout=log, stderr=subprocess.STDOUT, timeout=remaining, check=False)
        except subprocess.TimeoutExpired as exc:
            raise SpatialAnalysisError("timeout", f"{name} timed out; raw logs retained in {out}.",
                details={"stage": name, "max_wall_seconds": max_wall_seconds,
                         "completed_stages": list(completed_stages),
                         "elapsed_seconds": time.monotonic() - started}) from exc
        except OSError as exc:
            raise SpatialAnalysisError("process_failed", f"{name} could not start: {exc}; logs: {out}",
                details={"stage": name}) from exc
        if result.returncode != 0:
            raise SpatialAnalysisError("process_failed", f"{name} exited {result.returncode}; logs: {out}",
                details={"stage": name, "returncode": result.returncode})
        completed_stages.append(name)
        return (out / f"{name}.log").read_text()

    call(m2m, [], "input.mo\n", "m2m")
    if not (out / "input.molden").is_file():
        raise SpatialAnalysisError("parse_failed", "Converter produced no orbital file.")
    validate_molden((out / "input.molden").read_text(), output, symbols, xyz, coeff)
    transitions = [f"Excited State {state_root} 1 {energy}"]
    transitions.extend(f"{c['from_orbital']} -> {c['to_orbital']} {c['coefficient']:.7f}" for c in coeff)
    (out / "transitions.txt").write_text("\n".join(transitions) + "\n\n")
    grid = {"medium": 2, "high": 3}[spatial_grid]
    log = call(multiwfn, ["input.molden"], f"18\n1\ntransitions.txt\n1\n{grid}\n0\n0\n0\nq\n", "spatial")
    try:
        result = parse_spatial(log, dipole)
    except SpatialAnalysisError as exc:
        exc.details.update(stage="spatial_validation", spatial_grid=spatial_grid, state_root=state_root,
                           completed_stages=list(completed_stages), scientific_evidence=False,
                           elapsed_seconds=time.monotonic() - started, max_wall_seconds=max_wall_seconds)
        raise
    atomic = call(multiwfn, ["input.molden"], "18\n1\ntransitions.txt\n3\n1\n0\n0\n0\nq\n", "atomic")
    atoms = []
    for m in re.finditer(rf"(\d+)\((\w+)\s*\)\s+Hole:\s*({NUMBER})\s*%\s+Electron:\s*({NUMBER})", atomic):
        idx, element, h, e = int(m[1]) - 1, m[2], float(m[3]), float(m[4])
        if not 0 <= idx < len(symbols) or symbols[idx] != element or not all(math.isfinite(v) and -3 <= v <= 103 for v in (h, e)):
            raise SpatialAnalysisError("parse_failed", "Invalid atomic contribution or atom mapping.")
        atoms.append({"atom_index": idx, "element": element, "hole_percent": h, "electron_percent": e})
    if [a["atom_index"] for a in atoms] != [i for i, s in enumerate(symbols) if s != "H"]:
        raise SpatialAnalysisError("parse_failed", "Incomplete/duplicate heavy-atom contribution table.")
    if hashes != {k: fingerprint(p) for k, p in paths.items()}:
        raise SpatialAnalysisError("runtime_failed", "Input artifacts changed during postprocessing.")
    return {**result, "state_index": state_root, "excitation_energy_ev": energy,
            "method": "TDA-aTB1", "density_scope": "unrelaxed_vertical_excitation",
            "atom_index_base": 0, "atom_mapping_scope": "selected calculation geometry, not canonical SMILES order",
            "geometry": [{"atom_index": i, "element": s, "xyz_angstrom": c} for i, (s, c) in enumerate(zip(symbols, xyz))],
            "heavy_atom_contributions": atoms, "partition": "Mulliken", "hydrogens_omitted_from_contribution_table": True,
            "fragment_contributions": fragment_contributions(atoms, symbols, fragments),
            "coefficient_squared_sum": norm, "coefficient_count": len(coeff), "source_sha256": hashes,
            "grid": spatial_grid, "cross_term_threshold": .001, "elapsed_seconds": time.monotonic() - started,
            "postprocessor_banner": log.splitlines()[:12], "new_quantum_calculations": 0,
            "limitations": [BOUNDARY]}
