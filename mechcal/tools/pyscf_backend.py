"""Validated local CIS derivative, isolated from the main MCP interpreter."""

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from ase.units import Bohr, Hartree
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from mechcal.tools import pyscf_worker
from mechcal.tools.qc_process import QCError

Finite = Annotated[float, Field(allow_inf_nan=False)]
Vector = Annotated[list[Finite], Field(min_length=3, max_length=3)]


class Transition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    occupied: int = Field(ge=0)
    virtual: int = Field(ge=0)
    largest_normalized_weight: Finite = Field(ge=0, le=1)


class RawCISResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["pyscf_cis_s1_v1"]
    status: Literal["success"]
    version: str
    method: str
    units: dict[str, str]
    request: dict
    scf_converged: StrictBool
    td_converged: list[StrictBool] = Field(min_length=3, max_length=3)
    ground_energy_hartree: Finite
    excitations_hartree: Vector
    s1_total_energy_hartree: Finite
    gradient_hartree_per_bohr: list[Vector] = Field(min_length=2, max_length=pyscf_worker.MAX_ATOMS)
    oscillator_strengths: Vector
    dominant_transitions: list[Transition] = Field(min_length=3, max_length=3)
    td_residual_norms: Vector
    scf_seconds: Finite = Field(ge=0)
    td_seconds: Finite = Field(ge=0)
    gradient_seconds: Finite = Field(ge=0)
    memory_setting_mb: Literal[1000]
    nao: int = Field(gt=0)

    @model_validator(mode="after")
    def consistent_result(self):
        pyscf_worker.validate_request(self.request)
        if self.version != pyscf_worker.VERSION or self.method != pyscf_worker.METHOD:
            raise ValueError("Unexpected scientific method or PySCF version.")
        if self.units != {"energy": "hartree", "gradient": "hartree/bohr", "coordinates": "angstrom"}:
            raise ValueError("Unexpected physical units.")
        if not self.scf_converged or not all(self.td_converged):
            raise ValueError("Unconverged reference or excitations.")
        if min(self.excitations_hartree) <= 0 or self.excitations_hartree != sorted(self.excitations_hartree):
            raise ValueError("Nonpositive or unordered excitations.")
        if min(self.td_residual_norms) < 0 or max(self.td_residual_norms) > 1.01e-5:
            raise ValueError("Excitation residual check failed.")
        if min(self.oscillator_strengths) < 0:
            raise ValueError("Negative oscillator strength.")
        if len(self.gradient_hartree_per_bohr) != len(self.request["symbols"]):
            raise ValueError("Gradient does not preserve source atom count.")
        occupied = sum({"H":1,"C":6,"N":7,"O":8}[s] for s in self.request["symbols"]) // 2
        if any(t.occupied >= occupied or t.virtual >= self.nao - occupied for t in self.dominant_transitions):
            raise ValueError("Orbital transition index is outside the occupied/virtual spaces.")
        if abs(self.ground_energy_hartree + self.excitations_hartree[0] - self.s1_total_energy_hartree) > 1e-9:
            raise ValueError("S1 total energy is inconsistent with the reference.")
        return self


class RawCISContrastResult(RawCISResult):
    schema_version: Literal["pyscf_cis_s1_v2"]
    ground_gradient_hartree_per_bohr: list[Vector] = Field(min_length=2, max_length=pyscf_worker.MAX_ATOMS)
    ground_gradient_seconds: Finite = Field(ge=0)

    @model_validator(mode="after")
    def matched_gradients(self):
        if len(self.ground_gradient_hartree_per_bohr) != len(self.gradient_hartree_per_bohr):
            raise ValueError("S0 and S1 gradients must have the same source atom count.")
        return self


def evaluate(job, atoms):
    request = dict(symbols=atoms.get_chemical_symbols(), coordinates_angstrom=atoms.positions.tolist(),
        cpu_threads=job.config.threads)
    pyscf_worker.validate_request(request)
    directory = job.workdir / "cis_s1"
    directory.mkdir()
    input_path, output_path = directory / "request.json", directory / "result.json"
    input_path.write_text(json.dumps(request, indent=2, allow_nan=False))
    worker_path = Path(pyscf_worker.__file__).resolve()
    job.execute([job.config.pyscf_python, str(worker_path), str(input_path), str(output_path)], directory)
    try:
        raw = RawCISContrastResult.model_validate_json(output_path.read_text())
        if raw.request != request:
            raise ValueError("Worker result does not match the requested geometry/resources.")
    except (OSError, ValueError, TypeError) as exc:
        raise QCError("parse_failed", str(exc)) from exc
    gradient = np.asarray(raw.gradient_hartree_per_bohr) * Hartree / Bohr
    ground_gradient = np.asarray(raw.ground_gradient_hartree_per_bohr) * Hartree / Bohr
    difference = gradient - ground_gradient
    change_norms = np.linalg.norm(difference, axis=1)
    largest_changes = np.argsort(-change_norms, kind="stable")[:5]
    states = [dict(state=f"S{i+1}", excitation_ev=e*Hartree, oscillator_strength=f,
        dominant_transition=t.model_dump()) for i, (e, f, t) in enumerate(zip(
            raw.excitations_hartree, raw.oscillator_strengths, raw.dominant_transitions))]
    return dict(method=raw.method, software_version=raw.version, states=states,
        ground_energy_hartree=raw.ground_energy_hartree, s1_total_energy_hartree=raw.s1_total_energy_hartree,
        s2_minus_s1_ev=(raw.excitations_hartree[1]-raw.excitations_hartree[0])*Hartree,
        atom_index_base=0, symbols_in_source_order=request["symbols"],
        orbital_index_convention="Zero-based within occupied and virtual orbital spaces separately; virtual 0 is the first unoccupied orbital.",
        coordinates_angstrom=request["coordinates_angstrom"],
        s1_gradient_ev_per_angstrom=gradient.tolist(), s1_forces_ev_per_angstrom=(-gradient).tolist(),
        s0_gradient_ev_per_angstrom=ground_gradient.tolist(), s0_forces_ev_per_angstrom=(-ground_gradient).tolist(),
        excitation_gradient_ev_per_angstrom=difference.tolist(),
        s1_minus_s0_force_ev_per_angstrom=(-difference).tolist(),
        contrast_definition="At identical coordinates, excitation gradient = d(E_S1-E_S0)/dR = gradient_S1-gradient_S0; force change = force_S1-force_S0 = -excitation_gradient.",
        largest_force_changes=[dict(atom_index=int(i),symbol=request["symbols"][i],
            magnitude_ev_per_angstrom=float(change_norms[i])) for i in largest_changes],
        s0_gradient_norm_ev_per_angstrom=float(np.linalg.norm(ground_gradient)),
        excitation_gradient_norm_ev_per_angstrom=float(np.linalg.norm(difference)),
        gradient_norm_ev_per_angstrom=float(np.linalg.norm(gradient)),
        reference_converged=True, td_residual_norms=raw.td_residual_norms,
        component_seconds={"reference":raw.scf_seconds, "excitation":raw.td_seconds,
            "gradient":raw.gradient_seconds, "ground_gradient":raw.ground_gradient_seconds},
        provenance={"request":str(input_path), "raw_result":str(output_path), "log":str(output_path.with_suffix('.log')),
            "request_sha256":hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "worker_sha256":hashlib.sha256(worker_path.read_bytes()).hexdigest()},
        interpretation=f"At the unchanged source geometry, CIS gives S1 excitation {states[0]['excitation_ev']:.6f} eV. Matched S0 and S1 forces separate the initial local force from its change upon excitation. The difference describes local excitation-energy sensitivity, not a trajectory; no relaxation was performed.",
        limitations=["Neutral closed-shell, connected CHNO molecules with 2 to 40 atoms; fixed RHF/6-31G/CIS, no density fitting.",
            "Local derivative on energy-ordered S1, not state-followed relaxation, a reaction barrier or a conical intersection.",
            "CIS and this small basis are qualitative, especially for charge transfer, Rydberg states and strong correlation.",
            "Compare geometry trends within this method; CIS energies are not interchangeable with Amesp or DFTB energies.",
            "No SOC, nonradiative rate, quantum yield, mechanism classification or experimental validation.",
            "1000 MB is the library memory setting, not an operating-system memory limit. Runtime and peak RSS are measured."])
