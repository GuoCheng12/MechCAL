"""Broader-element state analysis at explicit, unchanged geometries."""

import json
from pathlib import Path
import numpy as np

from mechcal.tools import pyscf_states_worker
from mechcal.tools.qc_process import QCError
from mechcal.tools.qc_logs import log_excerpt


def evaluate(job, atoms, args, charge):
    partition = [i for group in args.fragment_atom_indices for i in group]
    if partition and sorted(partition) != list(range(len(atoms))):
        raise QCError("invalid_args", "Fragment groups must partition every source atom exactly once.")
    request = dict(symbols=atoms.get_chemical_symbols(), coordinates_angstrom=atoms.positions.tolist(),
        charge=charge, threads=job.config.pyscf_states_threads, max_memory_mb=job.config.pyscf_states_max_memory_mb,
        **args.model_dump(exclude={"source_artifact_id", "max_wall_seconds"}))
    directory = job.workdir / "spin_states"
    directory.mkdir()
    source, output = directory/"request.json", directory/"result.json"
    source.write_text(json.dumps(request, indent=2, allow_nan=False))
    error = None
    try:
        job.execute([job.config.pyscf_python, str(Path(pyscf_states_worker.__file__).resolve()), str(source), str(output)], directory,
                    threads=job.config.pyscf_states_threads)
    except QCError as exc:
        error = exc
    if not output.exists():
        log = directory / "process.log"
        if log.exists() and "SCF did not converge" in log.read_text(errors="replace"):
            raise QCError("not_converged", "SCF did not converge; no states were computed. Technical options: "
                          "scf_solver='newton' or a level_shift (e.g. 0.2-0.5 Hartree).") from error
        raise error or QCError("parse_failed", "No state result was produced.")
    raw = json.loads(output.read_text())
    if raw.get("schema_version") != "pyscf_spin_states_v1" or raw.get("request") != request or raw.get("scf_converged") is not True:
        raise QCError("parse_failed", "State result does not match the request/converged reference.")
    rows = raw.get("states", [])
    if not rows:
        completed = ["ground_state_scf"] if raw.get("scf_converged") else []
        raise QCError(error.code if error else "not_converged",
            "No spin-state result completed after the ground-state calculation.", details={
                "stage": "excited_states", "completed_stages": completed,
                "scf_converged": raw.get("scf_converged"), "scf_seconds": raw.get("scf_seconds"),
                "completed_state_count": 0, "stability_check_status": raw.get("stability_check_status"),
                "max_wall_seconds": job.max_wall_seconds,
                "elapsed_seconds": job.timing()["total_wall_seconds"],
                "raw_result": str(output), "calculation_log": str(output.with_suffix('.log')),
                "calculation_diagnostic": log_excerpt(output.with_suffix('.log'), job.workdir),
                "scientific_evidence": False}) from error
    for row in rows:
        if not np.isfinite(row["excitation_ev"]) or row["excitation_ev"] <= 0 or not 0 <= row["td_residual_norm"] <= 2e-5:
            raise QCError("parse_failed", "Nonfinite energy or invalid response residual.")
        for field in ("hole_population_by_atom", "electron_population_by_atom"):
            if len(row[field]) != len(atoms) or not np.isfinite(row[field]).all() or not np.isclose(sum(row[field]), 1., atol=1e-6):
                raise QCError("parse_failed", "Invalid atom population table.")
    failures = list(raw.get("spin_manifold_failures", [])) + list(raw.get("validation_failures", []))
    if error:
        failures.append({"reason": str(error), "failure_code": error.code})
    if raw.get("stability_check_status") != "completed":
        failures.append({"reason":"Reference stability check was not completed; state energies/populations are provisional.","failure_code":"not_converged"})
        raw["interpretation"]=f"Retained {len(rows)} converged spin-free states, but reference stability is not established. Do not treat this as validated triplet behavior."
    per_spin = {p: sum(r["state"].startswith(p) for r in rows) for p in ("S", "T")}
    states_complete = all(n >= request["nstates_per_spin"] for n in per_spin.values())
    raw["completion"] = ("complete" if states_complete and raw.get("stability_check_status") == "completed"
                         else "states_complete_stability_unverified" if states_complete else "incomplete_states")
    raw["points"] = [dict(status="success", member_id=r["state"]) for r in rows] + [dict(status="failed", **f) for f in failures]
    raw.setdefault("interpretation", f"Retained {len(rows)} converged spin-free states; another calculation did not complete.")
    raw["atom_index_base"] = 0
    raw["geometry_optimization_status"] = "not_requested"
    raw["provenance"] = {"request": str(source), "raw_result": str(output), "log": str(output.with_suffix('.log'))}
    raw["spin_gaps"] = [dict(singlet=s["state"], triplet=t["state"], singlet_minus_triplet_ev=s["excitation_ev"]-t["excitation_ev"])
        for s in rows if s["state"].startswith("S") for t in rows if t["state"].startswith("T")]
    return raw
