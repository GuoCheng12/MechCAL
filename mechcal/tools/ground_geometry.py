"""Planner-requested recovery of an embedded seed, without force-field parameters."""

import json
import time
from math import isfinite
from pathlib import Path
from uuid import uuid4

import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from ase.units import Hartree
from rdkit import Chem

from mechcal.chem.structure_prep import StructurePrepError, validate_geometry_coordinates
from mechcal.schemas import (
    ArtifactRecord,
    CaseRun,
    DispatchRequest,
    EvidenceUnit,
    ToolExecutionResult,
)
from mechcal.schemas.tool_arguments import GroundGeometryArgs
from mechcal.tools.amesp_baseline import (
    AmespBaselineError,
    AmespBaselineRunner,
    _read_xyz_symbols_and_coordinates,
    _write_xyz,
)
from mechcal.tools.amesp_microscopic import AmespMicroscopicError
from mechcal.tools.amesp_scan import AmespForceCalculator


def optimize_ground_geometry(
    request: DispatchRequest,
    case_run: CaseRun,
    tool_args: dict,
    runner: AmespBaselineRunner,
) -> ToolExecutionResult:
    args = GroundGeometryArgs.model_validate(tool_args)
    sources = [
        a
        for a in case_run.artifact_manifest.artifacts
        if a.kind == "initial_structure"
        and a.status == "available"
        and (args.source_artifact_id is None or a.artifact_id == args.source_artifact_id)
    ]
    if not sources:
        raise AmespMicroscopicError(
            "precondition_missing", "An available initial_structure is required."
        )
    source = sources[-1]
    try:
        mol = Chem.MolFromMolFile(source.file_paths["sdf"], removeHs=False)
        symbols, coordinates = _read_xyz_symbols_and_coordinates(Path(source.file_paths["xyz"]))
        if mol is None or [a.GetSymbol() for a in mol.GetAtoms()] != symbols:
            raise ValueError("Initial SDF and XYZ atom order differ.")
        charge, multiplicity = source.metadata["charge"], source.metadata["multiplicity"]
        electrons = sum(a.GetAtomicNum() for a in mol.GetAtoms()) - charge
        if multiplicity != 1 or electrons % 2 or Chem.GetFormalCharge(mol) != charge:
            raise ValueError("Recovery requires a consistent closed-shell singlet charge/state.")
        validate_geometry_coordinates(mol, coordinates)
    except (KeyError, OSError, ValueError, StructurePrepError) as exc:
        raise AmespMicroscopicError(
            "precondition_missing", f"Invalid initial geometry: {exc}"
        ) from exc

    unique = uuid4().hex
    workdir = Path(source.file_paths["xyz"]).parents[1] / "microscopic" / request.round_id / unique
    workdir.mkdir(parents=True, exist_ok=False)
    atoms = Atoms(symbols=symbols, positions=coordinates)
    calculator = AmespForceCalculator(
        runner, workdir, charge, multiplicity, None, time.monotonic() + args.max_wall_seconds
    )
    atoms.calc = calculator
    optimizer_log = workdir / "optimizer.log"
    evaluations = workdir / "evaluations.json"
    paths = {"optimizer_log": str(optimizer_log), "evaluations": str(evaluations)}
    try:
        optimizer = BFGS(atoms, logfile=str(optimizer_log), maxstep=0.1)
        converged = optimizer.run(fmax=0.02, steps=args.max_opt_cycles)
        max_force = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
        if not converged or not isfinite(max_force) or max_force > 0.02:
            raise AmespBaselineError(
                "optimization_not_converged",
                f"Maximum atomic force {max_force:.6f} eV/angstrom exceeds 0.02.",
            )
        xyz = atoms.positions.tolist()
        validate_geometry_coordinates(mol, xyz)
        energy = float(atoms.get_potential_energy() / Hartree)
        iterations = optimizer.nsteps
        if not isfinite(energy):
            raise AmespBaselineError("parse_failed", "Missing finite final ground-state energy.")
    except (AmespBaselineError, StructurePrepError) as exc:
        if calculator.records:
            last_step = calculator.records[-1]
            paths.update(aip=str(last_step.aip_path), aop=str(last_step.aop_path))
        raise AmespBaselineError(
            exc.code,
            f"Ground-state geometry recovery failed: {exc.message} No prepared geometry was published.",
            file_paths={**paths, **getattr(exc, "file_paths", {})},
            details={
                "requested_parameters": args.model_dump(exclude_none=True),
                "force_evaluations": len(calculator.records),
            },
        ) from exc
    finally:
        evaluations.write_text(
            json.dumps([step.to_json() for step in calculator.records], indent=2)
        )
    final_step = calculator.records[-1]
    paths.update(
        aip=str(final_step.aip_path),
        aop=str(final_step.aop_path),
        stdout=str(final_step.stdout_path),
        stderr=str(final_step.stderr_path),
    )

    for i, position in enumerate(xyz):
        mol.GetConformer().SetAtomPosition(i, position)
    xyz_path, sdf_path = workdir / "ground_geometry.xyz", workdir / "ground_geometry.sdf"
    _write_xyz(xyz_path, label="aTB1 ground-state optimized seed", symbols=symbols, coordinates=xyz)
    with Chem.SDWriter(str(sdf_path)) as writer:
        writer.write(mol)
    observations = {
        "method": "aTB1",
        "state": "ground_state",
        "environment": "gas_phase",
        "optimization_converged": True,
        "optimization_steps": iterations,
        "optimizer": "ASE_BFGS",
        "convergence_criterion": "maximum_atomic_force_at_most_0.02_eV_per_angstrom",
        "max_force_ev_per_angstrom": max_force,
        "force_evaluations": len(calculator.records),
        "final_energy_hartree": energy,
        "atom_order_preserved": True,
        "minimum_verified_by_frequencies": False,
        "requested_parameters": args.model_dump(exclude_none=True),
        "effective_parameters": {
            "max_opt_cycles": args.max_opt_cycles,
            "optimization_coordinates": "cartesian",
            "fmax_ev_per_angstrom": 0.02,
            "max_step_angstrom": 0.1,
            "scf_conver": 10,
            "max_wall_seconds": args.max_wall_seconds,
            "single_force_timeout_seconds": runner.timeout_seconds,
        },
        "molecular_context": {"charge": charge, "multiplicity": multiplicity},
        "available_descriptors": ["ground_state_geometry", "ground_state_energy"],
        "missing_descriptors": [],
    }
    metadata = {
        **source.metadata,
        "force_field": None,
        "geometry_status": "ground_state_optimized",
        "geometry_source": "amesp_aTB1",
        "parent_artifact_id": source.artifact_id,
        "selected_conformer_id": 0,
        "conformer_count": 1,
        "xyz_path": str(xyz_path),
        "sdf_path": str(sdf_path),
        "computed_observables": observations,
    }
    summary = workdir / "geometry_summary.json"
    metadata["summary_path"] = str(summary)
    summary.write_text(json.dumps(metadata, indent=2) + "\n")
    artifact = ArtifactRecord(
        artifact_id=f"geometry:{unique}",
        kind="prepared_structure",
        status="available",
        created_by_round=request.round_id,
        created_by_agent="microscopic",
        file_paths={"xyz": str(xyz_path), "sdf": str(sdf_path), "summary": str(summary), **paths},
        observable_tags=["ground_state_geometry", "geometry_precondition"],
        reusable_for=["macro", "microscopic", "artifact_inspection"],
        metadata=metadata,
    )
    report_id = f"{request.round_id}:microscopic:{unique}"
    message = "Ground-state geometry recovery converged; a checked input geometry is available."
    evidence = EvidenceUnit(
        evidence_id=f"{report_id}:geometry",
        round_id=request.round_id,
        source_report_id=report_id,
        agent_name="microscopic",
        capability_id=request.capability_id,
        claim=message,
        context="gas-phase input geometry preparation",
        basis="computed",
        support="not_applicable",
        summary=message,
        family="geometry_precondition",
        relation="neutral",
        artifact_refs=[artifact.artifact_id],
        metrics={"computed_observables": observations},
        limits=[
            "Convergence is not a frequency-verified minimum or evidence for any emission mechanism.",
            "One recovered conformer, not an ensemble or experimental molecular geometry.",
        ],
    )
    return ToolExecutionResult(
        tool_result_id=report_id,
        round_id=request.round_id,
        agent_name="microscopic",
        dispatch_id=request.dispatch_id,
        capability_id=request.capability_id,
        selected_route=request.route,
        tool_calls=["amesp.optimize_ground_state_geometry"],
        status="success",
        raw_results={"file_paths": paths, "tool_args": tool_args},
        structured_results={"selected_route": request.route, "parsed_observables": observations},
        evidence_units=[evidence],
        artifact_updates=[artifact],
        summary=message,
    )
