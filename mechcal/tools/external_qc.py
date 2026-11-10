"""Existing Worker/MCP contract for opt-in scientific backend computations."""

import json
from pathlib import Path
import time
from uuid import uuid4
import numpy as np
from ase.io import read
from ase.io.formats import UnknownFileTypeError
from rdkit import Chem

from mechcal.capabilities.external_qc import ROUTES
from mechcal.schemas import ArtifactRecord, EvidenceUnit, ToolExecutionResult
from mechcal.tools.failure_feedback import member_failures, tool_failure
from mechcal.schemas.tool_arguments import merge_tool_args
from mechcal.schemas.external_qc import EXTERNAL_ARGUMENTS
from mechcal.tools.qc_process import QCConfig, QCError, QCJob


def source_preflight(manifest, artifact_id, backend):
    candidates = [a for a in manifest.artifacts if a.kind == "prepared_structure"
                  and a.status in {"available", "partial"}
                  and (artifact_id is None or a.artifact_id == artifact_id)]
    if not candidates:
        raise ValueError(f"A registered prepared_structure is required; requested artifact_id={artifact_id!r}.")
    source = candidates[-1]
    if not source.file_paths.get("xyz") or not source.file_paths.get("sdf"):
        raise ValueError("Prepared structure requires XYZ and SDF files with a shared atom map.")
    try:
        atoms = read(source.file_paths["xyz"])
        mol = Chem.MolFromMolFile(source.file_paths["sdf"], removeHs=False)
    except (OSError, ValueError, IndexError, StopIteration, UnknownFileTypeError) as exc:
        raise ValueError(f"Prepared structure cannot be read: {exc}") from exc
    if mol is None or [a.GetSymbol() for a in mol.GetAtoms()] != atoms.get_chemical_symbols():
        raise ValueError("XYZ and SDF atom order disagree.")
    if mol.GetNumConformers() != 1 or not np.allclose(
        mol.GetConformer().GetPositions(), atoms.positions, atol=1e-3, rtol=0
    ):
        raise ValueError("XYZ and SDF coordinates disagree.")
    if not np.isfinite(atoms.positions).all() or len(atoms) > 150:
        raise ValueError("Trial requires finite coordinates and at most 150 atoms.")
    charge, spin = source.metadata.get("charge"), source.metadata.get("multiplicity")
    if (type(charge) is not int or charge != Chem.GetFormalCharge(mol) or spin != 1
            or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
            or (backend != "pyscf_tda" and charge != 0)):
        raise ValueError("Initial backend trial only supports neutral closed-shell singlet structures.")
    if backend in {"pyscf_tda", "amesp_soc"}:
        from mechcal.tools.pyscf_states_worker import ELEMENTS
        if not set(atoms.get_chemical_symbols()) <= ELEMENTS or not -2 <= charge <= 2:
            raise ValueError("TDA route supports H C N O F S Cl Br B Si Au Fe, charge -2 to +2, closed-shell reference.")
    if backend == "crest" and len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("This conformer sampler requires a single molecule; use explicit dimer tools for contacts.")
    if backend in {"dftb", "pysoc"} and not set(atoms.get_chemical_symbols()) <= {"C", "H", "N", "O"}:
        raise ValueError("DFTB trial parameter coverage is restricted to CHNO.")
    if backend == "pysoc":
        occupied = sum({"C":4,"H":1,"N":5,"O":6}[s] for s in atoms.get_chemical_symbols()) // 2
        orbitals = sum(1 if s == "H" else 4 for s in atoms.get_chemical_symbols())
        if occupied * (orbitals - occupied) <= 3:
            raise ValueError("SOC requires an excitation space larger than the three requested roots.")
    if backend == "pyscf":
        from mechcal.tools.pyscf_worker import ELEMENTS, MAX_ATOMS
        if not 2 <= len(atoms) <= MAX_ATOMS or not set(atoms.get_chemical_symbols()) <= ELEMENTS:
            raise ValueError("CIS local-gradient trial requires CHNO and 2 to 40 atoms.")
        if len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError("CIS local-gradient trial requires a single connected molecule.")
    if (sum(atoms.numbers)-charge) % 2:
        raise ValueError("Odd electron count cannot be a closed-shell singlet.")
    return source, atoms


def contact_scan_preflight(source, atoms, args=None):
    fragments = source.metadata.get("fragment_atom_indices")
    if (source.metadata.get("modeled_dimer") is not True or not isinstance(fragments, list)
            or len(fragments) != 2 or not all(isinstance(f, list) and f for f in fragments)
            or any(type(i) is not int for f in fragments for i in f)
            or fragments[0] + fragments[1] != list(range(len(atoms)))):
        raise ValueError("A registered modeled dimer with two contiguous, ordered fragments is required.")
    if args and (args.atom_indices[0] not in fragments[0] or args.atom_indices[1] not in fragments[1]):
        raise ValueError("First selected atom must belong to fragment A, second to fragment B.")
    return fragments


def prepared_contact_geometries(observables, source, partner, workdir, request):
    """Publish computed systems without reinterpreting them as the input monomer."""
    template = Chem.MolFromMolFile(source.file_paths["sdf"], removeHs=False)
    if partner:
        other = Chem.MolFromMolFile(partner.file_paths["sdf"], removeHs=False)
        if template is None or other is None:
            raise QCError("parse_failed", "Cannot read the two source topologies.")
        template = Chem.CombineMols(template, other)
    if template is None:
        raise QCError("parse_failed", "Cannot read source topology.")
    rows = observables.get("candidates", observables.get("points", []))
    artifacts = []
    for row in rows:
        if row["status"] != "success":
            continue
        atoms = read(row["xyz"])
        if atoms.get_chemical_symbols() != [a.GetSymbol() for a in template.GetAtoms()]:
            raise QCError("parse_failed", "Computed contact geometry lost source atom order.")
        mol = Chem.Mol(template)
        mol.RemoveAllConformers()
        conf = Chem.Conformer(len(atoms))
        for i, position in enumerate(atoms.positions):
            conf.SetAtomPosition(i, position.tolist())
        mol.AddConformer(conf)
        name = row["member_id"].replace(":", "_")
        sdf = workdir / f"{name}.sdf"
        Chem.MolToMolFile(mol, str(sdf))
        artifact_id = f"geometry:{workdir.name}:{name}"
        row["prepared_artifact_id"] = artifact_id
        context = {}
        is_dimer = request.route != "sample_conformers_crest"
        context.update(charge=source.metadata.get("charge", 0), multiplicity=1, atom_count=len(atoms), heavy_atom_count=mol.GetNumHeavyAtoms(),
            input_smiles=Chem.MolToSmiles(Chem.RemoveHs(mol)), canonical_smiles=Chem.MolToSmiles(Chem.RemoveHs(mol)),
            geometry_source="explicit_computed_selection", modeled_dimer=is_dimer,
            fragment_atom_indices=row.get("fragment_atom_indices", [list(range(len(atoms)))]),
            parent_source_artifact_ids=[source.artifact_id] + ([partner.artifact_id] if partner else []),
            parent_member_id=row["member_id"], geometry_converged=row.get("geometry_converged"),
            geometry_optimization_status=row.get("geometry_optimization_status", "converged" if row.get("geometry_converged") else "unspecified_legacy"),
            geometry_scope=("modeled dimer" if is_dimer else "sampled conformer")+"; preserve these coordinates for vertical comparisons",
            atom_index_base=0)
        artifacts.append(ArtifactRecord(artifact_id=artifact_id, kind="prepared_structure",
            created_by_round=request.round_id, created_by_agent="microscopic", status="available",
            file_paths={"xyz": row["xyz"], "sdf": str(sdf)}, metadata=context,
            reusable_for=["microscopic"], observable_tags=["modeled_dimer" if is_dimer else "sampled_conformer", "geometry_reuse"]))
    return artifacts


def execute_external(request, case_run, execution_plan=None, *, config=None, amesp_enabled=None):
    started = time.monotonic()
    config = config or QCConfig.from_environment()
    backend, _, description = ROUTES[request.route]
    job = None
    try:
        if request.capability_id != f"microscopic.{request.route}" or request.agent_name != "microscopic":
            raise QCError("invalid_args", "Capability ID, route and owner must agree.")
        reason = config.backend_reason(backend)
        if reason:
            raise QCError("unsupported", reason)
        if backend == "amesp_soc":
            import os
            enabled = amesp_enabled if amesp_enabled is not None else os.environ.get("MECHCAL_ENABLE_AMESP", "false").lower() in {"1", "true", "yes"}
            if not enabled:
                raise QCError("unsupported", "Amesp is disabled by the run environment.")
        if request.route == "scan_excited_state_coordinate" and not config.allow_excited_scan:
            raise QCError("unsupported", "Excited-state scans are disabled pending validation.")
        try:
            values = merge_tool_args(request, execution_plan)
            args = EXTERNAL_ARGUMENTS[request.route].model_validate(values)
        except ValueError as exc:
            raise QCError("invalid_args", str(exc)) from exc
        try:
            source, atoms = source_preflight(case_run.artifact_manifest, args.source_artifact_id, backend)
            if request.route in {"compare_singlet_triplet_states", "evaluate_spin_orbit_couplings"}:
                from mechcal.tools.dftb_spin import spin_parameters
                spin_parameters(config, atoms.get_chemical_symbols())
            partner, partner_source = None, None
            if request.route == "search_dimer_contacts":
                partner_source, partner = source_preflight(case_run.artifact_manifest, args.partner_artifact_id, backend)
                if len(atoms) + len(partner) > 150:
                    raise ValueError("Dimer calculations support at most 150 atoms in total.")
            if request.route == "scan_intermolecular_distance":
                fragments = contact_scan_preflight(source, atoms, args)
            if request.route == "compare_contact_torsion_response":
                from mechcal.tools.contact_motion import coordinate_preflight
                fragments = contact_scan_preflight(source, atoms)
                coordinate_preflight(Chem.MolFromMolFile(source.file_paths['sdf'], removeHs=False), atoms, args, fragments[0])
        except ValueError as exc:
            raise QCError("missing_input", str(exc)) from exc
        workdir = Path(source.file_paths["xyz"]).resolve().parent / "external_qc" / uuid4().hex
        job = QCJob(workdir, config, args.max_wall_seconds)
        from mechcal.tools import xtb_backend, dftb_backend
        if request.route == "estimate_soc_amesp":
            from mechcal.tools.amesp_soc_backend import evaluate
            observables = evaluate(job, atoms, args)
        elif request.route == "scan_cis_excitation_coordinate":
            from mechcal.tools.cis_path import scan
            observables = scan(job, atoms, args, source)
        elif request.route == "compare_contact_torsion_response":
            from mechcal.tools.contact_motion import evaluate
            observables = evaluate(job, atoms, args, source, fragments)
        elif request.route == "sample_conformers_crest":
            from mechcal.tools.crest_backend import sample
            observables = sample(job, atoms, args, source)
        elif request.route == "compare_spin_states_pyscf":
            from mechcal.tools.pyscf_states import evaluate
            observables = evaluate(job, atoms, args, source.metadata['charge'])
        elif request.route == "estimate_vertical_ip_ea":
            observables = xtb_backend.ip_ea(job, atoms)
        elif request.route == "analyze_ground_state_vibrations":
            observables = xtb_backend.vibrations(job, atoms)
        elif request.route == "search_dimer_contacts":
            observables = xtb_backend.dimer(job, atoms, partner, args.max_candidates, args.max_opt_cycles)
        elif request.route == "scan_intermolecular_distance":
            observables = xtb_backend.scan_contacts(job, atoms, args, fragments)
        elif request.route == "evaluate_excited_state_gradient":
            observables = dftb_backend.evaluate(job, atoms, workdir / "gradient", args.state_root, args.nstates)
        elif request.route == "compare_singlet_triplet_states":
            from mechcal.tools.dftb_spin import compare_states
            observables = compare_states(job, atoms, args.nstates_per_spin)
        elif request.route == "evaluate_spin_orbit_couplings":
            from mechcal.tools.dftb_soc import evaluate
            observables = evaluate(job, atoms)
        elif request.route == "evaluate_cis_s1_gradient":
            from mechcal.tools.pyscf_backend import evaluate
            observables = evaluate(job, atoms)
        else:
            observables = dftb_backend.scan(job, atoms, args)
        if request.route == "scan_excited_state_coordinate" and not all(p["converged"] for p in observables["points"]):
            status = "partial"
        else:
            status = "success"
        if request.route in {"scan_intermolecular_distance", "compare_contact_torsion_response", "sample_conformers_crest", "compare_spin_states_pyscf", "scan_cis_excitation_coordinate"}:
            successes = sum(p["status"] == "success" for p in observables["points"])
            status = "success" if successes == len(observables["points"]) else "partial" if successes else "failed"
        if request.route == "search_dimer_contacts":
            complete = all(a["status"] in {"success", "duplicate"} for a in observables["search_attempts"])
            status = "success" if complete and observables["candidates"] else "partial" if observables["candidates"] else "failed"
        geometries = prepared_contact_geometries(observables, source, partner_source, workdir, request) \
            if request.route in {"search_dimer_contacts", "scan_intermolecular_distance", "compare_contact_torsion_response", "sample_conformers_crest"} else []
        timing = {**job.timing(), "preflight_seconds": job.started-started,
                  "adapter_wall_seconds": time.monotonic()-started}
        timing["noncompute_seconds"] = timing["total_wall_seconds"]-timing["compute_wall_seconds"]
        payload = {"backend": backend, "tool_args": values, "parsed_observables": observables,
                   "timing": timing, "molecular_context": {"smiles": case_run.input.smiles,
                   "source_artifact_id": source.artifact_id, "charge": source.metadata['charge'], "multiplicity": 1,
                   "symbols_in_source_order": atoms.get_chemical_symbols()},
                   "limitations": observables["limitations"]}
        if geometries:
            payload["modeled_systems"] = [{"artifact_id": a.artifact_id, "smiles": a.metadata["input_smiles"],
                "atom_count": a.metadata["atom_count"], "fragment_atom_indices": a.metadata["fragment_atom_indices"],
                "scope": a.metadata["geometry_scope"]} for a in geometries]
        manifest = workdir / "result.json"
        manifest.write_text(json.dumps(payload, indent=2, allow_nan=False))
        result_id = f"{request.dispatch_id}:{backend}:{workdir.name}"
        artifact = ArtifactRecord(artifact_id=result_id, kind=f"{request.route}_bundle",
            created_by_round=request.round_id, created_by_agent="microscopic",
            status="available" if status == "success" else "partial" if status == "partial" else "failed",
            file_paths={"manifest": str(manifest), "process_records": str(workdir / "process_records.json")},
            metadata=payload, reusable_for=["artifact_inspection", "microscopic"])
        evidence = EvidenceUnit(evidence_id=result_id+":E001", round_id=request.round_id,
            source_report_id=result_id, agent_name="microscopic", capability_id=request.capability_id,
            claim=observables["interpretation"], summary=observables["interpretation"], context=observables["method"], basis="computed",
            support="not_applicable", limits=observables["limitations"], observable=request.route,
            family=request.evidence_goal_family, relation="unknown", status="present" if status == "success" else "partial" if status == "partial" else "failed",
            artifact_refs=list(dict.fromkeys([source.artifact_id, artifact.artifact_id]
                + ([args.partner_artifact_id] if request.route == "search_dimer_contacts" else []))), metrics=payload)
        return ToolExecutionResult(tool_result_id=result_id, round_id=request.round_id,
            agent_name="microscopic", dispatch_id=request.dispatch_id, capability_id=request.capability_id,
            selected_route=request.route, tool_calls=[f"{backend}.{request.route}"],
            raw_results={"manifest": str(manifest), "workdir": str(workdir), "steps": job.records}, structured_results=payload,
            status=status, evidence_units=[evidence], artifact_updates=[artifact, *geometries], summary=observables["interpretation"],
            failure=member_failures(observables, status))
    except (QCError, OSError, ValueError, IndexError) as exc:
        code = exc.code if isinstance(exc, QCError) else "parse_failed"
        status = {"unsupported": "unsupported", "missing_input": "precondition_missing"}.get(code, "failed")
        result_id = f"{request.dispatch_id}:{backend}:failed"
        boundary = EvidenceUnit(evidence_id=result_id+":status", round_id=request.round_id,
            source_report_id=result_id, agent_name="microscopic", capability_id=request.capability_id,
            claim=f"No computed result: {exc}", summary=f"No computed result: {exc}", context="tool availability or execution status",
            basis="checklist", support="unresolved", limits=["Status only; no scientific support was produced."],
            observable=request.route, family=request.evidence_goal_family, relation="unknown",
            status={"unsupported": "unsupported", "precondition_missing": "missing"}.get(status, "failed"))
        return ToolExecutionResult(tool_result_id=result_id, round_id=request.round_id,
            agent_name="microscopic", dispatch_id=request.dispatch_id, capability_id=request.capability_id,
            selected_route=request.route, status=status, evidence_units=[boundary], artifact_updates=[],
            raw_results={"workdir": str(job.workdir) if job else None, "steps": job.records if job else []},
            structured_results={"boundary_only": True, "failure_code": code,
                                "timing": job.timing() if job else {"total_wall_seconds": time.monotonic()-started}},
            summary=f"No scientific result: {exc}", failure=tool_failure(code, str(exc),
                details={"workdir": str(job.workdir) if job else None}))
