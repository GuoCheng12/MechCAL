"""Explicit, atom-preserving export of a converged computed geometry."""

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from rdkit import Chem

from mechcal.schemas import ArtifactRecord, EvidenceUnit, ToolExecutionResult
from mechcal.schemas.tool_arguments import GeometrySelectionArgs
from mechcal.tools.amesp_baseline import _read_xyz_symbols_and_coordinates, _write_xyz
from mechcal.tools.amesp_microscopic import AmespMicroscopicError, EXCITED_FORCE_BOUNDARY


def geometry_selection_input(manifest, source_artifact_id, member_id=None):
    """Read and validate the same source for discovery and actual selection."""
    artifacts = {a.artifact_id: a for a in manifest.artifacts}
    source = artifacts.get(source_artifact_id)
    if source is None or source.status not in {"available", "partial"}:
        raise AmespMicroscopicError(
            "precondition_missing", "Selected source artifact is unavailable in this case."
        )
    bundle = source.metadata.get("bundle", {})
    observations = bundle.get("parsed_observables", {})
    if source.kind == "excited_state_relaxation_bundle":
        raise AmespMicroscopicError("capability_unsupported", EXCITED_FORCE_BOUNDARY)
    elif source.kind == "relaxed_coordinate_scan_bundle":
        if observations.get("state_root") != 0:
            raise AmespMicroscopicError("capability_unsupported", EXCITED_FORCE_BOUNDARY)
        row = next(
            (
                r
                for r in observations.get("route_records", [])
                if (r.get("member_id") == member_id if member_id is not None
                    else r.get("geometry_converged") is True)
            ),
            None,
        )
        if row is None or row.get("geometry_converged") is not True:
            raise AmespMicroscopicError(
                "precondition_missing", "An explicit, converged scan member is required."
            )
        geometry_path = row.get("xyz_path")
    else:
        raise AmespMicroscopicError(
            "precondition_missing", "Source must be a relaxation or relaxed-scan bundle."
        )

    parents = [artifacts.get(i) for i in bundle.get("source_artifact_ids", [])]
    templates = [
        a for a in parents if a and a.kind == "prepared_structure" and a.status == "available"
    ]
    if len(templates) != 1 or not geometry_path:
        raise AmespMicroscopicError(
            "precondition_missing", "Missing unambiguous prepared topology or computed geometry."
        )
    template = templates[0]
    context = observations.get("molecular_context", {})
    for key in ("charge", "multiplicity"):
        if context.get(key) != template.metadata.get(key):
            raise AmespMicroscopicError(
                "precondition_missing", "Computed and source charge/multiplicity differ."
            )
    try:
        mol = Chem.MolFromMolFile(template.file_paths["sdf"], removeHs=False)
        symbols, coordinates = _read_xyz_symbols_and_coordinates(Path(geometry_path))
        if mol is None or [a.GetSymbol() for a in mol.GetAtoms()] != symbols:
            raise ValueError("Computed geometry atom sequence differs from source topology.")
        xyz = np.asarray(coordinates, dtype=float)
        if xyz.shape != (mol.GetNumAtoms(), 3) or not np.isfinite(xyz).all():
            raise ValueError("Computed coordinates are incomplete or non-finite.")
        if Chem.GetFormalCharge(mol) != context["charge"]:
            raise ValueError("Source topology charge differs from calculation charge.")
        if mol.GetNumConformers() != 1:
            raise ValueError("Source topology needs exactly one coordinate set.")
    except (OSError, KeyError, ValueError) as exc:
        raise AmespMicroscopicError(
            "precondition_missing", f"Cannot reuse selected geometry: {exc}"
        ) from exc
    return source, template, geometry_path, mol, xyz


def select_geometry(request, case_run, tool_args):
    args = GeometrySelectionArgs.model_validate(tool_args)
    source, template, geometry_path, mol, xyz = geometry_selection_input(
        case_run.artifact_manifest, args.source_artifact_id, args.member_id
    )
    try:
        conf = mol.GetConformer()
        for i, position in enumerate(xyz):
            conf.SetAtomPosition(i, position.tolist())
        unique = uuid4().hex
        directory = Path(template.file_paths["xyz"]).parent
        xyz_path = directory / f"selected_{unique}.xyz"
        sdf_path = directory / f"selected_{unique}.sdf"
        summary_path = directory / f"selected_{unique}.json"
        _write_xyz(
            xyz_path,
            label=f"{source.artifact_id}:{args.member_id}",
            symbols=[a.GetSymbol() for a in mol.GetAtoms()],
            coordinates=xyz.tolist(),
        )
        with Chem.SDWriter(str(sdf_path)) as writer:
            writer.write(mol)
    except (OSError, KeyError, ValueError) as exc:
        raise AmespMicroscopicError(
            "precondition_missing", f"Cannot reuse selected geometry: {exc}"
        ) from exc
    metadata = {
        k: template.metadata[k]
        for k in (
            "input_smiles",
            "canonical_smiles",
            "charge",
            "multiplicity",
            "heavy_atom_count",
            "atom_count",
        )
        if k in template.metadata
    }
    metadata.update(
        geometry_source="explicit_computed_selection",
        parent_artifact_id=source.artifact_id,
        parent_member_id=args.member_id,
        topology_artifact_id=template.artifact_id,
        source_geometry_xyz=str(geometry_path),
        xyz_path=str(xyz_path),
        sdf_path=str(sdf_path),
        summary_path=str(summary_path),
        atom_index_base=0,
    )
    summary_path.write_text(json.dumps(metadata, indent=2))
    artifact = ArtifactRecord(
        artifact_id=f"geometry:{unique}",
        kind="prepared_structure",
        created_by_round=request.round_id,
        created_by_agent="microscopic",
        file_paths={"xyz": str(xyz_path), "sdf": str(sdf_path), "summary": str(summary_path)},
        reusable_for=["macro", "microscopic"],
        observable_tags=["geometry_reuse"],
        metadata=metadata,
    )
    report_id = f"{request.round_id}:microscopic:{unique}"
    summary = (
        "Selected a converged computed geometry, preserving source topology and atom numbering."
    )
    evidence = EvidenceUnit(
        evidence_id=f"{report_id}:selection",
        round_id=request.round_id,
        source_report_id=report_id,
        agent_name="microscopic",
        capability_id=request.capability_id,
        claim=summary,
        context="geometry reuse",
        basis="checklist",
        support="not_applicable",
        summary=summary,
        family="geometry_precondition",
        relation="neutral",
        artifact_refs=[artifact.artifact_id],
        limits=[
            "Geometry selection adds no independent mechanistic evidence and does not verify a minimum."
        ],
    )
    return ToolExecutionResult(
        tool_result_id=report_id,
        round_id=request.round_id,
        agent_name="microscopic",
        dispatch_id=request.dispatch_id,
        capability_id=request.capability_id,
        selected_route=request.route,
        tool_calls=["geometry.select_computed_geometry"],
        status="success",
        evidence_units=[evidence],
        artifact_updates=[artifact],
        structured_results=metadata,
        summary=summary,
    )
