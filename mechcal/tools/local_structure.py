from __future__ import annotations

import json
from pathlib import Path

from mechcal.chem.structure_prep import (
    StructurePrepError,
    StructurePrepRequest,
    prepare_structure_from_smiles,
)
from mechcal.schemas import ArtifactRecord, CaseInput
from mechcal.schemas.failures import FailureReport
from mechcal.utils.smiles import extract_smiles_features


class LocalStructureTool:
    def prepare(
        self,
        case_input: CaseInput,
        *,
        round_id: str,
        workspace: Path,
    ) -> tuple[ArtifactRecord, FailureReport]:
        structure_dir = workspace / "artifacts" / "structure"
        structure_dir.mkdir(parents=True, exist_ok=True)
        try:
            _, prepared = prepare_structure_from_smiles(
                StructurePrepRequest(
                    smiles=case_input.smiles,
                    label=case_input.case_id,
                    workdir=structure_dir,
                    allow_unoptimized_seed=True,
                )
            )
            initial_only = prepared.geometry_status == "embedded_unoptimized"
            artifact = ArtifactRecord(
                artifact_id="structure:initial" if initial_only else "structure:prepared",
                kind="initial_structure" if initial_only else "prepared_structure",
                created_by_round=round_id,
                created_by_agent="structure",
                status="available",
                file_paths={
                    "xyz": str(prepared.xyz_path),
                    "sdf": str(prepared.sdf_path),
                    "summary": str(prepared.summary_path),
                },
                observable_tags=[prepared.geometry_status, "geometry_precondition"],
                reusable_for=(
                    ["microscopic.optimize_ground_state_geometry"] if initial_only
                    else ["macro", "microscopic", "artifact_inspection"]
                ),
                metadata=prepared.model_dump(mode="json"),
            )
            if initial_only:
                return artifact, FailureReport(
                    kind="precondition_missing", recoverable=True,
                    message=("Force-field preparation unavailable. Preserved an unoptimized initial "
                             "geometry; Planner may request ground-state optimization before analysis."),
                    details={"code": "forcefield_failed", "reason": prepared.preparation_warning,
                             "recovery_capability": "microscopic.optimize_ground_state_geometry"},
                )
            return artifact, FailureReport()
        except StructurePrepError as exc:
            return self._fallback_structure_context(
                case_input,
                round_id,
                structure_dir,
                exc.to_payload(),
            )
        except Exception as exc:
            return self._fallback_structure_context(
                case_input,
                round_id,
                structure_dir,
                {"code": type(exc).__name__, "message": str(exc)},
            )

    def _fallback_structure_context(
        self,
        case_input: CaseInput,
        round_id: str,
        structure_dir: Path,
        error_payload: dict[str, str],
    ) -> tuple[ArtifactRecord, FailureReport]:
        features = extract_smiles_features(case_input.smiles)
        context_path = structure_dir / "smiles_structure_context.json"
        context = {
            "input_smiles": case_input.smiles,
            "structure_source": "smiles_only_fallback",
            "features": features.__dict__,
            "structure_prep_error": error_payload,
        }
        context_path.write_text(
            json.dumps(context, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        artifact = ArtifactRecord(
            artifact_id="structure:smiles_context",
            kind="smiles_structure_context",
            created_by_round=round_id,
            created_by_agent="structure",
            status="partial",
            file_paths={"summary": str(context_path)},
            observable_tags=[
                "smiles_structure_context",
                "smiles_only_fallback",
                "geometry_precondition",
            ],
            reusable_for=["macro"],
            metadata=context,
        )
        failure = FailureReport(
            kind="precondition_missing",
            message=(
                "Full 3D structure preparation was unavailable; MechCAL recorded a SMILES-only "
                "fallback context and will keep downstream failures typed."
            ),
            recoverable=True,
            details=error_payload,
        )
        return artifact, failure
