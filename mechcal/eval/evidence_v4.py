"""Submission contract v4 and evaluator-side artifact records (evaluation v2).

Every evaluated method submits typed evidence claims. Computed claims cite artifact
records by ID and JSON pointer so that grounding can be checked by code. Artifact
records are method-agnostic: MechCAL tool results, ReAct/Codex tool calls or any
other recorder produce the same shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mechcal.mechanism_pool import MECHANISM_POOL

ClaimKind = Literal["structural", "computed", "prior"]
ArtifactStatus = Literal["success", "partial", "failed"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QuantitySource(_Model):
    artifact_id: str = Field(min_length=1)
    pointer: str = Field(pattern=r"^(/.*)?$", description="RFC 6901 JSON pointer into outputs")
    unit: str | None = Field(default=None, description="Unit of the artifact value, if not the claim unit")


class Quantity(_Model):
    name: str = Field(min_length=1)
    value: float
    unit: str = ""
    source: QuantitySource | None = None


class StructuralAssertion(_Model):
    """A deterministic, RDKit-checkable statement about the input graph (1-based atom ids)."""

    type: Literal["element", "aromatic", "in_ring", "bonded", "rotatable_bond", "conjugated_path"]
    atoms: list[int] = Field(min_length=1)
    element: str | None = None


class EvidenceClaim(_Model):
    statement: str = Field(min_length=1)
    kind: ClaimKind
    quantities: list[Quantity] = Field(default_factory=list)
    atom_ids: list[int] = Field(default_factory=list)
    assertions: list[StructuralAssertion] = Field(default_factory=list)


class PredictionV4(_Model):
    label: str
    rank: int = Field(ge=1, le=3)
    evidence: list[EvidenceClaim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _known_label(self):
        if self.label not in MECHANISM_POOL:
            raise ValueError(f"Unknown mechanism label: {self.label}")
        return self


class SubmissionV4(_Model):
    case_id: str
    mechanism_predictions: list[PredictionV4] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def _ranking(self):
        labels = [p.label for p in self.mechanism_predictions]
        if len(set(labels)) != len(labels):
            raise ValueError("Predicted labels must be distinct")
        if [p.rank for p in self.mechanism_predictions] != list(range(1, len(labels) + 1)):
            raise ValueError("Predictions must be ordered with ranks 1..n")
        return self


class ArtifactRecordV4(_Model):
    artifact_id: str
    case_id: str
    tool: str
    status: ArtifactStatus
    molecule_smiles: str = Field(description="Structure the calculation was run for")
    outputs: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


class ArtifactStore(_Model):
    case_id: str
    artifacts: dict[str, ArtifactRecordV4] = Field(default_factory=dict)

    def add(self, record: ArtifactRecordV4) -> None:
        if record.artifact_id in self.artifacts and self.artifacts[record.artifact_id] != record:
            raise ValueError(f"Conflicting artifact record: {record.artifact_id}")
        self.artifacts[record.artifact_id] = record


def resolve_pointer(document: Any, pointer: str) -> Any:
    """RFC 6901 JSON pointer resolution; raises KeyError when the path does not exist."""
    if pointer == "":
        return document
    current = document
    for raw in pointer.lstrip("/").split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if not token.isdigit() or int(token) >= len(current):
                raise KeyError(pointer)
            current = current[int(token)]
        elif isinstance(current, dict) and token in current:
            current = current[token]
        else:
            raise KeyError(pointer)
    return current


_STATUS_MAP = {"success": "success", "partial": "partial"}


def store_from_mechcal_run(run_dir: Path, case_id: str, smiles: str) -> ArtifactStore:
    """Convert persisted MechCAL ``ToolExecutionResult`` files into artifact records.

    The artifact ID is the tool_result_id; each evidence unit ID is also registered as an
    alias pointing at the same outputs so that either can be cited. Outputs are the
    structured results plus evidence-unit metrics, loaded through the snapshot reader so
    externalized arrays are restored losslessly.
    """
    from mechcal.runtime.snapshot_storage import decode_json
    from mechcal.schemas.tool_results import ToolExecutionResult

    root = Path(run_dir).resolve()
    store = ArtifactStore(case_id=case_id)
    for path in sorted((root / "rounds").glob("R*/*_tool_result.json")):
        record = ToolExecutionResult.model_validate(decode_json(path.read_bytes(), blob_root=root))
        outputs = {
            "structured_results": record.structured_results,
            "evidence_metrics": {e.evidence_id: e.metrics for e in record.evidence_units},
        }
        base = dict(case_id=case_id, tool=record.capability_id,
                    status=_STATUS_MAP.get(record.status, "failed"), molecule_smiles=smiles,
                    outputs=outputs,
                    provenance={"source_file": str(path.relative_to(root)), "round_id": record.round_id,
                                "dispatch_id": record.dispatch_id, "selected_route": record.selected_route})
        store.add(ArtifactRecordV4(artifact_id=record.tool_result_id, **base))
        for unit in record.evidence_units:
            store.add(ArtifactRecordV4(artifact_id=unit.evidence_id, **{
                **base, "outputs": {"metrics": unit.metrics, **outputs},
                "provenance": {**base["provenance"], "alias_of": record.tool_result_id}}))
    return store


def load_submission(path: Path) -> SubmissionV4:
    return SubmissionV4.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
