"""Bounded judge inputs selected from persisted execution records, not agent prose."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from mechcal.schemas.artifacts import ArtifactManifest
from mechcal.schemas.tool_results import ToolExecutionResult
from mechcal.schemas.mechanisms import INPUT_SMILES_REF
from mechcal.agents.observation_view import observation_view
from mechcal.runtime.snapshot_storage import decode_json

MAX_JUDGE_INPUT_BYTES: int | None = None
# Optional caller limits count prompts, schemas and repair feedback as well.
MAX_SOURCE_BYTES = MAX_JUDGE_INPUT_BYTES
DEDUPLICATION_THRESHOLD_BYTES = 200_000
PACKET_VERSION = "cited-tool-sources-v4-execution-refs"


class ToolSourcePacket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    packet_version: Literal["cited-tool-sources-v1", "cited-tool-sources-v2-coefficient-view",
                            "cited-tool-sources-v3-lossless-dedup",
                            "cited-tool-sources-v4-execution-refs"] = PACKET_VERSION
    requested_evidence_refs: list[str]
    unresolved_evidence_refs: list[str]
    execution_index: list[dict[str, Any]]
    tool_records: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    experimental_observations: list[dict[str, Any]]
    tool_action_count: int
    note: str


def json_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def check_judge_input_size(
    prompt: str,
    payload: dict[str, Any],
    response_format: dict[str, Any],
    schema_feedback: str | None = None,
    *,
    max_bytes: int | None = MAX_JUDGE_INPUT_BYTES,
) -> int:
    """Measure input bytes; enforce a limit only when explicitly supplied."""
    user_payload = dict(payload)
    if schema_feedback:
        user_payload["schema_feedback"] = schema_feedback
    size = len(prompt.encode("utf-8")) + json_bytes(user_payload) + json_bytes(response_format)
    if max_bytes is not None and size > max_bytes:
        raise ValueError(f"Judge input exceeds byte limit ({size} > {max_bytes}); not sent.")
    return size


def _read_json(path: Path, root: Path) -> tuple[Any, dict[str, str]]:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Source is outside the case run: {path.name}")
    data = resolved.read_bytes()
    return decode_json(data, blob_root=root), {
        "source_file": str(resolved.relative_to(root)),
        "source_sha256": hashlib.sha256(data).hexdigest(),
    }


def _without_file_paths(value: Any) -> Any:
    # Remove storage locations only. Keep every member, state, number and failure.
    if isinstance(value, dict):
        return {
            key: _without_file_paths(item)
            for key, item in value.items()
            if key not in {"file_paths", "initial_xyz", "evaluation_manifest"}
            and not key.endswith(("_path", "_paths"))
        }
    if isinstance(value, list):
        return [_without_file_paths(item) for item in value]
    return value


def deduplicate_packet(packet: ToolSourcePacket) -> ToolSourcePacket:
    """Replace exact repeated values with backward JSON pointers, never omit data."""
    seen = {}

    def visit(value, path):
        if not isinstance(value, (dict, list, str)):
            return value
        if isinstance(value, dict) and "packet_ref" in value:
            return value
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        # Small records stay inline for readability; only exact equality qualifies.
        if len(encoded.encode("utf-8")) >= 128:
            if encoded in seen:
                return {"packet_ref": seen[encoded]}
            seen[encoded] = path
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return [visit(item, path + f"/{index}") for index, item in enumerate(value)]
        return {key: visit(item, path + "/" + key.replace("~", "~0").replace("/", "~1"))
                for key, item in value.items()}

    data = packet.model_dump()
    # Preserve the public packet shape and source index. Only nested records compact.
    for section in ("tool_records", "artifacts"):
        data[section] = [visit(record, f"/{section}/{i}") for i, record in enumerate(data[section])]
    return ToolSourcePacket.model_validate(data)


def build_tool_source_packet(
    run_dir: Path, evidence_refs: list[str], *, max_bytes: int | None = MAX_SOURCE_BYTES
) -> ToolSourcePacket:
    root = run_dir.resolve()
    records: dict[tuple[str, str], tuple[ToolExecutionResult, dict[str, str]]] = {}
    for path in sorted((root / "rounds").glob("R*/*_tool_result.json")):
        raw, source = _read_json(path, root)
        record = ToolExecutionResult.model_validate(raw)
        identity = (record.dispatch_id, record.selected_route)
        if identity in records:
            if records[identity][0] != record:
                raise ValueError(f"Conflicting execution records: {identity}")
            continue
        records[identity] = (record, source)

    requested = list(dict.fromkeys(evidence_refs))
    ref_owners: dict[str, set[tuple[str, str]]] = {}
    for identity, (record, _) in records.items():
        for ref in [record.tool_result_id, *(e.evidence_id for e in record.evidence_units)]:
            ref_owners.setdefault(ref, set()).add(identity)
    for ref in requested:
        if len(ref_owners.get(ref, set())) > 1:
            raise ValueError(f"Ambiguous cited execution or evidence ID: {ref}")
    known_refs = set(ref_owners)
    cited_identities = {identity for ref in requested for identity in ref_owners.get(ref, set())}
    cited_routes = {
        record.capability_id for identity, (record, _) in records.items()
        if identity in cited_identities
    }
    index, details = [], []
    artifact_ids: set[str] = set()
    producers: dict[str, tuple[ToolExecutionResult, int]] = {}
    for identity, (record, source) in records.items():
        cited = identity in cited_identities
        selected = record.capability_id in cited_routes
        entry = {
            "tool_result_id": record.tool_result_id,
            "dispatch_id": record.dispatch_id, "round_id": record.round_id,
            "selected_route": record.selected_route, "status": record.status,
            "evidence_ids": [e.evidence_id for e in record.evidence_units],
            "detail_included": selected, **source,
        }
        index.append(entry)
        if not selected:
            continue
        structured = dict(record.structured_results)
        prior = structured.get("structural_prior", {})
        if prior == {structured.get("result_name"): structured.get("metrics")}:
            structured.pop("structural_prior", None)
        detail = {
            **entry, "selection_reason": "cited" if cited else "same_capability_attempt",
            "capability_id": record.capability_id, "tool_calls": record.tool_calls,
            "tool_args": record.raw_results.get("tool_args", {}),
            "step_records": record.raw_results.get("step_records", []),
            "structured_results": structured,
            "failure": record.failure.model_dump(),
            "evidence_context": [
                {"evidence_id": e.evidence_id, "basis": e.basis,
                 "context": e.context, "limits": e.limits, "artifact_refs": e.artifact_refs}
                for e in record.evidence_units
            ],
        }
        # Provenance is already present in the execution index, with no loss.
        detail.pop("source_file", None)
        detail.pop("source_sha256", None)
        detail.pop("detail_included", None)
        detail["source_record"] = {"packet_ref": f"/execution_index/{len(index) - 1}"}
        details.append(_without_file_paths(detail))
        for evidence in record.evidence_units:
            artifact_ids.update(evidence.artifact_refs)
        for artifact in record.artifact_updates:
            artifact_ids.add(artifact.artifact_id)
            producers[artifact.artifact_id] = (record, len(details) - 1)

    manifest_path = root / "artifacts/manifest.json"
    manifest, manifest_source = ({}, {})
    if manifest_path.exists():
        raw, manifest_source = _read_json(manifest_path, root)
        manifest = {a.artifact_id: (i, a) for i, a in enumerate(
            ArtifactManifest.model_validate(raw).artifacts
        )}
    artifacts, visited = [], set()
    while artifact_ids:
        artifact_id = min(artifact_ids)
        artifact_ids.remove(artifact_id)
        if artifact_id in visited:
            continue
        visited.add(artifact_id)
        if artifact_id not in manifest:
            artifacts.append({"artifact_id": artifact_id, "record_available": False})
            continue
        position, artifact = manifest[artifact_id]
        metadata = _without_file_paths(artifact.metadata)
        bundle = metadata.get("bundle", {})
        artifact_ids.update(bundle.get("source_artifact_ids", []))
        if artifact_id in producers:
            producer, detail_index = producers[artifact_id]
            # Equal duplicated fields become pointers; divergent fields remain visible.
            for key, original, target in (
                ("parsed_observables", producer.structured_results.get("parsed_observables"),
                 "structured_results/parsed_observables"),
                ("step_records", producer.raw_results.get("step_records"), "step_records"),
            ):
                if key in bundle and bundle[key] == _without_file_paths(original):
                    bundle[key] = {"packet_ref": f"/tool_records/{detail_index}/{target}"}
        files = {"available_count": 0, "missing": [], "outside_run": []}
        for name, filename in artifact.file_paths.items():
            path = Path(filename)
            path = (path if path.is_absolute() else root / path).resolve()
            if not path.is_relative_to(root):
                files["outside_run"].append(name)
            elif path.is_file():
                files["available_count"] += 1
            else:
                files["missing"].append(name)
        artifacts.append({
            "artifact_id": artifact_id, "kind": artifact.kind, "status": artifact.status,
            "record_available": True, "metadata": metadata, "raw_files": files,
            **manifest_source, "source_pointer": f"/artifacts/{position}",
        })

    packet = ToolSourcePacket(
        requested_evidence_refs=requested,
        unresolved_evidence_refs=[ref for ref in requested if ref not in known_refs and ref != INPUT_SMILES_REF],
        execution_index=index, tool_records=observation_view(details), artifacts=observation_view(artifacts),
        experimental_observations=[], tool_action_count=len(records),
        note=(
            "Citations resolve by exact tool_result_id or evidence_id, not operation IDs. "
            "Machine-recorded parameters/results for cited evidence and all attempts of those "
            "capabilities. Every scan member and failure is retained. Other executions are indexed "
            "but not expanded; missing detail does not mean no execution. Raw file contents are "
            "not included, and source hashes are provenance, not independent verification of parsers. "
            "Evidence context supplies scope/limitations, not an independent observation. "
            "These are computational records, not experimental measurements. "
            "Large orbital-coefficient tables show the eight largest absolute coefficients, "
            "with total/omitted row counts and full/shown coefficient squared sums. "
            "The full tables remain in the referenced source files at the same state_index. "
            "Omitted coefficients are not zero and cannot be checked individually from this view. "
            "All states, scan members, method conditions, spatial descriptors, atom contributions "
            "and failures are retained. Packet references resolve to other fields in this packet."
        ),
    )
    size = json_bytes(packet.model_dump())
    threshold = min(DEDUPLICATION_THRESHOLD_BYTES, max_bytes) if max_bytes is not None else DEDUPLICATION_THRESHOLD_BYTES
    if size > threshold:
        packet = deduplicate_packet(packet)
        size = json_bytes(packet.model_dump())
    if max_bytes is not None and size > max_bytes:
        raise ValueError(f"Tool source packet exceeds byte limit ({size} > {max_bytes}); not sent.")
    return packet
