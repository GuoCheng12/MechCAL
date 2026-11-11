from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from mechcal.eval.tool_sources import (
    build_tool_source_packet,
    check_judge_input_size,
    json_bytes,
)
from mechcal.runtime.persistence import RunStore
from mechcal.schemas import ToolExecutionResult


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def execution(round_id="R001", route="scan", *, status="success"):
    rid = f"{round_id}:{route}"
    return {
        "tool_result_id": rid, "round_id": round_id, "agent_name": "microscopic",
        "dispatch_id": rid, "capability_id": f"microscopic.{route}", "selected_route": route,
        "tool_calls": [route], "status": status,
        "raw_results": {"tool_args": {"angles_deg": [-90, 0, 90]}, "step_records": [
            {"method_label": "aTB1", "exit_code": 0, "terminated_normally": True,
             "aop_path": "/large/local/result.aop"}
        ]},
        "structured_results": {
            "parsed_observables": {"route_records": [
                {"member_id": "1", "status": "success", "oscillator_strength": 0.0},
                {"member_id": "2", "status": "failed", "error": "SCF did not converge"},
                {"member_id": "3", "status": "success", "oscillator_strength": 0.4},
            ], "missing_descriptors": ["charge_redistribution"], "minimum_verified": False},
            "limitations": ["Proxy only; no nonradiative rates."],
        },
        "failure": ({"kind": "partial_evidence", "message": "A point failed"}
                    if status != "success" else {}),
        "evidence_units": [{
            "evidence_id": f"{rid}:e", "round_id": round_id, "source_report_id": rid,
            "agent_name": "microscopic", "capability_id": f"microscopic.{route}",
            "claim": "Agent interpretation, not independent source",
            "context": "gas phase", "basis": "proxy", "summary": "Agent summary",
            "family": "torsion_sensitivity", "limits": ["No measured rates"],
            "artifact_refs": [],
        }],
    }


def save_execution(root, record, name=None):
    path = root / "rounds" / record["round_id"] / (name or f"{record['selected_route']}_tool_result.json")
    write_json(path, record)
    return path


def test_citations_keep_all_attempts_of_same_capability_and_complete_results(tmp_path):
    first = execution(status="partial")
    save_execution(tmp_path, first)
    save_execution(tmp_path, execution("R002"))
    save_execution(tmp_path, execution("R003", "unrelated"))
    packet = build_tool_source_packet(tmp_path, ["R001:scan:e"])
    assert packet.tool_action_count == 3
    assert len(packet.tool_records) == 2
    assert packet.execution_index[2]["detail_included"] is False
    record = packet.tool_records[0]
    assert record["structured_results"] == first["structured_results"]
    assert record["status"] == "partial"
    assert record["failure"]["kind"] == "partial_evidence"
    assert record["step_records"] == [
        {"method_label": "aTB1", "exit_code": 0, "terminated_normally": True}
    ]
    assert record["tool_args"] == first["raw_results"]["tool_args"]
    assert record["evidence_context"][0]["limits"] == ["No measured rates"]
    assert packet.tool_records[1]["selection_reason"] == "same_capability_attempt"
    assert "Agent summary" not in packet.model_dump_json()
    assert "Agent interpretation" not in packet.model_dump_json()


def test_missing_citation_does_not_expand_everything(tmp_path):
    save_execution(tmp_path, execution())
    packet = build_tool_source_packet(tmp_path, ["unknown", "unknown"])
    assert packet.requested_evidence_refs == ["unknown"]
    assert packet.unresolved_evidence_refs == ["unknown"]
    assert packet.tool_records == []
    assert packet.tool_action_count == 1


@pytest.mark.parametrize("refs", [["R001:scan"], ["R001:scan", "R001:scan:e"]])
def test_execution_and_evidence_citations_select_same_source(tmp_path, refs):
    record = execution()
    save_execution(tmp_path, record)
    save_execution(tmp_path, execution("R002"))
    packet = build_tool_source_packet(tmp_path, refs)
    assert packet.unresolved_evidence_refs == []
    assert len(packet.tool_records) == 2
    assert packet.tool_records[0]["tool_result_id"] == "R001:scan"
    assert packet.tool_records[0]["selection_reason"] == "cited"
    assert packet.tool_records[1]["selection_reason"] == "same_capability_attempt"
    assert packet.tool_records[0]["structured_results"] == record["structured_results"]


def test_failed_execution_citation_keeps_diagnostics_not_operation_alias(tmp_path):
    record = execution(status="failed")
    record["dispatch_id"] = "operation-id"
    save_execution(tmp_path, record)
    packet = build_tool_source_packet(tmp_path, ["R001:scan", "operation-id"])
    assert packet.unresolved_evidence_refs == ["operation-id"]
    assert packet.tool_records[0]["status"] == "failed"
    assert packet.tool_records[0]["evidence_context"][0]["evidence_id"] == "R001:scan:e"
    assert packet.tool_records[0]["failure"]["message"] == "A point failed"


def test_ambiguous_legacy_result_id_does_not_select_multiple_executions(tmp_path):
    first, second = execution(), execution("R002")
    first["tool_result_id"] = second["tool_result_id"] = "legacy-result"
    for record in (first, second):
        record["evidence_units"][0]["source_report_id"] = "legacy-result"
    save_execution(tmp_path, first)
    save_execution(tmp_path, second)
    with pytest.raises(ValueError, match="Ambiguous cited"):
        build_tool_source_packet(tmp_path, ["legacy-result"])
    assert build_tool_source_packet(tmp_path, ["R001:scan:e"]).unresolved_evidence_refs == []


def test_duplicate_alias_and_conflicting_alias(tmp_path):
    record = execution()
    save_execution(tmp_path, record)
    alias = save_execution(tmp_path, record, "generic_tool_result.json")
    assert build_tool_source_packet(tmp_path, ["R001:scan:e"]).tool_action_count == 1
    record["structured_results"]["limitations"].append("Different record")
    write_json(alias, record)
    with pytest.raises(ValueError, match="Conflicting"):
        build_tool_source_packet(tmp_path, ["R001:scan:e"])


def test_same_round_repeated_route_keeps_both_execution_records(tmp_path):
    store = RunStore(tmp_path, "case")
    store.initialize()
    refs = []
    for index in (1, 2):
        record = execution()
        rid = f"R001:scan:{index}"
        record.update(tool_result_id="legacy_route_result_id", dispatch_id=rid)
        record["evidence_units"][0].update(
            evidence_id=f"{rid}:e", source_report_id=record["tool_result_id"],
        )
        record["raw_results"]["tool_args"] = {"angles_deg": [0, 30 * index, 90]}
        refs.append(f"{rid}:e")
        store.write_tool_result(ToolExecutionResult.model_validate(record))
    packet = build_tool_source_packet(store.root, refs)
    assert packet.tool_action_count == 2
    assert packet.unresolved_evidence_refs == []
    assert {r["dispatch_id"] for r in packet.tool_records} == {"R001:scan:1", "R001:scan:2"}
    assert len(list((store.root / "rounds/R001").glob("execution_*_tool_result.json"))) == 2


def test_artifact_dependencies_keep_extra_states_without_loading_raw_logs(tmp_path, monkeypatch):
    record = execution()
    artifact = {
        "artifact_id": "bundle", "kind": "scan", "created_by_round": "R001",
        "created_by_agent": "microscopic", "status": "partial",
        "file_paths": {"aop": str(tmp_path / "large.aop"), "absent": "missing.xyz",
                       "external": str(tmp_path.parent / "external.log")},
        "metadata": {"charge": 0, "multiplicity": 1, "bundle": {
            "source_artifact_ids": ["structure:prepared", "not_registered"],
            "parsed_observables": copy.deepcopy(record["structured_results"]["parsed_observables"]),
            "step_records": copy.deepcopy(record["raw_results"]["step_records"]),
            "excited_states": [{"state_index": 1, "oscillator_strength": 0.0},
                               {"state_index": 2, "oscillator_strength": 0.2}],
        }},
    }
    record["artifact_updates"] = [artifact]
    save_execution(tmp_path, record)
    (tmp_path / "large.aop").write_text("RAW_LOG_NOT_SENT" * 10000)
    prepared = {
        "artifact_id": "structure:prepared", "kind": "prepared_structure",
        "created_by_round": "R000", "created_by_agent": "structure",
        "metadata": {"canonical_smiles": "CC", "charge": 0, "multiplicity": 1},
    }
    write_json(tmp_path / "artifacts/manifest.json", {
        "case_id": "CASE", "artifacts": [artifact, prepared],
    })
    read_bytes = Path.read_bytes

    def guarded_read(path):
        assert path.suffix == ".json", "Raw file contents must not be read"
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    packet = build_tool_source_packet(tmp_path, ["R001:scan:e"])
    artifacts = {a["artifact_id"]: a for a in packet.artifacts}
    bundle = artifacts["bundle"]
    assert bundle["metadata"]["bundle"]["excited_states"] == artifact["metadata"]["bundle"]["excited_states"]
    assert bundle["metadata"]["bundle"]["parsed_observables"] == {
        "packet_ref": "/tool_records/0/structured_results/parsed_observables"
    }
    assert bundle["raw_files"] == {
        "available_count": 1, "missing": ["absent"], "outside_run": ["external"]
    }
    assert bundle["status"] == "partial"
    assert artifacts["not_registered"]["record_available"] is False
    assert artifacts["structure:prepared"]["metadata"]["canonical_smiles"] == "CC"
    assert "RAW_LOG_NOT_SENT" not in packet.model_dump_json()


def test_missing_artifact_manifest_is_explicit(tmp_path):
    record = execution()
    record["evidence_units"][0]["artifact_refs"] = ["missing"]
    save_execution(tmp_path, record)
    packet = build_tool_source_packet(tmp_path, ["R001:scan:e"])
    assert packet.artifacts == [{"artifact_id": "missing", "record_available": False}]


def test_source_outside_run_is_rejected(tmp_path):
    outside = tmp_path / "outside.json"
    write_json(outside, execution())
    root = tmp_path / "run"
    path = root / "rounds/R001/scan_tool_result.json"
    path.parent.mkdir(parents=True)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="outside the case run"):
        build_tool_source_packet(root, ["R001:scan:e"])


def test_no_silent_truncation_on_large_source(tmp_path):
    record = execution()
    record["structured_results"]["many_states"] = list(range(20000))
    save_execution(tmp_path, record)
    with pytest.raises(ValueError, match="exceeds byte limit"):
        build_tool_source_packet(tmp_path, ["R001:scan:e"], max_bytes=64_000)


def test_large_source_has_no_default_limit_and_keeps_all_states(tmp_path):
    record = execution()
    states = list(range(50000))
    record["structured_results"]["many_states"] = states
    save_execution(tmp_path, record)
    packet = build_tool_source_packet(tmp_path, ["R001:scan:e"])
    assert json_bytes(packet.model_dump()) > 200_000
    assert packet.tool_records[0]["structured_results"]["many_states"] == states


def test_judge_guard_counts_utf8_and_retry_feedback():
    payload = {"evidence": "证据", "output_schema": {"x": "y"}}
    schema = {"schema": payload["output_schema"]}
    size = check_judge_input_size("prompt", payload, schema)
    assert size == 6 + json_bytes(payload) + json_bytes(schema)
    with pytest.raises(ValueError, match="not sent"):
        check_judge_input_size("prompt", payload, schema, "retry", max_bytes=size)
    assert "schema_feedback" not in payload


def test_empty_run_is_not_fabricated_evidence(tmp_path):
    packet = build_tool_source_packet(tmp_path, [])
    assert packet.tool_action_count == 0
    assert packet.tool_records == packet.artifacts == packet.experimental_observations == []


def test_packed_manifest_preserves_judge_sources_and_detects_missing_array(tmp_path):
    from mechcal.runtime.snapshot_storage import SnapshotStorage
    record = execution(status="partial")
    record["evidence_units"][0]["artifact_refs"] = ["bundle"]
    save_execution(tmp_path, record)
    path = tmp_path / "artifacts/manifest.json"
    manifest = {"case_id": "CASE", "artifacts": [{
        "artifact_id": "bundle", "kind": "scan", "created_by_round": "R001",
        "created_by_agent": "microscopic", "status": "partial",
        "metadata": {"values": [{"state_index": i, "energy": i / 1000, "status": "partial"}
                                for i in range(2000)]}}]}
    write_json(path, manifest)
    before = build_tool_source_packet(tmp_path, ["R001:scan:e"]).model_dump()
    SnapshotStorage(tmp_path, "CASE").write(path, manifest)
    after = build_tool_source_packet(tmp_path, ["R001:scan:e"]).model_dump()
    # Storage bytes (and their provenance hash) change; all scientific fields do not.
    for packet in (before, after):
        for artifact in packet["artifacts"]:
            artifact.pop("source_sha256", None)
    assert after == before
    next((tmp_path / "_payloads").glob("*.json")).unlink()
    with pytest.raises(FileNotFoundError):
        build_tool_source_packet(tmp_path, ["R001:scan:e"])
