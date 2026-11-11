"""Source identities and scoped reads must agree without weakening science commits."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest
from ase import Atoms

from mechcal.agents.command_worker import worker_turn
from mechcal.runtime.command_events import stop_reason
from mechcal.runtime.command_loop import _apply, _comment, planner_payload, save
from mechcal.runtime.command_memory import json_bytes, read_directory
from mechcal.runtime.command_operations import OperationRejected, protocol, put, record_identity, records, revise
from mechcal.runtime.command_reads import read_selection_results
from mechcal.runtime.command_review import review_observations
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import ReadEvidence
from mechcal.tools import xtb_backend
from test_command_handoff import prose, worker_fixture
from test_command_protocol import case, judgment, orchestrator, store
from test_native_commands import NativeFake, reply


def pending_case():
    return put(case(), experiments={"request:pending": {
        "status": "blocked", "command": {"kind": "compute", "capability_id": "microscopic.run_torsion_snapshots"},
        "issues": [{"code": "execution_limit", "message": "Not executed."}], "attempts": []}})


def test_pending_id_is_readable_limitation_but_never_positive_support(tmp_path):
    c = pending_case()
    payload = planner_payload(orchestrator(tmp_path), c)
    entry = payload["pending_experiments"][0]
    assert entry["ref"] == entry["experiment_id"] == "request:pending"
    assert entry["citation_uses"] == ["limitation"] and entry["record_kind"] == "operation"
    cmd = ReadEvidence(kind="read_evidence", records=[{"ref": entry["ref"]}])
    assert read_selection_results(c, cmd)[0]["records"][0]["data"]["status"] == "blocked"
    edit = judgment()
    edit.changes[0].limitation_refs = [entry["ref"]]
    assert protocol(revise(c, edit))["revision"] == 1
    edit.changes[0].refs = [entry["ref"]]
    with pytest.raises(OperationRejected) as exc:
        revise(c, edit)
    assert exc.value.issues[0]["code"] == "not_observation"
    assert exc.value.issues[0]["details"]["record_kind"] == "operation"
    assert protocol(c)["revision"] == 0


@pytest.mark.parametrize("fault", ["owner", "collision"])
def test_operation_storage_corruption_is_not_local_repair(fault):
    c = pending_case()
    if fault == "owner":
        protocol(c)["experiments"]["request:pending"]["case_id"] = "foreign-case"
    else:
        protocol(c)["arguments"] = {"request:pending": {"text": "collision"}}
    with pytest.raises(RuntimeError):
        records(c)


def test_partial_fields_keep_valid_data_and_exact_memory_but_citations_stay_atomic():
    c = put(case(), results={"result:1": {"status": "success", "structured_results": {"energy": 2.89}}})
    query = {"ref": "result:1", "paths": ["/structured_results/energy", "/structured_results/invented"]}
    outcome, handles = read_selection_results(c, ReadEvidence(kind="read_evidence", records=[query]))
    assert outcome["status"] == "partial_read"
    assert outcome["records"][0]["data"] == {"/structured_results/energy": 2.89}
    assert outcome["issues"][0]["path"] == ["arguments", "records", 0, "paths", 1]
    assert [h["path"] for h in handles] == ["/structured_results/energy"]
    edit = judgment()
    edit.changes[0].refs = ["result:1"]
    edit.changes[0].citations = ReadEvidence(kind="read_evidence", records=[query]).records
    before = c.model_dump()
    with pytest.raises(OperationRejected):
        revise(c, edit)
    assert c.model_dump() == before


def test_geometry_directory_advertises_real_metadata_not_guessed_fields():
    value = {**{f"metadata_{i}": i for i in range(10)}, "metadata": {"atom_count": 56}, "file_paths": {"xyz": "local"}}
    layout = read_directory(value)
    assert "/file_paths" in layout["paths"]
    assert layout["containers"][0]["paths"][0] == "/metadata/atom_count"
    assert "natoms" not in json.dumps(layout)


def test_contact_atoms_and_energies_are_self_describing(monkeypatch, tmp_path):
    monkeypatch.setattr(xtb_backend, "single", lambda *args: "TOTAL ENERGY -1.0 Eh")
    geometry = Atoms("AuH", positions=[[0, 0, 0], [0, 0, 3.0]])
    row = xtb_backend.contact_observables(None, geometry, 1, tmp_path)
    contact = row["contacts_under_3_5_angstrom"][0]
    assert (contact["atom_a"], contact["element_a"], contact["atom_b"], contact["element_b"]) == (0, "Au", 1, "H")
    assert contact["fragment_atom_b"] == 0 and contact["fragment_b"] == 1
    assert row["energy_component_order"] == ["complex", "fragment_a", "fragment_b"]
    assert row["fragment_energies_hartree"] == [-1, -1, -1]


def review_case():
    c = put(case(), results={"result:large": {"status": "success", "summary": "Exact data retained.",
        "structured_results": {"table": [{"value": i} for i in range(1000)], "units": "eV"},
        "raw_results": {"log": "x" * 100000}}})
    edit = judgment()
    edit.changes[0].refs = ["result:large"]
    return put(revise(c, edit), phase="review", review_role="reviewer", review_changes=[edit.changes[0].model_dump(mode="json")])


def test_reviewer_reads_full_deferred_data_then_prose_across_checkpoint(tmp_path):
    c, o = review_case(), orchestrator(tmp_path, reviews=True)
    o.config = replace(o.config, command_native_tools=True)
    client = NativeFake([reply(arguments={"records": [{"ref": "result:large",
        "paths": ["/structured_results/table/999/value"]}]}), prose("The saved last-row value is 999 eV.")])
    o.deps.scientific_reviewer.llm_client = client
    s = store(tmp_path, c)
    updated = save(_comment(o, c, s), s)
    assert protocol(updated)["phase"] == "review"
    assert protocol(updated)["reviewer_session"]["queries"] == 1
    material = client.requests[0]["payload"]
    observation = material["sections"]["observations:result:large"]
    assert json_bytes(observation) < 12000 and "raw_results" not in observation
    table = material["read_scope"]
    indexed = next(row for values in table["rows"]
                   if (row := {**table["defaults"], **dict(zip(table["columns"], values))})["ref"] == "result:large")
    assert "/structured_results" in material["read_path_sets"][indexed["path_set"]]
    restored = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    finished = _comment(o, restored, s)
    assert protocol(finished)["review"]["status"] == "available"
    assert protocol(finished)["review"]["read_record_ids"] == ["result:large"]
    for key in ("results", "ranking", "arguments", "revision", "rounds"):
        assert protocol(c)[key] == protocol(finished)[key]
    assert client.requests[1]["outcome"]["records"][0]["data"]["/structured_results/table/999/value"] == 999


def test_worker_invalid_field_does_not_waste_valid_sibling_and_no_query_reset(tmp_path):
    c, payload = worker_fixture()
    client = NativeFake([reply(arguments={"records": [{"ref": "result:1",
        "paths": ["/structured_results/observed_value", "/made_up"]}]}), prose()])
    s = store(tmp_path, c)
    _, session = worker_turn(client, c, payload, s)
    assert session["queries"] == 1 and session["outcome"]["status"] == "partial_read"
    c = put(c, worker_session=session)
    feedback, session = worker_turn(client, c, payload, s)
    assert feedback["status"] == "available" and session is None
    assert len(list((s.root / "events").glob("*.json"))) == 1


def test_reviewer_view_keeps_failed_status_and_scope_limits():
    c = review_case()
    available = records(c)
    available["result:large"]["status"] = "failed"
    available["result:large"]["failure"] = {"kind": "unconverged"}
    compact = review_observations(c, {"result:large"}, available)["result:large"]
    assert compact["status"] == "failed"
    assert any(f["path"] == "/failure" and f["value"]["kind"] == "unconverged" for f in compact["fields"])


def test_stop_reason_and_elapsed_freeze(monkeypatch, tmp_path):
    import mechcal.runtime.command_events as events
    c, s = case(), store(tmp_path, case())
    monkeypatch.setattr(events.time, "time", lambda: 100.)
    c = save(c, s)
    monkeypatch.setattr(events.time, "time", lambda: 110.)
    (s.root / "STOP_REQUESTED").write_text(json.dumps({"reason": "observed_tokens", "input_tokens": 100}))
    c = save(put(c.touch(status="paused"), interruption=stop_reason(s.root)), s)
    assert protocol(c)["clock"]["elapsed_seconds"] == 10
    assert protocol(c)["interruption"]["kind"] == "acceptance_budget"
    monkeypatch.setattr(events.time, "time", lambda: 1000.)
    c = save(c, s)
    assert protocol(c)["clock"]["elapsed_seconds"] == 10
    assert len(list((s.root / "events").glob("*.json"))) == 2


def test_old_checkpoints_remain_readable_without_record_migration():
    c = CaseRun.model_validate_json(pending_case().model_dump_json())
    before = deepcopy(c.model_dump())
    assert record_identity(c, "request:pending")["citation_uses"] == ["limitation"]
    assert c.model_dump() == before




def test_reviewer_transport_without_reads_keeps_legacy_data(tmp_path):
    from test_command_protocol import ScriptClient
    c, o = review_case(), orchestrator(tmp_path, reviews=True)
    client = ScriptClient(["Review the complete record supplied by the legacy text transport."])
    o.deps.scientific_reviewer.llm_client = client
    _comment(o, c, store(tmp_path, c))
    assert len(client.calls[0]["payload"]["observations"]["result:large"]["structured_results"]["table"]) == 1000
