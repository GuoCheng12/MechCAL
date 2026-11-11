"""Interface acceptance, not tests of a model's scientific performance."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from mechcal.agents.command_tools import ask_tools, replay_exchange
from mechcal.agents.command_worker import worker_scope, worker_turn
from mechcal.runtime.command_delivery import reviewed
from mechcal.runtime.command_loop import _apply, _comment, planner_payload, save
from mechcal.runtime.command_operations import OperationRejected, protocol, put, read_records, records, revise
from mechcal.schemas import CaseRun, DispatchRequest
from mechcal.schemas.artifacts import ArtifactRecord
from mechcal.schemas.commands import Finalize, ReadBatch, ReadEvidence, SubmitFinal
from test_command_protocol import REGISTRY, ScriptClient, case, judgment, orchestrator, store
from test_native_commands import NativeFake, reply


def batch(*replies):
    result = deepcopy(replies[0])
    result["calls"] = [c for r in replies for c in r["calls"]]
    result["items"] = [c for r in replies for c in r["items"]]
    return result


def unchanged_science(before, after):
    for key in ("arguments", "ranking", "revision", "rounds", "results"):
        assert protocol(before).get(key) == protocol(after).get(key)
    for key in ("portfolio", "evidence_ledger", "artifact_manifest"):
        assert getattr(before, key) == getattr(after, key)


def test_every_visible_reference_is_directly_queryable(tmp_path):
    c = revise(case(), judgment())
    payload = planner_payload(orchestrator(tmp_path), c)
    assert payload["argument_index"][0]["ref"] == "A1"
    for name in ("argument_index", "evidence_index", "result_index", "artifact_index", "action_index", "explanation_index"):
        for entry in payload[name]:
            read_records(c, ReadEvidence(kind="read_evidence", records=[{"ref": entry["ref"]}]))
    with pytest.raises(OperationRejected):
        read_records(c, ReadEvidence(kind="read_evidence", records=[{"ref": "argument:A1"}]))


@pytest.mark.parametrize("bad", ["foreign:E1", "missing"])
def test_partial_read_retains_valid_data_and_only_remembers_success(tmp_path, bad):
    c = case()
    cmd = ReadEvidence(kind="read_evidence", records=[{"ref": bad}, {"ref": "input:smiles"}])
    updated = _apply(orchestrator(tmp_path), c, cmd, store(tmp_path, c))
    result = protocol(updated)["outcome"]
    assert result["status"] == "partial_read"
    assert result["records"][0]["status"] == "rejected"
    assert result["records"][1]["data"]["smiles"] == c.input.smiles
    assert all(h["ref"] == "input:smiles" for h in protocol(updated)["read_memory"])
    unchanged_science(c, updated)


@pytest.mark.parametrize("protocol_name", ["responses", "chat_completions"])
def test_batch_results_correlate_to_each_call_and_persist(tmp_path, protocol_name):
    c, o = case(), orchestrator(tmp_path)
    s = store(tmp_path, c)
    incoming = batch(reply(call_id="good"), reply(arguments={"records": [{"ref": "absent"}]}, call_id="bad"))
    incoming["protocol"] = protocol_name
    cmd, exchange = ask_tools(NativeFake([incoming]), REGISTRY, [], {}, s)
    assert isinstance(cmd, ReadBatch)
    pending = save(put(c, latest_exchange=exchange, pending_command={"read_batch": cmd.model_dump(mode="json")}), s)
    restored = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    parsed = ReadBatch.model_validate(protocol(restored)["pending_command"]["read_batch"])
    updated = save(_apply(o, restored, parsed, s), s)
    outputs = replay_exchange(exchange, protocol(updated)["outcome"], protocol_name)[-2:]
    get_id = lambda row: row.get("call_id", row.get("tool_call_id"))
    get_body = lambda row: json.loads(row.get("output", row.get("content")))
    assert [get_id(row) for row in outputs] == ["good", "bad"]
    assert get_body(outputs[0])["status"] == "read"
    assert get_body(outputs[1])["status"] == "rejected"
    unchanged_science(pending, updated)
    save(updated, s)
    assert len(list((s.root / "operation_results").glob("*.json"))) == 1


def test_malformed_read_call_does_not_cancel_other_read(tmp_path):
    incoming = batch(reply(call_id="a"), reply(call_id="b"))
    incoming["calls"][1]["arguments"] = '{"records":'
    c, o = case(), orchestrator(tmp_path)
    s = store(tmp_path, c)
    cmd, _ = ask_tools(NativeFake([incoming]), REGISTRY, [], {}, s)
    updated = _apply(o, c, cmd, s)
    results = protocol(updated)["outcome"]["call_results"]
    assert results["a"]["status"] == "read"
    assert results["b"]["issues"][0]["code"] == "invalid_json"


@pytest.mark.parametrize("fault", ["count", "selections", "mixed", "incomplete"])
def test_invalid_batch_is_never_partly_executed(tmp_path, fault):
    incoming = batch(reply(call_id="a"), reply(call_id="b"))
    if fault == "count":
        incoming = batch(*(reply(call_id=str(i)) for i in range(5)))
    elif fault == "selections":
        incoming = batch(*(reply(call_id=str(i), arguments={"records": [{"ref": "input:smiles"}] * 4}) for i in range(2)))
    elif fault == "mixed":
        incoming["calls"][1]["name"] = "revise_judgment"
    else:
        incoming["complete"] = False
    with pytest.raises(OperationRejected):
        ask_tools(NativeFake([incoming]), REGISTRY, [], {}, store(tmp_path, case()))


def test_receipt_call_id_mismatch_is_storage_error():
    with pytest.raises(RuntimeError, match="receipts"):
        replay_exchange(reply(), {"call_results": {"other": {}}}, "responses")


def draft_and_review(tmp_path, text="Final interpretation.", reply_text="Check limitations."):
    c, o = revise(case(), judgment()), orchestrator(tmp_path, reviews=True)
    s = store(tmp_path, c)
    c = _apply(o, c, Finalize(kind="finalize", answer=text), s)
    o.deps.scientific_reviewer.llm_client = ScriptClient([reply_text])
    return o, _comment(o, c, s), s


def test_submit_exact_reviewed_draft_without_rewriting(tmp_path):
    o, c, s = draft_and_review(tmp_path)
    draft = protocol(c)["final_draft"]
    assert reviewed(c, draft)
    assert draft["draft_id"] in records(c)
    finished = _apply(o, c, SubmitFinal(kind="submit_final", draft_id=draft["draft_id"]), s)
    assert finished.status == "finalized"
    assert finished.final_answer.model_dump(mode="json") == draft["content"]


def test_changed_wording_requires_new_review_even_at_same_revision(tmp_path):
    o, c, s = draft_and_review(tmp_path)
    old_id = protocol(c)["final_draft"]["draft_id"]
    changed = _apply(o, c, Finalize(kind="finalize", answer="Changed scientific explanation."), s)
    assert protocol(changed)["phase"] == "review" and changed.final_answer is None
    assert protocol(changed)["revision"] == protocol(c)["revision"]
    assert protocol(changed)["final_draft"]["draft_id"] != old_id
    assert not reviewed(changed, protocol(changed)["final_draft"])


def test_changed_argument_invalidates_saved_submission(tmp_path):
    o, c, s = draft_and_review(tmp_path)
    edit = judgment().model_copy(deep=True)
    edit.changes[0].text = "Updated bounded interpretation."
    changed = revise(c, edit)
    with pytest.raises(OperationRejected, match="changed"):
        _apply(o, changed, SubmitFinal(kind="submit_final", draft_id=protocol(c)["final_draft"]["draft_id"]), s)


def test_unavailable_review_is_not_completed_review(tmp_path):
    o, c, s = draft_and_review(tmp_path, reply_text="")
    assert protocol(c)["final_review"]["status"] == "unavailable"
    with pytest.raises(OperationRejected, match="no completed"):
        _apply(o, c, SubmitFinal(kind="submit_final", draft_id=protocol(c)["final_draft"]["draft_id"]), s)


def test_final_review_sees_exact_assembled_answer_and_no_full_tool_catalog(tmp_path):
    o, c, _ = draft_and_review(tmp_path)
    payload = o.deps.scientific_reviewer.llm_client.calls[0]["payload"]
    assert payload["assembled_answer"] == protocol(c)["final_draft"]["content"]
    assert "tool_catalog" not in payload
    assert "run_limits" in payload


def test_incremental_review_has_previous_and_changed_arguments(tmp_path):
    c, o = revise(case(), judgment()), orchestrator(tmp_path, reviews=True)
    s = store(tmp_path, c)
    update = judgment().model_copy(deep=True)
    update.changes[0].text = "Revised assessment, still a structural hypothesis."
    c = _apply(o, c, update, s)
    client = ScriptClient(["The revised wording is bounded."])
    o.deps.scientific_reviewer.llm_client = client
    _comment(o, c, s)
    payload = client.calls[0]["payload"]
    assert payload["previous_arguments"][0]["text"] == judgment().changes[0].text
    assert payload["arguments"][0]["text"] == update.changes[0].text


def worker_fixture():
    c = put(case(), pending_result="result:1", results={
        "result:1": {"case_id": case().case_id, "tool_result_id": "result:1", "status": "success",
                     "structured_results": {"observed_value": 1.25, "units": "eV"}},
        "other:1": {"status": "success", "value": 99}}, phase="explain")
    request = DispatchRequest(dispatch_id="fixture:dispatch", round_id="R001", agent_name="microscopic",
        capability_id="microscopic.run_baseline_bundle", route="run_baseline_bundle", task="Explain this calculation.",
        objective="Inspect recorded observations.", evidence_goal_family="state_ordering_brightness")
    return c, {"request": request.model_dump(mode="json"), "result": {"source_result_id": "result:1"}}


def prose(text="The observed value is 1.25 eV; it does not establish the mechanism."):
    return {"request_id": "worker-final", "protocol": "responses", "items": [], "calls": [],
            "complete": True, "text": text}


def test_worker_reads_deferred_fields_then_prose_without_mutating_science(tmp_path):
    c, payload = worker_fixture()
    client = NativeFake([reply(arguments={"records": [{"ref": "result:1", "paths": ["/structured_results"]}]}), prose()])
    s = store(tmp_path, c)
    feedback, session = worker_turn(client, c, payload, s)
    assert feedback is None and session["outcome"]["records"][0]["data"]["/structured_results"]["observed_value"] == 1.25
    saved = save(put(c, worker_session=session), s)
    restored = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    feedback, session = worker_turn(client, restored, payload, s)
    assert session is None and feedback["status"] == "available"
    assert feedback["read_record_ids"] == ["result:1"]
    assert feedback["refs"] == ["result:1"]
    unchanged_science(c, saved)


@pytest.mark.parametrize("ref", ["other:1", "foreign:result", "/etc/passwd"])
def test_worker_cannot_read_unrelated_records(tmp_path, ref):
    c, payload = worker_fixture()
    client = NativeFake([reply(arguments={"records": [{"ref": ref}]})])
    _, session = worker_turn(client, c, payload, store(tmp_path, c))
    assert session["outcome"]["issues"][0]["code"] == "out_of_scope"
    assert not session["read_record_ids"]
    assert "other:1" not in worker_scope(c, "result:1", payload["request"])


@pytest.mark.parametrize("name", ["compute_macro_macro_structure_scan", "revise_judgment", "finalize"])
def test_worker_cannot_write_or_compute(tmp_path, name):
    c, payload = worker_fixture()
    client = NativeFake([reply(name=name)])
    _, session = worker_turn(client, c, payload, store(tmp_path, c))
    assert session["outcome"]["issues"][0]["code"] == "unknown_tool"


def test_worker_exhausted_reads_cannot_discard_calculation(tmp_path):
    c, payload = worker_fixture()
    c = put(c, worker_session={"result_id": "result:1", "queries": 2})
    client = NativeFake([reply()])
    feedback, session = worker_turn(client, c, payload, store(tmp_path, c))
    assert feedback["status"] == "unavailable" and session is None
    assert client.requests[0]["tools"] == []
    assert protocol(c)["results"]["result:1"]["structured_results"]["observed_value"] == 1.25


def test_worker_session_cannot_cross_computations(tmp_path):
    c, payload = worker_fixture()
    c = put(c, worker_session={"result_id": "other:1", "queries": 0})
    with pytest.raises(RuntimeError, match="another computation"):
        worker_turn(NativeFake([]), c, payload, store(tmp_path, c))


def test_native_batch_full_loop_resume_does_not_repeat_compute(tmp_path):
    o = orchestrator(tmp_path)
    o.config = replace(o.config, command_native_tools=True)
    fake = NativeFake([batch(reply(call_id="one"), reply(call_id="two"))])
    original = fake.complete_tool_turn
    def stopping(**kw):
        response = original(**kw)
        (tmp_path / "batch-resume/STOP_REQUESTED").touch()
        return response
    fake.complete_tool_turn = stopping
    o.deps.planner.llm_client = fake
    c = o.run(smiles="CCO", user_query="Interface fixture", case_id="batch-resume")
    assert c.status == "paused"
    assert set(protocol(c)["outcome"]["call_results"]) == {"one", "two"}
    (tmp_path / "batch-resume/STOP_REQUESTED").unlink()
    fake = NativeFake([reply("revise_judgment", judgment().model_dump(mode="json", exclude={"kind"}), call_id="revise"),
                       reply("finalize", {"answer": "Offline fixture.", "limitations": []}, call_id="final")])
    o.deps.planner.llm_client = fake
    completed = o.run(smiles="CCO", user_query="Interface fixture", case_id="batch-resume", resume=True)
    assert completed.status == "finalized" and protocol(completed)["rounds"] == 0
    assert set(fake.requests[0]["outcome"]["call_results"]) == {"one", "two"}


def test_read_turn_size_limit_does_not_rewrite_source(tmp_path):
    c = put(case(), results={"r1": {"value": "a" * 15000}, "r2": {"value": "b" * 15000}})
    original = deepcopy(protocol(c)["results"])
    incoming = batch(reply(arguments={"records": [{"ref": "r1"}]}, call_id="one"),
                     reply(arguments={"records": [{"ref": "r2"}]}, call_id="two"))
    s = store(tmp_path, c)
    cmd, _ = ask_tools(NativeFake([incoming]), REGISTRY, [], {}, s)
    updated = _apply(orchestrator(tmp_path), c, cmd, s)
    outcomes = protocol(updated)["outcome"]["call_results"]
    assert outcomes["one"]["status"] == "read"
    assert outcomes["two"]["issues"][0]["code"] == "response_too_large"
    assert protocol(updated)["results"] == original


def test_worker_second_read_keeps_first_exact_small_fact(tmp_path):
    c, payload = worker_fixture()
    client = NativeFake([
        reply(arguments={"records": [{"ref": "result:1", "paths": ["/structured_results/observed_value"]}]}),
        reply(arguments={"records": [{"ref": "result:1", "paths": ["/structured_results/units"]}]}),
        prose(),
    ])
    s = store(tmp_path, c)
    for _ in range(2):
        _, session = worker_turn(client, c, payload, s)
        c = put(c, worker_session=session)
    worker_turn(client, c, payload, s)
    facts = client.requests[-1]["payload"]["working_memory"]["findings"]
    assert any(row["data"] == 1.25 for row in facts)
    assert not any(row["data"] == "eV" for row in facts)  # Already in the last tool response.


def test_failed_result_is_queryable_but_cannot_support_argument(tmp_path):
    c = put(case(), results={"failed": {"status": "failed", "diagnostic": "No convergence"}})
    updated = _apply(orchestrator(tmp_path), c,
                     ReadEvidence(kind="read_evidence", records=[{"ref": "failed"}]), store(tmp_path, c))
    assert protocol(updated)["outcome"]["records"][0]["data"]["diagnostic"] == "No convergence"
    j = judgment().model_copy(deep=True)
    j.changes[0].refs = ["failed"]
    with pytest.raises(OperationRejected, match="not scientific evidence"):
        revise(updated, j)


def test_worker_scope_includes_real_dispatch_input_refs_and_only_linked_artifacts():
    c, payload = worker_fixture()
    artifacts = [ArtifactRecord(artifact_id=ref, kind="fixture", created_by_round="R001",
                                created_by_agent="microscopic")
                 for ref in ("geometry:input", "geometry:output", "geometry:unrelated")]
    c = c.touch(artifact_manifest=c.artifact_manifest.with_records(artifacts))
    p = protocol(c)
    c = put(c, results={**p["results"], "result:1": {**p["results"]["result:1"],
        "artifact_updates": [artifacts[1].model_dump(mode="json")]}})
    request = DispatchRequest.model_validate({**payload["request"], "input_refs": ["geometry:input"]})
    scope = worker_scope(c, "result:1", request.model_dump(mode="json"))
    assert "geometry:input" in scope and "geometry:output" in scope
    assert "geometry:unrelated" not in scope
