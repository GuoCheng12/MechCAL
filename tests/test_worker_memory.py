"""Worker receipt continuity; synthetic facts are not scientific acceptance."""

from copy import deepcopy

from mechcal.runtime import worker_tasks as tasks
from mechcal.runtime.command_loop import _execute
from mechcal.runtime.command_memory import (
    WORKING_BYTES, digest, json_bytes, remember_report, remember_values, result_report,
    resolve_pointer,
)
from mechcal.runtime.command_operations import protocol, put
from mechcal.runtime.command_reads import merge_handles
from mechcal.runtime.worker_memory import coalesce, receipt_handles, task_memory, directory_memory
from mechcal.schemas import CaseRun
from test_native_commands import reply
from test_worker_tasks import setup, start, step, load, compute_reply


def observation(value):
    return {"status": "success", "structured_results": {
        "method_provenance": "Recorded method details. " * 80,
        "limitations": ["Synthetic proxy; not an experiment."],
        "parsed_observables": {"condition": "same synthetic atom map",
            "measurements": {"a": value, "b": value + 1},
            "description": "Field description. " * 45}}}


def receipt(ref, raw):
    return {"status": raw["status"], "result_id": ref, "result_report": result_report(ref, raw)}


def test_computation_receipt_survives_query_error_and_resume(tmp_path):
    o, c, s, contract = setup(tmp_path)
    c = load(o, start(o, c, s, contract), s)
    c, _ = step(o, c, s, compute_reply())
    c = _execute(o, c, s)
    task = tasks.state(c)
    rid = task.result_ids[0]
    assert any(h["ref"] == rid for h in task.read_handles)
    before = deepcopy(protocol(c)["results"])
    immediate = tasks.payload(o, c, task)["working_memory"]["findings"]
    assert not any(f["ref"] == rid for f in immediate)  # Already in tool receipt.
    c, _ = step(o, c, s, reply("read_evidence", {"records": [{"ref": "foreign"}]}))
    restored = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    memory = tasks.payload(o, restored, tasks.state(restored))["working_memory"]
    assert any(f["ref"] == rid for f in memory["findings"])
    assert protocol(restored)["results"] == before
    assert protocol(restored)["phase"] == "worker_task"  # No automatic scientific completion.
    assert protocol(restored)["rounds"] == 1 and protocol(restored)["revision"] == 0


def test_two_sources_survive_other_read_without_parent_child_duplicates(tmp_path):
    o, c, s, contract = setup(tmp_path)
    c = start(o, c, s, contract)
    task = tasks.state(c)
    sources = {"baseline": observation(10), "new": observation(20)}
    for ref, raw in sources.items():
        task.read_handles = merge_handles(task.read_handles, receipt_handles(receipt(ref, raw)))
    task.read_handles = remember_values(task.read_handles, [(
        "baseline", "/structured_results/parsed_observables/measurements",
        {"a": 10, "b": 11}, None)])
    task.exchange = {"sent": True}
    task.outcome = {"status": "read", "records": [{"ref": "baseline", "data": {
        "/structured_results/parsed_observables/measurements": {"a": 10, "b": 11}},
        "selection": {"paths": ["/structured_results/parsed_observables/measurements"], "view": "data", "page": None}}]}
    memory = task_memory(task, sources)
    findings = memory["findings"]
    values = [f for f in findings if f["path"] == "/structured_results/parsed_observables"]
    assert {f["ref"] for f in values} == {"baseline", "new"}
    for f in findings:
        assert resolve_pointer(sources[f["ref"]], f["path"]) == f["data"]
        assert not any(other is not f and other["ref"] == f["ref"] and
                       other["path"].startswith(f["path"] + "/") for other in findings)
    assert json_bytes(findings) <= WORKING_BYTES
    for row in values:
        context = {f["path"]: f for f in row["source_context"]}
        method = context["/structured_results/method_provenance"]
        assert method["deferred"] and method["read"]["ref"] == row["ref"]
        assert context["/structured_results/limitations"]["data"] == ["Synthetic proxy; not an experiment."]


def test_reused_result_retained_without_new_execution(tmp_path):
    o, c, s, contract = setup(tmp_path)
    c = load(o, start(o, c, s, contract), s)
    c, _ = step(o, c, s, compute_reply())
    c = _execute(o, c, s)
    c = tasks.retain(c, tasks.state(c).model_copy(update={"read_handles": []}))
    c, _ = step(o, c, s, compute_reply())
    task = tasks.state(c)
    assert task.outcome["status"] == "reused" and task.outcome["executed"] is False
    assert task.read_handles and task.tool_attempts == 1 and protocol(c)["rounds"] == 1


def test_receipts_retain_only_accepted_values_not_directories_or_rejected_fields():
    outcome = {"status": "read_batch", "call_results": {
        "partial": {"records": [{"ref": "r", "status": "partial_read", "data": {"/a": 7},
            "selection": {"paths": ["/a"], "view": "data", "page": None},
            "requested_selection": {"paths": ["/a", "/absent"]}}]},
        "directory": {"records": [{"ref": "r", "data": {"": {"path": "/"}},
            "selection": {"paths": [], "view": "directory", "page": None}}]},
        "bad": {"status": "rejected", "issues": []}}}
    handles = receipt_handles(outcome)
    assert [(h["ref"], h["path"], h["sha256"]) for h in handles] == [("r", "/a", digest(7))]


def test_coalesce_does_not_confuse_sources_pages_or_stale_parents():
    raw = {"a": {"x": 1, "y": 2}, "b": {"x": 1, "y": 2}, "rows": [0, 1, 2, 3]}
    sources = {"r1": raw, "r2": deepcopy(raw)}
    entries = remember_values([], [(ref, path, resolve_pointer(raw, path), None)
        for ref, path in [("r1", "/a"), ("r1", "/a/x"), ("r1", "/b"), ("r2", "/a")]])
    entries = remember_values(entries, [("r1", "/rows", [1, 2], {"offset": 1, "limit": 2}),
                                        ("r1", "/rows", [3], {"offset": 3, "limit": 1})])
    before = deepcopy((entries, sources))
    kept = coalesce(entries, sources)
    assert len(kept) == 5 and not any(h["path"] == "/a/x" for h in kept)
    assert [h["page"]["offset"] for h in kept if h["page"]] == [1, 3]
    assert (entries, sources) == before
    sources["r1"]["a"]["y"] = 8
    kept = coalesce(entries, sources)
    assert any(h["ref"] == "r1" and h["path"] == "/a/x" for h in kept)
    assert not any(h["ref"] == "r1" and h["path"] == "/a" for h in kept)
    assert not coalesce(entries, {})


def test_failed_report_remains_failed_and_cannot_become_complete(tmp_path):
    o, c, s, contract = setup(tmp_path)
    c = start(o, c, s, contract)
    task = tasks.state(c)
    raw = {"status": "failed", "capability_id": contract.allowed_capability_ids[0],
           "summary": "Synthetic failure.", "structured_results": {"value": 1}}
    task.read_handles = remember_report([], result_report("failed", raw))
    task.result_ids = ["failed"]
    memory = task_memory(task, {"failed": raw})
    assert memory["findings"] and all(f["source_status"] == "failed" for f in memory["findings"])
    task.contract.input_refs = ["failed"]
    c = put(tasks.retain(c, task), results={"failed": raw})
    c, _ = step(o, c, s, reply("complete_worker_task", {
        "status": "complete", "refs": ["failed"], "summary": "Should not pass."}))
    assert tasks.state(c).outcome["issues"][0]["code"] == "no_usable_result"


def test_new_task_has_no_previous_task_reads_or_loaded_tools(tmp_path):
    o, c, s, contract = setup(tmp_path)
    c = load(o, start(o, c, s, contract), s)
    c, _ = step(o, c, s, reply())
    c, _ = step(o, c, s, reply("complete_worker_task", {
        "status": "blocked", "refs": [], "summary": "Offline boundary check."}))
    c = start(o, c, s, contract)
    task = tasks.state(c)
    assert not task.read_handles and not task.loaded_tools
    assert not tasks.payload(o, c, task)["working_memory"]["findings"]


def test_directory_memory_survives_data_read_and_checkpoint_without_becoming_evidence(tmp_path):
    o, c, s, contract = setup(tmp_path, refs=["source"])
    raw = {"fields": {str(i): "unread-secret" * 20 for i in range(20)}, "value": 3}
    c = start(o, put(c, results={"source": raw}), s, contract)
    c, _ = step(o, c, s, reply("read_evidence", {"records": [{"ref": "source", "paths": ["/fields"],
        "view": "directory", "page": {"offset": 5, "limit": 2}}]}))
    assert not tasks.state(c).read_handles
    c, _ = step(o, c, s, reply("read_evidence", {"records": [{"ref": "source", "paths": ["/value"]}]}))
    restored = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    task = tasks.state(restored)
    memory = directory_memory(task, tasks.scope(restored, task))
    assert len(memory["directories"]) == 1
    row = memory["directories"][0]
    assert [child[0] for child in row["children"]] == ["/fields/5", "/fields/6"]
    assert row["next_offset"] == 7 and row["total_children"] == 20
    assert "unread-secret" not in str(memory)
    assert not directory_memory(task, {})["directories"]
    changed = deepcopy(raw)
    changed["fields"]["5"] = "changed"
    assert not directory_memory(task, {"source": changed})["directories"]


def test_repeated_directory_reads_deduplicate_and_old_state_is_compatible(tmp_path):
    o, c, s, contract = setup(tmp_path)
    c = start(o, c, s, contract)
    for _ in range(3):
        c, _ = step(o, c, s, reply("read_evidence", {"records": [{"ref": "input:smiles", "view": "directory"}]}))
    assert len(tasks.state(c).directory_reads) == 1
    legacy = protocol(c)["worker_task"].copy()
    legacy.pop("directory_reads")
    legacy.pop("progress_note")
    assert tasks.state(put(c, worker_task=legacy)).directory_reads == []
    assert tasks.state(put(c, worker_task=legacy)).progress_note is None


def test_public_work_note_survives_intervening_reads_without_becoming_evidence(tmp_path):
    o, c, s, contract = setup(tmp_path)
    c = start(o, c, s, contract)
    response = reply(call_id="read-one")
    response["text"] = "I have checked the first input; next inspect the second input."
    response["items"].insert(0, {"type": "reasoning", "encrypted_content": "private-do-not-summarize"})
    c, _ = step(o, c, s, response)
    assert tasks.payload(o, c, tasks.state(c))["previous_work_note"] is None
    c, _ = step(o, c, s, reply(call_id="read-two"))
    restored = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    note = tasks.payload(o, restored, tasks.state(restored))["previous_work_note"]
    assert note["text"] == response["text"] and "private-do-not" not in str(note)
    assert "cannot be cited" in note["policy"]
    assert not protocol(restored)["results"] and not restored.evidence_ledger.items
    truncated = reply(call_id="bad", complete=False)
    truncated["text"] = "Not a complete response."
    c, _ = step(o, restored, s, truncated)
    assert tasks.state(c).progress_note["text"] == response["text"]
    c, _ = step(o, c, s, {**reply(call_id="revised"), "text": "New finding changes the next action."})
    assert tasks.state(c).progress_note["text"] == "New finding changes the next action."
