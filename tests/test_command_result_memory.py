"""Offline result delivery/retrieval acceptance; no model or chemistry calls."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from mechcal.runtime.command_loop import _apply, _comment, _execute, _local_error, planner_payload, save
from mechcal.runtime.command_memory import (
    REPORT_BYTES, WORKING_BYTES, json_bytes, remember_report, resolve_pointer, result_report, working_reads,
)
from mechcal.runtime.command_operations import OperationRejected, check_refs, protocol, put, read_records, records
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import ReadEvidence
from test_command_protocol import ScriptClient, case, command, computed, orchestrator, store


def result():
    return {"capability_id": "microscopic.test_fixture", "status": "success", "summary": "Fixture only.",
            "failure": {}, "structured_results": {
                "method_provenance": "Fixture method and gas-phase conditions.", "limitations": ["Not an experiment."],
                "parsed_observables": {"energy_eV": 2.8931, "oscillator_strength": .017,
                    "table": [{"state": i, "components": list(range(1000))} for i in range(30)]}},
            "evidence_units": [{"context": "Fixture conditions.", "limits": ["No causal conclusion."]}]}


def read(ref="r1", paths=None, **kw):
    return ReadEvidence(kind="read_evidence", records=[{"ref": ref, "paths": paths or [], **kw}])


def test_report_preserves_exact_values_conditions_and_full_raw_data():
    raw = result()
    before = deepcopy(raw)
    report = result_report("r1", raw)
    assert json_bytes(report) < REPORT_BYTES
    for field in report["fields"]:
        assert resolve_pointer(raw, field["path"]) == field["value"]
    fields = {f["path"]: f["value"] for f in report["fields"]}
    assert fields["/structured_results/parsed_observables/energy_eV"] == 2.8931
    assert fields["/structured_results/parsed_observables/oscillator_strength"] == .017
    assert fields["/structured_results/limitations"] == ["Not an experiment."]
    assert fields["/structured_results/method_provenance"] == raw["structured_results"]["method_provenance"]
    assert report["deferred"][0]["path"] == "/structured_results/parsed_observables/table"
    assert raw == before


def test_sizes_and_readable_report_are_known_before_any_query(tmp_path):
    c = put(case(), results={"r1": result()})
    index = planner_payload(orchestrator(tmp_path), c)["result_index"][0]
    assert index["bytes"] > 24000
    entries = {e["path"]: e for e in index["directory"]}
    assert entries["/structured_results/parsed_observables/table"]["bytes"] > 24000
    assert entries["/structured_results/parsed_observables/energy_eV"]["bytes"] < 100
    report = read_records(c, read(index["report_ref"]))[0]["data"]
    assert report["source_result_id"] == "r1"
    with pytest.raises(OperationRejected, match="not a raw observation"):
        check_refs(c, [index["report_ref"]], [], scientific=True)


def test_directory_pagination_reaches_all_children_without_values():
    c = put(case(), results={"r1": {str(i): "secret-value" * 1000 for i in range(40)}})
    paths = []
    for offset in (0, 16, 32):
        page = read_records(c, read(view="directory", page={"offset": offset, "limit": 16}))[0]
        assert "secret-value" not in str(page)
        paths.extend(v["path"] for v in page["data"][""]["children"])
    assert paths == ["/" + str(i) for i in range(40)]
    assert page["data"][""]["next_offset"] is None
    beyond = read_records(c, read(view="directory", page={"offset": 100, "limit": 16}))[0]["data"][""]
    assert not beyond["children"] and beyond["omitted_children"] == 40


def test_list_pages_are_exact_and_keep_original_indices():
    raw = result()
    c = put(case(), results={"r1": raw})
    path = "/structured_results/parsed_observables/table"
    row = read_records(c, read(paths=[path], page={"offset": 5, "limit": 2}))[0]
    assert row["data"][path] == raw["structured_results"]["parsed_observables"]["table"][5:7]
    assert row["pages"][path] == {"offset": 5, "total_rows": 30, "next_offset": 7}
    with pytest.raises(OperationRejected) as err:
        read_records(c, read(paths=[path], page={"offset": 5, "limit": 12}))
    layout = err.value.issues[0]["details"]["record_layouts"][0]
    assert layout["children"][0]["path"] == path + "/0"
    assert layout["total_children"] == 30  # Directory addresses the source, not a renumbered slice.


@pytest.mark.parametrize("query,code", [
    (read(paths=["/summary"], page={"offset": 0, "limit": 1}), "not_a_list"),
    (read(paths=["/summary", "/status"], page={"offset": 0, "limit": 1}), "ambiguous_page"),
    (read(paths=["/structured_results/wrong"]), "unknown_path"),
    (read(ref="another-case"), "unknown_reference"),
])
def test_invalid_reads_are_local_and_do_not_mutate_science(tmp_path, query, code):
    c = put(case(), results={"r1": result()})
    before = c.model_dump(mode="json")
    updated = _apply(orchestrator(tmp_path), c, query, store(tmp_path, c))
    outcome = protocol(updated)["outcome"]
    assert outcome["status"] == "rejected" and outcome["issues"][0]["code"] == code
    assert protocol(updated)["revision"] == protocol(c)["revision"]
    assert protocol(updated)["results"] == protocol(c)["results"]
    assert c.model_dump(mode="json") == before


@pytest.mark.parametrize("page", [{"offset": -1, "limit": 2}, {"offset": 0, "limit": 0}, {"offset": 0, "limit": 33}])
def test_invalid_page_schema(page):
    with pytest.raises(ValidationError):
        read(page=page)


def test_oversize_feedback_prioritizes_actual_large_field():
    c = put(case(), results={"r1": {"small": 1, "large": "x" * 30000}})
    q = ReadEvidence(kind="read_evidence", records=[{"ref": "r1", "paths": ["/small"]}] * 5
                     + [{"ref": "r1", "paths": ["/large"]}])
    with pytest.raises(OperationRejected) as err:
        read_records(c, q)
    assert err.value.issues[0]["details"]["record_layouts"][0]["path"] == "/large"


def test_reads_survive_other_operations_errors_and_checkpoint(tmp_path):
    c = put(case(), results={"r1": result()})
    o, s = orchestrator(tmp_path), store(tmp_path, c)
    q = read(paths=["/structured_results/parsed_observables/energy_eV"])
    c = save(_apply(o, c, q, s), s)
    assert not planner_payload(o, c)["working_memory"]["findings"]  # Already in the tool message.
    c = _local_error(c, [{"path": ["test"], "code": "test", "message": "Fixture"}], s)
    restored = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    finding = planner_payload(o, restored)["working_memory"]["findings"][0]
    assert finding["data"] == 2.8931 and finding["ref"] == "r1"
    assert protocol(restored)["rounds"] == protocol(restored)["revision"] == 0
    assert protocol(restored)["results"]["r1"] == result()
    assert not planner_payload(o, case())["working_memory"]["findings"]


def test_repeated_reads_deduplicate_and_directory_does_not_displace_facts(tmp_path):
    c = put(case(), results={"r1": result()})
    o, s = orchestrator(tmp_path), store(tmp_path, c)
    for _ in range(3):
        c = _apply(o, c, read(paths=["/structured_results/parsed_observables/energy_eV"]), s)
    assert len(protocol(c)["read_memory"]) == 1
    c = _apply(o, c, read(view="directory"), s)
    assert len(protocol(c)["read_memory"]) == 1
    assert planner_payload(o, c)["working_memory"]["findings"][0]["data"] == 2.8931


def test_large_successful_read_becomes_bounded_source_addressable_findings(tmp_path):
    c = put(case(), results={"r1": result()})
    o, s = orchestrator(tmp_path), store(tmp_path, c)
    path = "/structured_results/parsed_observables/table"
    c = _apply(o, c, read(paths=[path], page={"offset": 5, "limit": 2}), s)
    c = put(c, outcome={"status": "other"})
    memory = planner_payload(o, c)["working_memory"]
    assert json_bytes(memory["findings"]) <= WORKING_BYTES
    for entry in memory["findings"]:
        assert entry["path"].startswith(path + "/5/") or entry["path"].startswith(path + "/6/")
        assert resolve_pointer(protocol(c)["results"]["r1"], entry["path"]) == entry["data"]
    assert memory["other_read_count"] > 0


def test_updated_or_removed_records_do_not_replay_stale_text(tmp_path):
    c = put(case(), arguments={"A1": {"label": "RIM_RIR_RIV", "refs": [], "text": "Old."}})
    c = _apply(orchestrator(tmp_path), c, read("A1", ["/text"]), store(tmp_path, c))
    c = put(c, outcome={}, arguments={"A1": {"label": "RIM_RIR_RIV", "refs": [], "text": "New."}})
    assert not planner_payload(orchestrator(tmp_path), c)["working_memory"]["findings"]
    c = put(c, arguments={})
    assert not planner_payload(orchestrator(tmp_path), c)["working_memory"]["findings"]


def test_failed_result_and_worker_interpretation_are_not_positive_evidence():
    raw = {**result(), "status": "failed", "failure": {"message": "Quality check failed."}}
    report = result_report("r1", raw)
    c = put(case(), results={"r1": raw})
    entries = remember_report([], report)
    memory = working_reads(entries, records(c))
    assert memory["findings"] and all(f["source_status"] == "failed" for f in memory["findings"])
    with pytest.raises(OperationRejected, match="Failed computation"):
        check_refs(c, ["r1"], [], scientific=True)


def test_worker_uses_compact_facts_and_delivery_is_not_lost_after_next_action(tmp_path):
    o, c = orchestrator(tmp_path), computed()
    s = store(tmp_path, c)
    c = _execute(o, _apply(o, c, command(), s), s)
    rid = protocol(c)["pending_result"]
    raw_before = deepcopy(protocol(c)["results"])
    writer = o.deps.workers["macro"].report_writer
    writer.enable_llm = True
    writer.llm_client = ScriptClient(["Bounded interpretation, not a mechanism conclusion."])
    c = _comment(o, c, s)
    worker_input = writer.llm_client.calls[0]["payload"]["result"]
    assert worker_input["source_result_id"] == rid and "fields" in worker_input
    assert "raw_results" not in worker_input
    assert "result_report" in protocol(c)["outcome"]
    assert not planner_payload(o, c)["working_memory"]["findings"]
    c = put(c, outcome={"status": "other"})
    findings = planner_payload(o, c)["working_memory"]["findings"]
    explanation = next(f for f in findings if f["ref"].startswith("explanation:"))
    assert explanation["source_result_id"] == rid
    assert protocol(c)["results"] == raw_before
    assert protocol(c)["revision"] == 0


def test_old_checkpoint_without_memory_or_reports_is_readable(tmp_path):
    c = put(case(), results={"r1": result()})
    old = CaseRun.model_validate_json(c.model_dump_json())
    assert not protocol(old).get("result_reports")
    assert read_records(old, read("report:r1"))[0]["data"]["source_result_id"] == "r1"
    assert not planner_payload(orchestrator(tmp_path), old)["working_memory"]["findings"]


def test_native_directory_read_update_and_final_delivery(tmp_path):
    from dataclasses import replace
    from mechcal.agents.command_tools import tool_contracts
    from test_command_protocol import REGISTRY, TOOL, judgment
    from test_native_commands import NativeFake, reply

    o = orchestrator(tmp_path)
    o.config = replace(o.config, command_native_tools=True)
    compute_name = next(t["name"] for t in tool_contracts(REGISTRY, [TOOL])[0] if t["name"].startswith("compute_"))
    fake = NativeFake([])
    seen = {}

    def next_step(**kw):
        fake.requests.append(kw)
        step = len(fake.requests)
        if step == 1:
            return reply("describe_tools", {"capability_ids": [TOOL]}, call_id="describe")
        if step == 2:
            return reply(compute_name, {"arguments": command().arguments.model_dump(), "purpose": "Fixture."}, call_id="compute")
        if step == 3:
            seen["rid"] = kw["outcome"]["result_id"]
            assert "result_report" in kw["outcome"]
            assert not kw["payload"]["working_memory"]["findings"]
            return reply(arguments={"records": [{"ref": seen["rid"], "paths": [], "view": "directory"}]}, call_id="directory")
        if step == 4:
            assert kw["payload"]["working_memory"]["findings"]
            return reply(arguments={"records": [{"ref": seen["rid"], "paths": ["/summary"]}]}, call_id="read")
        if step == 5:
            return reply("revise_judgment", judgment().model_dump(exclude={"kind"}), call_id="revise")
        assert step == 6
        assert any(f["ref"] == seen["rid"] and f["path"] == "/summary"
                   for f in kw["payload"]["working_memory"]["findings"])
        return reply("finalize", {"answer": "Scripted interface fixture, not a scientific finding.", "limitations": []}, call_id="finalize")

    fake.complete_tool_turn = next_step
    o.deps.planner.llm_client = fake
    final = o.run(smiles="CCO", user_query="Offline interface test.", case_id="memory-final")
    assert final.status == "finalized" and len(fake.requests) == 6
    assert protocol(final)["rounds"] == 1 and protocol(final)["revision"] == 1
    assert len(protocol(final)["results"]) == 1
    assert final.final_answer.mechanism_predictions[0].evidence_refs == ["input:smiles"]
