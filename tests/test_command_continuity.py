"""Regressions for task-local memory and the 0005 delivery interruption."""

from copy import deepcopy

import pytest

from mechcal.agents.command_tools import parse_tool_turn, tool_contracts
from mechcal.runtime.command_memory import digest, json_bytes, remember_values, working_reads
from mechcal.runtime.command_reads import merge_handles
from mechcal.runtime.command_action_policy import action_policy
from mechcal.runtime.command_loop import _apply, planner_payload
from mechcal.runtime.command_operations import protocol, put, OperationRejected, record_index
from mechcal.runtime.command_task_context import prepare_planner
from mechcal.schemas.commands import ReadEvidence
from test_command_task_context import ready
from test_delivery_recovery import exhausted, science
from test_native_commands import reply
from mechcal.schemas.commands import AmendFinal, Finalize
from mechcal.runtime.command_delivery import amended_final, reviewed
from mechcal.runtime import command_review_workflow as workflow
from mechcal.runtime.command_delivery_review import fact_packets
from mechcal.runtime.command_review import review_projection
from mechcal.runtime.command_edit_context import edit_context
from test_command_context_efficiency import review_material, with_review
from test_command_protocol import case
from test_command_review_workflow import accept, report


def test_repeated_read_refreshes_recency_without_duplicates():
    handles = remember_values([], [("r", "/" + key, i, None) for i, key in enumerate("abc")])
    result = merge_handles(handles, [handles[0]])
    assert [h["path"] for h in result] == ["/b", "/c", "/a"]
    assert handles[0]["path"] == "/a"


def test_large_source_context_does_not_evict_exact_scan_table():
    table = [{"angle": i, "energy": i / 10, "note": "x" * 120} for i in range(20)]
    source = {"status": "success", "structured_results": {
        "method_provenance": "Long execution description. " * 200,
        "limitations": ["Ground-state proxy, not actual packing."], "table": table}}
    handles = remember_values([], [("r", "/structured_results/table", table, None)])
    # A real handoff preserves a whole 4 KB table, not only its leaf handles.
    handles = [{"ref": "r", "path": "/structured_results/table", "page": None,
                "sha256": digest(table), "bytes": json_bytes(table), "requested": True}]
    memory = working_reads(handles, {"r": source}, active_refs=["r"])
    assert memory["findings"][0]["data"] == table
    assert any(c.get("deferred") for c in memory["findings"][0]["source_context"])
    assert json_bytes(memory["findings"]) <= 6000


def test_explicit_recent_reads_take_precedence_over_old_mode_values():
    source = {"status": "success", "structured_results": {"modes": ["x" * 1250 for _ in range(6)]}}
    handles = remember_values([], [("r", f"/structured_results/modes/{i}", value, None)
                                   for i, value in enumerate(source["structured_results"]["modes"])])
    handles = merge_handles(handles, [handles[0], handles[2]])
    memory = working_reads(handles, {"r": source}, explicit_handles=handles)
    assert {h["path"] for h in memory["findings"]} >= {"/structured_results/modes/0", "/structured_results/modes/2"}
    assert all(h["visibility"] == "stored_not_shown" for h in memory["other_reads"])


def test_followup_is_readable_after_science_allowance_exhausted(tmp_path):
    c, o, store = ready(tmp_path)
    c = exhausted(c)
    before = science(c)
    assert "followup:state" in action_policy(c).workspace_refs
    c = _apply(o, c, ReadEvidence(kind="read_evidence", records=[{"ref": "followup:state", "view": "directory"}]), store)
    assert protocol(c)["outcome"]["status"] == "read"
    assert science(c) == before


def test_mixed_context_and_evidence_reads_keep_per_call_receipts(tmp_path):
    c, o, store = ready(tmp_path)
    c, _ = prepare_planner(c, planner_payload(o, c, native=True))
    tools, models = tool_contracts(o.deps.capability_registry, [], task_context=True)
    first = reply("read_context", {"fields": ["tool_catalog"]}, call_id="context")
    second = reply("read_evidence", {"records": [{"ref": "delivery:answer", "paths": ["/answer"],
                      "view": "data", "page": None}]}, call_id="answer")
    first["calls"].extend(second["calls"])
    batch = parse_tool_turn(first, tools, models)
    before = science(c)
    c = _apply(o, c, batch, store)
    results = protocol(c)["outcome"]["call_results"]
    assert results["context"]["status"] == results["answer"]["status"] == "read"
    assert protocol(c)["context_reads"]["count"] == 1
    assert science(c) == before


def test_field_replacement_is_atomic_version_bound_and_requires_review(tmp_path):
    c, o, store = ready(tmp_path)
    c = _apply(o, c, Finalize(kind="finalize", answer="Current scientific text.", limitations=["Old boundary."]), store)
    draft_id = protocol(c)["final_draft"]["draft_id"]
    before = science(c)
    changed = _apply(o, c, AmendFinal(kind="amend_final", draft_id=draft_id,
                    fields=[{"path": "/limitations/0", "value": "Corrected source boundary."}]), store)
    assert protocol(changed)["final_draft"]["command"]["answer"] == "Current scientific text."
    assert protocol(changed)["final_draft"]["command"]["limitations"] == ["Corrected source boundary."]
    assert science(changed) == before
    assert not reviewed(changed, protocol(changed)["final_draft"])
    with pytest.raises(OperationRejected):
        amended_final(changed, AmendFinal(kind="amend_final", draft_id=draft_id,
                         fields=[{"path": "/answer", "value": "Stale edit."}]))
    original = c.model_dump_json()
    for fields, replacements, limitations in [
        ([{"path": "/answer", "value": "Good"}, {"path": "/limitations/99", "value": "Bad"}], [], None),
        ([{"path": "/answer", "value": "One"}, {"path": "/answer", "value": "Two"}], [], None),
        ([{"path": "/answer", "value": "One"}], [{"old_text": "Current", "new_text": "Different"}], None),
        ([{"path": "/limitations/0", "value": "One"}], [], ["Whole replacement"]),
    ]:
        with pytest.raises(OperationRejected):
            amended_final(c, AmendFinal(kind="amend_final", draft_id=draft_id,
                          fields=fields, replacements=replacements, limitations=limitations))
        assert c.model_dump_json() == original


def test_native_final_edit_exposes_one_contract_but_legacy_storage_still_loads(tmp_path):
    _, o, _ = ready(tmp_path)
    tools, models = tool_contracts(o.deps.capability_registry, [], review_workflow=True,
                                  allowed_operations=["amend_final"])
    schema = tools[0]["parameters"]
    assert set(schema["properties"]) == {"draft_id", "fields"}
    assert set(schema["required"]) == {"draft_id", "fields"}
    parsed = parse_tool_turn(reply("amend_final", {"draft_id": "final:current",
        "fields": [{"path": "/limitations/0", "value": "Updated source condition."}]}), tools, models)
    assert isinstance(parsed, AmendFinal) and parsed.replacements == []
    assert parsed.fields[0].path == "/limitations/0"
    legacy = {"draft_id": "old", "replacements": [{"old_text": "before", "new_text": "after"}]}
    assert AmendFinal.model_validate({"kind": "amend_final", **legacy}).replacements
    with pytest.raises(OperationRejected):
        parse_tool_turn(reply("amend_final", legacy), tools, models)


def test_open_delivery_issue_supplies_current_edit_text_without_read(tmp_path):
    c, _, _ = ready(tmp_path)
    command = report(c, target="delivery:answer", focus={"path": "/answer", "quote": "A bounded comparison."})
    c = accept(c, command)
    view = edit_context(c)
    assert view["retained"][0]["data"] == "A bounded comparison."
    assert view["write_contract"]["draft_id"] == protocol(c)["final_draft"]["draft_id"]


def test_fact_packet_shares_planner_read_without_claiming_reviewer_checked_it():
    c = workflow.enable(case())
    raw = {"status": "success", "structured_results": {"fragments": [{"hole": 49, "electron": 51}]}}
    handles = remember_values([], [("r", "/structured_results/fragments", raw["structured_results"]["fragments"], None)])
    c = put(c, results={"r": raw}, planner_read_handles=handles)
    original = c.model_dump_json()
    packet = fact_packets(c, {"r"})["packets"][0]
    assert packet["value"] == [{"hole": 49, "electron": 51}]
    assert packet["visibility"] == "in_request"
    assert c.model_dump_json() == original
    assert not fact_packets(c, {"foreign"})["packets"]
    changed = put(c, results={"r": {**raw, "structured_results": {"fragments": []}}})
    assert not fact_packets(changed, {"r"})["packets"]


def test_reuse_bound_to_claim_sources_not_unrelated_results():
    c = put(workflow.enable(case()), results={"R1": {"status": "success", "value": 1}})
    payload = review_material()
    payload["arguments"][0]["refs"] = ["R1"]
    c, _, _ = with_review(c, payload)
    unrelated = put(c, results={**protocol(c)["results"], "OTHER": {"status": "success", "value": 9}})
    material, _, _ = review_projection(unrelated, payload, "final_reviewer", index=record_index(unrelated))
    assert "arguments:A1" in material["reused_sections"]
    changed = put(c, results={"R1": {"status": "success", "value": 2}})
    material, _, _ = review_projection(changed, payload, "final_reviewer", index=record_index(changed))
    assert "arguments:A1" not in material["reused_sections"]
    assert "arguments:A1" in material["changed_sections"]


def test_changed_raw_parent_invalidates_reuse_of_unchanged_evidence_unit():
    unit = {"evidence_id": "R1:E1", "source_report_id": "R1", "value": 1}
    raw = {"status": "success", "evidence_units": [unit], "method": "original"}
    c = put(workflow.enable(case()), results={"R1": raw})
    payload = review_material()
    payload["arguments"][0]["refs"] = ["R1:E1"]
    c, _, _ = with_review(c, payload)
    changed = put(c, results={"R1": {**raw, "method": "changed conditions"}})
    material, _, _ = review_projection(changed, payload, "final_reviewer", index=record_index(changed))
    assert "arguments:A1" not in material["reused_sections"]
