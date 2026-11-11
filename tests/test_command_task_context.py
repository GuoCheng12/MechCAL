"""Task routing is reversible and must not hide changes or approve science."""

from copy import deepcopy
from dataclasses import replace

import pytest

from mechcal.runtime.command_compact import expand_text, table_rows
from mechcal.runtime.command_loop import planner_payload, _apply, _comment
from mechcal.runtime.command_memory import json_bytes, resolve_pointer
from mechcal.runtime.command_operations import protocol, put, record_index, read_records, check_refs, OperationRejected
from mechcal.runtime.command_review import review_projection
from mechcal.runtime.command_task_context import prepare_planner, review_view, operation_guides
from mechcal.schemas.commands import ReadEvidence, ReadContext, Finalize
from test_command_context_efficiency import review_material, with_review
from test_command_review_workflow import setup
from test_command_protocol import case
from test_native_commands import NativeFake, reply
from mechcal.runtime import command_review_workflow as workflow


def ready(tmp_path):
    c, o, s = setup(tmp_path)
    o.config = replace(o.config, command_compact_context=True, command_task_context=True)
    c = _apply(o, c, Finalize(kind="finalize", answer="A bounded comparison."), s)
    return c, o, s


def test_deferred_fields_exact_readable_and_not_evidence(tmp_path):
    c, o, _ = ready(tmp_path)
    original = c.model_dump(mode="json")
    full = expand_text(planner_payload(o, c, native=True))
    routed, compact = prepare_planner(c, full)
    compact = expand_text(compact)
    assert compact["context_task"] == "delivery"
    assert c.model_dump(mode="json") == original
    assert json_bytes(compact) < json_bytes(full)
    assert set(full["mechanism_pool"]) == set(compact["mechanism_pool"])
    for entry in compact["deferred_context"]:
        rows = read_records(routed, ReadEvidence(kind="read_evidence", records=[
            {"ref": entry["ref"], "paths": [entry["path"]]}]))
        expected = operation_guides(full) if entry["field"] == "operation_guides" else full[entry["field"]]
        assert rows[0]["data"][entry["path"]] == expected
    with pytest.raises(OperationRejected):
        check_refs(routed, ["context:planner"], ["refs"], scientific=True)
    for key in ("ranking", "arguments", "results", "rounds", "revision", "final_draft"):
        assert protocol(routed).get(key) == protocol(c).get(key)


def test_context_case_isolation_and_refresh(tmp_path):
    c, o, _ = ready(tmp_path)
    c, view = prepare_planner(c, planner_payload(o, c, native=True))
    snapshot = protocol(c)["planner_context"]
    bad = put(c, planner_context={**snapshot, "case_id": "foreign"})
    with pytest.raises(RuntimeError, match="different case"):
        record_index(bad)
    value = expand_text(planner_payload(o, c, native=True))
    value["tool_catalog"]["molecular_input"] = "Changed directory metadata"
    current, _ = prepare_planner(c, value)
    assert protocol(current)["planner_context"] != snapshot
    assert protocol(c)["planner_context"] == snapshot


def test_delivery_keeps_source_provenance_and_failed_citation_permissions(tmp_path):
    c, o, _ = ready(tmp_path)
    full = expand_text(planner_payload(o, c, native=True))
    full["result_index"] = {"columns": ["ref", "tool", "status", "citation_uses"], "rows": [
        ["result:failed", "tool:soc", "failed", ["limitation"]],
        ["result:ok", "tool:geometry", "success", ["support", "limitation"]]]}
    full["evidence_index"] = {"columns": ["ref", "tool", "source_result_ref"], "rows": [
        ["evidence:one", "tool:geometry", "result:ok"]]}
    _, view = prepare_planner(c, full)
    view = expand_text(view)
    for key in ("result_index", "evidence_index"):
        assert view[key] == full[key]
        assert key not in [r["field"] for r in view["deferred_context"]]


def test_new_results_return_to_exploration_not_delivery(tmp_path):
    c, o, _ = ready(tmp_path)
    payload = expand_text(planner_payload(o, c, native=True))
    payload["exploration_state"]["new_result_ids"] = ["new:result"]
    current, view = prepare_planner(c, payload)
    view = expand_text(view)
    assert view["context_task"] == "exploration"
    assert [r["capability_id"] for r in table_rows(view["tool_catalog"])] == [
        r["capability_id"] for r in table_rows(payload["tool_catalog"])]
    assert protocol(current)["planner_context"]["fields"]["tool_catalog"] == payload["tool_catalog"]
    assert view["deferred_context"] and view["planner_interface"] == "full"
    assert view["exploration_state"]["new_result_ids"] == ["new:result"]


def test_repair_retains_draft_and_defers_discoverable_background(tmp_path):
    c, o, _ = ready(tmp_path)
    payload = expand_text(planner_payload(o, c, native=True))
    payload["judgment_draft"] = {"draft_id": "draft:one", "issues": [{"code": "missing_ref"}]}
    _, view = prepare_planner(c, payload)
    assert expand_text(view)["judgment_draft"] == payload["judgment_draft"]
    assert view["context_task"] == "repair" and view["deferred_context"]
    assert view["planner_interface"] == "repair"


def test_read_does_not_recursively_cache_previous_context(tmp_path):
    c, o, _ = ready(tmp_path)
    c, view = prepare_planner(c, planner_payload(o, c, native=True))
    snapshot = deepcopy(protocol(c)["planner_context"])
    for _ in range(3):
        full = expand_text(planner_payload(o, c, native=True))
        full["working_memory"] = {"previous_context": snapshot}
        c, _ = prepare_planner(c, full)
        assert protocol(c)["planner_context"] == snapshot


def test_exhausted_scientific_allowance_does_not_hide_context(tmp_path):
    c, o, s = ready(tmp_path)
    full = expand_text(planner_payload(o, c, native=True))
    full["exploration_state"]["query_turns_remaining"] = 0
    c, value = prepare_planner(c, full)
    value = expand_text(value)
    assert "tool_catalog" not in value and value["context_reads_remaining"] == 2
    queries = deepcopy(protocol(c)["exploration"])
    new = _apply(o, c, ReadContext(kind="read_context", fields=["tool_catalog"]), s)
    assert protocol(new)["exploration"] == queries
    assert protocol(new)["rounds"] == protocol(c)["rounds"]
    assert protocol(new)["context_reads"]["count"] == 1
    new, _ = prepare_planner(new, full)
    new = _apply(o, new, ReadContext(kind="read_context", fields=["ranking"]), s)
    _, value = prepare_planner(new, full)
    value = expand_text(value)
    assert "tool_catalog" not in value and "task_context_fallback" not in value
    assert value["deferred_context"] and value["context_reads_remaining"] == 0
    assert value["planner_interface"] == "full"
    assert protocol(new)["planner_context"]["fields"]["tool_catalog"] == full["tool_catalog"]


def test_bad_context_reads_consume_allowance_without_claim_mutation(tmp_path):
    c, o, s = ready(tmp_path)
    c, _ = prepare_planner(c, planner_payload(o, c, native=True))
    original = deepcopy(protocol(c)["arguments"])
    # This field is never in the deferred snapshot; ordinary operation error.
    c = put(c, planner_context={"case_id": c.case_id, "fields": {"ranking": []}})
    for expected in (1, 2):
        c = _apply(o, c, ReadContext(kind="read_context", fields=["tool_catalog"]), s)
        assert protocol(c)["outcome"]["status"] == "rejected"
        assert protocol(c)["context_reads"]["count"] == expected
    assert protocol(c)["arguments"] == original
    with pytest.raises(OperationRejected, match="Two context reads"):
        _apply(o, c, ReadContext(kind="read_context", fields=["ranking"]), s)


def test_context_tool_not_available_when_disabled_or_exhausted(tmp_path):
    from mechcal.agents.command_tools import ask_tools
    c, o, s = ready(tmp_path)
    for enabled in (False, True):
        payload = planner_payload(o, c, native=True)
        if enabled:
            c, payload = prepare_planner(c, payload)
        fake = NativeFake([reply("finalize", {"answer": "Bounded.", "limitations": []})])
        ask_tools(fake, o.deps.capability_registry, [], payload, s)
        assert ("read_context" in [t["name"] for t in fake.requests[0]["tools"]]) == enabled


def test_pending_impacts_remain_visible_during_delivery(tmp_path):
    c, o, _ = ready(tmp_path)
    full = expand_text(planner_payload(o, c, native=True))
    impacts = [{"id": "impact:one", "disposition": "deferred", "reason": "Unresolved comparison."}]
    full["exploration_state"]["evidence_followup"]["pending_impacts"] = impacts
    _, value = prepare_planner(c, full)
    assert expand_text(value)["exploration_state"]["evidence_followup"]["pending_impacts"] == impacts


def test_review_catalog_reuse_keeps_changed_conditions_and_raw_receipt():
    c = workflow.enable(case())
    payload = review_material()
    payload["tool_catalog"] = {"columns": ["capability_id", "description", "status"], "rows": [
        ["tool_" + str(i), "Description. " * 10, "ready"] for i in range(20)]}
    c, prior, _ = with_review(c, payload)
    new = deepcopy(payload)
    new["previous_arguments"] = payload["arguments"]
    new["arguments"][0]["text"] += " A new numerical assertion is 7.0."
    new["changes"] = [{"kind": "argument", **new["arguments"][0]}]
    material, receipt, _ = review_projection(c, new, "reviewer", index=record_index(c), task_context=True)
    assert "tool_catalog" in material["reused_sections"]
    assert "7.0" in str(material["sections"]["arguments:A1"])
    assert material["sections"]["changes"] == [{"kind": "argument", "id": "A1", "label": "RIM_RIR_RIV"}]
    assert "text" not in material["sections"]["previous_arguments"][0]
    assert receipt["sections"]["changes"] == new["changes"]
    assert receipt["sections"]["previous_arguments"] == payload["arguments"]
    assert resolve_pointer(prior, material["sections"]["tool_catalog"]["path"]) == payload["tool_catalog"]
    changed = deepcopy(new)
    changed["tool_catalog"]["rows"][0][-1] = "disabled"
    mat, _, _ = review_projection(c, changed, "reviewer", index=record_index(c), task_context=True)
    assert "tool_catalog" in mat["changed_sections"] and "disabled" in str(mat["sections"]["tool_catalog"])


def test_previous_intermediate_argument_is_not_lost():
    c = workflow.enable(case())
    old = review_material()
    c, _, _ = with_review(c, old)
    new = deepcopy(old)
    new["previous_arguments"] = [{**old["arguments"][0], "text": "An intermediate version never reviewed."}]
    new["arguments"][0]["text"] += " Another claim."
    mat, _, _ = review_projection(c, new, "reviewer", index=record_index(c), task_context=True)
    assert mat["sections"]["previous_arguments"] == new["previous_arguments"]


def test_optional_wording_grouping_does_not_close_findings_or_mutate_receipt():
    issues = [{"id": str(i), "category": kind} for i, kind in enumerate(("fact", "science", "wording"))]
    sections = {"review_workflow": {"issues": issues}}
    original = deepcopy(sections)
    mat = review_view({"sections": sections, "review_contract": "contract"}, sections)
    assert mat["sections"]["review_workflow"]["issue_groups"] == {
        "factual_corrections": ["0"], "scientific_questions": ["1"], "optional_wording": ["2"]}
    assert sections == original


def test_review_native_projection_accepts_reused_catalog(tmp_path):
    c, o, s = ready(tmp_path)
    client = NativeFake([reply("record_review", {"text": "Remaining uncertainty is scientific.", "findings": []}),
                         reply("record_review", {"text": "New wording reviewed.", "findings": []})])
    o.deps.scientific_reviewer.llm_client = client
    c = _comment(o, c, s)
    c = _apply(o, c, Finalize(kind="finalize", answer="A different bounded comparison."), s)
    c = _comment(o, c, s)
    view = expand_text(client.requests[-1]["payload"])
    assert view["task_context_version"] == 1
    assert protocol(c)["final_review"]["status"] == "available"
