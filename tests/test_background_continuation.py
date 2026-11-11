"""Small task packets must retain exact facts and a route to deferred manuals."""

from copy import deepcopy
import json

import pytest

from mechcal.agents.command_tools import ask_tools
from mechcal.runtime.command_compact import expand_text, table_rows
from mechcal.runtime.command_context_memory import bind_scoped_context
from mechcal.runtime.command_loop import planner_payload, _apply
from mechcal.runtime.command_memory import digest, json_bytes
from mechcal.runtime.command_operations import protocol, put, read_records, record_index, check_refs, OperationRejected
from mechcal.runtime.command_task_context import prepare_planner, operation_guides
from mechcal.runtime.llm import OpenAICompatibleSettings
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import ReadContext, ReadEvidence
from mechcal.schemas.commands import UpdateHypothesesWithInvestigation
from test_command_task_context import ready
from test_command_protocol import REGISTRY, store
from test_native_commands import reply


def initial(tmp_path):
    c, o, s = ready(tmp_path)
    payload = expand_text(planner_payload(o, c, native=True))
    # Use the same scientific payload but exercise exploration rather than final editing.
    payload["exploration_state"]["new_result_ids"] = ["result:new"]
    return c, o, s, payload


def continuing(c, name="describe_tools"):
    exchange = reply(name, {"capability_ids": []})
    return put(c, latest_exchange=exchange, outcome={"status": "described", "tools": []})


def test_continuation_keeps_whole_directory_and_exact_tool_details(tmp_path):
    c, _, _, raw = initial(tmp_path)
    c, first = prepare_planner(c, raw)
    original_fields = deepcopy(protocol(c)["planner_context"]["fields"])
    current, second = prepare_planner(continuing(c), raw)
    first, second = expand_text(first), expand_text(second)
    assert first["input_phase"] == "decision" and second["input_phase"] == "query_continuation"
    assert protocol(current)["planner_context"]["fields"] == original_fields
    before, after = table_rows(first["tool_catalog"]), table_rows(second["tool_catalog"])
    assert [{k: r[k] for k in ("capability_id", "status", "purpose_preview")} for r in before] == [
        {k: r[k] for k in ("capability_id", "status", "purpose_preview")} for r in after]
    assert json_bytes(second["tool_catalog"]) < json_bytes(first["tool_catalog"])
    fields = original_fields["tool_catalog"]
    read = read_records(current, ReadEvidence(kind="read_evidence", records=[{
        "ref": "context:planner", "paths": ["/fields/tool_catalog"]}]))
    assert read[0]["data"]["/fields/tool_catalog"] == fields
    for name in ("exploration_state", "artifact_index", "ranking", "review_workflow", "working_memory"):
        assert second[name] == first[name]


def test_saved_manuals_are_exact_queryable_and_never_evidence(tmp_path):
    c, _, _, raw = initial(tmp_path)
    current, view = prepare_planner(c, raw)
    view = expand_text(view)
    assert set(view["operation_guide_names"]) == set(operation_guides(raw))
    for name, text in operation_guides(raw).items():
        path = "/fields/operation_guides/" + name
        result = read_records(current, ReadEvidence(kind="read_evidence", records=[{
            "ref": "context:planner", "paths": [path]}]))
        assert result[0]["data"][path] == text
        assert text not in json.dumps(view)
    with pytest.raises(OperationRejected):
        check_refs(current, ["context:planner"], ["refs"], scientific=True)
    assert current.portfolio == c.portfolio and current.evidence_ledger == c.evidence_ledger


def test_long_review_advice_is_exactly_retrievable_without_losing_findings(tmp_path):
    c, _, _, raw = initial(tmp_path)
    raw["review_feedback"] = {"status": "available", "text": "Review alternatives and source conditions. " * 150,
                              "findings": [{"id": "F1", "category": "fact", "text": "Wrong units"}]}
    current, view = prepare_planner(c, raw)
    feedback = expand_text(view)["review_feedback"]
    assert feedback["text_truncated"] and len(feedback["text_preview"]) == 480
    assert feedback["findings"] == raw["review_feedback"]["findings"]
    result = read_records(current, ReadEvidence(kind="read_evidence", records=[feedback["read"]]))
    assert result[0]["data"][feedback["read"]["paths"][0]] == raw["review_feedback"]["text"]


def test_snapshot_and_query_allowance_survive_continuation_and_restore(tmp_path):
    c, o, s, raw = initial(tmp_path)
    c, _ = prepare_planner(c, raw)
    c = _apply(o, c, ReadContext(kind="read_context", fields=["tool_catalog"]), s)
    used = deepcopy(protocol(c)["context_reads"])
    c = CaseRun.model_validate_json(c.model_dump_json())
    c, view = prepare_planner(continuing(c, "read_context"), raw)
    assert protocol(c)["context_reads"] == used
    assert used["snapshot"] == digest(protocol(c)["planner_context"])
    assert view["context_reads_remaining"] == 1


def test_compact_prompt_does_not_change_offered_operations(tmp_path):
    c, _, s, raw = initial(tmp_path)
    _, current = prepare_planner(c, raw)
    old = {**current, "task_context_version": 3}
    class Captured(Exception):
        pass
    class Client:
        settings = OpenAICompatibleSettings(base_url="https://api.deepseek.com", model="offline", api_key="offline")
        def complete_tool_turn(self, **kw):
            self.request = kw
            raise Captured()
    client = Client()
    with pytest.raises(Captured):
        ask_tools(client, REGISTRY, [], old, s)
    before = client.request
    with pytest.raises(Captured):
        ask_tools(client, REGISTRY, [], current, s)
    after = client.request
    assert before["tools"] == after["tools"]
    context = next(t for t in after["tools"] if t["name"] == "read_context")
    assert context["parameters"]["properties"]["fields"]["items"]["enum"] == sorted(
        entry["field"] for entry in current["deferred_context"])
    assert len(after["system_prompt"]) < .4 * len(before["system_prompt"])
    assert "hidden reference" in after["system_prompt"]
    assert "outside Top-3" in after["system_prompt"]
    assert "negative physical" in after["system_prompt"]


def test_scoped_continuation_defers_only_catalog_detail_and_preserves_source_scope(tmp_path):
    c, _, _, raw = initial(tmp_path)
    material = {"sections": {"tool_catalog": raw["tool_catalog"],
        "arguments": [{"text": "A current numerical claim", "refs": ["input:smiles"]}],
        "execution_context": [{"method": "method", "state": 1}],
        "review_workflow": {"issues": [{"id": "I1", "text": "Check conditions"}]}},
        "working_memory": {"findings": [{"ref": "input:smiles", "data": "known"}]}}
    index = record_index(c)
    first, _, _, state = bind_scoped_context(index, index.available, {"queries": 0}, material,
        role="coverage_reviewer", focus_refs={"input:smiles"}, store=store(tmp_path, c))
    state["exchange"] = reply()
    second, idx, scope, state = bind_scoped_context(index, index.available, state, material,
        role="coverage_reviewer", focus_refs={"input:smiles"}, store=store(tmp_path, c))
    assert json_bytes(second) < json_bytes(first)
    for field in ("arguments", "execution_context", "review_workflow"):
        assert second["sections"][field] == first["sections"][field]
    assert second["working_memory"] == first["working_memory"]
    assert set(index.available) <= set(scope)
    record = state["task_context_record"]
    assert record["fields"]["tool_catalog"] == raw["tool_catalog"]
    assert record["ref"] == first["task_memory"]["fields"][0]["ref"]


def test_provisional_contract_explains_its_own_changes_and_rejects_formal_edits():
    from pydantic import ValidationError

    schema = UpdateHypothesesWithInvestigation.model_json_schema()
    assert "no kind or status" in schema["properties"]["changes"]["description"]
    change = {"label": "ESIPT_PT", "confidence": None, "note": "Not yet assessed."}
    command = {"kind": "update_hypotheses", "note": "Retain uncertainty.", "changes": [change]}
    assert UpdateHypothesesWithInvestigation.model_validate(command).changes[0].confidence is None
    with pytest.raises(ValidationError):
        UpdateHypothesesWithInvestigation.model_validate({**command, "changes": [
            {"kind": "candidate", "label": "ESIPT_PT", "status": "deprioritized", "note": "Prior only."}]})


def test_compact_grouping_classifies_metadata_loading_without_changing_execution():
    from mechcal.runtime.command_action_policy import turn_contract

    names = {"read_evidence", "read_context", "describe_tools", "update_hypotheses",
             "delegate_computation", "compute_macro_example"}
    contract = turn_contract(names)
    assert contract["single_call_operations"] == ["delegate_computation", "describe_tools", "update_hypotheses"]
    assert set(contract["batch_read_operations"]) == {"read_evidence", "read_context"}
    assert contract["hypothesis_followup_controls"] == ["delegate_computation"]
    assert "including metadata queries" in contract["rule"]


def test_each_advertised_background_field_is_readable_via_context_command(tmp_path):
    c, o, s, raw = initial(tmp_path)
    raw["review_feedback"] = {"text": "Complete review advice. " * 100, "status": "available"}
    c, view = prepare_planner(c, raw)
    for entry in view["deferred_context"]:
        command = ReadContext(kind="read_context", fields=[entry["field"]])
        updated = _apply(o, c, command, s)
        assert protocol(updated)["context_reads"]["count"] == 1
        assert protocol(updated)["planner_context"] == protocol(c)["planner_context"]
    with pytest.raises(OperationRejected):
        check_refs(c, ["context:planner"], [], scientific=True)


def test_scoped_reviewer_also_receives_actual_query_grouping(tmp_path):
    from mechcal.agents.command_worker import scoped_commentary_turn
    from test_command_handoff import prose
    from test_native_commands import NativeFake

    c, _, s, _ = initial(tmp_path)
    client = NativeFake([prose("No further query is needed for this offline test.")])
    feedback, _ = scoped_commentary_turn(client, c, {"sections": {}}, s,
        scope={"input:smiles": record_index(c).available["input:smiles"]}, state={"queries": 0},
        role="coverage_reviewer", prompt_name="command_worker_reads.md", compact_context=True,
        describe_tools=lambda command: [])
    assert feedback["status"] == "available"
    groups = expand_text(client.requests[0]["payload"])["response_contract"]["call_groups"]
    assert groups["single_call_operations"] == ["describe_tools"]
    assert "describe_tools" not in groups["batch_read_operations"]
