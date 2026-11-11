"""Task memory reduces transmission, not evidence fidelity or read permissions."""

from copy import deepcopy
import json

import pytest

from mechcal.agents.command_worker import scoped_commentary_turn
from mechcal.runtime.command_compact import expand_text, table_rows
from mechcal.runtime.command_context import record_table
from mechcal.runtime.command_context_memory import project_scoped, bind_scoped_context, context_identity
from mechcal.runtime.command_memory import json_bytes, resolve_pointer
from mechcal.runtime.command_operations import check_refs, record_index, OperationRejected, put, protocol
from mechcal.runtime.command_read_handoff import selection_handles
from mechcal.runtime.command_reads import query_results
from mechcal.schemas.commands import ReadEvidence
from test_command_protocol import case, store
from test_command_handoff import prose
from test_native_commands import NativeFake, reply


def material(count=100):
    return {"role": "exploration_reviewer", "mechanism_pool": ["A", "B", "C"],
            "results": [{"source_result_id": "R0", "value": 1.25, "unit": "eV", "state": 2}],
            "private_note": "A current unresolved alternative, not a scientific fact.",
            "open_questions": [{"id": f"Q{i}", "labels": ["A"], "status": "open",
                "question": "A lengthy historical experiment design. " * 50,
                "conditions": {"state": i, "solvent": "water"}} for i in range(count)],
            "read_scope": record_table([{"ref": f"R{i}", "record_kind": "observation",
                "citation_uses": ["support", "limitation"], "path_set": "p"} for i in range(count)]),
            "read_path_sets": {"p": ["/status", "/structured_results"]}}


def sources(count=100):
    return put(case(), results={f"R{i}": {"status": "success", "structured_results": {"value": i}}
                                for i in range(count)})


def test_projection_exact_readback_bounded_growth_and_scientific_state(tmp_path):
    c = sources()
    original_case, original = c.model_dump(mode="json"), material()
    index = record_index(c)
    scope = {k: v for k, v in index.available.items() if k.startswith("R")}
    value, bound, permitted, state = bind_scoped_context(index, scope, {"queries": 0}, original,
        role="exploration_reviewer", focus_refs={"R0"}, store=store(tmp_path, c))
    record = state["task_context_record"]
    assert json_bytes(value) < .2 * json_bytes(original)
    assert value["results"] == original["results"] and value["private_note"] == original["private_note"]
    assert "R0" in [r["ref"] for r in table_rows(value["read_scope"])]
    assert len(value["read_scope"]["rows"]) <= 8
    for field, expected in record["fields"].items():
        assert expected == original[field]
    out, _ = query_results(c, ReadEvidence(kind="read_evidence", records=[{
        "ref": record["ref"], "paths": ["/fields/open_questions/61"]}]), allowed_refs=permitted, index=bound)
    assert out["records"][0]["data"]["/fields/open_questions/61"] == original["open_questions"][61]
    assert c.model_dump(mode="json") == original_case
    assert bound.observations == index.observations and record["ref"] not in index.available
    with pytest.raises(OperationRejected, match="not a raw observation"):
        check_refs(c, [record["ref"]], ["refs"], scientific=True, index=bound)
    assert list((store(tmp_path, c).root / "memory/task_contexts").glob("*.json"))
    large, _ = project_scoped(material(500), case_id=c.case_id, role="exploration_reviewer", focus_refs={"R0"})
    assert json_bytes(large) < 1.2 * json_bytes(value)


def test_context_read_cannot_expand_role_scope_or_cross_case():
    c = sources(2)
    _, record = project_scoped(material(), case_id=c.case_id, role="worker")
    index = record_index(c).with_contexts([record])
    out, _ = query_results(c, ReadEvidence(kind="read_evidence", records=[{"ref": "R1"}]),
                          allowed_refs={"R0", record["ref"]}, index=index)
    assert out["issues"][0]["code"] == "out_of_scope"
    other = {**record, "case_id": "foreign"}
    other["ref"] = context_identity(other)
    with pytest.raises(RuntimeError, match="ownership"):
        index.with_contexts([other])
    with pytest.raises(RuntimeError, match="identity"):
        index.with_contexts([{**record, "fields": {}}])


def test_role_memory_contains_only_provided_background():
    c = sources()
    c = put(c, ranking={"PRIVATE_PLANNER": {"text": "PRIVATE_CANARY"}}, arguments={})
    _, record = project_scoped(material(), case_id=c.case_id, role="exploration_reviewer")
    assert "PRIVATE_CANARY" not in json.dumps(record)
    assert "PRIVATE_PLANNER" not in json.dumps(record)


def test_prior_context_survives_smaller_view_but_never_crosses_roles(tmp_path):
    c = sources()
    index = record_index(c)
    _, _, _, state = bind_scoped_context(index, index.available, {}, material(),
        role="worker", focus_refs={"R0"}, store=store(tmp_path, c))
    previous = state["task_context_record"]
    _, bound, scope, _ = bind_scoped_context(index, index.available, state, {"result": {}},
        role="worker", focus_refs={"R0"}, store=store(tmp_path, c))
    assert bound.available[previous["ref"]] == previous
    assert previous["ref"] in scope
    with pytest.raises(RuntimeError, match="different case or role"):
        bind_scoped_context(index, index.available, state, {"result": {}},
            role="exploration_reviewer", focus_refs={"R0"}, store=store(tmp_path, c))


def test_latest_read_delivered_once_and_prior_read_survives_local_correction(tmp_path):
    c = sources()
    payload = {k: v for k, v in material().items() if k not in {"read_scope", "read_path_sets"}}
    s, scope = store(tmp_path, c), record_index(c).available
    # Learn the exact registered address from the first actual task view.
    class ReadThenDeliver(NativeFake):
        def complete_tool_turn(self, **kw):
            self.requests.append(kw)
            view = expand_text(kw["payload"])
            if len(self.requests) == 1:
                ref = next(x["ref"] for x in view["task_memory"]["fields"] if x["field"] == "open_questions")
                return reply("read_evidence", {"records": [{"ref": ref, "paths": ["/fields/open_questions/61"]}]})
            return prose("The historical conditions remain available; no new calculation was performed.")
    client = ReadThenDeliver([])
    feedback, state = scoped_commentary_turn(client, c, payload, s, scope=scope, state={"queries": 0},
        role="worker", prompt_name="command_worker_reads.md", compact_context=True)
    assert feedback is None and state["queries"] == 1
    # A real checkpoint round-trip retains the exact role-local context.
    state = json.loads(json.dumps(state))
    feedback, _ = scoped_commentary_turn(client, c, payload, s, scope=scope, state=state,
        role="worker", prompt_name="command_worker_reads.md", compact_context=True)
    assert feedback["status"] == "available"
    assert not feedback["read_record_ids"]  # Private navigation is not a cross-agent source ID.
    second = expand_text(client.requests[-1]["payload"])
    assert not any(r["path"] == "/fields/open_questions/61" for r in second["working_memory"]["findings"])
    assert client.requests[-1]["outcome"]["records"][0]["data"]["/fields/open_questions/61"]["conditions"]["state"] == 61
    # If the latest exchange is now an error, the exact read returns in memory.
    state.update(outcome={"status": "rejected", "issues": []}, latest_handles=[], latest_source_reads=[])
    scoped_commentary_turn(client, c, payload, s, scope=scope, state=state,
        role="worker", prompt_name="command_worker_reads.md", compact_context=True)
    third = expand_text(client.requests[-1]["payload"])
    assert any(r["path"] == "/fields/open_questions/61/conditions" and r["data"]["state"] == 61
               for r in third["working_memory"]["findings"])
    assert not selection_handles({"records": [{"ref": state["task_context_record"]["ref"],
        "selection": {"view": "data", "paths": ["/fields/open_questions/61"], "page": None},
        "data": {"/fields/open_questions/61": {"not_evidence": True}}}]})


def test_empty_or_small_context_adds_no_memory_record_or_lookup():
    original = {"results": [{"value": 1}], "open_questions": [], "read_scope": record_table([])}
    view, record = project_scoped(original, case_id="c", role="worker")
    assert view == original and record is None


def test_no_current_result_does_not_expand_entire_source_directory(tmp_path):
    c = sources()
    client = NativeFake([prose("No new result; existing records remain queryable.")])
    feedback, _ = scoped_commentary_turn(client, c, {"private_note": "Review alternatives."},
        store(tmp_path, c), scope=record_index(c).available, state={"queries": 0},
        role="exploration_reviewer", prompt_name="command_worker_reads.md", compact_context=True)
    view = expand_text(client.requests[0]["payload"])
    assert feedback["status"] == "available"
    assert len(view["read_scope"]["rows"]) <= 8
    assert view["source_directory_counts"]["total"] >= 100
    assert any(field["field"] == "read_scope" for field in view["task_memory"]["fields"])


def test_nested_review_keeps_claims_conditions_and_current_objections():
    data = material()
    data["arguments"] = [{"text": "Current numeric claim", "refs": ["R0"]}]
    data["execution_context"] = [{"method": "method", "basis": "basis", "state": 2}]
    data["review_workflow"] = {"issues": [{"id": "issue:1", "text": "Check this attribution"}]}
    original = {"sections": data, "read_scope": data.pop("read_scope"), "read_path_sets": data.pop("read_path_sets")}
    view, record = project_scoped(original, case_id="c", role="reviewer")
    for field in ("arguments", "execution_context", "review_workflow", "results"):
        assert view["sections"][field] == original["sections"][field]
    assert record and original["sections"]["open_questions"] != view["sections"]["open_questions"]


@pytest.mark.parametrize("protocol_name", ["responses", "chat_completions"])
def test_wire_dedup_preserves_exact_values_and_call_pairs(protocol_name):
    from mechcal.runtime.command_review_wire import operation_wire
    from mechcal.agents.command_tools import replay_exchange
    values = [{"energy": i / 100, "unit": "eV", "state": 2} for i in range(200)]
    outcome = {"status": "read", "records": [{"ref": "R1", "data": values}]}
    exchange = {"protocol": protocol_name, "items": [], "calls": [{"id": "c1"}]}
    original = {"working_memory": {"ref": "R1", "data": values}, "current_claim": "An uncertain comparison."}
    payload, receipt = operation_wire(original, exchange, outcome)
    wire = replay_exchange(exchange, receipt, protocol_name)
    restored = expand_text(payload)
    assert restored["working_memory"] == original["working_memory"]
    assert restored["operation_result"] == outcome
    returned = json.loads(wire[-1].get("content", wire[-1].get("output")))
    assert resolve_pointer(restored, returned["result_in_current_user_at"]) == outcome
    assert wire[-1].get("call_id", wire[-1].get("tool_call_id")) == "c1"
    assert json_bytes(payload) + json_bytes(receipt) < .65 * (json_bytes(original) + json_bytes(outcome))
    assert original["working_memory"]["data"] == values and outcome["records"][0]["data"] == values


def test_wire_keeps_invalid_or_unmatched_receipts_visible():
    from mechcal.runtime.command_review_wire import operation_wire
    exchange = {"calls": [{"id": "expected"}]}
    with pytest.raises(RuntimeError, match="receipts"):
        operation_wire({"large": "x" * 5000}, exchange, {"call_results": {"wrong": {"status": "read"}}})


def test_native_projection_preserves_existing_text_compression():
    from mechcal.runtime.command_context import share_text
    from mechcal.runtime.command_review_wire import operation_wire
    original = {"one": "A repeated scientific explanation with a condition. " * 50,
                "two": "A repeated scientific explanation with a condition. " * 50}
    prior = share_text(original)
    current, _ = operation_wire(prior, None, None)
    assert expand_text(current) == original
    assert json_bytes(current) <= json_bytes(prior)


def test_native_projection_does_not_reexpand_reviewer_shared_tables():
    from mechcal.runtime.command_review_wire import operation_wire, share_values
    table = [{"energy": i / 100, "unit": "eV", "state": 2} for i in range(200)]
    original = {"fact": {"ref": "R1", "value": table}, "query_result": {"ref": "R2", "value": table}}
    prior = share_values(original)
    current, _ = operation_wire(prior, None, None)
    assert json_bytes(current) <= json_bytes(prior)
    restored = expand_text(current)
    for key, value in original.items():
        assert restored[key] == value
    again, _ = operation_wire(current, None, None)
    assert again == current
