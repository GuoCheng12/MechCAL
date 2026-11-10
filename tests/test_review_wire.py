"""Review wire optimization must preserve facts, ownership and native receipts."""

from copy import deepcopy
import json

import pytest

from mechcal.agents.command_tools import replay_exchange
from mechcal.agents.command_worker import scoped_commentary_turn
from mechcal.runtime.command_compact import expand_text
from mechcal.runtime.command_context import share_text
from mechcal.runtime.command_memory import digest, json_bytes, resolve_pointer
from mechcal.runtime.command_operations import protocol, put, record_index
from mechcal.runtime.command_review_wire import share_values, review_wire, task_binding, method_index
from test_command_protocol import case, store
from test_native_commands import NativeFake, reply


def values():
    return [{"angle_deg": i * 10, "energy_eV": 3.0 + i / 97,
             "oscillator_strength": i / 10000, "status": "success"} for i in range(32)]


def unpack(packed):
    result = expand_text(packed)
    result.pop("review_values", None)
    result.pop("shared_value_policy", None)
    return result


def test_exact_fact_pool_roundtrip_preserves_unequal_sources_and_precision():
    table = values()
    data = {"a": {"ref": "R1", "method": "gas", "value": table},
            "b": {"ref": "R2", "method": "solvent", "value": table},
            "c": {"ref": "R3", "value": [*table, {"status": "failed"}]}}
    original = deepcopy(data)
    packed = share_text(share_values(data))
    assert unpack(packed) == data == original
    assert json_bytes(packed) < json_bytes(data) * .8
    assert packed["a"]["ref"] != packed["b"]["ref"]
    assert packed["a"]["method"] != packed["b"]["method"]


def test_nested_duplicates_pointers_escaping_and_text_sharing():
    text = "Observed trend, not an experimental validation. " * 25
    table = {"/~ strange": values(), "note": text}
    data = {"/source~one": table, "two": table, "three": {"rows": values(), "note": text}}
    assert unpack(share_text(share_values(data))) == data


def test_no_shared_values_does_not_inflate_small_requests():
    data = {"a": [1], "b": [1], "c": "different"}
    assert share_values(data) == data


def test_reserved_fields_do_not_overwrite_model_data():
    data = {"review_values": {"old": 1}, "a": values(), "b": values()}
    assert share_values(data) == data


def test_shared_value_cycles_fail_explicitly():
    with pytest.raises(ValueError, match="Cyclic"):
        expand_text({"a": {"same_data_at": "/b"}, "b": {"same_data_at": "/a"}})


@pytest.mark.parametrize("change", ["case", "role", "context"])
def test_task_cannot_cross_case_role_or_source_claim_version(change):
    payload = {"argument": "A", "source_version": "v1"}
    binding = task_binding("C1", payload, {}, "reviewer")
    state = {"review_task": binding}
    with pytest.raises(RuntimeError, match="different"):
        task_binding("C2" if change == "case" else "C1",
                     {**payload, "source_version": "v2"} if change == "context" else payload,
                     state, "final_reviewer" if change == "role" else "reviewer")


def test_only_readable_worker_prose_is_deferred_not_observations():
    text = "Worker interpretation. " * 500
    row = {"source_result_ids": ["R1"], "status": "available", "full_text_ref": "explanation:R1",
           "text_preview": text, "text_truncated": False, "scientific_approval": False,
           "source_observations": {"findings": [{"ref": "R1", "path": "/table", "data": values()}]}}
    data = {"sections": {"worker_interpretations": [row], "arguments:A": {"text": "Current claim"},
                         "review_workflow": {"issues": [{"category": "fact", "summary": "Check unit"}]}}}
    binding = task_binding("C1", data, {}, "reviewer")
    state = {"queries": 2}
    result, _ = review_wire(data, binding=binding, state=state, scope={"explanation:R1": {"text": text}})
    shown = unpack(result)["sections"]
    assert shown["worker_interpretations"][0]["source_observations"] == row["source_observations"]
    assert shown["worker_interpretations"][0]["text_visibility"] == "stored_not_shown"
    assert shown["arguments:A"] == data["sections"]["arguments:A"]
    assert shown["review_workflow"] == data["sections"]["review_workflow"]
    absent, _ = review_wire(data, binding=binding, state=state, scope={})
    assert unpack(absent)["sections"]["worker_interpretations"][0] == row
    assert data["sections"]["worker_interpretations"][0] == row


def test_legacy_reused_section_is_not_treated_as_new_worker_list():
    data = {"sections": {"worker_interpretations": {"previous_review_ref": "review:old", "path": "/sections/worker_interpretations"}}}
    projected, _ = review_wire(data, binding={}, state={}, scope={})
    assert unpack(projected)["sections"] == data["sections"]


def test_unchanged_summary_cannot_hide_a_changed_raw_source_in_continuation():
    sources = {"R1": {"method": "aTB1"}}
    binding = task_binding("C1", {}, {}, "reviewer", sources=sources)
    with pytest.raises(RuntimeError, match="different"):
        task_binding("C1", {}, {"review_task": binding}, "reviewer", sources={"R1": {"method": "DFTB"}})


def test_method_index_preserves_variants_failures_and_missing_labels():
    atb = {"step_id": "vertical", "source": "submitted_input", "executed_method_label": "aTB1 tda"}
    dftb = {**atb, "executed_method_label": "TD-DFTB"}
    scope = {"R1": {"status": "failed", "structured_results": {"method_provenance": [atb, atb, dftb, {}]}},
             "R2": {"status": "success", "structured_results": {"method_provenance": [{}]}}}
    source = method_index(scope)["sources"]
    assert len(source) == 1 and source[0]["status"] == "failed"
    assert source[0]["variants"] == [{**atb, "row_indices": [0, 1]}, {**dftb, "row_indices": [2]}]


@pytest.mark.parametrize("api", ["responses", "chat_completions"])
def test_query_data_delivered_once_with_correct_native_call_receipts(api):
    exchange = reply(call_id="read/1~")
    exchange["calls"] += reply(call_id="read/2")["calls"]
    exchange["protocol"] = api
    exchange["items"] = []
    outcome = {"status": "read", "call_results": {
        "read/1~": {"status": "read", "records": [{"ref": "R1", "data": values()}]},
        "read/2": {"status": "rejected", "issues": [{"code": "unknown_reference", "record_id": "bad"}]}}}
    original = deepcopy(outcome)
    payload = {"working_memory": {"findings": [{"ref": "R1", "data": values()}]},
               "sections": {"execution_context:R1": {"status": "failed", "failure": "not evidence"}}}
    state = {"queries": 1, "exchange": exchange, "outcome": outcome}
    material, receipt = review_wire(payload, binding={}, state=state, scope={})
    restored = unpack(share_text(material))
    assert restored["query_result"] == outcome == original
    assert restored["sections"] == payload["sections"]
    messages = replay_exchange(exchange, receipt, api)
    for call, msg in zip(exchange["calls"], messages):
        ack = json.loads(msg.get("content", msg.get("output")))
        assert resolve_pointer(restored, ack["result_in_current_user_at"]) == outcome["call_results"][call["id"]]
        assert "data" not in ack
    assert restored["review_task"]["phase"] == "continue"


def test_query_mismatched_call_ids_cannot_be_hidden_by_projection():
    with pytest.raises(RuntimeError, match="tool calls"):
        review_wire({}, binding={}, state={"exchange": reply(),
                    "outcome": {"call_results": {"foreign": {"status": "read"}}}}, scope={})


def test_session_resume_keeps_facts_and_rejections_without_scientific_mutation(tmp_path):
    c = put(case(), results={"R1": {"status": "failed", "values": values()}})
    original = c.model_dump_json()
    fake = NativeFake([reply(arguments={"records": [{"ref": "R1", "paths": ["/values"]}]}),
                       {"request_id": "done", "complete": True, "calls": [], "text": "Calculation failed; no positive evidence."}])
    payload = {"sections": {"arguments:A": {"refs": ["R1"], "text": "Check this claim."}}}
    scope = {"R1": record_index(c).available["R1"]}
    feedback, state = scoped_commentary_turn(fake, c, payload, store(tmp_path, c), scope=scope,
        state={"queries": 0}, role="reviewer", prompt_name="command_reviewer_text.md", compact_context=True)
    assert feedback is None and state["queries"] == 1 and state["review_task"]
    restored = json.loads(json.dumps(state))
    feedback, state = scoped_commentary_turn(fake, c, payload, store(tmp_path, c), scope=scope,
        state=restored, role="reviewer", prompt_name="command_reviewer_text.md", compact_context=True)
    assert feedback["status"] == "available" and state is None
    assert c.model_dump_json() == original
    wire = unpack(fake.requests[-1]["payload"])
    assert wire["query_result"]["status"] == "read"
    assert wire["review_task"]["phase"] == "continue"
    assert not protocol(c).get("review_receipts")
