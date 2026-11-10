"""Offline retrieval contracts; synthetic values are not chemistry results."""

from copy import deepcopy
import json

import pytest

from mechcal.agents.planner_references import PlannerReferences
from mechcal.runtime.case_memory import CaseMemoryStore, pointer, serialized
from mechcal.runtime.memory_continuation import continuation_context
from mechcal.runtime.memory_views import retrieved_working_set
from mechcal.runtime.planner_memory import PlannerMemorySession
from mechcal.runtime.questions import public_payload, replace_state
from mechcal.runtime.structured_output import PlannerOutputContract
from mechcal.schemas.memory import MemoryQuery
from test_case_memory import Client, REGISTRY, fixture, session
from test_working_context import context


STATES = "/structured_results/parsed_observables/excited_states"


def calculation(rid="X1"):
    return {
        "result_id": rid, "capability_id": "fixture.electronic", "status": "partial",
        "requested_parameters": {"charge": 0, "multiplicity": 1, "source_artifact_id": "geometry:far"},
        "calculation_identity": {"arguments": {"nstates": 3}, "resolved_source_artifact_id": "geometry:far"},
        "failure": {"kind": "timeout", "message": "A separate substep timed out"},
        "evidence_refs": ["E1"], "delivery": {"missing": ["density"], "available": ["states"]},
        "structured_results": {
            "method_provenance": {"method": "fixture", "basis": "fixture"},
            "limitations": ["Fixed geometry only"],
            "parsed_observables": {
                "excited_states": [{"state": 1, "energy_ev": 2.5, "oscillator_strength": 0.01}],
                "transition_conventions": {"root": "one-based", "units": "eV"},
                "dominant_transitions": [{"amplitude": 0.1}] * 4000,
            },
        },
    }


def indexed(tmp_path, *, enabled=True, rows=None):
    case, store = fixture(tmp_path, scoped_evidence_updates=enabled)
    case = replace_state(case, results=rows or [calculation()], pending_results=[])
    return case, store, CaseMemoryStore(store, case)


def read(index, **kwargs):
    return index.read(MemoryQuery(action="read_case_records", **kwargs))["records"][0]


def test_large_record_defaults_to_recoverable_directory_not_json_prefix(tmp_path):
    case, store, index = indexed(tmp_path)
    result = read(index, record_ids=["X1"])
    assert "json_fragment" not in result
    outline = result["record"]
    assert outline["preview_only"] and outline["view"] == "outline"
    assert STATES in {r["path"] for r in outline["fields"]}
    assert len(serialized(result)) < 12000
    source = calculation()
    assert all(pointer(source, f["path"]) is not None for f in outline["fields"])
    assert case.runtime["question_loop"]["revision"] == 1


def test_selected_quantities_keep_conditions_and_failure_without_bulk_arrays(tmp_path):
    _, _, index = indexed(tmp_path)
    result = read(index, record_ids=["X1"], paths=[STATES])["record"]
    assert result["selected_fields"][STATES] == pointer(calculation(), STATES)
    c = result["source_context"]
    assert c["status"] == "partial" and c["failure"]["kind"] == "timeout"
    assert c["requested_parameters"]["source_artifact_id"] == "geometry:far"
    assert c["/structured_results/limitations"] == ["Fixed geometry only"]
    assert c["/structured_results/method_provenance"]["method"] == "fixture"
    assert c["/structured_results/parsed_observables/transition_conventions"]["units"] == "eV"
    assert "dominant_transitions" not in serialized(result)


def test_unknown_path_keeps_explicit_error_and_can_be_corrected(tmp_path):
    _, _, index = indexed(tmp_path)
    missing = read(index, record_ids=["X1"], paths=["/not_here"])
    assert "error" in missing["record"]["selected_fields"]["/not_here"]
    assert read(index, record_ids=["unknown"])["error"] == "unknown_record"
    assert read(index, record_ids=["X1"], paths=[STATES])["record"]["selected_fields"][STATES]


@pytest.mark.parametrize("path", [[], ["/structured_results"]])
def test_explicit_full_read_is_reconstructable_and_never_silently_truncated(tmp_path, path):
    _, _, index = indexed(tmp_path)
    start, parts = 0, []
    while True:
        row = read(index, record_ids=["X1"], paths=path, view="full", start=start)
        parts.append(row["json_fragment"])
        if row["next_start"] is None:
            break
        start = row["next_start"]
    actual = json.loads("".join(parts))
    assert (actual["selected_fields"][path[0]] if path else actual) == (
        pointer(calculation(), path[0]) if path else calculation())


def test_small_reads_and_legacy_fragments_are_unchanged(tmp_path):
    _, _, index = indexed(tmp_path / "legacy", enabled=False)
    assert "json_fragment" in read(index, record_ids=["X1"])
    _, _, focused = indexed(tmp_path / "focused")
    assert read(focused, record_ids=["Q1"])["record"]["question_id"] == "Q1"


def test_subtree_and_escaped_paths_and_paginated_directory(tmp_path):
    row = calculation()
    row["structured_results"] = {"a/b~c": {f"field{i}": i for i in range(260)}}
    _, _, index = indexed(tmp_path, rows=[row])
    paths = ["/structured_results/a~1b~0c"]
    offset, fields = 0, []
    while True:
        result = read(index, record_ids=["X1"], view="outline", paths=paths, offset=offset)["record"]
        fields += result["fields"]
        assert len(serialized(result)) <= 12000
        if result["next_offset"] is None:
            break
        assert result["next_offset"] > offset
        offset = result["next_offset"]
    assert len({f["path"] for f in fields}) == 261
    assert read(index, record_ids=["X1"], paths=[paths[0] + "/field200"])["record"]["selected_fields"] == {
        paths[0] + "/field200": 200}


def test_large_selected_value_becomes_directory_and_can_be_narrowed(tmp_path):
    _, _, index = indexed(tmp_path)
    path = "/structured_results/parsed_observables/dominant_transitions"
    directory = read(index, record_ids=["X1"], paths=[path])["record"]
    assert directory["view"] == "outline"
    assert directory["fields"] == [{"path": path, "type": "array", "items": 4000}]
    assert read(index, record_ids=["X1"], paths=[path + "/0"])["record"]["selected_fields"] == {
        path + "/0": {"amplitude": 0.1}}


def test_same_snapshot_reads_keep_distinct_fields_errors_and_fragments():
    outline = {"records": [{"record_id": "X1", "record": {"view": "outline"}}]}
    a = {"records": [{"record_id": "X1", "record": {"selected_fields": {"/a": 1}}}]}
    b = {"records": [{"record_id": "X1", "record": {"selected_fields": {"/b": 2}}}]}
    err = {"records": [{"record_id": "unknown", "error": "unknown_record"}]}
    fragment = {"records": [{"record_id": "X1", "json_fragment": "{", "next_start": 1}]}
    responses = [outline, a, a, b, err, fragment]
    before = deepcopy(responses)
    assert retrieved_working_set(responses) == [a, b, err, fragment]
    assert responses == before
    full = {"records": [{"record_id": "X1", "record": {"a": 1, "b": 2}}]}
    assert retrieved_working_set([a, b, full]) == [full]


def test_continuation_preserves_current_science_and_only_thins_unrequested_history():
    payload = context("planner_question_act")
    payload["scoped_evidence_updates"] = True
    payload["observations"] = []
    payload["pending_results"] = []
    payload["execution_history"].append({"result_id": "X2", "requested_parameters": {"far": True}})
    payload["execution_history"][0]["calculation_identity"] = {"long": "details" * 300}
    original = deepcopy(payload)
    after = continuation_context(payload, [{"request": {"record_ids": ["X2"]}}])
    for k in ("portfolio", "previous_support_arguments", "mechanism_pool", "argument_review",
              "evidence_index", "observations", "pending_results"):
        assert after[k] == payload[k]
    assert after["execution_history"][0]["failure"] == payload["execution_history"][0]["failure"]
    assert after["execution_history"][0]["delivery"] == payload["execution_history"][0]["delivery"]
    assert "requested_parameters" not in after["execution_history"][0]
    assert after["execution_history"][0]["read_before_reuse"] == ["/requested_parameters", "/calculation_identity"]
    assert after["execution_history"][1] == payload["execution_history"][1]
    assert payload == original


@pytest.mark.parametrize("reason", ["new", "pending", "audit", "stop", "queried_source"])
def test_active_sources_pin_execution_conditions(reason):
    payload = context()
    payload.update(scoped_evidence_updates=True, observations=[], pending_results=[])
    queries = []
    if reason == "new":
        payload["observations"] = [{"evidence_id": "old"}]
    elif reason == "pending":
        payload["pending_results"] = [{"result_id": "X1"}]
    elif reason == "audit":
        payload.update(argument_review_pending=True, argument_review={"evidence_refs": ["old"]})
    elif reason == "stop":
        payload["stop_review_feedback"] = {"evidence_refs": ["old"]}
    else:
        queries = [{"request": {"record_ids": ["old"]}}]
    assert continuation_context(payload, queries)["execution_history"] == payload["execution_history"]


@pytest.mark.parametrize("final,enabled", [(True, True), (False, False)])
def test_final_revision_and_legacy_context_are_not_thinned(final, enabled):
    payload = {**context(), "scoped_evidence_updates": enabled}
    if final:
        payload["stage"] = "planner_final_revision"
    assert continuation_context(payload, []) == payload


def test_strict_query_loop_directory_then_fields_then_decision_is_read_only(tmp_path):
    case, store, _ = indexed(tmp_path)
    case = case.touch(runtime={**case.runtime, "planner_strict_outputs": True})
    before = case.model_dump_json()
    payload = public_payload(case, REGISTRY, stage="planner_question_act")
    payload["output_contract"] = {"type": "object", "properties": {"done": {"type": "boolean"}},
                                  "required": ["done"], "additionalProperties": False}
    memory = PlannerMemorySession(case, store, REGISTRY, payload["stage"], payload)
    contract = PlannerOutputContract(memory.augment(memory.payload))
    client = Client([
        contract.encode(MemoryQuery(action="read_case_records", record_ids=["X1"]).model_dump(), kind="memory_query"),
        contract.encode(MemoryQuery(action="read_case_records", record_ids=["X1"], paths=[STATES]).model_dump(),
                        kind="memory_query"),
        contract.encode({"done": True}),
    ])
    assert memory.complete(client, PlannerReferences(case), payload=memory.payload, system_prompt="fixture") == {
        "done": True}
    assert len(client.calls) == 3 and memory.used == 2
    last = client.calls[-1]["payload"]
    assert len(last["retrieved_memory"]) == 1  # Directory replaced by its selected values.
    assert last["retrieved_memory"][0]["records"][0]["record"]["selected_fields"][STATES]
    assert last["memory_access"]["queries_remaining"] == 2
    assert last["portfolio"] == memory.payload["portfolio"]
    assert case.model_dump_json() == before
    restored = session(case, store)
    assert restored.used == 2 and len(restored.augment(restored.payload)["retrieved_memory"]) == 1
    events = [json.loads(line) for line in memory.log_path.read_text().splitlines()]
    assert any(e["kind"] == "memory_request_context" and e["continuation"] for e in events)


def test_selected_scalar_retains_sibling_units_and_state(tmp_path):
    row = calculation()
    row["structured_results"].update(energy=2.3, units="eV", state=2, conditions={"solvent": "none"})
    _, _, index = indexed(tmp_path, rows=[row])
    record = read(index, record_ids=["X1"], paths=["/structured_results/energy"])["record"]
    assert record["field_context"]["/structured_results"]["units"] == "eV"
    assert record["field_context"]["/structured_results"]["state"] == 2
    assert record["field_context"]["/structured_results"]["conditions"] == {"solvent": "none"}


def test_tiny_continuation_does_not_increase_context_size():
    payload = {"stage": "planner_question_act", "scoped_evidence_updates": True,
               "execution_history": [{"result_id": "X1", "requested_parameters": {"seed": 1}}]}
    assert continuation_context(payload, []) == payload


def test_query_response_can_replace_exact_duplicate_parameters_without_changing_values():
    payload = {"stage": "planner_question_act", "scoped_evidence_updates": True,
               "execution_history": [{"result_id": "X1", "requested_parameters": {"values": list(range(1000))},
                                      "status": "partial", "failure": {"kind": "timeout"}}]}
    responses = [{"request": {"record_ids": ["X1"]}, "records": [{"record_id": "X1", "record": {
        "selected_fields": {"/value": 2},
        "source_context": {"requested_parameters": deepcopy(payload["execution_history"][0]["requested_parameters"])},
    }}]}]
    before = deepcopy((payload, responses))
    after = continuation_context(payload, responses)
    row = after["execution_history"][0]
    assert "requested_parameters" not in row and row["parameters_in_retrieved_record"] == "X1"
    assert row["failure"] == payload["execution_history"][0]["failure"]
    assert (payload, responses) == before
    responses[0]["records"][0]["record"]["source_context"]["requested_parameters"]["values"][-1] = 998.9
    assert continuation_context(payload, responses) == payload


def test_unrequested_old_test_text_is_retrievable_but_outcomes_remain_visible():
    test = {"memory_record_id": "test:1:X1", "result_id": "X1", "revision": 1,
            "tested_claim": {"target_label": "A", "alternative_label": "B", "tested_claim": "old detail " * 100},
            "assessment": {"basis_effect": "weakened", "comparison_effect": "non_discriminating"}}
    payload = {"stage": "planner_question_act", "scoped_evidence_updates": True,
               "hypothesis_test_context": {"recent_tests": [test], "current_comparison": {"leader": "B"}}}
    after = continuation_context(payload, [])
    row = after["hypothesis_test_context"]["recent_tests"][0]
    assert row["assessment"] == test["assessment"] and row["memory_record_id"] == test["memory_record_id"]
    assert "tested_claim" not in row["tested_claim"]
    assert after["hypothesis_test_context"]["current_comparison"] == {"leader": "B"}
    assert continuation_context(payload, [{"request": {"record_ids": ["test:1:X1"]}}]) == payload


def test_missing_metadata_values_are_retained_as_unknown(tmp_path):
    row = calculation()
    row["structured_results"].update(energy=2.3, units=None)
    _, _, index = indexed(tmp_path, rows=[row])
    record = read(index, record_ids=["X1"], paths=["/structured_results/energy"])["record"]
    assert "units" in record["field_context"]["/structured_results"]
    assert record["field_context"]["/structured_results"]["units"] is None


def test_directories_do_not_hide_read_errors_or_partial_fragments():
    records = [
        {"records": [{"record_id": "X1", "record": {"selected_fields": {"/a": 1, "/missing": {"error": "absent"}}}}]},
        {"records": [{"record_id": "X1", "record": {"selected_fields": {"/a": 1}}}]},
        {"records": [{"record_id": "X1", "json_fragment": "{", "start": 0, "next_start": 1}]},
        {"records": [{"record_id": "X1", "json_fragment": "}", "start": 1, "next_start": None}]},
    ]
    projected = retrieved_working_set(records)
    assert projected[0]["records"][0]["record"]["selected_fields"] == {"/missing": {"error": "absent"}}
    assert projected[1:] == records[1:]


def test_retry_keeps_query_budget_failures_and_does_not_change_case(tmp_path):
    case, store, _ = indexed(tmp_path)
    memory = session(case, store)
    before = case.model_dump_json()
    memory.query({"action": "read_case_records", "record_ids": ["missing"]})
    client = Client([{"done": True}])
    memory.complete(client, PlannerReferences(case), payload=memory.payload, system_prompt="fixture",
                    schema_feedback="Correct previous formal decision")
    assert client.calls[0]["schema_feedback"] == "Correct previous formal decision"
    assert client.calls[0]["payload"]["memory_access"]["queries_remaining"] == 3
    assert client.calls[0]["payload"]["retrieved_memory"][0]["records"][0]["error"] == "unknown_record"
    assert case.model_dump_json() == before
