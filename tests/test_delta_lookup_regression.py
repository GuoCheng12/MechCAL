"""Regression for the paid probe's growing, self-referential lookup context."""

from copy import deepcopy
import json

from mechcal.agents.planner_references import PlannerReferences
from mechcal.runtime.case_memory import serialized
from mechcal.runtime.context_compaction import expand_payload
from mechcal.runtime.memory_views import retrieved_working_set
from mechcal.schemas.memory import MemoryQuery
from test_delta_context import DONE, READY, WireClient, query, setup, reopen


def context_query(paths=(), view="auto"):
    def respond(payload):
        return MemoryQuery(action="read_case_records", record_ids=[payload["context_record_id"]],
                           paths=list(paths), view=view).model_dump(mode="json")
    return respond, "memory_query"


def test_context_id_does_not_change_when_memory_is_read(tmp_path):
    case, _, memory = setup(tmp_path)
    client = WireClient([context_query(view="outline"), context_query(["/revision"]), READY, DONE])
    memory.complete(client, PlannerReferences(case), payload=memory.payload, system_prompt="unused")
    ids = {call["payload"]["context_record_id"] for call in client.calls}
    assert len(ids) == 1
    row = memory.index.read(MemoryQuery(action="read_case_records", record_ids=list(ids),
                                       paths=["/retrieved_memory"]))["records"][0]
    assert "error" in row["record"]["selected_fields"]["/retrieved_memory"]


def test_lookup_keeps_latest_values_and_receipts_not_all_prior_bodies(tmp_path):
    case, _, memory = setup(tmp_path)
    payload = {**memory.payload, "first": "UNIQUE_FIRST_VALUE " * 150, "second": "UNIQUE_SECOND_VALUE " * 150}
    reopen(memory, payload)
    client = WireClient([context_query(["/first"]), context_query(["/second"]), READY, DONE])
    memory.complete(client, PlannerReferences(case), payload=payload, system_prompt="unused")
    lookup = expand_payload(client.calls[2]["payload"])
    assert "UNIQUE_FIRST_VALUE" not in serialized(lookup)
    assert "UNIQUE_SECOND_VALUE" in serialized(lookup)
    assert "/first" in serialized(lookup["memory_read_receipts"])
    decision = expand_payload(client.calls[3]["payload"])
    assert "UNIQUE_FIRST_VALUE" in serialized(decision["retrieved_memory"])
    assert "UNIQUE_SECOND_VALUE" in serialized(decision["retrieved_memory"])


def test_overlapping_fields_are_sent_once_without_losing_errors_or_conditions():
    def response(fields):
        return {"records": [{"record_id": "context:stable", "record": {
            "selected_fields": fields, "source_context": {"limits": ["Fixed geometry only"]},
            "field_context": {"/metrics": {"unit": "eV"}},
        }}]}
    old = response({"/a": 1, "/b": 2, "/missing": {"error": "Field not found"}})
    new = response({"/b": 2, "/c": 3})
    original = deepcopy([old, new])
    output = retrieved_working_set([old, new])
    fields = [row["record"]["selected_fields"] for item in output for row in item["records"]]
    assert sum("/b" in item for item in fields) == 1
    assert {k: v for item in fields for k, v in item.items()} == {
        "/a": 1, "/b": 2, "/c": 3, "/missing": {"error": "Field not found"}}
    assert output[0]["records"][0]["record"]["source_context"] == old["records"][0]["record"]["source_context"]
    assert output[0]["records"][0]["record"]["field_context"] == old["records"][0]["record"]["field_context"]
    assert [old, new] == original


def test_large_context_read_returns_whole_fields_and_explicit_unread_paths(tmp_path):
    _, _, memory = setup(tmp_path)
    source = {**memory.payload, "table": list(range(9000)), "condition": "Fixed geometry only", "energy": 2.5}
    rid = memory.index.register_context(source)
    row = memory.index.read(MemoryQuery(action="read_case_records", record_ids=[rid], view="full",
                                       paths=["/table", "/condition", "/energy"]))["records"][0]
    assert "json_fragment" not in row
    assert row["record"]["selected_fields"] == {"/condition": "Fixed geometry only", "/energy": 2.5}
    assert row["record"]["deferred_paths"] == ["/table"]
    assert len(serialized(row)) < 12000
    # An explicit single-field full read remains recoverable, including old fragment clients.
    start, pieces = 0, []
    while True:
        part = memory.index.read(MemoryQuery(action="read_case_records", record_ids=[rid], view="full",
                                            paths=["/table"], start=start))["records"][0]
        pieces.append(part["json_fragment"])
        if part["next_start"] is None:
            break
        start = part["next_start"]
    assert json.loads("".join(pieces))["selected_fields"]["/table"] == source["table"]


def test_context_root_directory_is_shallow_but_nested_reads_remain_available(tmp_path):
    _, _, memory = setup(tmp_path)
    source = {**memory.payload, "details": {f"field{i}": {"unit": "eV", "value": i} for i in range(500)}}
    rid = memory.index.register_context(source)
    row = memory.index.read(MemoryQuery(action="read_case_records", record_ids=[rid], view="outline"))["records"][0]
    fields = row["record"]["fields"]
    assert all(f["path"].count("/") == 1 for f in fields)
    assert len(serialized(row)) < 6000
    nested = memory.index.read(MemoryQuery(action="read_case_records", record_ids=[rid],
                                          view="outline", paths=["/details/field20"]))["records"][0]
    assert "/details/field20/unit" in {f["path"] for f in nested["record"]["fields"]}


def test_latest_duplicate_query_is_still_latest_after_resume(tmp_path):
    case, store, memory = setup(tmp_path)
    first = query(record_ids=["Q1"], paths=["/question"])[0]
    second = query(record_ids=["Q1"], paths=["/answer_state"])[0]
    for read in [first, second, first]:
        memory.query(read)
    from mechcal.runtime.planner_memory import PlannerMemorySession
    restored = PlannerMemorySession(case, store, memory.registry, memory.stage, memory.payload)
    assert list(restored.loaded.values())[-1]["request"] == first
    assert restored.used == 3


def test_later_complete_field_resolves_old_deferred_receipt():
    old = {"records": [{"record_id": "C", "record": {
        "selected_fields": {"/a": 1}, "deferred_paths": ["/b", "/c"]}}]}
    new = {"records": [{"record_id": "C", "record": {"selected_fields": {"/b": 2}}}]}
    output = retrieved_working_set([old, new])
    assert output[0]["records"][0]["record"]["deferred_paths"] == ["/c"]
    assert old["records"][0]["record"]["deferred_paths"] == ["/b", "/c"]


def test_lookup_receipts_preserve_failures_and_pagination():
    from mechcal.runtime.memory_views import lookup_memory
    replies = [
        {"records": [{"record_id": "unknown", "error": "unknown_record"}]},
        {"records": [{"record_id": "X", "json_fragment": "OLD_RAW_FRAGMENT", "start": 0, "next_start": 12000}]},
        {"records": [{"record_id": "C", "record": {
            "selected_fields": {"/absent": {"error": "missing"}}, "deferred_paths": ["/table"]}}]},
        {"records": [{"record_id": "D", "record": {"view": "outline", "fields": [], "next_offset": 20}}]},
        {"error": "query_rejected", "request": {}},
    ]
    before = deepcopy(replies)
    latest, receipts = lookup_memory(replies)
    assert latest == replies[-1:]
    assert receipts[0]["records"][0]["error"] == "unknown_record"
    assert receipts[1]["records"][0]["partial"] and receipts[1]["records"][0]["next_start"] == 12000
    assert "OLD_RAW_FRAGMENT" not in serialized(receipts)
    assert receipts[2]["records"][0]["field_errors"] == {"/absent": {"error": "missing"}}
    assert receipts[2]["records"][0]["deferred_paths"] == ["/table"]
    assert receipts[3]["records"][0]["next_offset"] == 20
    assert replies == before


def test_discovery_and_retry_keep_same_source_snapshot(tmp_path):
    import pytest
    from mechcal.capabilities.discovery import ToolDiscoverySession
    from mechcal.runtime.planner_memory import PlannerMemorySession
    from mechcal.runtime.retry import RejectedResponseError
    from test_question_loop import TOOL

    case, store, memory = setup(tmp_path)
    discovery = ToolDiscoverySession(memory.registry, case, "R001")
    bad = WireClient([
        ({"action": "describe_tools", "capability_ids": [TOOL]}, "describe_tools"),
        ({**READY[0], "focus_capability_ids": ["not_loaded"]}, "decision"),
    ])
    with pytest.raises(RejectedResponseError):
        memory.complete(bad, PlannerReferences(case), payload=memory.payload, system_prompt="unused", discovery=discovery)
    restored = PlannerMemorySession(case, store, memory.registry, memory.stage, memory.payload)
    good = WireClient([({**READY[0], "focus_capability_ids": [TOOL]}, "context_read"), DONE])
    restored.complete(good, PlannerReferences(case), payload=memory.payload, system_prompt="unused", discovery=discovery)
    assert len({c["payload"]["context_record_id"] for c in [*bad.calls, *good.calls]}) == 1
    assert discovery.batches_used == 1


def test_context_multi_read_retains_small_zero_false_null_and_missing_fields(tmp_path):
    _, _, memory = setup(tmp_path)
    source = {**memory.payload, "large": list(range(9000)), "zero": 0, "false": False, "null": None}
    rid = memory.index.register_context(source)
    row = memory.index.read(MemoryQuery(action="read_case_records", record_ids=[rid], view="full",
                                       paths=["/large", "/zero", "/false", "/null", "/absent"]))["records"][0]
    fields = row["record"]["selected_fields"]
    assert fields["/zero"] == 0 and fields["/false"] is False and fields["/null"] is None
    assert fields["/absent"]["error"] == "Field not found: /absent"
    assert row["record"]["deferred_paths"] == ["/large"]
