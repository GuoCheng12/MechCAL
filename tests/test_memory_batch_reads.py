"""Whole-field retrieval for every record kind, not just Planner context snapshots."""

from copy import deepcopy
import json

import pytest

from mechcal.agents.planner_references import PlannerReferences
from mechcal.runtime.case_memory import serialized
from mechcal.runtime.context_compaction import expand_payload
from mechcal.runtime.delta_session import read_needs_review
from mechcal.runtime.memory_views import allocate_read_budgets, render_record, retrieved_working_set, selected_record
from mechcal.schemas.memory import MemoryQuery
from test_delta_context import DONE, READY, WireClient, setup, reopen


@pytest.mark.parametrize("kind", ["context", "observation", "execution", "judgment", "question", "artifact"])
def test_multi_field_full_reads_never_split_json(kind):
    data = {"large": {"states": list(range(10000)), "energy": 2.5}, "limits": ["Fixed geometry only"], "status": "partial"}
    request = MemoryQuery(action="read_case_records", record_ids=["X"], view="full", paths=["/large", "/limits", "/status"])
    row = render_record("X", kind, serialized(data), request, focused=True)
    assert "json_fragment" not in row
    record = row["record"]
    assert record["selected_fields"] == {"/limits": data["limits"], "/status": "partial"}
    assert record["deferred_paths"] == ["/large"]
    assert "/large/energy" in serialized(record["deferred_fields"])
    assert read_needs_review({"records": [row]})


@pytest.mark.parametrize("paths", [[], ["/large"]])
def test_batched_single_field_or_whole_record_is_not_a_fragment(paths):
    request = MemoryQuery(action="read_case_records", record_ids=["X", "Y"], view="full", paths=paths)
    row = render_record("X", "observation", serialized({"large": list(range(10000))}), request, focused=True)
    assert "json_fragment" not in row
    assert read_needs_review({"records": [row]})


@pytest.mark.parametrize("detail", [
    {"record": {"view": "outline", "fields": []}},
    {"record": {"selected_fields": {"/status": "ok"}, "deferred_paths": ["/metrics"]}},
    {"json_fragment": "{", "start": 0, "next_start": 12000},
    {"json_fragment": "}", "start": 12000, "next_start": None},
    {"record": {"selected_fields": {"/energy": 2.5}, "context_not_loaded": True}},
])
def test_incomplete_delivery_requires_a_lookup_turn(detail):
    assert read_needs_review({"records": [{"record_id": "X", **detail}]})


def test_batch_reuses_unused_space_without_raising_total_budget(tmp_path):
    _, _, memory = setup(tmp_path)
    source = {"case_id": memory.case.case_id, "data": "a" * 8000}
    large = memory.index.register_context(source)
    ids = [large] + [memory.index.register_context({"case_id": memory.case.case_id, "i": i, "data": i}) for i in range(5)]
    rows = memory.index.read(MemoryQuery(action="read_case_records", record_ids=ids, view="full", paths=["/data"]))["records"]
    assert all("json_fragment" not in r for r in rows)
    assert rows[0]["record"]["selected_fields"]["/data"] == source["data"]
    assert sum(len(serialized(r["record"])) for r in rows) <= 12000


def test_partial_batch_does_not_silently_enter_scientific_decision(tmp_path):
    case, _, memory = setup(tmp_path)
    payload = {**memory.payload, "large": {"table": list(range(10000)), "energy": 2.5}, "condition": "fixed geometry"}
    original = deepcopy(payload)
    reopen(memory, payload)

    def broad(p):
        return {**READY[0], "memory_queries": [MemoryQuery(
            action="read_case_records", record_ids=[p["context_record_id"]],
            paths=["/large", "/condition"], view="full").model_dump(mode="json")]}

    def narrow(p):
        return {**READY[0], "memory_queries": [MemoryQuery(
            action="read_case_records", record_ids=[p["context_record_id"]],
            paths=["/large/energy"], view="auto").model_dump(mode="json")]}

    client = WireClient([(broad, "decision"), (narrow, "decision"), DONE])
    memory.complete(client, PlannerReferences(case), payload=payload, system_prompt="unused")
    assert [c["payload"]["stage"] for c in client.calls] == ["planner_context_lookup", "planner_context_lookup", "planner_question_act"]
    assert memory.used == 2 and payload == original
    final = expand_payload(client.calls[-1]["payload"])
    assert '"/large/energy":2.5' in serialized(final["retrieved_memory"])
    assert '"/large"' in serialized(final["retrieved_memory"])  # Narrowing did not load the entire parent.


def test_exact_single_record_full_pagination_is_compatible():
    data = {"large": list(range(10000))}
    parts, start = [], 0
    while True:
        request = MemoryQuery(action="read_case_records", record_ids=["X"], paths=["/large"], view="full", start=start)
        row = render_record("X", "observation", serialized(data), request, focused=True)
        parts.append(row["json_fragment"])
        if row["next_start"] is None:
            break
        start = row["next_start"]
    assert json.loads("".join(parts))["selected_fields"]["/large"] == data["large"]


def test_error_metadata_is_not_scientific_error_measurement():
    assert not read_needs_review({"records": [{"record": {"selected_fields": {"/metric": {"error": 0.2, "unit": "eV"}}}}]})


@pytest.mark.parametrize("demands", [[0, 0], [100, 150], [100, 20000, 300], [300000] * 6, [8000, 2000, 3000]])
def test_budget_redistribution_never_increases_allowance(demands):
    allocated = allocate_read_budgets(demands)
    assert len(allocated) == len(demands)
    assert sum(allocated) <= 12000
    assert all(0 <= actual <= wanted for actual, wanted in zip(allocated, demands))
    if sum(demands) <= 12000:
        assert allocated == demands


def test_conditions_are_kept_once_and_never_inferred():
    data = {"metrics": {"data": list(range(10000))}, "context": {"unit": "eV", "temperature": None},
            "limits": ["Fixed geometry only"] * 10, "status": "partial"}
    request = MemoryQuery(action="read_case_records", record_ids=["X", "Y"], view="full",
                          paths=["/metrics", "/limits", "/context", "/status"])
    before = deepcopy(data)
    record = render_record("X", "observation", serialized(data), request, focused=True)["record"]
    assert record["selected_fields"]["/limits"] == data["limits"]
    assert record["selected_fields"]["/context"] == data["context"]
    assert "limits" not in record["source_context"]
    assert record["deferred_paths"] == ["/metrics"]
    assert data == before


def test_full_field_fragment_continuations_from_old_multi_field_requests_still_work():
    data = {"large": list(range(10000)), "status": "partial"}
    paths = ["/large", "/status"]
    body = serialized(selected_record(data, paths, focused=True))
    request = MemoryQuery(action="read_case_records", record_ids=["X"], view="full", paths=paths, start=2400)
    row = render_record("X", "execution", serialized(data), request, focused=True)
    assert row["json_fragment"] == body[2400:14400]
    assert row["next_start"] == 14400


def test_unread_field_descriptors_do_not_survive_a_complete_read():
    first = {"records": [{"record_id": "X", "record": {
        "selected_fields": {"/status": "partial"}, "deferred_paths": ["/a", "/b"],
        "deferred_fields": [{"path": "/a/value", "type": "value"}, {"path": "/b", "type": "array"}],
        "directory_incomplete": False}}]}
    second = {"records": [{"record_id": "X", "record": {"selected_fields": {"/a": {"value": 1}}}}]}
    result = retrieved_working_set([first, second])
    assert result[0]["records"][0]["record"]["deferred_fields"] == [{"path": "/b", "type": "array"}]


def test_exhausted_allowance_requires_visible_gap_review_not_automatic_ready(tmp_path):
    case, store, memory = setup(tmp_path)
    payload = {**memory.payload, "large": list(range(10000)), "condition": "fixed geometry"}
    reopen(memory, payload)
    def broad(p):
        q = MemoryQuery(action="read_case_records", record_ids=[p["context_record_id"]],
                        paths=["/large", "/condition"], view="full").model_dump(mode="json")
        return {**READY[0], "memory_queries": [q] * 4}
    client = WireClient([(broad, "decision"), READY, DONE])
    memory.complete(client, PlannerReferences(case), payload=payload, system_prompt="unused")
    assert client.calls[1]["payload"]["stage"] == "planner_context_lookup"
    assert client.calls[1]["payload"]["memory_access"]["queries_remaining"] == 0
    assert memory.used == 4
    final = expand_payload(client.calls[-1]["payload"])
    assert final["retrieved_memory"][0]["records"][0]["record"]["deferred_paths"] == ["/large"]
    events = [json.loads(line) for line in (store.root / "history/memory_queries.jsonl").read_text().splitlines()]
    assert sum(e["kind"] == "delta_context_read_followup" for e in events) == 1
    assert sum(e["kind"] == "delta_context_ready" for e in events) == 1


def test_small_narrow_batch_includes_numeric_signs_zero_and_null_context(tmp_path):
    _, _, memory = setup(tmp_path)
    originals = [{"case_id": memory.case.case_id, "metrics": {"energy": energy, "unit": "eV", "state": 1},
                  "context": {"solvent": None}, "limits": ["proxy only"], "status": "partial"}
                 for energy in (-2.5, 0, 1.1, 4.3, -0.01, 2.9)]
    ids = [memory.index.register_context(o) for o in originals]
    rows = memory.index.read(MemoryQuery(action="read_case_records", record_ids=ids, paths=["/metrics/energy"], view="auto"))["records"]
    for source, row in zip(originals, rows):
        record = row["record"]
        assert record["selected_fields"]["/metrics/energy"] == source["metrics"]["energy"]
        assert record["source_context"]["/metrics/unit"] == "eV"
        assert record["source_context"]["/metrics/state"] == 1
        assert record["source_context"]["context"] == {"solvent": None}
        assert record["source_context"]["limits"] == ["proxy only"]
    assert not read_needs_review({"records": rows})
