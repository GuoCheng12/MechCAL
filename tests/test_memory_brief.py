"""Compact retrieval is exact projection, not a generated scientific summary."""

from copy import deepcopy

import pytest

from mechcal.runtime.case_memory import pointer, serialized
from mechcal.runtime.delta_session import read_needs_review
from mechcal.runtime.memory_brief import brief_entries, brief_record
from mechcal.runtime.memory_views import render_record, retrieved_working_set, lookup_memory
from mechcal.schemas.memory import MemoryQuery
from test_focused_memory import calculation, indexed


def test_brief_retains_quantities_conditions_and_failure_without_directory(tmp_path):
    _, _, index = indexed(tmp_path)
    row = index.read(MemoryQuery(action="read_case_records", record_ids=["X1"], view="brief"))["records"][0]
    record = row["record"]
    assert record["view"] == "brief" and record["complete_record"] is False
    assert record["next_offset"] is None and not read_needs_review({"records": [row]})
    assert record["selected_fields"]["/structured_results/parsed_observables/excited_states"] == pointer(
        calculation(), "/structured_results/parsed_observables/excited_states")
    assert "/structured_results/parsed_observables/dominant_transitions" in record["bulk_paths"]
    assert "Fixed geometry only" in serialized(record)
    assert "timeout" in serialized(record)
    assert "geometry:far" in serialized(record)
    assert len(serialized(record)) < 12000
    for path, value in record["selected_fields"].items():
        assert value == pointer(calculation(), path)


def test_brief_projects_nested_values_with_ancestor_conditions():
    source = {"metrics": {"block": {"unit": "eV", "state": 2, "limits": ["not measured"],
              "properties": {"negative": -2.5, "zero": 0, "missing": None, "table": list(range(5000))}}}}
    before = deepcopy(source)
    record = brief_record(source, ["/metrics/block/properties"], offset=0, budget=12000)
    for name in ("negative", "zero", "missing"):
        path = "/metrics/block/properties/" + name
        assert record["selected_fields"][path] == pointer(source, path)
    assert record["field_context"]["/metrics/block"] == {"unit": "eV", "state": 2, "limits": ["not measured"]}
    assert source == before


def test_many_values_page_exactly_without_hiding_unread_fields():
    original = {"values": {f"v{i:03}": i for i in range(200)}, "large": list(range(5000))}
    offset, seen, bulk = 0, {}, []
    while True:
        row = brief_record(original, [], offset=offset, budget=2500)
        assert len(serialized(row)) <= 2500
        seen.update(row["selected_fields"])
        bulk += row["bulk_paths"]
        if row["next_offset"] is None:
            break
        assert read_needs_review({"records": [{"record_id": "X", "record": row}]})
        assert row["next_offset"] > offset
        offset = row["next_offset"]
    assert seen == {f"/values/v{i:03}": i for i in range(200)}
    assert bulk == ["/large"]


def test_brief_never_drops_long_boundaries_to_fit():
    original = {"limits": ["unresolved " * 800], "metrics": {"x": 1}}
    record = brief_record(original, ["/metrics/x"], offset=0, budget=2000)
    assert not record["selected_fields"]
    assert record["error"] == "brief_context_exceeds_budget"
    assert read_needs_review({"records": [{"record_id": "X", "record": record}]})


@pytest.mark.parametrize("path", ["/missing", "/metrics/wrong"])
def test_missing_path_is_an_error_not_a_missing_measurement(path):
    record = brief_record({"metrics": {}}, [path], offset=0, budget=12000)
    assert path in record["field_errors"]
    assert read_needs_review({"records": [{"record_id": "X", "record": record}]})


def test_brief_and_full_have_distinct_contracts():
    source = {"data": {"distance": 2.5, "table": list(range(20000))}}
    q = MemoryQuery(action="read_case_records", record_ids=["X"], paths=["/data"], view="brief")
    record = render_record("X", "observation", serialized(source), q, focused=True)["record"]
    assert record["selected_fields"] == {"/data/distance": 2.5}
    assert record["bulk_paths"] == ["/data/table"]
    full = render_record("X", "observation", serialized(source), q.model_copy(update={"view": "full"}), focused=True)
    assert "json_fragment" in full


def test_escaped_paths_and_empty_values_are_exact():
    source = {"a/b~c": {"yes": False, "empty": [], "nested": {}}}
    for path, kind, value in brief_entries(source, []):
        assert kind == "value" and pointer(source, path) == value


def test_brief_does_not_mark_a_record_fully_loaded_or_hide_old_fields():
    old = {"records": [{"record_id": "X", "record": {"selected_fields": {"/different": 3}}}]}
    new = {"records": [{"record_id": "X", "record": {"view": "brief", "selected_fields": {"/a": 1},
           "bulk_paths": ["/big"], "next_offset": 3}}]}
    result = retrieved_working_set([old, new])
    assert len(result) == 2
    _, receipts = lookup_memory([new, old])
    assert receipts[0]["records"][0]["bulk_paths"] == ["/big"]
    assert receipts[0]["records"][0]["next_offset"] == 3


def test_brief_uses_shared_budget_and_keeps_all_small_records(tmp_path):
    _, _, index = indexed(tmp_path, rows=[calculation("X1"), calculation("X2")])
    response = index.read(MemoryQuery(action="read_case_records", record_ids=["X1", "X2"], view="brief"))
    assert all(row["record"]["next_offset"] is None for row in response["records"])
    assert sum(len(serialized(row["record"])) for row in response["records"]) <= 12000


def test_brief_rejects_fragment_cursor():
    with pytest.raises(ValueError, match="offset"):
        MemoryQuery(action="read_case_records", record_ids=["X"], view="brief", start=1)


def test_brief_empty_record_and_invalid_page_are_explicit():
    assert brief_record({}, [], offset=0, budget=12000)["selected_fields"] == {}
    invalid = brief_record({"x": 1}, [], offset=99, budget=12000)
    assert invalid["error"] == "offset_beyond_brief"
    assert read_needs_review({"records": [{"record_id": "X", "record": invalid}]})


def test_long_context_already_loaded_is_not_marked_as_unread_bulk():
    original = {"metrics": {"limitations": ["important limitation " * 25], "x": 0}}
    result = brief_record(original, [], offset=0, budget=12000)
    assert result["source_context"]["/metrics/limitations"] == original["metrics"]["limitations"]
    assert "/metrics/limitations" not in result["bulk_paths"]


def test_delta_auto_returns_values_without_an_additional_narrowing_call():
    source = {"metrics": {"analysis": {"distance": 2.5, "table": list(range(10000))}}, "limits": ["proxy"]}
    q = MemoryQuery(action="read_case_records", record_ids=["X"], paths=["/metrics/analysis", "/limits"])
    auto = render_record("X", "observation", serialized(source), q, focused=True, brief_auto=True)["record"]
    assert auto["view"] == "brief"
    assert auto["selected_fields"]["/metrics/analysis/distance"] == 2.5
    assert not read_needs_review({"records": [{"record_id": "X", "record": auto}]})
    old = render_record("X", "observation", serialized(source), q, focused=True)["record"]
    assert old["deferred_paths"] == ["/metrics/analysis"]
    context = render_record("X", "context", serialized(source), q, focused=True, brief_auto=True)["record"]
    assert context["deferred_paths"] == ["/metrics/analysis"]


def test_brief_measurement_error_is_not_a_protocol_error():
    original = {"metrics": {"measurement": {"error": 0.2}}}
    record = brief_record(original, ["/metrics/measurement"], offset=0, budget=12000)
    response = {"records": [{"record_id": "X", "record": record}]}
    assert not read_needs_review(response)
    _, receipts = lookup_memory([response, {"records": []}])
    assert receipts[0]["records"][0]["loaded_paths"] == ["/metrics/measurement"]
    assert "field_errors" not in receipts[0]["records"][0]
