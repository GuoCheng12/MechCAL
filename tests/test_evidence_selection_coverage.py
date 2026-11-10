"""Complete-table checks and partial retrieval must remain distinguishable."""

from copy import deepcopy

import pytest

from mechcal.runtime import command_numeric as numeric
from mechcal.runtime.command_delivery_review import fact_packets
from mechcal.runtime.command_memory import READ_BYTES, json_bytes, working_reads
from mechcal.runtime.command_operations import OperationRejected, protocol, put, read_records, record_index
from mechcal.runtime.command_reads import query_results
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import ReadEvidence, ReduceEvidence
from test_command_protocol import case


PATH = "/structured_results/states/0/components"


def source_case(n=6000):
    rows = [{"from_orbital": 7, "to_orbital": i + 10, "coefficient": .0001} for i in range(n)]
    rows[-1] = {"from_orbital": 6, "to_orbital": 10, "coefficient": -.75}
    return put(case(), results={"R1": {"status": "success", "structured_results": {
        "states": [{"state_index": 3, "components": rows}]}}})


def read(**kwargs):
    return ReadEvidence(kind="read_evidence", records=[{"ref": "R1", "paths": [PATH], **kwargs}])


def reduction(**kwargs):
    return ReduceEvidence(kind="reduce_evidence", source={"ref": "R1", "path": PATH,
        "reduction": "max_abs", "value_path": "/coefficient", **kwargs})


def test_full_table_reduction_finds_signed_extremum_beyond_first_page_without_mutation():
    c = source_case()
    before = c.model_dump(mode="json")
    page = read_records(c, read(page={"offset": 0, "limit": 4}))[0]
    assert all(row["coefficient"] == .0001 for row in page["data"][PATH])
    assert page["coverage"][PATH] == {"scope": "partial_array", "complete": False,
                                    "offset": 0, "returned_rows": 4, "total_rows": 6000}
    result, handles = query_results(c, reduction(), allowed_refs={"R1"})
    check = result["numeric_check"]
    op = check["operands"][0]
    assert check["value"] == .75 and not handles
    assert op["considered_count"] == op["array_size"] == 6000
    assert op["selection_scope"] == "complete_array"
    assert op["selection_context"]["scalar_fields"]["state_index"] == 3
    assert op["matching_indices"] == [5999]
    assert op["matches"][0]["path"] == PATH + "/5999/coefficient"
    assert op["matches"][0]["value"] == -.75
    assert op["matches"][0]["context"]["scalar_fields"]["from_orbital"] == 6
    assert json_bytes(result) < READ_BYTES
    assert c.model_dump(mode="json") == before


def test_explicit_subset_is_not_reported_as_complete():
    result, _ = query_results(source_case(), reduction(row_indices=[0, 1]))
    op = result["numeric_check"]["operands"][0]
    assert op["value"] == .0001 and op["considered_count"] == 2
    assert op["selection_scope"] == "explicit_rows" and op["array_size"] == 6000


def test_tied_large_table_has_bounded_exact_match_metadata():
    c = source_case(100000)
    protocol(c)["results"]["R1"]["structured_results"]["states"][0]["components"][-1]["coefficient"] = -.0001
    result, _ = query_results(c, reduction())
    op = result["numeric_check"]["operands"][0]
    assert op["matching_count"] == 100000
    assert len(op["matching_indices"]) == 32 and op["omitted_matching_indices"] == 99968
    assert len(op["matches"]) == 4 and op["omitted_match_details"] == 99996
    assert json_bytes(result) < READ_BYTES


@pytest.mark.parametrize("bad", [None, "0.3", True, float("nan"), float("inf")])
def test_reduction_never_skips_invalid_rows(bad):
    c = source_case()
    protocol(c)["results"]["R1"]["structured_results"]["states"][0]["components"][3333]["coefficient"] = bad
    with pytest.raises(OperationRejected, match="finite numeric"):
        query_results(c, reduction())


def test_oversize_source_and_out_of_scope_are_local_errors():
    with pytest.raises(OperationRejected, match="100000"):
        query_results(source_case(100001), reduction())
    with pytest.raises(OperationRejected, match="outside"):
        query_results(source_case(), reduction(), allowed_refs=set())
    c = source_case()
    protocol(c)["results"]["R1"]["case_id"] = "foreign"
    with pytest.raises(RuntimeError, match="ownership"):
        query_results(c, reduction())


@pytest.mark.parametrize("offset,count", [(5998, 2), (6000, 0), (9000, 0)])
def test_last_page_and_empty_page_are_still_partial(offset, count):
    row = read_records(source_case(), read(page={"offset": offset, "limit": 4}))[0]
    assert row["pages"][PATH]["next_offset"] is None
    assert row["coverage"][PATH]["complete"] is False
    assert row["coverage"][PATH]["returned_rows"] == count


def test_directory_contains_no_data_and_small_whole_page_is_complete():
    c = source_case(3)
    row = read_records(c, read(view="directory"))[0]
    assert row["coverage"][PATH] == {"scope": "directory_only", "complete": False}
    assert "coefficient" not in str(row["data"][PATH]["children"])
    row = read_records(c, read(page={"offset": 0, "limit": 4}))[0]
    assert row["coverage"][PATH]["scope"] == "complete_array"


def test_partial_read_coverage_survives_old_style_handles_resume_and_reviewer_handoff():
    c = source_case()
    outcome, handles = query_results(c, read(page={"offset": 5, "limit": 2}))
    assert outcome["records"][0]["coverage"][PATH]["complete"] is False
    assert all("coverage" not in h for h in handles)  # Backward-compatible persisted format.
    c = put(c, read_memory=handles)
    restored = CaseRun.model_validate_json(c.model_dump_json())
    findings = working_reads(handles, record_index(restored).available)["findings"]
    assert findings[0]["coverage"]["offset"] == 5
    assert findings[0]["coverage"]["total_rows"] == 6000
    packets = fact_packets(restored, {"R1"})["packets"]
    assert packets[0]["coverage"] == findings[0]["coverage"]
    assert packets[0]["value"] == findings[0]["data"]
    assert not fact_packets(restored, {"foreign"})["packets"]


def test_read_valid_field_retained_but_missing_field_not_certified():
    query = read()
    query.records[0].paths = [PATH + "/0/coefficient", "/missing"]
    outcome, _ = query_results(source_case(), query)
    assert outcome["status"] == "partial_read"
    coverage = outcome["records"][0]["coverage"]
    assert list(coverage) == [PATH + "/0/coefficient"]
    assert coverage[PATH + "/0/coefficient"]["scope"] == "selected_value"


def test_numeric_receipt_reusable_until_original_source_changes():
    c = source_case()
    result, _ = query_results(c, reduction())
    c = numeric.remember(c, result["numeric_check"])
    restored = CaseRun.model_validate_json(c.model_dump_json())
    assert numeric.view(restored)["checks"][0]["value"] == .75
    rows = deepcopy(protocol(restored)["results"])
    rows["R1"]["structured_results"]["states"][0]["components"][-1]["coefficient"] = .8
    assert not numeric.view(put(restored, results=rows))["checks"]


def test_byte_limited_page_reports_actual_not_requested_coverage():
    c = source_case(20)
    rows = protocol(c)["results"]["R1"]["structured_results"]["states"][0]["components"]
    for row in rows:
        row["detail"] = "x" * 500
    result, handles = query_results(c, read(page={"offset": 3, "limit": 16}), allowance=4000)
    row = result["records"][0]
    count = len(row["data"][PATH])
    assert 0 < count < 16
    assert row["coverage"][PATH]["returned_rows"] == count
    assert row["coverage"][PATH]["total_rows"] == 20
    assert row["pages"][PATH]["next_offset"] == 3 + count
    retained = working_reads(handles, record_index(c).available)["findings"]
    for item in retained:
        if item["path"] == PATH:
            assert item["coverage"] == row["coverage"][PATH]


def test_native_tool_schema_and_parser_accept_new_reduction():
    from jsonschema import Draft202012Validator
    from mechcal.agents.command_tools import tool_contracts, parse_tool_turn
    from test_command_protocol import REGISTRY
    from test_native_commands import reply

    tools, models = tool_contracts(REGISTRY, [], allowed_operations={"reduce_evidence"})
    Draft202012Validator.check_schema(tools[0]["parameters"])
    response = reply("reduce_evidence", reduction().model_dump(exclude={"kind"}))
    parsed = parse_tool_turn(response, tools, models)
    assert parsed.source.reduction == "max_abs"
