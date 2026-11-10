"""Exact source handoff and byte-bounded reads, without external model calls."""

from dataclasses import replace

import pytest

from mechcal.agents.command_worker import scoped_commentary_turn
from mechcal.runtime import command_read_handoff as handoff
from mechcal.runtime.command_compact import expand_text, table_rows
from mechcal.runtime.command_impact_review import association_sources, review_payload, validate_suggestions
from mechcal.runtime.command_loop import _comment, planner_payload, save
from mechcal.runtime.command_memory import READ_BYTES, WORKING_BYTES, json_bytes, resolve_pointer, working_reads
from mechcal.runtime.command_operations import OperationRejected, protocol, put, read_records, records
from mechcal.runtime.command_reads import read_selection_results
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import ReadEvidence, ReportEvidenceImpacts
from test_command_followup import report
from test_command_handoff import prose, unchanged_science, worker_fixture
from test_command_protocol import case, orchestrator, store
from test_native_commands import NativeFake, reply

PATH = "/structured_results/parsed_observables/atoms"


def fixture():
    c, payload = worker_fixture()
    raw = protocol(c)["results"]["result:1"]
    raw.update(agent_name="microscopic", structured_results={
        "method": "Offline fixture", "limitations": ["Same source, not independent evidence."],
        "parsed_observables": {"atoms": [{"index": i, "hole": i / 1000, "electron": i / 2000,
            "unit": "dimensionless", "note": "Fixture, not science."} for i in range(31)]}})
    c = put(c, ticket={"request": payload["request"]}, outcome={"status": "computed"})
    return c


def read(paths=None, ref="result:1", **kw):
    return ReadEvidence(kind="read_evidence", records=[{"ref": ref, "paths": paths or [], **kw}])


def with_handoff(c):
    outcome, _ = read_selection_results(c, read([PATH]))
    return put(c, read_handoffs={"result:1": handoff.capture(c, "result:1", handoff.selection_handles(outcome))})


@pytest.mark.parametrize("explanation_available", [True, False])
def test_worker_reads_survive_handoff_resume_and_explanation_failure(tmp_path, explanation_available):
    c, o = fixture(), orchestrator(tmp_path)
    o.config = replace(o.config, command_native_tools=True, command_compact_context=True)
    answer = prose() if explanation_available else {**prose(), "complete": False, "text": ""}
    client = NativeFake([reply(arguments={"records": [{"ref": "result:1", "paths": [PATH]}]}), answer])
    worker = o.deps.workers["microscopic"]
    worker.report_writer.enable_llm, worker.report_writer.llm_client = True, client
    s = store(tmp_path, c)
    c = _comment(o, c, s)
    assert protocol(c)["worker_session"]["source_reads"][0]["path"] == PATH
    updated = save(_comment(o, c, s), s)
    restored = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    assert protocol(restored)["read_handoffs"] == protocol(updated)["read_handoffs"]
    memory = expand_text(planner_payload(o, restored, native=True))["working_memory"]
    exact = next(item for item in memory["findings"] if item["path"] == PATH)
    assert exact["data"] == resolve_pointer(protocol(c)["results"]["result:1"], PATH)
    assert exact["independence"] == "not_recorded" and exact["source_context"]
    assert json_bytes(memory["findings"]) <= WORKING_BYTES
    unchanged_science(c, updated)
    assert len(handoff.retained(restored)[0]) == 1  # Whole selection, not 31 fragments.
    assert protocol(restored)["explanations"]["result:1"]["status"] == (
        "available" if explanation_available else "unavailable")


def test_reviewer_gets_exact_source_values_before_any_new_read(tmp_path):
    c = with_handoff(fixture())
    client = NativeFake([prose()])
    scoped_commentary_turn(client, c, {}, store(tmp_path, c), scope=records(c), state={"queries": 0},
        role="reviewer", prompt_name="command_reviewer_text.md", compact_context=True)
    findings = expand_text(client.requests[0]["payload"])["working_memory"]["findings"]
    assert len(findings) == 1 and findings[0]["path"] == PATH
    assert len(findings[0]["data"]) == 31 and findings[0]["source_status"] == "success"


def test_explicit_reread_is_not_also_repeated_in_working_memory(tmp_path):
    c = with_handoff(fixture())
    client = NativeFake([reply(arguments={"records": [{"ref": "result:1", "paths": [PATH]}]}), prose()])
    kwargs = dict(scope=records(c), role="reviewer", prompt_name="command_reviewer_text.md", compact_context=True)
    s = store(tmp_path, c)
    _, session = scoped_commentary_turn(client, c, {}, s, state={"queries": 0}, **kwargs)
    scoped_commentary_turn(client, c, {}, s, state=session, **kwargs)
    assert not expand_text(client.requests[-1]["payload"])["working_memory"]["findings"]
    # The native tool message acknowledges delivery. Values appear once in the
    # current user document, rather than also in the tool output and memory.
    from mechcal.runtime.command_memory import resolve_pointer
    delivered = resolve_pointer(expand_text(client.requests[-1]["payload"]),
                                client.requests[-1]["outcome"]["result_in_current_user_at"])
    assert delivered["records"][0]["data"][PATH]


def test_partial_failures_directories_and_repeats_do_not_create_extra_evidence():
    c = fixture()
    outcome, _ = read_selection_results(c, read([PATH, "/missing"]))
    assert outcome["status"] == "partial_read"
    handles = handoff.selection_handles(outcome)
    assert len(handles) == 1 and handles[0]["path"] == PATH
    batch = {"status": "read_batch", "call_results": {"a": outcome, "b": outcome}}
    assert handoff.selection_handles(batch) == handles
    directory, _ = read_selection_results(c, read([PATH], view="directory"))
    assert not handoff.selection_handles(directory)
    rejected, _ = read_selection_results(c, read([PATH]), allowed_refs={"input:smiles"})
    assert not handoff.selection_handles(rejected)


def test_scope_source_version_and_case_isolation():
    c = with_handoff(fixture())
    assert not handoff.retained(c, allowed_refs={"input:smiles"})[0]
    assert not handoff.retained(c, result_ids={"other:1"})[0]
    changed = c.model_copy(deep=True)
    protocol(changed)["results"]["result:1"]["structured_results"]["method"] = "Changed conditions"
    handles, issues = handoff.retained(changed)
    assert not handles and issues[0]["status"] == "source_changed_or_missing"
    deleted = put(c, results={})
    assert not handoff.retained(deleted)[0]
    foreign = c.model_copy(deep=True)
    protocol(foreign)["read_handoffs"]["result:1"]["case_id"] = "foreign"
    with pytest.raises(RuntimeError, match="another case"):
        handoff.retained(foreign)
    assert handoff.retained(case()) == ([], [])


def test_failed_calculation_stays_failed_in_inherited_view():
    c = fixture()
    protocol(c)["results"]["result:1"]["status"] = "failed"
    c = with_handoff(c)
    handles, _ = handoff.retained(c)
    findings = working_reads(handles, records(c))["findings"]
    assert all(row["source_status"] == "failed" for row in findings)


def test_byte_pages_reconstruct_all_rows_and_preserve_exact_receipts():
    values = [{"index": i, "text": "x" * 3000} for i in range(70)]
    c = put(case(), results={"result:1": {"values": values}})
    before = c.model_dump(mode="json")
    offset, restored = 0, []
    while offset is not None:
        query = read(["/values"], page={"offset": offset, "limit": 32})
        outcome, handles = read_selection_results(c, query)
        assert outcome["status"] == "read" and json_bytes(outcome) <= READ_BYTES
        row = outcome["records"][0]
        actual = row["selection"]["page"]
        assert row["data"]["/values"] == values[offset:offset + actual["limit"]]
        restored.extend(row["data"]["/values"])
        for h in handles:
            assert resolve_pointer(records(c)[h["ref"]], h["path"]) is not None
        receipt = handoff.selection_handles(outcome)[0]
        assert receipt["page"] == actual
        if row.get("requested_page"):
            assert actual["limit"] < row["requested_page"]["limit"]
        offset = row["pages"]["/values"]["next_offset"]
    assert restored == values and c.model_dump(mode="json") == before
    with pytest.raises(OperationRejected):  # Citations still mean the exact requested range.
        read_records(c, read(["/values"], page={"offset": 0, "limit": 32}))


def test_oversized_nested_row_exposes_executable_child_selections():
    values = [{"state": 1, "components": [{"v": i, "details": "x" * 1200} for i in range(60)]}]
    c = put(case(), results={"result:1": {"values": values}})
    outcome, handles = read_selection_results(c, read(["/values"], page={"offset": 0, "limit": 6}))
    assert outcome["status"] == "rejected" and not handles
    issue = outcome["issues"][0]
    assert issue["code"] == "nested_page_required"
    fields = issue["details"]["record_layout"]["children"]
    child = next(item for item in fields if item["path"].endswith("/components"))
    nested, _ = read_selection_results(c, read(**child["read"]))
    assert nested["status"] == "read"
    query = read([child["path"]], page={"offset": 0, "limit": 32})
    nested, _ = read_selection_results(c, query)
    assert nested["records"][0]["pages"][child["path"]]["next_offset"]
    assert json_bytes(nested) <= READ_BYTES
    offset, rebuilt = 0, []
    while offset is not None:
        nested, _ = read_selection_results(c, read([child["path"]], page={"offset": offset, "limit": 32}))
        row = nested["records"][0]
        rebuilt.extend(row["data"][child["path"]])
        offset = row["pages"][child["path"]]["next_offset"]
    assert rebuilt == values[0]["components"]


def test_exhausted_page_budget_is_not_misreported_as_an_oversized_source():
    c = put(case(), results={"result:1": {"values": [1, 2]}})
    outcome, handles = read_selection_results(c, read(["/values"], page={"offset": 0, "limit": 2}), allowance=0)
    assert not handles and outcome["issues"][0]["code"] == "response_too_large"
    assert "no remaining" in outcome["issues"][0]["message"]


def test_directory_queries_preserve_escaped_keys_and_missing_paths_stay_invalid():
    c = put(case(), results={"result:1": {"a/b~c": [1.3, 2.4]}})
    outcome, _ = read_selection_results(c, read(view="directory"))
    child = outcome["records"][0]["data"][""]["children"][0]
    assert child["path"] == "/a~1b~0c"
    value, _ = read_selection_results(c, read(**child["read"]))
    assert value["records"][0]["data"][child["path"]] == [1.3, 2.4]
    invalid, _ = read_selection_results(c, read(["/a/b~c"]))
    assert invalid["issues"][0]["code"] == "unknown_path"


def test_impact_readable_and_associable_sets_have_one_definition(tmp_path):
    c = fixture()
    client = NativeFake([reply("report_evidence_impacts", report("result:1").model_dump(exclude={"kind"}))])
    feedback, _ = scoped_commentary_turn(client, c, review_payload(c, ["result:1"]), store(tmp_path, c),
        scope=records(c), state={"queries": 0}, role="impact_reviewer", prompt_name="command_impact_reviewer.md",
        completion=("report_evidence_impacts", ReportEvidenceImpacts,
                    lambda cmd: validate_suggestions(c, ["result:1"], cmd)), compact_context=True)
    rows = table_rows(expand_text(client.requests[0]["payload"])["read_scope"])
    assert {r["ref"] for r in rows if r["impact_source"]} == association_sources(c, ["result:1"])
    assert not next(r for r in rows if r["ref"] == "other:1")["impact_source"]
    assert feedback["status"] == "available"
    with pytest.raises(OperationRejected, match="new group"):
        validate_suggestions(c, ["result:1"], report("other:1"))
