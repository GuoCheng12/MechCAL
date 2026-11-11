"""Smaller wire directories must preserve science and available retrieval routes."""

from copy import deepcopy
from dataclasses import replace

import pytest

from mechcal.runtime.command_compact import commentary_view, eligible_tools, expand_text, planner_view, table_rows
from mechcal.runtime.command_context import catalog_table, record_table, share_text
from mechcal.runtime.command_loop import _comment, planner_payload
from mechcal.runtime.command_operations import protocol, put, records
from mechcal.schemas.commands import RecordComputationalReview
from test_command_handoff import unchanged_science
from test_command_protocol import case, orchestrator
from test_computational_review import fixture
from test_native_commands import NativeFake, reply


def test_planner_preserves_beliefs_sources_and_new_result_preview(tmp_path):
    o = orchestrator(tmp_path)
    c = put(case(), results={"R_old": {"status": "failed", "capability_id": "test", "summary": "Old failure"},
                            "R_new": {"status": "success", "capability_id": "test", "summary": "New exact observation"}})
    before = expand_text(planner_payload(o, c, native=True))
    before["exploration_state"] = {"new_result_ids": ["R_new"], "confidence_ranking": [
        {"label": "ICT_TICT_CT", "confidence": .4, "note": "A bounded assessment", "refs": ["R_new"]}]}
    original = deepcopy(before)
    after = expand_text(planner_view(before))
    assert before == original
    for field in ("exploration_state", "ranking", "working_memory", "review_feedback", "judgment_draft"):
        assert after[field] == before[field]
    old_rows = {r["ref"]: r for r in table_rows(before["result_index"])}
    new_rows = {r["ref"]: r for r in table_rows(after["result_index"])}
    assert new_rows["R_new"] == old_rows["R_new"]
    assert not new_rows["R_old"].get("summary_preview")
    assert new_rows["R_old"]["status"] == "failed"
    assert new_rows["R_old"]["citation_uses"] == ["limitation"]
    assert set(old_rows) == set(new_rows) <= records(c).keys()


def test_question_deduplication_keeps_unique_route_and_trial_information(tmp_path):
    payload = planner_payload(orchestrator(tmp_path), case(), native=True)
    question = {"id": "Q1", "question": "Which state?", "outcomes": "Unresolved", "status": "open"}
    payload["exploration_state"] = {"evidence_followup": {"questions": [question]}}
    option = {**question, "route_status": {"status": "missing_inputs"}, "trials": [{"result_id": "R1"}]}
    payload["comparison_options"] = [option, {"id": "Q2", "question": "Not in the bounded question view"}]
    after = expand_text(planner_view(payload))
    compact = after["comparison_options"][0]
    assert {**question, **{k: v for k, v in compact.items() if k != "question_location"}} == option
    assert after["comparison_options"][1] == payload["comparison_options"][1]


def test_review_directory_is_the_exact_old_allowed_source_set():
    material = {"sections": {"reference_ids": ["R1"], "reference_contract": {"support_refs": ["R1"]},
                "review_targets": {"A1": "version"}, "arguments:A1": {"text": "Scientific text", "refs": ["R1"]}},
                "read_scope": record_table([{"ref": "R1", "citation_uses": ["support", "limitation"]},
                                             {"ref": "prior-review", "citation_uses": []}]),
                "read_path_sets": {"paths_1": ["/values"]}}
    original = deepcopy(material)
    compact = commentary_view(material)
    assert material == original
    assert {r["ref"] for r in table_rows(compact["read_scope"]) if r["review_source"]} == {"R1"}
    assert compact["sections"]["arguments:A1"] == material["sections"]["arguments:A1"]
    assert compact["read_path_sets"] == material["read_path_sets"]
    assert compact["sections"]["review_targets"] == ["A1"]


def test_compact_final_review_uses_same_submission_validator(tmp_path):
    c, o, s = fixture(tmp_path)
    o.config = replace(o.config, command_compact_context=True)
    command = RecordComputationalReview(kind="record_review", text="Bounded conclusion.",
        findings=[{"id": None, "target": "A1", "category": "science", "status": "open",
                   "text": "An unresolved interpretation", "source_refs": ["input:smiles"]}], next_computations=[])
    client = NativeFake([reply("record_review", command.model_dump(mode="json", exclude={"kind"}))])
    o.deps.scientific_reviewer.llm_client = client
    after = _comment(o, c, s)
    assert protocol(after)["review"]["status"] == "available"
    sent = expand_text(client.requests[0]["payload"])
    assert "reference_ids" not in sent["sections"]
    assert any(r["ref"] == "input:smiles" and r["review_source"] for r in table_rows(sent["read_scope"]))
    unchanged_science(c, after)


@pytest.mark.parametrize("remaining", [None, 0, 3])
def test_function_filter_only_removes_inapplicable_commands(remaining):
    names = ["compute_x", "describe_tools", "read_evidence", "update_hypotheses", "manage_experiments",
             "amend_judgment", "finalize", "amend_final", "submit_final", "revise_judgment"]
    tools = [{"name": n, "parameters": {"type": "object"}} for n in names]
    models = {n: object() for n in names}
    payload = {"context_profile": "compact_v1", "computation_allowance": {"remaining": remaining},
               "argument_index": {"rows": []}, "judgment_status": {"delivery_available": False}}
    selected, allowed = eligible_tools(tools, models, payload)
    expected = {"describe_tools", "read_evidence", "update_hypotheses", "revise_judgment"}
    if remaining != 0:
        expected.add("compute_x")
    assert set(allowed) == expected == {t["name"] for t in selected}
    assert all(t == tools[names.index(t["name"])] for t in selected)
    payload.update(final_draft={"draft_id": "D"}, pending_experiments=[{"id": "P"}],
                   argument_index={"rows": [["A"]]}, judgment_status={"delivery_available": True})
    selected, _ = eligible_tools(tools, models, payload)
    assert set(names) - {t["name"] for t in selected} == ({"compute_x"} if remaining == 0 else set())


def test_default_profile_is_unchanged():
    tools, models = [{"name": "finalize"}], {"finalize": object()}
    assert eligible_tools(tools, models, {}) == (tools, models)


def test_missing_and_disabled_capabilities_remain_discoverable(tmp_path):
    payload = planner_payload(orchestrator(tmp_path), case(), native=True)
    rows = [{"capability_id": "scan", "description": "A scan, not packing evidence.",
             "status": "missing_inputs", "missing_inputs": ["geometry"], "outputs": ["energies"]},
            {"capability_id": "s1", "description": "Excited-state scan", "status": "disabled", "reason": "Paused"}]
    payload["tool_catalog"] = catalog_table(rows)
    compact = table_rows(planner_view(payload)["tool_catalog"])
    for row, expected in zip(compact, rows):
        assert all(row[k] == v for k, v in expected.items())


def test_shared_text_references_survive_relocation(tmp_path):
    payload = planner_payload(orchestrator(tmp_path), case(), native=True)
    text = "A precise interpretation that must remain unchanged. " * 10
    payload.update(first=text, second=text)
    after = expand_text(planner_view(share_text(payload)))
    assert after["first"] == after["second"] == text
