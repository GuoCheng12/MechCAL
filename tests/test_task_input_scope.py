from copy import deepcopy

import pytest

from mechcal.agents.command_tools import ask_tools, tool_contracts
from mechcal.runtime.command_compact import expand_text
from mechcal.runtime.command_loop import _apply, _comment
from mechcal.runtime.command_memory import digest, json_bytes, resolve_pointer
from mechcal.runtime.command_operations import OperationRejected, check_refs, protocol, put, read_records, record_index
from mechcal.runtime.command_review_context import prepare_review
from mechcal.runtime.command_task_context import planner_task_tools, prepare_planner
from mechcal.schemas.commands import ReadContext, ReadEvidence
from test_command_task_context import ready
from test_native_commands import NativeFake, reply


def repair_payload():
    return {"judgment_draft": {"draft_id": "draft-one", "issues": [{"code": "missing_ref"}]},
            "tool_catalog": {"tools": [{"id": "some-tool", "description": "route" * 1000}]},
            "worker_delegation": {"available": True},
            "exploration_state": {"query_turns_remaining": 2, "new_result_ids": [], "confidence_ranking": []}}


def test_planner_repair_schema_and_prompt_scope(tmp_path):
    c, o, s = ready(tmp_path)
    _, view = prepare_planner(c, repair_payload())
    client = NativeFake([reply("discard_judgment", {})])
    ask_tools(client, o.deps.capability_registry, [], view, s)
    request = client.requests[0]
    names = {t["name"] for t in request["tools"]}
    assert names == {"read_evidence", "reduce_evidence", "compare_evidence", "read_context",
                     "repair_judgment", "discard_judgment"}
    assert "planner_worker_tasks" not in request["system_prompt"]
    assert len(request["system_prompt"].encode()) < 6000
    assert "read_context" in request["system_prompt"]


def test_scientific_interface_can_be_restored_without_changing_science(tmp_path):
    c, o, s = ready(tmp_path)
    initial = repair_payload()
    c, view = prepare_planner(c, initial)
    saved = {key: deepcopy(protocol(c).get(key)) for key in ("rounds", "ranking", "results", "exploration")}
    c = _apply(o, c, ReadContext(kind="read_context", fields=["tool_catalog"]), s)
    c, view = prepare_planner(c, initial)
    assert view["planner_interface"] == "full"
    assert "tool_catalog" not in expand_text(view)
    assert protocol(c)["planner_context"]["fields"]["tool_catalog"] == initial["tool_catalog"]
    assert any(row["field"] == "tool_catalog" for row in view["deferred_context"])
    assert all(protocol(c).get(key) == data for key, data in saved.items())
    assert protocol(c)["context_reads"]["count"] == 1
    # A new source/catalog snapshot requires a new explicit expansion request.
    changed = deepcopy(initial)
    changed["tool_catalog"]["tools"][0]["description"] = "changed definition"
    _, view = prepare_planner(c, changed)
    assert view["planner_interface"] == "repair"


def test_task_filter_cannot_lock_out_discovery_or_expand_permissions():
    tools, models = tool_contracts(None, [], with_draft=True)
    assert planner_task_tools(tools, models, {"planner_interface": "repair"}) == (tools, models)
    tools, models = tool_contracts(None, [], with_draft=True, task_context=True,
                                  allowed_operations=["read_context", "repair_judgment"])
    filtered, selected = planner_task_tools(tools, models, {"planner_interface": "repair"})
    assert set(selected) == {t["name"] for t in filtered} == {"read_context", "repair_judgment"}


def test_repair_reads_name_stored_paths_not_display_paths(tmp_path):
    from mechcal.runtime.command_draft import draft_view
    c, _, _ = ready(tmp_path)
    body = {"case_id": c.case_id, "base_revision": protocol(c)["revision"],
            "command": {"changes": [{"kind": "argument", "id": "a1", "label": "RIM_RIR_RIV", "text": "exact text"}]},
            "issues": []}
    draft = {"draft_id": digest(body), **body}
    c = put(c, judgment_draft=draft)
    payload = {**repair_payload(), "judgment_draft": draft_view(c)}
    c, view = prepare_planner(c, payload)
    selection = expand_text(view)["judgment_draft"]["changes"][0]["read"]
    rows = read_records(c, ReadEvidence(kind="read_evidence", records=[selection]))
    assert rows[0]["data"][selection["paths"][0]] == draft["command"]["changes"][0]


def review_material():
    return {"arguments:arg1": {"label": "RIM_RIR_RIV", "text": "Check the actual conformer result."},
            "observations:one": {"energy": -12.3456789, "status": "failed"},
            "execution_context:one": {"charge": 0, "method": "exact", "status": "failed"},
            "citations:one": {"ref": "one", "path": "/energy", "value": -12.3456789},
            "review_workflow": {"issues": [{"id": "issue", "text": "Wrong source", "status": "open"}]},
            "tool_catalog": [{"id": "tool", "description": "complete limitations " * 100}],
            "candidate_overview": [{"label": "ICT_TICT_CT", "ranked": False, "confidence": .2,
                                    "note": "alternative consideration " * 100}],
            "evidence_followup": {"questions": [{"question": "undistinguished mechanism " * 100}]},
            "computational_review": {"basis": "basis", "executions": [{"result_id": "one", "status": "failed"}],
                "comparisons": [{"question": "scientific context " * 100}],
                "question_evidence": {"questions": [{"data": "exact source result " * 100}]},
                "candidate_comparison": {"explanation": "Full comparison " * 100}, "previous": None},
            "worker_interpretations": [{"source_observations": {"energy": -12.3456789}}]}


def test_review_deferral_is_exact_readable_owned_and_not_evidence(tmp_path):
    c, _, _ = ready(tmp_path)
    literal = review_material()
    material, receipt = {"sections": deepcopy(literal)}, {"sections": deepcopy(literal)}
    before = c.model_dump(mode="json")
    new, compact, ref = prepare_review(c, material, receipt, "reviewer")
    assert ref and c.model_dump(mode="json") == before
    assert receipt["sections"] == literal and material["sections"] == literal
    assert json_bytes(compact) < json_bytes(material)
    index = record_index(new)
    for item in compact["review_background"]["fields"]:
        rows = read_records(new, ReadEvidence(kind="read_evidence", records=[item["read"]]), index=index)
        expected = (literal["computational_review"][item["field"].split(".", 1)[1]]
                    if item["field"].startswith("computational_review.") else literal[item["field"]])
        assert rows[0]["data"][item["read"]["paths"][0]] == expected
    with pytest.raises(OperationRejected):
        check_refs(new, [ref], ["refs"], scientific=True, index=index)
    for name in ("arguments:arg1", "observations:one", "execution_context:one", "citations:one",
                 "review_workflow", "worker_interpretations"):
        assert compact["sections"][name] == literal[name]
    assert compact["candidate_directory"][0] == {"label": "ICT_TICT_CT", "ranked": False, "confidence": .2}
    assert compact["sections"]["computational_review"]["executions"] == literal["computational_review"]["executions"]
    assert protocol(new)["revision"] == protocol(c)["revision"]


def test_review_context_resumes_and_invalidates_on_changed_source(tmp_path):
    c, _, _ = ready(tmp_path)
    literal = review_material()
    new, view, ref = prepare_review(c, {"sections": literal}, {"sections": literal}, "reviewer")
    restored = type(new).model_validate_json(new.model_dump_json())
    assert record_index(restored).available[ref] == protocol(new)["review_context"]
    again, same, ref2 = prepare_review(restored, {"sections": literal}, {"sections": literal}, "reviewer")
    assert same == view and ref == ref2
    changed = deepcopy(literal)
    changed["observations:one"]["energy"] += 1
    fresh, _, ref3 = prepare_review(again, {"sections": changed}, {"sections": changed}, "reviewer")
    assert ref3 != ref and ref not in record_index(fresh).available
    for tamper in ({"case_id": "foreign"}, {"review_basis": "bad"}):
        broken = put(new, review_context={**protocol(new)["review_context"], **tamper})
        with pytest.raises(RuntimeError, match="Review context identity"):
            record_index(broken)


@pytest.mark.parametrize("role", ["final_reviewer", "coverage_reviewer", "worker"])
def test_other_roles_keep_original_packet(tmp_path, role):
    c, _, _ = ready(tmp_path)
    payload = {"sections": review_material()}
    new, projected, ref = prepare_review(c, payload, payload, role)
    assert new is c and projected == payload and ref is None


def test_small_and_reused_sections_are_not_replaced(tmp_path):
    c, _, _ = ready(tmp_path)
    sections = {"tool_catalog": {"previous_review_ref": "review:1", "path": "/sections/tool_catalog"},
                "computational_review": {"previous_review_ref": "review:1", "path": "/sections/computational_review"}}
    _, view, ref = prepare_review(c, {"sections": sections}, {"sections": review_material()}, "reviewer")
    assert view == {"sections": sections} and ref is None


def test_review_keeps_route_availability_and_exact_row_reads(tmp_path):
    c, _, _ = ready(tmp_path)
    sections = review_material()
    sections["tool_catalog"] = {"columns": ["capability_id", "description", "status", "missing_inputs"],
        "defaults": {}, "rows": [["tool-a", "Complete purpose " * 100, "disabled", []],
                                   ["tool-b", "Complete inputs " * 100, "missing_inputs", ["geometry"]]]}
    new, view, ref = prepare_review(c, {"sections": sections}, {"sections": sections}, "reviewer")
    directory = view["route_status_directory"]
    assert directory["rows"] == [[0, "tool-a", "disabled", []], [1, "tool-b", "missing_inputs", ["geometry"]]]
    path = directory["row_path_prefix"] + "1"
    result = read_records(new, ReadEvidence(kind="read_evidence", records=[{"ref": ref, "paths": [path]}]))
    assert result[0]["data"][path] == sections["tool_catalog"]["rows"][1]


def test_registered_review_background_survives_query_then_completion(tmp_path, monkeypatch):
    import mechcal.runtime.command_loop as loop
    c, o, s = ready(tmp_path)
    c = put(c, phase="review", review_role="reviewer", reviewer_session=None,
            review_changes=[{"kind": "argument", **a} for a in protocol(c)["arguments"].values()])
    catalog = [{"capability_id": "test-route", "description": "Exact limitations " * 200,
                "status": "disabled", "missing_inputs": [], "reason": "Test fixture"}]
    monkeypatch.setattr(loop, "_catalog", lambda *_: catalog)
    calls = []

    class Reviewer:
        def complete_tool_turn(self, **kw):
            calls.append(expand_text(kw["payload"]))
            if len(calls) == 1:
                entry = next(x for x in calls[-1]["review_background"]["fields"] if x["field"] == "tool_catalog")
                return reply("read_evidence", {"records": [entry["read"]]})
            assert calls[-1]["query_result"]["status"] == "read"
            return reply("record_review", {"text": "Source and current limitations checked.", "findings": []})

    o.deps.scientific_reviewer.llm_client = Reviewer()
    c = _comment(o, c, s)
    assert protocol(c)["reviewer_session"]["queries"] == 1
    assert protocol(c)["reviewer_session"]["outcome"]["status"] == "read"
    c = _comment(o, c, s)
    assert protocol(c)["review"]["status"] == "available"
    assert len(calls) == 2 and protocol(c)["reviewer_session"] is None
