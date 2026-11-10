"""Model edits, repair requests and stored records have consistent delta semantics."""

from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from mechcal.agents.command_tools import ask_tools, tool_contracts
from mechcal.runtime import command_followup as followup
from mechcal.runtime import command_review_workflow as workflow
from mechcal.runtime.command_draft import draft_view, repair
from mechcal.runtime.command_loop import _apply, planner_payload
from mechcal.runtime.command_operations import OperationRejected, protocol, put
from mechcal.schemas.commands import RepairJudgment, ReviseJudgment, UpdateHypothesesWithWorkflow
from test_command_draft_repair import reject_and_store
from test_command_followup import question
from test_command_protocol import LABELS, REGISTRY, case, orchestrator, rank, store
from test_native_commands import NativeFake, reply


def pending(tmp_path):
    o = orchestrator(tmp_path)
    c = workflow.enable(put(case(), results={"result:local": {
        "status": "success", "measurements": {"distance": 1.5, "overlap": .7}}}))
    s = store(tmp_path, c)
    cmd = ReviseJudgment(kind="revise_judgment", note="Original note.", request_review=True, changes=[
        {"kind": "argument", "id": "A1", "label": LABELS[0], "text": "Preserve the scientific text.",
         "refs": ["result:local"], "citations": [{"ref": "result:local", "paths": ["/measurements/wrong", "/measurements/overlap"]}]},
        rank(LABELS[0], ["A1"]), rank(LABELS[1]), rank(LABELS[2]),
    ])
    return o, reject_and_store(o, c, cmd, s), s


def path_patch(**kw):
    return {"kind": "argument_patch", "change_index": 0,
            "citation_path_edits": [{"citation_index": 0, "path_index": 0, "path": "/measurements/distance"}], **kw}


@pytest.mark.parametrize("metadata", [{}, {"note": None, "request_review": None},
                                      {"note": "Corrected reference.", "request_review": True},
                                      {"note": "", "request_review": False}])
def test_repair_metadata_and_one_path_survive_linked_translation_and_atomic_commit(tmp_path, metadata):
    o, c, s = pending(tmp_path)
    old = deepcopy(protocol(c)["judgment_draft"]["command"])
    args = {"edits": [path_patch()], **metadata}
    client = NativeFake([reply("repair_judgment", args)])
    command, _ = ask_tools(client, REGISTRY, [], planner_payload(o, c, native=True), s)
    proposed = repair(c, workflow.internal_judgment(c, command))
    expected = deepcopy(old)
    expected["changes"][0]["citations"][0]["paths"][0] = "/measurements/distance"
    expected.update({k: v for k, v in metadata.items() if v is not None})
    assert proposed.model_dump(mode="json") == expected
    done = _apply(o, c, command, s)
    assert protocol(done)["judgment_draft"] is None
    assert protocol(done)["revision"] == protocol(c)["revision"] + 1
    assert protocol(done)["arguments"]["A1"]["text"] == old["changes"][0]["text"]
    assert protocol(done)["results"] == protocol(c)["results"]
    assert protocol(done)["rounds"] == protocol(c)["rounds"]


def test_draft_points_to_exact_local_edit_without_choosing_replacement(tmp_path):
    _, c, _ = pending(tmp_path)
    view = draft_view(c)
    assert view["transaction"]["request_review"] is True
    issue = view["issues"][0]
    assert issue["repair_target"] == {"kind": "argument_patch", "change_index": 0,
        "field": "citation_path_edits", "citation_index": 0, "path_index": 0}
    assert issue["current_value"] == "/measurements/wrong"
    assert "path" not in issue["repair_target"]  # No guessed scientific source selection.


@pytest.mark.parametrize("fault", ["duplicate", "index", "unknown_path", "foreign_ref", "both", "extra"])
def test_bad_patch_never_changes_accepted_or_pending_science(tmp_path, fault):
    o, c, s = pending(tmp_path)
    edit = path_patch()
    args = {"kind": "repair_judgment", "draft_id": protocol(c)["judgment_draft"]["draft_id"], "edits": [edit]}
    if fault == "duplicate":
        edit["citation_path_edits"] *= 2
    elif fault == "index":
        edit["citation_path_edits"][0]["citation_index"] = 8
    elif fault == "unknown_path":
        edit["citation_path_edits"][0]["path"] = "/invented"
    elif fault == "foreign_ref":
        edit = {"kind": "argument_patch", "change_index": 0,
                "citations": [{"ref": "other-case:result", "paths": ["/measurements/distance"]}]}
        args["edits"] = [edit]
    elif fault == "both":
        edit["citations"] = []
    else:
        args["silently_ignore_me"] = True
    before = c.model_dump()
    with pytest.raises((OperationRejected, ValidationError)):
        _apply(o, c, RepairJudgment.model_validate(args), s)
    assert c.model_dump() == before


@pytest.mark.parametrize("strict", [False, True])
def test_native_repair_contract_declares_metadata_and_small_path_edits(strict):
    tools, _ = tool_contracts(REGISTRY, [], with_draft=True, review_workflow=True, strict_schema=strict, draft_id="bound")
    schema = next(t["parameters"] for t in tools if t["name"] == "repair_judgment")
    Draft202012Validator.check_schema(schema)
    assert {"note", "request_review"} <= schema["properties"].keys()
    assert "draft_id" not in schema["properties"]
    if not strict:
        Draft202012Validator(schema).validate({"edits": [path_patch()], "request_review": True})


def update(**changes):
    return UpdateHypothesesWithWorkflow(kind="update_hypotheses", note="Local question edit.", **changes)


def question_case():
    return followup.apply_update(followup.enable(case()), update(questions=[question(
        assessment="inconclusive", route={"capability_id": "macro.macro_structure_scan", "input_refs": ["input:smiles"]})]))


def test_existing_question_partial_update_retains_text_design_route_and_history():
    c = question_case()
    previous = followup.state(c).questions["compare"]
    done = followup.apply_update(c, update(questions=[{"id": "compare", "status": "deferred"}]))
    row = followup.state(done).questions["compare"]
    assert row.value.model_dump() == {**previous.value.model_dump(), "status": "deferred"}
    assert row.revision == previous.revision + 1
    assert row.trials == previous.trials
    # Null is no-change even when strict transports require spelling out all fields.
    unchanged = followup.apply_update(done, update(questions=[{"id": "compare", "question": None, "labels": None, "route": None}]))
    assert followup.state(unchanged) == followup.state(done)


def test_optional_question_fields_can_be_cleared_explicitly():
    c = question_case()
    done = followup.apply_update(c, update(questions=[{"id": "compare", "clear_fields": ["route", "assessment"]}]))
    row = followup.state(done).questions["compare"].value
    assert row.route is row.assessment is None
    assert row.question == followup.state(c).questions["compare"].value.question
    with pytest.raises(OperationRejected, match="Clear each field once"):
        followup.apply_update(c, update(questions=[{"id": "compare", "assessment": "distinguished", "clear_fields": ["assessment"]}]))


@pytest.mark.parametrize("question_id", ["new", "other_case_question"])
def test_incomplete_new_question_cannot_borrow_existing_scientific_text(question_id):
    c = question_case()
    before = c.model_dump()
    with pytest.raises(OperationRejected) as error:
        followup.apply_update(c, update(questions=[{"id": "compare", "status": "resolved"},
                                                   {"id": question_id, "status": "deferred"}]))
    assert {tuple(i["path"]) for i in error.value.issues} == {("questions", 1, "labels"), ("questions", 1, "question")}
    assert c.model_dump() == before
