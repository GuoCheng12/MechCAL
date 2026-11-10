"""Review savings preserve all edits, source ownership and exact finalization."""

from copy import deepcopy
import random

import pytest

from mechcal.runtime import command_review_workflow as workflow
from mechcal.runtime.command_memory import digest, json_bytes, resolve_pointer
from mechcal.runtime.command_delivery import reviewed
from mechcal.runtime.command_loop import _apply
from mechcal.runtime.command_operations import protocol, put, record_index
from mechcal.runtime.command_review import review_projection
from mechcal.runtime.command_review_delta import changes, retained_reads, text_edits
from mechcal.schemas.commands import AmendJudgment, Finalize, RecordReview
from test_command_context_efficiency import review_material, with_review
from test_command_protocol import case
from test_command_review_workflow import accept, report, setup


def apply_changes(value, edits):
    value = deepcopy(value)
    for edit in edits:
        path = edit["path"]
        if edit["op"] == "text_edits":
            old = resolve_pointer(value, path)
            for part in reversed(edit["edits"]):
                assert old[part["start"]:part["end"]] == part["before"]
                old = old[:part["start"]] + part["after"] + old[part["end"]:]
            updated = old
        else:
            updated = edit.get("value")
        if not path:
            value = updated
            continue
        parent_path, key = path.rsplit("/", 1)
        key = key.replace("~1", "/").replace("~0", "~")
        parent = resolve_pointer(value, parent_path)
        if edit["op"] == "remove":
            del parent[key]
        else:
            parent[key] = updated
    return value


@pytest.mark.parametrize("old,new", [
    ("", "new"), ("old", ""), ("  a\nb  ", "\ta b\n"),
    ("one two three", "one THIRTY three"), ("x " * 150, "x " * 100 + "y " + "x " * 50),
    ("Distance 3.20, not 3.02.\n" * 40, "Distance 3.20, not 3.02.\n" * 39 + "Removed qualification."),
])
def test_exact_text_edits_reconstruct_whitespace_and_all_changes(old, new):
    result = old
    for e in reversed(text_edits(old, new)):
        assert result[e["start"]:e["end"]] == e["before"]
        result = result[:e["start"]] + e["after"] + result[e["end"]:]
    assert result == new


def test_nested_delta_preserves_removal_numbers_lists_and_escaped_pointers():
    old = {"a/b": {"~n": .55, "gone": "do not retain"}, "list": [1, 2], "text": "Observation. " * 100}
    new = {"a/b": {"~n": .35, "added": None}, "list": [2, 1], "text": "Observation. " * 99 + "Changed evidence."}
    assert apply_changes(old, changes(old, new)) == new
    assert json_bytes(changes(old, new)) < json_bytes({"before": old, "after": new})


def test_randomized_delta_reconstruction():
    rng = random.Random(3)
    for _ in range(100):
        before = "".join(rng.choice(["number 3.2 ", "observed\n", "not ", "  "]) for _ in range(80))
        a, b = sorted(rng.sample(range(len(before) + 1), 2))
        after = before[:a] + "changed 4.5 " + before[b:]
        assert apply_changes(before, changes(before, after)) == after


def test_cross_role_reuse_and_exact_local_delta():
    c = workflow.enable(case())
    payload = review_material()
    c, receipt, _ = with_review(c, payload)
    new = deepcopy(payload)
    new.update(stage="command_final_reviewer", role="final_reviewer")
    new["arguments"][0]["text"] += " Now reports a limitation."
    material, current, ref = review_projection(c, new, "final_reviewer", index=record_index(c))
    block = material["sections"]["arguments:A1"]
    assert block["view"] == "exact_delta" and ref == receipt["ref"]
    assert apply_changes(receipt["sections"]["arguments:A1"], block["changes"]) == new["arguments"][0]
    assert current["sections"]["arguments:A1"] == new["arguments"][0]
    assert material["sections"]["ranking"] == new["ranking"]
    assert receipt["ref"] in material["review_receipt_refs"]


def test_prediction_scopes_reuse_two_unchanged_predictions():
    c = workflow.enable(case())
    payload = review_material()
    payload["assembled_answer"] = {"answer": "A bounded answer.", "mechanism_predictions": [
        {"label": name, "evidence": ["A long source-grounded statement. " * 60], "confidence": .5}
        for name in ["RIM", "ICT", "PACKING"]]}
    c, _, _ = with_review(c, payload)
    new = deepcopy(payload)
    new["assembled_answer"]["mechanism_predictions"][1]["confidence"] = .6
    material, _, _ = review_projection(c, new, "final_reviewer", index=record_index(c))
    assert "assembled_answer:prediction:RIM" in material["reused_sections"]
    assert "assembled_answer:prediction:PACKING" in material["reused_sections"]
    assert "assembled_answer:prediction:ICT" in material["delta_sections"]
    assert material["sections"]["assembled_answer:prediction_order"] == ["RIM", "ICT", "PACKING"]
    assert json_bytes(material) < json_bytes(payload)


def test_unavailable_other_role_receipt_cannot_be_reused():
    c, _, _ = with_review(workflow.enable(case()), review_material(), {"status": "unavailable"})
    material, _, _ = review_projection(c, review_material(), "final_reviewer", index=record_index(c))
    assert not material["reused_sections"] and not material["delta_sections"]


def test_shared_reads_keep_source_version_and_permissions():
    c = put(workflow.enable(case()), results={"R1": {"status": "success", "value": 1.2}})
    c, receipt, _ = with_review(c, review_material())
    receipt = {**receipt, "source_reads": [{"ref": "R1", "path": "/value", "page": None,
                                            "sha256": digest(1.2), "bytes": 3}],
               "read_versions": {"R1": digest(protocol(c)["results"]["R1"])}}
    c = put(c, review_memory={"reviewer": receipt}, review_receipts={receipt["ref"]: receipt})
    assert retained_reads(c, "final_reviewer", record_index(c), {"R1"}) == receipt["source_reads"]
    assert not retained_reads(c, "final_reviewer", record_index(c), set())
    changed = put(c, results={"R1": {"status": "success", "value": 9.9}})
    assert not retained_reads(changed, "final_reviewer", record_index(changed), {"R1"})


def scoped_case(tmp_path):
    c, _, _ = setup(tmp_path)
    c = accept(c, report(c, focus={"path": "/text", "quote": protocol(c)["arguments"]["A1"]["text"]}))
    issue = workflow.issue_view(c)[0]
    c = accept(c, report(c, id=issue["id"], status="resolved", focus=issue["focus"]))
    return c, issue


@pytest.mark.parametrize("edit", ["unrelated_field", "unrelated_sentence"])
def test_resolved_focus_survives_unrelated_changes(tmp_path, edit):
    c, issue = scoped_case(tmp_path)
    args = deepcopy(protocol(c)["arguments"])
    if edit == "unrelated_field":
        args["A1"]["limitations"] = ["A new independent limitation."]
    else:
        args["A1"]["text"] += " A separate statement."
    c = put(c, arguments=args)
    assert not workflow.issue_view(c)
    assert workflow.state(c).issues[issue["id"]].status == "resolved"


@pytest.mark.parametrize("change", ["quote", "source", "missing_target", "duplicate_quote"])
def test_changed_focus_or_source_requires_recheck(tmp_path, change):
    c, _ = scoped_case(tmp_path)
    args = deepcopy(protocol(c)["arguments"])
    if change == "source":
        c = c.touch(input=c.input.model_copy(update={"smiles": "CCN"}))
    elif change == "missing_target":
        del args["A1"]
        c = put(c, arguments=args)
    else:
        args["A1"]["text"] = "A different fact." if change == "quote" else args["A1"]["text"] * 2
        c = put(c, arguments=args)
    assert workflow.issue_view(c)[0]["status"] == "needs_recheck"


def test_invalid_optional_scope_keeps_review_but_cannot_narrow_reuse(tmp_path):
    c, _, _ = setup(tmp_path)
    original = c.model_dump(mode="json")
    command = report(c, focus={"path": "/text", "quote": "not in the target"})
    feedback = workflow.validate_report(c, command, targets=workflow.review_targets(c), source_refs=["input:smiles"])
    assert feedback["scope_warnings"][0]["code"] == "broad_issue_scope"
    assert feedback["findings"][0]["focus"] is None and feedback["text"] == command.text
    assert feedback["findings"][0]["text"] == command.findings[0].text
    assert c.model_dump(mode="json") == original
    assert workflow.issue_view(workflow.accept_report(c, feedback))


def test_repeated_closed_issue_does_not_create_new_history_or_feedback(tmp_path):
    c, issue = scoped_case(tmp_path)
    command = report(c, id=issue["id"], status="resolved", text="Rephrased but still resolved.", focus=issue["focus"])
    feedback = workflow.validate_report(c, command, targets=workflow.review_targets(c), source_refs=["input:smiles"])
    assert feedback["findings"] == [] and feedback["retained_resolutions"] == 1
    assert workflow.state(workflow.accept_report(c, feedback)) == workflow.state(c)


def test_omitted_issues_remain_open_and_new_fact_is_not_suppressed(tmp_path):
    c, _ = scoped_case(tmp_path)
    c = accept(c, report(c, text="A separate counterexample remains."))
    before = workflow.issue_view(c)
    c = accept(c, RecordReview(kind="record_review", text="No new finding.", findings=[]))
    assert workflow.issue_view(c) == before


def test_closed_broad_issue_is_context_but_changed_science_still_needs_final_review(tmp_path):
    c, o, store = setup(tmp_path)
    c = accept(c, report(c))
    iid = workflow.issue_view(c)[0]["id"]
    c = accept(c, report(c, id=iid, status="resolved"))
    c = _apply(o, c, Finalize(kind="finalize", answer="A bounded answer."), store)
    old = protocol(c)["final_draft"]
    c = put(c, phase="choose", final_review={"draft_id": old["draft_id"], "status": "available"})
    c = _apply(o, c, AmendJudgment(kind="amend_judgment", edits=[{"id": "A1", "text": "A changed claim."}]), store)
    assert not workflow.issue_view(c)
    assert workflow.prior_resolutions(c)[0]["id"] == iid
    c = _apply(o, c, Finalize(kind="finalize", answer="A bounded answer."), store)
    assert protocol(c)["phase"] == "review" and not reviewed(c, protocol(c)["final_draft"])
    changed = c.touch(input=c.input.model_copy(update={"smiles": "CCN"}))
    assert workflow.issue_view(changed)[0]["status"] == "needs_recheck"


def test_old_prediction_array_reuses_original_readable_paths():
    c = workflow.enable(case())
    value = {"label": "ICT", "evidence": ["Exact computed observations. " * 80]}
    payload = review_material()
    payload["assembled_answer"] = {"mechanism_predictions": [value]}
    c, receipt, _ = with_review(c, payload)
    legacy = deepcopy(receipt)
    del legacy["sections"]["assembled_answer:prediction:ICT"]
    del legacy["sections"]["assembled_answer:prediction_order"]
    legacy["sections"]["assembled_answer:mechanism_predictions"] = [value]
    c = put(c, review_memory={"reviewer": legacy}, review_receipts={legacy["ref"]: legacy})
    material, _, _ = review_projection(c, payload, "final_reviewer", index=record_index(c))
    pointer = material["sections"]["assembled_answer:prediction:ICT"]
    assert resolve_pointer(legacy, pointer["path"]) == value
    assert "assembled_answer:mechanism_predictions" not in material["removed_sections"]
