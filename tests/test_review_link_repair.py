"""One local link correction must never regenerate or approve scientific claims."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from mechcal.runtime import review_link_repair as repair
from mechcal.runtime.command_loop import _comment, _review_reference_ids, save
from mechcal.runtime.command_operations import OperationRejected, protocol, put, record_index
from mechcal.runtime.command_review_workflow import state
from mechcal.runtime.command_review_workflow import review_targets, validate_report
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import RepairReviewLinks, RecordComputationalReview
from test_command_context_efficiency import expand
from test_command_handoff import unchanged_science
from test_computational_review import fixture
from test_native_commands import NativeFake, reply


def proposal(**kw):
    return RecordComputationalReview(kind="record_review", text="Preserve this scientific judgment verbatim.",
        findings=[{"id": "bad-id", "target": "A1", "category": "science", "status": "open",
                   "text": "This source supports a preliminary interpretation, not certainty.",
                   "source_refs": ["input:smiles"], **kw}], next_computations=[])


def correction(field="id", value=None):
    return {"edits": [{"finding_index": 0, "field": field, "value": value}]}


def test_visible_computational_sources_are_in_reference_contract():
    payload = {"computational_review": {"executions": [{"result_id": "R2"}, {"result_id": "foreign"}]},
               "assembled_answer": {"evidence_refs": ["R3"]}, "task": "R4"}
    assert _review_reference_ids(payload, {"R2": {}, "R3": {}, "R4": {}}) == ["R2", "R3"]


def test_computational_review_can_cite_unranked_execution(tmp_path):
    c, o, s = fixture(tmp_path)
    c = put(c, results={**protocol(c)["results"], "R_unranked": {
        "status": "success", "capability_id": "macro.macro_structure_scan", "summary": "An actual stored result."}})
    cmd = proposal(id=None, source_refs=["R_unranked"])
    client = NativeFake([reply("record_review", cmd.model_dump(mode="json", exclude={"kind"}))])
    o.deps.scientific_reviewer.llm_client = client
    after = _comment(o, c, s)
    assert protocol(after)["review"]["status"] == "available"
    assert "R_unranked" in expand(client.requests[0]["payload"])["sections"]["reference_ids"]
    unchanged_science(c, after)


def test_exhausted_reads_get_one_small_correction_without_planner_roundtrip(tmp_path):
    c, o, s = fixture(tmp_path)
    bad = proposal()
    client = NativeFake([reply(call_id="r1"), reply(call_id="r2"),
        reply("record_review", bad.model_dump(mode="json", exclude={"kind"}), call_id="bad"),
        reply("repair_review_links", correction(), call_id="repair")])
    o.deps.scientific_reviewer.llm_client = client
    before = c
    for _ in range(3):
        c = _comment(o, c, s)
    session = protocol(c)["reviewer_session"]
    assert session["queries"] == 2 and session["review_link_repair"]["proposal"] == bad.model_dump(mode="json")
    assert protocol(c)["phase"] == "review" and not state(c).issues
    save(c, s)
    c = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    c = _comment(o, c, s)
    assert protocol(c)["review"]["status"] == "available"
    assert protocol(c)["final_review"]["draft_id"] == protocol(c)["final_draft"]["draft_id"]
    assert protocol(c)["review"]["text"] == bad.text
    finding = next(iter(state(c).issues.values()))
    assert finding.text == bad.findings[0].text and finding.category == "science"
    request = client.requests[-1]
    assert {t["name"] for t in request["tools"]} == {"repair_review_links"}
    assert request["exchange"] is None
    assert "sections" not in request["payload"] and "read_scope" not in request["payload"]
    assert request["payload"]["valid_targets"]
    assert protocol(c)["review"]["read_record_ids"] == ["input:smiles"]
    unchanged_science(before, c)


@pytest.mark.parametrize("bad_value", ["foreign", "another-case:issue"])
def test_failed_local_repair_is_not_retried_or_approved(tmp_path, bad_value):
    c, o, s = fixture(tmp_path)
    client = NativeFake([reply("record_review", proposal().model_dump(mode="json", exclude={"kind"})),
                        reply("repair_review_links", correction(value=bad_value))])
    o.deps.scientific_reviewer.llm_client = client
    pending = _comment(o, c, s)
    after = _comment(o, pending, s)
    assert protocol(after)["review"]["status"] == "unavailable" and not state(after).issues
    assert protocol(after)["reviewer_session"] is None
    assert protocol(after)["final_review"]["status"] == "unavailable"
    unchanged_science(c, after)


@pytest.mark.parametrize("reply_kind", ["truncated", "compute", "rewrite", "duplicate", "out_of_range"])
def test_repair_boundary_failures_preserve_all_science(tmp_path, reply_kind):
    c, o, s = fixture(tmp_path)
    edit = correction()
    name, complete = "repair_review_links", True
    if reply_kind == "truncated":
        complete = False
    elif reply_kind == "compute":
        name = "compute_macro_macro_structure_scan"
    elif reply_kind == "rewrite":
        edit["edits"][0]["field"] = "text"
    elif reply_kind == "duplicate":
        edit["edits"] *= 2
    else:
        edit["edits"][0]["finding_index"] = 30
    client = NativeFake([reply("record_review", proposal().model_dump(mode="json", exclude={"kind"})),
                        reply(name, edit, complete=complete)])
    o.deps.scientific_reviewer.llm_client = client
    after = _comment(o, _comment(o, c, s), s)
    assert protocol(after)["review"]["status"] == "unavailable"
    assert not state(after).issues
    unchanged_science(c, after)


def test_patch_changes_only_associations_and_revalidates_entire_report():
    cmd = proposal()
    original = cmd.model_dump(mode="json")
    edited = repair.apply_links(type(cmd), original, RepairReviewLinks(kind="repair_review_links", **correction()))
    expected = deepcopy(original)
    expected["findings"][0]["id"] = None
    assert edited.model_dump(mode="json") == expected
    assert original == cmd.model_dump(mode="json")
    with pytest.raises(ValidationError):
        RepairReviewLinks(kind="repair_review_links", edits=[{"finding_index": 0, "field": "text", "value": "New science"}])
    with pytest.raises(OperationRejected):
        repair.apply_links(type(cmd), original, RepairReviewLinks(kind="repair_review_links", edits=correction()["edits"] * 2))


def test_compact_read_directory_preserves_ownership_and_exact_paths(tmp_path):
    c, o, s = fixture(tmp_path)
    client = NativeFake([reply("record_review", proposal(id=None).model_dump(mode="json", exclude={"kind"}))])
    o.deps.scientific_reviewer.llm_client = client
    _comment(o, c, s)
    payload = expand(client.requests[0]["payload"])
    table = payload["read_scope"]
    available = record_index(c).available
    for values in table["rows"]:
        row = {**table["defaults"], **dict(zip(table["columns"], values))}
        assert row["ref"] in available
        assert payload["read_path_sets"][row["path_set"]]


def test_all_link_errors_return_together_without_partial_commit(tmp_path):
    c, _, _ = fixture(tmp_path)
    first = proposal(source_refs=["foreign-1", "foreign-2"]).findings[0]
    second = first.model_copy(update={"id": "bad-2", "target": "foreign-target", "source_refs": []})
    cmd = proposal().model_copy(update={"findings": [first, second]})
    with pytest.raises(OperationRejected) as rejected:
        validate_report(c, cmd, targets=review_targets(c), source_refs=["input:smiles"])
    issues = rejected.value.issues
    assert len(issues) == 5
    assert {r["record_id"] for r in issues} == {"bad-id", "bad-2", "foreign-target", "foreign-1", "foreign-2"}
    assert repair.repairable(cmd, issues)
    assert not state(c).issues


def test_repair_preserves_issue_text_needed_to_distinguish_same_target():
    issue = {"id": "issue:1", "target": "A1", "category": "fact", "text": "Check state identity.",
             "source_refs": ["input:smiles"]}
    payload = {"sections": {"review_workflow": {"issues": [issue]}}}
    material = repair.repair_material(payload, {"proposal": {}, "issues": []})
    assert material["existing_issues"] == [issue]
