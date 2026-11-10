"""Versioned issues are advisory; edits preserve sources and exact final review."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest
from jsonschema import Draft202012Validator

from mechcal.agents.command_tools import parse_tool_turn, tool_contracts
from mechcal.runtime import command_followup as f, command_hypotheses as h
from mechcal.runtime import command_review_workflow as w
from mechcal.runtime.command_delivery import amended_final, final_draft, reviewed, saved_final
from mechcal.runtime.command_loop import _apply, _comment, _hypothesis_followup, planner_payload, save
from mechcal.runtime.command_operations import OperationRejected, check_owner, protocol, put, record_index, revise
from mechcal.runtime.command_review import review_projection
from mechcal.runtime.command_views import candidate_view, computation_allowance, ordered_ranking
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import (
    AmendFinal, AmendJudgment, Finalize, RecordReview, SubmitFinal, UpdateHypothesesWithWorkflow,
    LinkedAmendJudgment, LinkedReviseJudgment, LinkedRepairJudgment,
)
from test_command_context_efficiency import expand, review_material, with_review
from test_command_protocol import LABELS, REGISTRY, TOOL, case, computed, judgment, orchestrator, rank, store
from test_command_hypotheses import update
from test_command_followup import question
from test_native_commands import NativeFake, reply


def setup(tmp_path):
    c, _ = h.update(w.enable(f.enable(computed())), update())
    c = revise(c, judgment())
    c = f.bind_formal(c, judgment().changes)
    o = orchestrator(tmp_path, reviews=True)
    o.config = replace(o.config, command_native_tools=True, command_hypothesis_tracking=True,
                       command_evidence_followup=True, command_review_workflow=True)
    return c, o, store(tmp_path, c)


def report(c, **updates):
    finding = {"target": "A1", "category": "fact", "status": "open",
               "text": "Attribute the number to its own source.", "source_refs": ["input:smiles"], **updates}
    return RecordReview(kind="record_review", text="Focused review, not a scientific score.", findings=[finding])


def accept(c, command):
    index = record_index(c)
    feedback = w.validate_report(c, command, targets=w.review_targets(c, index=index),
                                 source_refs=index.available, index=index)
    return w.accept_report(c, feedback)


def test_sorted_ranking_matches_delivery_without_mutating_storage(tmp_path):
    c, o, _ = setup(tmp_path)
    cmd = judgment().model_copy(deep=True)
    cmd.changes[-1].priority = .9
    c = revise(c, cmd)
    before = deepcopy(protocol(c)["ranking"])
    view = ordered_ranking(c)
    assert [r["label"] for r in view] == [h.name for h in c.portfolio.sorted_hypotheses()]
    assert [r["rank"] for r in view] == [1, 2, 3]
    assert planner_payload(o, c)["ranking"][0]["label"] == LABELS[2]
    assert protocol(c)["ranking"] == before


def test_one_current_argument_and_versioned_assessment_location(tmp_path):
    c, _, _ = setup(tmp_path)
    row = h.state(c).entries[LABELS[0]].model_dump(mode="json")
    v = candidate_view(c, row)
    assert "note" not in v and v["formal_matches_assessment"]
    old_version = v["argument_refs"][0]["version"]
    edit = AmendJudgment(kind="amend_judgment", edits=[{"id": "A1", "text": "A revised source-grounded account."}])
    changed = revise(c, w.amended_judgment(c, edit))
    assert candidate_view(changed, row)["argument_refs"][0]["version"] != old_version
    assert protocol(changed)["arguments"]["A1"]["refs"] == ["input:smiles"]
    newer, _ = h.update(changed, update(.7))
    current = h.view(newer)["confidence_ranking"][0]
    assert not current["formal_matches_assessment"] and "note" in current


def test_allowance_counts_only_scientific_computations():
    c = put(case(), rounds=16, revision=43)
    c = c.touch(runtime={**c.runtime, "run_limits": {"max_exploration_rounds": 20, "scientific_tools_allowed": True}})
    assert computation_allowance(c)["remaining"] == 4
    assert computation_allowance(c)["used"] == 16
    c.runtime["run_limits"].update(max_exploration_rounds=None, max_rounds=20)
    assert computation_allowance(c)["remaining"] == 4
    c.runtime["run_limits"]["unlimited_exploration"] = True
    assert computation_allowance(c)["remaining"] is None


def test_linked_wire_has_no_duplicate_rationale_and_argument_edit_updates_delivery(tmp_path):
    c, o, s = setup(tmp_path)
    tools, models = tool_contracts(REGISTRY, [], review_workflow=True, with_draft=True)
    for name in ("revise_judgment", "amend_judgment", "repair_judgment"):
        definition = next(t for t in tools if t["name"] == name)
        assert '"rationale"' not in json.dumps(definition["parameters"])
    ranking = {k: v for k, v in rank(LABELS[0], ["A1"]).items() if k != "rationale"}
    edit = LinkedAmendJudgment(kind="amend_judgment", edits=[{"id": "A1", "text": "Correct ratio with its source."}],
                               rankings=[ranking])
    changed = _apply(o, c, edit, s)
    assert all("rationale" not in r for r in ordered_ranking(changed))
    assert all("rationale" not in r for r in record_index(changed).available["judgment:ranking"]["entries"])
    hypothesis = next(h for h in changed.portfolio.hypotheses if h.name == LABELS[0])
    assert hypothesis.rationale == "Correct ratio with its source."
    final, _ = final_draft(changed, Finalize(kind="finalize", answer="A bounded comparison."))
    prediction = next(p for p in final.final_answer.mechanism_predictions if p.label == LABELS[0])
    assert prediction.evidence == ["Correct ratio with its source."]
    assert protocol(changed)["ranking"][LABELS[0]]["rationale"] == protocol(c)["ranking"][LABELS[0]]["rationale"]


def test_linked_ranking_creation_and_local_repair_share_the_same_contract(tmp_path):
    c, o, s = setup(tmp_path)
    value = {k: v for k, v in rank(LABELS[0], ["missing"]).items() if k != "rationale"}
    cmd = LinkedReviseJudgment(kind="revise_judgment", changes=[value])
    with pytest.raises(OperationRejected) as caught:
        _apply(o, c, cmd, s)
    c = put(c, judgment_draft=caught.value.judgment_draft)
    correction = LinkedRepairJudgment(kind="repair_judgment", draft_id=protocol(c)["judgment_draft"]["draft_id"],
                                      edits=[{"kind": "ranking_patch", "change_index": 0, "argument_ids": ["A1"]}])
    accepted = _apply(o, c, correction, s)
    assert protocol(accepted)["judgment_draft"] is None
    assert protocol(accepted)["ranking"][LABELS[0]]["argument_ids"] == ["A1"]


def test_open_question_disposition_may_be_absent_without_a_repair_call(tmp_path):
    c, o, s = setup(tmp_path)
    cmd = UpdateHypothesesWithWorkflow(kind="update_hypotheses", note="Untried comparison.",
        questions=[question(disposition_reason=None)])
    changed = _apply(o, c, cmd, s)
    assert f.state(changed).questions["compare"].value.disposition_reason is None


def test_issue_lifecycle_only_reopens_affected_versions(tmp_path):
    c, _, _ = setup(tmp_path)
    original = c.model_dump(mode="json")
    c = accept(c, report(c))
    iid = w.issue_view(c)[0]["id"]
    assert protocol(c)["arguments"] == original["runtime"]["command_protocol"]["arguments"]
    c = accept(c, report(c, id=iid, status="resolved", text="Corrected against source."))
    assert not w.issue_view(c)
    unrelated = put(c, results={"unrelated": {"status": "success", "summary": "Unrelated test record."}})
    assert not w.issue_view(unrelated)
    changed = revise(c, w.amended_judgment(c, AmendJudgment(kind="amend_judgment", edits=[{"id": "A1", "text": "Changed claim."}])))
    assert not w.issue_view(changed)
    assert w.prior_resolutions(changed)[0]["id"] == iid
    changed_source = c.touch(input=c.input.model_copy(update={"smiles": "CCN"}))
    assert w.issue_view(changed_source)[0]["status"] == "needs_recheck"
    saved = CaseRun.model_validate_json(changed.model_dump_json())
    assert w.issue_view(saved) == w.issue_view(changed)
    assert len(w.state(saved).issues[iid].history) == 1


@pytest.mark.parametrize("changes", [
    {"target": "foreign:argument"}, {"source_refs": ["foreign:result"]},
    {"id": "issue:foreign"}, {"status": "resolved"},
])
def test_review_bad_identity_is_local_and_atomic(tmp_path, changes):
    c, _, _ = setup(tmp_path)
    before = c.model_dump(mode="json")
    with pytest.raises(OperationRejected):
        accept(c, report(c, **changes))
    assert c.model_dump(mode="json") == before


def test_wrong_case_and_stale_report_cannot_be_committed(tmp_path):
    c, _, _ = setup(tmp_path)
    feedback = w.validate_report(c, report(c), targets=w.review_targets(c), source_refs=["input:smiles"])
    changed = c.touch(input=c.input.model_copy(update={"smiles": "CCN"}))
    with pytest.raises(RuntimeError, match="stale"):
        w.accept_report(changed, feedback)
    protocol(c)["review_workflow"]["case_id"] = "another-case"
    with pytest.raises(RuntimeError, match="ownership"):
        check_owner(c)


def test_existing_issue_cannot_change_target_or_duplicate(tmp_path):
    c, _, _ = setup(tmp_path)
    c = accept(c, report(c))
    iid = w.issue_view(c)[0]["id"]
    with pytest.raises(OperationRejected):
        accept(c, report(c, id=iid, target="judgment:ranking"))
    cmd = report(c, id=iid)
    cmd.findings *= 2
    with pytest.raises(OperationRejected):
        accept(c, cmd)


def test_findings_not_returned_are_retained_and_duplicate_report_is_idempotent(tmp_path):
    c, _, _ = setup(tmp_path)
    c = accept(c, report(c))
    again = accept(c, report(c))
    assert w.state(c) == w.state(again)
    no_changes = RecordReview(kind="record_review", text="No new errors.", findings=[])
    assert w.issue_view(accept(c, no_changes)) == w.issue_view(c)


def test_amendment_keeps_unrelated_records_and_all_links_atomic(tmp_path):
    c, o, s = setup(tmp_path)
    before = c.model_dump(mode="json")
    cmd = AmendJudgment(kind="amend_judgment", edits=[{"id": "A1", "text": "Corrected interpretation."}],
                        rankings=[{**rank(LABELS[0], ["A1"]), "confidence": .6}])
    new = _apply(o, c, cmd, s)
    assert protocol(new)["ranking"][LABELS[1]] == protocol(c)["ranking"][LABELS[1]]
    assert protocol(new)["arguments"]["A1"]["refs"] == ["input:smiles"]
    assert protocol(new)["arguments"]["A1"]["text"] == "Corrected interpretation."
    cmd.edits[0].refs = ["foreign:source"]
    with pytest.raises(OperationRejected):
        _apply(o, c, cmd, s)
    assert c.model_dump(mode="json") == before


@pytest.mark.parametrize("edits", [[{"id": "unknown", "text": "x"}], [{"id": "A1"}],
                                    [{"id": "A1", "text": "x"}, {"id": "A1", "text": "y"}]])
def test_invalid_accepted_edits_do_not_commit(tmp_path, edits):
    c, _, _ = setup(tmp_path)
    with pytest.raises(OperationRejected):
        w.amended_judgment(c, AmendJudgment(kind="amend_judgment", edits=edits))


def test_local_final_patch_requires_exact_new_review(tmp_path):
    c, o, s = setup(tmp_path)
    c = _apply(o, c, Finalize(kind="finalize", answer="One claim. One limitation."), s)
    draft = protocol(c)["final_draft"]
    c = put(c, final_review={"draft_id": draft["draft_id"], "status": "available"})
    edit = AmendFinal(kind="amend_final", draft_id=draft["draft_id"],
                      replacements=[{"old_text": "One claim.", "new_text": "One corrected claim."}])
    newer = _apply(o, c, edit, s)
    assert protocol(newer)["final_draft"]["command"]["answer"] == "One corrected claim. One limitation."
    assert not reviewed(newer, protocol(newer)["final_draft"])
    assert protocol(newer)["phase"] == "review"
    with pytest.raises(OperationRejected):
        saved_final(newer, draft["draft_id"])


@pytest.mark.parametrize("old", ["absent", "One"])
def test_ambiguous_or_missing_passage_cannot_partially_replace(tmp_path, old):
    c, o, s = setup(tmp_path)
    c = _apply(o, c, Finalize(kind="finalize", answer="One claim. One limitation."), s)
    before = c.model_dump(mode="json")
    edit = AmendFinal(kind="amend_final", draft_id=protocol(c)["final_draft"]["draft_id"],
                      replacements=[{"old_text": old, "new_text": "Changed"}])
    with pytest.raises(OperationRejected):
        amended_final(c, edit)
    assert c.model_dump(mode="json") == before


def test_new_observation_invalidates_final_review_even_if_text_unchanged(tmp_path):
    c, _, _ = setup(tmp_path)
    cmd = Finalize(kind="finalize", answer="A bounded conclusion.")
    _, before = final_draft(c, cmd)
    c = put(c, final_review={"draft_id": before["draft_id"], "status": "available"},
            results={"new": {"status": "failed"}})
    _, after = final_draft(c, cmd)
    assert before["draft_id"] != after["draft_id"] and not reviewed(c, after)


def test_live_comparison_routes_preserve_uncertainty_and_do_not_execute(tmp_path):
    c, o, _ = setup(tmp_path)
    c = put(c, results={"result:failed": {"status": "failed"}})
    cmd = UpdateHypothesesWithWorkflow(kind="update_hypotheses", note="A discriminating test remains untried.",
        questions=[question(route={"capability_id": TOOL}, disposition_reason="")])
    catalog = [{"capability_id": TOOL, "status": "missing_inputs", "missing_inputs": ["geometry"], "reason": "Need geometry"}]
    w.validate_routes(c, cmd, catalog)
    c, _ = h.update(c, cmd)
    option = w.comparison_options(c, catalog)[0]
    assert option["route_status"]["status"] == "missing_inputs" and not option["trials"]
    available = [{**catalog[0], "status": "ready", "missing_inputs": []}]
    assert w.comparison_options(c, available)[0]["route_status"]["status"] == "ready"
    assert protocol(c)["rounds"] == 0 and protocol(c)["ranking"][LABELS[0]]["confidence"] == .4
    restored = CaseRun.model_validate_json(c.model_dump_json())
    assert f.state(restored).questions["compare"].value.route.capability_id == TOOL
    assert "comparison_options" in planner_payload(o, restored)


@pytest.mark.parametrize("route", [{"capability_id": "unknown"}, {"capability_id": TOOL, "input_refs": ["foreign"]}])
def test_invalid_comparison_route_does_not_mutate(tmp_path, route):
    c, o, s = setup(tmp_path)
    cmd = UpdateHypothesesWithWorkflow(kind="update_hypotheses", note="Test", questions=[question(route=route)])
    before = c.model_dump(mode="json")
    with pytest.raises(OperationRejected):
        _apply(o, c, cmd, s)
    assert c.model_dump(mode="json") == before


def test_native_reviewer_records_issue_without_changing_science(tmp_path):
    c, o, s = setup(tmp_path)
    c = put(c, phase="review", review_role="reviewer", review_changes=[a.model_dump(mode="json") for a in judgment().changes])
    cmd = report(c)
    client = NativeFake([reply("record_review", cmd.model_dump(mode="json", exclude={"kind"}))])
    o.deps.photophysics_arbiter.llm_client = client
    after = _comment(o, c, s)
    assert w.issue_view(after)[0]["status"] == "open"
    assert protocol(after)["arguments"] == protocol(c)["arguments"]
    assert protocol(after)["ranking"] == protocol(c)["ranking"]
    payload = expand(client.requests[0]["payload"])["sections"]
    assert "review_targets" in payload and payload["ranking"][0]["rank"] == 1
    save(after, s)
    assert w.issue_view(CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())) == w.issue_view(after)


def test_workflow_native_contracts_and_combined_local_amendment(tmp_path):
    c, o, s = setup(tmp_path)
    tools, models = tool_contracts(REGISTRY, [TOOL], hypothesis_tracking=True, evidence_followup=True, review_workflow=True)
    for tool in tools:
        Draft202012Validator.check_schema(tool["parameters"])
    update_call = reply("update_hypotheses", {"changes": [], "note": "No change.", "responses": [], "questions": []})["calls"][0]
    patch = AmendJudgment(kind="amend_judgment", edits=[{"id": "A1", "text": "Fixed."}])
    patch_call = reply("amend_judgment", patch.model_dump(mode="json", exclude={"kind"}))["calls"][0]
    patch_call["id"] = "patch-call"
    exchange = {**reply("unused", {}), "calls": [update_call, patch_call]}
    command = parse_tool_turn(exchange, tools, models)
    c = _apply(o, put(c, latest_exchange=exchange), command, s)
    c = _hypothesis_followup(o, c, s)
    assert protocol(c)["arguments"]["A1"]["text"] == "Fixed."


def test_scientific_disagreement_is_not_a_finalization_gate(tmp_path):
    c, o, s = setup(tmp_path)
    c = accept(c, report(c, category="science", status="disagreement"))
    c = _apply(o, c, Finalize(kind="finalize", answer="The comparison remains unresolved."), s)
    fid = protocol(c)["final_draft"]["draft_id"]
    c = put(c, final_review={"draft_id": fid, "status": "available"})
    final = _apply(o, c, SubmitFinal(kind="submit_final", draft_id=fid), s)
    assert final.status == "finalized" and w.issue_view(final)


def test_unrelated_sources_do_not_resend_unchanged_arguments_but_source_changes_are_visible():
    c = w.enable(case())
    material = review_material()
    c, _, _ = with_review(c, material)
    c = put(c, results={"result:new": {"status": "success"}})
    payload, _, _ = review_projection(c, material, "reviewer", index=record_index(c))
    assert "arguments:A1" in payload["reused_sections"]
    assert "text" not in payload["previous_feedback"]
    changed = deepcopy(material)
    changed["observations"]["input:smiles"] = {"smiles": "CCN"}
    payload, _, _ = review_projection(c, changed, "reviewer", index=record_index(c))
    assert "observations:input:smiles" in payload["changed_sections"]


def test_partial_bad_review_then_local_correction_and_resume(tmp_path):
    c, o, s = setup(tmp_path)
    c = put(c, phase="review", review_role="reviewer", review_changes=[a.model_dump(mode="json") for a in judgment().changes])
    bad = report(c, target="foreign")
    client = NativeFake([reply("record_review", bad.model_dump(mode="json", exclude={"kind"})),
                        reply("repair_review_links", {"edits": [{"finding_index": 0, "field": "target", "value": "A1"}]})])
    o.deps.photophysics_arbiter.llm_client = client
    c = _comment(o, c, s)
    assert protocol(c)["reviewer_session"]["queries"] == 0 and not w.state(c).issues
    assert protocol(c)["reviewer_session"]["review_link_repair"]
    save(c, s)
    c = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    c = _comment(o, c, s)
    assert len(w.state(c).issues) == 1 and protocol(c)["phase"] == "choose"


def test_truncated_review_does_not_close_existing_issues(tmp_path):
    c, o, s = setup(tmp_path)
    c = accept(c, report(c))
    c = put(c, phase="review", review_role="reviewer", review_changes=[])
    broken = reply("record_review", {})
    broken["complete"] = False
    o.deps.photophysics_arbiter.llm_client = NativeFake([broken, broken, broken])
    after = c
    for _ in range(2):
        after = _comment(o, after, s)
        assert protocol(after)["phase"] == "review"
        assert protocol(after)["review_workflow"] == protocol(c)["review_workflow"]
    after = _comment(o, after, s)
    assert protocol(after)["review"]["status"] == "unavailable"
    assert w.state(after) == w.state(c)


def test_comparison_failure_and_disable_are_not_negative_results(tmp_path):
    c, _, _ = setup(tmp_path)
    cmd = UpdateHypothesesWithWorkflow(kind="update_hypotheses", note="Untried.",
        questions=[question(route={"capability_id": TOOL})])
    c, _ = h.update(c, cmd)
    c = put(c, results={"failed": {"status": "failed"}})
    c = f.record_trial(c, f.question_snapshot(c, "compare"), capability_id=TOOL, purpose="Test", result_id="failed")
    row = w.comparison_options(c, [{"capability_id": TOOL, "status": "disabled", "reason": "Ablation"}])[0]
    assert row["route_status"]["status"] == "disabled"
    assert row["trials"][0]["status"] == "failed" and row["status"] == "open"
    assert protocol(c)["ranking"][LABELS[0]]["confidence"] == .4
