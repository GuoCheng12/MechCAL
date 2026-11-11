"""Stopping advice is grounded in registry facts, not a scientific veto."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest
from jsonschema import Draft202012Validator

from mechcal.agents.command_tools import ask_tools, parse_tool_turn, tool_contracts
from mechcal.runtime import command_followup as f, command_hypotheses as h
from mechcal.runtime import computational_review as cr
from mechcal.runtime.command_loop import _apply, _catalog, _comment, planner_payload, save
from mechcal.runtime.command_operations import OperationRejected, protocol, put
from mechcal.runtime.command_review_workflow import comparison_options
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import (Finalize, RecordComputationalReview, SubmitFinalAfterReview,
                                     UpdateHypothesesWithWorkflow, envelope_model)
from test_command_context_efficiency import expand
from test_command_followup import question
from test_command_protocol import LABELS, REGISTRY, TOOL
from test_command_review_workflow import setup
from test_native_commands import NativeFake, reply


def fixture(tmp_path):
    c, o, s = setup(tmp_path)
    o.config = replace(o.config, command_computational_review=True)
    c = _apply(o, c, Finalize(kind="finalize", answer="A preliminary computational judgment."), s)
    return c, o, s


def report(**kw):
    return RecordComputationalReview(kind="record_review", text="A computational comparison, not experimental proof.",
                                    findings=[], next_computations=[], **kw)


def reviewed(c, o, suggestions):
    material = cr.context(c, _catalog(o, c))
    cmd = RecordComputationalReview.model_validate({**report().model_dump(), "next_computations": suggestions})
    feedback = cr.accept_review(c, cmd, {"status": "available"}, material, _catalog(o, c))
    return put(c, computational_review=feedback["computational_review"],
               final_review={"draft_id": protocol(c)["final_draft"]["draft_id"], "status": "available"})


def test_final_reviewer_sees_full_catalog_and_can_describe_without_scientific_mutation(tmp_path):
    c, o, s = fixture(tmp_path)
    original = deepcopy(protocol(c))
    fake = NativeFake([reply("describe_tools", {"capability_ids": [TOOL]}),
                       reply("record_review", report().model_dump(exclude={"kind"}), call_id="report")])
    o.deps.scientific_reviewer.llm_client = fake
    queried = _comment(o, c, s)
    payload = expand(fake.requests[0]["payload"])["sections"]
    assert payload["tool_catalog"] == expand(planner_payload(o, c, native=True))["tool_catalog"]
    assert "computational_review" in payload
    assert protocol(queried)["reviewer_session"]["outcome"]["status"] == "described"
    after = _comment(o, queried, s)
    for key in ("results", "arguments", "ranking", "rounds", "revision", "loaded_tools"):
        assert protocol(after).get(key) == original.get(key)
    assert not protocol(after)["review"]["read_record_ids"]
    assert protocol(after)["computational_review"]["suggestions"] == []
    assert cr.context(after, _catalog(o, after))["previous"]


@pytest.mark.parametrize("status,assessment,visible", [
    ("resolved", None, True), ("resolved", "inconclusive", True),
    ("resolved", "distinguished", False), ("deferred", "inconclusive", True)])
def test_question_disposition_does_not_hide_scientific_uncertainty(tmp_path, status, assessment, visible):
    c, o, _ = fixture(tmp_path)
    c, _ = h.update(c, UpdateHypothesesWithWorkflow(kind="update_hypotheses", note="Assessment separate from execution.",
        questions=[question(status=status, assessment=assessment)]))
    assert bool(comparison_options(c, _catalog(o, c))) is visible
    assert bool(f.view(c)["questions"]) is visible
    assert len(cr.context(c, _catalog(o, c))["comparisons"]) == 1
    restored = CaseRun.model_validate_json(c.model_dump_json())
    assert f.state(restored).questions["compare"].value.assessment == assessment


def test_suggestions_must_be_addressed_but_disagreement_does_not_change_ranking(tmp_path):
    c, o, s = fixture(tmp_path)
    c = reviewed(c, o, [{"capability_id": TOOL, "question_id": None, "purpose": "Check an alternative."}])
    fid = protocol(c)["final_draft"]["draft_id"]
    before = c.model_dump(mode="json")
    with pytest.raises(OperationRejected, match="Each distinct recommended capability"):
        _apply(o, c, SubmitFinalAfterReview(kind="submit_final", draft_id=fid), s)
    assert c.model_dump(mode="json") == before
    final = _apply(o, c, SubmitFinalAfterReview(kind="submit_final", draft_id=fid,
        route_responses=[{"capability_id": TOOL, "reason": "This repeats the same descriptors and cannot distinguish this comparison."}]), s)
    assert final.status == "finalized"
    assert protocol(final)["ranking"] == protocol(c)["ranking"]
    assert protocol(final)["results"] == protocol(c)["results"]
    assert len(protocol(final)["outcome"]["route_responses"]) == 1


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("suggestions", [[], [
    {"capability_id": TOOL, "question_id": None, "purpose": "Check an alternative."},
    {"capability_id": TOOL, "question_id": None, "purpose": "Compare a second geometry."},
]])
def test_submission_schema_matches_current_stop_requirements(tmp_path, strict, suggestions):
    c, o, s = fixture(tmp_path)
    c = reviewed(c, o, suggestions)
    tools, models = tool_contracts(REGISTRY, [], computational_review=True,
        strict_schema=strict, stopping_review=cr.context(c, _catalog(o, c)))
    schema = next(t["parameters"] for t in tools if t["name"] == "submit_final")
    Draft202012Validator.check_schema(schema)
    value = {"draft_id": protocol(c)["final_draft"]["draft_id"], "route_responses": []}
    if suggestions:
        assert list(Draft202012Validator(schema).iter_errors(value))
        value["route_responses"] = [{"capability_id": TOOL,
            "reason": "Neither comparison changes the preliminary judgment given the existing limitations."}]
    assert not list(Draft202012Validator(schema).iter_errors(value))
    command = parse_tool_turn(reply("submit_final", value), tools, models)
    final = _apply(o, c, command, s)
    assert final.status == "finalized"
    assert protocol(final)["results"] == protocol(c)["results"]
    assert protocol(final)["ranking"] == protocol(c)["ranking"]
    assert protocol(final)["arguments"] == protocol(c)["arguments"]
    assert protocol(final)["rounds"] == protocol(c)["rounds"]


def test_native_planner_receives_bound_submission_and_useful_missing_ids(tmp_path):
    c, o, s = fixture(tmp_path)
    c = reviewed(c, o, [{"capability_id": TOOL, "question_id": None, "purpose": "Compare alternative."}])
    payload = planner_payload(o, c, native=True)
    value = {"draft_id": protocol(c)["final_draft"]["draft_id"], "route_responses": []}
    fake = NativeFake([reply("submit_final", value)])
    with pytest.raises(OperationRejected):
        ask_tools(fake, REGISTRY, [], payload, s, allowed_operations=["submit_final"])
    schema = fake.requests[0]["tools"][0]["parameters"]
    assert schema["properties"]["route_responses"]["minItems"] == 1
    assert TOOL in schema["properties"]["route_responses"]["description"]
    with pytest.raises(OperationRejected) as error:
        _apply(o, c, SubmitFinalAfterReview(kind="submit_final", **value), s)
    assert TOOL in error.value.issues[0]["message"]
    assert "Compare alternative." in error.value.issues[0]["message"]


@pytest.mark.parametrize("responses", [
    [{"capability_id": "unknown", "reason": "Not needed."}],
    [{"capability_id": TOOL, "reason": "Not needed."}] * 2])
def test_unknown_and_duplicate_responses_are_local_errors(tmp_path, responses):
    c, o, s = fixture(tmp_path)
    c = reviewed(c, o, [{"capability_id": TOOL, "question_id": None, "purpose": "Check alternative."}])
    with pytest.raises(OperationRejected):
        _apply(o, c, SubmitFinalAfterReview(kind="submit_final", draft_id=protocol(c)["final_draft"]["draft_id"],
                                          route_responses=responses), s)
    assert c.status != "finalized"


def test_repeat_finalize_cannot_bypass_advice_acknowledgment(tmp_path):
    c, o, s = fixture(tmp_path)
    c = reviewed(c, o, [{"capability_id": TOOL, "question_id": None, "purpose": "Compare."}])
    with pytest.raises(OperationRejected):
        _apply(o, c, Finalize.model_validate(protocol(c)["final_draft"]["command"]), s)


def test_no_suggestions_does_not_force_more_calculations(tmp_path):
    c, o, s = fixture(tmp_path)
    c = reviewed(c, o, [])
    final = _apply(o, c, SubmitFinalAfterReview(kind="submit_final", draft_id=protocol(c)["final_draft"]["draft_id"]), s)
    assert final.status == "finalized" and protocol(final)["rounds"] == protocol(c)["rounds"]


def test_reuse_survives_wording_changes_but_not_new_results_catalog_or_candidate_change(tmp_path):
    c, o, _ = fixture(tmp_path)
    c = reviewed(c, o, [])
    catalog = _catalog(o, c)
    same = put(c, final_draft={**protocol(c)["final_draft"], "draft_id": "new-prose-only-id"})
    assert cr.context(same, catalog)["previous"]
    changed_catalog = deepcopy(catalog)
    changed_catalog[0]["status"] = "disabled"
    assert cr.context(c, changed_catalog)["previous"] is None
    results = {**protocol(c)["results"], "new": {"capability_id": TOOL, "status": "failed"}}
    assert cr.context(put(c, results=results), catalog)["previous"] is None
    exploration = deepcopy(protocol(c)["exploration"])
    exploration["entries"][LABELS[0]]["confidence"] = .99
    assert cr.context(put(c, exploration=exploration), catalog)["previous"] is None


def test_reuse_null_requires_current_receipt_and_cannot_erase_suggestions(tmp_path):
    c, o, _ = fixture(tmp_path)
    cmd = report().model_copy(update={"next_computations": None})
    with pytest.raises(OperationRejected):
        cr.accept_review(c, cmd, {}, cr.context(c, _catalog(o, c)), _catalog(o, c))
    c = reviewed(c, o, [{"capability_id": TOOL, "question_id": None, "purpose": "Compare."}])
    result = cr.accept_review(c, cmd, {}, cr.context(c, _catalog(o, c)), _catalog(o, c))
    assert result["computational_review"] == protocol(c)["computational_review"]


@pytest.mark.parametrize("has_previous", [False, True])
def test_review_wire_advertises_reuse_only_when_it_exists(tmp_path, has_previous):
    c, o, s = fixture(tmp_path)
    if has_previous:
        c = reviewed(c, o, [])
    fake = NativeFake([reply("record_review", report().model_dump(exclude={"kind"}))])
    o.deps.scientific_reviewer.llm_client = fake
    _comment(o, c, s)
    schema = next(t["parameters"] for t in fake.requests[0]["tools"] if t["name"] == "record_review")
    value = report().model_dump(exclude={"kind"})
    validator = Draft202012Validator(schema)
    assert not list(validator.iter_errors(value))
    value["next_computations"] = None
    assert bool(list(validator.iter_errors(value))) is not has_previous
    del value["next_computations"]
    assert "next_computations" in schema["required"]


@pytest.mark.parametrize("proposal", [
    {"capability_id": "foreign_tool", "purpose": "Compare."},
    {"capability_id": TOOL, "question_id": "foreign_case_question", "purpose": "Compare."}])
def test_unknown_proposals_are_not_saved(tmp_path, proposal):
    c, o, _ = fixture(tmp_path)
    cmd = RecordComputationalReview(kind="record_review", text="Check.", next_computations=[proposal])
    with pytest.raises(OperationRejected):
        cr.accept_review(c, cmd, {}, cr.context(c, _catalog(o, c)), _catalog(o, c))
    assert protocol(c).get("computational_review") is None


def test_cross_case_review_is_not_reused(tmp_path):
    c, o, _ = fixture(tmp_path)
    c = reviewed(c, o, [])
    receipt = {**protocol(c)["computational_review"], "case_id": "other"}
    with pytest.raises(RuntimeError, match="another case"):
        cr.context(put(c, computational_review=receipt), _catalog(o, c))


def test_native_wire_checkpoint_and_combined_update_use_same_response_schema(tmp_path):
    c, o, s = fixture(tmp_path)
    c = reviewed(c, o, [])
    tools, models = tool_contracts(REGISTRY, [], hypothesis_tracking=True, evidence_followup=True,
                                  review_workflow=True, computational_review=True)
    for tool in tools:
        Draft202012Validator.check_schema(tool["parameters"])
    response = {"draft_id": protocol(c)["final_draft"]["draft_id"], "route_responses": []}
    command = parse_tool_turn(reply("submit_final", response), tools, models)
    envelope = envelope_model(REGISTRY, [], hypothesis_tracking=True, evidence_followup=True,
                              review_workflow=True, computational_review=True)
    restored = envelope.model_validate_json(json.dumps({"command": command.model_dump(mode="json")})).command
    assert restored == command
    save(c, s)
    checkpoint = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    assert cr.context(checkpoint, _catalog(o, checkpoint))["previous"]
    assert _apply(o, checkpoint, restored, s).status == "finalized"


def test_unknown_description_is_read_only_and_shares_bounded_query_allowance(tmp_path):
    c, o, s = fixture(tmp_path)
    fake = NativeFake([reply("describe_tools", {"capability_ids": ["unknown"]}, call_id=str(i)) for i in range(2)] +
                      [reply("record_review", report().model_dump(exclude={"kind"}), call_id="finish")])
    o.deps.scientific_reviewer.llm_client = fake
    original = deepcopy(protocol(c)["results"])
    c = _comment(o, c, s)
    assert protocol(c)["reviewer_session"]["queries"] == 1
    assert protocol(c)["reviewer_session"]["outcome"]["status"] == "rejected"
    c = _comment(o, c, s)
    c = _comment(o, c, s)
    assert {t["name"] for t in fake.requests[-1]["tools"]} == {"record_review"}
    assert protocol(c)["results"] == original
