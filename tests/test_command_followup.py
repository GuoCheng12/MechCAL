"""Associations are durable work, not automatic scientific conclusions."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest
from jsonschema import Draft202012Validator

from mechcal.agents.command_tools import parse_tool_turn, tool_contracts
from mechcal.runtime import command_followup as f, command_hypotheses as h
from mechcal.runtime.command_experiments import next_member
from mechcal.runtime.command_impact_review import run_turn, validate_suggestions
from mechcal.runtime.command_loop import _apply, _execute, _hypothesis_followup, planner_payload, save
from mechcal.runtime.command_operations import OperationRejected, check_refs, prepare_computation, protocol, put, read_records
from mechcal.runtime.command_review import candidate_overview
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import (
    ReadEvidence, ReportEvidenceImpacts, ReviseJudgment, UpdateHypothesesWithFollowup,
    computation_model, envelope_model,
)
from test_command_protocol import REGISTRY, TOOL, LABELS, case, computed, judgment, orchestrator, store
from test_command_hypotheses import update
from test_exploration_batches import batch_reply, drain
from test_native_commands import NativeFake, reply


def enabled(tmp_path):
    o = orchestrator(tmp_path)
    o.config = replace(o.config, command_native_tools=True, command_hypothesis_tracking=True,
                       command_evidence_followup=True)
    return o


def observed():
    c, _ = h.update(f.enable(computed()), update())
    return put(c, results={"result:new": {"tool_result_id": "result:new", "status": "success",
        "summary": "Offline fixture, not a scientific observation.", "structured_results": {"test": 1}}})


def report(ref="result:new", labels=None):
    return ReportEvidenceImpacts(kind="report_evidence_impacts", text="Offline association fixture.", suggestions=[
        {"label": label, "refs": [ref], "reason": "Reconsider the current assessment."} for label in (labels or LABELS[:2])])


def associated(c=None):
    c = c or observed()
    return f.accept_review(c, ["result:new"], validate_suggestions(c, ["result:new"], report()))


def question(**changes):
    return {"id": "compare", "labels": LABELS[:2], "question": "Which explanation does this test constrain?",
            "next_test": "An explicit available computation.", "outcomes": "It may support either, both, or neither.",
            "status": "open", **changes}


def assessment(**changes):
    return UpdateHypothesesWithFollowup(kind="update_hypotheses", note="Bounded assessment.", **changes)


def test_overall_assessment_does_not_clear_unhandled_candidates():
    c = associated()
    c, _ = h.update(c, update(.6))
    assert not h.view(c)["assessment_required"]
    assert len(f.view(c)["pending_impacts"]) == 2
    assert all(item.disposition == "pending" for item in f.state(c).impacts.values())
    assert not f.unreviewed_results(c)


@pytest.mark.parametrize("disposition", ["reviewed", "deferred", "dismissed"])
def test_explicit_response_does_not_force_score_movement(disposition):
    c = associated()
    iid = next(iter(f.state(c).impacts))
    old = h.state(c).entries
    c, _ = h.update(c, assessment(responses=[{"impact_id": iid, "disposition": disposition,
                                            "note": "The observation does not distinguish this candidate."}]))
    assert h.state(c).entries == old
    assert f.state(c).impacts[iid].disposition == disposition
    assert len(f.view(c)["pending_impacts"]) == (2 if disposition == "deferred" else 1)
    assert protocol(c)["ranking"] == {} and protocol(c)["rounds"] == 0


@pytest.mark.parametrize("bad", ["unknown", "duplicate", "question_duplicate"])
def test_followup_update_is_atomic_with_scientific_change(bad):
    c = associated()
    iid = next(iter(f.state(c).impacts))
    responses = [{"impact_id": "foreign:impact" if bad == "unknown" else iid,
                  "disposition": "reviewed", "note": "Checked."}]
    if bad == "duplicate": responses *= 2
    q = [question(), question()] if bad == "question_duplicate" else []
    before = c.model_dump()
    with pytest.raises(OperationRejected):
        h.update(c, assessment(changes=update(.8).changes, responses=responses, questions=q))
    assert c.model_dump() == before


def test_response_and_question_survive_resume_without_reopening(tmp_path):
    c = associated()
    iid = next(iter(f.state(c).impacts))
    command = assessment(responses=[{"impact_id": iid, "disposition": "reviewed", "note": "No score change."}],
                         questions=[question()])
    c, _ = h.update(c, command)
    s = store(tmp_path, c)
    save(c, s)
    restored = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    same, receipt = h.update(restored, command)
    assert receipt["status"] == "unchanged"
    assert f.state(same) == f.state(c) and not f.unreviewed_results(same)
    assert f.state(case()) is None


def test_failed_result_creates_only_a_question_not_negative_evidence():
    c = put(observed(), results={"result:new": {"status": "failed"}})
    c = associated(c)
    c, _ = h.update(c, assessment())
    assert h.state(c).entries[LABELS[0]].confidence == .4
    with pytest.raises(OperationRejected, match="not scientific evidence"):
        check_refs(c, ["result:new"], [], scientific=True)
    assert f.view(c)["pending_impact_count"] == 2


@pytest.mark.parametrize("ref", ["input:smiles", "foreign:result", "explanation:result:new", "call:test"])
def test_reviewer_cannot_associate_from_wrong_source(ref):
    with pytest.raises(OperationRejected, match="new group"):
        validate_suggestions(observed(), ["result:new"], report(ref))


def test_cross_case_ownership_and_advice_not_citable():
    c = associated()
    with pytest.raises(OperationRejected, match="not a raw observation"):
        check_refs(c, ["followup:state"], [], scientific=True)
    rows = read_records(c, ReadEvidence(kind="read_evidence", records=[{"ref": "followup:state", "paths": ["/impacts"]}]))
    assert rows
    protocol(c)["evidence_followup"]["case_id"] = "other"
    with pytest.raises(RuntimeError, match="ownership"):
        f.state(c)


def test_trial_links_exact_question_and_does_not_close_it(tmp_path):
    c, _ = h.update(f.enable(computed()), assessment(questions=[question()]))
    o, s = enabled(tmp_path), store(tmp_path, c)
    model = computation_model(REGISTRY.require(TOOL), with_question=True)
    command = model(kind="compute", capability_id=TOOL, arguments={}, purpose="Offline test.", question_id="compare")
    unlinked = command.model_copy(update={"question_id": None})
    assert prepare_computation(c, command, REGISTRY, amesp_enabled=False)[1] == prepare_computation(c, unlinked, REGISTRY, amesp_enabled=False)[1]
    c = _execute(o, _apply(o, c, command, s), s)
    record = f.state(c).questions["compare"]
    assert record.value.status == "open" and len(record.trials) == 1
    assert record.trials[0]["result_id"] in protocol(c)["results"]
    assert "question_id" not in protocol(c)["actions"][record.trials[0]["dispatch_id"]]["tool_args"]
    c = put(c, phase="choose")
    c, _ = h.update(c, assessment(questions=[question(question="A refined comparison.")]))
    assert f.state(c).questions["compare"].revision == 2
    c = _apply(o, c, command, s)
    assert protocol(c)["outcome"]["status"] == "reused" and protocol(c)["rounds"] == 1
    trials = f.state(c).questions["compare"].trials
    assert [t["revision"] for t in trials] == [1, 2]
    assert trials[0]["question"]["question"] != trials[1]["question"]["question"]


def test_unknown_question_does_not_execute(tmp_path):
    c, _ = h.update(f.enable(computed()), assessment())
    command = computation_model(REGISTRY.require(TOOL), with_question=True)(
        kind="compute", capability_id=TOOL, arguments={}, purpose="Offline", question_id="foreign")
    with pytest.raises(OperationRejected, match="not in this case"):
        _apply(enabled(tmp_path), c, command, store(tmp_path, c))
    assert protocol(c)["results"] == {}


def test_single_source_for_candidate_prose_and_formal_staleness(tmp_path):
    o, c = enabled(tmp_path), f.enable(case())
    c, _ = h.update(c, update())
    c = put(c, candidates={LABELS[0]: {"note": "Stale legacy prose."}})
    s = store(tmp_path, c)
    c = _apply(o, c, judgment(), s)
    assert not f.view(c)["stale_formal_labels"]
    c, _ = h.update(c, update(.2, note="New evidence changes this judgment."))
    assert f.view(c)["stale_formal_labels"] == [LABELS[0]]
    assert candidate_overview(c)[0]["note"] != "Stale legacy prose."
    candidate = ReviseJudgment(kind="revise_judgment", changes=[{"kind": "candidate", "label": LABELS[0],
        "status": "considering", "note": "Corrected interpretation.", "refs": ["input:smiles"]}])
    c = _apply(o, c, candidate, s)
    assert h.state(c).entries[LABELS[0]].note == "Corrected interpretation."
    assert next(r for r in candidate_overview(c) if r["label"] == LABELS[0])["note"] == "Corrected interpretation."
    assert f.view(c)["stale_formal_labels"] == [LABELS[0]]


def test_native_contract_and_checkpoint_carry_questions_without_json_strings(tmp_path):
    o, c = enabled(tmp_path), f.enable(computed())
    update_call = reply("update_hypotheses", assessment(questions=[question()]).model_dump(mode="json", exclude={"kind"}), call_id="update")
    model = computation_model(REGISTRY.require(TOOL), with_question=True)
    data = model(kind="compute", capability_id=TOOL, purpose="Offline", arguments={}, question_id="compare")
    compute_call = reply("compute_" + TOOL.replace(".", "_"), data.model_dump(mode="json", exclude={"kind", "capability_id"}), call_id="run")
    incoming = batch_reply(update_call, compute_call)
    tools, models = tool_contracts(REGISTRY, [TOOL], hypothesis_tracking=True, evidence_followup=True)
    for tool in tools:
        Draft202012Validator.check_schema(tool["parameters"])
    parsed = parse_tool_turn(incoming, tools, models)
    s = store(tmp_path, c)
    c = save(_apply(o, put(c, latest_exchange=incoming), parsed, s), s)
    c = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    c = _hypothesis_followup(o, c, s)
    assert protocol(c)["phase"] == "execute" and protocol(c)["ticket"]["question"]["question_id"] == "compare"
    saved_model = envelope_model(REGISTRY, [TOOL], hypothesis_tracking=True, evidence_followup=True)
    assert saved_model.model_validate({"command": data.model_dump()}).command.question_id == "compare"
    assert saved_model.model_validate({"command": assessment(questions=[question()]).model_dump()}).command.questions


def test_batch_links_questions_and_reuses_identical_computations(tmp_path):
    o, c = enabled(tmp_path), f.enable(computed())
    c, _ = h.update(c, assessment(questions=[question()]))
    model = computation_model(REGISTRY.require(TOOL), with_question=True)
    data = model(kind="compute", capability_id=TOOL, purpose="Offline", arguments={}, question_id="compare")
    args = data.model_dump(mode="json", exclude={"kind", "capability_id"})
    incoming = batch_reply(*(reply("compute_" + TOOL.replace(".", "_"), args, call_id=i) for i in ["a", "b"]))
    tools, models = tool_contracts(REGISTRY, [TOOL], evidence_followup=True)
    s = store(tmp_path, c)
    c = _apply(o, put(c, latest_exchange=incoming), parse_tool_turn(incoming, tools, models), s)
    c = drain(o, c, s)
    assert protocol(c)["rounds"] == 1 and len(f.state(c).questions["compare"].trials) == 2


def test_impact_review_is_bounded_read_only_and_not_rescheduled(tmp_path):
    c, o = observed(), enabled(tmp_path)
    before = c.model_dump()
    client = NativeFake([reply("report_evidence_impacts", report().model_dump(exclude={"kind"}))])
    o.deps.photophysics_arbiter.llm_client = client
    c = run_turn(o, put(c, phase="impact_review"), store(tmp_path, c))
    assert f.view(c)["pending_impact_count"] == 2 and not f.unreviewed_results(c)
    for field in ["portfolio", "evidence_ledger", "artifact_manifest"]:
        assert c.model_dump()[field] == before[field]
    for field in ["rounds", "revision", "results", "ranking", "arguments", "exploration"]:
        assert protocol(c)[field] == before["runtime"]["command_protocol"][field]
    assert len(client.requests) == 1
    assert {t["name"] for t in client.requests[0]["tools"]} == {
        "read_evidence", "reduce_evidence", "compare_evidence", "report_evidence_impacts"}
    c = run_turn(o, c, store(tmp_path, c))
    assert len(client.requests) == 1


def test_impact_read_and_bad_ref_correction_survive_resume(tmp_path):
    c, o = observed(), enabled(tmp_path)
    bad = report("foreign:result").model_dump(exclude={"kind"})
    client = NativeFake([reply(arguments={"records": [{"ref": "result:new", "paths": ["/status"]}]}, call_id="read"),
                        reply("report_evidence_impacts", bad, call_id="bad"),
                        reply("report_evidence_impacts", report().model_dump(exclude={"kind"}), call_id="good")])
    o.deps.photophysics_arbiter.llm_client = client
    s = store(tmp_path, c)
    c = save(run_turn(o, put(c, phase="impact_review"), s), s)
    assert protocol(c)["impact_session"]["queries"] == 1
    c = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    c = run_turn(o, c, s)
    assert protocol(c)["impact_session"]["queries"] == 1
    assert protocol(c)["impact_session"]["corrections"] == 1
    c = run_turn(o, c, s)
    assert protocol(c)["impact_session"] is None
    assert f.view(c)["pending_impact_count"] == 2 and len(client.requests) == 3
    assert {t["name"] for t in client.requests[-1]["tools"]} == {
        "read_evidence", "reduce_evidence", "compare_evidence", "report_evidence_impacts"}


@pytest.mark.parametrize("mode", ["truncated", "invalid", "text"])
def test_unavailable_review_is_preserved_without_infinite_retry(tmp_path, mode):
    c, o = observed(), enabled(tmp_path)
    response = reply("report_evidence_impacts", report("foreign").model_dump(exclude={"kind"}))
    if mode == "truncated": response["complete"] = False
    if mode == "text": response.update(calls=[], text="No machine-readable associations.")
    client = NativeFake([deepcopy(response) for _ in range(3)])
    o.deps.photophysics_arbiter.llm_client = client
    s = store(tmp_path, c)
    c = put(c, phase="impact_review")
    for _ in range(3):
        c = run_turn(o, c, s)
        if protocol(c)["phase"] == "choose": break
    assert protocol(c)["phase"] == "choose" and not f.unreviewed_results(c)
    assert f.view(c)["unavailable_reviews"] == 1 and not f.state(c).impacts
    assert len(client.requests) <= 3 and protocol(c)["results"]


def test_planner_gets_small_pending_work_not_review_history(tmp_path):
    c = associated()
    c, _ = h.update(c, assessment(questions=[question()]))
    payload = planner_payload(enabled(tmp_path), c, native=True)
    info = payload["exploration_state"]["evidence_followup"]
    assert info["pending_impact_count"] == 2 and len(info["questions"]) == 1
    assert "reviews" not in info and "trials" not in json.dumps(info)
    assert "candidate_overview" not in payload


def test_old_mode_wire_unchanged_and_opt_in_requires_tracking(tmp_path):
    legacy, _ = tool_contracts(REGISTRY, [TOOL], hypothesis_tracking=True)
    update_tool = next(t for t in legacy if t["name"] == "update_hypotheses")
    compute_tool = next(t for t in legacy if t["name"].startswith("compute_"))
    assert "responses" not in update_tool["parameters"]["properties"]
    assert "question_id" not in compute_tool["parameters"]["properties"]
    with pytest.raises(ValueError, match="requires command_hypothesis_tracking"):
        replace(orchestrator(tmp_path).config, command_evidence_followup=True)


def test_candidate_sources_shown_to_impact_reviewer_are_readable(tmp_path):
    c, o = observed(), enabled(tmp_path)
    c = put(c, results={**protocol(c)["results"], "result:old": {"status": "success", "summary": "Old fixture."}})
    c, _ = h.update(c, update(refs=["result:old"]))
    c = f.accept_review(c, ["result:old"], {"status": "available", "suggestions": [], "text": "Previous fixture."})
    client = NativeFake([reply(arguments={"records": [{"ref": "result:old", "paths": ["/status"]}]}, call_id="oldread"),
                        reply("report_evidence_impacts", report().model_dump(exclude={"kind"}), call_id="report")])
    o.deps.photophysics_arbiter.llm_client = client
    s = store(tmp_path, c)
    c = run_turn(o, put(c, phase="impact_review"), s)
    assert protocol(c)["impact_session"]["outcome"]["status"] == "read"
    c = run_turn(o, c, s)
    assert f.view(c)["pending_impact_count"] == 2


def test_new_result_reopens_work_only_via_a_new_association():
    c = associated()
    responses = [{"impact_id": iid, "disposition": "reviewed", "note": "No change."} for iid in f.state(c).impacts]
    c, _ = h.update(c, assessment(responses=responses))
    c = put(c, results={**protocol(c)["results"], "result:later": {"status": "success"}})
    assert f.unreviewed_results(c) == ["result:later"] and f.view(c)["pending_impact_count"] == 0
    feedback = validate_suggestions(c, ["result:later"], report("result:later", labels=[LABELS[0]]))
    c = f.accept_review(c, ["result:later"], feedback)
    assert f.view(c)["pending_impact_count"] == 1 and len(f.state(c).impacts) == 3
    assert h.state(c).entries[LABELS[0]].confidence == .4


@pytest.mark.parametrize("review_enabled", [True, False])
def test_full_native_loop_and_final_delivery_with_unresolved_work(tmp_path, review_enabled):
    o = enabled(tmp_path)
    o.config = replace(o.config, enable_support_auditor=review_enabled, enable_agenda_coverage_reviewer=review_enabled)

    class Client:
        settings = None
        def __init__(self):
            self.roles, self.step = [], 0

        def complete_tool_turn(self, **kw):
            role, payload = kw.get("role", "planner"), kw["payload"]
            self.roles.append(role)
            call_id = str(len(self.roles))
            if role == "impact_reviewer":
                rid = payload["results"][0]["source_result_id"]
                return reply("report_evidence_impacts", report(rid).model_dump(exclude={"kind"}), call_id=call_id)
            if role != "planner":
                return {"request_id": role + call_id, "protocol": "responses", "complete": True,
                        "calls": [], "items": [], "text": "Offline review fixture. No scientific endorsement."}
            self.step += 1
            if self.step == 1:
                return reply("describe_tools", {"capability_ids": [TOOL]}, call_id=call_id)
            if self.step == 2:
                data = computation_model(REGISTRY.require(TOOL), with_question=True)(
                    kind="compute", capability_id=TOOL, arguments={}, purpose="Offline screen.", question_id="compare")
                return batch_reply(
                    reply("update_hypotheses", assessment(questions=[question()]).model_dump(exclude={"kind"}), call_id=call_id + "u"),
                    reply("compute_" + TOOL.replace(".", "_"), data.model_dump(exclude={"kind", "capability_id"}), call_id=call_id + "c"))
            if self.step == 3:
                pending = payload["exploration_state"]["evidence_followup"]["pending_impacts"]
                responses = [{"impact_id": pending[0]["id"], "disposition": "reviewed", "note": "No score change."}] if pending else []
                return batch_reply(
                    reply("update_hypotheses", assessment(responses=responses).model_dump(exclude={"kind"}), call_id=call_id + "u"),
                    reply("revise_judgment", judgment().model_dump(exclude={"kind"}), call_id=call_id + "j"))
            if self.step == 4:
                return reply("finalize", {"answer": "Offline bounded delivery.", "limitations": ["Comparison unresolved."]}, call_id=call_id)
            assert self.step == 5 and review_enabled
            return reply("submit_final", {"draft_id": payload["final_draft"]["draft_id"]}, call_id=call_id)

    client = Client()
    o.deps.planner.llm_client = client
    o.deps.photophysics_arbiter.llm_client = client
    o.deps.mechanism_agenda.llm_client = client
    c = o.run(smiles="CCO", user_query="Offline protocol acceptance, not scientific performance.", case_id="followup-full")
    assert c.status == "finalized" and protocol(c)["rounds"] == 1
    assert len(c.final_answer.mechanism_predictions) == 3
    assert client.roles.count("impact_reviewer") == int(review_enabled)
    assert f.view(c)["pending_impact_count"] == int(review_enabled)
    assert f.state(c).questions["compare"].value.status == "open"
    assert len(f.state(c).questions["compare"].trials) == 1
