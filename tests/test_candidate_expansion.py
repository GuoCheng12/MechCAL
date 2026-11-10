"""Candidate control-flow acceptance, not evidence that the scientific idea works."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from mechcal.agents.mechanism_agenda import MechanismAgendaAgent
from mechcal.agents.observation_view import repair_payload
from mechcal.runtime import MechCALOrchestrator, OrchestratorConfig
from mechcal.runtime.candidate_expansion import (
    admit, candidate_context, new_observations, proposal_payload, validate_proposals, validate_responses,
)
from mechcal.runtime.field_repair import correction_contract
from mechcal.runtime.questions import apply_assimilation, replace_state, state
from mechcal.runtime.retry import ModelContractError, RequestValidationError, SchemaRetryError
from mechcal.schemas import CaseRun
from mechcal.schemas.candidate_expansion import CandidateExpansion, CandidateResponse
from mechcal.schemas.question_loop import QuestionAssimilation
from test_question_loop import NOTE, assimilation, initial_case, loop, question
from test_hypothesis_tests import TestClient
from test_runtime import _evidence_unit


def case_with_result(mode="independent"):
    case = initial_case()
    unit = _evidence_unit(evidence_id="E1", family="state_ordering_brightness", round_id="R001",
                          source_report_id="report1", agent_name="microscopic", summary=NOTE)
    result = dict(result_id="X1", capability_id="microscopic.run_bright_dark_state_ordering",
                  status="success", evidence_refs=["E1"], question_ids=["Q1"],
                  structured_results={"parsed_observables": {"excited_states": [
                      {"energy_ev": 3., "oscillator_strength": .1}], "method": "fixture"}},
                  investigation_plan={"purpose": "discrimination"})
    case = case.touch(evidence_ledger=case.evidence_ledger.with_items([unit]),
                      runtime={**case.runtime, "candidate_expansion_mode": mode})
    return replace_state(case, phase="candidate_expansion", round_index=2, results=[result], pending_results=[result])


def proposal(ref="E1"):
    return CandidateExpansion(proposals=[dict(hypothesis=NOTE, related_labels=["ICT_TICT_CT"],
        evidence_refs=[ref], observation_basis=NOTE, distinguishing_observation=NOTE, boundary=NOTE)], rationale=NOTE)


def update(case, decision="defer"):
    data = assimilation(dict(case_id=case.case_id, revision=state(case)["revision"],
                             pending_results=state(case).get("pending_results", [])))
    data["candidate_responses"] = [dict(candidate_id=p["candidate_id"], decision=decision, reason=NOTE,
        question_ids=["Q1"] if decision in {"investigate", "merge"} else [])
        for p in candidate_context(case)["pending"]]
    return QuestionAssimilation.model_validate(data)


def test_independent_view_blinds_judgment_intent_interpretations_and_private_metadata():
    c = case_with_result()
    marker = "DO_NOT_SEND_THIS_JUDGMENT"
    c.input.metadata.update(hidden_reference=marker, rubric=marker, judge_feedback=marker)
    c.input.user_query = marker
    c.portfolio.current = marker
    s = state(c)
    s.update(questions={"Q1": {"assessment": marker}}, assimilation={"decision_assessment": marker},
             argument_review=marker, result_review=marker)
    for r in s["results"]:
        r.update(expected_observation=marker, worker_explanation=marker, explanation_review=marker)
        r["investigation_plan"]["hypothesis_test"] = marker
    for e in c.evidence_ledger.items:
        e.summary = e.claim = e.context = marker
        e.hypothesis_refs = [marker]
    payload = proposal_payload(c, new_observations(c))
    assert marker not in json.dumps(payload)
    assert payload["computations"][0]["computed"]["parsed_observables"]["excited_states"][0]["energy_ev"] == 3.
    reflection = c.touch(runtime={**c.runtime, "candidate_expansion_mode": "reflection"})
    control = proposal_payload(reflection, new_observations(reflection))
    assert marker in json.dumps(control["current_judgment"])
    control.pop("current_judgment")
    control["mode"] = "independent"
    assert control == payload


@pytest.mark.parametrize("kind", ["preparation", "failure", "boundary", "empty"])
def test_no_expansion_without_new_scientific_observation(kind):
    c = case_with_result()
    r = state(c)["results"][0]
    if kind == "preparation":
        r["investigation_plan"]["purpose"] = "preparation"
    elif kind == "failure":
        r["status"] = "failed"
        c.evidence_ledger.items[-1].status = "failed"
    elif kind == "boundary":
        r["structured_results"]["boundary_only"] = True
    else:
        r["structured_results"] = {}
    assert new_observations(c) == []


def test_legacy_geometry_family_does_not_hide_dimer_observations():
    c = case_with_result()
    c.evidence_ledger.items[-1].family = "geometry_precondition"
    r = state(c)["results"][0]
    r["capability_id"] = "microscopic.search_dimer_contacts"
    r["structured_results"]["parsed_observables"] = {"candidates": [{"interaction_energy_ev": -.1}]}
    assert new_observations(c)


def test_partial_results_survive_failed_substep_and_failures_remain_visible():
    c = case_with_result()
    r = state(c)["results"][0]
    r.update(status="failed", failure={"message": "one substep failed"})
    assert new_observations(c)
    assert proposal_payload(c, new_observations(c))["computations"][0]["failure"]


@pytest.mark.parametrize("fault", ["no_fresh_ref", "foreign_ref", "unknown_label", "duplicate"])
def test_proposals_reject_invented_sources_labels_and_duplicates(fault):
    c = case_with_result()
    p = proposal()
    if fault == "no_fresh_ref":
        p.proposals[0].evidence_refs = ["input:smiles"]
    elif fault == "foreign_ref":
        p.proposals[0].evidence_refs += ["othercase:E2"]
    elif fault == "unknown_label":
        p.proposals[0].related_labels = ["made_up"]
    else:
        p.proposals.append(p.proposals[0].model_copy(deep=True))
    with pytest.raises(ModelContractError):
        validate_proposals(p, proposal_payload(c, new_observations(c)))


def test_at_most_two_and_no_scores_or_actions_in_proposals():
    p = proposal().model_dump(mode="json")
    for extra in ({"ranking": []}, {"dispatch_requests": []}, {"confidence": .8}):
        with pytest.raises(ValueError):
            CandidateExpansion.model_validate(p | extra)
    p["proposals"] *= 3
    with pytest.raises(ValueError):
        CandidateExpansion.model_validate(p)


def test_admission_does_not_wait_for_agreement_change_rank_or_add_evidence():
    c = case_with_result()
    before = c.model_dump(mode="json")
    fresh = new_observations(c)
    admitted = admit(c, fresh, proposal())
    assert candidate_context(admitted)["pending"][0]["candidate_id"] == "B001:C01"
    assert admitted.evidence_ledger == c.evidence_ledger
    assert admitted.portfolio == c.portfolio
    assert admitted.round_ids == c.round_ids
    assert c.model_dump(mode="json") == before
    assert new_observations(admitted) == []
    with pytest.raises(ValueError):
        admit(admitted, fresh, proposal())
    assert candidate_context(case_with_result())["pending"] == []
    assert CaseRun.model_validate_json(admitted.model_dump_json()) == admitted


def test_no_proposal_is_valid_and_repeated_values_do_not_retrigger():
    c = case_with_result()
    c = admit(c, new_observations(c), CandidateExpansion(rationale="No concrete alternative justified"))
    assert candidate_context(c)["pending"] == []
    duplicate = deepcopy(state(c)["results"][0])
    duplicate["result_id"] = "X2"
    c = replace_state(c, pending_results=[duplicate], results=[*state(c)["results"], duplicate])
    assert new_observations(c) == []
    duplicate["structured_results"]["parsed_observables"]["environment"] = "solvent"
    assert new_observations(c)


def test_identical_numbers_under_different_conditions_are_not_silently_discarded():
    c = case_with_result()
    c = admit(c, new_observations(c), CandidateExpansion(rationale=NOTE))
    r = deepcopy(state(c)["results"][0])
    r["resolved_source_artifact_id"] = "a_different_geometry"
    c = replace_state(c, pending_results=[r])
    assert new_observations(c)


def test_new_question_can_be_added_during_scoped_receipt_repair():
    from mechcal.runtime.field_repair import apply_corrections
    c = case_with_result()
    c = admit(c, new_observations(c), proposal())
    u = update(c, "investigate")
    u.candidate_responses[0].question_ids = ["Q2"]
    with pytest.raises(RequestValidationError) as err:
        validate_responses(c, u)
    paths = err.value.issues[0]["repair_paths"]
    assert ["questions"] in paths
    corrected = apply_corrections(u.model_dump(mode="json"), {"repairs": [
        {"path": ["candidate_responses"], "value": [r.model_dump(mode="json") for r in u.candidate_responses]},
        {"path": ["questions"], "value": [q.model_dump(mode="json") for q in u.questions] + [question("Q2")]},
    ]}, paths)
    validate_responses(c, QuestionAssimilation.model_validate(corrected))


@pytest.mark.parametrize("decision", ["investigate", "merge", "defer", "reject"])
def test_all_dispositions_are_allowed_without_forced_ranking_or_tools(decision):
    c = case_with_result()
    c = admit(c, new_observations(c), proposal())
    u = update(c, decision)
    validate_responses(c, u)
    finished = apply_assimilation(c, u)
    assert finished.portfolio == u.portfolio
    assert finished.evidence_ledger == c.evidence_ledger
    assert candidate_context(finished)["pending"] == []
    assert candidate_context(finished)["history"][0]["response"]["decision"] == decision
    assert candidate_context(c)["pending"]


@pytest.mark.parametrize("fault", ["omitted", "duplicate", "foreign_candidate", "foreign_source", "missing_question", "closed_question"])
def test_receipts_cannot_disappear_or_claim_investigation_without_question(fault):
    c = case_with_result()
    c = admit(c, new_observations(c), proposal())
    u = update(c, "investigate")
    if fault == "omitted":
        u.candidate_responses = []
    elif fault == "duplicate":
        u.candidate_responses *= 2
    elif fault == "foreign_candidate":
        u.candidate_responses[0].candidate_id = "wrong"
    elif fault == "foreign_source":
        u.candidate_responses[0].evidence_refs = ["B001:C01"]
    elif fault == "missing_question":
        u.candidate_responses[0].question_ids = []
    else:
        u.questions[0].answer_state = "answered"
    with pytest.raises(RequestValidationError) as err:
        validate_responses(c, u)
    correction_contract(QuestionAssimilation, err.value.issues[0]["repair_paths"])


def test_empty_reason_and_historical_replies_rejected():
    with pytest.raises(ValueError):
        CandidateResponse(candidate_id="C", decision="reject", reason="   ")
    c = case_with_result()
    c = admit(c, new_observations(c), proposal())
    u = update(c)
    c = apply_assimilation(c, u)
    with pytest.raises(RequestValidationError):
        validate_responses(c, u)


def test_field_repair_keeps_candidate_and_question_context():
    c = case_with_result()
    c = admit(c, new_observations(c), proposal())
    ctx = {"case_id": c.case_id, "candidate_context": candidate_context(c), "questions": [question()]}
    repair = repair_payload(ctx, {"issues": []})
    assert repair["candidate_context"] == ctx["candidate_context"]
    assert repair["questions"] == ctx["questions"]


class ProposalClient:
    def __init__(self, invalid=False):
        self.calls = []
        self.invalid = invalid

    def is_configured(self):
        return True

    def complete_json(self, *, payload, **kwargs):
        self.calls.append(deepcopy(payload))
        if self.invalid:
            return proposal("foreign").model_dump(mode="json")
        return proposal(payload["new_observation_refs"][0]).model_dump(mode="json")


def test_bounded_correction_no_empty_fallback_or_state_mutation():
    c = case_with_result()
    before = c.model_dump(mode="json")
    client = ProposalClient(invalid=True)
    agent = MechanismAgendaAgent(llm_client=client)
    events = []
    with pytest.raises(SchemaRetryError):
        agent.propose_candidates(c, new_observations(c), event_sink=events.append)
    assert len(client.calls) == 3
    assert len([e for e in events if e["kind"] == "model_response"]) == 3
    assert c.model_dump(mode="json") == before


def test_disabled_config_and_legacy_records(tmp_path):
    assert OrchestratorConfig(run_base_dir=tmp_path).candidate_expansion_mode == "off"
    with pytest.raises(ValueError):
        OrchestratorConfig(run_base_dir=tmp_path, candidate_expansion_mode="independent")
    with pytest.raises(ValueError):
        OrchestratorConfig(run_base_dir=tmp_path, candidate_expansion_mode="unknown")
    c = initial_case()
    assert candidate_context(c) == {"enabled": False}
    assert CaseRun.model_validate_json(c.model_dump_json()) == c


class ExpansionPlanner(TestClient):
    __test__ = False

    def complete_json(self, *, payload, **kwargs):
        data = super().complete_json(payload=payload, **kwargs)
        if payload["stage"] == "planner_assimilate":
            data["candidate_responses"] = [dict(candidate_id=p["candidate_id"], decision="merge",
                reason=NOTE, question_ids=["Q1"]) for p in payload.get("candidate_context", {}).get("pending", [])]
        return data


@pytest.mark.parametrize("mode", ["independent", "reflection"])
def test_full_loop_no_extra_science_rounds_or_duplicate_admission_on_resume(tmp_path, monkeypatch, mode):
    from mechcal.schemas.decisions import StopReview
    import mechcal.runtime.question_loop as runtime

    o, _ = loop(tmp_path, unlimited=True, discover=True)
    planner = ExpansionPlanner(1, True)
    o.deps.planner.llm_client = planner
    proposer = ProposalClient()
    o.deps.mechanism_agenda.llm_client = proposer
    config = replace(o.config, comparative_evidence=True, decision_value_checks=True, hypothesis_tests=True,
                     enable_agenda_coverage_reviewer=True, candidate_expansion_mode=mode)
    o = MechCALOrchestrator(config, o.deps)
    monkeypatch.setattr(o, "_review_agenda_coverage", lambda case, *a, **kw: case)
    monkeypatch.setattr(o.deps.mechanism_agenda, "review_stop", lambda *a, **kw: StopReview(status="reviewed", rationale=NOTE))
    original_checkpoint = runtime.checkpoint
    def interrupt_after_admission(c, store):
        saved = original_checkpoint(c, store)
        if candidate_context(c).get("pending"):
            raise KeyboardInterrupt("fixture interruption after persisted admission")
        return saved
    monkeypatch.setattr(runtime, "checkpoint", interrupt_after_admission)
    with pytest.raises(KeyboardInterrupt):
        o.run(smiles="CCO", user_query=NOTE, case_id="expansion")
    assert len(proposer.calls) == 1
    monkeypatch.setattr(runtime, "checkpoint", original_checkpoint)
    result = o.run(smiles="CCO", user_query=NOTE, case_id="expansion", resume=True)
    assert result.status == "finalized"
    assert result.round_ids == ["R001", "R002"]
    assert result.runtime["worker_call_counts"]["macro"] == 1
    assert len(proposer.calls) == 1
    assert len(candidate_context(result)["history"]) == 1
    assert candidate_context(result)["pending"] == []
    assert result.portfolio.current == "RIM_RIR_RIV"
    assert not any(e.evidence_id.startswith("B001") for e in result.evidence_ledger.items)
    n = len(planner.calls)
    assert o.run(smiles="CCO", user_query=NOTE, case_id="expansion", resume=True) == result
    assert len(planner.calls) == n
