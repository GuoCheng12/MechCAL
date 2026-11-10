"""Offline contract and end-to-end tests, not performance evidence."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from mechcal.runtime import MechCALOrchestrator, OrchestratorConfig
from mechcal.runtime.field_repair import apply_corrections, correction_contract
from mechcal.runtime.hypothesis_tests import test_context as context, validate_comparison, validate_tests
from mechcal.runtime.questions import apply_assimilation, replace_state, state, validate_assimilation
from mechcal.runtime.retry import RequestValidationError
from mechcal.schemas.hypothesis_tests import HypothesisTest, LeadingComparison, TestAssessment as Assessment
from mechcal.schemas.question_loop import QuestionAction, QuestionAssimilation
from test_decision_value import outcomes, value
from test_evidence_comparison import ComparisonClient, initial, plan, update_for
from test_question_loop import NOTE, action, loop


def comparison(leader="RIM_RIR_RIV", challenger="ICT_TICT_CT"):
    return LeadingComparison(leader=leader, challenger=challenger, leading_hypothesis=NOTE,
                             alternative_hypothesis=NOTE, basis_for_preference=NOTE,
                             preference_status="prior_only", evidence_refs=["input:smiles"],
                             previous_basis_update=NOTE)


def planned_test():
    p = plan()
    p.hypothesis_test = HypothesisTest(target_label="RIM_RIR_RIV", alternative_label="ICT_TICT_CT",
                                      tested_claim=NOTE, relation_to_current_comparison=NOTE)
    p.decision_value = value()
    p.possible_outcomes = outcomes("support", "no_change")
    return p


def receipt(**changes):
    return Assessment(matched_outcome_indices=[0], observed_result=NOTE,
                      basis_effect="unchanged", comparison_effect="non_discriminating",
                      explanation=NOTE, **changes)


def start():
    c = initial()
    c = c.touch(runtime={**c.runtime, "hypothesis_tests": True})
    c = replace_state(c, round_index=1)
    d = update_for(c)
    d["ranking_update"]["leading_comparison"] = comparison().model_dump()
    return c, QuestionAssimilation.model_validate(d)


def pending():
    c, u = start()
    c = apply_assimilation(c, u)
    p = planned_test()
    r = dict(result_id="X", question_ids=["Q1"], status="success", evidence_refs=[],
             structured_results={}, investigation_plan=p.model_dump(mode="json"))
    c = replace_state(c, phase="assimilate", results=[r], pending_results=[r])
    d = update_for(c)
    d["ranking_update"]["leading_comparison"] = comparison().model_dump()
    d["dispositions"][0]["test_assessment"] = receipt().model_dump()
    return c, QuestionAssimilation.model_validate(d)


@pytest.mark.parametrize("fault", ["missing", "leader", "challenger", "foreign_refs", "unsupported_preference"])
def test_invalid_comparison_is_scoped_and_never_changes_ranking(fault):
    c, u = start()
    comp = u.ranking_update.leading_comparison
    if fault == "missing":
        u.ranking_update.leading_comparison = None
    elif fault == "leader":
        comp.leader = "ESIPT_PT"
    elif fault == "challenger":
        comp.challenger = comp.leader
    elif fault == "foreign_refs":
        comp.evidence_refs = ["other_case:E1"]
    else:
        comp.preference_status = "evidence_favors_leader"
        comp.evidence_refs = []
    before = deepcopy(c.model_dump())
    with pytest.raises(RequestValidationError) as exc:
        validate_assimilation(u, c)
    assert c.model_dump() == before
    path = exc.value.issues[0]["repair_paths"][0]
    assert path == ["ranking_update", "leading_comparison"]
    correction_contract(QuestionAssimilation, [path])


def test_planner_may_keep_or_change_order_without_forced_confidence():
    c, u = start()
    c = apply_assimilation(c, u)
    before = deepcopy(c.portfolio)
    d = update_for(c)
    d["ranking_update"]["leading_comparison"] = comparison().model_dump()
    kept = apply_assimilation(replace_state(c, phase="assimilate"), QuestionAssimilation.model_validate(d))
    assert kept.portfolio == before
    d["portfolio"]["hypotheses"][2]["differential_priority"] = .8
    d["portfolio"].update(current="ESIPT_PT", runner_up="RIM_RIR_RIV")
    d["ranking_update"].update(basis="reinterpretation", leading_comparison=comparison("ESIPT_PT", "RIM_RIR_RIV").model_dump())
    changed = apply_assimilation(replace_state(c, phase="assimilate"), QuestionAssimilation.model_validate(d))
    assert changed.portfolio.current == "ESIPT_PT"


@pytest.mark.parametrize("fault", ["missing", "duplicate_index", "bad_index", "negative_index", "foreign_refs", "non_discriminating_favors", "no_result_weakens"])
def test_invalid_receipts_have_bounded_repair_paths(fault):
    c, u = pending()
    disp = u.dispositions[0]
    item = disp.test_assessment
    if fault == "missing":
        disp.test_assessment = None
    elif fault in ("duplicate_index", "bad_index", "negative_index"):
        item.matched_outcome_indices = {"duplicate_index": [0, 0], "bad_index": [8], "negative_index": [-1]}[fault]
    elif fault == "foreign_refs":
        item.evidence_refs = ["other:E1"]
    elif fault == "non_discriminating_favors":
        item.comparison_effect = "favors_target"
    else:
        disp.outcome = "no_result"
        item.basis_effect = "weakened"
    with pytest.raises(RequestValidationError) as exc:
        validate_comparison(c, u)
    path = exc.value.issues[0]["repair_paths"][0]
    assert path == ["dispositions", 0, "test_assessment"]
    correction_contract(QuestionAssimilation, [path])


def test_unavailable_and_unexpected_are_preserved_not_negative_evidence():
    c, u = pending()
    d = u.dispositions[0]
    d.outcome = "no_result"
    d.test_assessment.matched_outcome_indices = []
    d.test_assessment.basis_effect = "unresolved"
    d.test_assessment.comparison_effect = "unavailable"
    validate_comparison(c, u)
    d.outcome = "incomplete"
    d.test_assessment.comparison_effect = "non_discriminating"
    validate_comparison(c, u)


def failed_execution(status="failed", boundary_only=False):
    from test_runtime import _evidence_unit

    c, u = pending()
    c.evidence_ledger.items.append(_evidence_unit(
        evidence_id="X:E1", round_id="R001", source_report_id="X",
        agent_name="microscopic_worker", family="geometry_precondition",
        summary="No usable dimer poses were returned.", status=status,
        metrics={"computed_observables": {"boundary_only": boundary_only}},
    ))
    for key in ("results", "pending_results"):
        c.runtime["question_loop"][key][0].update(
            status=status, evidence_refs=["X:E1"],
        )
    d = u.dispositions[0]
    d.outcome = "no_result"
    d.evidence_refs = ["X:E1"]
    d.test_assessment = Assessment(
        observed_result="All candidate optimizations failed; no contact data.",
        basis_effect="unresolved", comparison_effect="unavailable",
        operational_refs=["X:E1"],
        explanation="The execution failed, so the planned comparison is unavailable.",
    )
    return c, u


@pytest.mark.parametrize("status,boundary", [
    ("failed", False), ("unsupported", False), ("missing", False), ("present", True),
])
def test_failure_provenance_is_preserved_without_scientific_evidence(status, boundary):
    c, u = failed_execution(status, boundary)
    before = deepcopy(c.portfolio)
    accepted = apply_assimilation(c, u)
    assert accepted.portfolio == before
    assessment = state(accepted)["hypothesis_test_history"][-1]["assessment"]
    assert assessment["operational_refs"] == ["X:E1"]
    assert assessment["evidence_refs"] == []
    assert accepted.evidence_ledger == c.evidence_ledger
    assert accepted.mechanism_support_arguments == c.mechanism_support_arguments


def test_failure_citation_repair_has_exact_allowed_refs_and_preserves_judgment():
    c, u = failed_execution()
    a = u.dispositions[0].test_assessment
    a.evidence_refs, a.operational_refs = a.operational_refs, []
    before = deepcopy(u.portfolio)
    with pytest.raises(RequestValidationError) as exc:
        validate_assimilation(u, c)
    assert "Use [] when none were delivered" in str(exc.value)
    assert "operational_refs using ['X:E1']" in str(exc.value)
    paths = exc.value.issues[0]["repair_paths"]
    correction_contract(QuestionAssimilation, paths)
    replacement = a.model_dump()
    replacement.update(evidence_refs=[], operational_refs=["X:E1"])
    repaired = apply_corrections(u.model_dump(mode="json"), {
        "repairs": [{"path": paths[0], "value": replacement}],
    }, paths)
    fixed = QuestionAssimilation.model_validate(repaired)
    validate_assimilation(fixed, c)
    assert fixed.portfolio == before


@pytest.mark.parametrize("ref", ["other_case:E1", "review:E1", "other_execution:E1", "input:smiles"])
def test_operational_refs_cannot_cite_foreign_or_scientific_sources(ref):
    from test_runtime import _evidence_unit

    c, u = failed_execution()
    # An operational record in the same case is still invalid for another execution.
    c.evidence_ledger.items.append(_evidence_unit(
        evidence_id="other_execution:E1", round_id="R001", source_report_id="other_execution",
        agent_name="microscopic_worker", family="geometry_precondition",
        summary=NOTE, status="failed",
    ))
    u.dispositions[0].test_assessment.operational_refs = [ref]
    with pytest.raises(RequestValidationError, match="operational-only records from this execution"):
        validate_comparison(c, u)


@pytest.mark.parametrize("field,value", [
    ("basis_effect", "weakened"), ("basis_effect", "strengthened"),
    ("comparison_effect", "favors_target"), ("comparison_effect", "favors_alternative"),
])
def test_operational_refs_never_justify_scientific_changes(field, value):
    c, u = failed_execution()
    setattr(u.dispositions[0].test_assessment, field, value)
    with pytest.raises(RequestValidationError, match="requires delivered observations"):
        validate_comparison(c, u)


def test_partial_failure_retains_surviving_observations_and_separate_failure_refs():
    from test_runtime import _evidence_unit

    c, u = failed_execution()
    c.evidence_ledger.items.append(_evidence_unit(
        evidence_id="X:E2", round_id="R001", source_report_id="X",
        agent_name="microscopic_worker", family="charge_localization", summary=NOTE,
    ))
    for key in ("results", "pending_results"):
        c.runtime["question_loop"][key][0]["evidence_refs"] = ["X:E1", "X:E2"]
    d = u.dispositions[0]
    d.outcome = "incomplete"
    d.test_assessment.evidence_refs = ["X:E2"]
    d.test_assessment.basis_effect = "unchanged"
    d.test_assessment.comparison_effect = "non_discriminating"
    validate_assimilation(u, c)
    d.test_assessment.operational_refs = ["X:E2"]
    with pytest.raises(RequestValidationError, match="operational-only records"):
        validate_comparison(c, u)


def test_legacy_test_assessment_remains_readable():
    raw = receipt().model_dump()
    raw.pop("operational_refs")
    assert Assessment.model_validate(raw).operational_refs == []


def test_operational_record_cannot_become_scientific_comparison():
    c, u = pending()
    c.runtime["question_loop"]["pending_results"][0]["investigation_plan"]["purpose"] = "preparation"
    u.dispositions[0].test_assessment.basis_effect = "strengthened"
    with pytest.raises(RequestValidationError):
        validate_comparison(c, u)


def test_real_observation_can_weaken_a_tested_basis_without_automatic_reranking():
    from test_runtime import _evidence_unit

    c, u = pending()
    e = _evidence_unit(evidence_id="E1", round_id="R001", source_report_id="report",
                       agent_name="microscopic_worker", family="charge_localization", summary=NOTE)
    c.evidence_ledger.items.append(e)
    for key in ("results", "pending_results"):
        c.runtime["question_loop"][key][0]["evidence_refs"] = ["E1"]
    d = u.dispositions[0]
    d.outcome = "discriminating"
    d.test_assessment.evidence_refs = ["E1"]
    d.test_assessment.basis_effect = "weakened"
    d.test_assessment.comparison_effect = "favors_alternative"
    before = deepcopy(u.portfolio)
    validate_comparison(c, u)
    assert u.portfolio == before
    c.runtime["question_loop"]["pending_results"][0]["investigation_plan"]["purpose"] = "preparation"
    with pytest.raises(RequestValidationError, match="Preparation enables"):
        validate_comparison(c, u)


def test_receipt_repair_keeps_original_test_and_pending_observations_only():
    from mechcal.agents.observation_view import repair_payload

    p = planned_test().model_dump(mode="json")
    payload = {"case_id": "C", "hypothesis_test_context": {"enabled": True},
               "pending_results": [{"result_id": "X", "evidence_refs": ["E1"], "investigation_plan": p}],
               "observations": [{"evidence_id": "E1", "metrics": {"value": 1}},
                                {"evidence_id": "old", "metrics": {"value": 2}}]}
    before = deepcopy(payload)
    repaired = repair_payload(payload, {"issues": []})
    assert repaired["pending_results"][0]["investigation_plan"] == p
    assert repaired["observations"] == [payload["observations"][0]]
    assert payload == before


def test_missing_candidate_label_can_repair_either_side_without_rewriting_plan():
    c, u = start()
    c = apply_assimilation(c, u)
    a = action(c.case_id, 1, dispatch=True)
    p = planned_test()
    p.candidate_labels = ["RIM_RIR_RIV"]
    a.investigation_plans = [p]
    with pytest.raises(RequestValidationError) as exc:
        validate_tests(c, a)
    paths = exc.value.issues[0]["repair_paths"]
    assert paths == [["investigation_plans", 0, "candidate_labels"],
                     ["investigation_plans", 0, "hypothesis_test"]]
    correction_contract(QuestionAction, paths)
    repaired = apply_corrections(a.model_dump(mode="json"), {"repairs": [{"path": paths[0], "value": planned_test().candidate_labels}]}, paths)
    validate_tests(c, QuestionAction.model_validate(repaired))


def test_old_records_readable_flag_off_and_case_history_isolated(tmp_path):
    c, u = start()
    u.ranking_update.leading_comparison = None
    off = c.touch(runtime={**c.runtime, "hypothesis_tests": False})
    apply_assimilation(off, u)
    assert context(off) == {"enabled": False}
    assert context(start()[0])["recent_tests"] == []
    with pytest.raises(ValueError, match="requires decision_value_checks"):
        OrchestratorConfig(run_base_dir=tmp_path, hypothesis_tests=True)


class TestClient(ComparisonClient):
    __test__ = False

    def complete_json(self, *, payload, system_prompt, schema_feedback=None):
        data = super().complete_json(payload=payload, system_prompt=system_prompt, schema_feedback=schema_feedback)
        if data.get("action") == "describe_tools":
            return data
        stage = payload["stage"]
        if stage in ("planner_assimilate", "planner_final_revision"):
            update = data["update"] if stage == "planner_final_revision" else data
            update["ranking_update"]["leading_comparison"] = comparison().model_dump()
            for d in update["dispositions"]:
                d["test_assessment"] = receipt(evidence_refs=d["evidence_refs"]).model_dump()
        if stage == "planner_question_act":
            for p in data["investigation_plans"]:
                prototype = planned_test().model_dump(mode="json")
                p.update({k: prototype[k] for k in ("hypothesis_test", "decision_value", "possible_outcomes")})
        return data


@pytest.mark.parametrize("audit", [False, True])
def test_complete_loop_resume_no_extra_tools_or_evidence(tmp_path, audit, monkeypatch):
    from mechcal.schemas.question_loop import ResultReview

    o, _ = loop(tmp_path, unlimited=True, discover=True)
    client = TestClient(1, True)
    o.deps.planner.llm_client = client
    config = replace(o.config, comparative_evidence=True, decision_value_checks=True,
                     hypothesis_tests=True, enable_support_auditor=audit)
    monkeypatch.setattr(o.deps.photophysics_arbiter, "review_results", lambda case, **kw: ResultReview(
        interpretations=[dict(result_id=r["result_id"], question_ids=r["question_ids"],
                              delivered=NOTE, missing=NOTE, boundary=NOTE, evidence_refs=r["evidence_refs"])
                         for r in state(case).get("pending_results", [])]))
    o = MechCALOrchestrator(config, o.deps)
    result = o.run(smiles="CCO", user_query=NOTE, case_id="hypothesis-test")
    assert result.status == "finalized"
    s = state(result)
    assert len(s["results"]) == len(s["hypothesis_test_history"]) == 1
    assert s["hypothesis_test_history"][0]["assessment"]["comparison_effect"] == "non_discriminating"
    assert s["hypothesis_test_history"][0]["plan"] == s["results"][0]["investigation_plan"]
    assert result.portfolio.current == "RIM_RIR_RIV"
    assert context(result)["comparison_outcomes"] == {"non_discriminating": 1}
    assert not any("test_assessment" in e.evidence_id for e in result.evidence_ledger.items)
    before = len(client.calls)
    assert o.run(smiles="CCO", user_query=NOTE, case_id="hypothesis-test", resume=True) == result
    assert len(client.calls) == before
    if audit:
        assert s["final_revision_complete"]
    checkpoint = json.loads((tmp_path / "hypothesis-test/question_checkpoint.json").read_text())
    assert checkpoint["runtime"]["question_loop"]["hypothesis_test_history"] == s["hypothesis_test_history"]
    assert all("rubric" not in payload and "hidden_reference" not in payload for payload in client.calls)
