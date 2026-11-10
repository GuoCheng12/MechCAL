"""Offline contract checks, including real read-only preflight; no API calls."""

from copy import deepcopy
from dataclasses import replace

import pytest

from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime import MechCALOrchestrator, OrchestratorConfig
from mechcal.runtime.boundary_decisions import required_pairs, validate_continuation, validate_ranking
from mechcal.runtime.field_repair import correction_contract
from mechcal.runtime.questions import apply_assimilation, public_payload, replace_state, state, validate_action, validate_assimilation
from mechcal.runtime.retry import RequestValidationError
from mechcal.schemas import ArtifactRecord, CaseRun
from mechcal.schemas.boundary_decisions import ContinuationAssessment
from mechcal.schemas.question_loop import QuestionAction, QuestionAssimilation
from test_decision_value import outcomes, value
from test_evidence_comparison import ComparisonClient, initial, update_for
from test_question_loop import NOTE, REGISTRY, TOOL, action, loop, request


def complete_update(data, previous):
    ranked = [h["name"] for h in data["portfolio"]["hypotheses"]]
    for label in MECHANISM_POOL:
        if label not in ranked:
            data["portfolio"]["hypotheses"].append(dict(
                name=label, differential_priority=.09 / (len(ranked) + 1), rationale=NOTE))
            ranked.append(label)
    ranking = data["ranking_update"]
    ranking["candidate_bases"] = [dict(label=label, prior_rationale=NOTE,
        supported_observation="No delivered observation in this fixture.",
        positive_refs=[], counter_refs=[], inference_gap=NOTE) for label in ranked]
    challengers = [ranked[3], *[label for label in previous[:3] if label in ranked[3:]]]
    ranking["boundary_comparisons"] = [dict(included_label=ranked[2], excluded_label=label,
        outcome="unresolved", basis="prior_only", evidence_refs=[], common_standard=NOTE,
        reason_for_order=NOTE, remaining_difference=NOTE) for label in dict.fromkeys(challengers)]
    return data


def start():
    case = initial()
    case = case.touch(runtime={**case.runtime, "evidence_decision_contracts": True,
                               "decision_value_checks": True})
    data = complete_update(update_for(case), [h.name for h in case.portfolio.sorted_hypotheses()])
    return case, QuestionAssimilation.model_validate(data)


def ending(case, **route_changes):
    result = action(case.case_id, state(case)["revision"])
    result.continuation_assessment = ContinuationAssessment(
        scope="bounded_ranking", remaining_uncertainty=NOTE, decision_reason=NOTE,
        routes=[dict(capability_id=TOOL, disposition="not_useful", reason=NOTE, **route_changes)])
    return result


def test_explicit_bases_and_tentative_order_do_not_force_scores_or_new_evidence():
    case, update = start()
    before = deepcopy(case.model_dump())
    validate_assimilation(update, case)
    assert case.model_dump() == before
    result = apply_assimilation(case, update)
    assert result.portfolio == update.portfolio
    assert result.evidence_ledger == case.evidence_ledger
    assert state(result)["last_comparison"]["ranking_update"]["candidate_bases"]
    assert CaseRun.model_validate_json(result.model_dump_json()) == result
    validate_action(ending(result), result, REGISTRY)


@pytest.mark.parametrize("fault", ["missing_bases", "duplicate_label", "foreign_ref", "wrong_pair", "unreferenced_preference"])
def test_bad_ranking_records_have_scoped_repair_without_reranking(fault):
    case, update = start()
    ranking = update.ranking_update
    if fault == "missing_bases":
        ranking.candidate_bases = []
    elif fault == "duplicate_label":
        ranking.candidate_bases[-1].label = ranking.candidate_bases[0].label
    elif fault == "foreign_ref":
        ranking.candidate_bases[0].positive_refs = ["another_case:E1"]
    elif fault == "wrong_pair":
        ranking.boundary_comparisons[0].included_label = update.portfolio.current
    else:
        ranking.boundary_comparisons[0].basis = "observations"
    before = deepcopy(update.portfolio)
    with pytest.raises(RequestValidationError) as error:
        validate_ranking(case, update)
    paths = error.value.issues[0]["repair_paths"]
    assert paths[0][0] == "ranking_update"
    correction_contract(QuestionAssimilation, paths)
    assert update.portfolio == before


def test_cutoff_and_displaced_members_cannot_be_skipped():
    case, update = start()
    original = deepcopy(update.portfolio)
    ranked = update.portfolio.sorted_hypotheses()
    ranked[0].differential_priority = .0001
    ordered = update.portfolio.sorted_hypotheses()
    pairs = required_pairs(original, update.portfolio)
    assert (ordered[2].name, ordered[3].name) in pairs
    assert (ordered[2].name, original.current) in pairs


def test_comparator_can_favor_excluded_candidate_without_automatic_promotion():
    case, update = start()
    comparison = update.ranking_update.boundary_comparisons[0]
    comparison.outcome = "favors_excluded"
    comparison.reason_for_order = "Synthetic explicit unresolved conflict between comparison and overall prior."
    before = deepcopy(update.portfolio)
    validate_ranking(case, update)
    assert update.portfolio == before


@pytest.mark.parametrize("fault", ["missing", "useful", "false_block", "unknown", "duplicate", "omitted_named_route"])
def test_stop_requires_consistent_bounded_continuation_assessment(fault):
    case, update = start()
    case = apply_assimilation(case, update)
    result = ending(case)
    if fault == "missing":
        result.continuation_assessment = None
    elif fault == "omitted_named_route":
        state(case)["questions"]["Q1"]["acceptable_capability_ids"] = ["macro.screen_hbond_network"]
    elif fault == "duplicate":
        result.continuation_assessment.routes *= 2
    else:
        route = result.continuation_assessment.routes[0]
        if fault == "unknown":
            route.capability_id = "invented.method"
        else:
            route.disposition = "useful" if fault == "useful" else "blocked"
    with pytest.raises(RequestValidationError) as error:
        validate_action(result, case, REGISTRY)
    correction_contract(QuestionAction, error.value.issues[0]["repair_paths"])


def test_missing_input_needs_preparation_assessment_and_limit_is_not_convergence():
    case, update = start()
    case = apply_assimilation(case, update)
    case.runtime["tool_environment"] = {"amesp_enabled": True}
    result = ending(case)
    route = result.continuation_assessment.routes[0]
    route.capability_id = "microscopic.run_relaxed_coordinate_scan"
    route.disposition = "blocked"
    with pytest.raises(RequestValidationError, match="Missing inputs"):
        validate_continuation(case, result, REGISTRY)
    route.preparation_assessment = NOTE
    validate_continuation(case, result, REGISTRY)
    case.runtime["run_limits"] = {"scientific_tools_allowed": False}
    result.continuation_assessment = None
    validate_continuation(case, result, REGISTRY)


def test_real_au_preflight_rejects_scan_before_dispatch(tmp_path):
    case, update = start()
    case = apply_assimilation(case, update)
    xyz = tmp_path / "source.xyz"
    xyz.write_text("2\nScope-test geometry only, not a scientific result\nAu 0 0 0\nH 0 0 1.5\n")
    case.artifact_manifest.artifacts.append(ArtifactRecord(
        artifact_id="prepared", kind="prepared_structure", created_by_round="R001",
        created_by_agent="test", file_paths={"xyz": str(xyz)}, metadata={"charge": 0, "multiplicity": 1}))
    case.runtime["tool_environment"] = {"amesp_enabled": True}
    cid = "microscopic.run_relaxed_coordinate_scan"
    result = action(case.case_id, state(case)["revision"], dispatch=True)
    result.dispatch_requests = [request(cid=cid, args={"source_artifact_id": "prepared"})]
    before = deepcopy(case.model_dump())
    assert REGISTRY.readiness(cid, case.artifact_manifest, amesp_enabled=True)["status"] == "disabled"
    with pytest.raises(RequestValidationError, match="No calculation ran"):
        validate_continuation(case, result, REGISTRY)
    assert case.model_dump() == before
    stop = ending(case)
    route = stop.continuation_assessment.routes[0]
    route.capability_id, route.disposition = cid, "blocked"
    validate_continuation(case, stop, REGISTRY)


def test_comparison_sources_remain_visible_in_action_and_repair():
    from mechcal.agents.observation_view import repair_payload
    from test_runtime import _evidence_unit

    case, update = start()
    record = _evidence_unit(evidence_id="E1", round_id="R001", source_report_id="report",
                           agent_name="microscopic_worker", family="charge_localization", summary=NOTE)
    case.evidence_ledger.items.append(record)
    update.ranking_update.candidate_bases[-1].counter_refs = ["E1"]
    case = apply_assimilation(case, update)
    payload = public_payload(case, REGISTRY, stage="planner_question_act")
    assert any(item["evidence_id"] == "E1" for item in payload["observations"])
    repaired = repair_payload(payload, {"issues": []})
    assert repaired["observations"] == payload["observations"]
    assert repaired["boundary_decision_context"]["enabled"]
    assert "hidden_reference" not in repaired and "rubric" not in repaired


def test_old_records_still_read_and_new_mode_requires_comparative_context(tmp_path):
    case = initial()
    validate_ranking(case, QuestionAssimilation.model_validate(update_for(case)))
    with pytest.raises(ValueError, match="requires decision_value_checks"):
        OrchestratorConfig(run_base_dir=tmp_path, evidence_decision_contracts=True)


def test_final_audit_can_stop_incomplete_instead_of_falsely_claiming_convergence():
    from mechcal.runtime.questions import apply_final_revision, decision_from_action
    from mechcal.schemas.question_loop import FinalArgumentRevision

    case, update = start()
    case = apply_assimilation(case, update)
    original = decision_from_action(ending(case), case, "R001")
    case = replace_state(case, phase="final_revision", decision=original.model_dump(mode="json"),
                         final_audit={"status": "reviewed"})
    data = complete_update(update_for(case), [h.name for h in case.portfolio.sorted_hypotheses()])
    corrected = QuestionAssimilation.model_validate(data)
    final = ending(case)
    final.revision = corrected.revision
    final.action = "stop"
    final.stopping_assessment.finish_basis = None
    final.continuation_assessment.routes[0].disposition = "useful"
    revision = FinalArgumentRevision(update=corrected, final_action=final, audit_responses={})
    result, decision = apply_final_revision(case, revision, REGISTRY)
    assert decision.action == "stop" and decision.stopping_assessment.finish_basis is None
    assert result.evidence_ledger == case.evidence_ledger
    # The same useful opportunity cannot justify an ordinary voluntary stop.
    result.runtime["question_loop"].pop("final_audit")
    with pytest.raises(RequestValidationError, match="useful route"):
        validate_continuation(result, final, REGISTRY)


def test_previously_named_unregistered_gap_does_not_create_an_unrepairable_stop():
    case, update = start()
    case = apply_assimilation(case, update)
    state(case)["questions"]["Q1"]["acceptable_capability_ids"] = ["not_registered"]
    final = ending(case)
    from mechcal.schemas.boundary_decisions import ConsideredRoute
    final.continuation_assessment.routes.append(ConsideredRoute(
        capability_id="not_registered", disposition="blocked", reason="Named earlier but absent from this registry."))
    validate_continuation(case, final, REGISTRY)
    final.continuation_assessment.routes[-1].disposition = "useful"
    with pytest.raises(RequestValidationError, match="unregistered route"):
        validate_continuation(case, final, REGISTRY)


@pytest.mark.parametrize("audit", [False, True])
@pytest.mark.parametrize("repair_final", [False, True])
def test_full_loop_persists_contracts_and_resumes_without_calls(tmp_path, audit, repair_final, monkeypatch):
    from mechcal.schemas.question_loop import ResultReview
    class Client(ComparisonClient):
        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            if payload.get("stage") == "planner_field_repair":
                assert repair_final and audit
                assert payload["allowed_paths"] == [["final_action", "continuation_assessment"]]
                self.repaired_update = deepcopy(payload["rejected_response"]["update"])
                return {"repairs": [{"path": payload["allowed_paths"][0], "value": dict(
                    scope="bounded_ranking", remaining_uncertainty=NOTE, decision_reason=NOTE,
                    routes=[dict(capability_id=TOOL, disposition="not_useful", reason=NOTE)])}]}
            data = super().complete_json(payload=payload, system_prompt=system_prompt, schema_feedback=schema_feedback)
            if data.get("action") == "describe_tools":
                return data
            stage = payload["stage"]
            if stage in {"planner_assimilate", "planner_final_revision"}:
                update = data["update"] if stage == "planner_final_revision" else data
                complete_update(update, payload["comparison_context"]["current_order"])
            actions = [data["final_action"]] if stage == "planner_final_revision" else [data]
            for a in actions:
                for plan in a.get("investigation_plans", []):
                    plan["decision_value"] = value().model_dump(mode="json")
                    plan["possible_outcomes"] = [o.model_dump() for o in outcomes("support", "no_change")]
                if a.get("action") in {"finalize", "stop"}:
                    a["continuation_assessment"] = dict(scope="bounded_ranking", remaining_uncertainty=NOTE,
                        decision_reason=NOTE, routes=[dict(capability_id=TOOL, disposition="not_useful", reason=NOTE)])
            if repair_final and stage == "planner_final_revision":
                data["final_action"]["continuation_assessment"] = None
            return data

    o, _ = loop(tmp_path, unlimited=True, discover=True)
    client = Client(1, True)
    o.deps.planner.llm_client = client
    monkeypatch.setattr(o.deps.photophysics_arbiter, "review_results", lambda case, **kw: ResultReview(
        interpretations=[dict(result_id=r["result_id"], question_ids=r["question_ids"],
                              delivered=NOTE, missing=NOTE, boundary=NOTE, evidence_refs=r["evidence_refs"])
                         for r in state(case).get("pending_results", [])]))
    o = MechCALOrchestrator(replace(o.config, comparative_evidence=True, decision_value_checks=True,
                                    evidence_decision_contracts=True, enable_support_auditor=audit), o.deps)
    result = o.run(smiles="CCO", user_query=NOTE, case_id="boundary-loop")
    assert result.status == "finalized", result.failure
    assert len(state(result)["results"]) == 1
    assert state(result)["assimilation"]["ranking_update"]["boundary_comparisons"]
    if repair_final and audit:
        from mechcal.schemas.evidence import HypothesisPortfolio
        assert result.portfolio == HypothesisPortfolio.model_validate(client.repaired_update["portfolio"])
    assert all("hidden_reference" not in p and "rubric" not in p for p in client.calls)
    before = len(client.calls)
    assert o.run(smiles="CCO", user_query=NOTE, case_id="boundary-loop", resume=True) == result
    assert len(client.calls) == before
