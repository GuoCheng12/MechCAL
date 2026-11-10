"""Scripted interface acceptance, not evidence of improved scientific accuracy."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from mechcal.agents.mechanism_agenda import MechanismAgendaAgent
from mechcal.agents.planner import PlannerAgent
from mechcal.runtime.persistence import RunStore
from mechcal.runtime.predecision import comparison_input, ensure_comparison, input_id, validate_review
from mechcal.runtime.questions import apply_assimilation, replace_state, state
from mechcal.runtime.retry import RequestValidationError
from mechcal.runtime.structured_output import PlannerOutputContract
from mechcal.schemas.predecision import PredecisionReview
from mechcal.schemas.ranking_explanations import RankingExplanations
from test_judgment_commands import command, explanation_response
from test_question_loop import REGISTRY
from test_ranking_explanation_transaction import prepared_case, proposal, draft_for


def with_observations():
    case = prepared_case()
    case.runtime["predecision_comparison"] = True
    return replace_state(case, results=[{
        "result_id": "r", "evidence_refs": ["e"], "status": "success",
        "capability_id": "test:observation", "structured_results": {"fixture_only": True},
    }])


def review_for(payload, outcome="unresolved"):
    return PredecisionReview(comparisons=[{
        "candidate_a": payload["mechanism_pool"][0], "candidate_b": payload["mechanism_pool"][1],
        "outcome": outcome, "observation": "Interface fixture only.",
        "interpretation": "No scientific inference from this synthetic record.",
        "remaining_gap": "No physical validation.", "evidence_refs": ["e"],
    }])


def test_reviewer_input_is_independent_of_old_order_and_has_no_private_data():
    case = with_observations()
    before = comparison_input(case)
    changed = case.model_copy(deep=True)
    changed.portfolio.hypotheses.reverse()
    for h in changed.portfolio.hypotheses:
        h.confidence, h.differential_priority, h.rationale = .99, .01, "PRIOR_SENTINEL"
    changed.runtime.update(hidden_reference="SECRET_GT", rubric="SECRET_RUBRIC", judge="SECRET_JUDGE")
    assert comparison_input(changed) == before
    assert input_id(comparison_input(changed)) == input_id(before)
    assert not any(term in json.dumps(before) for term in ("SECRET", "PRIOR_SENTINEL", "differential_priority"))


@pytest.mark.parametrize("outcome", ["favors_a", "favors_b", "unresolved", "compatible"])
def test_pairwise_advice_can_favor_either_or_neither(outcome):
    payload = comparison_input(with_observations())
    validate_review(review_for(payload, outcome), payload)


def test_review_provenance_and_no_operational_evidence_promotion():
    payload = comparison_input(with_observations())
    review = review_for(payload, "favors_a")
    payload["operational_only_evidence_refs"] = ["e"]
    with pytest.raises(RequestValidationError):
        validate_review(review, payload)
    review.comparisons[0].outcome = "unresolved"
    validate_review(review, payload)
    review.comparisons[0].evidence_refs = ["other-case:e"]
    with pytest.raises(RequestValidationError):
        validate_review(review, payload)


def test_review_checkpoint_reuse_and_new_observation_invalidation(tmp_path):
    case = with_observations()
    calls = []
    def compare(payload, **kwargs):
        calls.append(deepcopy(payload))
        return review_for(payload)
    o = SimpleNamespace(deps=SimpleNamespace(mechanism_agenda=SimpleNamespace(compare_evidence=compare)))
    store = RunStore(tmp_path, case.case_id)
    store.initialize()
    reviewed = ensure_comparison(o, case, store, "R002")
    assert reviewed.portfolio == case.portfolio
    assert reviewed.evidence_ledger == case.evidence_ledger
    assert reviewed.round_ids == case.round_ids
    assert len(calls) == 1
    ensure_comparison(o, reviewed, store, "R002")
    assert len(calls) == 1
    changed = reviewed.model_copy(deep=True)
    state(changed)["results"][0]["structured_results"]["new_value"] = 1.3
    changed = ensure_comparison(o, changed, store, "R003")
    assert len(calls) == 2
    assert state(changed)["predecision_review"]["review_id"] != state(reviewed)["predecision_review"]["review_id"]


def test_initial_state_does_not_invent_observations_or_require_review(tmp_path):
    case = prepared_case()
    case.runtime["predecision_comparison"] = True
    o = SimpleNamespace(deps=None)
    store = RunStore(tmp_path, case.case_id)
    store.initialize()
    assert ensure_comparison(o, case, store, "R001") == case


def test_initial_full_pool_command_uses_joint_ranking_schema(monkeypatch):
    from test_boundary_decisions import start
    from test_judgment_commands import initial_command
    case, original = start()
    case.runtime.update(incremental_judgment=True, predecision_comparison=True, decision_economy=True)
    cmd = initial_command(original).model_dump(mode="json")
    class Client:
        def is_configured(self):
            return True
        def complete_json(self, **kwargs):
            from jsonschema import Draft202012Validator
            schema = kwargs["payload"]["output_contract"]
            assert schema["properties"]["ranking_update"]["$ref"] == "#/$defs/RankingUpdate"
            Draft202012Validator(schema).validate(cmd)
            return deepcopy(cmd)
    monkeypatch.setattr("mechcal.agents.judgment_commands.explain_ranking", lambda *args, **kwargs: pytest.fail("Unexpected frozen explanation"))
    update = PlannerAgent(llm_client=Client(), enable_llm_decisions=True, capability_registry=REGISTRY).assimilate(case)
    assert update.portfolio == original.portfolio




@pytest.mark.parametrize("strict,memory", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("change", [False, True])
def test_planner_commits_ranking_and_comparisons_together_not_a_frozen_defense(monkeypatch, tmp_path, strict, change, memory):
    case = with_observations()
    case.runtime["planner_strict_outputs"] = strict
    if memory:
        case.runtime.update(planner_case_memory=True, planner_delta_context=True, scoped_evidence_updates=True)
    report = {"review_id": input_id(comparison_input(case)), **review_for(comparison_input(case)).model_dump(mode="json")}
    case = replace_state(case, predecision_review=report)
    changes = [dict(name=case.portfolio.current, differential_priority=0)] if change else []
    # This helper prepares a coherent scripted response; it is not a model policy.
    fixture_case = case.model_copy(deep=True)
    fixture_case.runtime["predecision_comparison"] = False
    draft = draft_for(fixture_case, proposal(fixture_case, changes))
    expected = draft.apply(RankingExplanations.model_validate(explanation_response(draft.payload())))
    cmd = command(case, hypotheses=changes, ranking_update=expected.ranking_update,
                  comparison_response={"review_id": report["review_id"], "response": "I disagree with the advisory interpretation; retain the actual observations."})
    calls = []
    class Client:
        def is_configured(self):
            return True
        def complete_json(self, **kwargs):
            payload = kwargs["payload"]
            calls.append(deepcopy(payload))
            assert payload["stage"] == "planner_assimilate"
            from mechcal.runtime.context_compaction import expand_payload
            assert expand_payload(payload)["predecision_review"]["review_id"] == report["review_id"]
            data = cmd.model_dump(mode="json")
            from test_delta_context import WireClient
            return WireClient([(data, "decision")]).complete_json(**kwargs) if strict else data
    def forbidden(*args, **kwargs):
        raise AssertionError("No post-ranking explanation call in comparison-first mode")
    monkeypatch.setattr("mechcal.agents.judgment_commands.explain_ranking", forbidden)
    agent = PlannerAgent(llm_client=Client(), enable_llm_decisions=True, capability_registry=REGISTRY)
    before = case.model_dump_json()
    store = RunStore(tmp_path, case.case_id)
    store.initialize()
    update = agent.assimilate(case, **({"memory_store": store} if memory else {}))
    assert len(calls) == 1
    assert case.model_dump_json() == before
    assert update.portfolio == expected.portfolio
    assert update.comparison_response.review_id == report["review_id"]
    committed = apply_assimilation(case, update)
    assert committed.portfolio == expected.portfolio


def test_stale_review_response_is_rejected_without_changing_judgment():
    from mechcal.runtime.predecision import validate_response
    from mechcal.schemas.predecision import ComparisonResponse
    case = replace_state(with_observations(), predecision_review={"review_id": "current"})
    before = case.model_dump_json()
    with pytest.raises(RequestValidationError):
        validate_response(case, ComparisonResponse(review_id="stale", response="acknowledged"))
    assert case.model_dump_json() == before


def test_reviewer_uses_strict_schema_without_model_scoring():
    payload = comparison_input(with_observations())
    class Client:
        def is_configured(self):
            return True
        def complete_json(self, **kwargs):
            assert kwargs["response_format"]["json_schema"]["strict"]
            assert "portfolio" not in kwargs["payload"]
            return PlannerOutputContract(kwargs["payload"]).encode(review_for(payload).model_dump(mode="json"))
    review = MechanismAgendaAgent(llm_client=Client(), capability_registry=REGISTRY).compare_evidence(payload)
    assert review.comparisons[0].outcome == "unresolved"


def test_transport_failure_does_not_replace_review_or_ranking(tmp_path):
    case = with_observations()
    store = RunStore(tmp_path, case.case_id)
    store.initialize()
    def fail(*args, **kwargs):
        raise TimeoutError("offline transport fixture")
    o = SimpleNamespace(deps=SimpleNamespace(mechanism_agenda=SimpleNamespace(compare_evidence=fail)))
    before = case.model_dump_json()
    with pytest.raises(TimeoutError):
        ensure_comparison(o, case, store, "R002")
    assert case.model_dump_json() == before
    assert "predecision_review" not in state(case)


@pytest.mark.parametrize("fault", ["stale_review", "wrong_leader"])
def test_new_contract_repairs_only_the_rejected_field(monkeypatch, fault):
    case = with_observations()
    report = {"review_id": "current", **review_for(comparison_input(case)).model_dump(mode="json")}
    case = replace_state(case, predecision_review=report)
    cmd = command(case, ranking_update=state(case)["assimilation"]["ranking_update"],
                  comparison_response={"review_id": "current", "response": "Advisory comparison considered."}).model_dump(mode="json")
    cmd["ranking_update"]["basis"] = "unchanged"
    invalid = deepcopy(cmd)
    if fault == "stale_review":
        invalid["comparison_response"]["review_id"] = "old"
        field, value = "comparison_response", cmd["comparison_response"]
    else:
        invalid["ranking_update"]["leading_comparison"]["leader"] = case.portfolio.runner_up
        field, value = "leading_comparison", cmd["ranking_update"]["leading_comparison"]
    calls = []
    class Client:
        def is_configured(self):
            return True
        def complete_json(self, **kwargs):
            payload = kwargs["payload"]
            calls.append(payload)
            if len(calls) == 1:
                return invalid
            assert payload["stage"] == "planner_judgment_repair"
            assert len(payload["targets"]) == 1
            target = payload["targets"][0]
            assert target["field"] == field
            return {"action": "repair_judgment", "draft_id": payload["draft_id"],
                    "edits": [{"target_id": target["target_id"], "operation": "replace", "value": value}]}
    monkeypatch.setattr("mechcal.agents.judgment_commands.explain_ranking", lambda *args, **kwargs: pytest.fail("Unexpected frozen explanation"))
    agent = PlannerAgent(llm_client=Client(), enable_llm_decisions=True, capability_registry=REGISTRY)
    update = agent.assimilate(case)
    assert len(calls) == 2
    assert update.portfolio == case.portfolio
