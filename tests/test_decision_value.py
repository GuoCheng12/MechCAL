from copy import deepcopy
from dataclasses import replace

import pytest

from mechcal.runtime import MechCALOrchestrator
from mechcal.runtime.decision_value import validate_action_value, validate_stop_opportunity
from mechcal.runtime.questions import state, validate_action
from mechcal.schemas.decisions import StopReview
from mechcal.schemas.evidence_comparison import DecisionValue, OutcomeExpectation
from test_evidence_comparison import ComparisonClient, plan, ready
from test_question_loop import NOTE, REGISTRY, action, loop

CID = "macro.screen_rotor_torsion_topology"


def value(**changes):
    return DecisionValue(delivered_observable=NOTE, observation_scope=NOTE,
                         unresolved_extension=NOTE, why_not_redundant=NOTE, **changes)


def outcomes(*effects):
    return [OutcomeExpectation(observation=f"Synthetic outcome {i}", judgment_effect=NOTE,
                               decision_effect=effect) for i, effect in enumerate(effects)]


def prepared():
    c = ready()
    c = c.touch(runtime={**c.runtime, "decision_value_checks": True})
    a = action(c.case_id, state(c)["revision"], dispatch=True)
    p = plan()
    p.decision_value = value()
    p.possible_outcomes = outcomes("support", "no_change")
    a.investigation_plans = [p]
    return c, a


def test_meaningful_comparison_preserves_scientific_choice():
    c, a = prepared()
    before = deepcopy(c.model_dump(mode="json"))
    validate_action(a, c, REGISTRY)
    assert c.model_dump(mode="json") == before


@pytest.mark.parametrize("fault", ["missing", "all_no_change", "missing_effect", "whitespace"])
def test_vacuous_proposal_rejected_before_dispatch(fault):
    c, a = prepared()
    p = a.investigation_plans[0]
    if fault == "missing":
        p.decision_value = None
    elif fault == "whitespace":
        p.decision_value.delivered_observable = " "
    else:
        p.possible_outcomes = outcomes("no_change", "no_change" if fault == "all_no_change" else None)
    with pytest.raises(ValueError):
        validate_action(a, c, REGISTRY)


def test_preparation_allowed_with_available_downstream_not_invented_route():
    c, a = prepared()
    p = a.investigation_plans[0]
    p.purpose = "preparation"
    p.possible_outcomes = outcomes("preparation", "no_change")
    with pytest.raises(ValueError, match="downstream"):
        validate_action(a, c, REGISTRY)
    p.decision_value = value(downstream_capability_id=CID, downstream_question=NOTE)
    validate_action(a, c, REGISTRY)
    p.decision_value.downstream_capability_id = "invented.tool"
    with pytest.raises(ValueError):
        validate_action(a, c, REGISTRY)


@pytest.mark.parametrize("status", ["ready", "missing_input", "disabled"])
def test_downstream_preconditions_do_not_require_everything_already_computed(status):
    review = StopReview(rationale=NOTE, challenge=NOTE, capability_ids=[CID], continuation_opportunity={
        "capability_id": CID, "decision_value": value(downstream_capability_id="next", downstream_question=NOTE),
        "possible_outcomes": outcomes("preparation", "no_change")})
    catalog = {CID: {"status": "ready"}, "next": {"status": status}}
    if status == "disabled":
        with pytest.raises(ValueError):
            validate_stop_opportunity(review, catalog)
    else:
        validate_stop_opportunity(review, catalog)


def test_untried_tool_is_not_itself_continuation_justification():
    with pytest.raises(ValueError, match="explicit opportunity"):
        validate_stop_opportunity(StopReview(rationale=NOTE, capability_ids=[CID]), {CID: {"status": "ready"}})
    validate_stop_opportunity(StopReview(rationale=NOTE, challenge="Reasoning needs clarification."), {})


def test_flag_off_reads_old_records_and_planner_can_end_with_unresolved_scope():
    c, a = prepared()
    a.investigation_plans[0].decision_value = None
    validate_action_value(c.touch(runtime={**c.runtime, "decision_value_checks": False}), a, REGISTRY)
    ending = action(c.case_id, state(c)["revision"])
    validate_action(ending, c, REGISTRY)


def test_full_loop_preserves_decision_value_and_unlimited_planner_ending(tmp_path):
    class Client(ComparisonClient):
        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            data = super().complete_json(payload=payload, system_prompt=system_prompt, schema_feedback=schema_feedback)
            for p in data.get("investigation_plans", []):
                p["decision_value"] = value().model_dump(mode="json")
                for outcome in p["possible_outcomes"]:
                    outcome["decision_effect"] = "support"
            return data

    o, _ = loop(tmp_path, unlimited=True)
    o.deps.planner.llm_client = Client(1)
    o = MechCALOrchestrator(replace(o.config, comparative_evidence=True, decision_value_checks=True), o.deps)
    result = o.run(smiles="CCO", user_query=NOTE, case_id="decision-value-loop")
    assert result.status == "finalized"
    assert len(state(result)["results"]) == 1
    assert state(result)["results"][0]["investigation_plan"]["decision_value"]["delivered_observable"] == NOTE
