"""Current model-authored decision value, not a stale role, governs dispatch."""

from copy import deepcopy

import pytest

from mechcal.runtime.questions import state, validate_action
from test_decision_economy import meaningful_plan
from test_decision_value import outcomes
from test_question_loop import REGISTRY


@pytest.mark.parametrize("role", ["material", "limitation"])
def test_concrete_new_plan_can_revisit_question_without_mutating_judgment(role):
    case, action, _ = meaningful_plan()
    for question in state(case)["questions"].values():
        question["decision_role"] = role
    before = deepcopy(case.model_dump())
    validate_action(action, case, REGISTRY)
    assert case.model_dump() == before


@pytest.mark.parametrize("role", ["material", "limitation"])
@pytest.mark.parametrize("fault", ["no_plan", "no_benefit", "no_value", "unknown_question"])
def test_reconsideration_still_requires_complete_plan_and_existing_question(role, fault):
    case, action, plan = meaningful_plan()
    for question in state(case)["questions"].values():
        question["decision_role"] = role
    if fault == "no_plan":
        action.investigation_plans = []
    elif fault == "no_benefit":
        plan.possible_outcomes = outcomes("no_change")
    elif fault == "no_value":
        plan.decision_value = None
    else:
        action.dispatch_requests[0].question_ids = ["UNKNOWN"]
    before = deepcopy(case.model_dump())
    with pytest.raises(ValueError):
        validate_action(action, case, REGISTRY)
    assert case.model_dump() == before
