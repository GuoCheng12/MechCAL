"""Upsert regression: preserve model content, never choose a scientific answer."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from mechcal.runtime.decision_economy import expand_retained
from mechcal.runtime.questions import apply_assimilation, replace_state, state, validate_assimilation
from mechcal.runtime.retry import RequestValidationError
from mechcal.schemas import CaseRun
from mechcal.schemas.question_loop import QuestionAssimilation
from test_decision_economy import economy_start, next_update, retained_update
from test_evidence_comparison import argument


@pytest.mark.parametrize("field", ["hypothesis", "argument", "question", "basis"])
def test_explicit_update_overrides_selected_base_and_leaves_case_untouched(field):
    case, update = economy_start()
    case = replace_state(apply_assimilation(case, update), phase="assimilate")
    update = retained_update(case, next_update(case))
    if field == "hypothesis":
        item = case.portfolio.hypotheses[0].model_copy(deep=True)
        item.rationale = "New scientific explanation supplied by the model."
        update.portfolio.hypotheses = [item]
        get = lambda u: u.portfolio.hypotheses
    elif field == "argument":
        item = case.mechanism_support_arguments[0].model_copy(deep=True)
        item.warrant = "New model-authored warrant."
        update.mechanism_support_arguments = [item]
        get = lambda u: u.mechanism_support_arguments
    elif field == "question":
        from mechcal.schemas.mechanisms import EvidenceQuestion
        item = EvidenceQuestion.model_validate(next(iter(state(case)["questions"].values())))
        item.assessment = "New model-authored assessment."
        update.questions = [item]
        get = lambda u: u.questions
    else:
        from mechcal.schemas.boundary_decisions import CandidateBasis
        item = CandidateBasis.model_validate(state(case)["assimilation"]["ranking_update"]["candidate_bases"][0])
        item.inference_gap = "New model-authored limitation."
        update.ranking_update.candidate_bases = [item]
        get = lambda u: u.ranking_update.candidate_bases
    before = deepcopy(case.model_dump())
    expand_retained(case, update)
    assert item in get(update)
    assert case.model_dump() == before
    again = deepcopy(update.model_dump())
    expand_retained(case, update)
    assert update.model_dump() == again


def test_new_record_may_be_listed_as_base_only_when_fully_submitted():
    case, update = economy_start()
    case = replace_state(apply_assimilation(case, update), phase="assimilate")
    update = retained_update(case, next_update(case))
    update.retained_judgment.support_ids.append("new")
    with pytest.raises(RequestValidationError, match="Unknown IDs"):
        expand_retained(case, update)
    new = argument("new")
    update.mechanism_support_arguments = [new]
    expand_retained(case, update)
    assert new in update.mechanism_support_arguments


def test_full_snapshot_and_base_selection_have_one_unambiguous_result():
    case, update = economy_start()
    case = replace_state(apply_assimilation(case, update), phase="assimilate")
    full = next_update(case)
    selected = retained_update(case, full.model_copy(deep=True)).retained_judgment
    full.retained_judgment = selected
    before = full.model_copy(deep=True)
    validate_assimilation(full, case)
    assert full.portfolio == before.portfolio
    assert full.mechanism_support_arguments == before.mechanism_support_arguments
    assert full.ranking_update == before.ranking_update
