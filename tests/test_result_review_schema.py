"""Result review wire constraints and bounded recovery, without model calls."""

from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from mechcal.agents.photophysics_arbiter import PhotophysicsArbiterAgent
from mechcal.runtime.llm import _responses_request
from mechcal.runtime.retry import SchemaRetryError
from mechcal.schemas.question_loop import ResultReview
from test_worker_explanation import case_with_explanation, review_payload


class Client:
    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def complete_json(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return deepcopy(self.replies[min(len(self.calls) - 1, len(self.replies) - 1)])


def fixture():
    case = case_with_explanation()
    case.runtime['planner_strict_outputs'] = True
    return case, ResultReview.model_validate(review_payload()).model_dump(mode='json', exclude={'impact_review'})


@pytest.mark.parametrize('strict', [False, True])
def test_existing_switch_selects_review_wire_mode_without_mutating_science(strict):
    case, reply = fixture()
    case.runtime['planner_strict_outputs'] = strict
    before = case.model_dump_json()
    client = Client([reply])
    review = PhotophysicsArbiterAgent(llm_client=client).review_results(case)
    assert review.status == 'reviewed'
    call = client.calls[0]
    assert ('response_format' in call) == strict
    if strict:
        spec = call['response_format']['json_schema']
        assert spec['strict'] and spec['name'] == 'mechcal_result_review_v1'
        assert spec['schema'] == call['payload']['output_contract']
        Draft202012Validator.check_schema(spec['schema'])
        Draft202012Validator(spec['schema']).validate(reply)
        converted = _responses_request({'messages': [], 'response_format': call['response_format']})
        assert converted['text']['format'] == {'type': 'json_schema', **spec}
    assert case.model_dump_json() == before


def test_wire_missing_default_field_is_repaired_before_pydantic_can_fill_it():
    case, reply = fixture()
    bad = deepcopy(reply)
    bad.pop('reason')
    client = Client([bad, reply])
    review = PhotophysicsArbiterAgent(llm_client=client).review_results(case)
    assert review.status == 'reviewed' and len(client.calls) == 2
    assert 'reason' in client.calls[1]['schema_feedback']


@pytest.mark.parametrize('fault', ['wrong_reference', 'missing_interpretation', 'missing_wire_field'])
def test_schema_and_case_validation_still_reject_invalid_reviews_with_bounded_retries(fault):
    case, reply = fixture()
    before = case.model_dump_json()
    if fault == 'wrong_reference':
        reply['interpretations'][0]['evidence_refs'] = ['FOREIGN:E1']
    elif fault == 'missing_interpretation':
        reply['interpretations'] = []
    else:
        reply.pop('reason')
    client = Client([reply])
    with pytest.raises(SchemaRetryError):
        PhotophysicsArbiterAgent(llm_client=client).review_results(case)
    assert len(client.calls) == 3
    assert client.calls[1]['schema_feedback'] and client.calls[2]['schema_feedback']
    assert case.model_dump_json() == before
