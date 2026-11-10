"""Bookkeeping retries must not restart scientific generation. No API calls."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from mechcal.runtime.field_repair import FieldRepairProducer
from mechcal.runtime.questions import validate_assimilation
from mechcal.runtime.retry import RequestValidationError, SchemaRetryError
from mechcal.schemas import CaseRun
from mechcal.schemas.question_loop import QuestionAssimilation
from test_field_repair import corrections, missing_receipt, runner


def replay(case, original, bad_receipt, *, exhaust=False):
    before = case.model_dump(mode="json")
    full_calls, repairs = [], []

    def full(feedback):
        full_calls.append(feedback)
        assert feedback is None, "Bookkeeping retry incorrectly restarted full generation"
        return deepcopy(original)

    def repair(payload, feedback):
        repairs.append(deepcopy(payload))
        assert payload["allowed_paths"] == [["argument_changes"]]
        receipt = deepcopy(bad_receipt)
        if len(repairs) == 2:
            assert payload["rejected_response"]["argument_changes"] == [bad_receipt]
            assert receipt["support_id"] in payload["issues"][0]["retained_previous_ids"]
            if not exhaust:
                # Scripted model correction, not a production fallback.
                receipt["replacement_ids"] = []
        return corrections(["argument_changes"], [receipt])

    producer = FieldRepairProducer(full, repair, response_model=QuestionAssimilation)
    if exhaust:
        with pytest.raises(SchemaRetryError, match="Revised arguments retain"):
            runner(case).run(producer)
        result = None
    else:
        result = runner(case).run(producer)
        expected = QuestionAssimilation.model_validate(original).model_dump()
        actual = result.model_dump()
        expected.pop("argument_changes")
        actual.pop("argument_changes")
        assert actual == expected
        validate_assimilation(result, case)
    assert len(full_calls) == 1 and len(repairs) == 2
    assert case.model_dump(mode="json") == before
    return result


@pytest.mark.parametrize("exhaust", [False, True])
def test_invalid_replacement_stays_scoped_with_original_attempt_budget(exhaust):
    case, original, receipt = missing_receipt()
    original["mechanism_support_arguments"].append({**original["mechanism_support_arguments"][0], "support_id": "new"})
    replay(case, original, {**receipt, "replacement_ids": ["new"]}, exhaust=exhaust)


@pytest.mark.parametrize("change", [
    {"disposition": "withdrawn"},
    {"disposition": "superseded", "replacement_ids": ["new"]},
    {"replacement_ids": ["new"]},
    {"replacement_ids": ["unknown"]},
    {"evidence_refs": ["unknown:evidence"]},
    {"reason": " "},
])
def test_change_record_errors_all_have_scoped_repairs(change):
    case, original, receipt = missing_receipt()
    original["argument_changes"] = [{**receipt, **change}]
    with pytest.raises(RequestValidationError) as caught:
        validate_assimilation(QuestionAssimilation.model_validate(original), case)
    assert all(i["repair_paths"] == [["argument_changes"]] for i in caught.value.issues)
    assert all("current_argument_ids" in i for i in caught.value.issues)
