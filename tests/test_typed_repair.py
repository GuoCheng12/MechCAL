"""Typed repair instructions and the actual October 12 incident, without API calls."""

from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from mechcal.runtime.evidence_comparison import comparison_context, comparison_refs
from mechcal.runtime.field_repair import FieldRepairProducer, correction_contract
from mechcal.runtime.questions import apply_assimilation, state, validate_assimilation
from mechcal.runtime.retry import SchemaRetryError
from mechcal.schemas import CaseRun
from mechcal.schemas.evidence_comparison import ArgumentChange
from mechcal.schemas.question_loop import FinalArgumentRevision, QuestionAssimilation
from test_field_repair import corrections, missing_receipt, runner


@pytest.mark.parametrize("model,prefix", [(QuestionAssimilation, []), (FinalArgumentRevision, ["update"])])
def test_contract_contains_exact_target_type_and_rejects_internal_history_fields(model, prefix):
    path = [*prefix, "argument_changes"]
    schema = correction_contract(model, [path])
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    record = dict(support_id="S1", disposition="revised", reason="Clarified boundary.")
    validator.validate(corrections(path, [record]))
    assert schema["$defs"]["ArgumentChange"] == ArgumentChange.model_json_schema()
    for extra in ({"revision": 9}, {"observed_change": True}, {"previous_argument": {}}):
        assert not validator.is_valid(corrections(path, [record | extra]))
    assert not validator.is_valid(corrections(path, [{"support_id": "S1"}]))
    assert not validator.is_valid(corrections(path, "arbitrary value"))
    assert not validator.is_valid(corrections(["portfolio"], {}))


@pytest.mark.parametrize("path,value", [
    (["questions", 0, "result_ids"], ["execution:example"]),
    (["portfolio", "hypotheses", 0, "evidence_refs"], ["input:smiles"]),
    (["mechanism_support_arguments", 0, "observation_refs"], ["input:smiles"]),
    (["ranking_update", "evidence_refs"], ["input:smiles"]),
    (["dispositions", 0, "result_id"], "execution:example"),
    (["dispositions", 0, "mechanism_effects", 0, "evidence_refs"], ["input:smiles"]),
])
def test_reference_fields_derive_correct_scalar_or_array_type(path, value):
    schema = correction_contract(QuestionAssimilation, [path])
    validator = Draft202012Validator(schema)
    validator.validate(corrections(path, value))
    assert not validator.is_valid(corrections(path, 42))


def test_history_projection_separates_author_record_and_program_metadata():
    case, data, receipt = missing_receipt()
    data["argument_changes"] = [receipt]
    updated = apply_assimilation(case, QuestionAssimilation.model_validate(data))
    before = updated.model_dump(mode="json")
    public = comparison_context(updated)
    record = public["latest_update"]["argument_changes"][0]
    ArgumentChange.model_validate(record)
    assert not set(record) & {"previous_argument", "observed_change", "revision"}
    snapshot = public["read_only_audit"]["argument_snapshots"][0]
    assert snapshot["before"] == state(updated)["argument_history"][0]["previous_argument"]
    assert "input:smiles" in comparison_refs(updated)
    assert updated.model_dump(mode="json") == before


def test_impossible_schema_path_is_a_program_error_not_a_model_retry():
    with pytest.raises(RuntimeError, match="absent"):
        correction_contract(QuestionAssimilation, [["does_not_exist"]])


def test_failed_correction_and_precise_errors_survive_the_next_request_and_log():
    case, original, receipt = missing_receipt()
    bad = corrections(["argument_changes"], [receipt | {"observed_change": True}])
    calls, events = [], []

    def repair(payload, feedback):
        calls.append(deepcopy(payload))
        if len(calls) == 1:
            assert payload["last_failed_correction"] is None
            return bad
        failure = payload["last_failed_correction"]
        assert failure["submitted_correction"] == bad
        assert failure["errors"][0]["loc"] == ["argument_changes", 0, "observed_change"]
        assert failure["errors"][0]["type"] == "extra_forbidden"
        assert payload["rejected_response"] == original
        return corrections(["argument_changes"], [receipt])

    result = runner(case, event_sink=events.append).run(FieldRepairProducer(
        lambda _: original, repair, response_model=QuestionAssimilation))
    assert len(calls) == 2
    assert result.portfolio == QuestionAssimilation.model_validate(original).portfolio
    logged = [e for e in events if e.get("repair_failure")]
    assert len(logged) == 1 and logged[0]["repair_failure"]["submitted_correction"] == bad
