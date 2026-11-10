"""Provider schema lowering must preserve the accepted scientific data contract."""

from copy import deepcopy
import json

from jsonschema import Draft202012Validator
import pytest

from mechcal.runtime.structured_output import PlannerOutputContract, provider_response_format
from mechcal.schemas.judgment_commands import JudgmentCommand


def sample_format():
    return {"type": "json_schema", "json_schema": {"name": "test", "strict": True, "schema": {
        "type": "object", "additionalProperties": False, "required": ["answer"],
        "$defs": {"Entry": {"type": "object", "additionalProperties": False,
                            "required": ["score"], "properties": {"score": {
                                "type": "number", "minimum": 0, "maximum": 1}}}},
        "properties": {"answer": {"anyOf": [{"$ref": "#/$defs/Entry"}, {"type": "null"}]}},
    }}}


@pytest.mark.parametrize("answer", [None, {"score": 0}, {"score": .5}, {"score": 1},
                                  {"score": 2}, {}, {"score": "0.5"}, {"score": 1, "extra": 0}])
def test_reference_expansion_preserves_validation_and_input(answer):
    fmt = sample_format()
    before = deepcopy(fmt)
    converted = provider_response_format(fmt, "https://api.deepseek.com")
    a = Draft202012Validator(fmt["json_schema"]["schema"])
    b = Draft202012Validator(converted["json_schema"]["schema"])
    assert a.is_valid({"answer": answer}) == b.is_valid({"answer": answer})
    assert fmt == before
    assert converted["json_schema"]["strict"] is True
    assert "$ref" not in json.dumps(converted)
    assert provider_response_format(converted, "https://api.deepseek.com") == converted


@pytest.mark.parametrize("base", ["https://gateway.example.test", "https://provider.test",
                                  "https://api.deepseek.com.attacker.test"])
def test_other_providers_are_unchanged(base):
    fmt = sample_format()
    assert provider_response_format(fmt, base) is fmt


def test_json_object_and_none_unchanged():
    assert provider_response_format(None, "https://api.deepseek.com") is None
    fmt = {"type": "json_object"}
    assert provider_response_format(fmt, "https://api.deepseek.com") is fmt


@pytest.mark.parametrize("ref", ["https://example.com/schema", "#/$defs/Missing", "#/$defs/Entry"])
def test_unresolvable_and_recursive_refs_fail_locally(ref):
    fmt = sample_format()
    fmt["json_schema"]["schema"]["$defs"]["Entry"] = {"$ref": ref}
    with pytest.raises(ValueError):
        provider_response_format(fmt, "https://api.deepseek.com")


def test_conflicting_sibling_does_not_weaken_schema():
    fmt = sample_format()
    fmt["json_schema"]["schema"]["properties"]["answer"]["anyOf"][0]["type"] = "string"
    with pytest.raises(ValueError, match="conflicting"):
        provider_response_format(fmt, "https://api.deepseek.com")


def test_actual_planner_schema_lowers_all_reference_branches():
    contract = PlannerOutputContract({"output_contract": JudgmentCommand.model_json_schema(),
                                      "context_version": "planner_delta_v1",
                                      "stage": "planner_assimilate"})
    fmt = provider_response_format(contract.response_format(), "https://api.deepseek.com")
    assert "$ref" not in json.dumps(fmt)
    assert "$defs" not in json.dumps(fmt)
    assert fmt["json_schema"]["strict"]


def test_actual_judgment_round_trip_retains_the_original_contract():
    from test_decision_economy import economy_start
    from test_judgment_commands import initial_command

    _, update = economy_start()
    command = initial_command(update).model_dump(mode="json")
    contract = PlannerOutputContract({"output_contract": JudgmentCommand.model_json_schema()})
    encoded = contract.encode(command)
    lowered = provider_response_format(contract.response_format(), "https://api.deepseek.com")
    Draft202012Validator(lowered["json_schema"]["schema"]).validate(encoded)
    assert contract.decode(encoded) == command
    broken = deepcopy(encoded)
    broken["response"]["data"]["hypotheses"][0]["confidence"] = 5
    for schema in (contract.schema, lowered["json_schema"]["schema"]):
        assert not Draft202012Validator(schema).is_valid(broken)
