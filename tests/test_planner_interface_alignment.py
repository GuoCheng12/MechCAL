"""Reproduce provider-independent contract failures without paid model calls."""

from copy import deepcopy
import json

from jsonschema import Draft202012Validator
from pydantic import BaseModel
import pytest

from mechcal.agents.planner_references import PlannerReferences
from mechcal.capabilities import default_capability_registry
from mechcal.runtime.retry import RequestValidationError, SchemaRetryRunner
from mechcal.runtime.schema_diagnostics import schema_issues
from mechcal.runtime.structured_output import PlannerOutputContract, provider_response_format, request_payload
from mechcal.schemas.memory import ContextReadPlan
from mechcal.schemas.question_loop import QuestionAction
from mechcal.schemas.tool_arguments import validate_tool_args
from test_incremental_judgment import prepared


def contract(**extra):
    return PlannerOutputContract({"stage": "planner_question_act", "context_version": "planner_delta_v1",
        "output_contract": QuestionAction.model_json_schema(),
        "memory_access": {"queries_remaining": 4}, "tool_discovery": {"remaining_batches": 2}, **extra})


def read(**extra):
    return {"action": "prepare_context", "memory_queries": [], "describe_capability_ids": [],
            "focus_capability_ids": [], **extra}


def test_tool_cards_do_not_allow_explicit_null_but_keep_optional_omission():
    registry = default_capability_registry()
    for card in registry.cards():
        schema = card["tool_arg_schema"]
        for name, prop in schema.get("properties", {}).items():
            assert not Draft202012Validator({**prop, "$defs": schema.get("$defs", {})}).is_valid(None), (
                card["capability_id"], name)
    card = next(c for c in registry.cards() if c["route"] == "screen_donor_acceptor_layout")
    schema = Draft202012Validator(card["tool_arg_schema"])
    assert schema.is_valid({}) and schema.is_valid({"focus_tags": ["donor"]})
    assert not schema.is_valid({"focus_tags": None})
    assert validate_tool_args(card["route"], {}, "macro") == {}
    with pytest.raises(ValueError, match="Omit optional"):
        validate_tool_args(card["route"], {"focus_tags": None}, "macro")


@pytest.mark.parametrize("enabled", [True, False])
def test_enabled_scientific_requirements_are_nonnull_on_wire(enabled):
    api = contract(hypothesis_test_context={"enabled": enabled}, decision_value_checks=enabled)
    for definition, name in [("InvestigationPlan", "hypothesis_test"),
                             ("InvestigationPlan", "decision_value"), ("OutcomeExpectation", "decision_effect")]:
        logical = api.logical["decision"]
        prop = logical["$defs"][definition]["properties"][name]
        assert Draft202012Validator({**prop, "$defs": logical["$defs"]}).is_valid(None) is not enabled
    original = QuestionAction.model_json_schema()
    assert any(s.get("type") == "null" for s in original["$defs"]["InvestigationPlan"]["properties"]["hypothesis_test"]["anyOf"])


@pytest.mark.parametrize("lookup", [True, False])
@pytest.mark.parametrize("remaining", [0, 1, 4])
def test_query_schema_tracks_current_allowance_without_disabling_focus(lookup, remaining):
    extra = {"memory_access": {"queries_remaining": remaining}, "tool_discovery": {"remaining_batches": 0}}
    if lookup:
        extra.update(stage="planner_context_lookup", output_contract=ContextReadPlan.model_json_schema(),
                     context_access={"lookup_active": True})
    api = contract(**extra)
    kind = "decision" if lookup else "context_read"
    schema = api.logical[kind]
    validator = Draft202012Validator(schema)
    assert schema["properties"]["memory_queries"]["maxItems"] == remaining
    assert not validator.is_valid(read(describe_capability_ids=["new_tool"]))
    assert validator.is_valid(read(focus_capability_ids=["already_loaded_tool"]))
    assert not validator.is_valid(read(focus_capability_ids=[str(i) for i in range(8)]))
    if lookup or remaining == 0:
        assert "context_needed" not in api.logical


def test_context_reopening_is_removed_when_exhausted():
    assert "context_needed" in contract().logical
    assert "context_needed" not in contract(context_access={"reopens_remaining": 0}).logical


def test_query_instructions_do_not_offer_exhausted_branches():
    api = contract(context_access={"reopens_remaining": 0})
    payload = request_payload({"context_version": "planner_delta_v1", "memory_access": {"queries_remaining": 0}},
                              api.response_format())
    assert "context_needed" not in payload["memory_access"]["query_contract"]


def test_union_error_reports_selected_branch_field_without_dumping_answer():
    api = contract()
    wire = api.encode(read(), "context_read")
    wire["response"]["data"]["focus_capability_ids"] = list(map(str, range(8)))
    error = next(Draft202012Validator(api.schema).iter_errors(wire))
    assert schema_issues(error) == [{"path": ["response", "data", "focus_capability_ids"],
                                   "code": "schema_maxItems", "message": "At most 6 items; received 8."}]
    # Provider-specific schema expansion preserves the same rejection.
    expanded = provider_response_format(api.response_format(), "https://api.deepseek.com")
    assert not Draft202012Validator(expanded["json_schema"]["schema"]).is_valid(wire)


def test_correction_receives_original_wire_not_internal_decision():
    class Answer(BaseModel):
        value: int

    case = prepared()
    case.runtime["planner_strict_outputs"] = True
    before = case.model_dump_json()
    refs = PlannerReferences(case)
    payload = {"output_contract": Answer.model_json_schema()}
    api = PlannerOutputContract(payload)
    first = api.encode({"value": 0})
    calls = []

    class Client:
        def complete_json(self, **kwargs):
            calls.append(deepcopy(kwargs))
            return first if len(calls) == 1 else api.encode({"value": 1})

    def validate(answer):
        if answer.value == 0:
            raise RequestValidationError([{"path": "value", "code": "test_requirement", "message": "Expected one."}])

    result = SchemaRetryRunner(Answer, validators=[validate], repair_context=True).run(
        lambda feedback: refs.complete(Client(), payload=payload, system_prompt="Test", schema_feedback=feedback))
    feedback = json.loads(calls[-1]["schema_feedback"])
    assert result.value == 1 and len(calls) == 2
    assert feedback["rejected_response"] == first
    assert feedback["response_representation"] == "planner_response_v1_wire"
    assert feedback["issue_paths_relative_to"] == "/response/data"
    assert case.model_dump_json() == before


def test_internal_materialized_snapshot_is_not_exposed_as_output_shape():
    refs = PlannerReferences(prepared())
    feedback = refs.wire_feedback(json.dumps({"error": "invalid", "issues": [],
                                            "rejected_response": {"action": "dispatch", "unknown": 1}}))
    result = json.loads(feedback)
    assert "rejected_response" not in result
    assert result["response_representation"] == "issues_only"
