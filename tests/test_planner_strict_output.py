"""Strict wire contracts, sparse updates and reproductions of the R003 failure."""

from copy import deepcopy
import json

import pytest
from jsonschema import Draft202012Validator, ValidationError as JSONSchemaError
from pydantic import ValidationError

from mechcal.agents.planner_references import PlannerReferences
from mechcal.runtime.judgment_draft import JudgmentDraft
from mechcal.runtime.question_changes import apply_question_changes
from mechcal.runtime.persistence import RunStore
from mechcal.runtime.questions import state
from mechcal.runtime.retry import RejectedResponseError
from mechcal.runtime.structured_output import PlannerOutputContract
from mechcal.schemas.judgment_commands import JudgmentCommand, FinalJudgmentCommand, QuestionUpdate
from mechcal.schemas.memory import MemoryQuery
from mechcal.schemas.question_loop import QuestionAction
from mechcal.schemas.boundary_decisions import ContinuationAssessment, ConsideredRoute
from test_decision_economy import economy_start
from test_incremental_judgment import prepared
from test_judgment_commands import initial_command, command, planner, repair_fields
from test_question_loop import action
from test_final_evidence_submission import submission_for


def contract(model=JudgmentCommand, **extra):
    return PlannerOutputContract({"output_contract": model.model_json_schema(), **extra})


def assert_strict(schema):
    if isinstance(schema, list):
        for item in schema:
            assert_strict(item)
    elif isinstance(schema, dict):
        assert not {"oneOf", "discriminator", "default", "prefixItems"} & schema.keys()
        if schema.get("type") == "object":
            assert schema["additionalProperties"] is False
            assert set(schema["required"]) == set(schema["properties"])
        for item in schema.values():
            assert_strict(item)


@pytest.mark.parametrize("model", [JudgmentCommand, FinalJudgmentCommand, QuestionAction])
def test_all_planner_contracts_compile_to_closed_strict_objects(model):
    result = contract(model, memory_access={"queries_remaining": 1, "query_schema": MemoryQuery.model_json_schema()},
                      tool_discovery={"remaining_batches": 1})
    assert_strict(result.schema)
    assert result.response_format()["json_schema"]["strict"] is True


def test_partial_object_intersections_fail_before_a_model_request():
    schema = {"type": "object", "properties": {"value": {"type": "string"}},
              "anyOf": [{"properties": {"value": {"const": "one"}}}]}
    with pytest.raises(ValueError, match="complete object alternatives"):
        PlannerOutputContract({"output_contract": schema})


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("invalid", ["empty_dispatch", "stop_with_requests", "empty_final"])
def test_action_shape_matches_local_and_wire_contracts(nested, invalid):
    _, update = economy_start()
    candidate = action("test", 1, dispatch=True).model_dump(mode="json")
    valid_action = deepcopy(candidate)
    expected_field = "dispatch_requests"
    if invalid == "empty_dispatch":
        candidate["dispatch_requests"] = []
    elif invalid == "stop_with_requests":
        candidate["action"] = "stop"
    else:
        candidate.update(action="finalize", dispatch_requests=[], final_answer_draft=" ")
        expected_field = "final_answer_draft"
    model, data, prefix = QuestionAction, candidate, ()
    if nested:
        model, prefix = FinalJudgmentCommand, ("final_action",)
        data = {"action": "revise_final", "update": initial_command(update).model_dump(mode="json"),
                "final_action": candidate, "audit_responses": {},
                "evidence_submission": submission_for(update.portfolio, update.mechanism_support_arguments)}
    with pytest.raises(ValidationError) as caught:
        model.model_validate(data)
    assert prefix + (expected_field,) in {tuple(e["loc"]) for e in caught.value.errors()}
    assert not Draft202012Validator(model.model_json_schema()).is_valid(data)
    api = contract(model)
    valid = {**data, "final_action": valid_action} if nested else valid_action
    wire = api.encode(valid)
    target = wire["response"]["data"]
    if nested:
        target = target["final_action"]
    target["action"] = candidate["action"]
    if not candidate["dispatch_requests"]:
        target["dispatch_requests"] = []
    target["final_answer_draft"] = candidate["final_answer_draft"]
    assert not Draft202012Validator(api.schema).is_valid(wire)


def test_create_requires_role_and_all_core_fields_in_both_contracts():
    _, update = economy_start()
    good = initial_command(update).model_dump(mode="json")
    bad = deepcopy(good)
    bad["questions"][0].pop("decision_role")
    bad["questions"][0].pop("observable")
    with pytest.raises(ValidationError) as caught:
        JudgmentCommand.model_validate(bad)
    assert {tuple(e["loc"]) for e in caught.value.errors()} == {
        ("questions", 0, "create", "decision_role"), ("questions", 0, "create", "observable"),
    }
    api = contract()
    raw = api.encode(good)
    del raw["response"]["data"]["questions"][0]["decision_role"]
    assert not Draft202012Validator(api.schema).is_valid(raw)


@pytest.mark.parametrize("final", [False, True])
def test_two_missing_roles_are_repaired_together_without_replacing_questions(final):
    case, update = economy_start()
    good = initial_command(update).model_dump(mode="json")
    good["questions"] += [{**deepcopy(good["questions"][0]), "question_id": "second",
                            "decision_role": "limitation"}]
    model = JudgmentCommand
    prefix = []
    if final:
        model = FinalJudgmentCommand
        good = {"action": "revise_final", "update": good,
                "final_action": action(case.case_id, 2).model_dump(mode="json"), "audit_responses": {},
                "evidence_submission": submission_for(update.portfolio, update.mechanism_support_arguments)}
        prefix = ["update"]
    bad = deepcopy(good)
    body = bad["update"] if final else bad
    for q in body["questions"]:
        del q["decision_role"]
    with pytest.raises(ValidationError) as caught:
        model.model_validate(bad)
    draft = JudgmentDraft(case.case_id, 5, "assimilate", model, {
        "rejected_response": bad, "error": "Missing roles", "issues": caught.value.errors(),
    })
    assert [t["path"] for t in draft.targets.values()] == [
        prefix + ["questions", 0, "decision_role"], prefix + ["questions", 1, "decision_role"],
    ]
    patch = {"action": "repair_judgment", "draft_id": draft.draft_id, "edits": [
        {"target_id": target_id, "operation": "replace", "value": role}
        for target_id, role in zip(draft.targets, ["material", "limitation"])
    ]}
    api = PlannerOutputContract(draft.payload())
    assert draft.apply(api.decode(api.encode(patch))) == good
    assert draft.original == bad


def test_role_only_update_preserves_all_previous_question_fields():
    _, update = economy_start()
    q = initial_command(update).questions[0]
    old = apply_question_changes({}, [q])[0].model_dump(mode="json")
    old["decision_role"] = None  # Historical records remain readable.
    result = apply_question_changes({q.question_id: old}, [QuestionUpdate(
        operation="update", question_id=q.question_id, decision_role="material",
    )])[0].model_dump(mode="json")
    assert result == {**old, "decision_role": "material"}


def test_maps_are_losslessly_decoded_and_logically_validated():
    data = command(prepared(), audit_responses={"S1": "Bounded evidence."}).model_dump(mode="json")
    api = contract()
    raw = api.encode(data)
    assert json.loads(raw["response"]["data"]["audit_responses"]) == data["audit_responses"]
    assert api.decode(raw) == data
    raw["response"]["data"]["audit_responses"] = '{"S1": 5}'
    with pytest.raises(JSONSchemaError):
        api.decode(raw)
    raw["response"]["data"]["audit_responses"] = "not JSON"
    with pytest.raises((ValueError, JSONSchemaError)):
        api.decode(raw)


@pytest.mark.parametrize("bad", ["", "  ", "[]", "null", '""'])
def test_encoded_object_fields_cannot_be_empty_strings_or_other_json_types(bad):
    data = ContinuationAssessment(scope="bounded_ranking", remaining_uncertainty="Test scope",
        decision_reason="Test decision", routes=[ConsideredRoute(
            capability_id="test", tool_args={}, disposition="not_useful", reason="Test reason",
        )]).model_dump(mode="json")
    api = contract(ContinuationAssessment)
    good = api.encode(data)
    assert good["response"]["data"]["routes"][0]["tool_args"] == "{}"
    assert api.decode(good) == data
    good["response"]["data"]["routes"][0]["tool_args"] = bad
    assert not Draft202012Validator(api.schema).is_valid(good)


def test_malformed_encoded_json_reports_nested_field_without_guessing_values():
    data = ContinuationAssessment(scope="bounded_ranking", remaining_uncertainty="Test scope",
        decision_reason="Test decision", routes=[ConsideredRoute(
            capability_id="test", tool_args={}, disposition="not_useful", reason="Test reason",
        )]).model_dump(mode="json")
    api = contract(ContinuationAssessment)
    wire = api.encode(data)
    wire["response"]["data"]["routes"][0]["tool_args"] = "{malformed}"
    with pytest.raises(ValueError, match=r"\['routes', 0, 'tool_args'\]"):
        api.decode(wire)
    assert wire["response"]["data"]["routes"][0]["tool_args"] == "{malformed}"


@pytest.mark.parametrize("remaining", [0, 1])
def test_query_alternatives_respect_allowances(remaining):
    api = contract(memory_access={"queries_remaining": remaining, "query_schema": MemoryQuery.model_json_schema()},
                   tool_discovery={"remaining_batches": remaining})
    assert ("memory_query" in api.logical) == bool(remaining)
    assert ("describe_tools" in api.logical) == bool(remaining)
    if remaining:
        query = MemoryQuery(action="find_compatible_artifacts", capability_id="example",
                            tool_args={"solvent": "water", "indices": [1, 2]}).model_dump(mode="json")
        assert api.decode(api.encode(query, "memory_query")) == query
        query = {"action": "describe_tools", "capability_ids": ["example"]}
        assert api.decode(api.encode(query, "describe_tools")) == query


def test_strict_planner_integration_records_wire_and_decoded_data_without_mutating_state():
    case = prepared()
    case.runtime["planner_strict_outputs"] = True
    good = command(case).model_dump(mode="json")

    def reply(payload):
        return PlannerOutputContract(payload).encode(good)

    agent, client = planner(case, [reply])
    before = case.model_dump(mode="json")
    events = []
    result = agent.assimilate(case, request_event_sink=events.append)
    assert result.coverage_response == good["coverage_response"]
    assert client.calls[0]["response_format"]["json_schema"]["strict"] is True
    assert any(e["kind"] == "planner_response_decoded" for e in events)
    assert case.model_dump(mode="json") == before


def test_invalid_wire_fails_explicitly_without_fallback_or_fake_command_draft():
    case = prepared()
    case.runtime["planner_strict_outputs"] = True
    calls = []

    class Client:
        def complete_json(self, **kwargs):
            calls.append(kwargs)
            return {"wrong": "shape"}

    with pytest.raises(RejectedResponseError, match="Strict Planner") as caught:
        PlannerReferences(case).complete(Client(), payload={"output_contract": JudgmentCommand.model_json_schema()},
                                         system_prompt="Test")
    assert isinstance(caught.value.response, str)
    assert len(calls) == 1 and calls[0]["response_format"]["type"] == "json_schema"


def test_disabled_mode_preserves_legacy_request_shape():
    case = prepared()
    agent, client = planner(case, [command(case).model_dump(mode="json")])
    agent.assimilate(case)
    assert "response_format" not in client.calls[0]


def test_strict_memory_queries_and_semantic_field_repair_share_existing_budget(tmp_path):
    case = prepared()
    state(case)["round_index"] = 1
    case.runtime["planner_strict_outputs"] = True
    bad = command(case, questions=[QuestionUpdate(operation="update", question_id="Q1",
                        evidence_refs=["foreign"], answer_state="partial")]).model_dump(mode="json")
    query = MemoryQuery(action="read_case_records", record_ids=["Q1"]).model_dump(mode="json")

    def ask(payload):
        return PlannerOutputContract(payload).encode(query, "memory_query")

    def submit(payload):
        return PlannerOutputContract(payload).encode(bad)

    def repair(payload):
        patch = repair_fields(evidence_refs=["input:smiles"])(payload)
        return PlannerOutputContract(payload).encode(patch)

    agent, client = planner(case, [ask, submit, ask, repair])
    case.runtime["planner_case_memory"] = True
    before = case.model_dump()
    store = RunStore(tmp_path, case.case_id)
    store.initialize()
    result = agent.assimilate(case, memory_store=store)
    assert len(client.calls) == 4
    assert client.calls[-1]["payload"]["memory_access"]["queries_remaining"] == 2
    assert result.questions[0].evidence_refs == ["input:smiles"]
    assert case.model_dump() == before
    logs = [json.loads(line) for line in (store.root / "history/memory_queries.jsonl").read_text().splitlines()]
    assert [e["operation"] for e in logs if e["kind"] == "memory_model_call"] == [
        "read_case_records", "decision", "read_case_records", "repair_judgment",
    ]
