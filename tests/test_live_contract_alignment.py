"""Contract congruence only: no external LLMs, chemistry, or score assertions."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime.decision_economy import validate_material_outcomes
from mechcal.runtime.schema_diagnostics import schema_issues
from mechcal.runtime.scoped_contracts import final_delivery_schema, result_review_schema, stop_review_schema, worker_schema
from mechcal.runtime.structured_output import PlannerOutputContract, provider_response_format
from mechcal.schemas.decisions import StoppingAssessment, StopReview
from mechcal.schemas.evidence_comparison import OutcomeExpectation
from mechcal.schemas.final_delivery import FinalDeliveryPatch
from mechcal.schemas.memory import MemoryQuery
from mechcal.schemas.observation_delivery import ObservableRequirement
from mechcal.schemas.question_loop import QuestionAction, ResultReview
from mechcal.schemas.tool_arguments import GeometryMeasurement, RelaxedScanArgs, supplied_argument_schema
from mechcal.schemas.worker_explanation import FindingAudit, WorkerExplanation
from test_question_loop import state
from test_worker_explanation import (ExplanationClient, case_with_explanation, fixture_result,
                                     review_payload)


def outcome_contract(**flags):
    return PlannerOutputContract({"output_contract": {
        "type": "object", "properties": {"outcome": {"$ref": "#/$defs/OutcomeExpectation"}},
        "required": ["outcome"], "additionalProperties": False,
        "$defs": {"OutcomeExpectation": OutcomeExpectation.model_json_schema()},
    }, "decision_economy": True, "decision_value_checks": True, **flags})


@pytest.mark.parametrize("effect", ["ranking", "support"])
@pytest.mark.parametrize("change", ["promote", "demote", "strengthen", "weaken", None])
def test_live_schema_wire_provider_and_runtime_agree_on_outcome_pairs(effect, change):
    label = MECHANISM_POOL[0]
    outcome = OutcomeExpectation(observation="Fixture observation.", judgment_effect="Fixture consequence.",
                                 decision_effect=effect, judgment_change=change, affected_labels=[label])
    valid = change in ({"promote", "demote"} if effect == "ranking" else {"strengthen", "weaken"})
    contract = outcome_contract()
    logical = {"outcome": outcome.model_dump(mode="json")}
    assert Draft202012Validator(contract.logical["decision"]).is_valid(logical) == valid
    baseline = outcome.model_copy(update={"judgment_change": "promote" if effect == "ranking" else "strengthen"})
    wire = contract.encode({"outcome": baseline.model_dump(mode="json")})
    wire["response"]["data"]["outcome"]["judgment_change"] = change
    provider = provider_response_format(contract.response_format(), "https://api.deepseek.com")
    for schema in (contract.schema, provider["json_schema"]["schema"]):
        assert Draft202012Validator(schema).is_valid(wire) == valid
    if valid:
        validate_material_outcomes([outcome], [label], ["test"])
        assert contract.decode(wire) == logical
    else:
        with pytest.raises(ValueError):
            validate_material_outcomes([outcome], [label], ["test"])
        error = next(Draft202012Validator(contract.schema).iter_errors(wire))
        issues = schema_issues(error)
        assert any(i["path"] == ["response", "data", "outcome", "judgment_change"] for i in issues)


@pytest.mark.parametrize("flags", [dict(decision_economy=False), dict(decision_value_checks=False)])
def test_historical_outcomes_remain_loadable_when_economy_rules_are_disabled(flags):
    old = OutcomeExpectation(observation="Fixture.", judgment_effect="Fixture.",
                             decision_effect="ranking", judgment_change="strengthen")
    contract = outcome_contract(**flags)
    assert contract.decode(contract.encode({"outcome": old.model_dump(mode="json")}))["outcome"] == old.model_dump(mode="json")


@pytest.mark.parametrize("model,data,valid", [
    (StoppingAssessment, dict(judgment_state="unresolved", finish_basis="judgment_sufficient", reason="Unresolved."), False),
    (StoppingAssessment, dict(judgment_state="unresolved", finish_basis="no_useful_available_step", reason="No useful route."), True),
    (StoppingAssessment, dict(judgment_state="adequate_for_bounded_ranking", finish_basis="judgment_sufficient", reason="Bounded answer."), True),
    (FindingAudit, dict(finding_id="F1", status="correction_required", reason="Check.", correction="  "), False),
    (FindingAudit, dict(finding_id="F1", status="correction_required", reason="Check.", correction="Restrict the claim."), True),
    (FindingAudit, dict(finding_id="F1", status="faithful", reason="Check.", correction=""), True),
    (ObservableRequirement, dict(path="/x", value_type="number", unit_path=None, unit="eV"), False),
    (ObservableRequirement, dict(path="/x", value_type="number", unit_path="/unit", unit=None), False),
    (ObservableRequirement, dict(path="/x", value_type="number", unit_path="/unit", unit="eV"), True),
    (ObservableRequirement, dict(path="/x", value_type="number", unit_path=None, unit=None), True),
])
def test_conditional_field_contracts_match_model_validation(model, data, valid):
    schema = model.model_json_schema()
    Draft202012Validator.check_schema(schema)
    assert Draft202012Validator(schema).is_valid(data) == valid
    if valid:
        saved = model.model_validate(data)
        contract = PlannerOutputContract({"output_contract": schema})
        assert contract.decode(contract.encode(saved.model_dump(mode="json"))) == saved.model_dump(mode="json")
    else:
        with pytest.raises(ValidationError):
            model.model_validate(data)


@pytest.mark.parametrize("kind,count", [("distance", 2), ("angle", 3), ("dihedral", 4)])
@pytest.mark.parametrize("size", [2, 3, 4])
def test_geometry_measurement_card_and_validator_agree(kind, count, size):
    data = dict(measurement_id="m1", kind=kind, atom_indices=list(range(size)))
    schema = supplied_argument_schema(GeometryMeasurement)
    assert "properties" in schema and schema["additionalProperties"] is False
    assert Draft202012Validator(schema).is_valid(data) == (size == count)
    if size == count:
        GeometryMeasurement.model_validate(data)
    else:
        with pytest.raises(ValidationError):
            GeometryMeasurement.model_validate(data)


@pytest.mark.parametrize("coordinate,indices,values,valid", [
    ("bond", [0, 1], [1.0, 1.5], True),
    ("bond", [0, 1], [1.0, 15], False),
    ("bond", [0, 0], [1.0, 1.5], False),
    ("bond", [0, 1, 2], [1.0, 1.5], False),
    ("dihedral", [0, 1, 2, 3], [-90, 90], True),
    ("dihedral", [0, 1], [-90, 90], False),
    ("dihedral", [0, 1, 2, 3], [-90, 190], False),
])
def test_relaxed_scan_card_and_validator_agree(coordinate, indices, values, valid):
    data = dict(coordinate=coordinate, atom_indices=indices, values=values)
    assert Draft202012Validator(supplied_argument_schema(RelaxedScanArgs)).is_valid(data) == valid
    if valid:
        RelaxedScanArgs.model_validate(data)
    else:
        with pytest.raises(ValidationError):
            RelaxedScanArgs.model_validate(data)


@pytest.mark.parametrize("action,ids,capability,valid", [
    ("read_case_records", [], None, False),
    ("read_case_records", ["E1"], None, True),
    ("read_case_records", ["E1", "E1"], None, False),
    ("find_compatible_artifacts", [], None, False),
    ("find_compatible_artifacts", [], "macro.example", True),
    ("search_case_records", [], None, True),
])
def test_memory_action_requirements_match_schema(action, ids, capability, valid):
    data = dict(action=action, record_ids=ids, capability_id=capability)
    assert Draft202012Validator(MemoryQuery.model_json_schema()).is_valid(data) == valid
    if valid:
        query = MemoryQuery.model_validate(data)
        contract = PlannerOutputContract({"output_contract": MemoryQuery.model_json_schema()})
        assert contract.decode(contract.encode(query.model_dump(mode="json"))) == query.model_dump(mode="json")
    else:
        with pytest.raises(ValidationError):
            MemoryQuery.model_validate(data)


def test_worker_contract_is_scoped_to_request_and_failed_outputs_cannot_invent_findings():
    request, result = fixture_result()
    client = ExplanationClient()
    data = client.complete_json(payload={"stage": "worker_result_explanation", "status": result.status,
        "dispatch_request": request.model_dump(mode="json"), "tool_execution_result": result.model_dump(mode="json")},
        system_prompt="Offline fixture.")
    schema = worker_schema(WorkerExplanation.model_json_schema(), request, result)
    validator = Draft202012Validator(schema)
    assert validator.is_valid(data)
    for field, bad in [("question_ids", ["FOREIGN"]), ("evidence_refs", ["FOREIGN"])]:
        changed = deepcopy(data)
        target = changed if field == "question_ids" else changed["findings"][0]
        target[field] = bad
        assert not validator.is_valid(changed)
    failed = result.model_copy(update={"status": "failed"})
    assert not Draft202012Validator(worker_schema(WorkerExplanation.model_json_schema(), request, failed)).is_valid(data)
    assert result.status == "success"


@pytest.mark.parametrize("fault", ["question", "evidence", "finding", "missing", "unavailable"])
def test_result_review_publishes_exact_supplied_scope(fault):
    case = case_with_explanation()
    before = case.model_dump_json()
    raw = review_payload()
    schema = result_review_schema(ResultReview.model_json_schema(), state(case)["pending_results"])
    validator = Draft202012Validator(schema)
    validator.validate(raw)
    if fault == "question":
        raw["interpretations"][0]["question_ids"] = ["FOREIGN"]
    elif fault == "evidence":
        raw["interpretations"][0]["evidence_refs"] = ["FOREIGN"]
    elif fault == "finding":
        raw["interpretations"][0]["finding_audits"][0]["finding_id"] = "FOREIGN"
    elif fault == "missing":
        raw["interpretations"] = []
    else:
        raw["status"] = "unavailable"
    assert not validator.is_valid(raw)
    assert before == case.model_dump_json()


def test_schema_diagnostic_does_not_guess_between_ambiguous_branches():
    schema = {"anyOf": [
        {"type": "object", "properties": {"x": {"type": "string", "enum": ["a", "b"]}}},
        {"type": "object", "properties": {"x": {"type": "string", "enum": ["b", "c"]}}},
    ]}
    error = next(Draft202012Validator(schema).iter_errors({"x": "d"}))
    assert schema_issues(error)[0]["path"] == []


@pytest.mark.parametrize("cards", [[], [{"capability_id": "micro.disabled", "status": "disabled"}]])
def test_stop_schema_does_not_offer_impossible_continuation_when_no_tools_enabled(cards):
    from test_decision_value import prepared

    case, _ = prepared()
    schema = stop_review_schema(StopReview.model_json_schema(), case, {"capability_cards": cards})
    assert len(schema["anyOf"]) == 1
    contract = PlannerOutputContract({"output_contract": schema})
    review = StopReview(rationale="No available continuation.").model_dump(mode="json")
    Draft202012Validator(schema).validate(review)
    assert contract.decode(contract.encode(review)) == review
    review["capability_ids"] = ["micro.disabled"]
    assert not Draft202012Validator(schema).is_valid(review)


def test_schema_diagnostic_supports_array_discriminators_in_correction_contracts():
    schema = {"anyOf": [
        {"type": "object", "properties": {"path": {"const": ["a"]}, "value": {"type": "number"}}},
        {"type": "object", "properties": {"path": {"const": ["b"]}, "value": {"type": "boolean"}}},
    ]}
    error = next(Draft202012Validator(schema).iter_errors({"path": ["a"], "value": "not a number"}))
    assert schema_issues(error)[0]["path"] == ["value"]


@pytest.mark.parametrize("fault", ["withdrawal", "no_evidence_label", "whitespace", "argument_ref"])
def test_final_delivery_schema_rejects_known_bad_scope_before_materialization(fault):
    from mechcal.runtime.final_delivery import apply_patch, open_draft
    from test_final_delivery_v2 import completion, fixture

    case = fixture()
    draft = open_draft(case)
    before = (case.model_dump_json(), draft.model_dump_json())
    data = completion(case, draft).model_dump(mode="json")
    schema = final_delivery_schema(FinalDeliveryPatch.model_json_schema(), case, draft)
    Draft202012Validator(schema).validate(data)
    if fault == "withdrawal":
        data["withdrawn_support_ids"] = ["FOREIGN"]
    elif fault == "no_evidence_label":
        data["no_evidence"] = [dict(label="FOREIGN", reason="No argument.")]
    elif fault == "whitespace":
        data["final_answer_draft"] = "  "
    else:
        data["arguments"] = [draft.arguments[0].model_dump(mode="json")]
        data["arguments"][0]["observation_refs"] = ["FOREIGN"]
    assert not Draft202012Validator(schema).is_valid(data)
    with pytest.raises(ValueError):
        apply_patch(case, draft, FinalDeliveryPatch.model_validate(data))
    assert before == (case.model_dump_json(), draft.model_dump_json())




def test_optional_worker_planning_publishes_omission_not_null_for_tool_parameters():
    from mechcal.agents.execution_planner import WorkerExecutionPlanner
    from test_question_loop import REGISTRY, initial_case

    class Capture(Exception):
        pass

    class Client:
        def is_configured(self):
            return True

        def complete_json(self, **kwargs):
            self.payload = kwargs["payload"]
            raise Capture()

    request, _ = fixture_result("microscopic.run_baseline_bundle")
    client = Client()
    planner = WorkerExecutionPlanner(capability_registry=REGISTRY, llm_client=client, enable_llm=True)
    with pytest.raises(Capture):
        planner.plan(agent_name=request.agent_name, request=request, case_run=initial_case())
    schema = client.payload["capability_card"]["tool_arg_schema"]
    validator = Draft202012Validator(schema)
    assert validator.is_valid({})
    assert not validator.is_valid({"source_artifact_id": None})
