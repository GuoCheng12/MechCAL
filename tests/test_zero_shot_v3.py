import copy
import json

import pytest
from jsonschema import Draft202012Validator
from mechcal.eval.zero_shot_v3 import (
    RUBRIC_PATH,
    EvidenceSubmission,
    Prediction,
    Predictions,
    ZeroShotV3Judge,
    empty_evidence_judgement,
    judge_payload,
    judge_response_schema,
    structure_facts,
    validate_judgement,
)
from pydantic import ValidationError


@pytest.fixture
def rubric():
    return json.loads(RUBRIC_PATH.read_text())


@pytest.fixture
def prediction():
    return Prediction(
        label="RIM_RIR_RIV",
        rank=1,
        confidence=0.4,
        claim_status="candidate_requires_validation",
        support_strength="weak_or_proxy",
        evidence=[
            "Finding: a phenyl group linked to an alkyl chain has a rotatable bond. "
            "Warrant: its motion can provide a nonradiative route."
        ],
        limitations=["This graph does not establish the route's activity."],
    )


def test_graph_is_connectivity_only():
    facts = structure_facts("CCc1ccccc1O")
    assert len(facts["atoms"]) == 9
    assert {a["id"] for a in facts["atoms"]} == set(range(1, 10))
    assert facts["atoms"][-1]["element"] == "O"
    assert facts["rotatable_bond_count"] == 1
    assert "geometry" not in facts
    assert "mapped_smiles" in facts
    with pytest.raises(ValueError):
        structure_facts("not-a-smiles")


def test_judge_receives_no_identity_or_reference(prediction, rubric):
    payload = judge_payload(prediction, structure_facts("CCc1ccccc1O"), rubric)
    text = json.dumps(payload)
    for field in ("hidden_reference", "reference_mechanisms", "case_id", "confidence", "rank"):
        assert field not in payload
        assert field not in payload["submitted_prediction"]
    assert "gpt-5.5" not in text
    assert payload["source_availability"]["subject_scientific_tool_calls"] == 0
    assert not payload["source_availability"]["tool_records"]


def judgement(prediction, rubric):
    route = rubric["labels"][prediction.label]["partial_support"]["any_of"][0]
    return {
        "status": "scored",
        "evidence_support_score": 0.5,
        "verifications": [
            {
                "submitted_quote": "a phenyl group linked to an alkyl chain",
                "outcome": "consistent",
                "atom_ids": [1, 2, 3],
                "explanation": "The graph has this connection.",
            }
        ],
        "selected_route_kind": "partial_support",
        "selected_route_id": route["id"],
        "conditions": [
            {
                "condition_index": index,
                "satisfied": True,
                "verification_indices": [1],
                "explanation": "Submitted finding supports it.",
            }
            for index in range(1, len(route["all_of"]) + 1)
        ],
        "overstatement": "none",
        "rationale": "Structure-specific proxy, not a dynamics result.",
    }


def test_positive_route_validation(prediction, rubric):
    raw = judgement(prediction, rubric)
    facts = structure_facts("CCc1ccccc1O")
    assert validate_judgement(raw, prediction, facts, rubric).evidence_support_score == 0.5
    invalid = copy.deepcopy(raw)
    invalid["conditions"][0]["satisfied"] = False
    with pytest.raises(ValueError, match="every selected route"):
        validate_judgement(invalid, prediction, facts, rubric)
    invalid = copy.deepcopy(raw)
    invalid["verifications"][0]["outcome"] = "beyond_source"
    with pytest.raises(ValueError, match="ineligible"):
        validate_judgement(invalid, prediction, facts, rubric)


def test_each_condition_requires_a_finding(prediction, rubric):
    raw = judgement(prediction, rubric)
    raw["conditions"][0]["verification_indices"] = []
    with pytest.raises(ValueError, match="must cite"):
        validate_judgement(raw, prediction, structure_facts("CCc1ccccc1O"), rubric)


def test_all_rubric_routes_are_representable(prediction, rubric):
    judge = ZeroShotV3Judge(None)
    facts = structure_facts("CCc1ccccc1O")
    for label, family in rubric["labels"].items():
        pred = prediction.model_copy(update={"label": label})
        schema = judge_response_schema(label, rubric)
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        assert schema["type"] == "object" and "anyOf" not in schema
        for kind, score in (("strong_support", 1.0), ("partial_support", 0.5)):
            for route in family[kind]["any_of"]:
                raw = judgement(pred, rubric)
                raw.update(evidence_support_score=score, selected_route_kind=kind,
                           selected_route_id=route["id"])
                condition = raw["conditions"][0]
                raw["conditions"] = [dict(condition, condition_index=i)
                                     for i in range(1, len(route["all_of"]) + 1)]
                response = {"judgement": raw}
                validator.validate(response)
                assert judge.validate_response(response, pred, facts).model_dump() == raw
        for status, score in (("scored", 0.0), ("pending", None)):
            raw = judgement(pred, rubric)
            raw.update(status=status, evidence_support_score=score,
                       selected_route_kind="none", selected_route_id=None, conditions=[])
            if status == "pending":
                raw["verifications"][0]["outcome"] = "unavailable"
            response = {"judgement": raw}
            validator.validate(response)
            assert judge.validate_response(response, pred, facts).model_dump() == raw


@pytest.mark.parametrize("change", [
    {"conditions": []},
    {"selected_route_kind": "none", "selected_route_id": None},
    {"selected_route_id": "invented_route"},
    {"evidence_support_score": 1.0},
    {"status": "pending"},
])
def test_wire_schema_rejects_inconsistent_positive_scores(prediction, rubric, change):
    raw = judgement(prediction, rubric)
    raw.update(change)
    assert not Draft202012Validator(judge_response_schema(prediction.label, rubric)).is_valid(
        {"judgement": raw}
    )


def test_wire_wrapper_does_not_bypass_semantic_validation(prediction, rubric):
    raw = judgement(prediction, rubric)
    raw["conditions"].reverse()
    with pytest.raises(ValueError, match="in order"):
        ZeroShotV3Judge(None).validate_response(
            {"judgement": raw}, prediction, structure_facts("CCc1ccccc1O")
        )
    with pytest.raises(ValidationError):
        ZeroShotV3Judge(None).validate_response(
            {"judgement": judgement(prediction, rubric), "extra": "bad"},
            prediction, structure_facts("CCc1ccccc1O"),
        )


def test_quote_and_atom_validation(prediction, rubric):
    raw = judgement(prediction, rubric)
    facts = structure_facts("CCc1ccccc1O")
    raw["verifications"][0]["atom_ids"] = [100]
    with pytest.raises(ValueError, match="outside"):
        validate_judgement(raw, prediction, facts, rubric)
    raw["verifications"][0]["atom_ids"] = [1]
    raw["verifications"][0]["submitted_quote"] = "an invented TDDFT result"
    with pytest.raises(ValueError, match="submitted evidence"):
        validate_judgement(raw, prediction, facts, rubric)


def test_pending_never_becomes_zero(prediction, rubric):
    raw = judgement(prediction, rubric)
    raw.update(
        status="pending",
        evidence_support_score=None,
        selected_route_kind="none",
        selected_route_id=None,
        conditions=[],
    )
    raw["verifications"][0]["outcome"] = "unavailable"
    facts = structure_facts("CCc1ccccc1O")
    assert validate_judgement(raw, prediction, facts, rubric).evidence_support_score is None
    raw["evidence_support_score"] = 0
    with pytest.raises(ValueError, match="Pending requires"):
        validate_judgement(raw, prediction, facts, rubric)


def test_duplicate_or_unordered_predictions_are_rejected(prediction, rubric):
    labels = list(rubric["labels"])[:3]
    rows = [
        prediction.model_copy(update={"label": label, "rank": rank}).model_dump()
        for rank, label in enumerate(labels, 1)
    ]
    assert (
        len(Predictions.model_validate({"mechanism_predictions": rows}).mechanism_predictions) == 3
    )
    rows[1]["label"] = rows[0]["label"]
    with pytest.raises(ValidationError):
        Predictions.model_validate({"mechanism_predictions": rows})


def test_explicit_empty_submission_does_not_relax_generation_contract(prediction):
    raw = prediction.model_dump() | {"evidence": []}
    submitted = EvidenceSubmission.model_validate(raw)
    assert submitted.evidence == []
    with pytest.raises(ValidationError):
        Prediction.model_validate(raw)
    assert Prediction.model_json_schema()["properties"]["evidence"]["minItems"] == 1


@pytest.mark.parametrize("evidence", [None, "", [""], [" \n"], [None], [42]])
def test_malformed_evidence_is_not_an_empty_submission(prediction, evidence):
    with pytest.raises(ValidationError):
        EvidenceSubmission.model_validate(prediction.model_dump() | {"evidence": evidence})


def test_missing_evidence_is_not_an_empty_submission(prediction):
    raw = prediction.model_dump()
    del raw["evidence"]
    with pytest.raises(ValidationError):
        EvidenceSubmission.model_validate(raw)


def test_empty_submission_has_explicit_zero_without_invented_quotes(prediction, rubric):
    submitted = EvidenceSubmission.model_validate(prediction.model_dump() | {"evidence": []})
    result = empty_evidence_judgement(submitted)
    assert result.evidence_support_score == 0
    assert result.status == "scored"
    assert result.verifications == [] and result.conditions == []
    assert "deterministic" in result.rationale
    assert validate_judgement(result.model_dump(), submitted, {}, rubric) == result
    assert submitted.evidence == []
    # A normal LLM response still has to quote submitted evidence.
    assert not Draft202012Validator(judge_response_schema(submitted.label, rubric)).is_valid(
        {"judgement": result.model_dump()}
    )


@pytest.mark.parametrize("change", [
    {"evidence_support_score": 0.5},
    {"status": "pending", "evidence_support_score": None},
    {"selected_route_kind": "partial_support"},
    {"selected_route_id": "invented_route"},
])
def test_empty_submission_cannot_get_positive_or_pending_score(prediction, rubric, change):
    submitted = EvidenceSubmission.model_validate(prediction.model_dump() | {"evidence": []})
    raw = empty_evidence_judgement(submitted).model_dump() | change
    with pytest.raises(ValueError, match="Empty submitted evidence"):
        validate_judgement(raw, submitted, {}, rubric)


def test_nonempty_submission_always_needs_judge_verification(prediction, rubric):
    # Self-reported lack of support cannot short-circuit actual evidence review.
    prediction.support_strength = "unsupported"
    assert empty_evidence_judgement(prediction) is None
    raw = judgement(prediction, rubric)
    raw.update(evidence_support_score=0, selected_route_kind="none",
               selected_route_id=None, conditions=[], verifications=[])
    with pytest.raises(ValueError, match="Nonempty evidence requires"):
        validate_judgement(raw, prediction, structure_facts("CCc1ccccc1O"), rubric)


def test_empty_submission_cannot_bypass_mechanism_pool(prediction):
    submitted = EvidenceSubmission.model_validate(
        prediction.model_dump() | {"label": "unknown", "evidence": []}
    )
    with pytest.raises(ValueError, match="Unknown mechanism"):
        empty_evidence_judgement(submitted)
