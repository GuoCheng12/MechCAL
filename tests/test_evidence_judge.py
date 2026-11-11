"""Evidence rubric and the Layer B judge validator, with scripted judge responses (no API)."""

import json

import pytest

from mechcal.eval import evidence_judge as ej
from mechcal.eval.submission import (
    EvaluatorArtifact, ArtifactStore, EvidenceClaim, Prediction, Quantity, QuantitySource,
    StructuralAssertion,
)
from mechcal.eval.grounding import check_prediction

SMILES = "CN(C)c1ccc(cc1)C#N"


@pytest.fixture(scope="module")
def rubric():
    return ej.load_rubric()


@pytest.fixture
def store():
    s = ArtifactStore(case_id="CASE")
    s.add(EvaluatorArtifact(artifact_id="HE1", case_id="CASE", tool="microscopic.analyze_hole_electron_from_bundle",
                           status="success", molecule_smiles=SMILES,
                           outputs={"states": [{"root": 1, "d_index_angstrom": 3.12, "f": 0.42}]}))
    return s


def ict_prediction(*extra):
    return Prediction(label="ICT_TICT_CT", rank=1, limitations=["vertical only"], evidence=[
        EvidenceClaim(statement="S1 moves charge 3.12 Å from the NMe2 donor to the nitrile.", kind="computed",
                      quantities=[Quantity(name="D_index", value=3.12, unit="Å",
                                           source=QuantitySource(artifact_id="HE1", pointer="/states/0/d_index_angstrom"))]),
        EvidenceClaim(statement="Donor N2 is conjugated to nitrile C10 through the ring.", kind="structural",
                      atom_ids=[2, 10], assertions=[StructuralAssertion(type="conjugated_path", atoms=[2, 10])]),
        EvidenceClaim(statement="Push-pull benzonitriles are classic CT emitters.", kind="prior"),
        *extra,
    ])


def response(score, kind, route, conditions, verifications):
    return {"judgement": {
        "evidence_support_score": score, "selected_route_kind": kind, "selected_route_id": route,
        "verifications": [{"claim_id": c, "outcome": o, "explanation": "checked"} for c, o in verifications],
        "conditions": [{"condition_index": i + 1, "satisfied": True, "claim_ids": ids, "explanation": "meets"}
                       for i, ids in enumerate(conditions)],
        "overstatement": "none", "rationale": "because"}}


def validate(raw, pred, store, rubric):
    check = check_prediction(pred, store, SMILES)
    judgement, used = ej.validate_judgement(raw, pred, check, rubric)
    return ej.final_score(judgement, check, used), used


ALL_OK = [("C1", "consistent"), ("C2", "consistent"), ("C3", "consistent")]


def test_strong_route_on_grounded_computed_claim(rubric, store):
    raw = response(1.0, "strong_support", "excited_state_charge_transfer", [["C1"], ["C1", "C3"]], ALL_OK)
    assert validate(raw, ict_prediction(), store, rubric) == (1.0, [0])


def test_structural_partial_route(rubric, store):
    raw = response(0.5, "partial_support", "specific_structural_charge_transfer", [["C2"], ["C2", "C3"]], ALL_OK)
    assert validate(raw, ict_prediction(), store, rubric) == (0.5, [1])


@pytest.mark.parametrize("conditions,verifications,message", [
    ([["C2"], ["C2"]], ALL_OK, "needs an eligible"),            # structural cannot satisfy computed-only
    ([["C3"], ["C3"]], ALL_OK, "needs an eligible"),            # prior cannot be a finding
    ([["C1"], ["C1"]], [("C1", "contradicted")], "not verified as consistent"),
    ([["C9"], ["C1"]], ALL_OK, "existing claims"),
])
def test_strong_route_violations(rubric, store, conditions, verifications, message):
    raw = response(1.0, "strong_support", "excited_state_charge_transfer", conditions, verifications)
    with pytest.raises(ValueError, match=message):
        validate(raw, ict_prediction(), store, rubric)


def test_fabricated_claim_cannot_carry_a_route(rubric, store):
    fake = EvidenceClaim(statement="S1 has D = 5.00 Å.", kind="computed", quantities=[Quantity(
        name="D_index", value=5.0, unit="Å", source=QuantitySource(artifact_id="HE1", pointer="/states/0/d_index_angstrom"))])
    pred = ict_prediction(fake)
    raw = response(1.0, "strong_support", "excited_state_charge_transfer", [["C4"], ["C4"]],
                   ALL_OK + [("C4", "consistent")])
    with pytest.raises(ValueError):
        validate(raw, pred, store, rubric)
    # A route resting on the genuine claim still scores; the unused fabrication does not cap it.
    ok = response(1.0, "strong_support", "excited_state_charge_transfer", [["C1"], ["C1"]], ALL_OK)
    assert validate(ok, pred, store, rubric)[0] == 1.0


def test_family_without_structural_route_rejects_structure_only(rubric, store):
    pred = Prediction(label="RADIATIVE_RATE_STATE_BALANCE", rank=1, evidence=[
        EvidenceClaim(statement="The planar push-pull core should be bright.", kind="structural", atom_ids=[4])])
    raw = response(0.5, "partial_support", "specific_state_or_rate_proxy", [["C1"], ["C1"]], [("C1", "consistent")])
    with pytest.raises(ValueError, match="needs an eligible"):
        validate(raw, pred, store, rubric)


def test_zero_and_route_shape(rubric, store):
    zero = response(0.0, "none", None, [], [("C1", "contradicted")])
    assert validate(zero, ict_prediction(), store, rubric) == (0.0, [])
    bad = response(0.5, "strong_support", "excited_state_charge_transfer", [["C1"], ["C1"]], ALL_OK)
    with pytest.raises(ValueError, match="route kind"):
        validate(bad, ict_prediction(), store, rubric)
    short = response(1.0, "strong_support", "excited_state_charge_transfer", [["C1"]], ALL_OK)
    with pytest.raises(ValueError, match="every condition"):
        validate(short, ict_prediction(), store, rubric)


def test_rubric_annotations_and_ceilings(rubric):
    from mechcal.mechanism_pool import MECHANISM_POOL

    assert rubric["release"] == "0.3.0" and set(rubric["labels"]) == set(MECHANISM_POOL)
    kinds = set(rubric["evidence_kinds"]) - {"prior"}
    for label, lab in rubric["labels"].items():
        for kind in ("strong_support", "partial_support"):
            for route in lab[kind]["any_of"]:
                for cond in route["all_of"]:
                    assert cond["text"] and cond["accepts"] and set(cond["accepts"]) <= kinds, (label, route["id"])
        assert lab["ceilings"]["structure_only"] <= lab["ceilings"]["closed_book_with_computation"]
    assert rubric["labels"]["RADIATIVE_RATE_STATE_BALANCE"]["ceilings"]["structure_only"] == 0.0
    assert rubric["labels"]["ICT_TICT_CT"]["ceilings"]["structure_only"] == 0.5


def test_judge_never_sees_tool_names_or_ceilings(rubric, store):
    pred = ict_prediction()
    payload = ej.judge_payload(pred, check_prediction(pred, store, SMILES), store, SMILES, rubric)
    text = json.dumps(payload)
    assert "microscopic." not in text and "ceilings" not in text and "evaluator_coverage_notes" not in text
    cited = payload["claims"][0]["cited_values"][0]
    assert cited["artifact_value"] == 3.12 and cited["artifact_context"]["root"] == 1
    assert payload["claims"][0]["grounding_outcome"] == "grounded"


def test_response_schema_has_one_variant_per_route_plus_zero(rubric):
    schema = ej.response_schema("ICT_TICT_CT", rubric)
    lab = rubric["labels"]["ICT_TICT_CT"]
    n_routes = sum(len(lab[k]["any_of"]) for k in ("strong_support", "partial_support"))
    assert len(schema["properties"]["judgement"]["anyOf"]) == n_routes + 1


class ScriptedClient:
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def complete_json(self, system, payload, schema):
        self.calls.append(payload)
        return self.answers.pop(0)


def test_judge_repairs_once_then_scores(store):
    bad = response(1.0, "strong_support", "excited_state_charge_transfer", [["C2"], ["C2"]], ALL_OK)
    good = response(1.0, "strong_support", "excited_state_charge_transfer", [["C1"], ["C1"]], ALL_OK)
    client = ScriptedClient(bad, good)
    out = ej.EvidenceJudge(client).judge(ict_prediction(), store, SMILES)
    assert out["status"] == "scored" and out["score"] == 1.0 and out["attempts"] == 2
    assert "needs an eligible" in client.calls[1]["schema_feedback"]


def test_judge_reports_invalid_after_repairs_and_skips_empty(store):
    bad = response(1.0, "strong_support", "excited_state_charge_transfer", [["C2"], ["C2"]], ALL_OK)
    out = ej.EvidenceJudge(ScriptedClient(bad, bad)).judge(ict_prediction(), store, SMILES)
    assert out["status"] == "invalid_judgement" and out["score"] is None
    empty = Prediction(label="PET_ET", rank=1, evidence=[])
    assert ej.EvidenceJudge(ScriptedClient()).judge(empty, store, SMILES)["score"] == 0.0


def test_freeze_manifest_matches_committed_files():
    frozen = json.loads(ej.FREEZE_PATH.read_text())
    current = ej.freeze_manifest()
    assert {k: frozen[k] for k in current} == current, (
        "Evidence rubric, prompt, schema or grounding changed without updating the freeze manifest")
