"""Citation transport and mixed-error repairs, without model or chemistry calls."""

from copy import deepcopy
import json

import pytest

from mechcal.agents.planner_references import PlannerReferences
from mechcal.runtime.field_repair import FieldRepairProducer
from mechcal.runtime.questions import replace_state, validate_assimilation
from mechcal.runtime.retry import RequestValidationError
from mechcal.schemas.question_loop import QuestionAssimilation
from test_evidence_references import citation_case
from test_field_repair import missing_receipt, runner

FIRST = "execution:bcf69493e86ba6359729"
SECOND = "execution:6a3d0e3e8e4ed9b0273c"
MIXED = "execution:bcf69493e8e4ed9b0273c"


def refs_case():
    return replace_state(citation_case([FIRST + ":E001", SECOND + ":E001", "E001"]),
                         results=[{"result_id": FIRST}], pending_results=[{"result_id": SECOND}])


def test_roundtrip_collision_and_case_isolation():
    case = refs_case()
    before = case.model_dump(mode="json")
    refs = PlannerReferences(case)
    value = {"evidence_refs": [FIRST + ":E001", "E001"], "result_ids": [FIRST, SECOND],
             "text": f"Compare {FIRST}:E001 and {SECOND}.", "artifact": "artifact:E002"}
    encoded = refs.encode(value)
    assert encoded["evidence_refs"] == ["E002", "E001"]
    assert encoded["result_ids"] == ["X001", "X002"]
    assert refs.restore(encoded) == value
    assert refs.restore({"evidence_refs": [MIXED + ":E001", "E999"]}) == {
        "evidence_refs": [MIXED + ":E001", "E999"]}
    assert PlannerReferences(citation_case([])).aliases == {}
    assert case.model_dump(mode="json") == before


def test_payload_feedback_and_field_repair_values_share_aliases():
    refs = PlannerReferences(refs_case())

    class Client:
        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            assert payload["execution_history"] == [{"result_id": "X002", "evidence_refs": ["E003"]}]
            assert json.loads(schema_feedback)["allowed_refs"] == ["E003"]
            return {"repairs": [{"path": ["update", "questions", 0, "result_ids"], "value": ["X002"]},
                                {"path": ["update", "questions", 0, "evidence_refs"], "value": ["E003"]}]}

    response = refs.complete(Client(), system_prompt="fixture", payload={
        "execution_history": [{"result_id": SECOND, "evidence_refs": [SECOND + ":E001"]}]},
        schema_feedback=json.dumps({"allowed_refs": [SECOND + ":E001"]}))
    assert response["repairs"][0]["value"] == [SECOND]
    assert response["repairs"][1]["value"] == [SECOND + ":E001"]


@pytest.mark.parametrize("partial", [False, True])
def test_mixed_questions_receipts_and_references_stay_field_scoped(partial):
    case, data, receipt = missing_receipt()
    data["questions"][0]["assessment"] = " "
    data["questions"][0]["related_labels"] = ["not_a_mechanism"]
    data["questions"][0]["result_ids"] = [MIXED]
    data["questions"][0]["evidence_refs"] = [MIXED + ":E001"]
    before = deepcopy(data)
    calls = []

    def full(feedback):
        calls.append("full")
        return deepcopy(data)

    def repair(payload, feedback):
        calls.append("repair")
        values = {("argument_changes",): [receipt], ("questions", 0, "assessment"): "Still an open question.",
                  ("questions", 0, "related_labels"): ["RIM_RIR_RIV"],
                  ("questions", 0, "result_ids"): [], ("questions", 0, "evidence_refs"): ["input:smiles"]}
        paths = payload["allowed_paths"]
        if partial and calls.count("repair") == 1:
            paths = [p for p in paths if p[-1] != "related_labels"]
        return {"repairs": [{"path": p, "value": values[tuple(p)]} for p in paths]}

    result = runner(case).run(FieldRepairProducer(full, repair, response_model=QuestionAssimilation))
    assert calls == ["full", "repair"] + (["repair"] if partial else [])
    assert result.portfolio == QuestionAssimilation.model_validate(before).portfolio
    assert result.mechanism_support_arguments == QuestionAssimilation.model_validate(before).mechanism_support_arguments
    assert data == before


def test_unknown_alias_remains_invalid_not_dropped():
    case, data, receipt = missing_receipt()
    data["argument_changes"] = [receipt]
    data["questions"][0]["evidence_refs"] = ["E999"]
    with pytest.raises(RequestValidationError) as raised:
        validate_assimilation(QuestionAssimilation.model_validate(PlannerReferences(case).restore(data)), case)
    assert raised.value.issues[0]["invalid_refs"] == ["E999"]
