"""R002 regression: enum corrections must not regenerate scientific judgments."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from mechcal.agents.planner_references import PlannerReferences
from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime.evidence_comparison import mechanism_effect_issues, validate_update
from mechcal.runtime.judgment_commands import materialize_judgment, snapshot_feedback
from mechcal.runtime.judgment_draft import JudgmentDraft
from mechcal.runtime.llm import _parse_json_object
from mechcal.runtime.result_contract import disposition_contract
from mechcal.runtime.retry import RejectedResponseError, RequestValidationError
from mechcal.runtime.structured_output import PlannerOutputContract, provider_response_format
from mechcal.schemas.evidence_comparison import RankingUpdate
from mechcal.schemas.judgment_commands import JudgmentCommand, RankingRationale
from test_incremental_judgment import prepared
from test_judgment_commands import command, planner, repair_fields


@pytest.mark.parametrize("model", [RankingUpdate, RankingRationale])
def test_alternative_labels_have_same_closed_schema_in_full_and_patch(model):
    prop = model.model_json_schema()["properties"]["alternatives_considered"]
    assert Draft202012Validator(prop).is_valid(list(MECHANISM_POOL))
    assert not Draft202012Validator(prop).is_valid(["ICT_TICT_CT because it looks plausible"])
    assert prop["items"]["enum"] == list(MECHANISM_POOL)


def test_semantic_label_error_points_to_leaf_without_scientific_context():
    case = prepared()
    update = materialize_judgment(case, command(case))
    update.ranking_update.alternatives_considered = ["ICT_TICT_CT needs evidence"]
    candidate = command(case, ranking_update=update.ranking_update).model_dump(mode="json")
    with pytest.raises(RequestValidationError) as caught:
        validate_update(case, update)
    enriched = snapshot_feedback(caught.value, update)
    draft = JudgmentDraft(case.case_id, 2, "assimilate", JudgmentCommand, {
        "error": str(enriched), "issues": enriched.issues, "rejected_response": candidate,
    })
    assert [t["path"] for t in draft.targets.values()] == [["ranking_update", "alternatives_considered"]]
    assert not draft.requires_context
    fixed = draft.apply(repair_fields(alternatives_considered=["ICT_TICT_CT"])(draft.payload()))
    expected = deepcopy(candidate)
    expected["ranking_update"]["alternatives_considered"] = ["ICT_TICT_CT"]
    assert fixed == expected


def test_invalid_enum_wire_uses_small_typed_edits_and_preserves_other_values():
    case = prepared()
    refs = PlannerReferences(case)
    update = materialize_judgment(case, command(case))
    good = command(case, ranking_update=update.ranking_update).model_dump(mode="json")
    good["ranking_update"]["alternatives_considered"] = ["ICT_TICT_CT", "PET_ET"]
    api = PlannerOutputContract({"output_contract": JudgmentCommand.model_json_schema()})
    wire = api.encode(good)
    bad = deepcopy(wire)
    bad["response"]["data"]["ranking_update"]["alternatives_considered"] = ["ICT explanation", "PET explanation"]
    with pytest.raises(RejectedResponseError) as caught:
        refs.decode_response(api, bad)
    err = caught.value
    assert refs.rejected_contract(err.response) is api
    assert PlannerReferences(case).rejected_contract(err.response) is None
    draft = JudgmentDraft(case.case_id, 2, "assimilate", JudgmentCommand, {
        "error": str(err), "issues": err.issues, "rejected_response": json.loads(err.response),
    }, schema=api.schema)
    assert not draft.requires_context
    assert [t["path"] for t in draft.targets.values()] == [
        ["response", "data", "ranking_update", "alternatives_considered", i] for i in (0, 1)]
    patch = {"action": "repair_judgment", "draft_id": draft.draft_id, "edits": [
        {"target_id": tid, "operation": "replace", "value": good["ranking_update"]["alternatives_considered"][t["field"]]}
        for tid, t in draft.targets.items()]}
    repair_api = PlannerOutputContract(draft.payload())
    expanded = provider_response_format(repair_api.response_format(), "https://api.deepseek.com")
    Draft202012Validator(expanded["json_schema"]["schema"]).validate(repair_api.encode(patch))
    fixed = draft.apply(repair_api.decode(repair_api.encode(patch)))
    assert fixed == wire and api.decode(fixed) == good
    assert len(json.dumps(patch)) < len(json.dumps(wire))


@pytest.mark.parametrize("bad_correction", [False, True])
def test_live_judgment_flow_repairs_wire_field_within_original_retry_budget(bad_correction):
    case = prepared()
    case.runtime["planner_strict_outputs"] = True
    good = command(case).model_dump(mode="json")

    def bad_reply(payload):
        raw = PlannerOutputContract(payload).encode(good)
        raw["response"]["data"]["scope"] = "free prose instead of enum"
        return raw

    def fix(payload):
        assert payload["stage"] == "planner_judgment_repair"
        assert "scientific_context" not in payload and "candidate_command" not in payload
        assert [t["field"] for t in payload["targets"]] == ["scope"]
        return PlannerOutputContract(payload).encode(repair_fields(scope="global")(payload))

    def invalid_fix(payload):
        raw = fix(payload)
        raw["response"]["data"]["edits"][0]["value"] = "still invalid"
        return raw

    responses = [bad_reply, invalid_fix, fix] if bad_correction else [bad_reply, fix]
    agent, client = planner(case, responses)
    before = case.model_dump_json()
    result = agent.assimilate(case)
    assert result.portfolio == case.portfolio
    assert len(client.calls) == (3 if bad_correction else 2)
    assert case.model_dump_json() == before


def test_all_missing_execution_mechanisms_are_reported_together():
    case = prepared()
    results = [{"result_id": f"X{i}", "evidence_refs": [],
                "investigation_plan": {"candidate_labels": ["ICT_TICT_CT", "PET_ET"]}} for i in (1, 2, 3)]
    case.runtime["question_loop"]["pending_results"] = results
    effect = SimpleNamespace(label="ICT_TICT_CT", reason="No new observation.", evidence_refs=[], effect="unavailable")
    update = SimpleNamespace(dispositions=[SimpleNamespace(result_id=r["result_id"], mechanism_effects=[effect]) for r in results])
    issues = mechanism_effect_issues(case, update, set())
    assert len(issues) == 3
    for i, issue in enumerate(issues):
        assert issue["path"] == ["dispositions", i, "mechanism_effects"]
        assert issue["missing_labels"] == ["PET_ET"]
        assert disposition_contract(results[i], case)["required_mechanism_labels"] == ["ICT_TICT_CT", "PET_ET"]


@pytest.mark.parametrize("raw", ['{"response":{"kind":"decision"}', '{"a":1]}'])
def test_malformed_json_is_not_silently_patched_or_accepted(raw):
    with pytest.raises(ValueError):
        _parse_json_object(raw)
