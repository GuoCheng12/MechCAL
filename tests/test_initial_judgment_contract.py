"""Initial-pool and repair-location regressions; no model or scientific calls."""

from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime.boundary_decisions import validate_ranking
from mechcal.runtime.judgment_commands import materialize_judgment, snapshot_feedback
from mechcal.runtime.judgment_draft import JudgmentDraft
from mechcal.runtime.questions import apply_assimilation, replace_state, validate_assimilation
from mechcal.runtime.retry import RequestValidationError
from mechcal.schemas.judgment_commands import FinalJudgmentCommand, JudgmentCommand
from test_boundary_decisions import start
from test_judgment_commands import command, initial_command, planner, proposal_wire, repair_fields


def fixture():
    case, update = start()
    case.runtime["incremental_judgment"] = True
    return case, initial_command(update).model_dump(mode="json")


@pytest.mark.parametrize("final", [False, True])
def test_direct_materialization_locations_do_not_regenerate_whole_command(final):
    case, good = fixture()
    bad = deepcopy(good)
    for i in range(3, len(bad["hypotheses"])):
        bad["hypotheses"][i] = {"name": bad["hypotheses"][i]["name"]}
    before = case.model_dump()
    with pytest.raises(RequestValidationError) as error:
        materialize_judgment(case, JudgmentCommand.model_validate(bad))
    assert len(error.value.issues) == len(MECHANISM_POOL) - 3
    assert all("confidence" in issue["message"] for issue in error.value.issues)
    issues = error.value.issues
    raw, model = bad, JudgmentCommand
    if final:
        raw, model = {"update": bad, "final_action": {}}, FinalJudgmentCommand
        issues = [{**issue, "command_field": "update"} for issue in issues]
    draft = JudgmentDraft(case.case_id, 0, "assimilate", model, {
        "rejected_response": raw, "error": str(error.value), "issues": issues,
    })
    prefix = ["update"] if final else []
    assert [t["path"] for t in draft.targets.values()] == [
        [*prefix, "hypotheses", i] for i in range(3, len(MECHANISM_POOL))]
    edits = [{"target_id": key, "value": good["hypotheses"][t["path"][-1]]}
             for key, t in draft.targets.items()]
    repaired = draft.apply({"action": "repair_judgment", "draft_id": draft.draft_id, "edits": edits})
    assert (repaired["update"] if final else repaired) == good
    assert case.model_dump() == before


def test_string_display_path_uses_structured_repair_path():
    _, good = fixture()
    draft = JudgmentDraft("case", 0, "assimilate", JudgmentCommand, {
        "rejected_response": good, "error": "Missing new candidate fields.",
        "issues": [{"path": "hypotheses.3", "repair_paths": [["hypotheses", "3"]]}],
    })
    assert next(iter(draft.targets.values()))["path"] == ["hypotheses", 3]
    Draft202012Validator.check_schema(draft.payload()["output_contract"])


@pytest.mark.parametrize("final", [False, True])
def test_pool_and_ranking_gaps_reported_together_with_local_targets(final):
    case, good = fixture()
    bad = deepcopy(good)
    bad["hypotheses"] = bad["hypotheses"][:3]
    bad["ranking_update"]["candidate_bases"] = bad["ranking_update"]["candidate_bases"][:3]
    bad["ranking_update"]["boundary_comparisons"] = []
    update = materialize_judgment(case, JudgmentCommand.model_validate(bad))
    with pytest.raises(RequestValidationError) as error:
        validate_ranking(case, update)
    feedback = snapshot_feedback(error.value, update)
    assert len(feedback.issues) == 3
    raw, model = bad, JudgmentCommand
    if final:
        raw, model = {"update": bad}, FinalJudgmentCommand
        feedback.issues = [{**issue, "command_field": "update"} for issue in feedback.issues]
    draft = JudgmentDraft(case.case_id, 0, "assimilate", model, {
        "rejected_response": raw, "error": str(feedback), "issues": feedback.issues,
    })
    prefix = ["update"] if final else []
    assert [t["path"] for t in draft.targets.values()] == [
        [*prefix, "hypotheses"], [*prefix, "ranking_update", "candidate_bases"],
        [*prefix, "ranking_update", "boundary_comparisons"]]
    fixed = draft.apply(repair_fields(hypotheses=good["hypotheses"],
                                     **{k: good["ranking_update"][k] for k in
                                        ("candidate_bases", "boundary_comparisons")})(draft.payload()))
    fixed = fixed["update"] if final else fixed
    assert fixed == good
    validate_assimilation(materialize_judgment(case, JudgmentCommand.model_validate(fixed)), case)


def test_initial_contract_rejects_names_only_top3_but_defers_comparisons():
    case, good = fixture()
    agent, client = planner(case, [good])
    agent.assimilate(case)
    payload = client.calls[0]["payload"]
    schema = payload["output_contract"]
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    validator.validate(proposal_wire(good))
    assert set(payload["hypothesis_write_contract"]["required_initial_labels"]) == set(MECHANISM_POOL)
    for fault in ("only_three", "name_only", "null_value", "no_bases", "no_ranking", "foreign_label"):
        bad = proposal_wire(good)
        if fault == "only_three":
            bad["hypotheses"] = bad["hypotheses"][:3]
        elif fault == "name_only":
            bad["hypotheses"][-1] = {"name": bad["hypotheses"][-1]["name"]}
        elif fault == "null_value":
            bad["hypotheses"][-1]["confidence"] = None
        elif fault == "no_bases":
            bad["ranking_update"]["candidate_bases"] = []
        elif fault == "foreign_label":
            bad["hypotheses"][-1]["name"] = "not_a_label"
        else:
            bad["ranking_update"] = None
        assert list(validator.iter_errors(bad)), fault
    accepted = replace_state(apply_assimilation(case, materialize_judgment(case,
                            JudgmentCommand.model_validate(good))), phase="assimilate")
    sparse = command(accepted).model_dump(mode="json")
    agent, client = planner(accepted, [sparse])
    agent.assimilate(accepted)
    Draft202012Validator(client.calls[0]["payload"]["output_contract"]).validate(sparse)


def test_first_round_gaps_can_be_fixed_in_one_bounded_patch():
    case, good = fixture()
    bad = deepcopy(good)
    bad["hypotheses"] = bad["hypotheses"][:3]
    bad["ranking_update"]["candidate_bases"] = bad["ranking_update"]["candidate_bases"][:3]
    bad["ranking_update"]["boundary_comparisons"] = []
    agent, client = planner(case, [bad, repair_fields(hypotheses=good["hypotheses"],
        **{k: good["ranking_update"][k] for k in ("candidate_bases", "boundary_comparisons")})])
    before = case.model_dump()
    result = agent.assimilate(case)
    assert len(client.calls) == 3  # Proposal, bounded repair, then frozen-order explanation.
    expected = materialize_judgment(case, JudgmentCommand.model_validate(good))
    assert result.portfolio == expected.portfolio
    validate_assimilation(result, case)
    assert case.model_dump() == before
    assert client.calls[1]["payload"]["candidate_command"] == proposal_wire(bad)
