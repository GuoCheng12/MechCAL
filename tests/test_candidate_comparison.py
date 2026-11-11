"""Comparable candidates share source access, not hard-coded scientific verdicts."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest
from jsonschema import Draft202012Validator

from mechcal.agents.command_tools import tool_contracts, parse_tool_turn
from mechcal.mechanism_pool import MECHANISM_POOL, MECHANISM_DEFINITIONS, mechanism_definitions
from mechcal.runtime import command_hypotheses as h, computational_review as cr
from mechcal.runtime.command_comparison import comparison_view, comparison_refs
from mechcal.runtime.command_compact import expand_text
from mechcal.runtime.command_impact_review import review_payload
from mechcal.runtime.command_loop import _comment, _catalog, planner_payload
from mechcal.runtime.command_operations import protocol, put, record_index
from mechcal.runtime.command_review import review_projection
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import UpdateHypotheses
from test_command_hypotheses import update
from test_command_protocol import LABELS, REGISTRY, case
from test_computational_review import fixture, report, reviewed
from test_native_commands import NativeFake, reply
from test_reviewer_execution_context import execution_case, OTHER, OTHER_UNIT, RESULT, UNIT


def assess(c, label, refs=(), limitation_refs=(), explanation=None):
    return h.update(c, update(.4, label, refs=list(refs), limitation_refs=list(limitation_refs),
                              explanation=explanation))[0]


def test_public_meanings_cover_existing_pool_without_scoring_or_case_information():
    assert len(MECHANISM_POOL) == 11 and tuple(MECHANISM_DEFINITIONS) == MECHANISM_POOL
    data = mechanism_definitions()
    assert "higher excited state" in data["definitions"]["SOKR_ANTI_KASHA"]
    assert "not mean only" in data["definitions"]["RADIATIVE_RATE_STATE_BALANCE"]
    assert "not required" in data["definitions"]["TRIPLET_METAL_ENERGY_TRANSFER"]
    assert not {"hidden_reference", "criteria", "score", "case_id"} & data.keys()
    data["definitions"].clear()
    assert len(mechanism_definitions()["definitions"]) == 11


def test_explanation_persists_separately_and_does_not_change_science_records():
    c = case()
    before = deepcopy(protocol(c))
    c = assess(c, LABELS[0], ["input:smiles"], explanation="A specific provisional physical account.")
    c = assess(c, LABELS[0], ["input:smiles"])
    assert h.state(c).entries[LABELS[0]].explanation == "A specific provisional physical account."
    assert h.state(c).entries[LABELS[0]].note != h.state(c).entries[LABELS[0]].explanation
    for key in ("results", "arguments", "ranking", "rounds", "revision"):
        assert protocol(c).get(key) == before.get(key)
    restored = CaseRun.model_validate_json(c.model_dump_json())
    assert h.state(restored) == h.state(c)
    cleared = assess(c, LABELS[0], ["input:smiles"], explanation="")
    assert h.state(cleared).entries[LABELS[0]].explanation == ""


def test_legacy_assessment_has_unknown_explanation_not_invented_from_note():
    c = assess(case(), LABELS[0], ["input:smiles"])
    p = deepcopy(protocol(c)["exploration"])
    p["entries"][LABELS[0]].pop("explanation")
    c = put(c, exploration=p)
    assert h.state(c).entries[LABELS[0]].explanation is None
    assert comparison_view(c)["interpretations"][0]["explanation"] is None


@pytest.mark.parametrize("strict", [True, False])
def test_explanation_native_schema_and_parser_agree(strict):
    tools, models = tool_contracts(REGISTRY, [], hypothesis_tracking=True, strict_schema=strict)
    schema = next(t["parameters"] for t in tools if t["name"] == "update_hypotheses")
    args = update(explanation="A revisable account.").model_dump(exclude={"kind"})
    Draft202012Validator(schema).validate(args)
    cmd = parse_tool_turn(reply("update_hypotheses", args), tools, models)
    assert cmd.changes[0].explanation == "A revisable account."


def test_parent_and_child_citations_share_one_source_without_removing_either_claim():
    c = execution_case()
    c = assess(c, LABELS[0], [OTHER])
    c = assess(c, LABELS[1], [OTHER_UNIT])
    before = c.model_dump(mode="json")
    view = comparison_view(c)
    assert len(view["source_directory"]) == 1
    row = view["source_directory"][0]
    assert row["source_ref"] == OTHER
    assert row["refs"] == [OTHER, OTHER_UNIT]
    assert row["used_by"] == sorted(LABELS[:2])
    assert row["independence"] == "not_recorded"
    assert comparison_refs(view) == {OTHER, OTHER_UNIT}
    assert c.model_dump(mode="json") == before


def test_failure_is_a_limitation_and_unknown_candidates_remain_unknown():
    c = assess(execution_case(), LABELS[0], limitation_refs=[UNIT])
    view = comparison_view(c)
    row = view["source_directory"][0]
    assert row["source_ref"] == RESULT and row["status"] == "failed"
    assert row["used_by"] == [] and row["limitations_for"] == [LABELS[0]]
    unknown = next(r for r in view["interpretations"] if r["label"] == MECHANISM_POOL[-1])
    assert unknown["confidence"] is None and unknown["explanation"] is None


def test_shared_input_is_dependency_not_claim_of_identical_calculation():
    c = execution_case()
    results = deepcopy(protocol(c)["results"])
    results[OTHER]["artifact_updates"] = [{"artifact_id": "geometry:1"}]
    actions = deepcopy(protocol(c)["actions"])
    actions[results[RESULT]["dispatch_id"]]["input_refs"] = ["geometry:1"]
    c = put(c, results=results, actions=actions)
    c = assess(c, LABELS[0], [OTHER], [RESULT])
    rows = {r["source_ref"]: r for r in comparison_view(c)["source_directory"]}
    assert len(rows) == 2
    assert rows[RESULT]["input_sources"] == [{"input_ref": "geometry:1", "produced_by": [OTHER]}]
    assert rows[RESULT]["independence"] == "not_recorded"


def test_multiple_artifact_producers_are_not_arbitrarily_collapsed():
    from mechcal.schemas.artifacts import ArtifactRecord

    c = execution_case()
    artifact = ArtifactRecord(artifact_id="artifact:shared", kind="fixture", status="available",
                              created_by_round="R001", created_by_agent="microscopic")
    c = c.touch(artifact_manifest=c.artifact_manifest.model_copy(update={"artifacts": [artifact]}))
    results = deepcopy(protocol(c)["results"])
    for raw in results.values():
        raw["artifact_updates"] = [artifact.model_dump(mode="json")]
    c = assess(put(c, results=results), LABELS[0], [artifact.artifact_id])
    view = comparison_view(c)
    row = view["source_directory"][0]
    assert row["source_ref"] == artifact.artifact_id
    assert row["produced_by"] == sorted([OTHER, RESULT])
    assert comparison_refs(view) == {artifact.artifact_id, OTHER, RESULT}


def test_definition_shared_across_planner_review_and_impact_roles(tmp_path):
    c, o, s = fixture(tmp_path)
    fake = NativeFake([reply("record_review", report().model_dump(exclude={"kind"}))])
    o.deps.scientific_reviewer.llm_client = fake
    _comment(o, c, s)
    assert expand_text(fake.requests[0]["payload"])["sections"]["mechanism_definitions"] == mechanism_definitions()
    assert planner_payload(o, c)["mechanism_definitions"] == mechanism_definitions()
    assert review_payload(c, [])["mechanism_definitions"] == mechanism_definitions()


def test_unranked_sources_are_readable_in_same_bounded_final_review(tmp_path):
    c, o, s = fixture(tmp_path)
    outsider = MECHANISM_POOL[-1]
    assert outsider not in protocol(c)["ranking"]
    results = {**protocol(c)["results"], "alternative:result": {
        "tool_result_id": "alternative:result", "case_id": c.case_id,
        "capability_id": "fixture", "status": "success", "values": {"observable": 2.0}}}
    c = assess(put(c, results=results), outsider, ["alternative:result"], explanation="Alternative physical account.")
    fake = NativeFake([
        reply("read_evidence", {"records": [{"ref": "alternative:result", "paths": ["/values"], "view": "data", "page": None}]}),
        reply("record_review", report().model_dump(exclude={"kind"}), call_id="finish")])
    o.deps.scientific_reviewer.llm_client = fake
    before = deepcopy(protocol(c)["results"])
    c = _comment(o, c, s)
    assert protocol(c)["reviewer_session"]["outcome"]["status"] != "rejected"
    sections = expand_text(fake.requests[0]["payload"])["sections"]
    comparison = sections["computational_review"]["candidate_comparison"]
    assert "alternative:result" in comparison_refs(comparison)
    # Outside sources are queryable but not inlined into the ordinary top-three observations.
    assert "observations:alternative:result" not in sections
    c = _comment(o, c, s)
    assert protocol(c)["final_review"]["status"] == "available"
    assert len(fake.requests) == 2 and protocol(c)["results"] == before


def test_comparison_reuse_tracks_explanation_and_sources_but_not_delivery_wording(tmp_path):
    c, o, _ = fixture(tmp_path)
    c = reviewed(c, o, [])
    same = put(c, final_draft={**protocol(c)["final_draft"], "draft_id": "wording-only"})
    assert cr.context(same, _catalog(o, same))["previous"]
    changed = assess(c, LABELS[0], ["input:smiles"], explanation="Changed physical interpretation.")
    assert cr.context(changed, _catalog(o, changed))["previous"] is None


def test_comparison_rejects_cross_case_sources_without_changing_state():
    c = assess(execution_case(), LABELS[0], [OTHER])
    results = deepcopy(protocol(c)["results"])
    results[OTHER]["case_id"] = "other-case"
    with pytest.raises(RuntimeError):
        comparison_view(put(c, results=results))


def test_no_scientific_values_are_copied_into_source_directory():
    c = assess(execution_case(), LABELS[0], [OTHER])
    text = json.dumps(comparison_view(c))
    assert "effective_parameters" not in text and "state_root" not in text
    assert "report_ref" in text


def test_changed_definition_contract_does_not_reuse_old_review_interpretations(tmp_path):
    from test_delivery_review_scope import reviewed_case

    c, o, s = reviewed_case(tmp_path)
    old = deepcopy(protocol(c)["review_memory"]["final_reviewer"])
    old.pop("context_contract")
    c = put(c, review_memory={"final_reviewer": old},
            review_receipts={**protocol(c)["review_receipts"], old["ref"]: old})
    payload = {"stage": "command_final_reviewer", "role": "final_reviewer",
               "mechanism_definitions": mechanism_definitions(),
               "assembled_answer": {"answer": "Existing prose " * 100},
               "computational_review": cr.context(c, _catalog(o, c))}
    material, receipt, previous = review_projection(c, payload, "final_reviewer", index=record_index(c), task_context=True)
    assert not material["reused_sections"] and previous is None
    assert "previous_feedback" not in material
    assert material["sections"]["assembled_answer:answer"] == payload["assembled_answer"]["answer"]
    assert receipt["context_contract"]


def test_two_comparisons_can_use_one_capability_without_forcing_a_second_tool(tmp_path):
    from test_command_protocol import TOOL
    from mechcal.schemas.commands import RecordComputationalReview, SubmitFinalAfterReview

    c, o, _ = fixture(tmp_path)
    suggestions = [{"capability_id": TOOL, "question_id": None, "purpose": text}
                   for text in ("Compare torsion A.", "Compare torsion B as a control.")]
    c = reviewed(c, o, suggestions)
    assert protocol(c)["computational_review"]["suggestions"] == suggestions
    command = SubmitFinalAfterReview(kind="submit_final", draft_id=protocol(c)["final_draft"]["draft_id"],
        route_responses=[{"capability_id": TOOL, "reason": "Neither comparison resolves the remaining uncertainty."}])
    assert len(cr.require_response(c, command, cr.context(c, _catalog(o, c)))) == 1
