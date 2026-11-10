"""Offline explanation-context checks; fixtures are not scientific calculations."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime.case_memory import CaseMemoryStore, pointer, serialized
from mechcal.runtime.persistence import RunStore
from mechcal.runtime.ranking_explanations import RankingExplanationDraft
from mechcal.runtime.retry import RequestValidationError
from mechcal.schemas import CaseRun
from mechcal.schemas.memory import MemoryQuery
from mechcal.schemas.question_loop import QuestionAssimilation
from mechcal.schemas.ranking_explanations import RankingExplanations
from test_judgment_commands import explanation_response
from test_ranking_explanation_transaction import prepared_case, draft_for, proposal
from test_runtime import _evidence_unit


def payloads(case, update):
    disabled = case.model_copy(deep=True)
    disabled.runtime["planner_case_memory"] = False
    enabled = case.model_copy(deep=True)
    enabled.runtime["planner_case_memory"] = True
    return (RankingExplanationDraft(disabled, update).payload(),
            RankingExplanationDraft(enabled, update).payload())


def read_full(index, record_id):
    fragments, start = [], 0
    while True:
        item = index.read(MemoryQuery(action="read_case_records", record_ids=[record_id], start=start))["records"][0]
        assert "error" not in item
        if "record" in item:
            return item["record"]
        fragments.append(item["json_fragment"])
        if item["next_start"] is None:
            return json.loads("".join(fragments))
        start = item["next_start"]


def restore(row, payload, index):
    result = {}
    if "memory_record_id" in row:
        result = read_full(index, row["memory_record_id"])
        if "memory_path" in row:
            result = pointer(result, row["memory_path"])
    result = deepcopy(result)
    for key, value in row.items():
        if key in {"memory_record_id", "memory_path"}:
            continue
        if key.endswith("_context_path"):
            result[key.removesuffix("_context_path")] = pointer(payload, value)
        else:
            result[key] = value
    return result


def test_projection_is_immutable_lossless_and_schema_identical(tmp_path):
    case = prepared_case()
    update = draft_for(case, proposal(case)).update
    case.runtime["planner_case_memory"] = True
    before = case.model_dump_json(), update.model_dump_json(), RankingExplanations.model_json_schema()
    draft = RankingExplanationDraft(case, update)
    legacy, compact = payloads(case, update)
    index = CaseMemoryStore(RunStore(tmp_path, case.case_id), case)
    assert compact["output_contract"] == legacy["output_contract"]
    assert compact["draft_id"] == legacy["draft_id"] == draft.draft_id
    assert compact["boundary_targets"] == legacy["boundary_targets"]
    assert compact["mechanism_pool"] == list(MECHANISM_POOL)
    for key in ("proposed_ranking", "support_arguments"):
        assert [restore(row, compact, index) for row in compact[key]] == legacy[key]
    assert [restore(row, compact, index) for row in compact["candidate_bases"]] == legacy["rationale"]["candidate_bases"]
    assert restore(compact["rationale"], compact, index) == {
        k: v for k, v in legacy["rationale"].items() if k != "candidate_bases"}
    assert compact["new_result_assessments"] == legacy["new_result_assessments"]
    contexts = {e.evidence_id: e.context for e in case.evidence_ledger.items}
    assert compact["observations"] == [
        {**e, "context": contexts[e["evidence_id"]], "memory_record_id": e["evidence_id"]}
        for e in legacy["observations"]]
    assert "source_results" not in compact
    response = explanation_response(compact)
    Draft202012Validator(compact["output_contract"]).validate(response)
    result = draft.apply(RankingExplanations.model_validate(response))
    assert result.portfolio == update.portfolio
    compact["previous_comparison"]["revision"] = -100
    assert (case.model_dump_json(), update.model_dump_json(), RankingExplanations.model_json_schema()) == before


@pytest.mark.parametrize("large", [False, True])
def test_new_numbers_conditions_and_complete_execution_are_recoverable(tmp_path, large):
    case = prepared_case()
    unit = _evidence_unit(evidence_id="E:new", round_id="R002", source_report_id="fixture",
        agent_name="macro", family="torsion_sensitivity", summary="Synthetic numeric fixture only.",
        context="Fixture geometry G1, solvent S1, units eV.", metrics={
            "energy": 2.123456789012345, "conditions": {"geometry": "G1", "solvent": "S1"},
            "signed_coefficients": [-.0123456789012345, .0987654321098765] * (2000 if large else 2),
        })
    case.evidence_ledger.items.append(unit)
    raw = {"result_id": "X:new", "status": "partial", "evidence_refs": [unit.evidence_id],
           "requested_parameters": {"state_root": 3, "solvent": "S1"},
           "resolved_source_artifact_id": "G1", "input_refs": ["G1"],
           "structured_results": {"exact_only_in_execution": 1.987654321098765,
                                  "trace": list(range(5000))}}
    case.runtime["question_loop"]["pending_results"] = [raw]
    update = QuestionAssimilation.model_validate(case.runtime["question_loop"]["assimilation"])
    legacy, compact = payloads(case, update)
    index = CaseMemoryStore(RunStore(tmp_path, case.case_id), case)
    observation = next(e for e in compact["observations"] if e["evidence_id"] == unit.evidence_id)
    assert restore(observation, compact, index) == unit.model_dump(mode="json")
    assert set(observation) == {"evidence_id", "claim", "summary", "limits", "basis", "context", "memory_record_id"}
    assert "source_results" not in compact
    assert read_full(index, "X:new") == raw
    assert len(serialized(compact)) < len(serialized(legacy))
    before = case.model_dump_json()
    observation["limits"].append("Changed payload only.")
    assert case.model_dump_json() == before


def test_changed_alternative_and_missing_history_never_point_to_stale_records():
    case = prepared_case()
    update = draft_for(case, proposal(case)).update
    tail = update.portfolio.sorted_hypotheses()[-1]
    tail.rationale = "New interpretation absent from committed case memory."
    basis = next(b for b in update.ranking_update.candidate_bases if b.label == tail.name)
    basis.inference_gap = "New inference gap, not the old accepted basis."
    case.runtime["question_loop"].pop("last_comparison")
    _, compact = payloads(case, update)
    row = next(h for h in compact["proposed_ranking"] if h["name"] == tail.name)
    assert row["rationale"] == tail.rationale and "memory_record_id" not in row
    assert all("memory_record_id" not in b for b in compact["candidate_bases"])
    assert "memory_record_id" not in compact["previous_comparison"]


def test_displaced_targets_and_unranked_challenger_remain_available():
    case = prepared_case()
    update = draft_for(case, proposal(case)).update
    for h in update.portfolio.sorted_hypotheses()[:2]:
        h.differential_priority = 0
    _, compact = payloads(case, update)
    assert len(compact["boundary_targets"]) == 3
    targets = {t[k] for t in compact["boundary_targets"] for k in ("included_label", "excluded_label")}
    assert all("memory_record_id" not in h for h in compact["proposed_ranking"] if h["name"] in targets)
    case.runtime.update(evidence_decision_contracts=False, planner_case_memory=True)
    removed = update.portfolio.hypotheses.pop()
    draft = RankingExplanationDraft(case, update)
    payload = draft.payload()
    assert removed.name in payload["mechanism_pool"]
    response = explanation_response(payload)
    response["leading"]["challenger"] = removed.name
    Draft202012Validator(payload["output_contract"]).validate(response)
    assert draft.apply(RankingExplanations.model_validate(response)).ranking_update.leading_comparison.challenger == removed.name


@pytest.mark.parametrize("fault", ["stale", "same_challenger", "foreign_ref", "operational_ref",
                                   "no_reference", "missing_pair", "scores"])
def test_existing_rejections_unchanged(fault):
    case = prepared_case()
    case.runtime["planner_case_memory"] = True
    if fault == "operational_ref":
        case.evidence_ledger.items.append(_evidence_unit(
            evidence_id="E:failed", round_id="R002", source_report_id="fixture", agent_name="macro",
            family="geometry_precondition", summary="Synthetic execution failure.", status="failed"))
    draft = draft_for(case, proposal(case))
    response = explanation_response(draft.payload())
    if fault == "stale":
        response["draft_id"] = "stale"
    elif fault == "same_challenger":
        response["leading"]["challenger"] = draft.ranked[0].name
    elif fault == "foreign_ref":
        response["leading"]["evidence_refs"] = ["foreign:E1"]
    elif fault == "operational_ref":
        response["leading"]["evidence_refs"] = ["E:failed"]
    elif fault == "no_reference":
        response["leading"]["preference_status"] = "evidence_favors_leader"
    elif fault == "missing_pair":
        response["boundaries"] = []
    else:
        response["hypotheses"] = [{"name": draft.ranked[0].name, "differential_priority": 0}]
    before = case.model_dump_json(), draft.update.model_dump_json()
    with pytest.raises((ValidationError, RequestValidationError)):
        draft.apply(RankingExplanations.model_validate(response))
    assert (case.model_dump_json(), draft.update.model_dump_json()) == before


@pytest.mark.parametrize("leading,boundary", [(False, False), (False, True), (True, False), (True, True)])
def test_feature_flags_preserve_contract_and_targets(leading, boundary):
    case = prepared_case()
    update = draft_for(case, proposal(case)).update
    case.runtime.update(hypothesis_tests=leading, evidence_decision_contracts=boundary)
    legacy, compact = payloads(case, update)
    for key in ("output_contract", "leading_target", "boundary_targets", "draft_id"):
        assert compact[key] == legacy[key]


@pytest.mark.parametrize("name,round_index", [
    ("mechcal_working_context_0003_20261024", 3),
    ("mechcal_evidence_refresh_0003_json_schema_20261024", 5),
    ("mechcal_evidence_refresh_0003_laicai_schema_fresh_20261024", 5),
])
def test_real_checkpoint_payload_reduction_and_retrieval(name, round_index, tmp_path):
    root = Path(__file__).resolve().parents[3] / "experiments"
    path = root / name / "smoke/runs/AIE_DDX_MECH_V4_0003/question_checkpoint.json"
    if not path.exists():
        pytest.skip("Local acceptance checkpoint is not distributed with the package")
    case = CaseRun.model_validate_json(path.read_text())
    assert case.runtime["question_loop"]["round_index"] == round_index
    before = case.model_dump_json()
    update = QuestionAssimilation.model_validate(case.runtime["question_loop"]["assimilation"])
    legacy, compact = payloads(case, update)
    sizes = [len(serialized(p).encode()) for p in (legacy, compact)]
    print(f"{name}: legacy={sizes[0]} compact={sizes[1]} reduction={1-sizes[1]/sizes[0]:.1%}")
    assert sizes[1] < .95 * sizes[0]
    assert compact["output_contract"] == legacy["output_contract"]
    assert compact["boundary_targets"] == legacy["boundary_targets"]
    index = CaseMemoryStore(RunStore(tmp_path, case.case_id), case)
    for key in ("proposed_ranking", "support_arguments"):
        assert [restore(r, compact, index) for r in compact[key]] == legacy[key]
    expected = {e.evidence_id: e.model_dump(mode="json") for e in case.evidence_ledger.items}
    assert compact["observations"] == [
        {**e, "context": expected[e["evidence_id"]]["context"], "memory_record_id": e["evidence_id"]}
        for e in legacy["observations"]]
    for row in compact["observations"]:
        assert restore(row, compact, index) == expected[row["evidence_id"]]
    results = {r["result_id"]: r for r in [*case.runtime["question_loop"]["results"],
                                          *case.runtime["question_loop"]["pending_results"]]}
    assert "source_results" not in compact
    for result_id, result in results.items():
        assert read_full(index, result_id) == result
    assert case.model_dump_json() == before
