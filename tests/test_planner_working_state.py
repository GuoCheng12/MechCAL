"""Working notes are accepted judgments; observations and repairs stay authoritative."""

from copy import deepcopy

import pytest

from mechcal.agents.planner_references import PlannerReferences
from mechcal.runtime.case_memory import pointer, serialized
from mechcal.runtime.context_compaction import expand_payload
from mechcal.runtime.delta_context import current_state, decision_context, lookup_context
from mechcal.runtime.judgment_commands import materialize_judgment
from mechcal.runtime.planner_memory import PlannerMemorySession
from mechcal.runtime.questions import apply_assimilation, public_payload, replace_state, state
from mechcal.schemas import CaseRun
from mechcal.schemas.memory import MemoryQuery
from test_delta_context import DONE, READY, WireClient, setup
from test_incremental_judgment import prepared
from test_judgment_commands import command
from test_question_loop import REGISTRY


def scientific_payload():
    p = public_payload(prepared(), REGISTRY, stage="planner_assimilate")
    p["previous_support_arguments"][0].update(finding="OLD_FINDING", warrant="OLD_WARRANT")
    p["questions"][0]["assessment"] = "OLD_QUESTION_DETAILS"
    p["hypothesis_test_context"] = {
        "enabled": True, "current_comparison": {"leader": "A", "challenger": "B",
            "leading_hypothesis": "OLD_LEADING_EXPLANATION", "preference_status": "prior_only"},
        "recent_tests": [{"result_id": "old", "revision": 1, "assessment": {"reason": "OLD_TEST"}}],
    }
    p["boundary_decision_context"] = {
        "enabled": True, "required_stop_route_ids": ["microscopic.fixture"],
        "current_candidate_bases": [{"label": "A", "positive_refs": ["E1"], "counter_refs": ["E2"],
            "supported_observation": "OLD_BASIS", "inference_gap": "OLD_GAP"}],
    }
    p["execution_history"] = [{"result_id": "old", "status": "failed",
        "failure": {"code": "unsupported", "message": "Wrong element"},
        "requested_parameters": {"state": "S1"}, "delivery": {"details": "OLD_DELIVERY"}}]
    return p


@pytest.mark.parametrize("stage", ["planner_assimilate", "planner_question_act"])
def test_default_packet_has_exact_note_and_indices_not_old_explanations(stage):
    source = scientific_payload()
    source["stage"] = stage
    original = deepcopy(source)
    result = decision_context(source, "context:test")
    assert result["current_working_state"]["assessment"] == source["previous_assessment"]
    assert result["current_working_state"]["base_revision"] == source["revision"]
    assert result["current_working_state"]["kind"] == "accepted_judgment_not_evidence"
    assert result["previous_support_arguments"][0]["finding"] == "OLD_FINDING"
    assert "OLD_WARRANT" not in serialized(result)
    assert "OLD_QUESTION_DETAILS" not in serialized(result)
    assert len(result["portfolio"]["hypotheses"]) == len(source["portfolio"]["hypotheses"])
    assert result["execution_history"][0]["failure"] == source["execution_history"][0]["failure"]
    assert result["boundary_decision_context"]["current_candidate_bases"][0]["counter_refs"] == ["E2"]
    assert result["boundary_decision_context"]["required_stop_route_ids"] == ["microscopic.fixture"]
    assert source == original

    def verify(value):
        if isinstance(value, dict):
            if "read_path" in value:
                row = pointer(source, value["read_path"])
                assert isinstance(row, dict)
                for k in row.keys() & value.keys():
                    assert row[k] == value[k]
            for child in value.values():
                verify(child)
        elif isinstance(value, list):
            for child in value:
                verify(child)
    verify(result)


def test_unknown_context_details_do_not_expand_default_packet_but_remain_addressable():
    source = scientific_payload()
    source["additional_history"] = {"data": "LONG_HISTORY " * 2000}
    result = decision_context(source, "context:test")
    assert "LONG_HISTORY" not in serialized(result)
    assert pointer(source, result["context_paths"]["additional_history"]) == source["additional_history"]


def test_no_saved_assessment_does_not_invent_or_reuse_a_summary():
    assert current_state({"revision": 0}) == {
        "base_revision": 0, "assessment": None, "kind": "no_saved_assessment"}


def test_lookup_exposes_current_note_and_pending_work_without_forcing_rereads():
    p = scientific_payload()
    p.update(argument_review_pending=True, argument_review={"support_ids": ["S1"]},
             assimilation_context={"required_disposition_result_ids": ["new"]},
             candidate_context={"pending": [{"proposal_id": "C1"}]})
    result = lookup_context(p, "context:test")
    assert result["current_working_state"]["assessment"] == p["previous_assessment"]
    assert result["attention"]["argument_review_pending"]
    assert result["attention"]["pending_candidate_count"] == 1
    assert "/argument_review" in result["attention"]["feedback_paths"]
    assert "next_useful_step" not in result  # Already present in the saved assessment.
    assert "OLD_" not in serialized(result)


def test_new_evidence_corrections_and_candidate_proposals_are_not_hidden_by_old_note():
    p = scientific_payload()
    p.update(
        pending_results=[{"result_id": "new", "question_ids": ["Q1"], "status": "partial",
            "failure": {"message": "Some output missing"}, "evidence_refs": ["E3"],
            "investigation_plan": {"purpose": "discrimination", "hypothesis_test": {"tested_claim": "compare"}},
            "worker_explanation": {"findings": [{"finding_id": "F1", "text": "Counterevidence", "limits": ["S1 only"]}]}}],
        observations=[{"evidence_id": "E3", "context": "S1 fixed geometry", "limits": ["No emission claim"],
            "metrics": {"value": -0.2, "unit": "eV"}}],
        argument_review_pending=True,
        argument_review={"support_ids": ["S1"], "correction": "Retract stale claim"},
        result_review={"status": "unavailable", "reason": "Not approved"},
        candidate_context={"pending": [{"proposal_id": "C1", "reason": "New explanation"}]},
        stop_review_feedback={"challenge": "Do not ignore the alternative"},
    )
    result = decision_context(p, "context:test")
    for key in ("observations", "argument_review", "result_review", "candidate_context", "stop_review_feedback"):
        assert result[key] == p[key]
    assert result["pending_results"][0]["failure"] == p["pending_results"][0]["failure"]
    assert result["pending_results"][0]["worker_explanation"]["findings"] == p["pending_results"][0]["worker_explanation"]["findings"]


def test_explicit_reads_restore_old_details_to_self_contained_decision(tmp_path):
    case, _, memory = setup(tmp_path, start_lookup=False)
    payload = {**memory.payload, "previous_assessment": {"current_best": "Accepted fixture judgment"},
               "historical_detail": {"conditions": "NOT_IN_SUMMARY", "value": 0}}
    def read(active):
        return {**READY[0], "memory_queries": [MemoryQuery(
            action="read_case_records", record_ids=[active["context_record_id"]],
            paths=["/historical_detail"],
        ).model_dump(mode="json")]}
    client = WireClient([(read, "context_read"), DONE])
    memory.complete(client, PlannerReferences(case), payload=payload, system_prompt="unused")
    first, final = [expand_payload(c["payload"]) for c in client.calls]
    assert "NOT_IN_SUMMARY" not in serialized(first)
    assert "NOT_IN_SUMMARY" in serialized(final["retrieved_memory"])
    assert "historical_detail" not in final
    assert memory.used == 1


def test_accepted_note_survives_checkpoint_and_changes_only_after_commit(tmp_path):
    case = prepared()
    old_note = deepcopy(state(case)["assimilation"]["decision_assessment"])
    proposal = command(case)
    proposal.decision_assessment.current_best = "A revised fixture interpretation, not a measured result."
    update = materialize_judgment(case, proposal)
    assert state(case)["assimilation"]["decision_assessment"] == old_note
    updated = apply_assimilation(case, update)
    restored = CaseRun.model_validate_json(updated.model_dump_json())
    p = public_payload(restored, REGISTRY, stage="planner_question_act")
    result = decision_context(p, "context:next")
    assert result["current_working_state"]["assessment"] == proposal.decision_assessment.model_dump(mode="json")
    assert result["current_working_state"]["base_revision"] == state(restored)["revision"]
    assert restored.portfolio == case.portfolio
    assert restored.mechanism_support_arguments == case.mechanism_support_arguments
    assert restored.evidence_ledger == case.evidence_ledger


def test_next_revision_starts_with_note_not_previous_query_bodies(tmp_path):
    case, store, memory = setup(tmp_path)
    memory.query(MemoryQuery(action="read_case_records", record_ids=["Q1"]).model_dump(mode="json"))
    changed = replace_state(case, revision=state(case)["revision"] + 1)
    payload = public_payload(changed, REGISTRY, stage=memory.stage)
    next_memory = PlannerMemorySession(changed, store, REGISTRY, memory.stage, payload)
    assert next_memory.used == 0
    assert next_memory.augment(next_memory.payload)["retrieved_memory"] == []


@pytest.mark.parametrize("stage", ["planner_final_revision", "planner_judgment_repair"])
def test_final_and_repair_contract_details_remain_visible(stage):
    p = scientific_payload()
    p.update(stage=stage, final_internal_audit={"must_read": "FINAL_AUDIT"},
             scientific_context={"required": "REPAIR_CONTEXT"}, targets=["S1"])
    result = decision_context(p, "context:test")
    for key in ("final_internal_audit", "scientific_context", "targets"):
        assert result[key] == p[key]
    assert result["previous_support_arguments"][0]["finding"] == "OLD_FINDING"
