"""Small working decisions retain exact source data and retrieval paths."""

from copy import deepcopy

from mechcal.runtime.case_memory import CaseMemoryStore
from mechcal.runtime.planner_context import build_context
from mechcal.runtime.planner_memory import PlannerMemorySession
from mechcal.runtime.questions import public_payload
from mechcal.schemas.memory import MemoryQuery
from test_case_memory import fixture, REGISTRY
from test_working_context import context


def expanded_context(stage="planner_assimilate"):
    payload = context(stage)
    payload["incremental_update"] = {"enabled": True, "interface": "judgment_proposal_v3"}
    payload["portfolio"]["hypotheses"][0].update(
        rationale="Full earlier rationale " * 40, differential_priority=0.7,
        validation_needed=["More measurements"],
    )
    payload["boundary_decision_context"] = {"current_candidate_bases": [
        {"label": "A", "prior_rationale": "A prior", "supported_observation": "Observed feature",
         "inference_gap": "Only these conditions", "positive_refs": ["old"], "counter_refs": []},
    ]}
    payload["questions"] = [{"question_id": "Q1", "question": "What next?", "answer_state": "open",
                             "assessment": "Material question", "observable": "Energy",
                             "missing_or_unresolved": ["Historical detail " * 40]}]
    return payload


def test_sparse_decision_keeps_all_labels_bases_scores_and_exact_new_evidence():
    before = expanded_context()
    original = deepcopy(before)
    after = build_context(before)
    assert before == original
    assert after["portfolio"]["hypotheses"][0]["memory_record_id"] == "hypothesis:A"
    for key in ("name", "confidence", "differential_priority", "evidence_refs"):
        assert after["portfolio"]["hypotheses"][0][key] == before["portfolio"]["hypotheses"][0][key]
    assert after["boundary_decision_context"] == before["boundary_decision_context"]
    assert after["observations"] == [before["observations"][1]]
    assert after["questions"][0]["assessment"] == "Material question"
    assert after["questions"][0]["question_id"] == "Q1"
    assert len(str(after)) < len(str(before))


def test_affected_hypothesis_and_final_delivery_do_not_lose_details():
    before = expanded_context()
    before["evidence_impact_context"] = {"targets": [{"label": "A", "current_support_ids": ["S1"]}]}
    after = build_context(before)
    assert after["portfolio"] == before["portfolio"]
    before["stage"] = "planner_final_revision"
    after = build_context(before)
    assert after["portfolio"] == before["portfolio"]
    assert after["questions"] == before["questions"]


def test_no_candidate_basis_means_no_hidden_rationale():
    before = expanded_context()
    before["boundary_decision_context"]["current_candidate_bases"] = []
    assert build_context(before)["portfolio"] == before["portfolio"]


def test_history_outcome_index_keeps_non_discrimination_and_original_record_handle():
    before = expanded_context()
    before["hypothesis_test_context"] = {"enabled": True, "recent_tests": [
        {"revision": 1, "round": 1, "result_id": "X1",
         "plan": {"hypothesis_test": {"tested_claim": "Original hypothesis", "target_label": "A"}},
         "assessment": {"basis_effect": "unresolved", "comparison_effect": "non_discriminating",
                        "evidence_refs": ["old"], "observed_result": "Full interpretation"}},
    ]}
    row = build_context(before)["hypothesis_test_context"]["recent_tests"][0]
    assert row["memory_record_id"] == "test:1:X1"
    assert row["assessment"]["comparison_effect"] == "non_discriminating"
    assert row["tested_claim"]["tested_claim"] == "Original hypothesis"


def test_retrieval_uses_same_small_working_set_with_exact_readable_records(tmp_path):
    current, store = fixture(tmp_path)
    payload = public_payload(current, REGISTRY, stage="planner_question_act")
    memory = PlannerMemorySession(current, store, REGISTRY, payload["stage"], payload)
    before = deepcopy(memory.payload)
    memory.query({"action": "read_case_records", "record_ids": ["Q1"], "paths": ["/question"]})
    after = memory.augment(memory.payload)
    assert all(after[key] == value for key, value in before.items())
    assert after["retrieved_memory"][0]["records"][0]["record"]["selected_fields"]["/question"]
    exact = CaseMemoryStore(store, current).read(MemoryQuery(action="read_case_records", record_ids=["Q1"]))
    assert exact["records"][0]["record"]["question_id"] == "Q1"
    assert memory.used == 1 and after["memory_access"]["queries_remaining"] == 3


def test_active_feedback_stays_full_and_inactive_feedback_is_retrievable():
    before = expanded_context("planner_question_act")
    before["argument_review_pending"] = True
    assert build_context(before)["argument_review"] == before["argument_review"]
    before["argument_review_pending"] = False
    assert build_context(before)["argument_review"] == {"memory_record_id": "review:argument"}
