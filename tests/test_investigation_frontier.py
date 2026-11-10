"""Exploration navigation survives delivery without forcing scientific choices."""

from copy import deepcopy

from mechcal.runtime import command_followup as f, command_hypotheses as h
from mechcal.runtime.command_compact import expand_text
from mechcal.runtime.command_loop import planner_payload
from mechcal.runtime.command_memory import json_bytes
from mechcal.runtime.command_operations import protocol, read_records
from mechcal.runtime.command_review_workflow import comparison_options, investigation_frontier, trial_summary
from mechcal.runtime.command_task_context import prepare_planner
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import ReadEvidence, UpdateHypothesesWithWorkflow
from test_command_followup import question
from test_command_protocol import LABELS, TOOL
from test_command_task_context import ready


def test_trial_summary_counts_results_once_and_does_not_convert_failure_to_refutation():
    trials = [{"dispatch_id": "a", "result_id": None}, {"dispatch_id": "a", "result_id": "r1"},
              {"dispatch_id": "a", "result_id": "r1"}, {"dispatch_id": "b", "result_id": "r2"},
              {"dispatch_id": "c", "result_id": "r3"}, {"dispatch_id": "d", "result_id": None}]
    result = trial_summary(trials, {"r1": {"status": "failed"}, "r2": {"status": "partial"}})
    assert result == {"linked_results": 3, "result_statuses": {"failed": 1, "partial": 1, "missing_record": 1},
                      "pending_dispatches": 1}


def test_untried_lower_confidence_question_remains_visible_in_delivery_and_resume(tmp_path):
    c, o, _ = ready(tmp_path)
    c, _ = h.update(c, UpdateHypothesesWithWorkflow(kind="update_hypotheses", note="Alternative remains unresolved.",
        questions=[question(id="comparison_new", labels=[LABELS[-1]], status="deferred",
            route={"capability_id": TOOL, "input_refs": []},
            disposition_reason="Not yet attempted; no negative result.")]))
    c = CaseRun.model_validate_json(c.model_dump_json())
    before = deepcopy(protocol(c))
    full = expand_text(planner_payload(o, c, native=True))
    routed, compact = prepare_planner(c, full)
    compact = expand_text(compact)
    assert compact["context_task"] == "delivery"
    frontier = compact["investigation_frontier"]
    row = frontier["questions"][0]
    assert row["labels"] == [LABELS[-1]] and row["status"] == "deferred"
    assert row["trial_summary"]["linked_results"] == 0
    assert row["read"]["paths"] == ["/questions/comparison_new"]
    assert "confidence" not in row
    assert frontier["omitted_questions"] == 0
    saved = read_records(routed, ReadEvidence(kind="read_evidence", records=[frontier["read_all"]]))[0]
    restored_option = saved["data"]["/fields/comparison_options"][0]
    assert restored_option["labels"] == row["labels"]
    assert restored_option["question"] == row["question"]
    assert "question_location" not in restored_option
    assert protocol(c) == before
    assert compact["exploration_state"]["confidence_ranking"]


def test_route_readiness_is_recomputed_not_inferred_from_confidence(tmp_path):
    c, _, _ = ready(tmp_path)
    c, _ = h.update(c, UpdateHypothesesWithWorkflow(kind="update_hypotheses", note="Fixture.",
        questions=[question(route={"capability_id": TOOL, "input_refs": []})]))
    before = c.model_dump(mode="json")
    for status in ("ready", "disabled", "missing_input"):
        options = comparison_options(c, [{"capability_id": TOOL, "status": status}])
        frontier = investigation_frontier(options)
        assert frontier["questions"][0]["route_status"] == status
    assert c.model_dump(mode="json") == before
    assert f.state(c).questions["compare"].value.status == "open"


def test_bounded_preview_preserves_ids_not_truncated_scientific_claims():
    options = [{"id": str(i), "labels": [LABELS[0] if i < 9 else LABELS[1]],
                "status": "open", "question": "Long exact text. " * 1000,
                "next_test": "Long exact test. " * 1000, "route_status": {"status": "ready"}}
               for i in range(12)]
    original = deepcopy(options)
    frontier = investigation_frontier(options)
    assert len(frontier["questions"]) == 8 and frontier["omitted_questions"] == 4
    assert frontier["questions"][1]["id"] == "9"  # Another label is visible, not another confidence rank.
    assert all(row["details_deferred"] and "question" not in row for row in frontier["questions"])
    assert json_bytes(frontier["questions"]) <= 5000
    assert options == original


def test_frontier_keeps_tool_status_distinct_from_selected_input_status():
    option = {"id": "q", "labels": [LABELS[0]], "question": "Change the conditions?",
        "route_status": {"status": "missing_input", "tool_status": "ready", "input_scope": "selected_records",
            "missing_inputs": ["prepared_structure"], "compatible_input_count": 1,
            "compatible_input_options": [{"source_artifact_id": "geometry:one", "kind": "prepared_structure", "status": "ready"}]}}
    row = investigation_frontier([option])["questions"][0]
    assert row["route_status"] == "missing_input" and row["tool_status"] == "ready"
    assert row["compatible_input_options"][0]["source_artifact_id"] == "geometry:one"
