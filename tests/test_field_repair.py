"""Offline correction tests; scripted repairs are not real-model success claims."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mechcal.runtime.field_repair import FieldRepairProducer, apply_corrections
from mechcal.runtime import MechCALOrchestrator
from mechcal.runtime.questions import apply_assimilation, replace_state, state, validate_assimilation
from mechcal.runtime.retry import (
    RejectedResponseError, RequestValidationError, RetryPolicy, SchemaRetryError, SchemaRetryRunner,
)
from mechcal.schemas import CaseRun
from mechcal.schemas.question_loop import QuestionAssimilation
from test_evidence_comparison import ComparisonClient, ready, update_for
from test_question_loop import NOTE, loop


def corrections(path, value):
    return {"repairs": [{"path": path, "value": value}]}


def missing_receipt():
    case = replace_state(ready(), phase="assimilate")
    data = deepcopy(update_for(case))
    data["mechanism_support_arguments"][0]["boundary"] = "Revised boundary, not a new finding."
    receipt = dict(support_id="structural", disposition="revised", reason="Clarified the boundary.")
    return case, data, receipt


def runner(case, **kwargs):
    return SchemaRetryRunner(QuestionAssimilation, repair_context=True,
                             validators=[lambda x: validate_assimilation(x, case)], **kwargs)


def test_collects_receipt_and_all_reference_errors_together_without_mutating_state():
    case, data, _ = missing_receipt()
    before = case.model_dump(mode="json")
    data["questions"][0]["result_ids"] = ["unknown:execution"]
    data["portfolio"]["hypotheses"][0]["evidence_refs"] = ["unknown:evidence"]
    with pytest.raises(RequestValidationError) as raised:
        validate_assimilation(QuestionAssimilation.model_validate(data), case)
    issues = raised.value.issues
    assert len(issues) == 3
    assert issues[0]["changes"][0]["after"]["boundary"] == data["mechanism_support_arguments"][0]["boundary"]
    assert issues[1]["allowed_refs"] == []
    assert case.model_dump(mode="json") == before


def test_scoped_retry_keeps_scientific_text_and_ranking_exactly_as_submitted(tmp_path):
    case, data, receipt = missing_receipt()
    original = deepcopy(data)
    events, calls = [], []
    o, _ = loop(tmp_path, rounds=0, unlimited=True)

    class Client:
        def is_configured(self):
            return True

        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            calls.append(payload["stage"])
            if schema_feedback is None:
                return data
            assert payload["stage"] == "planner_field_repair"
            assert payload["allowed_paths"] == [["argument_changes"]]
            return corrections(["argument_changes"], [receipt])

    o.deps.planner.llm_client = Client()
    result = o.deps.planner.assimilate(case, request_event_sink=events.append)
    expected = QuestionAssimilation.model_validate({**original, "argument_changes": [receipt]})
    assert result == expected
    assert data == original
    assert calls == ["planner_assimilate", "planner_field_repair"]
    assert [e["kind"] for e in events].count("attempt_started") == 2
    assert any(e["kind"] == "field_repair_applied" for e in events)
    updated = apply_assimilation(case, result)
    assert updated.portfolio == result.portfolio
    assert updated.evidence_ledger == case.evidence_ledger
    assert state(updated)["revision"] == state(case)["revision"] + 1


@pytest.mark.parametrize("bad", [
    {"repairs": [{"path": ["portfolio"], "value": {}}]},
    {"repairs": [{"path": ["argument_changes"], "value": []}] * 2},
    {"repairs": []},
    {"repairs": [{"path": ["argument_changes"], "value": [], "extra": 1}]},
    {"repairs": [{"path": [True], "value": []}]},
    {"repairs": [{"path": ["questions", -1], "value": {}}]},
    {"portfolio": {}, "argument_changes": []},
])
def test_malformed_or_unscoped_correction_does_not_mutate_candidate(bad):
    _, data, _ = missing_receipt()
    before = deepcopy(data)
    with pytest.raises(ValueError):
        apply_corrections(data, bad, [["argument_changes"]])
    assert data == before


def test_overlapping_paths_are_rejected():
    response = {"repairs": [{"path": ["x"], "value": {}}, {"path": ["x", "y"], "value": 2}]}
    with pytest.raises(ValueError, match="Overlapping"):
        apply_corrections({"x": {"y": 1}}, response, [["x"], ["x", "y"]])


def test_missing_optional_field_can_be_filled_and_other_fields_preserved():
    assert apply_corrections({"x": 1}, corrections(["argument_changes"], []), [["argument_changes"]]) == {
        "x": 1, "argument_changes": []}


def test_unknown_reference_is_never_fuzzy_matched_or_accepted():
    case, data, receipt = missing_receipt()
    data["argument_changes"] = [receipt]
    data["questions"][0]["evidence_refs"] = ["input:smile"]
    calls = []

    def repair(payload, feedback):
        calls.append(payload)
        assert payload["issues"][0]["allowed_refs"] == ["input:smiles"]
        return corrections(["questions", 0, "evidence_refs"], ["input:smile"])

    before = case.model_dump(mode="json")
    with pytest.raises(SchemaRetryError):
        runner(case).run(FieldRepairProducer(lambda _: data, repair, response_model=QuestionAssimilation))
    assert len(calls) == 2
    assert case.model_dump(mode="json") == before


@pytest.mark.parametrize("malformed", [False, True])
def test_failed_patch_retains_candidate_and_budget_then_allows_valid_correction(malformed):
    case, data, receipt = missing_receipt()
    attempts = []

    def repair(payload, feedback):
        attempts.append(payload)
        assert payload["rejected_response"] == data
        if len(attempts) == 1:
            if malformed:
                raise RejectedResponseError("Not JSON", "broken")
            return corrections(["portfolio"], {})
        return corrections(["argument_changes"], [receipt])

    result = runner(case).run(FieldRepairProducer(lambda _: data, repair, response_model=QuestionAssimilation))
    assert len(attempts) == 2
    assert result.portfolio == QuestionAssimilation.model_validate(data).portfolio


def test_local_producer_errors_are_not_paid_retries():
    case, data, _ = missing_receipt()
    calls = []

    def repair(payload, feedback):
        calls.append(payload)
        raise ValueError("Local logging failure")

    with pytest.raises(ValueError, match="Local logging failure"):
        runner(case).run(FieldRepairProducer(lambda _: data, repair, response_model=QuestionAssimilation))
    assert len(calls) == 1


def test_invalid_field_type_stays_in_scoped_correction_without_full_regeneration():
    case, data, receipt = missing_receipt()
    full_calls, patch_calls = [], []

    def full(feedback):
        full_calls.append(feedback)
        assert feedback is None
        return data

    def repair(payload, feedback):
        patch_calls.append(payload)
        assert payload["rejected_response"] == data
        return corrections(["argument_changes"], "wrong type" if len(patch_calls) == 1 else [receipt])

    result = runner(case).run(FieldRepairProducer(full, repair, response_model=QuestionAssimilation))
    assert len(full_calls) == 1 and len(patch_calls) == 2
    assert result.argument_changes[0].support_id == "structural"


def test_repair_still_runs_original_scientific_constraints():
    case, data, receipt = missing_receipt()
    data["ranking_update"]["basis"] = "initial"
    producer = FieldRepairProducer(lambda _: data, lambda p, f: corrections(["argument_changes"], [receipt]),
                                  response_model=QuestionAssimilation)
    with pytest.raises(SchemaRetryError, match="Use initial only"):
        runner(case, policy=RetryPolicy(max_attempts=2)).run(producer)








def test_nested_final_revision_path_cannot_modify_final_action():
    original = {"update": {"argument_changes": []}, "final_action": {"action": "stop"}}
    fixed = apply_corrections(original, corrections(["update", "argument_changes"], [1]),
                              [["update", "argument_changes"]])
    assert fixed["final_action"] == original["final_action"]
    with pytest.raises(ValueError):
        apply_corrections(original, corrections(["final_action"], {}), [["update", "argument_changes"]])


@pytest.mark.parametrize("stage", ["planner_assimilate", "planner_final_revision"])
def test_complete_workflow_repairs_references_without_extra_scientific_rounds(tmp_path, monkeypatch, stage):
    from mechcal.schemas.question_loop import ResultReview

    o, _ = loop(tmp_path, discover=True)
    client_repairs = []

    class Client(ComparisonClient):
        injected = False

        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            if payload["stage"] == "planner_field_repair":
                assert "evidence_index" in payload and "execution_history" in payload
                issue = payload["issues"][0]
                path = issue["repair_paths"][0]
                assert path == (["update"] if stage == "planner_final_revision" else []) + [
                    "questions", 0, "evidence_refs"]
                assert "input:smiles" in issue["allowed_refs"]
                client_repairs.append(payload)
                return corrections(path, ["input:smiles"])
            data = super().complete_json(payload=payload, system_prompt=system_prompt,
                                         schema_feedback=schema_feedback)
            if payload["stage"] == stage and not self.injected:
                self.injected = True
                update = data["update"] if stage == "planner_final_revision" else data
                update["questions"][0]["evidence_refs"] = ["input:smile"]
            return data

    o.deps.planner.llm_client = Client(1, True)
    config = replace(o.config, comparative_evidence=True, enable_support_auditor=True)
    o = MechCALOrchestrator(config, o.deps)
    monkeypatch.setattr(o.deps.photophysics_arbiter, "review_results", lambda c, **kw: ResultReview(
        interpretations=[dict(result_id=r["result_id"], question_ids=r["question_ids"], delivered=NOTE,
                              missing=NOTE, boundary=NOTE, evidence_refs=r["evidence_refs"])
                         for r in state(c).get("pending_results", [])]))
    result = o.run(smiles="CCO", user_query=NOTE, case_id="correction-loop")
    assert result.status == "finalized" and state(result)["final_revision_complete"]
    assert len(client_repairs) == 1
    assert result.round_ids == ["R001", "R002"]
    assert result.runtime["worker_call_counts"]["macro"] == 1
    events = [json.loads(s) for s in (tmp_path / "correction-loop/history/planner_requests.jsonl").read_text().splitlines()]
    assert any(e["kind"] == "field_repair_applied" for e in events)
