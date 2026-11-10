"""Fault-injection acceptance of the robustness audit; no external models."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest
from mechcal.runtime.field_repair import FieldRepairProducer
from mechcal.runtime.persistence import RunStore
from mechcal.runtime.question_loop import _find_completed, _stop_review
from mechcal.runtime.questions import replace_state, state, validate_assimilation
from mechcal.runtime.retry import SchemaRetryRunner
from mechcal.schemas.question_loop import QuestionAssimilation
from pydantic import BaseModel
from test_field_repair import corrections, missing_receipt
from test_question_loop import NOTE, action, initial_case, loop, ready_case, request


@pytest.mark.parametrize("partial", [False, True])
def test_mixed_bookkeeping_errors_stay_scoped(partial):
    case, data, receipt = missing_receipt()
    case = replace_state(
        case, argument_review_pending=True, argument_review={"support_ids": ["structural"]}
    )
    data["audit_responses"] = {}
    calls = []

    def full(feedback):
        calls.append("full")
        return deepcopy(data)

    def repair(payload, feedback):
        calls.append("repair")
        if calls.count("repair") == 1:
            assert payload["allowed_paths"] == [["argument_changes"], ["audit_responses"]]
            patches = corrections(["argument_changes"], [receipt])
            if not partial:
                patches["repairs"] += corrections(["audit_responses"], {"structural": NOTE})[
                    "repairs"
                ]
            return patches
        assert payload["allowed_paths"] == [["audit_responses"]]
        assert payload["rejected_response"]["argument_changes"]
        return corrections(["audit_responses"], {"structural": NOTE})

    before = case.model_dump(mode="json")
    result = SchemaRetryRunner(
        QuestionAssimilation,
        repair_context=True,
        validators=[lambda u: validate_assimilation(u, case)],
    ).run(FieldRepairProducer(full, repair, response_model=QuestionAssimilation))
    assert calls == ["full", "repair"] + (["repair"] if partial else [])
    assert result.portfolio == QuestionAssimilation.model_validate(data).portfolio
    assert (
        result.mechanism_support_arguments
        == QuestionAssimilation.model_validate(data).mechanism_support_arguments
    )
    assert case.model_dump(mode="json") == before


def test_first_response_field_type_error_is_scoped():
    case, data, receipt = missing_receipt()
    data["argument_changes"] = [receipt]
    data["revision"] = "not-an-integer"
    calls = []

    def full(feedback):
        calls.append("full")
        return data

    def repair(payload, feedback):
        calls.append("repair")
        assert payload["allowed_paths"] == [["revision"]]
        return corrections(["revision"], state(case)["revision"] + 1)

    result = SchemaRetryRunner(QuestionAssimilation, repair_context=True).run(
        FieldRepairProducer(full, repair, response_model=QuestionAssimilation)
    )
    assert result.revision == state(case)["revision"] + 1
    assert calls == ["full", "repair"]


def test_validator_program_error_is_not_a_paid_retry():
    class Reply(BaseModel):
        value: int

    calls = []

    def producer(feedback):
        calls.append(feedback)
        return {"value": 1}

    def broken_validator(reply):
        int("broken-local-configuration")

    with pytest.raises(ValueError, match="broken-local-configuration"):
        SchemaRetryRunner(Reply, repair_context=True, validators=[broken_validator]).run(producer)
    assert len(calls) == 1


def test_auditor_program_bug_stops_even_when_outages_are_tolerated(tmp_path, monkeypatch):
    orchestrator, planner = loop(tmp_path)
    orchestrator.config = replace(
        orchestrator.config, enable_support_auditor=True, continue_on_arbiter_failure=True
    )

    def broken_review(case, **kwargs):
        raise KeyError("local-result-parser-bug")

    monkeypatch.setattr(orchestrator.deps.photophysics_arbiter, "review_results", broken_review)
    with pytest.raises(KeyError, match="local-result-parser-bug"):
        orchestrator.run(smiles="CCO", user_query=NOTE, case_id="audit-bug")
    saved = json.loads((tmp_path / "audit-bug/question_checkpoint.json").read_text())
    assert saved["status"] == "failed"
    assert saved["runtime"]["question_loop"]["phase"] == "review"
    assert saved["runtime"]["question_loop"]["pending_results"]


def test_truncated_history_tail_is_preserved_without_hiding_next_report(tmp_path):
    store = RunStore(tmp_path, "audit")
    store.initialize()
    req = request()
    path = store.root / "history/tool_events.jsonl"
    # Simulate termination halfway through an append, before the newline.
    path.write_text('{"event":"agent_report","dispatch":')
    row = {
        "event": "agent_report",
        "dispatch": req.model_dump(mode="json"),
        "report": {"fixture": True},
    }
    store.append_jsonl("history/tool_events.jsonl", row)
    assert _find_completed(store, req) == row
    assert path.read_text().splitlines()[0] == '{"event":"agent_report","dispatch":'


def test_stop_review_propagates_local_bug(tmp_path, monkeypatch):
    from mechcal.runtime.questions import decision_from_action

    orchestrator, _ = loop(tmp_path)
    orchestrator.config = replace(
        orchestrator.config, enable_agenda_coverage_reviewer=True, continue_on_arbiter_failure=False
    )
    case = ready_case()
    decision = decision_from_action(action(case.case_id, state(case)["revision"]), case, "R001")
    store = RunStore(tmp_path, case.case_id)
    store.initialize()

    def broken_review(*args, **kwargs):
        raise KeyError("local-stop-review-bug")

    monkeypatch.setattr(orchestrator.deps.mechanism_agenda, "review_stop", broken_review)
    with pytest.raises(KeyError, match="local-stop-review-bug"):
        _stop_review(orchestrator, case, decision, store)
    assert not state(case).get("stop_review_done")


@pytest.mark.parametrize(
    "stage,expected_executions,expected_reports",
    [
        ("planned", 1, 1),
        ("executing", 0, 0),
        ("tool_returned", 1, 1),
        ("reported", 1, 1),
    ],
)
def test_execution_checkpoint_crash_boundaries(
    tmp_path, monkeypatch, stage, expected_executions, expected_reports
):
    orchestrator, _ = loop(tmp_path)
    worker = orchestrator.deps.workers["macro"]
    case, req = initial_case(), request()
    store = RunStore(tmp_path, case.case_id)
    store.initialize()
    calls = {"execute": 0, "report": 0}
    execute, report = worker.execute_tool, worker.report_result
    write = store.write_execution_checkpoint

    def count_execute(*args, **kwargs):
        calls["execute"] += 1
        return execute(*args, **kwargs)

    def count_report(*args, **kwargs):
        calls["report"] += 1
        return report(*args, **kwargs)

    def interrupt(*args, **kwargs):
        write(*args, **kwargs)
        if kwargs["stage"] == stage:
            raise KeyboardInterrupt("offline injected process interruption")

    monkeypatch.setattr(worker, "execute_tool", count_execute)
    monkeypatch.setattr(worker, "report_result", count_report)
    monkeypatch.setattr(store, "write_execution_checkpoint", interrupt)
    with pytest.raises(KeyboardInterrupt):
        orchestrator._durable_execution(worker, req, case, store)
    monkeypatch.setattr(store, "write_execution_checkpoint", write)
    if stage == "executing":
        with pytest.raises(RuntimeError, match="unknown completion"):
            orchestrator._durable_execution(worker, req, case, store)
    else:
        result, saved_report = orchestrator._durable_execution(worker, req, case, store)
        assert saved_report.evidence_units == result.evidence_units
        assert store.read_execution_checkpoint(req)["stage"] == "reported"
    assert calls == {"execute": expected_executions, "report": expected_reports}


def test_interrupted_derived_write_retains_previous_record(tmp_path, monkeypatch):
    from pathlib import Path

    store = RunStore(tmp_path, "audit")
    store.initialize()
    path = store.write_json("case_run.json", {"accepted": True})
    original = Path.write_text

    def interrupted_write(self, data, **kwargs):
        if self.name.startswith(".case_run.json."):
            original(self, data[:5], **kwargs)
            raise OSError("injected partial disk write")
        return original(self, data, **kwargs)

    monkeypatch.setattr(Path, "write_text", interrupted_write)
    with pytest.raises(OSError, match="partial disk write"):
        store.write_json("case_run.json", {"accepted": False})
    assert json.loads(path.read_text()) == {"accepted": True}
    assert not list(path.parent.glob(".case_run.json.*.tmp"))


@pytest.mark.parametrize("history", ["missing", "truncated", "stale"])
def test_reported_checkpoint_is_authoritative_over_history(tmp_path, history):
    o, _ = loop(tmp_path)
    case, req = initial_case(), request()
    store = RunStore(tmp_path, case.case_id)
    store.initialize()
    result, report = o._durable_execution(o.deps.workers["macro"], req, case, store)
    path = store.root / "history/tool_events.jsonl"
    if history == "truncated":
        path.write_text('{"event":')
    elif history == "stale":
        store.append_jsonl(
            "history/tool_events.jsonl",
            {
                "event": "agent_report",
                "dispatch": req.model_dump(mode="json"),
                "report": {"stale": True},
            },
        )
    recovered = _find_completed(store, req)
    expected = report.model_dump(mode="json")
    expected["structured_results"]["dispatch_request"] = req.model_dump(mode="json")
    assert recovered["report"] == expected
    assert recovered["tool_result"] == result.model_dump(mode="json")
    from mechcal.agents.evidence_context import calculation_record
    from mechcal.schemas import AgentReport

    record = calculation_record(AgentReport.model_validate(recovered["report"]))
    assert record["requested_parameters"] == req.tool_args
    assert record["capability_id"] == req.capability_id
    assert record["dispatch_id"] == req.dispatch_id


def test_worker_rejected_outputs_and_errors_are_saved_before_resume(tmp_path):
    from test_worker_explanation import ExplanationClient

    o, _ = loop(tmp_path)
    case, req = initial_case(), request()
    worker = o.deps.workers["macro"]
    worker.report_writer.enable_llm = True
    worker.report_writer.llm_client = ExplanationClient(
        lambda data: data["findings"][0].update(evidence_refs=["wrong-execution"])
    )
    store = RunStore(tmp_path, case.case_id)
    store.initialize()
    with pytest.raises(RuntimeError, match="schema validation"):
        o._durable_execution(worker, req, case, store)
    events = [
        json.loads(line)
        for line in (store.root / "history/component_requests.jsonl").read_text().splitlines()
    ]
    rejected = [e for e in events if e["kind"] == "validation_error"]
    assert len(rejected) == 3
    assert all(
        e["response"]["findings"][0]["evidence_refs"] == ["wrong-execution"] for e in rejected
    )
    assert len([e for e in events if e["kind"] == "model_response"]) == 3
    assert store.read_execution_checkpoint(req)["stage"] == "tool_returned"


def test_result_review_rejections_are_recorded_without_changing_observations():
    from mechcal.agents.photophysics_arbiter import PhotophysicsArbiterAgent
    from mechcal.runtime.retry import SchemaRetryError
    from test_worker_explanation import case_with_explanation, review_payload

    case = case_with_explanation()
    before = case.model_dump_json()
    events = []

    class Client:
        def complete_json(self, **kwargs):
            data = review_payload()
            data["interpretations"][0]["finding_audits"] = []
            return data

    with pytest.raises(SchemaRetryError, match="each worker finding"):
        PhotophysicsArbiterAgent(llm_client=Client()).review_results(case, event_sink=events.append)
    assert len([e for e in events if e["kind"] == "model_response"]) == 3
    assert len([e for e in events if e["kind"] == "validation_error"]) == 3
    assert case.model_dump_json() == before


def test_coverage_parser_bug_is_not_wrapped_as_model_failure(monkeypatch):
    import mechcal.agents.mechanism_agenda as module

    class Client:
        def is_configured(self):
            return True

        def complete_json(self, **kwargs):
            return {}

    def broken(*args, **kwargs):
        raise KeyError("local-coverage-parser")

    monkeypatch.setattr(module, "_sanitize_coverage", broken)
    with pytest.raises(KeyError, match="local-coverage-parser"):
        module.MechanismAgendaAgent(llm_client=Client()).review_coverage(
            ready_case(), round_id="R002"
        )


def test_coverage_local_error_message_cannot_masquerade_as_outage(tmp_path):
    o, _ = loop(tmp_path)
    assert not o._can_continue_after_agenda_coverage_failure(KeyError("timeout"))
    assert not o._can_continue_after_agenda_coverage_failure(ValueError("failed to produce a valid parser"))
    assert o._can_continue_after_agenda_coverage_failure(TimeoutError("network timeout"))
