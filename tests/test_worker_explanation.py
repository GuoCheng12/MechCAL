"""Offline contracts only: scripted explanations are not scientific results."""

from copy import deepcopy

import pytest

from mechcal.agents.report_writer import WorkerReportWriter
from mechcal.agents.worker_explanation import explain_result, resolve_source_path
from mechcal.runtime.questions import apply_result_review, public_payload, replace_state, state, validate_review
from mechcal.runtime.persistence import RunStore
from mechcal.schemas import AgentReport, ToolExecutionResult
from mechcal.schemas.failures import FailureReport
from mechcal.schemas.question_loop import ResultReview
from mechcal.schemas.worker_explanation import WorkerExplanation
from test_question_loop import NOTE, REGISTRY, initial_case, loop, pending_case, request
from test_runtime import _evidence_unit


class ExplanationClient:
    def __init__(self, change=None):
        self.calls = []
        self.change = change

    def is_configured(self):
        return True

    def complete_json(self, *, payload, system_prompt, schema_feedback=None):
        self.calls.append(deepcopy(payload))
        if payload["stage"] == "worker_explanation_repair":
            result = deepcopy(payload["rejected_response"])
            if self.change:
                self.change(result)
            repairs = []
            for path in payload["allowed_paths"]:
                value = result
                for key in path:
                    value = value[key]
                repairs.append({"path": path, "value": value})
            return {"repairs": repairs}
        assert payload["stage"] == "worker_result_explanation"
        units = payload["tool_execution_result"]["evidence_units"]
        usable = [(i, e) for i, e in enumerate(units) if e["status"] in {"present", "partial"}]
        result = {
            "question_ids": payload["dispatch_request"]["question_ids"],
            "answer_to_question": NOTE,
            "findings": [dict(
                finding_id="F1", observation=NOTE, interpretation=NOTE,
                molecular_scope="Not assigned in this offline fixture.", conditions=NOTE,
                limitations=NOTE, evidence_refs=[usable[0][1]["evidence_id"]],
                source_paths=[f"/evidence_units/{usable[0][0]}/summary"],
            )] if usable and payload["status"] in {"success", "partial"} else [],
            "missing_observations": [],
        }
        if self.change:
            self.change(result)
        return result


def fixture_result(cid="macro.screen_rotor_torsion_topology"):
    req = request(cid=cid)
    evidence = _evidence_unit(evidence_id="X1:E1", round_id="R001", source_report_id="X1",
                              agent_name=req.agent_name, family=req.evidence_goal_family, summary=NOTE)
    # A deeply nested record tests that no generic depth/list truncation occurs.
    atoms = [dict(atom_index=i, contribution=i / 100, element="C") for i in range(30)]
    structured = {"parsed": {"states": [{"state_index": i, "atom_contributions": atoms}
                                         for i in range(12)]}}
    return req, ToolExecutionResult(
        tool_result_id="X1", round_id="R001", agent_name=req.agent_name,
        dispatch_id=req.dispatch_id, capability_id=cid, selected_route=req.route,
        status="success", raw_results={"original": NOTE}, structured_results=structured,
        evidence_units=[evidence], summary=NOTE,
    )


@pytest.mark.parametrize("cid", ["macro.screen_rotor_torsion_topology", "microscopic.run_baseline_bundle"])
def test_both_workers_explain_without_mutating_sources_or_loading_private_data(cid):
    req, result = fixture_result(cid)
    before = result.model_dump(mode="json")
    case = initial_case()
    case.input.metadata["hidden_reference"] = "PRIVATE SENTINEL"
    client = ExplanationClient()
    explanation = explain_result(client, req, case, result, REGISTRY.require(cid), None)
    assert explanation.findings[0].evidence_refs == ["X1:E1"]
    data = client.calls[0]
    assert "PRIVATE SENTINEL" not in str(data)
    assert "rubric" not in data and "portfolio" not in data
    assert len(data["tool_execution_result"]["structured_results"]["parsed"]["states"]) == 12
    assert len(data["tool_execution_result"]["structured_results"]["parsed"]["states"][11]["atom_contributions"]) == 30
    assert result.model_dump(mode="json") == before


@pytest.mark.parametrize("field,value,match", [
    ("evidence_refs", ["other_execution:E1"], "this execution"),
    ("source_paths", ["/structured_results/not_here"], "does not resolve"),
    ("source_paths", ["/raw_results/original"], "inside structured_results"),
])
def test_worker_rejects_invalid_citations(field, value, match):
    req, result = fixture_result()
    client = ExplanationClient(lambda data: data["findings"][0].update({field: value}))
    with pytest.raises(ValueError, match=match):
        explain_result(client, req, initial_case(), result, REGISTRY.require(req.capability_id), None)


def test_question_ids_and_control_fields_cannot_be_changed():
    req, result = fixture_result()
    client = ExplanationClient(lambda data: data.update(question_ids=["different_question"]))
    with pytest.raises(ValueError, match="question_ids"):
        explain_result(client, req, initial_case(), result, REGISTRY.require(req.capability_id), None)
    for key in ("ranking", "next_tool", "claim_status", "finalize"):
        with pytest.raises(ValueError):
            WorkerExplanation.model_validate(dict(question_ids=["Q1"], answer_to_question=NOTE, **{key: NOTE}))


@pytest.mark.parametrize("path", ["/structured_results/items/-1", "/structured_results/items/01",
                                  "/structured_results/empty", "/structured_results"])
def test_invalid_or_empty_source_paths(path):
    with pytest.raises(ValueError):
        resolve_source_path({"structured_results": {"items": [1, 2], "empty": None}}, path)


def case_with_explanation():
    case = pending_case("success")
    req, result = fixture_result()
    explanation = explain_result(ExplanationClient(), req, case, result, REGISTRY.require(req.capability_id), None)
    packet = {**state(case)["results"][0], "evidence_refs": ["X1:E1"],
              "worker_explanation": explanation.model_dump(mode="json")}
    return replace_state(case, results=[packet], pending_results=[packet])


def review_payload():
    return dict(status="reviewed", interpretations=[dict(
        result_id="X1", question_ids=["Q1"], delivered=NOTE, missing=NOTE, boundary=NOTE,
        evidence_refs=["X1:E1"], finding_audits=[dict(finding_id="F1", status="correction_required",
                                                    reason=NOTE, correction="Retain only the bounded observation.")],
    )])


def test_review_must_check_every_finding_and_persist_corrections_for_planner():
    case = case_with_explanation()
    data = review_payload()
    checked = apply_result_review(case, ResultReview.model_validate(data))
    assert checked.portfolio == case.portfolio
    assert checked.evidence_ledger == case.evidence_ledger
    payload = public_payload(checked, REGISTRY, stage="planner_assimilate")
    assert payload["pending_results"][0]["explanation_review"]["assessment"]["finding_audits"][0]["correction"]
    assert payload["previous_worker_explanations"] == []
    later = public_payload(replace_state(checked, pending_results=[]), REGISTRY, stage="planner_assimilate")
    assert later["previous_worker_explanations"][0]["explanation_review"]["status"] == "reviewed"
    assert "worker_explanation" not in later["execution_history"][0]
    data["interpretations"][0]["finding_audits"] = []
    with pytest.raises(ValueError, match="each worker finding"):
        validate_review(ResultReview.model_validate(data), case)


@pytest.mark.parametrize("status", ["unavailable", "disabled"])
def test_unavailable_review_does_not_approve_explanation(status):
    checked = apply_result_review(case_with_explanation(), ResultReview(status=status, reason=NOTE))
    audit = state(checked)["results"][0]["explanation_review"]
    assert audit["status"] == status and audit["assessment"] is None


def test_durable_explanation_failure_does_not_rerun_calculation(tmp_path):
    orchestrator, _ = loop(tmp_path)
    worker = orchestrator.deps.workers["macro"]
    worker.report_writer.enable_llm = True
    client = ExplanationClient(lambda data: data["findings"][0].update(evidence_refs=["bad_ref"]))
    worker.report_writer.llm_client = client
    case = initial_case()
    req = request()
    store = RunStore(tmp_path, case.case_id)
    store.initialize()
    executions = []
    original = worker.execute_tool

    def execute(*args, **kwargs):
        executions.append(True)
        return original(*args, **kwargs)

    worker.execute_tool = execute
    with pytest.raises(RuntimeError, match="report writing failed schema validation"):
        orchestrator._durable_execution(worker, req, case, store)
    saved = store.read_execution_checkpoint(req)
    assert saved["stage"] == "tool_returned" and len(executions) == 1
    client.change = None
    result, report = orchestrator._durable_execution(worker, req, case, store)
    assert len(executions) == 1
    assert report.result_explanation is not None
    assert report.raw_results == result.raw_results
    assert report.structured_results == result.structured_results
    assert report.evidence_units == result.evidence_units
    assert report.artifact_updates == result.artifact_updates
    assert AgentReport.model_validate(report.model_dump()) == report
    calls = len(client.calls)
    assert orchestrator._durable_execution(worker, req, case, store)[1] == report
    assert len(client.calls) == calls and len(executions) == 1


def test_full_loop_forwards_worker_explanation_and_keeps_action_count(tmp_path):
    orchestrator, planner = loop(tmp_path)
    client = ExplanationClient()
    for worker in orchestrator.deps.workers.values():
        worker.report_writer.enable_llm = True
        worker.report_writer.llm_client = client
    result = orchestrator.run(smiles="CCO", user_query=NOTE, case_id="explanation-loop")
    assert result.status == "finalized"
    assert result.runtime["worker_call_counts"]["macro"] == 1
    assert len(client.calls) == 1
    packets = [p["pending_results"][0] for p in planner.calls if p.get("pending_results")]
    assert packets[0]["worker_explanation"]["findings"]
    assert packets[0]["explanation_review"]["status"] == "disabled"
    assert state(result)["results"][0]["worker_explanation"]


def test_legacy_report_still_readable_without_explanation():
    req, result = fixture_result()
    report = WorkerReportWriter(capability_registry=REGISTRY).deterministic_report(request=req, tool_result=result)
    old = report.model_dump()
    old.pop("result_explanation")
    assert AgentReport.model_validate(old).result_explanation is None


@pytest.mark.parametrize("status,kind", [("partial", "partial_evidence"), ("failed", "runtime_failed")])
def test_partial_and_failed_computations_are_not_synthetic_successes(status, kind):
    req, original = fixture_result()
    result = original.model_copy(update={
        "status": status,
        "failure": FailureReport(kind=kind, message=NOTE),
        "evidence_units": [original.evidence_units[0].model_copy(update={
            "status": "partial" if status == "partial" else "failed",
        })],
    })
    client = ExplanationClient()
    explanation = explain_result(client, req, initial_case(), result, REGISTRY.require(req.capability_id), None)
    assert bool(explanation.findings) == (status == "partial")
    if status == "failed":
        fake_finding = dict(finding_id="F1", observation=NOTE, interpretation=NOTE,
                           molecular_scope=NOTE, conditions=NOTE, limitations=NOTE,
                           evidence_refs=["X1:E1"], source_paths=["/evidence_units/0/summary"])
        client.change = lambda data: data.update(findings=[fake_finding])
        with pytest.raises(ValueError, match="cannot yield scientific findings"):
            explain_result(client, req, initial_case(), result, REGISTRY.require(req.capability_id), None)


def test_auditor_receives_worker_explanation_and_verifiable_sources():
    from mechcal.agents.photophysics_arbiter import PhotophysicsArbiterAgent

    case = case_with_explanation()
    _, result = fixture_result()
    case = case.touch(evidence_ledger=case.evidence_ledger.with_items(result.evidence_units))

    class ReviewClient:
        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            pending = payload["pending_results"][0]
            assert pending["worker_explanation"]["findings"][0]["finding_id"] == "F1"
            assert resolve_source_path(pending, "/evidence_units/0/summary") == NOTE
            assert "rubric" not in payload and "hidden_reference" not in payload
            return review_payload()

    review = PhotophysicsArbiterAgent(llm_client=ReviewClient()).review_results(case)
    assert review.interpretations[0].finding_audits[0].status == "correction_required"
