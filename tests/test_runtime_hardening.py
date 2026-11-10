"""Offline fault injection; fixtures are not scientific performance evidence."""

import json
import socket
from copy import deepcopy
from dataclasses import replace

import pytest

from mechcal.runtime.questions import calculation_identity, replace_state, state, validate_action
from mechcal.runtime.retry import RejectedResponseError, RequestValidationError, SchemaRetryError
from test_question_loop import NOTE, REGISTRY, TOOL, QuestionClient, action, loop, ready_case, request


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No network is permitted in fault injection")
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def events(directory):
    return [json.loads(line) for line in (directory / "history/planner_requests.jsonl").read_text().splitlines()]


def test_unknown_tools_and_all_reference_errors_are_correctable(tmp_path):
    o, _ = loop(tmp_path)

    class Client(QuestionClient):
        rejected = None

        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            if payload["stage"] == "planner_question_act" and payload["round_id"] == "R001":
                if self.rejected is None:
                    data = action(payload["case_id"], payload["revision"], dispatch=True).model_dump(mode="json")
                    second = deepcopy(data["dispatch_requests"][0])
                    data["dispatch_requests"][0]["capability_id"] = "unknown.tool"
                    second.update(dispatch_id="R001:second", question_ids=["unknown_question"], reuse_result_ids=["foreign"])
                    data["dispatch_requests"].append(second)
                    self.rejected = deepcopy(data)
                    return data
                repair = json.loads(schema_feedback)
                assert repair["rejected_response"] == self.rejected
                assert {x["code"] for x in repair["issues"]} >= {"unknown_capability", "unknown_question", "unknown_result"}
                schema_feedback = None
            return super().complete_json(payload=payload, system_prompt=system_prompt, schema_feedback=schema_feedback)

    client = Client()
    o.deps.planner.llm_client = client
    result = o.run(smiles="CCO", user_query=NOTE, case_id="correctable")
    assert result.status == "finalized"
    assert result.runtime["worker_call_counts"]["macro"] == 1
    rows = events(tmp_path / "correctable")
    bad = [e for e in rows if e["kind"] == "validation_error"]
    assert len(bad) == 1 and bad[0]["response"] == client.rejected
    assert any(e["kind"] == "model_response" and e["response"] == client.rejected for e in rows)


@pytest.mark.parametrize("phase,stage", [("assimilate", "planner_assimilate"), ("act", "planner_question_act"),
                                        ("final_revision", "planner_final_revision")])
def test_every_planner_stage_exhausts_without_losing_state_and_resumes(tmp_path, phase, stage):
    o, _ = loop(tmp_path, rounds=0, unlimited=True)
    o.config = replace(o.config, enable_support_auditor=True)

    class Client(QuestionClient):
        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            if payload["stage"] == stage:
                return {"audit_marker": "rejected", "case_id": payload["case_id"]}
            return super().complete_json(payload=payload, system_prompt=system_prompt, schema_feedback=schema_feedback)

    o.deps.planner.llm_client = Client(rounds=0)
    with pytest.raises(SchemaRetryError):
        o.run(smiles="CCO", user_query=NOTE, case_id="recoverable")
    directory = tmp_path / "recoverable"
    saved = json.loads((directory / "question_checkpoint.json").read_text())
    assert saved["status"] == "failed"
    assert saved["runtime"]["question_loop"]["phase"] == phase
    assert saved["runtime"]["termination"]["reason"] == "request_retries_exhausted"
    assert not (directory / "final.json").exists()
    failures = [r for r in events(directory) if r["kind"] == "validation_error" and r["phase"] == phase]
    assert len(failures) == 3
    assert all(r["response"]["audit_marker"] == "rejected" for r in failures)
    o.deps.planner.llm_client = QuestionClient(rounds=0)
    result = o.run(smiles="CCO", user_query=NOTE, case_id="recoverable", resume=True)
    assert result.status == "finalized"
    assert result.runtime["worker_call_counts"]["macro"] == 0
    resume = json.loads((directory / "history/resume_events.jsonl").read_text().splitlines()[-1])
    assert resume["previous_request_state"]["kind"] == "exhausted"
    assert resume["previous_request_state"]["attempt"] == 3
    assert resume["resume_generation"] == 1


def test_query_keeps_rejected_draft_and_feedback(tmp_path):
    o, _ = loop(tmp_path, discover=True)

    class Client(QuestionClient):
        count = 0
        feedback = None

        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            if payload["stage"] == "planner_question_act" and payload["round_id"] == "R001":
                self.count += 1
                if self.count == 1:
                    return {"action": "dispatch", "rationale": "ORIGINAL_DRAFT"}
                if self.count == 2:
                    self.feedback = schema_feedback
                    return {"action": "describe_tools", "capability_ids": [TOOL]}
                assert schema_feedback == self.feedback
                assert json.loads(schema_feedback)["rejected_response"]["rationale"] == "ORIGINAL_DRAFT"
                schema_feedback = None
            return super().complete_json(payload=payload, system_prompt=system_prompt, schema_feedback=schema_feedback)

    o.deps.planner.llm_client = Client(discover=True)
    result = o.run(smiles="CCO", user_query=NOTE, case_id="repair-query")
    assert result.status == "finalized"
    rows = events(tmp_path / "repair-query")
    assert len([r for r in rows if r["kind"] == "attempt_started" and r["phase"] == "act" and r["round_id"] == "R001"]) == 2


def test_unknown_short_citation_keeps_the_original_action_draft(tmp_path):
    o, _ = loop(tmp_path, rounds=0, unlimited=True, discover=True)
    class Client(QuestionClient):
        rejected = None
        repaired = False

        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            if payload["stage"] == "planner_question_act" and not self.repaired:
                if self.rejected is None:
                    self.rejected = action(payload["case_id"], payload["revision"]).model_dump(mode="json")
                    self.rejected["evidence_refs"] = ["unknown-short-citation"]
                    return deepcopy(self.rejected)
                feedback = json.loads(schema_feedback)
                assert feedback["rejected_response"] == self.rejected
                assert "Unknown evidence reference" in feedback["error"]
                self.repaired = True
                schema_feedback = None
            return super().complete_json(payload=payload, system_prompt=system_prompt,
                                         schema_feedback=schema_feedback)
    client = Client(rounds=0)
    o.deps.planner.llm_client = client
    result = o.run(smiles="CCO", user_query=NOTE, case_id="bad-citation")
    assert result.status == "finalized" and client.repaired
    assert not result.evidence_ledger.items


def test_recorded_retry_does_not_misclassify_producer_configuration_errors():
    from mechcal.runtime.retry import SchemaRetryRunner
    from mechcal.schemas.question_loop import QuestionAction

    calls = []
    def producer(feedback):
        calls.append(feedback)
        raise ValueError("Unsupported API configuration, not a model response")
    with pytest.raises(ValueError, match="Unsupported API configuration"):
        SchemaRetryRunner(QuestionAction, repair_context=True).run(producer)
    assert calls == [None]


@pytest.mark.parametrize("point", ["planning", "planned", "executing", "tool_returned", "reported"])
def test_execution_checkpoints_preserve_success_without_duplicate_tools(tmp_path, monkeypatch, point):
    from mechcal.runtime.persistence import RunStore

    o, _ = loop(tmp_path)
    worker = o.deps.workers["macro"]
    run_tool = worker.execute_tool
    calls = []

    def count(*args, **kwargs):
        calls.append(True)
        return run_tool(*args, **kwargs)

    monkeypatch.setattr(worker, "execute_tool", count)
    original = RunStore.write_execution_checkpoint

    def interrupt(store, request, **kwargs):
        original(store, request, **kwargs)
        if kwargs["stage"] == point:
            raise KeyboardInterrupt("Offline crash after durable checkpoint")

    monkeypatch.setattr(RunStore, "write_execution_checkpoint", interrupt)
    with pytest.raises(KeyboardInterrupt):
        o.run(smiles="CCO", user_query=NOTE, case_id="interrupt")
    monkeypatch.setattr(RunStore, "write_execution_checkpoint", original)
    if point == "executing":
        with pytest.raises(RuntimeError, match="unknown completion"):
            o.run(smiles="CCO", user_query=NOTE, case_id="interrupt", resume=True)
        assert calls == []
    else:
        result = o.run(smiles="CCO", user_query=NOTE, case_id="interrupt", resume=True)
        assert result.status == "finalized"
        assert len(calls) == 1
        assert len(state(result)["results"]) == 1
        assert len(result.evidence_ledger.items) == 1


def test_report_failure_resumes_report_only(tmp_path, monkeypatch):
    o, _ = loop(tmp_path)
    worker = o.deps.workers["macro"]
    normal = worker.report_result
    calls = []
    tool = worker.execute_tool

    def execute(*args, **kwargs):
        calls.append(True)
        return tool(*args, **kwargs)

    def fail(*args, **kwargs):
        raise TimeoutError("Report unavailable after successful tool")

    monkeypatch.setattr(worker, "execute_tool", execute)
    monkeypatch.setattr(worker, "report_result", fail)
    with pytest.raises(TimeoutError):
        o.run(smiles="CCO", user_query=NOTE, case_id="report-recovery")
    files = list((tmp_path / "report-recovery/executions").glob("*.json"))
    saved = json.loads(files[0].read_text())
    assert saved["stage"] == "tool_returned" and saved["tool_result"]["status"] == "success"
    assert saved["tool_result"]["evidence_units"]
    monkeypatch.setattr(worker, "report_result", normal)
    result = o.run(smiles="CCO", user_query=NOTE, case_id="report-recovery", resume=True)
    assert result.status == "finalized" and len(calls) == 1


@pytest.mark.parametrize("allow_continue", [True, False])
def test_review_failure_policy_is_consistent(tmp_path, monkeypatch, allow_continue):
    o, client = loop(tmp_path)
    o.config = replace(o.config, enable_support_auditor=True, continue_on_arbiter_failure=allow_continue)

    def fail(_, **kwargs):
        raise TimeoutError("Offline result review failure")

    monkeypatch.setattr(o.deps.photophysics_arbiter, "review_results", fail)
    if not allow_continue:
        with pytest.raises(TimeoutError):
            o.run(smiles="CCO", user_query=NOTE, case_id="review-policy")
        checkpoint = json.loads((tmp_path / "review-policy/question_checkpoint.json").read_text())
        assert checkpoint["runtime"]["question_loop"]["phase"] == "review"
    else:
        assert o.run(smiles="CCO", user_query=NOTE, case_id="review-policy").status == "finalized"
        assert any(p.get("result_review", {}).get("status") == "unavailable"
                   for p in client.calls if p.get("result_review"))


@pytest.mark.parametrize("change", ["prompt", "chemistry"])
def test_resume_rejects_changed_discovery_prompt_and_chemistry(tmp_path, monkeypatch, change):
    import mechcal.runtime.question_loop as module

    o, _ = loop(tmp_path, discover=True)
    original = o.deps.planner.plan_initial
    monkeypatch.setattr(o.deps.planner, "plan_initial", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        o.run(smiles="CCO", user_query=NOTE, case_id="version")
    monkeypatch.setattr(o.deps.planner, "plan_initial", original)
    if change == "prompt":
        load = module.load_prompt
        monkeypatch.setattr(module, "load_prompt", lambda n: load(n) + (" altered" if n == "planner_tool_discovery.md" else ""))
    else:
        monkeypatch.setenv("MECHCAL_AMESP_S1_NSTATES", "7")
    with pytest.raises(ValueError, match="differs"):
        o.run(smiles="CCO", user_query=NOTE, case_id="version", resume=True)


def test_shared_dependencies_are_not_hard_duplicates_and_all_exact_repeats_reported():
    c = ready_case()
    old = request(cid="microscopic.run_targeted_charge_redistribution_analysis", args={"s1_nstates": 1})
    histories = [dict(result_id=f"old:{i}", status="partial", capability_id=old.capability_id,
                      calculation_identity=calculation_identity(old, c, REGISTRY),
                      dependencies=REGISTRY.require(old.capability_id).metadata["execution_dependencies"])
                 for i in range(4)]
    c = replace_state(c, results=histories)
    candidate = action(c.case_id, 1, dispatch=True)
    for cid, args in [(old.capability_id, {"s1_nstates": 4}),
                      ("microscopic.run_ris_state_characterization", {"s1_nstates": 1})]:
        candidate.dispatch_requests = [request(cid=cid, args=args)]
        validate_action(candidate, c, REGISTRY)
    candidate.dispatch_requests = [old]
    with pytest.raises(RequestValidationError) as exc:
        validate_action(candidate, c, REGISTRY)
    assert all(f"old:{i}" in str(exc.value) for i in range(4))
    old.retry_reason = "Independent replication explicitly requested."
    validate_action(candidate, c, REGISTRY)


@pytest.mark.parametrize("cid,first,second", [
    ("microscopic.run_torsion_snapshots", {"dihedral_atoms": [0, 1, 2, 3], "angles": [-90, 90]},
     {"dihedral_atoms": [0, 1, 2, 3], "angles": [-60, 60]}),
    ("microscopic.run_solvation_polarity_proxy", {"solvents": ["water", "toluene"]},
     {"solvents": ["water", "acetonitrile"]}),
    ("microscopic.run_conformer_bundle", {"conformer_ids": ["c1", "c2"]},
     {"conformer_ids": ["c2", "c3"]}),
])
def test_distinct_scientific_parameters_are_not_duplicate_requests(cid, first, second):
    c = ready_case()
    old = request(cid=cid, args=first)
    new = request(cid=cid, args=second)
    old_identity = calculation_identity(old, c, REGISTRY)
    assert old_identity != calculation_identity(new, c, REGISTRY)
    c = replace_state(c, results=[{"result_id": "old", "calculation_identity": old_identity}])
    candidate = action(c.case_id, 1, dispatch=True)
    candidate.dispatch_requests = [new]
    validate_action(candidate, c, REGISTRY)


def test_resolved_geometry_identity_matches_explicit_source_and_tracks_new_geometry():
    c = ready_case()
    source = c.artifact_manifest.artifacts[-1].model_copy(update={
        "artifact_id": "baseline:first", "kind": "amesp_baseline_bundle",
    })
    c = c.touch(artifact_manifest=c.artifact_manifest.with_records([source]))
    req = request(cid="microscopic.run_targeted_state_characterization", args={"s1_nstates": 2})
    original = calculation_identity(req, c, REGISTRY)
    source_id = original["resolved_source_artifact_id"]
    explicit = req.model_copy(update={"tool_args": {**req.tool_args, "source_artifact_id": source_id},
                                      "input_refs": [source_id]})
    assert calculation_identity(explicit, c, REGISTRY) == original
    artifact = c.artifact_manifest.artifacts[-1].model_copy(update={"artifact_id": "geometry:second"})
    c = c.touch(artifact_manifest=c.artifact_manifest.with_records([artifact]))
    assert calculation_identity(req, c, REGISTRY) != original
    assert calculation_identity(explicit, c, REGISTRY) == original


def test_malformed_json_body_retained_and_not_treated_as_empty_success():
    from mechcal.runtime.retry import SchemaRetryRunner
    from mechcal.schemas.question_loop import QuestionAction

    seen = []
    def producer(feedback):
        if feedback:
            assert json.loads(feedback)["rejected_response"] == '{"action":'
        raise RejectedResponseError("Truncated JSON fixture", '{"action":')
    with pytest.raises(SchemaRetryError):
        SchemaRetryRunner(QuestionAction, event_sink=seen.append, repair_context=True).run(producer)
    assert len([e for e in seen if e["kind"] == "validation_error"]) == 3
    assert not any(e["kind"] == "accepted" for e in seen)


@pytest.mark.parametrize("failure", [TimeoutError("offline network failure"), OSError("offline disk failure"),
                                   KeyError("internal implementation error")])
def test_runtime_failures_do_not_spend_schema_retries(failure):
    from mechcal.runtime.question_loop import _failure_kind
    from mechcal.runtime.retry import SchemaRetryRunner
    from mechcal.schemas.question_loop import QuestionAction

    calls = []
    def produce(feedback):
        calls.append(feedback)
        raise failure
    with pytest.raises(type(failure)):
        SchemaRetryRunner(QuestionAction).run(produce)
    assert calls == [None]
    expected = ("model_or_transport_failure" if isinstance(failure, TimeoutError) else
                "storage_failure" if isinstance(failure, OSError) else "runtime_failure")
    assert _failure_kind(failure) == expected


def test_wrapped_report_retry_exhaustion_keeps_failure_category():
    from mechcal.runtime.question_loop import _failure_kind

    try:
        try:
            raise SchemaRetryError(["Invalid report fixture"])
        except SchemaRetryError as exc:
            raise RuntimeError("Worker report failed") from exc
    except RuntimeError as error:
        assert _failure_kind(error) == "request_retries_exhausted"


def test_same_dispatch_id_in_different_rounds_does_not_confuse_recovery(tmp_path, monkeypatch):
    from mechcal.runtime.persistence import RunStore

    o, _ = loop(tmp_path, rounds=2)
    class Client(QuestionClient):
        def complete_json(self, **kwargs):
            data = super().complete_json(**kwargs)
            for dispatch in data.get("dispatch_requests", []):
                dispatch["dispatch_id"] = "reused-id"
            return data
    o.deps.planner.llm_client = Client(rounds=2)
    worker = o.deps.workers["macro"]
    execute = worker.execute_tool
    calls = []
    def count(request, *args, **kwargs):
        calls.append(request.round_id)
        return execute(request, *args, **kwargs)
    monkeypatch.setattr(worker, "execute_tool", count)
    write = RunStore.write_execution_checkpoint
    def interrupt(store, request, **kwargs):
        write(store, request, **kwargs)
        if request.round_id == "R002" and kwargs["stage"] == "tool_returned":
            raise KeyboardInterrupt()
    monkeypatch.setattr(RunStore, "write_execution_checkpoint", interrupt)
    with pytest.raises(KeyboardInterrupt):
        o.run(smiles="CCO", user_query=NOTE, case_id="reused-id-case")
    monkeypatch.setattr(RunStore, "write_execution_checkpoint", write)
    result = o.run(smiles="CCO", user_query=NOTE, case_id="reused-id-case", resume=True)
    assert result.status == "finalized" and calls == ["R001", "R002"]
    assert len(state(result)["results"]) == len(result.evidence_ledger.items) == 2


def test_accepted_event_failure_is_not_a_model_validation_failure():
    from mechcal.runtime.retry import SchemaRetryRunner
    from mechcal.schemas.question_loop import QuestionAction

    calls = []
    def produce(feedback):
        calls.append(feedback)
        return action("test", 1)
    def sink(event):
        if event["kind"] == "accepted":
            raise ValueError("Diagnostic sink failed, not an invalid model response")
    with pytest.raises(ValueError, match="Diagnostic sink"):
        SchemaRetryRunner(QuestionAction, event_sink=sink).run(produce)
    assert calls == [None]


@pytest.mark.parametrize("tamper", ["case_id", "dispatch", "stage"])
def test_recovery_rejects_mismatched_execution_checkpoints(tmp_path, monkeypatch, tamper):
    from mechcal.runtime.persistence import RunStore

    o, _ = loop(tmp_path)
    normal = RunStore.write_execution_checkpoint
    def interrupt(store, request, **kwargs):
        normal(store, request, **kwargs)
        if kwargs["stage"] == "tool_returned":
            raise KeyboardInterrupt()
    monkeypatch.setattr(RunStore, "write_execution_checkpoint", interrupt)
    with pytest.raises(KeyboardInterrupt):
        o.run(smiles="CCO", user_query=NOTE, case_id="wrong-receipt")
    monkeypatch.setattr(RunStore, "write_execution_checkpoint", normal)
    receipt = next((tmp_path / "wrong-receipt/executions").glob("*.json"))
    contents = json.loads(receipt.read_text())
    contents[tamper] = "invalid"
    receipt.write_text(json.dumps(contents))
    def forbidden(*args, **kwargs):
        raise AssertionError("Mismatched receipt must not trigger execution")
    monkeypatch.setattr(o.deps.workers["macro"], "execute_tool", forbidden)
    with pytest.raises(ValueError, match="checkpoint"):
        o.run(smiles="CCO", user_query=NOTE, case_id="wrong-receipt", resume=True)
