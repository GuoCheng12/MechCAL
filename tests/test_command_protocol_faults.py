"""Fault boundaries and real local-MCP acceptance for the command interface."""

import json
from dataclasses import replace

import pytest

from mechcal.agents.command_client import ask, schema_issues
from mechcal.runtime import MechCALOrchestrator, mcp_deps
from mechcal.runtime.command_loop import _apply, _comment, _execute, save
from mechcal.runtime.command_operations import OperationRejected, protocol, put, revise
from mechcal.runtime.command_schema import native_format
from mechcal.runtime.llm import JsonCompletionResult
from mechcal.schemas.commands import DescribeTools, envelope_model
from mechcal.schemas.failures import FailureReport
from mechcal.tools import StdioMcpServerConfig, StdioMcpTransport
from test_command_protocol import (
    REGISTRY, TOOL, ScriptClient, case, command, computed, judgment, orchestrator, store, wire,
)


def test_valid_json_with_truncated_status_is_not_committed(tmp_path):
    model = envelope_model(REGISTRY, [])
    class Truncated:
        def complete_json_result(self, **kw):
            return JsonCompletionResult(data={}, raw_text=json.dumps(wire(model, {
                "command": judgment().model_dump(mode="json")})), finish_reason="length",
                usage={}, parse_error="Incomplete response")
    with pytest.raises(OperationRejected, match="Incomplete"):
        ask(Truncated(), model, {}, store(tmp_path, case()))


def test_known_schema_errors_point_to_command_field(tmp_path):
    with pytest.raises(OperationRejected) as error:
        ask(ScriptClient([{"command": {"kind": "finalize", "answer": "", "limitations": []}}]),
            envelope_model(REGISTRY, []), {}, store(tmp_path, case()))
    assert error.value.issues[0]["path"] == ["command", "answer"]


def test_native_schema_exposes_same_mechanism_pool_as_runtime():
    from mechcal.mechanism_pool import MECHANISM_POOL
    fmt = native_format(envelope_model(REGISTRY, []))
    definitions = fmt["json_schema"]["schema"]["$defs"]
    for name in ("ArgumentChange", "RankingChange"):
        assert definitions[name]["properties"]["label"]["enum"] == list(MECHANISM_POOL)


def test_root_and_nested_label_errors_reported_together():
    data = wire(envelope_model(REGISTRY, []), {"command": judgment().model_dump(mode="json")})
    data["command_kind"] = "unexpected"
    data["command"]["changes"][0]["label"] = "A narrative, not a canonical label"
    issues = schema_issues(native_format(envelope_model(REGISTRY, []))["json_schema"]["schema"], data)
    assert ["command_kind"] in [i["path"] for i in issues]
    assert ["command", "changes", 0, "label"] in [i["path"] for i in issues]


def test_ordinary_rejections_do_not_interrupt_the_run(tmp_path):
    from mechcal.runtime.command_loop import _local_error
    c = case().touch(status="running")
    s = store(tmp_path, c)
    for _ in range(5):
        c = _local_error(c, [{"path": ["command"], "code": "schema",
                             "message": "Invalid command", "remedy": "Correct or choose another operation."}], s)
    assert c.status == "running"
    assert protocol(c)["revision"] == 0
    assert protocol(c)["local_errors"] == 5


def test_duplicate_revision_has_no_scientific_side_effect_or_extra_review(tmp_path):
    o, c = orchestrator(tmp_path, reviews=True), revise(case(), judgment())
    updated = _apply(o, c, judgment(), store(tmp_path, c))
    assert protocol(updated)["revision"] == 1
    assert protocol(updated)["phase"] == "choose"
    assert protocol(updated)["outcome"]["status"] == "unchanged"


def test_receipt_is_published_only_after_checkpoint_commit(tmp_path, monkeypatch):
    c, o = case(), orchestrator(tmp_path)
    s = store(tmp_path, c)
    pending = _apply(o, c, judgment(), s)
    assert not list((s.root / "operations").glob("*.json"))
    original = s.write_json
    def fail_checkpoint(path, value):
        if path == "command_checkpoint.json":
            raise OSError("Injected storage failure")
        return original(path, value)
    monkeypatch.setattr(s, "write_json", fail_checkpoint)
    with pytest.raises(OSError):
        save(pending, s)
    assert not list((s.root / "operations").glob("*.json"))
    monkeypatch.setattr(s, "write_json", original)
    save(pending, s)
    save(pending, s)
    assert len(list((s.root / "operations").glob("*.json"))) == 1


def test_json_in_parameter_string_is_rejected(tmp_path):
    with pytest.raises(OperationRejected) as error:
        ask(ScriptClient([{"command": {"kind": "compute", "capability_id": TOOL,
            "arguments": "{}", "purpose": "test"}}]), envelope_model(REGISTRY, [TOOL]),
            {}, store(tmp_path, case()))
    assert error.value.issues[0]["path"] == ["command", "arguments"]


def test_scientific_argument_cannot_self_cite_or_cite_an_action():
    c = revise(case(), judgment())
    update = judgment().model_copy(deep=True)
    update.changes[0].refs = ["A1"]
    with pytest.raises(OperationRejected, match="not a raw observation"):
        revise(c, update)


def test_metadata_displays_native_fragment_objects(tmp_path):
    o, c = orchestrator(tmp_path), case()
    cid = "microscopic.analyze_hole_electron_from_bundle"
    loaded = _apply(o, c, DescribeTools(kind="describe_tools", capability_ids=[cid]), store(tmp_path, c))
    card = protocol(loaded)["outcome"]["tools"][0]
    assert "tool_arg_schema" not in card
    assert card["arguments_schema"]["properties"]["fragments"]["anyOf"][0]["type"] == "array"
    assert protocol(loaded)["rounds"] == 0


def test_second_description_batch_keeps_first_batch_executable(tmp_path):
    o, c = orchestrator(tmp_path), case()
    s = store(tmp_path, c)
    c = _apply(o, c, DescribeTools(kind="describe_tools", capability_ids=[TOOL]), s)
    c = _apply(o, c, DescribeTools(kind="describe_tools",
        capability_ids=["microscopic.analyze_hole_electron_from_bundle"]), s)
    model = envelope_model(REGISTRY, protocol(c)["visible_tools"])
    assert model.model_validate({"command": command().model_dump(mode="json")}).command.capability_id == TOOL


def test_failed_calculation_persisted_but_not_promoted(tmp_path, monkeypatch):
    o, c = orchestrator(tmp_path), computed()
    s = store(tmp_path, c)
    c = _apply(o, c, command(), s)
    worker = o.deps.workers["macro"]
    original = worker.execute_tool
    def fail(*a, **kw):
        result = original(*a, **kw)
        # Fault injection: even if a failing adapter mistakenly returns ordinary
        # observations, the runtime must not promote them to accepted evidence.
        return result.model_copy(update={
            "status": "failed", "failure": FailureReport(kind="runtime_failed", message="Injected failure"),
        })
    monkeypatch.setattr(worker, "execute_tool", fail)
    result = _execute(o, c, s)
    saved = protocol(result)["results"]
    assert len(saved) == 1 and next(iter(saved.values()))["status"] == "failed"
    assert not result.evidence_ledger.items


def test_unknown_execution_completion_cannot_replay(tmp_path):
    o, c = orchestrator(tmp_path), computed()
    s = store(tmp_path, c)
    c = _apply(o, c, command(), s)
    from mechcal.schemas import DispatchRequest
    req = DispatchRequest.model_validate(protocol(c)["ticket"]["request"])
    plan = o.deps.workers["macro"].execution_planner.deterministic_plan(
        agent_name="macro", request=req, case_run=c)
    s.write_execution_checkpoint(req, stage="executing", plan=plan)
    with pytest.raises(RuntimeError, match="unknown"):
        _execute(o, c, s)


def test_tool_returned_checkpoint_reused_before_case_commit(tmp_path, monkeypatch):
    o, c = orchestrator(tmp_path), computed()
    s = store(tmp_path, c)
    c = _apply(o, c, command(), s)
    completed = _execute(o, c, s)
    # Simulate a crash after result persistence but before case checkpoint commit.
    monkeypatch.setattr(o.deps.workers["macro"], "execute_tool",
                        lambda *a, **k: pytest.fail("Must not repeat the saved computation"))
    recovered = _execute(o, c, s)
    assert protocol(recovered)["results"] == protocol(completed)["results"]
    assert protocol(recovered)["rounds"] == 1


def test_commentary_unknown_refs_remains_advisory(tmp_path):
    o, c = orchestrator(tmp_path, reviews=True), revise(case(), judgment())
    s = store(tmp_path, c)
    c = put(c, phase="review", review_role="reviewer",
            review_changes=[c.model_dump(mode="json") for c in judgment().changes])
    before = c.portfolio.model_dump()
    o.deps.photophysics_arbiter.llm_client = ScriptClient(["A comment mentioning unverified foreign:E1."])
    reviewed = _comment(o, c, s)
    assert protocol(reviewed)["review"]["status"] == "available"
    assert "foreign:E1" not in protocol(reviewed)["review"]["reviewed_record_ids"]
    assert "foreign:E1" not in protocol(reviewed)["results"]
    assert reviewed.portfolio.model_dump() == before


def test_stop_after_current_compute_saves_and_never_dispatches_next(tmp_path, monkeypatch):
    o = orchestrator(tmp_path)
    tool_model = envelope_model(REGISTRY, [TOOL])
    o.deps.planner.llm_client = ScriptClient([
        wire(envelope_model(REGISTRY, []), {"command": {"kind": "describe_tools", "capability_ids": [TOOL]}}),
        wire(tool_model, {"command": command().model_dump(mode="json")}),
    ])
    worker = o.deps.workers["macro"]
    original = worker.execute_tool
    def compute_then_stop(*a, **kw):
        result = original(*a, **kw)
        (tmp_path / "stop-test/STOP_REQUESTED").touch()
        return result
    monkeypatch.setattr(worker, "execute_tool", compute_then_stop)
    c = o.run(smiles="CCO", user_query="Rank mechanisms.", case_id="stop-test")
    assert c.status == "paused" and protocol(c)["phase"] == "explain"
    assert len(protocol(c)["results"]) == 1
    assert protocol(c)["interruption"]["kind"] == "user_stop"
    assert len(o.deps.planner.llm_client.calls) == 2


def test_mcp_execution_and_command_final_delivery(tmp_path):
    local = orchestrator(tmp_path)
    server = replace(StdioMcpServerConfig.current_python_server(), timeout_seconds=60)
    deps = mcp_deps(local.config, StdioMcpTransport(server))
    tool_model = envelope_model(deps.capability_registry, [TOOL])
    deps.planner.llm_client = ScriptClient([
        wire(envelope_model(deps.capability_registry, []), {"command": {"kind": "describe_tools", "capability_ids": [TOOL]}}),
        wire(tool_model, {"command": command().model_dump(mode="json")}),
        wire(tool_model, {"command": judgment().model_dump(mode="json")}),
        wire(tool_model, {"command": {"kind": "finalize", "answer": "Offline protocol test, not a mechanism conclusion."}}),
    ])
    result = MechCALOrchestrator(local.config, deps).run(smiles="CCO", user_query="Test protocol.", case_id="mcp-test")
    assert result.status == "finalized"
    assert protocol(result)["rounds"] == 1
    assert len(protocol(result)["results"]) == 1
    assert result.evidence_ledger.items
    assert protocol(result)["arguments"]["A1"]["text"] in result.mechanism_predictions[0].evidence
