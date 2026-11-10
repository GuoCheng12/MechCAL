"""Retry identity, local progress and restart acceptance; no external LLM calls."""

from copy import deepcopy
from dataclasses import replace
import pytest
from jsonschema import Draft202012Validator

from mechcal.agents.command_tools import parse_tool_turn, tool_contracts
from mechcal.runtime import MechCALOrchestrator, mcp_deps
from mechcal.runtime.command_loop import (
    LocalProgressError, _apply, _check_local_progress, _comment, _execute, save,
)
from mechcal.runtime.command_operations import OperationRejected, prepare_computation, protocol, put
from mechcal.schemas import CaseRun
from mechcal.schemas.failures import FailureReport
from mechcal.tools import StdioMcpServerConfig, StdioMcpTransport
from test_command_hypotheses import combined, enabled
from test_command_protocol import REGISTRY, TOOL, case, command, computed, judgment, orchestrator, store
from test_exploration_batches import batch_reply, drain, stage
from test_native_commands import NativeFake, reply


def prior_result(status="failed"):
    c = computed()
    request, key = prepare_computation(c, command(), REGISTRY, amesp_enabled=False)
    ref = "execution:fixture"
    result = {"tool_result_id": ref, "status": status, "dispatch_id": request.dispatch_id,
              "capability_id": TOOL, "failure": {"message": "Offline injected process failure."}}
    return put(c, results={ref: result}, actions={request.dispatch_id: request.model_dump(mode="json")},
               computations={key: ref}, rounds=1), ref, key


def retry_command(ref, **changes):
    return command().model_copy(update={"retry_of": ref, **changes})


@pytest.mark.parametrize("status", ["failed", "partial"])
def test_explicit_retry_has_new_stable_identity_but_same_parameters(status):
    c, ref, base = prior_result(status)
    original = deepcopy(c.model_dump())
    request, key = prepare_computation(c, retry_command(ref), REGISTRY, amesp_enabled=False)
    again, same_key = prepare_computation(c, retry_command(ref, purpose="Rephrased purpose"), REGISTRY, amesp_enabled=False)
    assert request.dispatch_id == again.dispatch_id and key == same_key != base
    assert request.dispatch_id not in protocol(c)["actions"]
    assert request.tool_args == command().arguments.model_dump(exclude_none=True)
    assert request.constraints["command_retry"] == {"result_id": ref, "base_digest": base}
    assert c.model_dump() == original


def test_purpose_text_never_triggers_retry(tmp_path):
    c, ref, _ = prior_result()
    c = _apply(orchestrator(tmp_path), c, command().model_copy(update={"purpose": "Retry this failed computation now."}), store(tmp_path, c))
    assert protocol(c)["phase"] == "choose"
    assert protocol(c)["outcome"]["result_status"] == "failed"
    assert protocol(c)["outcome"]["retry_of"] == ref
    assert protocol(c)["rounds"] == 1


@pytest.mark.parametrize("fault", ["unknown", "success", "parameters", "other_tool", "foreign", "stale", "budget", "ablation", "missing_action"])
def test_bad_retry_cannot_dispatch_or_mutate_state(fault):
    c, ref, base = prior_result()
    cmd = retry_command(ref)
    error = OperationRejected
    if fault == "unknown": cmd = retry_command("execution:other-case")
    if fault == "success": protocol(c)["results"][ref]["status"] = "success"
    if fault == "parameters": protocol(c)["actions"][next(iter(protocol(c)["actions"]))]["tool_args"] = {"unexpected": 1}
    if fault == "other_tool": protocol(c)["actions"][next(iter(protocol(c)["actions"]))]["capability_id"] = "macro.other"
    if fault == "foreign":
        protocol(c)["results"][ref]["case_id"] = "other-case"
        error = RuntimeError
    if fault == "missing_action":
        c = put(c, actions={})
        error = RuntimeError
    if fault == "stale": c = put(c, computations={base: "execution:newer"})
    if fault == "budget": c = c.touch(runtime={**c.runtime, "run_limits": {"scientific_tools_allowed": False}})
    if fault == "ablation": c = c.touch(runtime={**c.runtime, "ablation": {"enabled_capability_owners": []}})
    before = c.model_dump()
    with pytest.raises(error):
        prepare_computation(c, cmd, REGISTRY, amesp_enabled=False)
    assert c.model_dump() == before


def test_retry_checks_current_availability(monkeypatch):
    c, ref, _ = prior_result()
    monkeypatch.setattr(REGISTRY, "availability_violations", lambda *a, **k: ["Route disabled in this runtime"])
    with pytest.raises(OperationRejected, match="disabled"):
        prepare_computation(c, retry_command(ref), REGISTRY, amesp_enabled=False)


@pytest.mark.parametrize("explicit", [False, True])
def test_native_retry_schema_and_old_saved_calls(explicit):
    tools, models = tool_contracts(REGISTRY, [TOOL])
    tool = next(t for t in tools if t["name"].startswith("compute_"))
    args = {"arguments": command().arguments.model_dump(mode="json"), "purpose": "Offline request"}
    if explicit:
        args["retry_of"] = "execution:fixture"
    # Strict providers emit null explicitly; historical/non-strict calls may omit it.
    Draft202012Validator(tool["parameters"]).validate({"retry_of": None, **args})
    parsed = parse_tool_turn(reply(tool["name"], args), tools, models)
    assert parsed.retry_of == ("execution:fixture" if explicit else None)
    assert "retry_of" not in parsed.arguments.model_dump()


def test_real_local_mcp_retry_retains_failure_and_resumes_without_reexecution(tmp_path, monkeypatch):
    local = orchestrator(tmp_path)
    server = replace(StdioMcpServerConfig.current_python_server(), timeout_seconds=60)
    o = MechCALOrchestrator(local.config, mcp_deps(local.config, StdioMcpTransport(server)))
    c = computed()
    s = store(tmp_path, c)
    worker = o.deps.workers["macro"]
    execute = worker.execute_tool
    calls = []

    def injected_first_failure(*args, **kwargs):
        result = execute(*args, **kwargs)
        calls.append(result.dispatch_id)
        if len(calls) == 1:
            # Inject only status after a real MCP round trip, never a scientific finding.
            result = result.model_copy(update={"status": "failed", "failure": FailureReport(
                kind="runtime_failed", message="Injected transport completion failure")})
        return result

    monkeypatch.setattr(worker, "execute_tool", injected_first_failure)
    c = _execute(o, _apply(o, c, command(), s), s)
    ref = protocol(c)["pending_result"]
    failed = deepcopy(protocol(c)["results"][ref])
    assert not c.evidence_ledger.items
    c = _comment(o, c, s)
    pending = save(_apply(o, c, retry_command(ref), s), s)
    pending = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    completed = _execute(o, pending, s)
    # Replay the pre-commit checkpoint after the tool_returned checkpoint exists.
    recovered = _execute(o, pending, s)
    assert len(calls) == 2 and len(set(calls)) == 2
    assert protocol(recovered)["results"] == protocol(completed)["results"]
    assert protocol(recovered)["results"][ref] == failed
    assert protocol(recovered)["rounds"] == 2
    assert len(protocol(recovered)["results"]) == 2
    new_ref = protocol(recovered)["pending_result"]
    assert new_ref != ref and protocol(recovered)["results"][new_ref]["status"] == "success"
    recovered = _comment(o, recovered, s)
    for cmd in (command(), retry_command(ref)):
        replayed = _apply(o, recovered, cmd, s)
        assert protocol(replayed)["outcome"]["result_id"] == new_ref
        assert not protocol(replayed)["outcome"]["executed"]
        assert protocol(replayed)["phase"] == "choose"


def test_failed_retry_chain_requires_new_explicit_target(tmp_path, monkeypatch):
    c, ref, base = prior_result()
    o, s = orchestrator(tmp_path), store(tmp_path, c)
    worker = o.deps.workers["macro"]
    execute = worker.execute_tool
    def fail(*a, **kw):
        return execute(*a, **kw).model_copy(update={"status": "failed", "failure": FailureReport(
            kind="runtime_failed", message="Injected second failure")})
    monkeypatch.setattr(worker, "execute_tool", fail)
    c = _execute(o, _apply(o, c, retry_command(ref), s), s)
    latest = protocol(c)["pending_result"]
    assert protocol(c)["computations"][base] == latest
    assert prepare_computation(c, retry_command(ref), REGISTRY, amesp_enabled=False)[0] is None
    next_request, _ = prepare_computation(c, retry_command(latest), REGISTRY, amesp_enabled=False)
    assert next_request.dispatch_id != protocol(c)["results"][latest]["dispatch_id"]


def test_duplicate_retries_in_batch_execute_only_once(tmp_path):
    c, ref, _ = prior_result()
    o, s = orchestrator(tmp_path), store(tmp_path, c)
    name = "compute_" + TOOL.replace(".", "_")
    args = retry_command(ref).model_dump(mode="json", exclude={"kind", "capability_id"})
    incoming = batch_reply(reply(name, args, call_id="a"), reply(name, args, call_id="b"))
    c = drain(o, stage(o, c, s, incoming), s)
    assert protocol(c)["rounds"] == 2 and len(protocol(c)["results"]) == 2
    receipts = protocol(c)["outcome"]["call_results"]
    first, second = receipts["a"]["experiments"][0], receipts["b"]["experiments"][0]
    assert first["executed"] and not second["executed"]
    assert first["result_id"] == second["result_id"]
    assert second["result_status"] == "success"


def test_local_progress_ignores_turn_churn_but_allows_model_and_tool_work():
    c = put(case(), phase="hypothesis_next", hypothesis_step={"update_call_id": "u", "followup_calls": []})
    seen = set()
    _check_local_progress(c, seen)
    with pytest.raises(LocalProgressError):
        _check_local_progress(put(c, turn=10000), seen)
    for phase in ("choose", "execute", "explain", "review", "impact_review"):
        _check_local_progress(put(c, phase=phase), seen)
        _check_local_progress(c, seen)


def test_local_progress_detects_local_cycle():
    c = put(case(), phase="hypothesis_next", hypothesis_step={"update_call_id": "u", "followup_calls": []})
    seen = set()
    _check_local_progress(c, seen)
    _check_local_progress(put(c, phase="batch_next", compute_batch={"ids": [], "active": None}), seen)
    with pytest.raises(LocalProgressError):
        _check_local_progress(c, seen)


def test_stalled_loop_pauses_with_checkpoint_and_can_resume_after_fix(tmp_path, monkeypatch):
    import mechcal.runtime.command_loop as loop
    o = enabled(tmp_path)
    incoming = combined(reply("revise_judgment", judgment().model_dump(mode="json", exclude={"kind"}), call_id="formal"))
    fake = NativeFake([incoming, reply("finalize", {"answer": "Offline control-flow test.", "limitations": []}, call_id="final")])
    o.deps.planner.llm_client = fake
    followup = loop._hypothesis_followup
    attempts = []
    def stuck(o, c, s):
        attempts.append(1)
        return put(c, turn=protocol(c)["turn"] + 1)
    monkeypatch.setattr(loop, "_hypothesis_followup", stuck)
    with pytest.raises(LocalProgressError):
        o.run(smiles="CCO", user_query="Offline fault test.", case_id="stalled")
    checkpoint = tmp_path / "stalled/command_checkpoint.json"
    saved = CaseRun.model_validate_json(checkpoint.read_text())
    assert saved.status == "paused" and len(attempts) == 1 and len(fake.requests) == 1
    assert protocol(saved)["interruption"]["kind"] == "local_no_progress"
    assert protocol(saved)["hypothesis_step"]["update_call_id"] == "assess"
    monkeypatch.setattr(loop, "_hypothesis_followup", followup)
    result = o.run(smiles="CCO", user_query="Offline fault test.", case_id="stalled", resume=True)
    assert result.status == "finalized" and len(fake.requests) == 2
    assert protocol(result)["exploration"]["assessments"] == 1
    assert protocol(result)["rounds"] == 0


def test_combined_unchanged_judgment_returns_to_planner(tmp_path):
    from mechcal.runtime.command_loop import _hypothesis_followup
    from mechcal.runtime.command_operations import revise
    from test_command_hypotheses import parse
    c, o = revise(case(), judgment()), enabled(tmp_path)
    incoming = combined(reply("revise_judgment", judgment().model_dump(mode="json", exclude={"kind"}), call_id="formal"))
    c = put(c, latest_exchange=incoming)
    s = store(tmp_path, c)
    c = save(_apply(o, c, parse(incoming), s), s)
    c = save(_hypothesis_followup(o, c, s), s)
    assert protocol(c)["phase"] == "choose" and protocol(c)["revision"] == 1
    assert protocol(c)["outcome"]["call_results"]["formal"]["status"] == "unchanged"
