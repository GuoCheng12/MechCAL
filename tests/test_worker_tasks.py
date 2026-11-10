"""Task ownership/recovery tests. Scripted choices are not model competence."""

from copy import deepcopy
from dataclasses import replace
import json
import time

import pytest

from mechcal.agents import worker_task as agent
from mechcal.agents.command_tools import parse_tool_turn, replay_exchange, tool_contracts
from mechcal.runtime.command_loop import _apply, _execute, save, planner_payload
from mechcal.runtime.command_operations import OperationRejected, protocol, put
from mechcal.runtime import worker_tasks as tasks
from mechcal.schemas import CaseRun
from mechcal.schemas.artifacts import ArtifactRecord
from mechcal.schemas.commands import DescribeTools, envelope_model
from mechcal.schemas.failures import FailureReport
from mechcal.schemas.worker_tasks import DelegateComputation, CompleteWorkerTask, ReadWorkerLog, FixedParameter
from test_command_protocol import REGISTRY, TOOL, case, command, orchestrator, store, judgment
from test_native_commands import NativeFake, reply


def setup(tmp_path, *, refs=None, caps=None, owner="macro", **config):
    c = case().touch(status="running")
    o = orchestrator(tmp_path)
    o.config = replace(o.config, command_native_tools=True, command_worker_tasks=True, **config)
    s = store(tmp_path, c)
    contract = DelegateComputation(owner=owner, objective="Obtain a checked structural observation.",
        input_refs=refs or ["input:smiles"], allowed_capability_ids=caps or [TOOL])
    return o, c, s, contract


def start(o, c, s, contract):
    parent = reply("delegate_computation", contract.model_dump(exclude={"kind"}), call_id="parent")
    return save(_apply(o, put(c, latest_exchange=parent), contract, s), s)


def step(o, c, s, response):
    client = NativeFake([response])
    o.deps.workers[tasks.state(c).contract.owner].report_writer.llm_client = client
    return save(tasks.turn(o, c, s), s), client


def compute_reply(**updates):
    return reply("compute_" + TOOL.replace(".", "_"),
        {**command().model_dump(mode="json", exclude={"kind", "capability_id"}), **updates}, call_id="worker-compute")


def load(o, c, s, caps=None):
    return step(o, c, s, reply("describe_tools", {"capability_ids": caps or [TOOL]}, call_id="worker-docs"))[0]


def test_opt_in_schemas_and_planner_contract(tmp_path):
    o, c, s, contract = setup(tmp_path)
    assert "delegate_computation" not in tool_contracts(REGISTRY, [])[1]
    tools, models = tool_contracts(REGISTRY, [], worker_tasks=True)
    parsed = parse_tool_turn(reply("delegate_computation", contract.model_dump(exclude={"kind"})), tools, models)
    restored = envelope_model(REGISTRY, [], worker_tasks=True).model_validate({"command": parsed.model_dump()})
    assert restored.command == contract
    assert planner_payload(o, c, native=True)["worker_delegation"]["enabled"]
    tools, models = agent.contracts(REGISTRY, [TOOL], NativeFake([]))
    assert not {"revise_judgment", "finalize", "update_hypotheses", "delegate_computation"} & models.keys()




def test_worker_real_local_compute_and_atomic_parent_delivery(tmp_path):
    o, c, s, contract = setup(tmp_path)
    c = start(o, c, s, contract)
    before = deepcopy(protocol(c)["latest_exchange"])
    c = load(o, c, s)
    assert protocol(c)["rounds"] == 0
    assert protocol(c).get("loaded_tools", {}) == {}
    c, _ = step(o, c, s, compute_reply())
    assert protocol(c)["phase"] == "execute"
    c = _execute(o, c, s)
    task = tasks.state(c)
    assert protocol(c)["phase"] == "worker_task" and protocol(c)["rounds"] == 1
    assert len(task.result_ids) == 1
    assert protocol(c)["latest_exchange"] == before and not protocol(c)["ranking"]
    rid = task.result_ids[0]
    c, _ = step(o, c, s, reply("complete_worker_task", {
        "status": "complete", "refs": [rid], "summary": "Structural observations returned; no mechanism is proven."}))
    assert protocol(c)["phase"] == "choose" and protocol(c)["worker_task"] is None
    result = protocol(c)["outcome"]
    assert result["status"] == "complete" and result["tool_attempts"] == 1
    assert result["model_calls"] == 3
    assert protocol(c)["revision"] == 0
    assert protocol(c)["explanations"][rid]["text"] == result["summary"]
    assert (s.root / "operation_results/planner-parent.json").exists()
    replay = replay_exchange(before, result, "responses")
    assert replay[-1]["call_id"] == "parent" and rid in replay[-1]["output"]


def test_failure_is_retained_and_only_worker_can_request_retry(tmp_path, monkeypatch):
    o, c, s, contract = setup(tmp_path)
    c = load(o, start(o, c, s, contract), s)
    worker = o.deps.workers["macro"]
    real_execute = worker.execute_tool
    calls = []

    def injected(*a, **kw):
        result = real_execute(*a, **kw)
        calls.append(result.dispatch_id)
        if len(calls) == 1:
            return result.model_copy(update={"status": "failed", "failure": FailureReport(
                kind="runtime_failed", message="Offline injected transient failure", recoverable=True)})
        return result

    monkeypatch.setattr(worker, "execute_tool", injected)
    c, _ = step(o, c, s, compute_reply())
    c = _execute(o, c, s)
    failed = tasks.state(c).result_ids[0]
    assert len(calls) == 1 and not c.evidence_ledger.items
    saved_failure = deepcopy(protocol(c)["results"][failed])
    c, _ = step(o, c, s, compute_reply(retry_of=failed))
    before_compute = CaseRun.model_validate_json(c.model_dump_json())
    c = _execute(o, c, s)
    resumed = _execute(o, before_compute, s)
    assert len(calls) == 2  # tool_returned checkpoint does not execute again
    assert tasks.state(resumed).result_ids == tasks.state(c).result_ids
    assert protocol(c)["results"][failed] == saved_failure
    assert len(c.evidence_ledger.items) > 0 and len(tasks.state(c).result_ids) == 2
    c, _ = step(o, c, s, compute_reply(retry_of=failed))
    assert protocol(c)["phase"] == "worker_task" and len(calls) == 2
    assert tasks.state(c).outcome["executed"] is False


@pytest.mark.parametrize("fault", ["truncated", "unknown", "missing", "wrong_scope", "ranking", "no_results"])
def test_invalid_operations_stay_local_and_do_not_reset_allowance(tmp_path, fault):
    o, c, s, contract = setup(tmp_path)
    c = start(o, c, s, contract)
    bad = {
        "truncated": reply("describe_tools", {"capability_ids": [TOOL]}, complete=False),
        "unknown": reply("compute_unknown", {}),
        "missing": reply("describe_tools", {}),
        "wrong_scope": reply("read_evidence", {"records": [{"ref": "foreign-case:evidence"}]}),
        "ranking": reply("revise_judgment", judgment().model_dump(exclude={"kind"})),
        "no_results": reply("complete_worker_task", {"status": "complete", "refs": [], "summary": "Done"}),
    }[fault]
    c, _ = step(o, c, s, bad)
    task = tasks.state(c)
    assert task.outcome["status"] == "rejected" and task.model_calls == 1
    assert task.tool_attempts == 0 and protocol(c)["phase"] == "worker_task"
    assert not protocol(c)["ranking"] and protocol(c)["revision"] == 0
    c = load(o, c, s)
    assert tasks.state(c).model_calls == 2 and tasks.state(c).outcome["status"] == "described"


def test_changed_physical_state_rejected_and_foreign_source_rejected(tmp_path):
    tool = "microscopic.analyze_hole_electron_from_bundle"
    o, c, s, contract = setup(tmp_path, owner="microscopic", caps=[tool])
    contract.fixed_parameters = [FixedParameter(name="state_root", value=1)]
    c = load(o, start(o, c, s, contract), s, [tool])
    model = envelope_model(REGISTRY, [tool])
    for root, code in [(2, "fixed_condition"), (1, "unknown_artifact")]:
        cmd = model.model_validate({"command": {"kind": "compute", "capability_id": tool,
            "arguments": {"source_artifact_id": "other-case:bundle", "state_root": root}, "purpose": "Inspect S1"}}).command
        with pytest.raises(OperationRejected) as exc:
            tasks.validate_computation(o, c, tasks.state(c), cmd)
        assert exc.value.issues[0]["code"] == code


@pytest.mark.parametrize("fault", ["disabled_mode", "wrong_owner", "ablation", "foreign_ref", "bad_lock"])
def test_bad_delegations_do_not_create_state(tmp_path, fault):
    o, c, s, contract = setup(tmp_path)
    if fault == "disabled_mode": o.config = replace(o.config, command_worker_tasks=False)
    if fault == "wrong_owner": contract.owner = "microscopic"
    if fault == "ablation": c = c.touch(runtime={**c.runtime, "ablation": {"enabled_capability_owners": []}})
    if fault == "foreign_ref": contract.input_refs = ["other-case:source"]
    if fault == "bad_lock": contract = contract.model_copy(update={"fixed_parameters": [] , "allowed_capability_ids": ["unknown"]})
    with pytest.raises(OperationRejected):
        tasks.begin(o, c, contract)
    assert not protocol(c).get("worker_task")


def test_task_limit_delivers_blocked_and_cross_case_resume_fails(tmp_path):
    o, c, s, contract = setup(tmp_path, worker_task_max_calls=1)
    c = load(o, start(o, c, s, contract), s)
    crossed = c.model_copy(update={"case_id": "another-case"})
    with pytest.raises(RuntimeError, match="another case"):
        tasks.state(crossed)
    c = save(tasks.turn(o, c, s), s)
    assert protocol(c)["phase"] == "choose"
    assert protocol(c)["outcome"]["status"] == "blocked"
    assert protocol(c)["outcome"]["completion_reason"] == "model_call_limit"
    assert not protocol(c)["results"]


def test_partial_result_survives_operational_time_limit(tmp_path, monkeypatch):
    o, c, s, contract = setup(tmp_path)
    c = load(o, start(o, c, s, contract), s)
    worker = o.deps.workers["macro"]
    execute = worker.execute_tool
    monkeypatch.setattr(worker, "execute_tool", lambda *a, **kw: execute(*a, **kw).model_copy(update={
        "status": "partial", "failure": FailureReport(kind="runtime_failed", message="Some observations unavailable")}))
    c, _ = step(o, c, s, compute_reply())
    c = _execute(o, c, s)
    task = tasks.state(c)
    task.deadline = time.time() - 1
    c = save(tasks.turn(o, tasks.retain(c, task), s), s)
    assert protocol(c)["outcome"]["status"] == "partial"
    assert protocol(c)["outcome"]["usable_result_ids"]
    assert c.evidence_ledger.items


def test_bounded_registered_log_read_and_no_arbitrary_path(tmp_path):
    o, c, s, contract = setup(tmp_path)
    path = tmp_path / "run.log"
    path.write_text("diagnostic" * 1000)
    artifact = ArtifactRecord(artifact_id="fixture:log", kind="diagnostic", created_by_round="R001",
        created_by_agent="macro", file_paths={"stdout": str(path)})
    c = c.touch(artifact_manifest=c.artifact_manifest.with_records([artifact]))
    contract.input_refs.append(artifact.artifact_id)
    c = start(o, c, s, contract)
    read = ReadWorkerLog(record_id=artifact.artifact_id, file_key="stdout", limit=10)
    result = tasks.read_log(c, tasks.state(c), read)
    assert len(result["text"]) == 10 and result["next_offset"] == 10
    with pytest.raises(OperationRejected):
        tasks.read_log(c, tasks.state(c), read.model_copy(update={"file_key": "/etc/passwd"}))
    path.unlink()
    path.symlink_to("/etc/passwd")
    with pytest.raises(OperationRejected, match="symbolic"):
        tasks.read_log(c, tasks.state(c), read)


def test_transport_failure_preserves_call_charge_and_completed_results(tmp_path):
    o, c, s, contract = setup(tmp_path)
    c = start(o, c, s, contract)
    class Broken:
        settings = None
        def complete_tool_turn(self, **kw): raise TimeoutError("Offline transport injection")
    o.deps.workers["macro"].report_writer.llm_client = Broken()
    with pytest.raises(TimeoutError):
        tasks.turn(o, c, s)
    saved = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    assert tasks.state(saved).model_calls == 1
    assert protocol(saved)["latest_exchange"] == protocol(c)["latest_exchange"]


def combine(*replies):
    first = deepcopy(replies[0])
    first["calls"] = [c for r in replies for c in r["calls"]]
    first["items"] = [c for r in replies for c in r["items"]]
    return first


def test_read_only_batch_keeps_valid_reads_and_correlates_every_call(tmp_path):
    o, c, s, contract = setup(tmp_path)
    c = start(o, c, s, contract)
    response = combine(reply("describe_tools", {"capability_ids": [TOOL]}, call_id="d"),
                       reply(call_id="r"),
                       reply("read_evidence", {"records": [{"ref": "outside"}]}, call_id="bad"))
    c, client = step(o, c, s, response)
    task = tasks.state(c)
    assert task.model_calls == 1 and task.tool_attempts == 0
    assert task.loaded_tools[TOOL] == REGISTRY.definition_version(TOOL)
    assert {k: v["status"] for k, v in task.outcome["call_results"].items()} == {
        "d": "described", "r": "read", "bad": "rejected"}
    assert len(replay_exchange(task.exchange, task.outcome, "responses")) == 6
    assert protocol(c)["rounds"] == 0 and protocol(c)["revision"] == 0


def test_mixed_compute_read_batch_executes_nothing(tmp_path):
    o, c, s, contract = setup(tmp_path)
    c = load(o, start(o, c, s, contract), s)
    c, _ = step(o, c, s, combine(compute_reply(), reply(call_id="read")))
    assert tasks.state(c).outcome["status"] == "rejected"
    assert protocol(c)["phase"] == "worker_task" and not protocol(c)["results"]
    assert tasks.state(c).tool_attempts == 0


@pytest.mark.parametrize("ref_kind", ["result", "evidence", "artifact"])
def test_every_visible_result_record_uses_one_completion_reference_contract(tmp_path, ref_kind):
    o, c, s, contract = setup(tmp_path)
    c = load(o, start(o, c, s, contract), s)
    c, _ = step(o, c, s, compute_reply())
    c = _execute(o, c, s)
    rid = tasks.state(c).result_ids[0]
    if ref_kind == "artifact":
        artifact = ArtifactRecord(artifact_id="fixture:generated", kind="diagnostic", created_by_round="R001",
                                  created_by_agent="macro")
        results = deepcopy(protocol(c)["results"])
        results[rid]["artifact_updates"].append(artifact.model_dump())
        c = put(c.touch(artifact_manifest=c.artifact_manifest.with_records([artifact])), results=results)
        ref = artifact.artifact_id
    else:
        ref = rid if ref_kind == "result" else protocol(c)["results"][rid]["evidence_units"][0]["evidence_id"]
    c, _ = step(o, c, s, reply("complete_worker_task", {"status": "complete", "refs": [ref],
                        "summary": "Task observation is recorded under the cited source. This is not mechanism approval."}))
    assert protocol(c)["phase"] == "choose"
    assert protocol(c)["outcome"]["refs"] == [ref]
    assert protocol(c)["outcome"]["source_result_ids"] == [rid]


def test_registered_failed_result_log_can_be_read_without_success_artifact(tmp_path):
    o, c, s, contract = setup(tmp_path)
    logdir = tmp_path / "failed_job"
    logdir.mkdir()
    (logdir / "spatial.log").write_text("numeric check failed")
    failure = {"tool_result_id": "fixture:failure", "status": "failed", "capability_id": TOOL,
               "failure": {"details": {"workdir": str(logdir)}}}
    c = put(c, results={"fixture:failure": failure})
    contract.input_refs.append("fixture:failure")
    c = start(o, c, s, contract)
    data = tasks.read_log(c, tasks.state(c), ReadWorkerLog(record_id="fixture:failure", file_key="spatial.log"))
    assert data["text"] == "numeric check failed"
    with pytest.raises(OperationRejected):
        tasks.read_log(c, tasks.state(c), ReadWorkerLog(record_id="fixture:failure", file_key="../unrelated.txt"))


@pytest.mark.parametrize("constraint", ["backend_disabled", "ablation", "global_limit", "task_limit"])
def test_loading_description_never_overrides_execution_permissions(tmp_path, constraint):
    o, c, s, contract = setup(tmp_path)
    c = load(o, start(o, c, s, contract), s)
    if constraint == "backend_disabled":
        from mechcal.capabilities.registry import CapabilityRegistry
        spec = o.deps.capability_registry.require(TOOL).model_copy(update={"backend_enabled": False,
            "disabled_reason": "Offline test: unavailable backend"})
        o.deps.capability_registry = CapabilityRegistry([spec])
        task = tasks.state(c)
        task.loaded_tools[TOOL] = o.deps.capability_registry.definition_version(TOOL)
        c = tasks.retain(c, task)
    if constraint == "ablation":
        c = c.touch(runtime={**c.runtime, "ablation": {"enabled_capability_owners": []}})
    if constraint == "global_limit":
        c = c.touch(runtime={**c.runtime, "run_limits": {"scientific_tools_allowed": False}})
    if constraint == "task_limit":
        task = tasks.state(c)
        task.tool_attempts = o.config.worker_task_max_computations
        c = tasks.retain(c, task)
    with pytest.raises(OperationRejected):
        tasks.validate_computation(o, c, tasks.state(c), command())
    assert not protocol(c)["results"]


def test_main_loop_delegates_and_receives_one_completed_worker_report(tmp_path):
    o, _, _, contract = setup(tmp_path)
    planner = NativeFake([
        reply("delegate_computation", contract.model_dump(exclude={"kind"}), call_id="delegate"),
        reply("revise_judgment", judgment().model_dump(exclude={"kind"}), call_id="judgment"),
        reply("finalize", {"answer": "Offline integration fixture.", "limitations": []}, call_id="final"),
    ])
    class Worker:
        settings = None
        def __init__(self): self.requests = []
        def complete_tool_turn(self, **kw):
            self.requests.append(kw)
            if len(self.requests) == 1:
                return reply("describe_tools", {"capability_ids": [TOOL]}, call_id="docs")
            if len(self.requests) == 2:
                return compute_reply()
            rid = kw["payload"]["attempts"][-1]["result_id"]
            return reply("complete_worker_task", {"status": "complete", "refs": [rid],
                "summary": "Computed structural features. These do not prove a mechanism."}, call_id="done")
    worker = Worker()
    o.deps.planner.llm_client = planner
    o.deps.workers["macro"].report_writer.llm_client = worker
    completed = o.run(smiles="CCO", user_query="Offline execution contract.", case_id="worker-loop")
    assert completed.status == "finalized" and protocol(completed)["rounds"] == 1
    assert len(worker.requests) == 3 and len(planner.requests) == 3
    received = planner.requests[1]["outcome"]
    assert received["completion_reason"] == "worker_report" and received["model_calls"] == 3
    assert received["all_attempt_result_ids"] == list(protocol(completed)["results"])
    assert len(list((tmp_path / "worker-loop/operation_results").glob("*.json"))) == 3


def test_failed_computation_can_be_delivered_as_blocked_without_fabrication(tmp_path, monkeypatch):
    o, c, s, contract = setup(tmp_path)
    c = load(o, start(o, c, s, contract), s)
    worker = o.deps.workers["macro"]
    execute = worker.execute_tool
    monkeypatch.setattr(worker, "execute_tool", lambda *a, **kw: execute(*a, **kw).model_copy(update={
        "status": "unsupported", "failure": FailureReport(kind="capability_unsupported", message="Offline injected unsupported element")}))
    c, _ = step(o, c, s, compute_reply())
    c = _execute(o, c, s)
    rid = tasks.state(c).result_ids[0]
    c, _ = step(o, c, s, reply("complete_worker_task", {"status": "blocked", "refs": [rid],
        "summary": "Backend cannot handle this element. No valid evidence was obtained."}))
    assert protocol(c)["outcome"]["status"] == "blocked"
    assert protocol(c)["outcome"]["completion_reason"] == "worker_report"
    assert not protocol(c)["outcome"]["usable_result_ids"] and not c.evidence_ledger.items
