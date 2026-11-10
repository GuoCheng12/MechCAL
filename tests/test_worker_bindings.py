"""Binding scope, not scientific ranking or numerical-method tests."""

from dataclasses import replace

import pytest
from jsonschema import Draft202012Validator

from mechcal.agents.command_tools import tool_contracts, parse_tool_turn
from mechcal.runtime import worker_tasks as tasks
from mechcal.runtime.command_operations import OperationRejected, protocol, put
from mechcal.schemas import CaseRun
from mechcal.schemas.artifacts import ArtifactRecord
from mechcal.schemas.worker_tasks import FixedParameter, DelegateComputation, WorkerTaskState
from test_command_protocol import REGISTRY, command
from test_native_commands import reply
from test_worker_tasks import setup, start


OPT = "microscopic.optimize_ground_state_geometry"
BASE = "microscopic.run_baseline_bundle"
HOLE = "microscopic.analyze_hole_electron_from_bundle"
CT = "microscopic.extract_ct_descriptors_from_bundle"
SELECT = "microscopic.select_computed_geometry"
FRONTIER = "microscopic.run_frontier_orbital_partition"


def binding(name, value, cid=None):
    return FixedParameter(name=name, value=value, capability_id=cid)


@pytest.mark.parametrize("bindings,code", [
    ([binding("source_artifact_id", "structure:initial")], "ambiguous_binding"),
    ([binding("max_wall_seconds", 120)], "ambiguous_binding"),
    ([binding("state_root", 11)], "invalid_binding_value"),
    ([binding("state_root", "1")], "invalid_binding_value"),
    ([binding("state_root", True)], "invalid_binding_value"),
    ([binding("source_artifact_id", "", OPT)], "invalid_binding_value"),
    ([binding("source_artifact_id", "x", "unknown")], "invalid_binding"),
    ([binding("state_root", 1, OPT)], "invalid_binding"),
    ([binding("state_root", 1), binding("state_root", 2, HOLE)], "conflicting_binding"),
    ([binding("state_root", 1), binding("state_root", 1)], "conflicting_binding"),
    ([binding("source_artifact_id", "x", OPT), binding("source_artifact_id", "y", OPT)], "conflicting_binding"),
])
def test_invalid_contract_returns_local_issue_before_worker_calls(tmp_path, bindings, code):
    o, c, s, contract = setup(tmp_path, owner="microscopic", caps=[OPT, BASE, HOLE, CT])
    contract.fixed_parameters = bindings
    before = c.model_dump_json()
    with pytest.raises(OperationRejected) as exc:
        tasks.begin(o, c, contract)
    assert exc.value.issues[0]["code"] == code
    assert exc.value.issues[0]["path"][:1] == ["fixed_parameters"]
    assert c.model_dump_json() == before
    assert not protocol(c).get("worker_task")
    assert not (s.root / "history/command_requests.jsonl").exists()


def test_distinct_tools_can_bind_same_parameter_to_distinct_inputs(tmp_path):
    o, c, s, contract = setup(tmp_path, owner="microscopic", caps=[OPT, BASE, HOLE])
    contract.fixed_parameters = [binding("source_artifact_id", "initial", OPT),
                                 binding("source_artifact_id", "prepared", BASE), binding("state_root", 1)]
    c = tasks.begin(o, c, contract)
    assert tasks.state(c).contract == contract


def test_single_tool_old_contract_remains_readable(tmp_path):
    o, c, s, contract = setup(tmp_path, owner="microscopic", caps=[HOLE])
    contract.fixed_parameters = [binding("source_artifact_id", "bundle"), binding("state_root", 1)]
    c = tasks.begin(o, c, contract)
    raw = tasks.state(c).model_dump()
    for item in raw["contract"]["fixed_parameters"]:
        item.pop("capability_id")
    old = WorkerTaskState.model_validate(raw)
    assert all(item.capability_id is None for item in old.contract.fixed_parameters)
    assert old.contract == contract


def test_global_scientific_condition_still_applies_to_every_exposed_tool(tmp_path, monkeypatch):
    caps = [HOLE, "microscopic.run_excited_state_relaxation"]
    o, c, s, contract = setup(tmp_path, owner="microscopic", caps=caps)
    contract.fixed_parameters = [binding("state_root", 1)]
    c = tasks.begin(o, c, contract)
    calls = []
    monkeypatch.setattr(tasks, "prepare_computation", lambda *a, **kw: calls.append(a) or (None, "validated"))
    for cid in caps:
        tasks.validate_computation(o, c, tasks.state(c), command({"source_artifact_id": "bundle", "state_root": 1}, cid))
        with pytest.raises(OperationRejected) as exc:
            tasks.validate_computation(o, c, tasks.state(c), command({"source_artifact_id": "bundle", "state_root": 2}, cid))
        assert exc.value.issues[0]["code"] == "fixed_condition"
    assert len(calls) == 2


@pytest.mark.parametrize("caps,locks,steps", [
    ([OPT, BASE, HOLE], [binding("source_artifact_id", "initial", OPT), binding("state_root", 1)],
     [(OPT, "initial"), (BASE, "new_geometry"), (HOLE, "new_bundle")]),
    ([SELECT, FRONTIER], [binding("source_artifact_id", "scan", SELECT), binding("member_id", "snapshot:06", SELECT)],
     [(SELECT, "scan"), (FRONTIER, "selected_geometry")]),
])
def test_upstream_binding_does_not_leak_into_downstream_call(tmp_path, monkeypatch, caps, locks, steps):
    o, c, s, contract = setup(tmp_path, owner="microscopic", caps=caps)
    contract.fixed_parameters = locks
    c = start(o, c, s, contract)
    # Persist/restore the contract across an execution checkpoint boundary.
    c = CaseRun.model_validate_json(c.model_dump_json())
    dispatched = []
    monkeypatch.setattr(tasks, "prepare_computation", lambda narrowed, cmd, *a, **kw:
                        dispatched.append((cmd.capability_id, cmd.arguments.source_artifact_id)) or (None, "validated"))
    for cid, source in steps:
        args = {"source_artifact_id": source}
        if cid == SELECT:
            args["member_id"] = "snapshot:06"
        if cid == FRONTIER:
            args.update(donor_atoms=[0], acceptor_atoms=[1])
        tasks.validate_computation(o, c, tasks.state(c), command(args, cid))
    assert dispatched == steps
    first_args = {"source_artifact_id": "substituted"}
    if caps[0] == SELECT:
        first_args["member_id"] = "snapshot:06"
    with pytest.raises(OperationRejected) as exc:
        tasks.validate_computation(o, c, tasks.state(c), command(first_args, caps[0]))
    assert exc.value.issues[0]["code"] == "fixed_condition"
    assert len(dispatched) == len(steps)


def test_new_artifact_access_requires_task_provenance(tmp_path):
    o, c, s, contract = setup(tmp_path, owner="microscopic", caps=[BASE])
    o.config = replace(o.config, enable_amesp=True)
    generated = ArtifactRecord(artifact_id="geometry:new", kind="prepared_structure", created_by_round="R001",
                               created_by_agent="microscopic")
    unrelated = generated.model_copy(update={"artifact_id": "geometry:unrelated"})
    c = c.touch(artifact_manifest=c.artifact_manifest.with_records([generated, unrelated]))
    c = tasks.begin(o, c, contract)
    t = tasks.state(c)
    t.loaded_tools = {BASE: REGISTRY.definition_version(BASE)}
    rid = "execution:task-local"
    c = put(c, results={rid: {"tool_result_id": rid, "status": "success", "capability_id": OPT,
                             "artifact_updates": [generated.model_dump(mode="json")]}})
    assert generated.artifact_id not in tasks.scope(c, t)
    t.result_ids.append(rid)
    c = tasks.retain(c, t)
    assert generated.artifact_id in tasks.scope(c, t)
    assert unrelated.artifact_id not in tasks.scope(c, t)
    for source in (unrelated.artifact_id, "other-case:geometry"):
        with pytest.raises(OperationRejected) as exc:
            tasks.validate_computation(o, c, t, command({"source_artifact_id": source}, BASE))
        assert exc.value.issues[0]["code"] == "unknown_artifact"


def test_planner_native_schema_exposes_binding_scope():
    tools, models = tool_contracts(REGISTRY, [], worker_tasks=True)
    schema = next(t["parameters"] for t in tools if t["name"] == "delegate_computation")
    Draft202012Validator.check_schema(schema)
    assert "capability_id" in str(schema)
    data = DelegateComputation(owner="microscopic", objective="Optimize then analyze.",
        input_refs=["input:smiles"], allowed_capability_ids=[OPT, BASE],
        fixed_parameters=[binding("source_artifact_id", "structure:initial", OPT)])
    parsed = parse_tool_turn(reply("delegate_computation", data.model_dump(exclude={"kind"})), tools, models)
    assert parsed == data
