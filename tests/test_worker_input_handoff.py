"""Input navigation and permission handoff, independent of scientific predictions."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from mechcal.capabilities.registry import CapabilityRegistry
from mechcal.runtime import worker_tasks as tasks
from mechcal.runtime.command_compact import expand_text, table_rows
from mechcal.runtime.command_context import record_table
from mechcal.runtime.command_operations import OperationRejected, protocol, put
from mechcal.runtime.command_task_context import prepare_planner
from mechcal.schemas import CaseRun
from test_command_protocol import REGISTRY, case, command
from test_native_commands import reply
from test_worker_tasks import setup, start, step
from test_worker_bindings import OPT, BASE, HOLE, SELECT, with_inputs


def prepared(tmp_path, *, caps=None):
    o, c, s, contract = setup(tmp_path, owner="microscopic", caps=caps or [OPT, BASE, HOLE])
    c = c.touch(artifact_manifest=c.artifact_manifest.model_copy(update={"artifacts": []}))
    c = with_inputs(c, contract, ("structure:initial", "initial_structure"))
    return o, c, s, contract


def computation(cid, arguments, purpose="Inspect input"):
    data = command(arguments, cid).model_dump(mode="json", exclude={"kind", "capability_id"})
    data["purpose"] = purpose
    return reply("compute_" + cid.replace(".", "_"), data)


def test_missing_delegation_input_is_caught_before_any_worker_call(tmp_path):
    o, c, s, contract = prepared(tmp_path)
    contract.input_refs = ["input:smiles"]
    before = c.model_dump_json()
    with pytest.raises(OperationRejected) as exc:
        tasks.begin(o, c, contract)
    issue = exc.value.issues[0]
    assert issue["code"] == "delegation_input_gap" and issue["path"] == ["input_refs"]
    assert issue["details"]["candidate_inputs"] == [
        {"ref": "structure:initial", "kind": "initial_structure", "status": "available"}]
    assert c.model_dump_json() == before
    assert not (s.root / "history/command_requests.jsonl").exists()
    assert not protocol(c).get("worker_task")


def test_three_step_chain_needs_only_starting_input_and_does_not_compute(tmp_path, monkeypatch):
    o, c, s, contract = prepared(tmp_path, caps=[HOLE, BASE, OPT])
    monkeypatch.setattr(o.deps.capability_registry, "readiness", lambda *a, **k: pytest.fail("No chemistry preflight"))
    c = tasks.begin(o, c, contract)
    paths = tasks.task_input_paths(o, c, tasks.state(c))
    assert paths["input_ready"] == [OPT]
    assert set(paths["potentially_reachable"]) == {OPT, BASE, HOLE}
    assert set(tasks.scope(c, tasks.state(c))) == {"input:smiles", "structure:initial"}
    assert not protocol(c)["results"] and protocol(c)["rounds"] == 0


def test_no_automatic_source_selection_and_cross_case_references_rejected(tmp_path):
    o, c, s, contract = prepared(tmp_path)
    c = with_inputs(c, contract, ("structure:second", "initial_structure"))
    contract.input_refs = ["input:smiles"]
    with pytest.raises(OperationRejected) as exc:
        tasks.begin(o, c, contract)
    assert exc.value.issues[0]["details"]["candidate_input_count"] == 2
    contract.input_refs = ["foreign-case:structure"]
    with pytest.raises(OperationRejected):
        tasks.begin(o, c, contract)
    assert not protocol(c).get("worker_task")


@pytest.mark.parametrize("fault", ["missing", "failed", "disabled", "amesp_off", "cycle"])
def test_no_unreachable_or_disabled_route_becomes_a_starting_path(tmp_path, fault):
    o, c, s, contract = prepared(tmp_path)
    if fault in {"missing", "failed"}:
        artifacts = [a.model_copy(update={"status": fault}) for a in c.artifact_manifest.artifacts]
        c = c.touch(artifact_manifest=c.artifact_manifest.model_copy(update={"artifacts": artifacts}))
    if fault == "amesp_off":
        o.config = replace(o.config, enable_amesp=False)
    if fault == "disabled":
        o.deps.capability_registry = CapabilityRegistry([s.model_copy(update={"backend_enabled": False})
            for s in o.deps.capability_registry.snapshot().capabilities])
    if fault == "cycle":
        # Neither member of a dependency cycle supplies the initial input.
        o.deps.capability_registry = CapabilityRegistry([
            REGISTRY.require(OPT).model_copy(update={"required_artifact_kinds": ["amesp_baseline_bundle"]}),
            REGISTRY.require(BASE)])
        contract.allowed_capability_ids = [OPT, BASE]
    with pytest.raises(OperationRejected) as exc:
        tasks.begin(o, c, contract)
    assert exc.value.issues[0]["code"] == "delegation_input_gap"


def test_alternative_source_kind_is_sufficient_and_partial_inputs_remain_usable(tmp_path):
    o, c, s, contract = setup(tmp_path, owner="microscopic", caps=[SELECT, BASE])
    c = with_inputs(c, contract, ("scan:one", "torsion_snapshot_bundle"))
    a = c.artifact_manifest.artifacts[-1].model_copy(update={"status": "partial"})
    c = c.touch(artifact_manifest=c.artifact_manifest.with_records([a]))
    c = tasks.begin(o, c, contract)
    paths = tasks.task_input_paths(o, c, tasks.state(c))
    assert SELECT in paths["input_ready"] and BASE in paths["potentially_reachable"]


def test_unreachable_optional_tool_hands_back_retaining_finished_results(tmp_path):
    o, c, s, contract = prepared(tmp_path, caps=[OPT, HOLE])
    c = start(o, c, s, contract)
    t = tasks.state(c)
    rid = "execution:retained"
    t.result_ids = [rid]
    t.loaded_tools = {HOLE: o.deps.capability_registry.definition_version(HOLE)}
    c = tasks.retain(put(c, results={rid: {"tool_result_id": rid, "capability_id": OPT,
        "status": "success", "artifact_updates": [], "evidence_units": []}}), t)
    before = deepcopy(protocol(c)["results"])
    c, _ = step(o, c, s, computation(HOLE, {"source_artifact_id": None, "state_root": 1}))
    assert protocol(c)["phase"] == "choose" and protocol(c)["worker_task"] is None
    report = protocol(c)["outcome"]
    assert report["completion_reason"] == "input_scope_gap" and report["status"] == "partial"
    assert report["source_result_ids"] == [rid] and report["summary_role"] == "runtime_input_handoff"
    assert protocol(c)["results"] == before and not protocol(c)["ranking"]
    assert not protocol(c)["explanations"]  # Runtime handoff is not a Worker scientific interpretation.


def test_identical_missing_input_after_reads_and_resume_yields_to_planner(tmp_path):
    o, c, s, contract = prepared(tmp_path)
    c = start(o, c, s, contract)
    t = tasks.state(c)
    t.loaded_tools = {OPT: REGISTRY.definition_version(OPT)}
    c = tasks.retain(c, t)
    def bad(purpose):
        return computation(OPT, {"source_artifact_id": None}, purpose)
    c, _ = step(o, c, s, bad("First attempt"))
    assert tasks.state(c).input_error and tasks.state(c).tool_attempts == 0
    c, _ = step(o, c, s, reply("read_evidence", {"records": [{"ref": "input:smiles"}]}))
    c = CaseRun.model_validate_json(c.model_dump_json())
    c, _ = step(o, c, s, bad("Same absent input, rephrased purpose"))
    assert protocol(c)["outcome"]["completion_reason"] == "repeated_input_error"
    assert protocol(c)["outcome"]["tool_attempts"] == 0
    assert not protocol(c)["results"] and protocol(c)["phase"] == "choose"


def test_corrected_input_can_execute_without_automatic_source_substitution(tmp_path, monkeypatch):
    o, c, s, contract = prepared(tmp_path)
    c = start(o, c, s, contract)
    t = tasks.state(c)
    t.loaded_tools = {OPT: REGISTRY.definition_version(OPT)}
    c = tasks.retain(c, t)
    c, _ = step(o, c, s, computation(OPT, {"source_artifact_id": None}))
    seen = []
    def prepare(narrowed, cmd, *a, **kw):
        seen.append(cmd.arguments.source_artifact_id)
        # Stop before executing any chemistry; the corrected operation reached execution validation.
        raise OperationRejected([{"path": ["arguments"], "code": "offline_marker", "message": "Offline fixture",
                                  "remedy": "", "record_id": None}])
    monkeypatch.setattr(tasks, "prepare_computation", prepare)
    c, _ = step(o, c, s, computation(OPT, {"source_artifact_id": "structure:initial"}))
    assert seen == ["structure:initial"] and protocol(c)["phase"] == "worker_task"


def test_changed_input_state_does_not_trigger_identical_error_handoff(tmp_path):
    o, c, s, contract = prepared(tmp_path)
    c = start(o, c, s, contract)
    t = tasks.state(c)
    t.loaded_tools = {OPT: REGISTRY.definition_version(OPT)}
    c = tasks.retain(c, t)
    c, _ = step(o, c, s, computation(OPT, {"source_artifact_id": None}))
    old = tasks.state(c).input_error["fingerprint"]
    changed = c.artifact_manifest.artifacts[0].model_copy(update={"status": "partial"})
    c = c.touch(artifact_manifest=c.artifact_manifest.with_records([changed]))
    c, _ = step(o, c, s, computation(OPT, {"source_artifact_id": None}))
    assert tasks.state(c).input_error["fingerprint"] != old
    assert protocol(c)["phase"] == "worker_task" and not protocol(c)["results"]


def test_bound_but_undelegated_source_is_not_silently_authorized(tmp_path):
    from mechcal.schemas.worker_tasks import FixedParameter
    o, c, s, contract = prepared(tmp_path)
    contract.input_refs = ["input:smiles"]
    contract.fixed_parameters = [FixedParameter(name="source_artifact_id", value="structure:initial", capability_id=OPT)]
    with pytest.raises(OperationRejected) as exc:
        tasks.begin(o, c, contract)
    assert exc.value.issues[0]["code"] == "binding_input_scope"
    assert not protocol(c).get("worker_task")


def test_old_task_state_without_error_memory_can_resume(tmp_path):
    from mechcal.schemas.worker_tasks import WorkerTaskState
    o, c, s, contract = prepared(tmp_path)
    c = start(o, c, s, contract)
    raw = tasks.state(c).model_dump(mode="json")
    raw.pop("input_error")
    restored = WorkerTaskState.model_validate(raw)
    assert restored.input_error is None
    assert tasks.task_input_paths(o, c, restored)["input_ready"] == [OPT]


@pytest.mark.parametrize("recent_refs", [[], [f"artifact:{i}" for i in range(20, 26)]])
def test_input_index_is_small_visible_and_full_directory_stays_exact(recent_refs):
    c = case()
    c = put(c, planner_read_handles=[{"ref": ref} for ref in recent_refs])
    rows = [{"ref": "structure:initial" if i == 0 else f"artifact:{i}",
             "kind": "initial_structure" if i == 0 else "prepared_structure", "status": "available",
             "metadata": "coordinates-not-for-the-working-view" * 100} for i in range(100)]
    full = {"artifact_index": record_table(rows)}
    before = c.model_dump_json()
    saved, view = prepare_planner(c, full)
    view = expand_text(view)
    shown = table_rows(view["artifact_index"])
    assert len(shown) <= 12 and any(r["ref"] == "structure:initial" for r in shown)
    assert set(recent_refs).issubset({r["ref"] for r in shown})
    assert any(r["ref"] == "artifact:99" for r in shown)
    assert all(set(r) == {"ref", "kind", "status"} for r in shown)
    assert "coordinates-not-for-the-working-view" not in json.dumps(view)
    assert protocol(saved)["planner_context"]["fields"]["artifact_index"] == full["artifact_index"]
    assert c.model_dump_json() == before
    assert view["artifact_index_counts"] == {"total": 100, "shown": len(shown)}
