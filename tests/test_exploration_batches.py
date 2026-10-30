"""Independent computations retain intent without relaxing scientific commits."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from mechcal.agents.command_tools import parse_tool_turn, replay_exchange, tool_contracts
from mechcal.runtime.command_experiments import finish_member, next_member, pending_experiments
from mechcal.runtime.command_loop import _apply, _comment, _execute, planner_payload, save
from mechcal.runtime.command_operations import OperationRejected, protocol, put, revise
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import ManageExperiments, ReviseJudgment
from test_command_protocol import REGISTRY, TOOL, case, command, computed, judgment, orchestrator, store
from test_native_commands import NativeFake, reply

SECOND = "macro.screen_donor_acceptor_architecture"


def compute_reply(cid, call_id, **args):
    name = next(t["name"] for t in tool_contracts(REGISTRY, [cid])[0] if t["name"].startswith("compute_"))
    return reply(name, {"arguments": command(args, tool=cid).arguments.model_dump(mode="json"),
                        "purpose": "Offline independent structural screen."}, call_id=call_id)


def batch_reply(*replies):
    return {**replies[0], "calls": [c for r in replies for c in r["calls"]],
            "items": [item for r in replies for item in r["items"]]}


def batch_case():
    return put(computed(), visible_tools=[TOOL, SECOND],
               loaded_tools={cid: REGISTRY.definition_version(cid) for cid in [TOOL, SECOND]})


def stage(o, c, s, incoming):
    tools, models = tool_contracts(REGISTRY, [TOOL, SECOND])
    parsed = parse_tool_turn(incoming, tools, models)
    return save(_apply(o, put(c, latest_exchange=incoming), parsed, s), s)


def drain(o, c, s):
    for _ in range(24):
        phase = protocol(c)["phase"]
        if phase == "choose":
            return c
        if phase == "batch_next":
            c = next_member(o, c)
        elif phase == "execute":
            c = _execute(o, c, s)
        elif phase == "explain":
            c = _comment(o, c, s)
            if protocol(c)["phase"] == "choose":
                c = finish_member(c)
        else:
            raise AssertionError(phase)
        c = save(c, s)
    raise AssertionError("Batch did not drain")


def test_batch_runs_two_real_macro_tools_without_planner_or_reviewer_calls(tmp_path):
    o, c = orchestrator(tmp_path), batch_case()
    s = store(tmp_path, c)
    incoming = batch_reply(compute_reply(TOOL, "a"), compute_reply(SECOND, "b"))
    c = stage(o, c, s, incoming)
    assert protocol(c)["phase"] == "batch_next" and protocol(c)["rounds"] == 0
    c = drain(o, c, s)
    p = protocol(c)
    assert p["rounds"] == 2 and len(p["results"]) == 2 and p["revision"] == 0
    assert not pending_experiments(c)
    assert set(p["outcome"]["call_results"]) == {"a", "b"}
    messages = replay_exchange(incoming, p["outcome"], "responses")
    assert {m["call_id"] for m in messages if m["type"] == "function_call_output"} == {"a", "b"}
    completed = json.loads((s.root / "operation_results" / (incoming["request_id"] + ".json")).read_text())
    assert completed["outcome"] == p["outcome"]
    assert {r["round_id"] for r in p["results"].values()} == {"R001", "R002"}


def test_bad_sibling_retained_and_good_computation_not_lost(tmp_path):
    o, c = orchestrator(tmp_path), batch_case()
    s = store(tmp_path, c)
    bad = compute_reply(SECOND, "bad")
    bad["calls"][0]["arguments"] = '{"arguments":'
    c = drain(o, stage(o, c, s, batch_reply(compute_reply(TOOL, "good"), bad)), s)
    assert protocol(c)["rounds"] == 1
    pending = pending_experiments(c)
    assert len(pending) == 1 and pending[0]["issues"][0]["code"] == "invalid_json"
    assert pending[0]["original_call"]["arguments"].endswith(":")
    c = _apply(o, c, ManageExperiments(kind="manage_experiments", action="cancel",
               experiment_ids=[pending[0]["experiment_id"]], reason="Replace malformed parameters."), s)
    assert not pending_experiments(c) and len(protocol(c)["results"]) == 1


def test_batch_never_guesses_missing_future_artifacts(tmp_path):
    o, c = orchestrator(tmp_path), batch_case()
    cid = "microscopic.run_baseline_bundle"
    c = put(c, visible_tools=[TOOL, cid], loaded_tools={**protocol(c)["loaded_tools"], cid: REGISTRY.definition_version(cid)})
    incoming = batch_reply(compute_reply(TOOL, "good"), compute_reply(cid, "future", source_artifact_id="future-geometry"))
    tools, models = tool_contracts(REGISTRY, [TOOL, cid])
    s = store(tmp_path, c)
    c = _apply(o, put(c, latest_exchange=incoming), parse_tool_turn(incoming, tools, models), s)
    pending = pending_experiments(c)
    assert next(r for r in pending if r["status"] == "blocked")["issues"][0]["code"] == "unknown_artifact"
    c = drain(o, c, s)
    assert protocol(c)["rounds"] == 1 and len(pending_experiments(c)) == 1


def test_batch_duplicate_reuses_result_and_checkpoint_resume_does_not_repeat(tmp_path):
    o, c = orchestrator(tmp_path), batch_case()
    s = store(tmp_path, c)
    incoming = batch_reply(compute_reply(TOOL, "a"), compute_reply(TOOL, "b"))
    c = stage(o, c, s, incoming)
    c = save(_execute(o, next_member(o, c), s), s)
    assert protocol(c)["phase"] == "explain"
    restored = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    c = drain(o, restored, s)
    assert protocol(c)["rounds"] == 1 and len(protocol(c)["results"]) == 1
    assert sorted(r["status"] for r in protocol(c)["experiments"].values()) == ["reused", "success"]


def test_ablation_failure_can_be_retried_only_after_permission_changes(tmp_path):
    o, c = orchestrator(tmp_path), batch_case()
    c = c.touch(runtime={**c.runtime, "ablation": {"enabled_capability_owners": ["microscopic"]}})
    s = store(tmp_path, c)
    incoming = batch_reply(compute_reply(TOOL, "a"), compute_reply(SECOND, "b"))
    c = drain(o, stage(o, c, s, incoming), s)
    ids = [r["experiment_id"] for r in pending_experiments(c)]
    assert len(ids) == 2 and protocol(c)["rounds"] == 0
    retry = ManageExperiments(kind="manage_experiments", action="retry", experiment_ids=ids)
    c = put(c, latest_exchange=reply("manage_experiments", retry.model_dump(exclude={"kind"}), call_id="retry1"))
    c = drain(o, _apply(o, c, retry, s), s)
    assert protocol(c)["rounds"] == 0 and len(pending_experiments(c)) == 2
    c = c.touch(runtime={**c.runtime, "ablation": {"enabled_capability_owners": ["macro"]}})
    c = put(c, latest_exchange=reply("manage_experiments", retry.model_dump(exclude={"kind"}), call_id="retry2"))
    c = drain(o, _apply(o, c, retry, s), s)
    assert protocol(c)["rounds"] == 2 and not pending_experiments(c)
    assert len(protocol(c)["outcome"]["call_results"]["retry2"]["experiments"]) == 2


def test_retry_unknown_or_completed_id_changes_nothing(tmp_path):
    o, c = orchestrator(tmp_path), batch_case()
    before = c.model_dump_json()
    with pytest.raises(OperationRejected, match="currently blocked"):
        _apply(o, c, ManageExperiments(kind="manage_experiments", action="cancel", experiment_ids=["foreign"]), store(tmp_path, c))
    assert c.model_dump_json() == before


def test_legacy_json_retry_does_not_require_a_native_call_id(tmp_path):
    o, c = orchestrator(tmp_path), batch_case()
    c = put(c, experiments={"saved": {"status": "blocked", "command": command().model_dump(mode="json"),
                                      "issues": [], "attempts": []}})
    s = store(tmp_path, c)
    c = _apply(o, c, ManageExperiments(kind="manage_experiments", experiment_ids=["saved"], action="retry"), s)
    c = drain(o, c, s)
    assert protocol(c)["rounds"] == 1 and not pending_experiments(c)


def test_typed_calculation_failure_does_not_cancel_independent_sibling(tmp_path, monkeypatch):
    o, c = orchestrator(tmp_path), batch_case()
    s = store(tmp_path, c)
    worker = o.deps.workers["macro"]
    original = worker.execute_tool
    def fail_first(request, case, **kw):
        result = original(request, case, **kw)
        if request.capability_id == TOOL:
            data = result.model_dump(mode="json")
            data.update(status="failed", artifact_updates=[], failure={"kind": "runtime_failed", "message": "Injected typed failure."})
            data["evidence_units"] = [{**data["evidence_units"][0], "status": "missing",
                                       "summary": "Injected diagnostic record, not positive evidence."}]
            return type(result).model_validate(data)
        return result
    monkeypatch.setattr(worker, "execute_tool", fail_first)
    c = drain(o, stage(o, c, s, batch_reply(compute_reply(TOOL, "fail"), compute_reply(SECOND, "good"))), s)
    assert protocol(c)["rounds"] == 2
    assert sorted(r["status"] for r in protocol(c)["results"].values()) == ["failed", "success"]
    failed = next(rid for rid, r in protocol(c)["results"].items() if r["status"] == "failed")
    assert all(e.source_report_id != failed for e in c.evidence_ledger.items)


def test_new_execution_limit_keeps_unstarted_member_pending(tmp_path):
    o, c = orchestrator(tmp_path), batch_case()
    s = store(tmp_path, c)
    c = stage(o, c, s, batch_reply(compute_reply(TOOL, "a"), compute_reply(SECOND, "b")))
    c = _execute(o, next_member(o, c), s)
    c = finish_member(_comment(o, c, s))
    c = c.touch(runtime={**c.runtime, "run_limits": {"scientific_tools_allowed": False}})
    c = drain(o, c, s)
    assert protocol(c)["rounds"] == 1
    assert pending_experiments(c)[0]["issues"][0]["code"] == "execution_limit"


def test_full_runtime_accepts_batch_and_pauses_before_dispatch_without_losing_intent(tmp_path):
    o = orchestrator(tmp_path)
    o.config = replace(o.config, command_native_tools=True, max_exploration_rounds=8)
    incoming = batch_reply(compute_reply(TOOL, "a"), compute_reply(SECOND, "b"))
    fake = NativeFake([reply("describe_tools", {"capability_ids": [TOOL, SECOND]}), incoming])
    original = fake.complete_tool_turn
    def stop(**kw):
        value = original(**kw)
        if len(fake.requests) == 2:
            (tmp_path / "queue-stop/STOP_REQUESTED").touch()
        return value
    fake.complete_tool_turn = stop
    o.deps.planner.llm_client = fake
    c = o.run(smiles="CCO", user_query="Offline queue fixture.", case_id="queue-stop")
    assert c.status == "paused" and protocol(c)["phase"] == "batch_next"
    assert len(pending_experiments(c)) == 2 and protocol(c)["rounds"] == 0


def test_candidate_records_do_not_force_ranking_and_science_edits_stay_atomic(tmp_path):
    c = revise(case(), judgment())
    label = "PACKING_HOST_MATRIX_CONFINEMENT"
    data = {"kind": "candidate", "label": label, "status": "considering", "note": "Unexamined alternative.", "refs": ["input:smiles"]}
    cmd = ReviseJudgment(kind="revise_judgment", changes=[data])
    updated = revise(c, cmd)
    assert protocol(updated)["ranking"] == protocol(c)["ranking"]
    overview = planner_payload(orchestrator(tmp_path), updated)["candidate_overview"]
    row = next(r for r in overview if r["label"] == label)
    assert not row["ranked"] and row["status"] == "considering"
    bad = deepcopy(data)
    bad["refs"] = ["foreign"]
    before = c.model_dump_json()
    with pytest.raises(OperationRejected):
        revise(c, ReviseJudgment(kind="revise_judgment", changes=[bad, *judgment().changes]))
    assert before == c.model_dump_json()
