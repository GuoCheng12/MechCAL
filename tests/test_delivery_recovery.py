"""Whole delivery paths with injected interface faults, never simulated science."""

from copy import deepcopy
from dataclasses import replace

import pytest

from mechcal.agents.command_tools import ask_tools, replay_exchange, tool_contracts
from mechcal.runtime import command_hypotheses as hypotheses
from mechcal.runtime import command_recovery as recovery
from mechcal.runtime.command_action_policy import action_policy
from mechcal.runtime.command_loop import _apply, _comment, _local_error, planner_payload
from mechcal.runtime.command_operations import OperationRejected, protocol, put, check_refs
from mechcal.runtime.command_task_context import prepare_planner
from mechcal.schemas.commands import AmendFinal, Finalize, ReadEvidence, ReviewFinal, SubmitFinal
from test_command_task_context import ready
from test_native_commands import NativeFake, reply


def exhausted(case):
    for call_id in ("first", "second"):
        case = hypotheses.record_read_attempt(case, reply(call_id=call_id))
    assert hypotheses.view(case)["query_turns_remaining"] == 0
    return case


def science(case):
    return deepcopy({"ledger": case.evidence_ledger.model_dump(mode="json"),
                     "artifacts": case.artifact_manifest.model_dump(mode="json"),
                     **{k: protocol(case).get(k) for k in ("results", "arguments", "ranking", "rounds", "revision")}})


@pytest.mark.parametrize("compact", [False, True])
def test_workspace_read_remedy_is_executable_with_exhausted_science(tmp_path, compact):
    c, o, s = ready(tmp_path)
    o.config = replace(o.config, command_compact_context=compact, command_task_context=compact)
    c = exhausted(c)
    original = science(c)
    draft = protocol(c)["final_draft"]
    with pytest.raises(OperationRejected, match="match exactly") as error:
        _apply(o, c, AmendFinal(kind="amend_final", draft_id=draft["draft_id"], replacements=[
            {"old_text": "not in this draft", "new_text": "Do not guess."}]), s)
    assert "read_evidence" in error.value.issues[0]["remedy"]
    payload = planner_payload(o, c, native=True)
    if compact:
        c, payload = prepare_planner(c, payload)
    incoming = reply(arguments={"records": [{"ref": "delivery:answer", "paths": ["/answer"]}]})
    fake = NativeFake([incoming])
    cmd, exchange = ask_tools(fake, o.deps.capability_registry, [], payload, s)
    tool = next(t for t in fake.requests[0]["tools"] if t["name"] == "read_evidence")
    assert "delivery:answer" in tool["parameters"]["$defs"]["ReadRecord"]["properties"]["ref"]["enum"]
    c = _apply(o, put(c, latest_exchange=exchange), cmd, s)
    assert protocol(c)["outcome"]["records"][0]["data"]["/answer"] == draft["command"]["answer"]
    assert science(c) == original
    assert hypotheses.view(c)["query_turns_remaining"] == 0
    c = _apply(o, c, AmendFinal(kind="amend_final", draft_id=draft["draft_id"], replacements=[
        {"old_text": draft["command"]["answer"], "new_text": "A corrected bounded comparison."}]), s)
    assert protocol(c)["phase"] == "review"
    assert science(c) == original


def test_workspace_reads_neither_spend_nor_bypass_source_allowance(tmp_path):
    c, o, s = ready(tmp_path)
    before = hypotheses.view(c)["query_turns_remaining"]
    exchange = reply(arguments={"records": [{"ref": "delivery:answer", "paths": []}]})
    c = _apply(o, put(c, latest_exchange=exchange), ReadEvidence(
        kind="read_evidence", records=[{"ref": "delivery:answer"}]), s)
    assert hypotheses.view(c)["query_turns_remaining"] == before
    c = exhausted(c)
    c = put(c, latest_exchange=None)
    with pytest.raises(OperationRejected, match="queries.*exhausted"):
        _apply(o, c, ReadEvidence(kind="read_evidence", records=[{"ref": "input:smiles"}]), s)
    with pytest.raises(OperationRejected):
        check_refs(c, ["delivery:answer"], ["refs"], scientific=True)


def test_workspace_scope_is_case_local_and_unknown_reference_is_not_returned(tmp_path):
    c, o, s = ready(tmp_path)
    c = exhausted(c)
    assert "foreign:answer" not in action_policy(c).workspace_refs
    with pytest.raises(OperationRejected):
        _apply(o, c, ReadEvidence(kind="read_evidence", records=[{"ref": "foreign:answer"}]), s)
    bad = put(c, case_id="foreign")
    with pytest.raises(RuntimeError, match="ownership"):
        action_policy(bad)


def test_unavailable_review_retry_uses_saved_draft_and_preserves_its_contents(tmp_path):
    c, o, s = ready(tmp_path)
    draft = deepcopy(protocol(c)["final_draft"])
    original = science(c)
    c = put(c, final_review={"draft_id": draft["draft_id"], "status": "unavailable"})
    with pytest.raises(OperationRejected) as error:
        _apply(o, c, SubmitFinal(kind="submit_final", draft_id=draft["draft_id"]), s)
    assert "review_final" in error.value.issues[0]["remedy"]
    for _ in range(3):
        c = _apply(o, c, ReviewFinal(kind="review_final", draft_id=draft["draft_id"]), s)
        assert protocol(c)["final_draft"] == draft and protocol(c)["phase"] == "review"
    assert science(c) == original
    with pytest.raises(OperationRejected):
        _apply(o, c, ReviewFinal(kind="review_final", draft_id="foreign"), s)


@pytest.mark.parametrize("fault", ["extra_read", "truncated", "no_call", "bad_json"])
def test_review_local_fault_then_completion_keeps_scope_and_science(tmp_path, fault):
    c, o, s = ready(tmp_path)
    original = science(c)
    first = reply(call_id="read1")
    second = reply(call_id="read2")
    broken = reply(call_id="bad")
    if fault == "truncated":
        broken["complete"] = False
    elif fault == "no_call":
        broken["calls"] = []
    elif fault == "bad_json":
        broken = reply("record_review", {}, call_id="bad")
        broken["calls"][0]["arguments"] = '{"text":'
    completed = reply("record_review", {"text": "Offline protocol fixture; scientific limits remain.", "findings": []}, call_id="record")
    fake = NativeFake([first, second, broken, completed])
    o.deps.scientific_reviewer.llm_client = fake
    for _ in range(3):
        c = _comment(o, c, s)
        assert protocol(c)["phase"] == "review"
    session = protocol(c)["reviewer_session"]
    assert session["queries"] == 2 and session["corrections"] == 1
    assert "input:smiles" in session["read_record_ids"]
    c = _comment(o, c, s)
    assert protocol(c)["final_review"]["status"] == "available"
    assert science(c) == original
    assert fake.requests[-1]["outcome"]["status"] == "rejected"
    replay_exchange(fake.requests[-1]["exchange"], fake.requests[-1]["outcome"], "responses")
    draft = protocol(c)["final_draft"]
    c = _apply(o, c, SubmitFinal(kind="submit_final", draft_id=draft["draft_id"]), s)
    assert c.status == "finalized" and c.final_answer is not None


def test_exhausted_review_retains_read_session_for_explicit_retry(tmp_path):
    c, o, s = ready(tmp_path)
    fake = NativeFake([reply(call_id=f"attempt{i}") for i in range(5)])
    o.deps.scientific_reviewer.llm_client = fake
    for _ in range(5):
        c = _comment(o, c, s)
    assert protocol(c)["phase"] == "choose"
    assert protocol(c)["final_review"]["status"] == "unavailable"
    assert c.final_answer is None
    c = _apply(o, c, ReviewFinal(kind="review_final", draft_id=protocol(c)["final_draft"]["draft_id"]), s)
    session = protocol(c)["reviewer_session"]
    assert session["queries"] == 2 and session["corrections"] == 0
    assert "input:smiles" in session["read_record_ids"]


def test_delivery_budget_does_not_reset_on_retries_reads_or_rewording(tmp_path):
    c, o, s = ready(tmp_path)
    original = science(c)
    for i in range(12):
        c = recovery.before_model_turn(c)
        c = _apply(o, c, Finalize(kind="finalize", answer=f"Offline protocol narrative {i}."), s)
        assert recovery.state(c).model_turns == i + 1
    with pytest.raises(recovery.DeliveryRecoveryLimit):
        recovery.before_model_turn(c)
    assert science(c) == original
    # Only a genuinely new calculation identity can begin another episode.
    c = put(c, results={**protocol(c)["results"], "offline:new-result": {
        "status": "failed", "summary": "Protocol fixture, not a physical result."}})
    assert recovery.state(recovery.begin(c)).model_turns == 0


def test_model_operation_policy_matches_runtime_with_all_compaction_modes(tmp_path):
    c, o, s = ready(tmp_path)
    c = exhausted(c)
    policy = action_policy(c)
    tools, models = policy.filter_tools(*tool_contracts(o.deps.capability_registry, [], review_workflow=True))
    assert {"read_evidence", "review_final", "submit_final"}.issubset(models)
    assert [tool["name"] for tool in tools] == list(models)
    for ref in policy.workspace_refs:
        result = _apply(o, c, ReadEvidence(kind="read_evidence", records=[{"ref": ref, "view": "directory"}]), s)
        assert protocol(result)["outcome"]["status"] == "read"


def test_disabled_tracking_does_not_advertise_a_stale_read_limit(tmp_path):
    from mechcal.runtime.command_compact import expand_text

    c, o, s = ready(tmp_path)
    c = put(exhausted(c), latest_exchange=None)
    o.config = replace(o.config, command_hypothesis_tracking=False, command_evidence_followup=False,
                       command_review_workflow=False, command_task_context=False)
    payload = expand_text(planner_payload(o, c, native=True))
    assert payload["operation_policy"]["scientific_reads_remaining"] is None
    policy = action_policy(c, scientific_read_limit_enabled=False)
    tools, _ = policy.filter_tools(*tool_contracts(o.deps.capability_registry, []))
    read_tool = next(tool for tool in tools if tool["name"] == "read_evidence")
    assert "enum" not in read_tool["parameters"]["$defs"]["ReadRecord"]["properties"]["ref"]
    current = _apply(o, c, ReadEvidence(kind="read_evidence", records=[{"ref": "input:smiles"}]), s)
    assert protocol(current)["outcome"]["status"] == "read"


def test_local_rejection_does_not_reset_delivery_episode(tmp_path):
    c, _, s = ready(tmp_path)
    c = recovery.before_model_turn(c)
    issue = {"path": ["draft_id"], "code": "review_unavailable", "message": "Not reviewed.", "remedy": "review_final"}
    c = _local_error(c, [issue], s)
    assert recovery.state(c).model_turns == 1


def test_configured_delivery_allowance_keeps_existing_episode_count(tmp_path):
    c, o, s = ready(tmp_path)
    o.config = replace(o.config, command_delivery_max_model_turns=30)
    c = put(c, delivery_recovery=None)
    c = _apply(o, c, Finalize(kind="finalize", answer="Offline allowance fixture."), s)
    original = science(c)
    assert recovery.state(c).max_model_turns == 30
    for i in range(30):
        c = recovery.before_model_turn(c, max_model_turns=30)
        assert recovery.state(c).model_turns == i + 1
        # A repeated finalization or a changed constructor default must not refill it.
        c = _apply(o, c, Finalize(kind="finalize", answer=f"Offline narrative {i}."), s)
        c = recovery.begin(c, max_model_turns=100)
        assert recovery.state(c).max_model_turns == 30
    with pytest.raises(recovery.DeliveryRecoveryLimit):
        recovery.before_model_turn(c, max_model_turns=30)
    assert science(c) == original


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_configured_delivery_allowance(value, tmp_path):
    from mechcal.runtime import OrchestratorConfig

    with pytest.raises(ValueError, match="positive integer"):
        OrchestratorConfig(run_base_dir=tmp_path, command_delivery_max_model_turns=value)


def test_production_loop_pauses_and_retains_state_at_delivery_limit(tmp_path):
    from mechcal.runtime.command_loop import run_command_loop, save
    from mechcal.runtime.question_loop import configuration_fingerprint
    from mechcal.schemas import CaseRun

    c, o, s = ready(tmp_path)
    original = science(c)
    fake = NativeFake([reply(call_id=f"read{i}") for i in range(10)])
    o.deps.scientific_reviewer.llm_client = fake
    c = put(c, fingerprint=configuration_fingerprint(o, c), delivery_recovery={
        **protocol(c)["delivery_recovery"], "max_model_turns": 2})
    save(c, s)
    with pytest.raises(recovery.DeliveryRecoveryLimit):
        run_command_loop(o, c, s, resume=True)
    after = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    assert after.status == "paused" and after.final_answer is None
    assert protocol(after)["interruption"]["kind"] == "delivery_recovery_limit"
    assert len(fake.requests) == 2 and science(after) == original


def test_review_retry_after_failed_corrections_can_complete_without_refilling_reads(tmp_path):
    c, o, s = ready(tmp_path)
    completed = reply("record_review", {"text": "Protocol fixture; remaining limits are explicit.", "findings": []}, call_id="ok")
    fake = NativeFake([reply(call_id=f"r{i}") for i in range(5)] + [completed])
    o.deps.scientific_reviewer.llm_client = fake
    for _ in range(5):
        c = _comment(o, c, s)
    draft = protocol(c)["final_draft"]
    c = _apply(o, c, ReviewFinal(kind="review_final", draft_id=draft["draft_id"]), s)
    c = _comment(o, c, s)
    assert protocol(c)["final_review"] == {"draft_id": draft["draft_id"], "status": "available"}
    assert not any(t["name"] == "read_evidence" for t in fake.requests[-1]["tools"])
