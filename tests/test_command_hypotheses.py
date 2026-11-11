"""Provisional beliefs are durable decisions, never generated scientific evidence."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest
from pydantic import ValidationError

from mechcal.agents.command_tools import ask_tools, parse_tool_turn, replay_exchange, tool_contracts
from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime import command_hypotheses as h
from mechcal.runtime.command_experiments import finish_member, next_member
from mechcal.runtime.command_loop import _apply, _comment, _execute, _hypothesis_followup, _local_error, planner_payload, save
from mechcal.runtime.command_operations import OperationRejected, protocol, put
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import ReadEvidence, UpdateHypotheses, envelope_model
from test_command_protocol import LABELS, REGISTRY, TOOL, case, command, computed, judgment, orchestrator, store
from test_native_commands import NativeFake, reply
from test_command_handoff import prose


def enabled(tmp_path):
    o = orchestrator(tmp_path)
    o.config = replace(o.config, command_native_tools=True, command_hypothesis_tracking=True)
    return o


def update(confidence=.4, label=LABELS[0], **extra):
    return UpdateHypotheses(kind="update_hypotheses", changes=[{
        "label": label, "confidence": confidence, "note": "Offline state fixture, not a scientific result.",
        "refs": ["input:smiles"], **extra}], note="Initial provisional assessment.")


def combined(followup=None, assessment=None):
    first = reply("update_hypotheses", (assessment or update()).model_dump(mode="json", exclude={"kind"}), call_id="assess")
    name = "compute_" + TOOL.replace(".", "_")
    other = followup or reply(name, {"arguments": command().arguments.model_dump(mode="json"),
                                    "purpose": "Offline computation."}, call_id="compute")
    return {**first, "calls": first["calls"] + other["calls"], "items": first["items"] + other["items"]}


def parse(incoming):
    tools, models = tool_contracts(REGISTRY, [TOOL], hypothesis_tracking=True)
    return parse_tool_turn(incoming, tools, models)


def test_initial_unknowns_are_not_zero_and_current_state_is_separate(tmp_path):
    c, o = case(), enabled(tmp_path)
    view = planner_payload(o, c)["exploration_state"]
    assert view["assessment_required"] and view["confidence_ranking"] == []
    assert len(view["unassessed"]) == len(MECHANISM_POOL)
    assert all(row["confidence"] is None for row in view["unassessed"])
    before = c.model_dump()
    c = _apply(o, c, update(), store(tmp_path, c))
    assert h.view(c)["confidence_ranking"][0]["confidence"] == .4
    assert protocol(c)["phase"] == "choose"
    for key in ("ranking", "arguments", "revision", "rounds", "results"):
        assert protocol(c)[key] == before["runtime"]["command_protocol"][key]
    assert c.portfolio.model_dump() == before["portfolio"]


def test_add_lower_reset_and_ties_do_not_force_top_three_or_normalization():
    c, _ = h.update(case(), update(.8))
    c, _ = h.update(c, update(.8, LABELS[1]))
    assert [r["rank"] for r in h.view(c)["confidence_ranking"]] == [1, 1]
    assert sum(r["confidence"] for r in h.view(c)["confidence_ranking"]) == 1.6
    c, _ = h.update(c, update(.2))
    assert h.view(c)["confidence_ranking"][0]["label"] == LABELS[1]
    c, _ = h.update(c, update(None, LABELS[1]))
    assert h.view(c)["confidence_ranking"][0]["label"] == LABELS[0]
    assert next(r for r in h.view(c)["unassessed"] if r["label"] == LABELS[1])["confidence"] is None


@pytest.mark.parametrize("bad", ["missing", "range", "foreign", "failed", "duplicate"])
def test_invalid_update_is_atomic(bad):
    c = put(case(), results={"failure:1": {"status": "failed"}})
    data = update().model_dump()
    if bad == "missing": del data["changes"][0]["confidence"]
    if bad == "range": data["changes"][0]["confidence"] = 1.1
    if bad in {"foreign", "failed"}: data["changes"][0]["refs"] = ["other-case:result" if bad == "foreign" else "failure:1"]
    if bad == "duplicate": data["changes"].append(data["changes"][0])
    before = c.model_dump()
    with pytest.raises((ValidationError, OperationRejected)):
        h.update(c, UpdateHypotheses.model_validate(data))
    assert c.model_dump() == before


def test_failed_observation_may_be_acknowledged_without_lowering_confidence():
    c, _ = h.update(case(), update())
    c = put(c, results={"failure:1": {"status": "failed"}})
    assert h.view(c)["assessment_required"]
    c, _ = h.update(c, UpdateHypotheses(kind="update_hypotheses", changes=[],
                                      note="Failed computation does not change the physical hypothesis."))
    assert h.view(c)["confidence_ranking"][0]["confidence"] == .4
    assert not h.view(c)["assessment_required"]
    c, _ = h.update(c, update(limitation_refs=["failure:1"]))
    assert h.state(c).entries[LABELS[0]].limitation_refs == ["failure:1"]


def test_query_allowance_persists_through_repair_and_unchanged_update(tmp_path):
    c, o = h.sync(case()), enabled(tmp_path)
    s = store(tmp_path, c)
    for i in range(2):
        incoming = reply(arguments={"records": []}, call_id=f"bad{i}")
        c = _local_error(c, [{"path": ["arguments", "records"], "code": "minItems"}], s, exchange=incoming)
    assert h.view(c)["query_turns_remaining"] == 0
    c = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    c, _ = h.update(c, update())
    assert h.view(c)["query_turns_remaining"] == 0
    fake = NativeFake([reply("update_hypotheses", update().model_dump(mode="json", exclude={"kind"}))])
    ask_tools(fake, REGISTRY, [], planner_payload(o, c), s)
    assert "read_evidence" not in {t["name"] for t in fake.requests[0]["tools"]}
    c = put(c, latest_exchange=reply(call_id="third"))
    with pytest.raises(OperationRejected, match="Scientific queries.*exhausted"):
        _apply(o, c, ReadEvidence(kind="read_evidence", records=[{"ref": "input:smiles"}]), s)
    c = put(c, results={"result:new": {"status": "success"}})
    assert h.view(c)["query_turns_remaining"] == 2


def test_new_computation_requires_assessment_but_read_is_allowed(tmp_path):
    c, o = computed(), enabled(tmp_path)
    with pytest.raises(OperationRejected, match="provisional assessment"):
        _apply(o, c, command(), store(tmp_path, c))
    c = put(c, latest_exchange=reply())
    c = _apply(o, c, ReadEvidence(kind="read_evidence", records=[{"ref": "input:smiles"}]), store(tmp_path, c))
    assert h.view(c)["query_turns_remaining"] == 1


def test_combined_update_and_execution_resume_without_second_model_call(tmp_path):
    o, c = enabled(tmp_path), computed()
    incoming = combined()
    c = put(c, latest_exchange=incoming)
    s = store(tmp_path, c)
    c = save(_apply(o, c, parse(incoming), s), s)
    assert protocol(c)["phase"] == "hypothesis_next" and h.state(c).assessments == 1
    assert not list((s.root / "operation_results").glob("*.json"))
    c = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    c = save(_hypothesis_followup(o, c, s), s)
    assert protocol(c)["phase"] == "execute"
    c = _execute(o, c, s)
    c = save(_comment(o, c, s), s)
    assert h.state(c).assessments == 1 and protocol(c)["rounds"] == 1
    output = json.loads((s.root / "operation_results/planner-assess.json").read_text())
    assert set(output["outcome"]["call_results"]) == {"assess", "compute"}
    assert output["outcome"]["call_results"]["compute"]["executed"]
    assert h.view(c)["assessment_required"]
    replay = replay_exchange(incoming, protocol(c)["outcome"], "responses")
    assert [m["call_id"] for m in replay[-2:]] == ["assess", "compute"]


def test_valid_update_survives_bad_followup_parameters(tmp_path):
    o, c = enabled(tmp_path), computed()
    bad = reply("compute_" + TOOL.replace(".", "_"), {"arguments": {"impossible": 1}, "purpose": "Fixture"}, call_id="compute")
    incoming = combined(bad)
    c = put(c, latest_exchange=incoming)
    s = store(tmp_path, c)
    c = save(_apply(o, c, parse(incoming), s), s)
    c = save(_hypothesis_followup(o, c, s), s)
    assert h.state(c).assessments == 1 and protocol(c)["rounds"] == 0
    receipts = protocol(c)["outcome"]["call_results"]
    assert receipts["assess"]["committed"] and receipts["compute"]["status"] == "rejected"
    assert receipts["compute"]["issues"][0]["path"][0] == "arguments"


@pytest.mark.parametrize("status", ["success", "partial", "failed"])
def test_combined_cached_computation_completes_once_after_resume(tmp_path, status):
    from mechcal.runtime.command_operations import prepare_computation
    o, c = enabled(tmp_path), computed()
    request, key = prepare_computation(c, command(), REGISTRY, amesp_enabled=False)
    c = put(c, computations={key: "saved:result"}, results={"saved:result": {
        "tool_result_id": "saved:result", "case_id": c.case_id, "status": status,
        "dispatch_id": request.dispatch_id, "failure": {"message": "Fixture failure"} if status == "failed" else {},
    }})
    incoming = combined()
    c = put(c, latest_exchange=incoming)
    s = store(tmp_path, c)
    c = save(_apply(o, c, parse(incoming), s), s)
    c = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    c = save(_hypothesis_followup(o, c, s), s)
    assert protocol(c)["phase"] == "choose"
    assert protocol(c)["hypothesis_step"] is None
    assert h.state(c).assessments == 1 and protocol(c)["rounds"] == 0
    receipt = protocol(c)["outcome"]["call_results"]
    assert set(receipt) == {"assess", "compute"}
    assert receipt["compute"]["result_status"] == status
    assert receipt["compute"]["executed"] is False
    assert receipt["compute"]["retry_of"] == ("saved:result" if status in {"failed", "partial"} else None)
    save(c, s)
    assert len(list((s.root / "operation_results").glob("*.json"))) == 1


def test_bad_update_never_dispatches_followup(tmp_path):
    o, c = enabled(tmp_path), computed()
    incoming = combined(assessment=update(refs=["unknown:record"]))
    before = c.model_dump()
    with pytest.raises(OperationRejected):
        _apply(o, put(c, latest_exchange=incoming), parse(incoming), store(tmp_path, c))
    assert c.model_dump() == before


@pytest.mark.parametrize("kind", ["truncated", "two_updates", "mixed_read"])
def test_invalid_combination_has_no_side_effects(kind):
    incoming = combined()
    if kind == "truncated": incoming["complete"] = False
    if kind == "two_updates": incoming["calls"].append({**incoming["calls"][0], "id": "duplicate"})
    if kind == "mixed_read": incoming = combined(reply())
    with pytest.raises(OperationRejected): parse(incoming)


def test_snapshot_identity_migration_and_idempotency(tmp_path):
    c = case()
    old = deepcopy(c.model_dump())
    assert h.view(c)["confidence_ranking"] == []
    assert c.model_dump() == old
    c, _ = h.update(c, update())
    s = store(tmp_path, c)
    c = save(c, s)
    c, receipt = h.update(c, update())
    c = save(c, s)
    assert receipt["status"] == "unchanged" and len(list((s.root / "events").glob("*.json"))) == 2
    assert h.state(case()).entries == {}
    protocol(c)["exploration"]["case_id"] = "another-case"
    with pytest.raises(RuntimeError, match="ownership"):
        h.view(c)


def test_formal_ranking_review_contract_unchanged(tmp_path):
    o, c = enabled(tmp_path), case()
    o.config = replace(o.config, enable_support_auditor=True)
    c = _apply(o, c, update(), store(tmp_path, c))
    assert protocol(c)["phase"] == "choose" and protocol(c)["revision"] == 0
    c = _apply(o, c, judgment(), store(tmp_path, c))
    assert protocol(c)["phase"] == "review" and protocol(c)["revision"] == 1
    assert h.state(c).entries[LABELS[0]].confidence == .4


def test_feature_is_opt_in(tmp_path):
    o, c = orchestrator(tmp_path), case()
    assert "exploration_state" not in planner_payload(o, c)
    assert "update_hypotheses" not in tool_contracts(REGISTRY, [])[1]
    assert "update_hypotheses" not in json.dumps(envelope_model(REGISTRY, []).model_json_schema())
    with pytest.raises(OperationRejected, match="disabled"):
        _apply(o, c, update(), store(tmp_path, c))


def test_batch_yields_one_assessment_boundary_and_preserves_all_receipts(tmp_path):
    o, c = enabled(tmp_path), computed()
    incoming = combined()
    second = deepcopy(incoming["calls"][1])
    second["id"] = "compute2"
    # Identical work is reused by the existing computation cache, not rerun.
    incoming["calls"].append(second)
    incoming["items"].append({"type": "function_call", "call_id": second["id"],
                              "name": second["name"], "arguments": second["arguments"]})
    c = put(c, latest_exchange=incoming)
    s = store(tmp_path, c)
    c = save(_apply(o, c, parse(incoming), s), s)
    c = save(_hypothesis_followup(o, c, s), s)
    assert protocol(c)["phase"] == "batch_next"
    c = save(next_member(o, c), s)
    c = _execute(o, c, s)
    c = save(finish_member(_comment(o, c, s)), s)
    assert h.state(c).observed_results == []
    c = save(next_member(o, c), s)
    assert protocol(c)["phase"] == "choose" and protocol(c)["rounds"] == 1
    receipt = protocol(c)["outcome"]["call_results"]
    assert set(receipt) == {"assess", "compute", "compute2"}
    assert receipt["compute2"]["experiments"][0]["status"] == "reused"
    c = h.sync(c)
    assert len(h.view(c)["new_result_ids"]) == 1
    assert h.state(c).assessments == 1


def test_combined_formal_update_retains_review_and_separate_confidence(tmp_path):
    o, c = enabled(tmp_path), case()
    o.config = replace(o.config, enable_support_auditor=True)
    o.deps.scientific_reviewer.llm_client = NativeFake([prose("Offline advisory review.")])
    formal = reply("revise_judgment", judgment().model_dump(mode="json", exclude={"kind"}), call_id="formal")
    incoming = combined(formal, update(.7))
    c = put(c, latest_exchange=incoming)
    s = store(tmp_path, c)
    c = save(_apply(o, c, parse(incoming), s), s)
    c = save(_hypothesis_followup(o, c, s), s)
    assert protocol(c)["phase"] == "review" and protocol(c)["revision"] == 1
    assert h.state(c).entries[LABELS[0]].confidence == .7
    assert protocol(c)["ranking"][LABELS[0]]["confidence"] == .4
    c = save(_comment(o, c, s), s)
    assert set(protocol(c)["outcome"]["call_results"]) == {"assess", "formal"}


def test_native_full_loop_needs_no_extra_assessment_call(tmp_path):
    o = enabled(tmp_path)
    followup = reply("finalize", {"answer": "Offline flow fixture only.", "limitations": []}, call_id="final")
    fake = NativeFake([
        combined(reply("revise_judgment", judgment().model_dump(mode="json", exclude={"kind"}), call_id="formal")),
        followup,
    ])
    o.deps.planner.llm_client = fake
    c = o.run(smiles="CCO", user_query="Offline state fixture.", case_id="hypothesis-final")
    assert c.status == "finalized" and len(fake.requests) == 2
    assert h.state(c).assessments == 1 and protocol(c)["revision"] == 1
    assert protocol(c)["rounds"] == 0




def test_combined_formal_errors_target_draft_without_rolling_back_hypotheses(tmp_path):
    o, c = enabled(tmp_path), case()
    cmd = judgment()
    cmd.changes[0].refs = ["unknown:source"]
    incoming = combined(reply("revise_judgment", cmd.model_dump(mode="json", exclude={"kind"}), call_id="formal"))
    c = put(c, latest_exchange=incoming)
    s = store(tmp_path, c)
    c = save(_apply(o, c, parse(incoming), s), s)
    c = save(_hypothesis_followup(o, c, s), s)
    outcomes = protocol(c)["outcome"]["call_results"]
    assert outcomes["assess"]["committed"]
    assert outcomes["formal"]["issues"][0]["path"] == ["draft", "changes", 0, "refs", 0]
    assert protocol(c)["revision"] == 0 and h.state(c).assessments == 1
