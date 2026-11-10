"""Control-flow acceptance, not a claim of improved scientific performance."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from mechcal.agents.command_tools import ask_tools, parse_tool_turn, tool_contracts
from mechcal.runtime import command_followup as f, command_hypotheses as h
from mechcal.runtime import command_investigation as inv, computational_review as cr
from mechcal.runtime.command_compact import expand_text
from mechcal.runtime.command_action_policy import turn_contract
from mechcal.runtime.command_loop import _apply, _execute, _hypothesis_followup, planner_payload, save
from mechcal.runtime.command_operations import OperationRejected, check_refs, protocol, put, record_index
from mechcal.runtime.command_review_workflow import comparison_options
from mechcal.runtime.command_task_context import prepare_planner
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import (
    Finalize, ReadEvidence, SubmitFinal, UpdateHypothesesWithInvestigation as Update,
    computation_model, envelope_model,
)
from test_command_followup import question, associated
from test_command_protocol import LABELS, REGISTRY, TOOL, command
from test_command_review_workflow import setup
from test_exploration_batches import batch_reply
from test_native_commands import NativeFake, reply


def prepared(tmp_path):
    c, o, s = setup(tmp_path)
    o.config = replace(o.config, command_computational_review=True, command_investigation_policy=True)
    return inv.enable(c), o, s


def choice(**fields):
    return {"decision": "investigate", "question_id": None, "considered_question_ids": [],
            "reason": "An informative comparison, not a confidence score.", **fields}


def select(c, **fields):
    return h.update(c, Update(kind="update_hypotheses", note="Retain beliefs.", investigation=choice(**fields)))[0]


def test_selection_does_not_change_beliefs_rank_evidence_or_rounds(tmp_path):
    c, _, _ = prepared(tmp_path)
    before = deepcopy(c.model_dump(mode="json"))
    out = select(c)
    for field in ("results", "arguments", "ranking", "rounds", "revision"):
        assert protocol(out)[field] == before["runtime"]["command_protocol"][field]
    for field in ("portfolio", "evidence_ledger", "artifact_manifest"):
        assert out.model_dump(mode="json")[field] == before[field]
    assert h.state(out).entries == h.state(c).entries
    assert inv.current(out)["basis"] == inv.basis(out)
    assert c.model_dump(mode="json") == before


def test_selected_lower_rank_question_is_not_filtered_by_confidence(tmp_path):
    c, _, _ = prepared(tmp_path)
    c, _ = h.update(c, Update(kind="update_hypotheses", note="Useful test for a lower-ranked mechanism.",
        questions=[question(id="low", labels=[LABELS[2]], route={"capability_id": TOOL})],
        investigation=choice(question_id="low")))
    options = comparison_options(c, [{"capability_id": TOOL, "status": "ready"}])
    assert options[0]["id"] == "low"
    inv.guard(c, command())
    assert protocol(c)["ranking"][LABELS[2]]["confidence"] == .4


@pytest.mark.parametrize("status", ["success", "partial", "failed", "unsupported"])
def test_new_observation_invalidates_choice_without_altering_science(tmp_path, status):
    c, _, _ = prepared(tmp_path)
    c = select(c)
    c = put(c, results={"new": {"status": status, "capability_id": TOOL}})
    beliefs = deepcopy(h.state(c).entries)
    with pytest.raises(OperationRejected, match="No current investigate"):
        inv.guard(c, command())
    c = select(c)
    inv.guard(c, command())
    assert h.state(c).entries == beliefs


def test_reads_and_redundant_trial_receipts_do_not_spend_choices(tmp_path):
    c, o, s = prepared(tmp_path)
    c, _ = h.update(c, Update(kind="update_hypotheses", note="Test.", questions=[question()], investigation=choice()))
    expected = inv.basis(c)
    c = _apply(o, c, ReadEvidence(kind="read_evidence", records=[{"ref": "input:smiles"}]), s)
    c = f.record_trial(c, f.question_snapshot(c, "compare"), capability_id=TOOL, purpose="Cached test", result_id="old")
    assert inv.basis(c) == expected
    inv.guard(c, command())


def test_choice_survives_resume_and_repeat_is_idempotent(tmp_path):
    c, _, s = prepared(tmp_path)
    c = select(c)
    save(c, s)
    c = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    again = select(c)
    assert inv.current(c) == inv.current(again)
    assert len(protocol(again)["investigation_policy"]["choices"]) == 1
    inv.guard(again, command())


@pytest.mark.parametrize("edit", [
    {"question_id": "foreign"}, {"considered_question_ids": ["foreign"]},
    {"decision": "conclude", "question_id": "q"},
])
def test_invalid_selection_is_local_and_entire_update_atomic(tmp_path, edit):
    c, _, _ = prepared(tmp_path)
    before = c.model_dump(mode="json")
    with pytest.raises((OperationRejected, ValidationError)):
        h.update(c, Update(kind="update_hypotheses", note="No partial mutation.", questions=[question(id="q")],
            changes=[{"label": LABELS[0], "confidence": .9, "note": "New provisional belief."}],
            investigation=choice(**edit)))
    assert c.model_dump(mode="json") == before


def test_selection_is_queryable_but_not_scientific_evidence_and_case_isolated(tmp_path):
    c, _, _ = prepared(tmp_path)
    c = select(c)
    assert inv.STATE_REF in record_index(c).available
    with pytest.raises(OperationRejected, match="not a raw observation"):
        check_refs(c, [inv.STATE_REF], [], scientific=True)
    protocol(c)["investigation_policy"]["case_id"] = "foreign"
    with pytest.raises(RuntimeError, match="ownership"):
        inv.current(c)


def test_all_related_impacts_persist_until_explicit_response():
    c = inv.enable(associated())
    ids = list(f.state(c).impacts)
    c, _ = h.update(c, Update(kind="update_hypotheses", note="One candidate checked, other retained pending.",
        responses=[{"impact_id": ids[0], "disposition": "reviewed", "note": "No confidence change."}],
        investigation=choice()))
    assert inv.view(c, [])["pending_impact_count"] == 1
    assert inv.view(c, [])["pending_impacts"][0]["id"] == ids[1]
    assert h.state(c).event["details"]["impact_coverage"][ids[1]]["disposition"] == "pending"


def test_conclusion_allowed_with_unresolved_and_failed_tests(tmp_path):
    c, _, _ = prepared(tmp_path)
    c, _ = h.update(c, Update(kind="update_hypotheses", note="Bounded conclusion.",
        questions=[question(status="deferred")],
        investigation=choice(decision="conclude", considered_question_ids=["compare"])))
    inv.guard(c, Finalize(kind="finalize", answer="Retain uncertainty."))
    with pytest.raises(OperationRejected, match="No current investigate"):
        inv.guard(c, command())
    assert f.state(c).questions["compare"].value.status == "deferred"


def test_existing_reviewer_receives_stop_choice_and_reuses_unchanged_basis(tmp_path):
    c, _, _ = prepared(tmp_path)
    c = select(c, decision="conclude")
    context = cr.context(c, [])
    assert context["investigation"]["current_choice"]["choice"]["decision"] == "conclude"
    c = put(c, computational_review={"case_id": c.case_id, "basis": context["basis"], "suggestions": []})
    assert cr.context(select(c, decision="conclude"), [])["previous"]
    assert cr.context(select(c, reason="A new experiment is now worth doing."), [])["previous"] is None


def test_minimal_task_view_keeps_selection_and_way_back_to_experiments(tmp_path):
    c, o, s = prepared(tmp_path)
    o.config = replace(o.config, command_compact_context=True, command_task_context=True)
    c = select(c, decision="conclude")
    c = _apply(o, c, Finalize(kind="finalize", answer="Bounded comparison."), s)
    c, payload = prepare_planner(c, planner_payload(o, c, native=True))
    payload = expand_text(payload)
    assert payload["investigation_policy"]["current_choice"]["decision"] == "conclude"
    client = NativeFake([reply("update_hypotheses", {"note": "Choose another experiment.",
        "investigation": choice()}, call_id="choice")])
    parsed, _ = ask_tools(client, REGISTRY, [], payload, s)
    assert isinstance(parsed, Update)
    assert parsed.investigation.decision == "investigate"
    assert "read_context" in {t["name"] for t in client.requests[0]["tools"]}


@pytest.mark.parametrize("strict", [False, True])
def test_wire_and_checkpoint_share_selection_schema_and_keep_old_contract(tmp_path, strict):
    tools, models = tool_contracts(REGISTRY, [TOOL], hypothesis_tracking=True, evidence_followup=True,
        review_workflow=True, investigation_policy=True, strict_schema=strict)
    for t in tools:
        Draft202012Validator.check_schema(t["parameters"])
    data = Update(kind="update_hypotheses", note="Test.", investigation=choice()).model_dump(mode="json")
    wire = reply("update_hypotheses", {k: v for k, v in data.items() if k != "kind"})
    parsed = parse_tool_turn(wire, tools, models)
    assert parsed.investigation.reason == data["investigation"]["reason"]
    restored = envelope_model(REGISTRY, [], hypothesis_tracking=True, review_workflow=True,
        investigation_policy=True).model_validate({"command": data}).command
    assert restored == parsed
    old, _ = tool_contracts(REGISTRY, [], hypothesis_tracking=True, review_workflow=True)
    assert "investigation" not in next(t for t in old if t["name"] == "update_hypotheses")["parameters"]["properties"]


def test_batched_selection_executes_without_an_extra_planner_call_and_caches(tmp_path):
    c, o, s = prepared(tmp_path)
    update = Update(kind="update_hypotheses", note="Choose test.", questions=[question()], investigation=choice(question_id="compare"))
    compute = computation_model(REGISTRY.require(TOOL), with_question=True)(kind="compute", capability_id=TOOL,
        arguments={}, purpose="Offline comparison.", question_id="compare")
    incoming = batch_reply(reply("update_hypotheses", update.model_dump(mode="json", exclude={"kind"}), call_id="u"),
        reply("compute_" + TOOL.replace(".", "_"), compute.model_dump(mode="json", exclude={"kind", "capability_id"}), call_id="c"))
    tools, models = tool_contracts(REGISTRY, [TOOL], hypothesis_tracking=True, evidence_followup=True,
        review_workflow=True, investigation_policy=True)
    parsed = parse_tool_turn(incoming, tools, models)
    c = save(_apply(o, put(c, latest_exchange=incoming), parsed, s), s)
    c = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    c = _hypothesis_followup(o, c, s)
    assert protocol(c)["phase"] == "execute"
    c = _execute(o, c, s)
    assert protocol(c)["rounds"] == 1 and len(protocol(c)["results"]) == 1
    c = select(c, question_id="compare")
    c = _apply(o, c, compute, s)
    assert protocol(c)["outcome"]["status"] == "reused" and protocol(c)["rounds"] == 1


def test_no_selection_has_actionable_error_and_no_calculation(tmp_path):
    c, o, s = prepared(tmp_path)
    with pytest.raises(OperationRejected) as error:
        _apply(o, c, command(), s)
    assert error.value.issues[0]["code"] == "investigation_choice_required"
    assert "same response" in error.value.issues[0]["remedy"]
    assert protocol(c)["rounds"] == 0


def test_new_flag_requires_existing_reviewer_and_old_cases_remain_readable(tmp_path):
    c, o, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="requires command_computational_review"):
        replace(o.config, command_investigation_policy=True)
    old = CaseRun.model_validate_json(c.model_dump_json())
    assert not inv.enabled(old)
    inv.guard(old, command())


def test_scoped_probe_advertises_only_the_operations_it_accepts(tmp_path):
    c, o, s = prepared(tmp_path)
    client = NativeFake([reply("update_hypotheses", {"note": "Choice only.", "investigation": choice()})])
    ask_tools(client, REGISTRY, [], planner_payload(o, c, native=True), s,
              allowed_operations={"update_hypotheses"})
    groups = client.requests[0]["payload"]["operation_groups"]
    assert groups["available_operations"] == ["update_hypotheses"]
    assert groups["read_calls"] == groups["read_selections"] == groups["compute_calls"] == 0
    assert groups["batch_read_operations"] == groups["hypothesis_followup_controls"] == []
    assert turn_contract(["read_evidence"])["read_calls"] == 4
    assert turn_contract(["compute_test"])["compute_calls"] == 6
