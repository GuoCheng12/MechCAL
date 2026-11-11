"""No-change defaults and local correction never invent scientific values."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from mechcal.agents.command_client import schema_issues
from mechcal.agents.command_shape_repair import retry_local_command, shape_only
from mechcal.agents.command_tools import ask_tools, parse_tool_turn, tool_contracts
from mechcal.runtime.command_loop import _apply, planner_payload
from mechcal.runtime.command_operations import OperationRejected, protocol
from mechcal.runtime.command_schema import native_format
from mechcal.schemas.commands import LinkedAmendJudgment, LinkedRankingChange
from test_command_protocol import REGISTRY, TOOL, command, LABELS, rank
from test_command_review_workflow import setup
from test_native_commands import NativeFake, reply


def amendment():
    return {"edits": [{"id": "A1", "text": "A corrected bounded interpretation."}],
            "note": "Correct wording only.", "request_review": False}


def contracts():
    return tool_contracts(REGISTRY, [], review_workflow=True)


def rejected(exchange, issues):
    return {"status": "rejected", "executed": False, "committed": False,
            "call_ids": [c["id"] for c in exchange["calls"]], "issues": issues}


def strict_issues(exchange, tools):
    call = exchange["calls"][0]
    schema = next(t["parameters"] for t in tools if t["name"] == call["name"])
    return schema_issues(schema, json.loads(call["arguments"]))


def test_wire_and_local_schemas_share_explicit_omission_semantics():
    strict = native_format(LinkedAmendJudgment)["json_schema"]["schema"]
    relaxed = native_format(LinkedAmendJudgment, strict=False)["json_schema"]["schema"]
    assert "rankings" in strict["required"] and "rankings" not in relaxed["required"]
    assert "x-mechcal" not in json.dumps(strict) + json.dumps(relaxed)
    assert "id" in relaxed["$defs"]["AcceptedArgumentEdit"]["required"]
    full = native_format(LinkedRankingChange, strict=False)["json_schema"]["schema"]
    assert {"label", "confidence", "priority", "support", "claim"} <= set(full["required"])


def test_missing_noop_fields_preserve_accepted_science(tmp_path):
    c, o, s = setup(tmp_path)
    tools, models = contracts()
    cmd = parse_tool_turn(reply("amend_judgment", amendment()), tools, models)
    assert cmd.rankings == [] and cmd.edits[0].refs is None
    updated = _apply(o, c, cmd, s)
    before, after = protocol(c), protocol(updated)
    assert before["ranking"] == after["ranking"]
    assert before["arguments"]["A1"]["refs"] == after["arguments"]["A1"]["refs"]
    assert before["results"] == after["results"]
    assert after["arguments"]["A1"]["text"] == amendment()["edits"][0]["text"]


def test_explicit_empty_reference_list_is_not_omission():
    tools, models = contracts()
    data = amendment()
    data["edits"][0]["refs"] = []
    assert parse_tool_turn(reply("amend_judgment", data), tools, models).edits[0].refs == []


@pytest.mark.parametrize("missing", ["label", "confidence", "priority", "support", "claim"])
def test_full_scientific_record_defaults_do_not_silently_fill(missing):
    tools, models = contracts()
    row = {k: v for k, v in rank(LABELS[0], ["A1"]).items() if k != "rationale"}
    row = LinkedRankingChange.model_validate(row).model_dump(mode="json")
    parse_tool_turn(reply("revise_judgment", {"changes": [row], "note": "", "request_review": False}), tools, models)
    row.pop(missing)
    with pytest.raises(OperationRejected):
        parse_tool_turn(reply("revise_judgment", {"changes": [row], "note": "", "request_review": False}), tools, models)


def test_missing_execution_parameters_remain_required():
    tools, models = tool_contracts(REGISTRY, [TOOL], strict_schema=False)
    name = next(t["name"] for t in tools if t["name"].startswith("compute_"))
    data = {"arguments": command().arguments.model_dump(mode="json"), "purpose": "Compute exact request."}
    assert data["arguments"]
    data["arguments"].pop(next(iter(data["arguments"])))
    with pytest.raises(OperationRejected):
        parse_tool_turn(reply(name, data), tools, models)


def test_old_missing_noop_rejection_resumes_without_model_call(tmp_path):
    c, o, s = setup(tmp_path)
    o.config = replace(o.config, command_compact_context=True)
    tools, _ = contracts()
    exchange = reply("amend_judgment", amendment())
    outcome = rejected(exchange, strict_issues(exchange, tools))
    fake = NativeFake([])
    cmd, returned = ask_tools(fake, REGISTRY, [], planner_payload(o, c, native=True), s,
                              exchange=exchange, outcome=outcome)
    assert not fake.requests and returned["no_change_normalized"]
    assert returned["current_payload_delivered"] is False
    assert cmd.rankings == []
    assert protocol(_apply(o, c, cmd, s))["revision"] == protocol(c)["revision"] + 1
    assert (s.root / "history/command_normalizations.jsonl").exists()


def test_local_repair_only_sends_one_command_and_schema(tmp_path):
    c, o, s = setup(tmp_path)
    o.config = replace(o.config, command_compact_context=True)
    data = {**amendment(), "rankings": "[]"}
    exchange = reply("amend_judgment", data)
    # Fill other no-change fields so this failure is a pure type error.
    semantic = native_format(LinkedAmendJudgment, strict=False)["json_schema"]["schema"]
    errors = schema_issues(semantic, {"kind": "amend_judgment", **data})
    fake = NativeFake([reply("amend_judgment", {**data, "rankings": []}, call_id="corrected")])
    cmd, returned = ask_tools(fake, REGISTRY, [], planner_payload(o, c, native=True), s,
                              exchange=exchange, outcome=rejected(exchange, errors))
    sent = fake.requests[0]
    assert len(fake.requests) == 1 and sent["role"] == "planner_repair"
    assert set(sent["payload"]) == {"stage", "original_call", "issues"}
    assert sent["exchange"] is None and sent["outcome"] is None
    assert [t["name"] for t in sent["tools"]] == ["amend_judgment", "defer_command_repair"]
    assert returned["command_repair_attempted"] and cmd.rankings == []
    assert returned["current_payload_delivered"] is False
    assert protocol(c)["arguments"]["A1"]["text"] != cmd.edits[0].text


@pytest.mark.parametrize("change", ["science", "tool", "truncated", "defer", "malformed"])
def test_local_repair_cannot_expand_its_scope(tmp_path, change):
    _, _, s = setup(tmp_path)
    tools, models = contracts()
    data = {**amendment(), "rankings": "[]"}
    old = reply("amend_judgment", data)
    fixed = {**deepcopy(data), "rankings": []}
    new = reply("amend_judgment", fixed, call_id="repair")
    if change == "science":
        fixed["edits"][0]["text"] = "New unsupported scientific claim."
        new = reply("amend_judgment", fixed)
    elif change == "tool":
        new = reply("describe_tools", {"capability_ids": [TOOL]})
    elif change == "truncated":
        new["complete"] = False
    elif change == "defer":
        new = reply("defer_command_repair", {"reason": "Need evidence."})
    elif change == "malformed":
        new["calls"][0]["arguments"] = "{"
    with pytest.raises(OperationRejected) as exc:
        retry_local_command(NativeFake([new]), tools, models, old,
                            rejected(old, [{"code": "type"}]), s, parse_tool_turn)
    assert exc.value.exchange["command_repair_attempted"]


def test_transport_failure_does_not_reset_local_attempt_allowance(tmp_path):
    _, _, s = setup(tmp_path)
    tools, models = contracts()
    old = reply("amend_judgment", {**amendment(), "rankings": "[]"})
    outcome = rejected(old, [{"code": "type"}])
    with pytest.raises(StopIteration):
        retry_local_command(NativeFake([]), tools, models, old, outcome, s, parse_tool_turn)
    fake = NativeFake([])
    assert retry_local_command(fake, tools, models, old, outcome, s, parse_tool_turn) is None
    assert not fake.requests


@pytest.mark.parametrize("code", ["unknown_reference", "stale_draft", "wrong_case", "minimum", "enum"])
def test_semantic_or_state_errors_stay_with_full_planner(tmp_path, code):
    _, _, s = setup(tmp_path)
    tools, models = contracts()
    old = reply("amend_judgment", amendment())
    assert retry_local_command(NativeFake([]), tools, models, old,
                               rejected(old, [{"code": code}]), s, parse_tool_turn) is None


def test_repaired_operation_still_checks_references_atomically(tmp_path):
    c, o, s = setup(tmp_path)
    tools, models = contracts()
    data = amendment()
    data["edits"].append({"id": "A2", "refs": ["other_case:unknown"]})
    cmd = parse_tool_turn(reply("amend_judgment", data), tools, models)
    before = c.model_dump(mode="json")
    with pytest.raises(OperationRejected):
        _apply(o, c, cmd, s)
    assert c.model_dump(mode="json") == before


def test_type_repair_preserves_numeric_meaning_and_valid_fields():
    issue = [{"path": ["confidence"], "code": "type"}]
    assert shape_only({"confidence": "0.4", "text": "same"}, {"confidence": .4, "text": "same"}, issue)
    assert not shape_only({"confidence": "0.4"}, {"confidence": .5}, issue)
    assert not shape_only({"confidence": "0.4", "text": "same"}, {"confidence": .4, "text": "new"}, issue)
    assert not shape_only({"confidence": ".4"}, {"confidence": .4}, issue)
