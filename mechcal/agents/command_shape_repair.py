"""Bounded native-command syntax correction; runtime still validates state."""

import json

from pydantic import ValidationError

from mechcal.agents.command_client import schema_issues
from mechcal.runtime.command_memory import digest, json_bytes
from mechcal.runtime.command_operations import OperationRejected
from mechcal.runtime.command_schema import command_parameters, native_format
from mechcal.runtime.prompts import load_prompt
from mechcal.schemas.commands import DeferCommandRepair


def same_value(before, after):
    return json.dumps(before, sort_keys=True) == json.dumps(after, sort_keys=True)


def shape_only(before, after, issues):
    """Only remove unknown fields or decode the same JSON value at a type error."""
    allowed = {tuple(i["path"]): i["code"] for i in issues}
    absent = object()

    def check(old, new, path):
        code = allowed.get(path)
        if code == "unexpected_field":
            return new is absent
        if code == "type" and isinstance(old, str):
            try:
                return same_value(json.loads(old), new)
            except (ValueError, TypeError):
                return False
        if old is absent or new is absent:
            return False
        if isinstance(old, dict) and isinstance(new, dict):
            return all(check(old.get(k, absent), new.get(k, absent), (*path, k)) for k in old.keys() | new.keys())
        if isinstance(old, list) and isinstance(new, list):
            return len(old) == len(new) and all(check(a, b, (*path, i)) for i, (a, b) in enumerate(zip(old, new)))
        return same_value(old, new)

    return check(before, after, ())


def retry_local_command(client, tools, models, exchange, outcome, store, parse):
    """Return an unexecuted command, or None to use the normal full Planner turn."""
    if (not exchange or not exchange.get("complete") or len(exchange.get("calls", [])) != 1
            or exchange.get("command_repair_attempted") or exchange.get("no_change_normalized")
            or not outcome or outcome.get("status") != "rejected"
            or outcome.get("executed") is not False or outcome.get("committed") is not False):
        return None
    call = exchange["calls"][0]
    if outcome.get("call_ids") != [call["id"]] or call["name"] not in models:
        return None
    if not outcome.get("issues") or any(i["code"] not in {"missing_field", "unexpected_field", "type"}
                                       for i in outcome["issues"]):
        return None
    try:
        data = json.loads(call["arguments"])
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    model, identity = models[call["name"]]
    schema = command_parameters(model, identity, strict=False)
    issues = schema_issues(schema, data)
    if not issues:
        prior = schema_issues(command_parameters(model, identity), data)
        if not prior or any(i["code"] != "missing_field" for i in prior):
            return None
        command = parse(exchange, tools, models)
        store.append_jsonl("history/command_normalizations.jsonl", {
            "request_id": exchange["request_id"], "call_id": call["id"],
            "kind": "declared_no_change_defaults", "fields": [i["path"] for i in prior],
            "model_calls": 0, "executed": False, "committed": False})
        return command, {**exchange, "no_change_normalized": True}
    # Missing scientific values, wrong enums, references and state errors need
    # the full decision context. A small repair must never invent their answers.
    if any(i["code"] not in {"unexpected_field", "type"} for i in issues):
        return None
    payload = {"stage": "local_command_repair", "original_call": {"name": call["name"], "arguments": data},
               "issues": issues}
    tool = next(t for t in tools if t["name"] == call["name"])
    defer = {"name": "defer_command_repair", "description": "Return to full Planner context without executing anything.",
             "parameters": native_format(DeferCommandRepair)["json_schema"]["schema"]}
    if json_bytes(payload) + json_bytes([tool, defer]) > 24000:
        return None
    ticket = "command_repairs/" + digest(exchange["request_id"]) + ".json"
    if (store.root / ticket).exists():
        return None
    store.write_json(ticket, {"request_id": exchange["request_id"], "attempted": True})
    store.activity("Planner local format correction")
    store.append_jsonl("history/command_repairs.jsonl", {
        "event": "attempt", "original_request_id": exchange["request_id"], "original_call_id": call["id"],
        "payload_bytes": json_bytes(payload), "issues": issues, "attempt_limit": 1})
    # No old reasoning, tool history, evidence ledger, or broad tool catalog is
    # replayed. This is a fresh, narrowly scoped format correction.
    reply = client.complete_tool_turn(system_prompt=load_prompt("command_shape_repair.md"), payload=payload,
        tools=[tool, defer], exchange=None, outcome=None, store=store, role="planner_repair")
    reply = {**reply, "command_repair_attempted": True, "repair_of": exchange["request_id"]}

    def reject(code, message):
        store.append_jsonl("history/command_repairs.jsonl", {
            "event": "not_applied", "request_id": reply["request_id"], "code": code})
        raise OperationRejected([dict(path=[], code=code, message=message,
            remedy="Return to full Planner context; no computation or judgment was committed.", record_id=None)], exchange=reply)

    calls = reply.get("calls", [])
    if not reply.get("complete") or len(calls) != 1:
        reject("repair_incomplete", "One complete correction or explicit deferral is required.")
    try:
        corrected = json.loads(calls[0]["arguments"])
    except (ValueError, TypeError):
        reject("repair_invalid_json", "Correction is not complete JSON.")
    if calls[0]["name"] == "defer_command_repair":
        try:
            reason = DeferCommandRepair.model_validate(corrected).reason
        except ValidationError:
            reason = "Correction requires full Planner context."
        reject("repair_deferred", reason)
    if calls[0]["name"] != call["name"] or not shape_only(data, corrected, issues):
        reject("repair_scope_exceeded", "Correction changed command identity, scientific content or unrelated fields.")
    try:
        command = parse(reply, [tool], {call["name"]: models[call["name"]]})
    except OperationRejected as exc:
        raise OperationRejected(exc.issues, exchange=reply) from exc
    store.append_jsonl("history/command_repairs.jsonl", {
        "event": "validated", "request_id": reply["request_id"], "original_request_id": exchange["request_id"],
        "executed": False, "committed": False})
    return command, reply
