"""Provider-native operations with explicit call/result correlation.

Only the latest completed exchange is replayed with the materialized case state.
The complete raw transcript stays in the audit log, not in every model request.
"""

from copy import deepcopy
import hashlib
import json
import time
import uuid
from urllib.parse import urlparse

from pydantic import ValidationError

from mechcal.agents.command_client import schema_issues
from mechcal.runtime.command_operations import OperationRejected
from mechcal.runtime.command_schema import native_format
from mechcal.runtime.llm import (
    OpenAIJsonClient, _add_generation_args, _is_transient_llm_exception,
    _response_usage, create_completion,
)
from mechcal.runtime.prompts import load_prompt
from mechcal.schemas.commands import (
    ComputeBatch, ComputeCall, DescribeTools, DiscardJudgment, Finalize, ManageExperiments,
    ReadBatch, ReadCall, ReadEvidence, RepairJudgment,
    HypothesisStep, ReviseJudgment, SubmitFinal, UpdateHypotheses, computation_model,
)


DESCRIPTIONS = {
    "read_evidence": "Read exact case fields or inspect directory paths and byte sizes. "
                     "Page large lists with offset/limit. Prefer report refs for a compact result. No computation.",
    "describe_tools": "Load and activate exact capability parameters, including missing-input routes.",
    "manage_experiments": "Retry or explicitly cancel saved pending computations. Retry never invents parameters; "
                          "for parameter changes cancel the old request and submit the corrected computation.",
    "revise_judgment": "Atomically revise related scientific arguments and ranking entries. Omitted records are retained.",
    "repair_judgment": "Patch the displayed uncommitted draft; null fields stay unchanged. Revalidate and commit the whole transaction.",
    "discard_judgment": "Discard an uncommitted draft without changing accepted arguments or ranking.",
    "finalize": "Submit the final narrative using the accepted ranking and evidence records.",
    "submit_final": "Deliver the exact saved, reviewed final draft without rewriting it.",
    "update_hypotheses": "Save provisional confidence changes or explicitly retain the current assessment. "
                         "Not a formal argument or final ranking. May accompany the next computation in one response.",
}


def tool_contracts(registry, capability_ids, *, allowed_operations=None, with_draft=False,
                   hypothesis_tracking=False, draft_id=None):
    models = {name: (model, {"kind": name}) for name, model in (
        ("read_evidence", ReadEvidence), ("describe_tools", DescribeTools),
        ("manage_experiments", ManageExperiments),
        ("revise_judgment", ReviseJudgment), ("finalize", Finalize), ("submit_final", SubmitFinal),
        ("repair_judgment", RepairJudgment), ("discard_judgment", DiscardJudgment))}
    if hypothesis_tracking:
        models["update_hypotheses"] = (UpdateHypotheses, {"kind": "update_hypotheses"})
    if draft_id is not None:
        if not with_draft:
            raise ValueError("A bound repair target requires a current draft")
        # Bind once when constructing this request, never to a later draft at
        # execution time. The internal command still carries the exact version.
        for name in ("repair_judgment", "discard_judgment"):
            models[name][1]["draft_id"] = draft_id
    for cid in capability_ids:
        name = "compute_" + cid.replace(".", "_")
        if len(name) > 64:
            name = name[:55] + "_" + hashlib.sha256(cid.encode()).hexdigest()[:8]
        if name in models:
            raise RuntimeError("Capability function-name collision")
        models[name] = (computation_model(registry.require(cid)), {"kind": "compute", "capability_id": cid})
    if allowed_operations is not None:
        unknown = set(allowed_operations) - models.keys()
        if unknown:
            raise ValueError("Unknown operation scope: " + ", ".join(sorted(unknown)))
        models = {name: value for name, value in models.items() if name in allowed_operations}
    if not with_draft:
        models.pop("repair_judgment", None)
        models.pop("discard_judgment", None)
    tools = []
    for name, (model, identity) in models.items():
        schema = native_format(model)["json_schema"]["schema"]
        for key in identity:
            schema["properties"].pop(key)
            schema["required"].remove(key)
        tools.append({"name": name, "description": DESCRIPTIONS.get(name,
            "Execute exactly " + identity.get("capability_id", name) + ". Explicit inputs only."),
            "parameters": schema})
    return tools, models


def replay_exchange(exchange, outcome, protocol):
    if not exchange:
        return []
    if exchange["protocol"] != protocol:
        raise RuntimeError("Cannot replay a command exchange through a different API protocol")
    messages = deepcopy(exchange["items"])
    # Accepted edits are stored/queryable, including text retained during repair.
    # Keep the durable receipt intact; acknowledge IDs once on the model wire.
    wire_outcome = deepcopy(outcome)
    if wire_outcome and wire_outcome.get("status") == "committed" and "changed" in wire_outcome:
        wire_outcome["changed"] = [{k: c[k] for k in ("kind", "id", "label") if k in c}
                                   for c in wire_outcome["changed"]]
    per_call = wire_outcome.get("call_results") if wire_outcome else None
    if per_call is not None and set(per_call) != {c["id"] for c in exchange["calls"]}:
        raise RuntimeError("Read receipts do not match the original tool-call IDs")
    for call in exchange["calls"]:
        result = json.dumps(per_call[call["id"]] if per_call is not None else wire_outcome,
                            ensure_ascii=False, separators=(",", ":"))
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": result}
                        if protocol == "chat_completions" else
                        {"type": "function_call_output", "call_id": call["id"], "output": result})
    return messages


class NativeCommandClient(OpenAIJsonClient):
    def complete_tool_turn(self, *, system_prompt, payload, tools, exchange, outcome, store,
                           role="planner", allow_text=False):
        settings = self.settings
        protocol = settings.api_protocol
        messages = [{"role": "system", "content": system_prompt}]
        messages += replay_exchange(exchange, outcome, protocol)
        # The current state/request follows the completed exchange; otherwise
        # the old result can appear to have already satisfied this new request.
        messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))})
        # The official main DeepSeek endpoint does not promise strict tool mode.
        # Do not silently move credentials to /beta or claim server enforcement.
        deepseek = urlparse(settings.base_url).hostname == "api.deepseek.com"
        definitions = ([{"type": "function", "function": {**t, "strict": False}} for t in tools]
                       if protocol == "chat_completions" else
                       [{"type": "function", **t, "strict": not deepseek} for t in tools])
        args = {"model": settings.model, "messages": messages}
        if definitions:
            args.update(tools=definitions, tool_choice="auto" if deepseek or allow_text else "required",
                        parallel_tool_calls=any(t["name"] == "read_evidence" or t["name"].startswith("compute_")
                                                for t in tools))
        if settings.reasoning_effort:
            args["reasoning_effort"] = settings.reasoning_effort
        if protocol == "responses" and not deepseek:
            args["include"] = ["reasoning.encrypted_content"]
        _add_generation_args(args, settings)
        request_id = role + "-" + uuid.uuid4().hex
        attempts = max(1, settings.max_retries + 1)
        for attempt in range(1, attempts + 1):
            started = time.monotonic()
            store.append_jsonl("history/command_requests.jsonl", {
                "event": "request", "id": request_id, "attempt": attempt, "role": role,
                "transport": "native_tools", "protocol": protocol, "timestamp": time.time(),
                "messages": messages, "tools": definitions,
                "messages_bytes": len(json.dumps(messages).encode()),
                "tools_bytes": len(json.dumps(definitions).encode()),
            })
            try:
                response = create_completion(settings, args)
            except Exception as exc:
                store.append_jsonl("history/command_requests.jsonl", {
                    "event": "transport_error", "id": request_id, "attempt": attempt,
                    "role": role, "error_type": type(exc).__name__,
                    "elapsed_seconds": time.monotonic() - started})
                if attempt == attempts or not _is_transient_llm_exception(exc):
                    raise
                time.sleep(min(2 * attempt, 8))
                continue
            raw = response.model_dump(mode="json", exclude_none=True)
            store.append_jsonl("history/command_requests.jsonl", {
                "event": "response", "id": request_id, "attempt": attempt, "role": role,
                "transport": "native_tools", "raw_response": raw,
                "usage": _response_usage(response) or {}, "elapsed_seconds": time.monotonic() - started})
            return extract_exchange(raw, protocol, request_id)
        raise RuntimeError("Tool request exhausted without a response")


def extract_exchange(raw, protocol, request_id):
    if protocol == "chat_completions":
        choice = raw["choices"][0]
        message = choice["message"]
        # Retain provider reasoning with the exact assistant tool message.
        items = [{k: v for k, v in message.items() if k in {"role", "content", "reasoning_content", "tool_calls"}}]
        calls = [{"id": c["id"], **c["function"]} for c in message.get("tool_calls", [])]
        complete = choice.get("finish_reason") in {"stop", "tool_calls"}
        text = message.get("content") or ""
        if message.get("refusal"):
            complete = False
    else:
        items = raw["output"]
        calls = [{"id": c["call_id"], "name": c["name"], "arguments": c["arguments"]}
                 for c in items if c["type"] == "function_call"]
        complete = raw.get("status") == "completed"
        parts = [part for item in items if item.get("type") == "message" for part in item.get("content", [])]
        text = "\n".join(part["text"] for part in parts if part.get("type") == "output_text")
        if any(part.get("type") == "refusal" for part in parts):
            complete = False
    ids = [c["id"] for c in calls]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise RuntimeError("Provider returned missing or duplicate tool-call IDs")
    return {"request_id": request_id, "protocol": protocol, "items": items,
            "calls": calls, "complete": complete, "text": text}


def ask_tools(client, registry, visible_tools, payload, store, *, exchange=None, outcome=None,
              allowed_operations=None):
    tracking = "exploration_state" in payload
    tools, models = tool_contracts(registry, visible_tools, allowed_operations=allowed_operations,
                                  with_draft=bool(payload.get("judgment_draft")), hypothesis_tracking=tracking,
                                  draft_id=(payload.get("judgment_draft") or {}).get("draft_id"))
    if tracking and payload["exploration_state"]["query_turns_remaining"] == 0:
        tools = [t for t in tools if t["name"] != "read_evidence"]
        models.pop("read_evidence", None)
    prompt = load_prompt("planner_tools.md")
    if tracking:
        prompt += "\n" + load_prompt("planner_exploration_state.md")
    reply = client.complete_tool_turn(system_prompt=prompt,
        payload=payload, tools=tools, exchange=exchange, outcome=outcome, store=store)

    return parse_tool_turn(reply, tools, models), reply


def parse_tool_turn(reply, tools, models):
    """Independent reads/computations may share a turn; judgment edits stay atomic."""

    def fail(code, message, path=None):
        raise OperationRejected([dict(path=path or [], code=code, message=message,
            remedy="Correct this operation or choose another. No operation executed or committed.",
            record_id=None)], exchange=reply)

    if not reply["complete"]:
        fail("incomplete_response", "Provider marked the response incomplete; no partial command may run.")
    calls = reply["calls"]
    updates = [c for c in calls if c["name"] == "update_hypotheses"]
    if len(calls) > 1 and updates:
        following = [c for c in calls if c["name"] != "update_hypotheses"]
        compute = following and all(c["name"].startswith("compute_") for c in following) and len(following) <= 6
        control = len(following) == 1 and following[0]["name"] in {"revise_judgment", "finalize", "submit_final"}
        if len(updates) != 1 or not (compute or control):
            fail("hypothesis_step_shape", "Use one hypothesis update with up to six computations, "
                 "or one formal judgment/final submission. Reads remain a separate turn.")
        try:
            update = parse_call(updates[0], tools, models)
        except OperationRejected as exc:
            raise OperationRejected(exc.issues, exchange=reply) from exc
        return HypothesisStep(update=update, update_call_id=updates[0]["id"], followup_calls=following)
    if len(calls) > 1:
        if all(c["name"].startswith("compute_") for c in calls) and len(calls) <= 6:
            computations = []
            for call in calls:
                try:
                    command = parse_call(call, tools, models)
                    computations.append(ComputeCall(call_id=call["id"], command=command.model_dump(mode="json"),
                                                    original_call=call))
                except OperationRejected as exc:
                    computations.append(ComputeCall(call_id=call["id"], issues=exc.issues, original_call=call))
            return ComputeBatch(calls=computations)
        if len(calls) > 4 or any(call["name"] != "read_evidence" for call in calls):
            fail("command_count", "Submit up to four reads, up to six independent computations, or one judgment/control "
                 "operation. Do not mix operation types. Nothing executed.")
        parsed = []
        for call in calls:
            try:
                parsed.append(ReadCall(call_id=call["id"], command=parse_call(call, tools, models)))
            except OperationRejected as exc:
                parsed.append(ReadCall(call_id=call["id"], issues=exc.issues))
        if sum(len(c.command.records) for c in parsed if c.command) > 6:
            fail("query_limit", "At most six record selections are allowed across a read turn.")
        return ReadBatch(calls=parsed)
    if not calls:
        fail("command_count", "Submit an operation; no executable call was returned.")
    try:
        return parse_call(calls[0], tools, models)
    except OperationRejected as exc:
        raise OperationRejected(exc.issues, exchange=reply) from exc


def parse_call(call, tools, models):
    def fail(code, message, path=None):
        raise OperationRejected([dict(path=path or [], code=code, message=message,
            remedy="Correct this operation or choose another. No operation executed or committed.", record_id=None)])

    if call["name"] not in models:
        fail("unknown_tool", "Function is not in the current tool set; load it through describe_tools.", ["name"])
    try:
        data = json.loads(call["arguments"])
    except (ValueError, TypeError):
        fail("invalid_json", "Function arguments are not complete JSON.", ["arguments"])
    schema = next(t["parameters"] for t in tools if t["name"] == call["name"])
    issues = schema_issues(schema, data)
    if issues:
        raise OperationRejected([{**i, "path": ["arguments", *i["path"]]} for i in issues])
    model, identity = models[call["name"]]
    try:
        command = model.model_validate({**data, **identity})
    except ValidationError as exc:
        raise OperationRejected([dict(path=["arguments", *e["loc"]], code=e["type"],
            message=e["msg"], remedy="Correct the indicated parameter.", record_id=None)
            for e in exc.errors(include_input=False, include_url=False)]) from exc
    return command
