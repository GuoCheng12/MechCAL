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
from mechcal.runtime.command_compact import eligible_tools
from mechcal.runtime.computational_review import response_requirements
from mechcal.runtime.command_action_policy import (
    ActionPolicy, turn_contract, BATCH_READS, READ_CALLS, READ_SELECTIONS, COMPUTE_CALLS, ASSESSMENT_CONTROLS,
)
from mechcal.runtime.command_reads import READ_KINDS, query_refs
from mechcal.runtime.command_schema import command_parameters
from mechcal.runtime.llm import (
    OpenAIJsonClient, _add_generation_args, _is_transient_llm_exception,
    _response_usage, create_completion,
)
from mechcal.runtime.prompts import load_prompt
from mechcal.schemas.commands import (
    ComputeBatch, ComputeCall, DescribeTools, DiscardJudgment, Finalize, ManageExperiments,
    ReadBatch, ReadCall, ReadEvidence, ReadContext, RepairJudgment, ReduceEvidence, CompareEvidence,
    HypothesisStep, ReviseJudgment, SubmitFinal, UpdateHypotheses, UpdateHypothesesWithFollowup, computation_model,
    AmendFinal, AmendFinalFields, LinkedAmendJudgment, LinkedReviseJudgment, LinkedRepairJudgment, UpdateHypothesesWithWorkflow,
    SubmitFinalAfterReview, ReviewFinal, reviewed_submission_model, UpdateHypothesesWithInvestigation,
)


DESCRIPTIONS = {
    "delegate_computation": "Delegate a computational objective and explicit scope to a Worker. "
                            "The Worker can diagnose and retry within that scope; Planner retains scientific judgments.",
    "reduce_evidence": "Compute an exact min/max/max_abs/sum over an explicitly selected array (up to 100000 rows), or read a numeric scalar. "
                       "Returns source paths, conditions and ties. Read-only; shares evidence-query allowance.",
    "compare_evidence": "Compute left/right ratio or left-right difference. Each operand can be a scalar or the min/max "
                        "of an explicit complete array. Returns exact sources and conditions; no scientific inference. "
                        "Read-only; shares evidence-query allowance.",
    "read_context": "Read up to three exact deferred task-context fields. No raw evidence query or computation. "
                    "Two context reads per unchanged context snapshot; never resets scientific read allowance.",
    "read_evidence": "Read exact case fields or inspect directory paths and byte sizes. "
                     "Copy directory child.read selections; use at most 8 paths per record. "
                     "Query pages may return fewer whole rows to fit; follow pages.next_offset. "
                     "Page large lists with offset/limit. Prefer report refs for a compact result. No computation.",
    "describe_tools": "Load and activate exact capability parameters, including missing-input routes.",
    "manage_experiments": "Retry or explicitly cancel saved pending computations. Retry never invents parameters; "
                          "for parameter changes cancel the old request and submit the corrected computation.",
    "revise_judgment": "Atomically revise related scientific arguments and ranking entries. Omitted records are retained.",
    "repair_judgment": "Patch the displayed uncommitted draft; null fields stay unchanged, including note/request_review. "
                       "Use citation_path_edits to correct individual paths without rewriting text or other citations. "
                       "Revalidate and commit the whole transaction.",
    "discard_judgment": "Discard an uncommitted draft without changing accepted arguments or ranking.",
    "finalize": "Submit the final narrative using the accepted ranking and evidence records.",
    "submit_final": "Deliver the exact saved, reviewed final draft without rewriting it.",
    "review_final": "Request or retry review of the exact saved draft_id. Never resubmit the narrative for a retry.",
    "update_hypotheses": "Save provisional confidence changes or explicitly retain the current assessment. "
                         "Not a formal argument or final ranking. May accompany the next computation in one response.",
    "amend_judgment": "Edit only changed fields of accepted arguments, optionally with linked ranking changes. "
                      "Null fields remain unchanged. Commit atomically; do not rewrite unaffected arguments.",
    "amend_final": "Use fields (path,value) to replace /answer or /limitations/<index>, bound to draft_id, without copying old text. "
                   "Rebuild against current claims and review the resulting exact draft; no automatic approval.",
}


def tool_contracts(registry, capability_ids, *, allowed_operations=None, with_draft=False,
                   hypothesis_tracking=False, draft_id=None, evidence_followup=False, review_workflow=False,
                   computational_review=False, strict_schema=True, task_context=False, stopping_review=None,
                   worker_tasks=False, investigation_policy=False):
    models = {name: (model, {"kind": name}) for name, model in (
        ("read_evidence", ReadEvidence), ("reduce_evidence", ReduceEvidence), ("compare_evidence", CompareEvidence),
        ("describe_tools", DescribeTools),
        ("manage_experiments", ManageExperiments),
        ("revise_judgment", ReviseJudgment), ("finalize", Finalize), ("submit_final", SubmitFinal),
        ("review_final", ReviewFinal),
        ("repair_judgment", RepairJudgment), ("discard_judgment", DiscardJudgment))}
    if task_context:
        models["read_context"] = (ReadContext, {"kind": "read_context"})
    if worker_tasks:
        from mechcal.schemas.worker_tasks import DelegateComputation
        models["delegate_computation"] = (DelegateComputation, {"kind": "delegate_computation"})
    if hypothesis_tracking:
        models["update_hypotheses"] = (UpdateHypothesesWithInvestigation if investigation_policy else
                                       UpdateHypothesesWithWorkflow if review_workflow else
                                       UpdateHypothesesWithFollowup if evidence_followup else UpdateHypotheses,
                                       {"kind": "update_hypotheses"})
    if review_workflow:
        models.update(revise_judgment=(LinkedReviseJudgment, {"kind": "revise_judgment"}),
                      repair_judgment=(LinkedRepairJudgment, {"kind": "repair_judgment"}),
                      amend_judgment=(LinkedAmendJudgment, {"kind": "amend_judgment"}),
                      amend_final=(AmendFinalFields, {"kind": "amend_final"}))
    if computational_review:
        submission = (reviewed_submission_model(response_requirements(stopping_review))
                      if stopping_review is not None else SubmitFinalAfterReview)
        models["submit_final"] = (submission, {"kind": "submit_final"})
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
        models[name] = (computation_model(registry.require(cid), with_question=evidence_followup),
                        {"kind": "compute", "capability_id": cid})
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
        schema = command_parameters(model, identity, strict=strict_schema)
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
                        parallel_tool_calls=any(t["name"] in READ_KINDS or t["name"].startswith("compute_")
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


def strict_tool_schema(client):
    settings = getattr(client, "settings", None)
    return not settings or (settings.api_protocol == "responses"
                            and urlparse(settings.base_url).hostname != "api.deepseek.com")


def ask_tools(client, registry, visible_tools, payload, store, *, exchange=None, outcome=None,
              allowed_operations=None):
    tracking = "exploration_state" in payload
    followup = "evidence_followup" in payload.get("exploration_state", {})
    workflow = "review_workflow" in payload
    tools, models = tool_contracts(registry, visible_tools, allowed_operations=allowed_operations,
                                  with_draft=bool(payload.get("judgment_draft")), hypothesis_tracking=tracking,
                                  evidence_followup=followup,
                                  review_workflow=workflow,
                                  computational_review="computational_review" in payload,
                                  investigation_policy="investigation_policy" in payload,
                                  stopping_review=payload.get("computational_review"),
                                  task_context=bool(payload.get("deferred_context")) and
                                               payload.get("context_reads_remaining", 0) > 0,
                                  worker_tasks=bool(payload.get("worker_delegation")),
                                  strict_schema=strict_tool_schema(client),
                                  draft_id=(payload.get("judgment_draft") or {}).get("draft_id"))
    if payload.get("operation_policy") is not None:
        tools, models = ActionPolicy.model_validate(payload["operation_policy"]).filter_tools(tools, models)
    elif tracking and payload["exploration_state"]["query_turns_remaining"] == 0:
        tools = [t for t in tools if t["name"] not in READ_KINDS]
        models = {name: model for name, model in models.items() if name not in READ_KINDS}
    tools, models = eligible_tools(tools, models, payload)
    from mechcal.runtime.command_task_context import planner_task_tools
    tools, models = planner_task_tools(tools, models, payload)
    if payload.get("context_profile") == "compact_v1":
        from mechcal.agents.command_shape_repair import retry_local_command
        repaired = retry_local_command(client, tools, models, exchange, outcome, store, parse_tool_turn)
        if repaired is not None:
            return repaired
    task_interface = payload.get("planner_interface", "full") != "full" and "read_context" in models
    prompt = load_prompt("planner_task_edit.md" if task_interface else "planner_tools.md")
    if payload.get("worker_delegation") and not task_interface:
        prompt += "\n" + load_prompt("planner_worker_tasks.md")
    if any(t["name"] in {"reduce_evidence", "compare_evidence"} for t in tools):
        prompt += "\n" + load_prompt("numeric_evidence.md")
    if tracking and not task_interface:
        prompt += "\n" + load_prompt("planner_exploration_state.md")
    if followup and not task_interface:
        prompt += "\n" + load_prompt("planner_evidence_followup.md")
    if workflow and not task_interface:
        prompt += "\n" + load_prompt("planner_review_workflow.md")
    if "computational_review" in payload and not task_interface:
        prompt += "\n" + load_prompt("computational_mission.md")
    if "investigation_policy" in payload:
        prompt += "\n" + load_prompt("planner_investigation.md")
    groups = turn_contract(models)
    if task_interface:
        groups["batch_read_operations"] = [n for n in groups["batch_read_operations"] if n in models]
        groups["hypothesis_followup_controls"] = ([n for n in groups["hypothesis_followup_controls"] if n in models]
                                                   if "update_hypotheses" in models else [])
        groups["compute_calls"] = 0
        groups["rule"] = ("One editing control or up to four read-only calls; never mix reads with mutations. "
                          "update_hypotheses, if available, may precede ONE listed control. "
                          "For scientific operations, read_context(fields=['tool_catalog']) restores the full interface.")
    reply = client.complete_tool_turn(system_prompt=prompt,
        payload={**payload, "operation_groups": groups}, tools=tools, exchange=exchange, outcome=outcome, store=store)

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
        compute = following and all(c["name"].startswith("compute_") for c in following) and len(following) <= COMPUTE_CALLS
        control = len(following) == 1 and following[0]["name"] in ASSESSMENT_CONTROLS
        if len(updates) != 1 or not (compute or control):
            fail("hypothesis_step_shape", turn_contract()["rule"] + " Allowed following controls: " +
                 ", ".join(sorted(ASSESSMENT_CONTROLS)))
        try:
            update = parse_call(updates[0], tools, models)
        except OperationRejected as exc:
            raise OperationRejected(exc.issues, exchange=reply) from exc
        return HypothesisStep(update=update, update_call_id=updates[0]["id"], followup_calls=following)
    if len(calls) > 1:
        if all(c["name"].startswith("compute_") for c in calls) and len(calls) <= COMPUTE_CALLS:
            computations = []
            for call in calls:
                try:
                    command = parse_call(call, tools, models)
                    computations.append(ComputeCall(call_id=call["id"], command=command.model_dump(mode="json"),
                                                    original_call=call))
                except OperationRejected as exc:
                    computations.append(ComputeCall(call_id=call["id"], issues=exc.issues, original_call=call))
            return ComputeBatch(calls=computations)
        if len(calls) > READ_CALLS or any(call["name"] not in BATCH_READS for call in calls):
            fail("command_count", turn_contract()["rule"] + " Nothing executed.")
        parsed = []
        for call in calls:
            try:
                parsed.append(ReadCall(call_id=call["id"], command=parse_call(call, tools, models)))
            except OperationRejected as exc:
                parsed.append(ReadCall(call_id=call["id"], issues=exc.issues))
        if sum(len(query_refs(c.command)) for c in parsed if c.command) > READ_SELECTIONS:
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
        discovery = ("read_context(fields=['tool_catalog']) to restore the scientific interface"
                     if "read_context" in models and "describe_tools" not in models else "describe_tools")
        fail("unknown_tool", "Function is not in the current tool set; use " + discovery + ".", ["name"])
    try:
        data = json.loads(call["arguments"])
    except (ValueError, TypeError):
        fail("invalid_json", "Function arguments are not complete JSON.", ["arguments"])
    model, identity = models[call["name"]]
    # Only schema-declared delta omissions are allowed. Full scientific records
    # and execution parameters stay required, even if Python has legacy defaults.
    schema = command_parameters(model, identity, strict=False)
    issues = schema_issues(schema, data)
    if issues:
        raise OperationRejected([{**i, "path": ["arguments", *i["path"]]} for i in issues])
    try:
        command = model.model_validate({**data, **identity})
    except ValidationError as exc:
        raise OperationRejected([dict(path=["arguments", *e["loc"]], code=e["type"],
            message=e["msg"], remedy="Correct the indicated parameter.", record_id=None)
            for e in exc.errors(include_input=False, include_url=False)]) from exc
    return AmendFinal.model_validate(command.model_dump()) if isinstance(command, AmendFinalFields) else command
