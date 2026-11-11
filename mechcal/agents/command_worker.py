"""One durable Worker turn: scoped read-only retrieval or final prose."""

from mechcal.agents.command_tools import NativeCommandClient, parse_tool_turn, tool_contracts, strict_tool_schema
from mechcal.agents.commentary_delivery import delivery_issue
from mechcal.runtime.command_context import scope_card, scoped_directory, share_text, record_table
from mechcal.runtime.command_memory import working_reads, json_bytes
from mechcal.runtime.command_operations import OperationRejected, protocol, record_identity, record_index, records
from mechcal.runtime.command_reads import batch_read_results, merge_handles, query_results, READ_KINDS
from mechcal.runtime.command_events import CommandEvent, write_event
from mechcal.runtime.prompts import load_prompt
from mechcal.runtime.command_schema import native_format
from mechcal.runtime import review_link_repair
from mechcal.runtime.command_compact import commentary_view
from mechcal.runtime import command_read_handoff as handoff
from mechcal.runtime import command_query_state as query_state
from mechcal.runtime import command_numeric as numeric
from mechcal.runtime.command_review_wire import review_wire, task_binding
from mechcal.runtime.command_context_memory import bind_scoped_context
from mechcal.runtime.command_action_policy import turn_contract
from mechcal.schemas.commands import ReadBatch, RepairReviewLinks


def rejected_commentary(state, reply, outcome, store, role):
    """Keep a local failed call in its review job; never turn it into approval."""
    corrections = state.get("corrections", 0)
    # The latest exchange now carries an error, not the previous query data.
    # Keep all read handles, but stop suppressing their values as already sent.
    session = {**state, "exchange": reply, "outcome": outcome, "corrections": corrections + 1,
               "latest_handles": [], "latest_source_reads": []}
    store.append_jsonl("history/commentary_recovery.jsonl", {
        "role": role, "request_id": reply["request_id"], "correction": corrections + 1,
        "status": "retry" if corrections < 2 else "exhausted", "issues": outcome.get("issues", [])})
    if corrections >= 2:
        return {"status": "unavailable", "issues": outcome.get("issues", []),
                "reason": "Two local corrections failed. No review was committed.",
                "failed_session": session}, None
    return None, session


def worker_scope(case, result_id, request, *, index=None):
    available = records(case, index=index)
    result = protocol(case)["results"][result_id]
    refs = {"input:smiles", result_id, "report:" + result_id}
    refs.update(unit["evidence_id"] for unit in result.get("evidence_units", []) if unit.get("evidence_id"))
    # Only explicit input/output artifact identities, never a transitive walk of
    # all strings in the result (which could admit unrelated records).
    refs.update(request.get("input_refs", []))
    if request.get("dispatch_id"):
        refs.add(request["dispatch_id"])
    refs.update(result.get("artifact_refs", []))
    refs.update(a["artifact_id"] for a in result.get("artifact_updates", []) if a.get("artifact_id"))
    return {ref: available[ref] for ref in sorted(refs) if ref in available}


def worker_turn(client, case, payload, store, *, compact_context=False):
    p = protocol(case)
    rid = p["pending_result"]
    index = record_index(case)
    scope = worker_scope(case, rid, payload["request"], index=index)
    state = p.get("worker_session") or {"result_id": rid, "queries": 0}
    if state["result_id"] != rid:
        raise RuntimeError("Worker retrieval session belongs to another computation")
    feedback, session = scoped_commentary_turn(client, case, payload, store, scope=scope, state=state,
        role="worker", prompt_name="command_worker_reads.md", index=index, compact_context=compact_context)
    if feedback is not None:
        feedback["refs"] = [rid]
    return feedback, session


def scoped_commentary_turn(client, case, payload, store, *, scope, state, role, prompt_name, index=None,
                           completion=None, describe_tools=None, compact_context=False, review_task_context=True):
    """Shared bounded reads; the agent returns prose, never mutations or computations."""
    remaining = max(0, 2 - state["queries"])
    proofreading = payload.get("sections", {}).get("delivery_task", {}).get("mode") == "proofread"
    index = record_index(case, index)
    allowed = sorted(READ_KINDS) if remaining else []
    if remaining and describe_tools is not None:
        allowed.append("describe_tools")
    tools, models = tool_contracts(None, [], allowed_operations=allowed, strict_schema=strict_tool_schema(client))
    for tool in tools:
        if tool["name"] == "describe_tools":
            tool["description"] = ("Read registry definitions for this role. Does not execute a scientific tool."
                                   if compact_context else
                                   "Read registry definitions only. Does not execute or activate any scientific tool.")
    if completion is not None:
        name, model, accept = completion
        schema = native_format(model, strict=strict_tool_schema(client))["json_schema"]["schema"]
        schema["properties"].pop("kind")
        schema["required"].remove("kind")
        tools.append({"name": name, "description": "Save concise review prose and the indicated advisory records. No scientific mutation.",
                      "parameters": schema})
        models[name] = (model, {"kind": name})
    repair = state.get("review_link_repair")
    if repair is not None:
        if completion is None or name != "record_review":
            raise RuntimeError("Review link correction requires its original review contract")
        schema = native_format(RepairReviewLinks)["json_schema"]["schema"]
        schema["properties"].pop("kind")
        schema["required"].remove("kind")
        tools = [{"name": "repair_review_links", "description": "Correct only IDs and source associations in the saved review.",
                  "parameters": schema}]
        models = {"repair_review_links": (RepairReviewLinks, {"kind": "repair_review_links"})}
    if proofreading:
        directory, paths = None, {}
        directory_table = {"columns": ["ref"], "rows": [[ref] for ref in sorted(scope)]}
    else:
        directory, paths = scoped_directory([scope_card(record_identity(case, ref, index=index), value,
                                                        producers=[rid for rid in index.artifact_producers.get(ref, []) if rid in scope])
                                            for ref, value in scope.items()])
        directory_table = None
    inherited, handoff_issues = handoff.retained(case, allowed_refs=scope, index=index) if compact_context else ([], [])
    reports = [payload["result"]] if "result" in payload else payload.get("results", [])
    display_focus = {r["source_result_id"] for r in reports if r.get("source_result_id")}
    focus = display_focus or set(scope)
    if compact_context and "review_focus_refs" in payload:
        focus = set(payload["review_focus_refs"])
        display_focus = focus
        from mechcal.runtime.command_review_delta import retained_reads
        inherited = merge_handles(inherited, retained_reads(case, role, index, scope))
    if proofreading:
        # Versioned prior facts are already joined to their conditions in
        # review_facts. Only new reads from this review need another value cache.
        inherited = []
    if completion is not None and name == "report_evidence_impacts":
        allowed_sources = set(payload["association_source_ids"])
        for row in directory:
            row["impact_source"] = row["ref"] in allowed_sources
        payload = {k: v for k, v in payload.items() if k != "association_source_ids"}
    material = {**payload, "read_scope": directory_table if directory_table is not None else record_table(directory),
                "read_path_sets": paths,
                "directory_format": "read_scope is a table: columns name row values; defaults apply to every row. "
                                    "path_set selects exact paths in read_path_sets. Reads require remaining query turns; "
                                    "with none left, complete from retained information or explicitly state its limits.",
                "query_turns_remaining": remaining,
                "local_corrections_remaining": max(0, 2 - state.get("corrections", 0)),
                "read_constraints": {"paths_per_record": 8, "records_per_turn": 6, "calls_per_turn": 4,
                                     "bytes_per_turn": 24000, "list_page_rows": 32},
                "working_memory": working_reads(merge_handles(inherited, state.get("read_handles", [])), scope,
                                                latest_rows=merge_handles(state.get("latest_handles", []),
                                                                          state.get("latest_source_reads", [])),
                                                focus_refs=focus if compact_context else None,
                                                explicit_handles=state.get("read_handles", []))}
    if protocol(case).get("numeric_checks"):
        material["numeric_checks"] = numeric.view(case, scope=scope, index=index,
            exclude_ids=numeric.outcome_ids(state.get("outcome", {})) if state.get("exchange") else ())
    if compact_context:
        last_query = query_state.session_outcome(state)
        receipt_case = query_state.remember(case, last_query, role=role, index=index)
        navigation_refs = (focus | {h["ref"] for h in state.get("read_handles", [])}) if proofreading else scope
        material["query_state"] = query_state.view(receipt_case, allowed_refs=navigation_refs,
            latest_outcome=last_query if state.get("exchange") else None)
    if handoff_issues:
        material["source_read_issues"] = handoff_issues
    method = getattr(client, "complete_tool_turn", None)
    prompt = load_prompt(prompt_name)
    if remaining:
        prompt += "\n" + load_prompt("numeric_evidence.md")
    if "computational_review" in payload.get("sections", payload):
        prompt += "\n" + load_prompt("computational_mission.md")
    if repair is not None:
        material = review_link_repair.repair_material(payload, repair)
        prompt = load_prompt("review_link_repair.md")
    elif compact_context:
        material = commentary_view(material)
    material["response_contract"] = {
        "delivery": tools[0]["name"] if repair is not None else name if completion is not None else "natural_language",
        "query_tools": [t["name"] for t in tools if t["name"] in READ_KINDS | {"describe_tools"}],
        "call_groups": turn_contract(models),
        "policy": "Queries use native tool calls only. Text describing a call does not execute it. "
                  "When no query tools remain, deliver from retained facts and state unresolved limits."}
    if (state.get("outcome") or {}).get("status") == "rejected" and not (state.get("exchange") or {}).get("calls"):
        # There is no native call ID to attach a tool-result message to. The
        # correction must still reach text-only and noncompact continuations.
        material["delivery_error"] = state["outcome"]
    if compact_context and review_task_context and repair is None:
        material, index, scope, state = bind_scoped_context(
            index, scope, state, material, role=role,
            focus_refs=display_focus | {h["ref"] for h in state.get("read_handles", [])[-6:]}, store=store)
        # A prior metadata read can outlive the latest exchange, just like a
        # source read. Retain it only while its exact task-local record exists.
        if state.get("task_context_record"):
            material["working_memory"] = working_reads(
                merge_handles(inherited, state.get("read_handles", [])), scope,
                latest_rows=merge_handles(state.get("latest_handles", []), state.get("latest_source_reads", [])),
                focus_refs=focus, explicit_handles=state.get("read_handles", []))
    wire_outcome = state.get("outcome")
    if compact_context and review_task_context and repair is None and role in {"reviewer", "final_reviewer"}:
        binding = task_binding(case.case_id, payload, state, role,
                               sources={ref: row for ref, row in scope.items() if ref not in index.contexts})
        state = {**state, "review_task": binding}
        original_bytes = json_bytes(material)
        material, wire_outcome = review_wire(material, binding=binding, state=state, scope=scope)
        store.append_jsonl("history/review_context.jsonl", {
            "task_id": binding["task_id"], "role": role, "query_turns": state["queries"],
            "phase": material["review_task"]["phase"], "wire_bytes": json_bytes(material),
            "original_material_bytes": original_bytes,
            "shared_values": len(material.get("review_values", {})), "scientific_mutation": False})
    kw = dict(system_prompt=prompt, payload=share_text(material),
              tools=tools, exchange=state.get("exchange") if repair is None else None,
              outcome=wire_outcome if repair is None else None,
              store=store, role=role, allow_text=True)
    reply = method(**kw) if method else NativeCommandClient.complete_tool_turn(client, **kw)
    if reply["complete"] and not reply["calls"] and reply.get("text", "").strip() and completion is None:
        issue = delivery_issue(reply["text"], can_read=bool(remaining))
        if issue is not None:
            return rejected_commentary(state, reply, {"status": "rejected", "executed": False,
                "committed": False, "issues": [issue]}, store, role)
        return {"status": "available", "text": reply["text"],
                "read_record_ids": sorted(set(state.get("read_record_ids", [])))}, None
    if completion is None and (not reply["complete"] or (not remaining and not reply["calls"])):
        return {"status": "unavailable",
                "reason": "No complete commentary within the bounded read allowance."}, None
    command = None
    try:
        command = parse_tool_turn(reply, tools, models)
        if repair is not None:
            command = review_link_repair.apply_links(model, repair["proposal"], command)
        if completion is not None and command.kind == name:
            issue = delivery_issue(getattr(command, "text", ""), can_read=bool(remaining))
            if issue is not None:
                raise OperationRejected([{**issue, "path": ["arguments", "text"]}])
            feedback = accept(command)
            feedback["read_record_ids"] = sorted(set(state.get("read_record_ids", [])))
            log = "review_issue_completions" if name == "record_review" else "impact_review_completions"
            store.append_jsonl(f"history/{log}.jsonl", {
                "role": role, "request_id": reply["request_id"], "completion": name,
                "outcome": {"status": "recorded", "scientific_mutation": False,
                            "local_link_repair": repair is not None}})
            return feedback, None
        if command.kind == "describe_tools" and describe_tools is not None:
            outcome, handles = {"status": "described", "tools": describe_tools(command)}, []
        elif isinstance(command, ReadBatch):
            outcome, handles = batch_read_results(case, command, allowed_refs=scope, index=index)
        else:
            outcome, handles = query_results(case, command, allowed_refs=scope, index=index)
    except OperationRejected as exc:
        outcome = {"status": "rejected", "executed": False, "committed": False, "issues": exc.issues}
        handles = []
        if repair is None and completion is not None and review_link_repair.repairable(command, exc.issues):
            session = {**state, "review_link_repair": {"proposal": command.model_dump(mode="json"), "issues": exc.issues},
                       "exchange": reply, "outcome": outcome}
            store.append_jsonl("history/review_link_repairs.jsonl", {
                "role": role, "request_id": reply["request_id"], "status": "pending", "issues": exc.issues})
            return None, session
    if completion is not None and outcome["status"] == "rejected" and repair is None:
        if remaining and any(call["name"] in READ_KINDS | {"describe_tools"} for call in reply["calls"]):
            state = {**state, "queries": state["queries"] + 1}
        if not remaining:
            outcome["issues"] = [{**issue, "remedy": f"No reads remain. Use {name} with the retained information; "
                                  "state unresolved limits rather than inventing evidence."}
                                 for issue in outcome.get("issues", [])]
        return rejected_commentary(state, reply, outcome, store, role)
    if repair is not None:
        store.append_jsonl("history/review_link_repairs.jsonl", {
            "role": role, "request_id": reply["request_id"], "status": "unavailable", "issues": outcome.get("issues", [])})
        return {"status": "unavailable", "issues": outcome.get("issues", []),
                "reason": "One local review-link correction failed; the review was not accepted."}, None
    if not remaining:
        return {"status": "unavailable", "issues": outcome.get("issues", []),
                "reason": "No valid advisory associations within the bounded allowance."}, None
    session = {**state, "queries": state["queries"] + 1, "exchange": reply, "outcome": outcome,
               "read_handles": merge_handles(state.get("read_handles", []), handles), "latest_handles": handles,
               "read_record_ids": sorted((set(state.get("read_record_ids", [])) | {h["ref"] for h in handles}
                                          | numeric.outcome_refs(outcome)) - set(index.contexts))}
    if compact_context:
        session["latest_source_reads"] = handoff.selection_handles(outcome)
        session["source_reads"] = merge_handles(state.get("source_reads", []), session["latest_source_reads"])
    store.append_jsonl(f"history/{'worker' if role == 'worker' else 'reviewer'}_reads.jsonl", session)
    write_event(store, CommandEvent(event_id=f"read:{reply['request_id']}:{session['queries']}", kind="read", role=role,
        status=outcome["status"], details={"request_id": reply["request_id"],
            "queries": session["queries"], "outcome": outcome}).model_dump())
    return None, session
