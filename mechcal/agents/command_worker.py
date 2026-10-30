"""One durable Worker turn: scoped read-only retrieval or final prose."""

from mechcal.agents.command_tools import NativeCommandClient, parse_tool_turn, tool_contracts
from mechcal.runtime.command_memory import read_directory, working_reads
from mechcal.runtime.command_operations import OperationRejected, protocol, record_identity, records
from mechcal.runtime.command_reads import batch_read_results, merge_handles, read_selection_results
from mechcal.runtime.command_events import CommandEvent, write_event
from mechcal.runtime.prompts import load_prompt
from mechcal.schemas.commands import ReadBatch


def worker_scope(case, result_id, request):
    available = records(case)
    result = protocol(case)["results"][result_id]
    refs = {"input:smiles", result_id, "report:" + result_id}
    refs.update(unit["evidence_id"] for unit in result.get("evidence_units", []) if unit.get("evidence_id"))
    # Only explicit input/output artifact identities, never a transitive walk of
    # all strings in the result (which could admit unrelated records).
    refs.update(request.get("input_refs", []))
    refs.update(result.get("artifact_refs", []))
    refs.update(a["artifact_id"] for a in result.get("artifact_updates", []) if a.get("artifact_id"))
    return {ref: available[ref] for ref in sorted(refs) if ref in available}


def worker_turn(client, case, payload, store):
    p = protocol(case)
    rid = p["pending_result"]
    scope = worker_scope(case, rid, payload["request"])
    state = p.get("worker_session") or {"result_id": rid, "queries": 0}
    if state["result_id"] != rid:
        raise RuntimeError("Worker retrieval session belongs to another computation")
    feedback, session = scoped_commentary_turn(client, case, payload, store, scope=scope, state=state,
        role="worker", prompt_name="command_worker_reads.md")
    if feedback is not None:
        feedback["refs"] = [rid]
    return feedback, session


def scoped_commentary_turn(client, case, payload, store, *, scope, state, role, prompt_name):
    """Shared bounded reads; the agent returns prose, never mutations or computations."""
    remaining = max(0, 2 - state["queries"])
    tools, models = tool_contracts(None, [], allowed_operations=["read_evidence"] if remaining else [])
    material = {**payload, "read_scope": [{**record_identity(case, ref), **read_directory(value)}
                                         for ref, value in scope.items()],
                "query_turns_remaining": remaining,
                "read_constraints": {"paths_per_record": 8, "records_per_turn": 6, "calls_per_turn": 4,
                                     "bytes_per_turn": 24000, "list_page_rows": 32},
                "working_memory": working_reads(state.get("read_handles", []), scope,
                                                latest_rows=state.get("latest_handles", []))}
    method = getattr(client, "complete_tool_turn", None)
    kw = dict(system_prompt=load_prompt(prompt_name), payload=material,
              tools=tools, exchange=state.get("exchange"), outcome=state.get("outcome"),
              store=store, role=role, allow_text=True)
    reply = method(**kw) if method else NativeCommandClient.complete_tool_turn(client, **kw)
    if reply["complete"] and not reply["calls"] and reply.get("text", "").strip():
        return {"status": "available", "text": reply["text"],
                "read_record_ids": sorted(set(state.get("read_record_ids", [])))}, None
    if not remaining or not reply["complete"]:
        return {"status": "unavailable",
                "reason": "No complete commentary within the bounded read allowance."}, None
    try:
        command = parse_tool_turn(reply, tools, models)
        if isinstance(command, ReadBatch):
            outcome, handles = batch_read_results(case, command, allowed_refs=scope)
        else:
            outcome, handles = read_selection_results(case, command, allowed_refs=scope)
    except OperationRejected as exc:
        outcome = {"status": "rejected", "executed": False, "committed": False, "issues": exc.issues}
        handles = []
    session = {**state, "queries": state["queries"] + 1, "exchange": reply, "outcome": outcome,
               "read_handles": merge_handles(state.get("read_handles", []), handles), "latest_handles": handles,
               "read_record_ids": sorted(set(state.get("read_record_ids", [])) | {h["ref"] for h in handles})}
    store.append_jsonl(f"history/{'worker' if role == 'worker' else 'reviewer'}_reads.jsonl", session)
    write_event(store, CommandEvent(event_id=f"read:{reply['request_id']}:{session['queries']}", kind="read", role=role,
        status=outcome["status"], details={"request_id": reply["request_id"],
            "queries": session["queries"], "outcome": outcome}).model_dump())
    return None, session
