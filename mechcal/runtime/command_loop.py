"""Opt-in native command loop. Durable computations and advisory science review."""

import fcntl
import json
import signal
import threading
import time

from mechcal.agents.command_client import ask, ask_reviewer_text, ask_worker_text
from mechcal.agents.command_tools import ask_tools, parse_tool_turn, tool_contracts
from mechcal.agents.command_worker import scoped_commentary_turn, worker_turn
from mechcal.agents.observation_view import observation_view
from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime.command_operations import (
    OperationRejected, check_owner, check_refs, prepare_computation,
    protocol, put, record_identity, records, reject, revise,
)
from mechcal.runtime.command_memory import (
    READ_BYTES, digest, json_bytes, record_layout, remember_report, remember_values,
    result_index, result_report, working_reads,
)
from mechcal.runtime.command_draft import draft_view, make_draft, repair, require_draft
from mechcal.runtime.command_events import CommandEvent, stop_reason, track_state, write_event
from mechcal.runtime import command_hypotheses as hypotheses
from mechcal.runtime.command_delivery import final_draft, reviewed, saved_final
from mechcal.runtime.command_reads import batch_read_results, merge_handles, read_selection_results
from mechcal.runtime.command_review import (
    candidate_overview, review_basis, review_citations, review_execution_context,
    review_observations, reviewer_scope,
)
from mechcal.runtime.command_experiments import (
    finish_member, manage_experiments, next_member, pending_experiments,
    retain_rejected_computations, stage_batch,
)
from mechcal.runtime.question_loop import configuration_fingerprint
from mechcal.schemas import AgentExecutionPlan, CaseRun, DispatchRequest, ToolExecutionResult
from mechcal.schemas.commands import ComputeBatch, HypothesisStep, ReadBatch, envelope_model, tool_argument_model


def save(case, store):
    # All accepted scientific mutations and their replay cursor share one atomic
    # checkpoint. History files are diagnostic, not the source of truth.
    case = hypotheses.complete_step(case)
    previous_path = store.root / "command_checkpoint.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
    case = track_state(case, previous)
    projections = []
    receipt = protocol(case).get("receipt")
    if receipt:
        relative = f"operations/{receipt['turn']:06d}.json"
        existing = store.root / relative
        if existing.exists() and json.loads(existing.read_text()) != receipt:
            raise RuntimeError("Operation receipt conflicts with its committed checkpoint.")
        projections.append((relative, receipt))
    p = protocol(case)
    if p.get("judgment_draft"):
        draft = p["judgment_draft"]
        relative = f"judgment_drafts/{draft['draft_id']}.json"
        existing = store.root / relative
        if existing.exists() and json.loads(existing.read_text()) != draft:
            raise RuntimeError("Stored judgment draft conflicts with its checkpoint.")
        projections.append((relative, draft))
    exchange = p.get("latest_exchange")
    if (exchange and p["phase"] in {"choose", "done"}
            and p.get("outcome_request_id") == exchange["request_id"]):
        # Separate durable completion from initial command admission. Resume must
        # never interpret an accepted compute request as a completed calculation.
        relative = f"operation_results/{exchange['request_id']}.json"
        completed = {
            "request_id": exchange["request_id"],
            "call_ids": [c["id"] for c in exchange["calls"]], "outcome": p.get("outcome"),
        }
        existing = store.root / relative
        if existing.exists() and json.loads(existing.read_text()) != completed:
            raise RuntimeError("Completed tool message conflicts with its committed checkpoint.")
        projections.append((relative, completed))
    store.write_json("command_checkpoint.json", case)
    write_event(store, protocol(case)["state_event"])
    exploration_event = (protocol(case).get("exploration") or {}).get("event")
    if exploration_event:
        write_event(store, exploration_event)
    for relative, data in projections:
        if not (store.root / relative).exists():
            store.write_json(relative, data)
        if relative.startswith("operations/"):
            write_event(store, CommandEvent(event_id="operation:" + str(data["turn"]),
                kind="operation", role="planner", status=data["status"], details=data).model_dump())
    store.write_case_run(case)
    if case.final_answer:
        store.write_final_answer(case.final_answer)
    return case


def _catalog(o, case):
    allowed = case.runtime.get("ablation", {}).get("enabled_capability_owners", ["macro", "microscopic"])
    registry = o.deps.capability_registry.for_owner_agents(tuple(allowed))
    return registry.compact_catalog(case.artifact_manifest, amesp_enabled=o.config.enable_amesp)


def planner_payload(o, case, *, native=False):
    p = protocol(case)
    active_refs = {ref for row in [*p.get("arguments", {}).values(), *p.get("candidates", {}).values()]
                   for field in ("refs", "limitation_refs") for ref in row.get(field, [])}
    if o.config.command_hypothesis_tracking:
        active_refs.update(ref for row in hypotheses.state(case).entries.values()
                           for ref in [*row.refs, *row.limitation_refs])
    outcome = p.get("outcome") or {}
    payload = {
        "stage": "planner_command", "smiles": case.input.smiles,
        "mechanism_pool": list(MECHANISM_POOL), "task": case.input.user_query,
        "ranking": list(p.get("ranking", {}).values()),
        "candidate_overview": candidate_overview(case),
        "pending_experiments": pending_experiments(case),
        "judgment_draft": draft_view(case),
        "judgment_status": {
            "revision": p.get("revision", 0), "ranked_mechanisms": len(p.get("ranking", {})),
            "delivery_available": len(case.portfolio.sorted_hypotheses()) >= 3 and bool(p.get("ranking"))
                                  and not p.get("judgment_draft"),
            "last_note": p.get("note", ""),
            "meaning": "Accepted records persist after reads. Delivery availability is structural, "
                       "not a judgment of scientific sufficiency. Planner decides whether to finalize.",
        },
        "argument_index": [{"ref": k, "label": v["label"], "refs": v["refs"],
                            "limitation_refs": v.get("limitation_refs", []),
                            "text_preview": v["text"][:160]} for k, v in p.get("arguments", {}).items()],
        "evidence_index": [{"ref": e.evidence_id, "tool": e.capability_id, "status": e.status,
                            "summary_preview": e.summary[:160]} for e in case.evidence_ledger.items],
        "result_index": [result_index(k, v) for k, v in p.get("results", {}).items()],
        "artifact_index": [{"ref": a.artifact_id, "kind": a.kind, "status": a.status}
                           for a in case.artifact_manifest.artifacts],
        "action_index": [{"ref": k, "tool": a["capability_id"], "purpose_preview": a["task"][:160]}
                         for k, a in p.get("actions", {}).items()],
        "explanation_index": [{"ref": "explanation:" + rid, "source_result_id": rid,
                               "status": e["status"], "bytes": json_bytes({"source_result_id": rid, **e}),
                               "directory": record_layout({"source_result_id": rid, **e})["children"]}
                              for rid, e in p.get("explanations", {}).items()],
        "read_limit_bytes": READ_BYTES,
        "working_memory": working_reads(p.get("read_memory", []), records(case), active_refs=active_refs,
                                        latest_rows=p.get("latest_read_handles", [])
                                        if outcome.get("status") in {"read", "partial_read", "read_batch"}
                                        or "result_report" in outcome else []),
        "tool_catalog": _catalog(o, case),
        "last_operation": p.get("outcome"),
        "review_feedback": p.get("review"),
        "tools_allowed": case.runtime["run_limits"].get("scientific_tools_allowed", True),
        "loaded_capabilities": p.get("visible_tools", []),
        "description_batches_remaining": max(0, 2 - p.get("description_batches", 0)),
        "final_draft": ({"draft_id": p["final_draft"]["draft_id"],
                         "revision": p["final_draft"]["revision"],
                         "review_status": (p.get("final_review") or {}).get("status"),
                         "read_ref": p["final_draft"]["draft_id"]} if p.get("final_draft") else None),
    }
    if o.config.command_hypothesis_tracking:
        payload["exploration_state"] = hypotheses.view(case)
        payload.pop("candidate_overview", None)
    # A fixture/resume may have a delivered result but no native call to replay.
    # Deduplication is valid only when that result actually has a message route.
    if native and p.get("latest_exchange"):
        payload.pop("last_operation", None)
    known = records(case)
    for name in ("argument_index", "evidence_index", "result_index", "artifact_index", "action_index", "explanation_index"):
        if any(item["ref"] not in known for item in payload[name]):
            raise RuntimeError("Advertised record is not queryable: " + name)
        for item in payload[name]:
            item.update(record_identity(case, item["ref"]))
    for item in payload["pending_experiments"]:
        item.update(record_identity(case, item["experiment_id"]))
    return payload


def _local_error(case, issues, store, *, exchange=None, judgment_draft=None):
    if judgment_draft is not None:
        case = put(case, judgment_draft=judgment_draft)
    p = protocol(case)
    failures = p.get("local_errors", 0) + 1
    outcome = {"status": "rejected", "executed": False, "committed": False, "issues": issues}
    exchange = exchange or p.get("latest_exchange")
    if p.get("exploration") is not None:
        case = hypotheses.record_read_attempt(case, exchange)
    case = retain_rejected_computations(case, exchange, issues)
    if exchange:
        outcome["call_ids"] = [c["id"] for c in exchange["calls"]]
    receipt = {"turn": p["turn"], "revision": p["revision"], **outcome}
    case = put(case, outcome=outcome, pending_command=None, local_errors=failures,
               turn=p["turn"] + 1, phase="choose", receipt=receipt, latest_exchange=exchange,
               outcome_request_id=exchange["request_id"] if exchange else None)
    # These are operation failures, not run failures. Explicit experiment
    # allowances or user stop requests bound model calls, not scientific fields.
    return save(case, store)


def _outcome(case, store, status, **details):
    p = protocol(case)
    result = {"status": status, **details}
    return put(case, outcome=result, local_errors=0, pending_command=None, turn=p["turn"] + 1,
               outcome_request_id=(p.get("latest_exchange") or {}).get("request_id"),
               receipt={"turn": p["turn"], "revision": p["revision"], **result})


def _apply(o, case, command, store):
    if o.config.command_hypothesis_tracking:
        case = hypotheses.sync(case)
        hypotheses.guard(case, command)
        if command.kind in {"read_evidence", "read_batch"}:
            case = hypotheses.record_read_attempt(case, protocol(case).get("latest_exchange"))
    p = protocol(case)
    if command.kind in {"update_hypotheses", "hypothesis_step"}:
        if not o.config.command_hypothesis_tracking:
            reject([], "operation_disabled", "Exploration-state tracking is disabled.", "Use the enabled operations.")
        update = command.update if command.kind == "hypothesis_step" else command
        candidate, result = hypotheses.update(case, update)
        if command.kind == "update_hypotheses":
            return _outcome(candidate, store, **result)
        return put(candidate, phase="hypothesis_next", pending_command=None,
                   hypothesis_step={"update_call_id": command.update_call_id, "update_result": result,
                                    "followup_calls": command.followup_calls})
    if command.kind == "compute_batch":
        candidate = stage_batch(o, case, command)
        return put(_outcome(candidate, store, "batch_accepted", executed=False), outcome_request_id=None)
    if command.kind == "manage_experiments":
        candidate = manage_experiments(o, case, command)
        candidate = _outcome(candidate, store, "experiments_" + command.action, experiment_ids=command.experiment_ids)
        return put(candidate, outcome_request_id=None) if command.action == "retry" else candidate
    if command.kind in {"read_evidence", "read_batch"}:
        outcome, handles = (batch_read_results(case, command) if command.kind == "read_batch"
                            else read_selection_results(case, command))
        case = put(case, read_memory=merge_handles(p.get("read_memory", []), handles),
                   latest_read_handles=handles)
        return _outcome(case, store, **outcome)
    if command.kind == "describe_tools":
        if p.get("description_batches", 0) >= 2:
            reject(["capability_ids"], "query_limit", "Two description batches are already used.",
                   "Use a loaded tool, revise judgment or finalize. Rejected commands do not reset the allowance.")
        allowed = {r["capability_id"] for r in _catalog(o, case)}
        unknown = set(command.capability_ids) - allowed
        if unknown:
            reject(["capability_ids"], "unknown_tool", "Unknown or disabled-owner tool IDs: " + ", ".join(sorted(unknown)),
                   "Choose an ID in the tool catalog.")
        details = o.deps.capability_registry.describe_tools(
            command.capability_ids, case.artifact_manifest, amesp_enabled=o.config.enable_amesp)
        details = [{
            **{k: v for k, v in detail.items() if k not in {"tool_arg_schema", "question_loop_arg_schema"}},
            "arguments_schema": tool_argument_model(
                o.deps.capability_registry.require(detail["capability_id"])).model_json_schema(),
        } for detail in details]
        loaded = {**p.get("loaded_tools", {}),
                  **{d["capability_id"]: d["definition_version"] for d in details}}
        # Loaded definitions remain case-local; only a bounded active set is
        # advertised per request. Any inactive capability can be loaded again.
        active = list(dict.fromkeys([*p.get("visible_tools", []), *command.capability_ids]))
        active = [cid for cid in active if cid not in command.capability_ids] + list(command.capability_ids)
        case = put(case, loaded_tools=loaded, visible_tools=active[-12:],
                   description_batches=p.get("description_batches", 0) + 1)
        return _outcome(case, store, "described", tools=details)
    if command.kind in {"revise_judgment", "repair_judgment"}:
        proposed = repair(case, command) if command.kind == "repair_judgment" else command
        try:
            candidate = revise(case, proposed)
        except OperationRejected as exc:
            draft = make_draft(case, proposed, exc.issues)
            raise OperationRejected([{**issue, "scope": "judgment_draft", "draft_id": draft["draft_id"]}
                                     for issue in exc.issues], judgment_draft=draft) from exc
        previous_draft = (p.get("judgment_draft") or {}).get("draft_id")
        candidate = put(candidate, judgment_draft=None)
        if protocol(candidate)["revision"] == p["revision"]:
            return _outcome(candidate, store, "unchanged", committed=False,
                            closed_draft_id=previous_draft,
                            reason="Identical records already accepted; no revision or review repeated.")
        # No mutation has reached disk until every linked argument/rank passes.
        candidate = _outcome(candidate, store, "committed", closed_draft_id=previous_draft,
                             changed=[c.model_dump(mode="json") for c in proposed.changes])
        affected_ids = {c.id for c in proposed.changes if c.kind in {"argument", "withdraw_argument"}}
        affected_ids.update(a for c in proposed.changes if c.kind == "ranking" for a in c.argument_ids)
        needs_review = proposed.request_review or review_basis(candidate) != review_basis(case)
        return put(candidate, phase="review" if o.config.effective_support_auditor and needs_review else "choose",
                   review_role="reviewer", review_changes=[c.model_dump(mode="json") for c in proposed.changes],
                   review_trigger="explicit" if proposed.request_review else "decision_or_sources" if needs_review else "record_fields_unchanged",
                   review_previous_arguments=[p["arguments"][a] for a in sorted(affected_ids) if a in p["arguments"]])
    if command.kind == "discard_judgment":
        require_draft(case, command.draft_id, current_revision=False)
        return _outcome(put(case, judgment_draft=None), store, "draft_discarded", committed=False,
                        closed_draft_id=command.draft_id)
    if command.kind == "compute":
        request, digest = prepare_computation(case, command, o.deps.capability_registry,
                                              amesp_enabled=o.config.enable_amesp)
        if request is None:
            return _outcome(case, store, "reused", result_id=p["computations"][digest],
                            executed=False, reason="Identical computation already has a saved result.")
        case = _outcome(case, store, "accepted", executed=False, dispatch_id=request.dispatch_id)
        return put(case, phase="execute", ticket={"request": request.model_dump(mode="json"), "digest": digest})
    if command.kind in {"finalize", "submit_final"}:
        candidate, draft = (saved_final(case, command.draft_id) if command.kind == "submit_final"
                            else final_draft(case, command))
        if o.config.effective_agenda_coverage_reviewer and not reviewed(case, draft):
            if command.kind == "submit_final":
                reject(["draft_id"], "review_unavailable", "This exact draft has no completed final review.",
                       "Use finalize to request review of a current draft; unavailable review is not endorsement.")
            return put(_outcome(case, store, "review_requested", committed=False),
                       phase="review", review_role="final_reviewer", review_changes=[],
                       final_draft=draft, final_proposal=command.model_dump(mode="json"))
        return _outcome(put(candidate, final_draft=draft), store, "finalized", committed=True,
                        draft_id=draft["draft_id"])
    raise RuntimeError("Unregistered command reached execution.")


def _execute(o, case, store):
    p = protocol(case)
    request = DispatchRequest.model_validate(p["ticket"]["request"])
    worker = o.deps.workers[request.agent_name]
    saved = store.read_execution_checkpoint(request)
    if saved and saved["stage"] == "executing":
        raise RuntimeError("Calculation completion is unknown. Reconcile its saved artifacts before resume; no automatic replay.")
    if saved and saved["stage"] in {"tool_returned", "reported"}:
        result = ToolExecutionResult.model_validate(saved["tool_result"])
        plan = AgentExecutionPlan.model_validate(saved["plan"])
        worker._validate_tool_result(result, request)
    else:
        # Planner already selected the full executable request. This binds it to
        # the existing calculator; no second agent changes its parameters.
        plan = worker.execution_planner.deterministic_plan(
            agent_name=request.agent_name, request=request, case_run=case,
            notes=["Exact Planner command; no scientific or parameter substitution."])
        plan = plan.model_copy(update={"planning_mode": "explicit_command"})
        worker._validate_execution_plan(plan, request)
        store.write_execution_checkpoint(request, stage="executing", plan=plan)
        store.activity("Compute " + request.capability_id, request.round_id)
        result = worker.execute_tool(request, case, execution_plan=plan)
        worker._validate_tool_result(result, request)
        store.write_execution_checkpoint(request, stage="tool_returned", plan=plan, result=result)
    store.write_tool_result(result)
    # Save the raw computation before report conversion or evidence indexing.
    # Those are projections and may fail independently of a completed tool.
    case = put(case, results={**p.get("results", {}), result.tool_result_id: result.model_dump(mode="json")})
    save(case, store)
    report = worker.report_writer.deterministic_report(request=request, execution_plan=plan, tool_result=result)
    if result.status not in {"success", "partial"}:
        report = report.model_copy(update={"evidence_units": [], "artifact_updates": []})
    else:
        report = report.model_copy(update={
            "evidence_units": [e for e in report.evidence_units if e.status in {"present", "partial"}],
        })
    case = o._apply_reports(case, [report])
    store.write_agent_report(report)
    store.append_jsonl("history/tool_events.jsonl", {
        "event": "agent_report", "dispatch": request.model_dump(mode="json"),
        "tool_result": result.model_dump(mode="json"), "report": report.model_dump(mode="json"),
    })
    result_id = result.tool_result_id
    brief = result_report(result_id, result.model_dump(mode="json"))
    round_ids = list(dict.fromkeys([*case.round_ids, request.round_id]))
    case = put(case.touch(round_ids=round_ids, current_round_id=request.round_id),
        results={**p.get("results", {}), result_id: result.model_dump(mode="json")},
        result_reports={**p.get("result_reports", {}), result_id: brief},
        read_memory=remember_report(p.get("read_memory", []), brief),
        latest_read_handles=remember_report([], brief),
        actions={**p.get("actions", {}), request.dispatch_id: request.model_dump(mode="json")},
        computations={**p.get("computations", {}), p["ticket"]["digest"]: result_id},
        rounds=p["rounds"] + 1, description_batches=0, pending_result=result_id, phase="explain", worker_session=None,
        outcome={"status": result.status, "result_id": result_id, "executed": True,
                 "result_report": brief,
                 "evidence_refs": [e.evidence_id for e in report.evidence_units],
                 "failure": result.failure.model_dump(mode="json")},
    )
    # Save raw facts and artifact updates before any explanatory model request.
    return save(case, store)


def _review_reference_ids(payload, available):
    """Use exact registered IDs exposed in the review, including nested raw units."""
    found = set()

    def visit(value):
        if isinstance(value, str):
            if value in available:
                found.add(value)
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(key)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for key in ("observations", "execution_context", "arguments", "ranking", "final_proposal", "citations", "candidate_overview", "pending_experiments"):
        visit(payload.get(key))
    return sorted(found)


def _comment(o, case, store):
    p = protocol(case)
    explaining = p["phase"] == "explain"
    role = "worker" if explaining else p.get("review_role", "reviewer")
    payload = {"stage": "command_" + role, "role": role,
               "smiles": case.input.smiles, "mechanism_pool": list(MECHANISM_POOL)}
    if explaining:
        result = p["results"][p["pending_result"]]
        worker = o.deps.workers[result["agent_name"]]
        if not worker.report_writer.enable_llm:
            return put(case, phase="choose", ticket=None)
        client = worker.report_writer.llm_client
        payload.update(result=p.get("result_reports", {}).get(p["pending_result"])
                              or result_report(p["pending_result"], result),
                       request=p["ticket"]["request"])
    else:
        client = (o.deps.mechanism_agenda if role in {"final_reviewer", "coverage_reviewer"}
                  else o.deps.photophysics_arbiter).llm_client
        changes = p.get("review_changes", [])
        affected = {c["id"] for c in changes if c["kind"] == "argument"}
        affected.update(aid for c in changes if c["kind"] == "ranking" for aid in c["argument_ids"])
        args = [p["arguments"][aid] for aid in sorted(affected) if aid in p.get("arguments", {})]
        if role == "final_reviewer":
            ids = ({aid for h in case.portfolio.sorted_hypotheses()[:3]
                    for aid in p["ranking"][h.name]["argument_ids"]} if p.get("ranking") else set())
            args = [p["arguments"][aid] for aid in sorted(ids)]
            payload["assembled_answer"] = (p.get("final_draft") or {}).get("content")
            if not p.get("final_draft"):
                payload["final_proposal"] = p.get("final_proposal")
            payload["draft_id"] = (p.get("final_draft") or {}).get("draft_id")
        previous = p.get("review_previous_arguments", []) if role == "reviewer" else []
        refs = {"input:smiles"} | {r for a in args for r in [*a["refs"], *a.get("limitation_refs", [])]}
        refs.update(r for a in previous for r in [*a["refs"], *a.get("limitation_refs", [])])
        candidate_changes = [c for c in changes if c["kind"] == "candidate"] if role == "reviewer" else []
        refs.update(r for c in candidate_changes for r in [*c["refs"], *c.get("limitation_refs", [])])
        available = records(case)
        payload.update(ranking=list(p.get("ranking", {}).values()), arguments=args,
                       observations=(review_observations(case, refs, available) if o.config.command_native_tools else
                                     {r: observation_view(available[r], preserve_paths=True) for r in sorted(refs)}),
                       citations=review_citations(case, args),
                       candidate_overview=candidate_overview(case), pending_experiments=pending_experiments(case),
                       review_trigger=p.get("review_trigger"),
                       execution_context=review_execution_context(case, refs, available),
                       run_limits=case.runtime.get("run_limits", {}))
        if role == "reviewer":
            payload.update(changes=changes, previous_arguments=previous)
        if role != "final_reviewer":
            payload["tool_catalog"] = [{k: row[k] for k in
                ("capability_id", "description", "status", "missing_inputs", "reason") if k in row}
                for row in _catalog(o, case)]
        payload["reference_ids"] = _review_reference_ids(payload, available)
    store.activity("Worker explanation" if explaining else "Scientific review", case.current_round_id)
    scope = {"refs": [p["pending_result"]]} if explaining else {"reviewed_record_ids": payload["reference_ids"]}
    for field, ids in scope.items():
        check_refs(case, ids, [field])
    try:
        if explaining and o.config.command_native_tools:
            feedback, session = worker_turn(client, case, payload, store)
            if session is not None:
                return put(case, worker_session=session)
        elif not explaining and o.config.command_native_tools:
            context_key = digest(payload)
            state = p.get("reviewer_session") or {"context_key": context_key, "queries": 0}
            if state["context_key"] != context_key:
                raise RuntimeError("Reviewer read session belongs to a different judgment")
            read_scope = reviewer_scope(case, set(payload["reference_ids"]), available)
            feedback, session = scoped_commentary_turn(client, case, payload, store,
                scope=read_scope, state=state, role=role, prompt_name="command_reviewer_text.md")
            if session is not None:
                return put(case, reviewer_session=session)
            feedback.update(scope)
        else:
            comment = ask_worker_text(client, payload, store) if explaining else ask_reviewer_text(
                client, payload, store, role=role)
            feedback = {"status": "available", **comment.model_dump(mode="json"), **scope}
    except OperationRejected as exc:
        feedback = {"status": "unavailable", "issues": exc.issues, **({} if explaining else scope),
                    "reason": "Commentary invalid; saved computations and judgment remain intact."}
    store.append_jsonl("history/command_commentary.jsonl", {
        "role": role, "revision": p["revision"], "result_id": p.get("pending_result") if explaining else None,
        **feedback,
    })
    if explaining:
        remembered = remember_report(p.get("read_memory", []), payload["result"])
        delivered = remember_report([], payload["result"])
        if feedback["status"] == "available":
            text = [("explanation:" + p["pending_result"], "/text", feedback["text"], None)]
            remembered, delivered = remember_values(remembered, text), remember_values(delivered, text)
        return put(case, phase="choose", ticket=None, worker_session=None,
                   read_memory=remembered, latest_read_handles=delivered,
                   explanations={**p.get("explanations", {}), p["pending_result"]: feedback},
                   outcome={**p["outcome"], "result_report": payload["result"], "worker_explanation": feedback})
    return put(case, phase="choose", review=feedback, reviewer_session=None,
               final_review=({"draft_id": (p.get("final_draft") or {}).get("draft_id"),
                              "status": feedback["status"]} if role == "final_reviewer" else p.get("final_review")),
               final_review_revision=p["revision"] if role == "final_reviewer" else p.get("final_review_revision"))


def run_command_loop(o, case, store, *, resume=False):
    """A stop request drains the current calculation and prevents new dispatch."""
    with (store.root / ".command_loop.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = store.root / "command_checkpoint.json"
        fingerprint = configuration_fingerprint(o, case)
        if resume:
            saved = CaseRun.model_validate_json(path.read_text())
            check_owner(saved)
            if saved.case_id != case.case_id or saved.input != case.input or protocol(saved)["fingerprint"] != fingerprint:
                raise ValueError("Resume case/configuration/code does not match command checkpoint.")
            case = put(saved.touch(status="running"), interruption=None, local_errors=0)
            store.append_jsonl("history/command_resume.jsonl", {"timestamp": time.time(), "revision": protocol(case)["revision"]})
        elif path.exists() or (store.root / "question_checkpoint.json").exists():
            raise FileExistsError("Use a new run directory; legacy failure scenes must not be overwritten.")
        else:
            store.write_case_input(case.input)
            store.write_capability_catalog(o.deps.capability_registry.snapshot())
            case = put(case, version=2, case_id=case.case_id, fingerprint=fingerprint,
                       phase="prepare", turn=1, revision=0, rounds=0, local_errors=0,
                       arguments={}, ranking={}, results={}, computations={})
        case = save(case, store)
        stopping = False

        def stop(signum, frame):
            nonlocal stopping
            stopping = True

        old_handlers = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                old_handlers[sig] = signal.signal(sig, stop)
        try:
            while protocol(case)["phase"] != "done" and case.status == "running":
                if stopping or (store.root / "STOP_REQUESTED").exists():
                    case = put(case.touch(status="paused"), interruption=stop_reason(store.root))
                    return save(case, store)
                p = protocol(case)
                allowed = o.config.unlimited_exploration or p["rounds"] < (
                    o.config.max_exploration_rounds if o.config.max_exploration_rounds is not None else o.config.max_rounds)
                case = case.touch(runtime={**case.runtime, "run_limits": {
                    **case.runtime["run_limits"], "scientific_tools_allowed": allowed,
                    "exploration_rounds_used": p["rounds"],
                }})
                if p["phase"] == "prepare":
                    case = o._prepare_structure(case, store)
                    case = put(case, phase="review" if o.config.effective_agenda_coverage_reviewer else "choose",
                               review_role="coverage_reviewer", review_changes=[])
                elif p["phase"] == "choose":
                    if o.config.command_hypothesis_tracking:
                        case = save(hypotheses.sync(case), store)
                        p = protocol(case)
                    model = envelope_model(o.deps.capability_registry, p.get("visible_tools", []),
                                           hypothesis_tracking=o.config.command_hypothesis_tracking)
                    if p.get("pending_command") is None:
                        store.activity("Planner command", case.current_round_id)
                        try:
                            if o.config.command_native_tools:
                                payload = planner_payload(o, case, native=True)
                                command, exchange = ask_tools(o.deps.planner.llm_client,
                                    o.deps.capability_registry, p.get("visible_tools", []), payload, store,
                                    exchange=p.get("latest_exchange"), outcome=p.get("outcome"))
                                case = put(case, latest_exchange=exchange)
                                key = ("hypothesis_step" if isinstance(command, HypothesisStep) else
                                       "read_batch" if isinstance(command, ReadBatch) else
                                       "compute_batch" if isinstance(command, ComputeBatch) else "command")
                                pending = {key:
                                           command.model_dump(mode="json")}
                            else:
                                response = ask(o.deps.planner.llm_client, model, planner_payload(o, case), store)
                                pending = response.model_dump(mode="json")
                        except OperationRejected as exc:
                            case = _local_error(case, exc.issues, store, exchange=exc.exchange,
                                                judgment_draft=exc.judgment_draft)
                            continue
                        case = save(put(case, pending_command=pending), store)
                    pending = protocol(case)["pending_command"]
                    command = (HypothesisStep.model_validate(pending["hypothesis_step"]) if "hypothesis_step" in pending else
                               ComputeBatch.model_validate(pending["compute_batch"]) if "compute_batch" in pending else
                               ReadBatch.model_validate(pending["read_batch"]) if "read_batch" in pending
                               else model.model_validate(pending).command)
                    try:
                        case = _apply(o, case, command, store)
                    except OperationRejected as exc:
                        issues = _model_issue_paths(exc.issues, native=o.config.command_native_tools)
                        case = _local_error(case, issues, store, judgment_draft=exc.judgment_draft)
                        continue
                elif p["phase"] == "hypothesis_next":
                    case = _hypothesis_followup(o, case, store)
                elif p["phase"] == "execute":
                    case = _execute(o, case, store)
                elif p["phase"] == "batch_next":
                    case = next_member(o, case)
                elif p["phase"] in {"explain", "review"}:
                    case = _comment(o, case, store)
                    if p["phase"] == "explain" and protocol(case)["phase"] == "choose" and p.get("compute_batch"):
                        case = finish_member(case)
                else:
                    raise RuntimeError("Unknown command-loop stage.")
                case = save(case, store)
            return case
        except Exception as exc:
            # Reload the authoritative checkpoint: an interrupted optional
            # explanation cannot erase a completed calculation.
            latest = CaseRun.model_validate_json(path.read_text())
            failure = {"kind": "system_interruption", "type": type(exc).__name__, "message": str(exc),
                       "phase": protocol(latest)["phase"], "requires_explicit_resume": True}
            store.append_jsonl("history/command_interruptions.jsonl", failure)
            save(put(latest.touch(status="paused"), interruption=failure), store)
            raise
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


def _hypothesis_followup(o, case, store):
    """Resume the saved follow-up without asking the model or repeating its update."""
    p = protocol(case)
    step = p["hypothesis_step"]
    exchange = {**p["latest_exchange"], "calls": step["followup_calls"]}
    tools, models = tool_contracts(o.deps.capability_registry, p.get("visible_tools", []),
                                  with_draft=bool(p.get("judgment_draft")))
    try:
        command = parse_tool_turn(exchange, tools, models)
    except OperationRejected as exc:
        return _local_error(case, exc.issues, store, exchange=p["latest_exchange"],
                            judgment_draft=exc.judgment_draft)
    try:
        return _apply(o, case, command, store)
    except OperationRejected as exc:
        return _local_error(case, _model_issue_paths(exc.issues, native=True), store,
                            exchange=p["latest_exchange"], judgment_draft=exc.judgment_draft)


def _model_issue_paths(issues, *, native):
    """Standalone and combined operations identify the same model-editable fields."""
    mapped = []
    for issue in issues:
        path = issue["path"]
        if issue.get("scope") == "judgment_draft":
            path = ["draft", *path]
        elif native and issue.get("code") in {"stale_draft", "stale_revision"}:
            path = ["judgment_draft", *path]
            issue = {**issue, "remedy": "The request-bound draft is no longer current. Inspect judgment_draft "
                     "and issue a new operation for that version; do not generate a draft_id."}
        elif native:
            path = ["name"] if path and path[0] in {"kind", "capability_id"} else ["arguments", *path]
        else:
            path = ["command", *path]
        mapped.append({**issue, "path": path})
    return mapped
