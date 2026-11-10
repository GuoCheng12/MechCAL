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
from mechcal.mechanism_pool import MECHANISM_POOL, mechanism_definitions
from mechcal.runtime.command_operations import (
    OperationRejected, cached_computation, check_owner, check_refs, prepare_computation,
    protocol, put, record_identity, record_index, records, reject, revise,
)
from mechcal.runtime.command_context import catalog_table, record_table, result_card, short_identity, share_text
from mechcal.runtime.command_compact import planner_view
from mechcal.runtime.command_memory import (
    READ_BYTES, digest, json_bytes, record_layout, remember_report, remember_values,
    result_index, result_report, working_reads,
)
from mechcal.runtime.command_draft import draft_view, make_draft, repair, require_draft
from mechcal.runtime.command_events import CommandEvent, stop_reason, track_state, write_event
from mechcal.runtime import command_hypotheses as hypotheses
from mechcal.runtime import command_followup as followup
from mechcal.runtime import command_review_workflow as workflow
from mechcal.runtime import computational_review
from mechcal.runtime import command_read_handoff as handoff
from mechcal.runtime import command_query_state as query_state
from mechcal.runtime import command_numeric as numeric
from mechcal.runtime import command_recovery as recovery
from mechcal.runtime import command_delivery_review as delivery_review
from mechcal.runtime import worker_tasks
from mechcal.runtime.command_comparison import comparison_view, comparison_refs
from mechcal.runtime.command_action_policy import action_policy
from mechcal.runtime.command_edit_context import edit_context
from mechcal.runtime.command_views import ANSWER_REF
from mechcal.runtime.command_views import enabled as workflow_enabled, ordered_ranking, computation_allowance
from mechcal.runtime.command_impact_review import run_turn as impact_review_turn
from mechcal.runtime.command_delivery import amended_final, final_draft, reviewed, saved_final
from mechcal.runtime.command_reads import batch_read_results, merge_handles, read_selection_results, query_results, query_refs, READ_KINDS
from mechcal.runtime.command_review import (
    candidate_overview, completed_interim_review, interim_review_trigger, reference_contract,
    review_citations, review_execution_context, review_worker_interpretations,
    review_observations, reviewer_scope, review_projection,
)
from mechcal.runtime.command_experiments import (
    finish_member, manage_experiments, next_member, pending_experiments,
    retain_rejected_computations, stage_batch,
)
from mechcal.runtime.question_loop import configuration_fingerprint
from mechcal.schemas import AgentExecutionPlan, CaseRun, DispatchRequest, ToolExecutionResult
from mechcal.schemas.commands import ComputeBatch, HypothesisStep, ReadBatch, RecordReview, RecordComputationalReview, RecordFreshComputationalReview, UpdateHypotheses, envelope_model, tool_argument_model


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


def _registry(o, case):
    allowed = case.runtime.get("ablation", {}).get("enabled_capability_owners", ["macro", "microscopic"])
    return o.deps.capability_registry.for_owner_agents(tuple(allowed))


def _catalog(o, case):
    return _registry(o, case).compact_catalog(case.artifact_manifest, amesp_enabled=o.config.enable_amesp)


def _route_context(o, case):
    return {"registry": _registry(o, case), "amesp_enabled": o.config.enable_amesp}


def _describe_tools(o, case, command):
    allowed = {r["capability_id"] for r in _catalog(o, case)}
    unknown = set(command.capability_ids) - allowed
    if unknown:
        reject(["capability_ids"], "unknown_tool", "Unknown or disabled-owner tool IDs: " + ", ".join(sorted(unknown)),
               "Choose an ID in the tool catalog.")
    details = o.deps.capability_registry.describe_tools(
        command.capability_ids, case.artifact_manifest, amesp_enabled=o.config.enable_amesp)
    return [{**{k: v for k, v in detail.items() if k not in {"tool_arg_schema", "question_loop_arg_schema"}},
             "arguments_schema": tool_argument_model(
                 o.deps.capability_registry.require(detail["capability_id"])).model_json_schema()}
            for detail in details]


def planner_payload(o, case, *, native=False):
    p = protocol(case)
    index = record_index(case)
    editing = edit_context(case, index=index, native_replay=native) if native else None
    edit_refs = {ANSWER_REF, editing["draft_id"]} if editing else set()
    active_refs = {ref for row in [*p.get("arguments", {}).values(), *p.get("candidates", {}).values()]
                   for field in ("refs", "limitation_refs") for ref in row.get(field, [])}
    if o.config.command_hypothesis_tracking:
        active_refs.update(ref for row in hypotheses.state(case).entries.values()
                           for ref in [*row.refs, *row.limitation_refs])
    outcome = p.get("outcome") or {}
    inherited, handoff_issues = handoff.retained(case, index=index) if o.config.command_compact_context else ([], [])
    focus_refs = active_refs | {p.get("pending_result")}
    if o.config.command_hypothesis_tracking:
        focus_refs.update(hypotheses.view(case)["new_result_ids"])
    payload = {
        "stage": "planner_command", "smiles": case.input.smiles,
        "operation_policy": action_policy(case, index=index,
            scientific_read_limit_enabled=o.config.command_hypothesis_tracking).model_dump(mode="json"),
        "delivery_recovery": recovery.view(case),
        "mechanism_pool": list(MECHANISM_POOL), "task": case.input.user_query,
        "mechanism_definitions": mechanism_definitions(),
        "ranking": ordered_ranking(case),
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
        "result_index": [(result_card if native else result_index)(k, v) for k, v in p.get("results", {}).items()],
        "artifact_index": [{"ref": a.artifact_id, "kind": a.kind, "status": a.status}
                           for a in case.artifact_manifest.artifacts],
        "action_index": [{"ref": k, "tool": a["capability_id"], "purpose_preview": a["task"][:160]}
                         for k, a in p.get("actions", {}).items()],
        "explanation_index": [{"ref": "explanation:" + rid, "source_result_id": rid,
                               "status": e["status"], "bytes": json_bytes({"source_result_id": rid, **e}),
                               **({"paths": ["/text", "/status"]} if native else
                                  {"directory": record_layout({"source_result_id": rid, **e})["children"]})}
                              for rid, e in p.get("explanations", {}).items()],
        "read_limit_bytes": READ_BYTES,
        "working_memory": working_reads([h for h in merge_handles(p.get("read_memory", []), inherited)
                                         if h["ref"] not in edit_refs], index.available, active_refs=active_refs,
                                        latest_rows=p.get("latest_read_handles", [])
                                        if outcome.get("status") in {"read", "partial_read", "read_batch"}
                                        or "result_report" in outcome else [],
                                        focus_refs=focus_refs if o.config.command_compact_context else None,
                                        explicit_handles=p.get("planner_read_handles", [])),
        "tool_catalog": catalog_table(_catalog(o, case)) if native else _catalog(o, case),
        "last_operation": p.get("outcome"),
        "review_feedback": p.get("review"),
        "interim_review_status": {
            "last_reviewed_revision": (p.get("interim_review") or {}).get("revision"),
            "automatic_review_reason": interim_review_trigger(case, index=index),
            "meaning": "Scheduling only, not approval. Same-source corrections return to action selection. "
                       "Prior objections remain advice; exact final content still needs final review.",
        },
        "tools_allowed": case.runtime["run_limits"].get("scientific_tools_allowed", True),
        "loaded_capabilities": p.get("visible_tools", []),
        "description_batches_remaining": max(0, 2 - p.get("description_batches", 0)),
        "final_draft": ({"draft_id": p["final_draft"]["draft_id"],
                         "revision": p["final_draft"]["revision"],
                         "review_status": (p.get("final_review") or {}).get("status"),
                         "read_ref": ANSWER_REF if editing else p["final_draft"]["draft_id"]}
                        if p.get("final_draft") else None),
    }
    if editing:
        payload["edit_context"] = editing
        issue_refs = {r for issue in workflow.issue_view(case, index=index) for r in issue["source_refs"]}
        if issue_refs:
            payload["review_facts"] = delivery_review.fact_packets(case, issue_refs, index=index)
    if o.config.command_worker_tasks:
        payload["worker_delegation"] = {"enabled": True, "maximum_model_calls": o.config.worker_task_max_calls,
            "maximum_computations": o.config.worker_task_max_computations,
            "seconds": o.config.worker_task_seconds}
    if p.get("numeric_checks"):
        payload["numeric_checks"] = numeric.view(case, index=index,
            exclude_ids=numeric.outcome_ids(outcome) if native and p.get("latest_exchange") else ())
    if handoff_issues:
        payload["source_read_issues"] = handoff_issues
    if o.config.command_compact_context:
        payload["query_state"] = query_state.view(case, index=index,
            latest_outcome=outcome if native and p.get("latest_exchange") else None)
        payload["shared_tool_guidance"] = query_state.tool_guidance(case, o.deps.capability_registry)
    if o.config.command_hypothesis_tracking:
        payload["exploration_state"] = hypotheses.view(case)
        payload.pop("candidate_overview", None)
        payload["evidence_associations"] = comparison_view(case, index=index)["source_directory"]
    if o.config.command_computational_review:
        stop_context = computational_review.context(case, _catalog(o, case), **_route_context(o, case))
        payload["computational_review"] = {k: stop_context[k] for k in ("basis", "previous", "policy")}
    if workflow_enabled(case):
        payload["computation_allowance"] = computation_allowance(case)
        payload["review_workflow"] = {"issues": workflow.issue_view(case, index=index), "state_ref": "review:issues",
            "stage": "delivery" if p.get("delivery_rounds") == p.get("rounds") else "exploration",
            "policy": "Resolve factual issues in their target records. Science disagreements are advice, not gates. "
                      "Do not rewrite settled claims or copy them into candidate notes and ranking rationale."}
        if editing:
            editing["issue_ids"] = [row["id"] for row in payload["review_workflow"]["issues"]
                                    if row["target"] in edit_refs]
        if payload["review_workflow"]["issues"]:
            from mechcal.runtime.command_edit_context import correction_context
            payload["correction_context"] = correction_context(case, index=index)
        payload["comparison_options"] = workflow.comparison_options(case, _catalog(o, case), **_route_context(o, case))
        feedback = payload.get("review_feedback")
        if feedback and feedback.get("findings") is not None:
            payload["review_feedback"] = {"status": feedback["status"], "text": feedback["text"]}
    # A fixture/resume may have a delivered result but no native call to replay.
    # Deduplication is valid only when that result actually has a message route.
    if native and p.get("latest_exchange"):
        payload.pop("last_operation", None)
    known = index.available
    for name in ("argument_index", "evidence_index", "result_index", "artifact_index", "action_index", "explanation_index"):
        if any(item["ref"] not in known for item in payload[name]):
            raise RuntimeError("Advertised record is not queryable: " + name)
        for item in payload[name]:
            identity = record_identity(case, item["ref"], index=index)
            item.update(short_identity(identity) if native else identity)
    for item in payload["pending_experiments"]:
        identity = record_identity(case, item["experiment_id"], index=index)
        item.update(short_identity(identity) if native else identity)
    if native:
        for name in ("argument_index", "evidence_index", "result_index", "artifact_index", "action_index", "explanation_index"):
            payload[name] = record_table(payload[name])
        payload["record_navigation"] = ("Indexes are previews, not complete evidence. Read report_ref for a bounded "
            "result report; read_evidence with view=directory lists exact raw paths. All listed IDs remain readable. "
            "Unshown fields are not absent or negative evidence. Index and tool catalog rows use their declared columns. "
            "For a cited evidence unit, source_result_ref names its registered raw parent; otherwise use the exact ref.")
    if native and o.config.command_compact_context:
        return planner_view(payload)
    return share_text(payload) if native else payload


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
    # Synchronous completion yields to Planner; asynchronous callers explicitly
    # select execute/review/batch_next. Final delivery must remain terminal.
    return put(case, phase="done" if p["phase"] == "done" else "choose",
               outcome=result, local_errors=0, pending_command=None, turn=p["turn"] + 1,
               outcome_request_id=(p.get("latest_exchange") or {}).get("request_id"),
               receipt={"turn": p["turn"], "revision": p["revision"], **result})


def _apply(o, case, command, store):
    if command.kind == "read_batch":
        # Context and evidence reads share receipts/byte limits, not permissions.
        # A rejected member never cancels another read-only member.
        calls = []
        for call in command.calls:
            try:
                selected = call.command
                if selected is not None and selected.kind == "read_context":
                    if not o.config.command_task_context:
                        reject(["kind"], "operation_disabled", "Task context reads are disabled.", "Use advertised operations.")
                    from mechcal.runtime.command_task_context import context_selection
                    selected, receipt, unknown = context_selection(case, selected)
                    case = put(case, context_reads=receipt)
                    if unknown:
                        reject(["fields"], "context_field", "Fields are not deferred: " + ", ".join(unknown),
                               "Use fields from deferred_context.")
                if selected is not None and o.config.command_hypothesis_tracking:
                    action_policy(case).guard_read(query_refs(selected))
                calls.append(call.model_copy(update={"command": selected}))
            except OperationRejected as exc:
                calls.append(call.model_copy(update={"command": None, "issues": exc.issues}))
        command = command.model_copy(update={"calls": calls})
    if command.kind == "read_context":
        if not o.config.command_task_context:
            reject(["kind"], "operation_disabled", "Task context reads are disabled.", "Use the advertised operations.")
        from mechcal.runtime.command_task_context import context_selection
        selection, receipt, unknown = context_selection(case, command)
        case = put(case, context_reads=receipt)
        if unknown:
            return _outcome(case, store, "rejected", executed=False, committed=False, issues=[{
                "path": ["fields"], "code": "context_field", "message": "Fields are not deferred: " + ", ".join(unknown),
                "remedy": "Choose field names from deferred_context; undeferred fields are already visible."}])
        outcome, handles = read_selection_results(case, selection)
        p = protocol(case)
        case = put(case, read_memory=merge_handles(p.get("read_memory", []), handles),
                   latest_read_handles=handles,
                   planner_read_handles=merge_handles(p.get("planner_read_handles", []), handles))
        case = query_state.remember(case, outcome, role="planner")
        return _outcome(case, store, **outcome)
    if o.config.command_hypothesis_tracking:
        case = hypotheses.sync(case)
        hypotheses.guard(case, command)
        if command.kind in READ_KINDS | {"read_batch"}:
            case = hypotheses.record_read_attempt(case, protocol(case).get("latest_exchange"))
    p = protocol(case)
    if command.kind in {"update_hypotheses", "hypothesis_step"}:
        if not o.config.command_hypothesis_tracking:
            reject([], "operation_disabled", "Exploration-state tracking is disabled.", "Use the enabled operations.")
        update = command.update if command.kind == "hypothesis_step" else command
        workflow.validate_routes(case, update, _catalog(o, case))
        candidate, result = hypotheses.update(case, update)
        if command.kind == "update_hypotheses":
            return _outcome(candidate, store, **result)
        return put(candidate, phase="hypothesis_next", pending_command=None,
                   hypothesis_step={"update_call_id": command.update_call_id, "update_result": result,
                                    "followup_calls": command.followup_calls})
    if command.kind == "compute_batch":
        candidate = stage_batch(o, case, command)
        return put(_outcome(candidate, store, "batch_accepted", executed=False), phase="batch_next", outcome_request_id=None)
    if command.kind == "manage_experiments":
        candidate = manage_experiments(o, case, command)
        candidate = _outcome(candidate, store, "experiments_" + command.action, experiment_ids=command.experiment_ids)
        return put(candidate, phase="batch_next", outcome_request_id=None) if command.action == "retry" else candidate
    if command.kind in READ_KINDS | {"read_batch"}:
        outcome, handles = (batch_read_results(case, command) if command.kind == "read_batch"
                            else query_results(case, command))
        case = put(case, read_memory=merge_handles(p.get("read_memory", []), handles),
                   latest_read_handles=handles,
                   planner_read_handles=merge_handles(p.get("planner_read_handles", []), handles))
        if o.config.command_compact_context or workflow_enabled(case) or numeric.outcome_ids(outcome):
            case = query_state.remember(case, outcome, role="planner")
        return _outcome(case, store, **outcome)
    if command.kind == "describe_tools":
        if p.get("description_batches", 0) >= 2:
            reject(["capability_ids"], "query_limit", "Two description batches are already used.",
                   "Use a loaded tool, revise judgment or finalize. Rejected commands do not reset the allowance.")
        details = _describe_tools(o, case, command)
        loaded = {**p.get("loaded_tools", {}),
                  **{d["capability_id"]: d["definition_version"] for d in details}}
        # Loaded definitions remain case-local; only a bounded active set is
        # advertised per request. Any inactive capability can be loaded again.
        active = list(dict.fromkeys([*p.get("visible_tools", []), *command.capability_ids]))
        active = [cid for cid in active if cid not in command.capability_ids] + list(command.capability_ids)
        case = put(case, loaded_tools=loaded, visible_tools=active[-12:],
                   description_batches=p.get("description_batches", 0) + 1)
        return _outcome(case, store, "described", tools=details)
    if command.kind in {"revise_judgment", "repair_judgment", "amend_judgment"}:
        if command.kind == "amend_judgment" and not workflow_enabled(case):
            reject(["kind"], "operation_disabled", "Local accepted-argument edits are disabled.", "Use the advertised operations.")
        proposed = (repair(case, workflow.internal_judgment(case, command)) if command.kind == "repair_judgment" else
                    workflow.amended_judgment(case, command) if command.kind == "amend_judgment" else
                    workflow.internal_judgment(case, command))
        try:
            candidate = revise(case, proposed)
        except OperationRejected as exc:
            draft = make_draft(case, proposed, exc.issues)
            raise OperationRejected([{**issue, "scope": "judgment_draft", "draft_id": draft["draft_id"]}
                                     for issue in exc.issues], judgment_draft=draft) from exc
        previous_draft = (p.get("judgment_draft") or {}).get("draft_id")
        if followup.state(candidate) is not None:
            # Older native candidate edits remain accepted, but write the same
            # current judgment displayed to both Planner and Reviewer.
            entries = hypotheses.state(candidate).entries
            notes = [{"label": c.label, "confidence": entries[c.label].confidence if c.label in entries else None,
                      "note": c.note, "refs": c.refs, "limitation_refs": c.limitation_refs}
                     for c in proposed.changes if c.kind == "candidate"]
            if notes:
                candidate, _ = hypotheses.update(candidate, UpdateHypotheses(
                    kind="update_hypotheses", changes=notes, note=proposed.note or "Candidate judgment revised."))
            candidate = followup.bind_formal(candidate, proposed.changes)
        candidate = put(candidate, judgment_draft=None)
        if protocol(candidate)["revision"] == p["revision"]:
            candidate = _outcome(candidate, store, "unchanged", committed=False,
                                 closed_draft_id=previous_draft,
                                 reason="Identical records already accepted; no revision repeated.")
            if proposed.request_review and o.config.effective_support_auditor:
                return put(candidate, phase="review", review_role="reviewer", review_trigger="explicit",
                           review_changes=[c.model_dump(mode="json") for c in proposed.changes],
                           review_previous_arguments=[])
            return candidate
        # No mutation has reached disk until every linked argument/rank passes.
        candidate = _outcome(candidate, store, "committed", closed_draft_id=previous_draft,
                             changed=[c.model_dump(mode="json") for c in proposed.changes])
        affected_ids = {c.id for c in proposed.changes if c.kind in {"argument", "withdraw_argument"}}
        affected_ids.update(a for c in proposed.changes if c.kind == "ranking" for a in c.argument_ids)
        trigger = "explicit" if proposed.request_review else interim_review_trigger(candidate)
        needs_review = trigger is not None
        return put(candidate, phase="review" if o.config.effective_support_auditor and needs_review else "choose",
                   review_role="reviewer", review_changes=[c.model_dump(mode="json") for c in proposed.changes],
                   review_trigger=trigger or "same_sources_return_to_planner",
                   review_previous_arguments=[p["arguments"][a] for a in sorted(affected_ids) if a in p["arguments"]])
    if command.kind == "discard_judgment":
        require_draft(case, command.draft_id, current_revision=False)
        return _outcome(put(case, judgment_draft=None), store, "draft_discarded", committed=False,
                        closed_draft_id=command.draft_id)
    if command.kind == "delegate_computation":
        candidate = worker_tasks.begin(o, case, command)
        candidate = _outcome(candidate, store, "worker_task_accepted", executed=False,
                             task_id=protocol(candidate)["worker_task"]["task_id"])
        return put(candidate, phase="worker_task")
    if command.kind == "compute":
        request, digest = prepare_computation(case, command, o.deps.capability_registry,
                                              amesp_enabled=o.config.enable_amesp)
        if request is None:
            case = followup.record_trial(case, followup.question_snapshot(case, getattr(command, "question_id", None)),
                capability_id=command.capability_id, purpose=command.purpose, result_id=p["computations"][digest])
            return _outcome(case, store, **cached_computation(case, p["computations"][digest]))
        case = _outcome(case, store, "accepted", executed=False, dispatch_id=request.dispatch_id)
        return put(case, phase="execute", ticket={"request": request.model_dump(mode="json"), "digest": digest,
                                                  "question": followup.question_snapshot(case, getattr(command, "question_id", None))})
    if command.kind == "amend_final":
        if not workflow_enabled(case):
            reject(["kind"], "operation_disabled", "Final text amendments are disabled.", "Use the advertised operations.")
        command = amended_final(case, command)
    if command.kind in {"finalize", "submit_final", "review_final"}:
        candidate, draft = (saved_final(case, command.draft_id) if command.kind in {"submit_final", "review_final"}
                            else final_draft(case, command))
        stop_context = (computational_review.context(case, _catalog(o, case), **_route_context(o, case))
                        if o.config.command_computational_review else None)
        needs_review = not reviewed(case, draft) or (stop_context is not None and stop_context["previous"] is None)
        if o.config.effective_agenda_coverage_reviewer and needs_review:
            if command.kind == "submit_final":
                reject(["draft_id"], "review_unavailable", "This exact draft has no completed final review.",
                       "Use review_final with this draft_id to retry its review without rewriting. "
                       "Unavailable review is not endorsement.", record_id=draft["draft_id"])
            failed = p.get("failed_review_session") or {}
            session = failed.get("session") if failed.get("draft_id") == draft["draft_id"] else None
            if session is not None:
                session = {**session, "corrections": 0}
            return recovery.begin(put(_outcome(case, store, "review_requested", committed=False,
                                               draft_id=draft["draft_id"]),
                       phase="review", review_role="final_reviewer", review_changes=[],
                       final_draft=draft, final_proposal=draft["command"], delivery_rounds=p["rounds"],
                       reviewer_session=session, failed_review_session=None),
                       max_model_turns=o.config.command_delivery_max_model_turns)
        if command.kind == "review_final":
            return _outcome(case, store, "review_already_completed", committed=False, draft_id=draft["draft_id"])
        responses = computational_review.require_response(case, command, stop_context) if stop_context is not None else None
        return _outcome(put(candidate, final_draft=draft), store, "finalized", committed=True,
                        draft_id=draft["draft_id"], **({"route_responses": responses} if responses is not None else {}))
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
        # Bind the accepted request exactly, whether from Planner or a delegated
        # Worker. No calculator-side retry changes these parameters.
        plan = worker.execution_planner.deterministic_plan(
            agent_name=request.agent_name, request=request, case_run=case,
            notes=["Exact accepted command; no scientific or parameter substitution."])
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
    computations = {**p.get("computations", {}), p["ticket"]["digest"]: result_id}
    if retry := request.constraints.get("command_retry"):
        computations[retry["base_digest"]] = result_id
    case = put(case.touch(round_ids=round_ids, current_round_id=request.round_id),
        results={**p.get("results", {}), result_id: result.model_dump(mode="json")},
        result_reports={**p.get("result_reports", {}), result_id: brief},
        read_memory=remember_report(p.get("read_memory", []), brief),
        latest_read_handles=remember_report([], brief),
        actions={**p.get("actions", {}), request.dispatch_id: request.model_dump(mode="json")},
        computations=computations,
        rounds=p["rounds"] + 1, description_batches=0, pending_result=result_id, phase="explain", worker_session=None,
        planner_read_handles=[],
        outcome={"status": result.status, "result_id": result_id, "executed": True,
                 "result_report": brief,
                 "evidence_refs": [e.evidence_id for e in report.evidence_units],
                 "failure": result.failure.model_dump(mode="json")},
    )
    case = followup.record_trial(case, p["ticket"].get("question"), capability_id=request.capability_id,
                                purpose=request.task, dispatch_id=request.dispatch_id, result_id=result_id)
    case = put(case, outcome={**protocol(case)["outcome"], "execution_context":
                             review_execution_context(case, {result_id}, record_index(case).available)})
    if p["ticket"].get("worker_task_id"):
        case = worker_tasks.accept_result(case, result_id)
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

    for key in ("observations", "execution_context", "arguments", "ranking", "final_proposal", "assembled_answer",
                "citations", "candidate_overview", "pending_experiments", "evidence_followup", "review_targets",
                "review_workflow", "computational_review", "worker_interpretations"):
        visit(payload.get(key))
    return sorted(found)


def _comment(o, case, store):
    p = protocol(case)
    index = record_index(case)
    review_receipt = None
    explaining = p["phase"] == "explain"
    role = "worker" if explaining else p.get("review_role", "reviewer")
    payload = {"stage": "command_" + role, "role": role,
               "smiles": case.input.smiles, "mechanism_pool": list(MECHANISM_POOL),
               "mechanism_definitions": mechanism_definitions()}
    if explaining:
        result = p["results"][p["pending_result"]]
        worker = o.deps.workers[result["agent_name"]]
        if not worker.report_writer.enable_llm:
            return put(case, phase="choose", ticket=None)
        client = worker.report_writer.llm_client
        payload.update(result=p.get("result_reports", {}).get(p["pending_result"])
                              or result_report(p["pending_result"], result),
                       request=p["ticket"]["request"],
                       execution_context=review_execution_context(case, {p["pending_result"]}, index.available))
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
        available = index.available
        payload.update(ranking=ordered_ranking(case), arguments=args,
                       observations=(review_observations(case, refs, available) if o.config.command_native_tools else
                                     {r: observation_view(available[r], preserve_paths=True) for r in sorted(refs)}),
                       citations=review_citations(case, args, index=index),
                       candidate_overview=candidate_overview(case), pending_experiments=pending_experiments(case),
                       review_trigger=p.get("review_trigger"),
                       execution_context=review_execution_context(case, refs, available),
                       run_limits=case.runtime.get("run_limits", {}))
        if followup.state(case) is not None:
            payload["evidence_followup"] = followup.view(case)
        if role == "reviewer":
            payload.update(changes=changes, previous_arguments=previous)
        catalog = _catalog(o, case)
        if role != "final_reviewer" or o.config.command_computational_review:
            payload["tool_catalog"] = [{k: row[k] for k in
                ("capability_id", "description", "status", "missing_inputs", "reason") if k in row}
                for row in catalog]
        if o.config.command_computational_review:
            payload["computational_review"] = computational_review.context(case, catalog, **_route_context(o, case))
            payload["tool_catalog"] = catalog_table(catalog)
            if role == "final_reviewer":
                comparison = payload["computational_review"]["candidate_comparison"]
                # Alternatives receive the same read permissions, not a dump of
                # all historical calculations into every review request.
                refs.update(comparison_refs(comparison))
                payload["execution_context"] = review_execution_context(case, refs, available)
        if workflow_enabled(case):
            payload["computation_allowance"] = computation_allowance(case)
            payload["comparison_options"] = workflow.comparison_options(case, _catalog(o, case), **_route_context(o, case))
            if o.config.command_computational_review:
                payload.pop("comparison_options")  # Already in the shared computational review context.
            if role != "coverage_reviewer":
                payload["review_targets"] = workflow.review_targets(case, index=index)
                payload["review_workflow"] = {"issues": workflow.issue_view(case, index=index),
                    "prior_resolutions": workflow.prior_resolutions(case, index=index),
                    "mode": "resolve_issues" if workflow.state(case).issues else "initial_review",
                    "policy": "Review changed targets and unresolved facts together. Read related unchanged claims "
                              "if affected. Prior resolutions are historical context, not a queue to close again. "
                              "Reopen a prior issue when the changed claim actually reintroduces it; "
                              "do not restate correct numbers or reopen settled issues without a reason."}
        payload["worker_interpretations"] = review_worker_interpretations(case, payload["execution_context"])
        payload["reference_ids"] = _review_reference_ids(payload, available)
        payload["reference_contract"] = reference_contract(case, payload["reference_ids"], index=index)
    store.activity("Worker explanation" if explaining else "Scientific review", case.current_round_id)
    scope = {"refs": [p["pending_result"]]} if explaining else {"reviewed_record_ids": payload["reference_ids"]}
    for field, ids in scope.items():
        check_refs(case, ids, [field], index=index)
    review_binding = None
    source_refs = payload.get("reference_ids", [])
    if not explaining and role == "final_reviewer" and workflow_enabled(case) and o.config.command_native_tools:
        payload, review_binding = delivery_review.prepare(case, payload, index=index)
    proofreading = (review_binding or {}).get("delivery_review_mode") == "proofread"
    try:
        if explaining and o.config.command_native_tools:
            feedback, session = worker_turn(client, case, payload, store, compact_context=o.config.command_compact_context)
            if session is not None:
                if o.config.command_compact_context or numeric.outcome_ids(session.get("outcome", {})):
                    case = query_state.remember_session(case, session, role="worker", index=index)
                return put(case, worker_session=session)
        elif not explaining and o.config.command_native_tools:
            completion = None
            prompt_name = "command_reviewer_text.md"
            if workflow_enabled(case) and role != "coverage_reviewer":
                targets = payload["review_targets"]
                stop_material = payload.get("computational_review") if role == "final_reviewer" else None

                def accept_review(cmd):
                    feedback = workflow.validate_report(case, cmd, targets=targets, source_refs=source_refs, index=index)
                    return (computational_review.accept_review(case, cmd, feedback, stop_material, catalog)
                            if stop_material is not None else feedback)

                review_model = RecordReview
                if stop_material is not None:
                    review_model = (RecordComputationalReview if stop_material["previous"] is not None
                                    else RecordFreshComputationalReview)
                completion = ("record_review", review_model, accept_review)
                prompt_name = "command_delivery_proofread.md" if proofreading else "command_reviewer_issues.md"
            read_scope = reviewer_scope(case, set(source_refs), available)
            payload, review_receipt, prior_ref = review_projection(
                case, payload, role, index=index, task_context=o.config.command_task_context)
            if o.config.command_compact_context and o.config.command_task_context:
                from mechcal.runtime.command_review_context import prepare_review
                case, payload, context_ref = prepare_review(case, payload, review_receipt, role)
                if context_ref is not None:
                    index = record_index(case)
                    available = index.available
                    read_scope[context_ref] = available[context_ref]
            if review_binding:
                review_receipt.update(review_binding)
            if proofreading:
                anchor = review_binding["scientific_review_ref"]
                payload["review_receipt_refs"] = sorted(set(payload.get("review_receipt_refs", [])) | {anchor})
            for ref in set(payload.get("review_receipt_refs", [])) | ({prior_ref} if prior_ref else set()):
                read_scope[ref] = available[ref]
            context_key = digest(payload)
            state = p.get("reviewer_session") or {"context_key": context_key, "queries": 0}
            if state["context_key"] != context_key:
                raise RuntimeError("Reviewer read session belongs to a different judgment")
            feedback, session = scoped_commentary_turn(client, case, payload, store,
                scope=read_scope, state=state, role=role, prompt_name=prompt_name, index=index, completion=completion,
                describe_tools=(lambda cmd: _describe_tools(o, case, cmd))
                    if o.config.command_computational_review and not proofreading else None,
                compact_context=o.config.command_compact_context)
            if session is not None:
                if o.config.command_compact_context or numeric.outcome_ids(session.get("outcome", {})):
                    case = query_state.remember_session(case, session, role=role,
                                                        registry=o.deps.capability_registry, index=index)
                return put(case, reviewer_session=session)
            failed_session = feedback.pop("failed_session", None)
            if role == "final_reviewer" and failed_session is not None:
                case = put(case, failed_review_session={"draft_id": p["final_draft"]["draft_id"],
                                                       "session": failed_session})
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
        handoffs = p.get("read_handoffs", {})
        selections = (p.get("worker_session") or {}).get("source_reads", [])
        if o.config.command_compact_context and selections:
            handoffs = {**handoffs, p["pending_result"]: handoff.capture(case, p["pending_result"], selections, index=index)}
        if feedback["status"] == "available":
            text = [("explanation:" + p["pending_result"], "/text", feedback["text"], None)]
            remembered, delivered = remember_values(remembered, text), remember_values(delivered, text)
        return put(case, phase="choose", ticket=None, worker_session=None,
                   read_handoffs=handoffs,
                   read_memory=remembered, latest_read_handles=delivered,
                   explanations={**p.get("explanations", {}), p["pending_result"]: feedback},
                   outcome={**p["outcome"], "result_report": payload["result"],
                            "execution_context": payload["execution_context"], "worker_explanation": feedback})
    interim = (completed_interim_review(case, index=index)
               if role == "reviewer" and feedback["status"] == "available" else p.get("interim_review"))
    if workflow_enabled(case):
        case = workflow.accept_report(case, feedback)
    if feedback.get("computational_review") is not None and feedback["status"] == "available":
        case = put(case, computational_review=feedback["computational_review"])
    memory = p.get("review_memory", {})
    receipts = p.get("review_receipts", {})
    if review_receipt is not None and feedback["status"] == "available":
        completed = {**review_receipt, "feedback": feedback}
        selections = (p.get("reviewer_session") or {}).get("source_reads", [])
        if selections:
            completed.update(source_reads=selections,
                             read_versions={h["ref"]: digest(index.available[h["ref"]]) for h in selections})
        completed["ref"] = f"review:{role}:" + digest({k: v for k, v in completed.items() if k != "ref"})
        memory = {**memory, role: completed}
        receipts = {**receipts, completed["ref"]: completed}
    return put(case, phase="choose", review=feedback, reviewer_session=None, review_memory=memory,
               review_receipts=receipts,
               interim_review=interim,
               final_review=({"draft_id": (p.get("final_draft") or {}).get("draft_id"),
                              "status": feedback["status"]} if role == "final_reviewer" else p.get("final_review")),
               final_review_revision=p["revision"] if role == "final_reviewer" else p.get("final_review_revision"))


class LocalProgressError(RuntimeError):
    """A local control cursor repeated without yielding to a model or calculator."""


def _check_local_progress(case, seen):
    p = protocol(case)
    if p["phase"] not in {"hypothesis_next", "batch_next"}:
        seen.clear()
        return
    batch = p.get("compute_batch") or {}
    step = p.get("hypothesis_step") or {}
    cursor = {"phase": p["phase"], "request_id": (p.get("latest_exchange") or {}).get("request_id"),
              "update_call_id": step.get("update_call_id"),
              "followup_call_ids": [c["id"] for c in step.get("followup_calls", [])],
              "active": batch.get("active"),
              "experiments": [(eid, p.get("experiments", {}).get(eid, {}).get("status"))
                              for eid in batch.get("ids", [])]}
    key = digest(cursor)
    if key in seen:
        raise LocalProgressError("Local command cursor repeated without model/tool work: " + json.dumps(cursor)
                                 + ". Checkpoint retained; explicit repair/resume required.")
    seen.add(key)


def run_command_loop(o, case, store, *, resume=False):
    """A stop request drains the current calculation and prevents new dispatch."""
    # Limits come from the current configuration, not a historical checkpoint.
    run_limits = dict(case.runtime["run_limits"])
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
        if o.config.command_evidence_followup:
            case = followup.enable(case)
        if o.config.command_review_workflow:
            case = workflow.enable(case)
        case = save(case, store)
        stopping = False
        local_cursors = set()

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
                _check_local_progress(case, local_cursors)
                p = protocol(case)
                if o.config.command_compact_context:
                    for name, role in (("worker_session", "worker"), ("impact_session", "impact_reviewer"),
                                       ("reviewer_session", p.get("review_role", "reviewer"))):
                        if p.get(name):
                            case = query_state.remember_session(case, p[name], role=role,
                                                                registry=o.deps.capability_registry)
                    p = protocol(case)
                allowed = o.config.unlimited_exploration or p["rounds"] < (
                    o.config.max_exploration_rounds if o.config.max_exploration_rounds is not None else o.config.max_rounds)
                case = case.touch(runtime={**case.runtime, "run_limits": {
                    **run_limits, "scientific_tools_allowed": allowed,
                    "exploration_rounds_used": p["rounds"],
                    "last_round_reserved_for_final_decision": False,
                }})
                case = save(case, store)
                if p["phase"] == "prepare":
                    case = o._prepare_structure(case, store)
                    case = put(case, phase="review" if o.config.effective_agenda_coverage_reviewer else "choose",
                               review_role="coverage_reviewer", review_changes=[])
                elif p["phase"] == "choose":
                    if o.config.command_compact_context:
                        case = save(query_state.refresh_tools(case, o.deps.capability_registry), store)
                        p = protocol(case)
                    if o.config.command_hypothesis_tracking:
                        case = save(hypotheses.sync(case), store)
                        p = protocol(case)
                    if (o.config.command_evidence_followup and o.config.effective_support_auditor
                            and followup.unreviewed_results(case)):
                        case = save(put(case, phase="impact_review"), store)
                        continue
                    model = envelope_model(o.deps.capability_registry, p.get("visible_tools", []),
                                           hypothesis_tracking=o.config.command_hypothesis_tracking,
                                           evidence_followup=o.config.command_evidence_followup,
                                           review_workflow=o.config.command_review_workflow,
                                           computational_review=o.config.command_computational_review,
                                           task_context=o.config.command_task_context,
                                           worker_tasks=o.config.command_worker_tasks)
                    if p.get("pending_command") is None:
                        case = save(recovery.before_model_turn(case,
                            max_model_turns=o.config.command_delivery_max_model_turns), store)
                        store.activity("Planner command", case.current_round_id)
                        try:
                            if o.config.command_native_tools:
                                payload = planner_payload(o, case, native=True)
                                if o.config.command_task_context:
                                    from mechcal.runtime.command_task_context import prepare_planner
                                    case, payload = prepare_planner(case, payload)
                                    case = save(case, store)
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
                elif p["phase"] == "worker_task":
                    case = worker_tasks.turn(o, case, store)
                elif p["phase"] == "batch_next":
                    case = next_member(o, case)
                elif p["phase"] == "impact_review":
                    case = save(recovery.before_model_turn(case,
                        max_model_turns=o.config.command_delivery_max_model_turns), store)
                    case = impact_review_turn(o, case, store)
                elif p["phase"] in {"explain", "review"}:
                    case = save(recovery.before_model_turn(case,
                        max_model_turns=o.config.command_delivery_max_model_turns), store)
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
            failure = {"kind": "delivery_recovery_limit" if isinstance(exc, recovery.DeliveryRecoveryLimit) else
                       "local_no_progress" if isinstance(exc, LocalProgressError) else "system_interruption",
                       "type": type(exc).__name__, "message": str(exc),
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
                                  with_draft=bool(p.get("judgment_draft")),
                                  evidence_followup=followup.state(case) is not None,
                                  review_workflow=workflow_enabled(case),
                                  computational_review=o.config.command_computational_review,
                                  worker_tasks=o.config.command_worker_tasks)
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
