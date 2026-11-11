"""Explicit staged loop with a case-local atomic checkpoint and safe resume."""

import fcntl
import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from mechcal.agents.evidence_context import relevant_evidence
from mechcal.capabilities.discovery import LoadedToolState
from mechcal.runtime.prompts import load_prompt

from mechcal.runtime.progress import atomic_json
from mechcal.runtime.persistence import RunStore
from mechcal.runtime.questions import (
    apply_assimilation, apply_final_revision, calculation_identity, delivery_summary, dependency_inputs,
    replace_state, request_signature, selected_source_id, state,
)
from mechcal.schemas import AgentReport, CaseRun, DispatchRequest, PlannerDecision, RoundRecord
from mechcal.schemas.decisions import StopReview
from mechcal.schemas.question_loop import ResultReview
from mechcal.runtime.retry import ComponentResponseRejected, SchemaRetryError
from mechcal.runtime.api_failures import api_pause_reason, is_review_service_failure


def checkpoint(case, store):
    # This file is authoritative for resume. Derived files can be reconstructed.
    atomic_json(store.root / "question_checkpoint.json", case.model_dump(mode="json"))
    store.write_case_run(case)
    return case


def _request_events(case: CaseRun, store: RunStore) -> Callable[[dict[str, Any]], None]:
    s = state(case)
    context = {"case_id": case.case_id, "round_id": f"R{s['round_index']:03d}",
               "phase": s["phase"], "revision": s["revision"], "resume_generation": s.get("resume_generation", 0)}
    attempt = 0
    max_attempts = 3

    def record(event):
        nonlocal attempt, max_attempts
        attempt = event.get("attempt", attempt)
        max_attempts = event.get("max_attempts", max_attempts)
        row = {**context, "timestamp": time.time(), "attempt": attempt, **event}
        store.append_jsonl("history/planner_requests.jsonl", row)
        if event["kind"] == "ranking_explanation_draft":
            store.activity("Planner explain frozen ranking", context["round_id"],
                           "Judgment staged; explanations pending; no scientific tool action")
        atomic_json(store.root / "planner_request_state.json", {
            **context, "attempt": attempt, "kind": event["kind"],
            "stage": event.get("stage"),
            "max_attempts": max_attempts, "error": event.get("error"),
        })
    return record


def _memory_kwargs(case, store):
    return {"memory_store": store} if case.runtime.get("planner_case_memory") else {}


def _failure_kind(exc: Exception) -> str:
    import httpx
    from openai import APIError

    cause = exc
    while cause is not None:
        if isinstance(cause, ComponentResponseRejected):
            return cause.failure_kind
        if isinstance(cause, SchemaRetryError):
            return "request_retries_exhausted"
        if isinstance(cause, (APIError, httpx.HTTPError, TimeoutError, ConnectionError)):
            return "model_or_transport_failure"
        if isinstance(cause, OSError):
            return "storage_failure"
        cause = cause.__cause__
    return "runtime_failure"


def configuration_fingerprint(orchestrator, case):
    """Shared by strict resume checks and explicit, audited checkpoint migration."""
    clients = [orchestrator.deps.planner, orchestrator.deps.photophysics_arbiter,
               orchestrator.deps.mechanism_agenda]
    if orchestrator.config.planner_command_protocol:
        clients = [orchestrator.deps.planner, orchestrator.deps.scientific_reviewer,
                   orchestrator.deps.exploration_agent]
        if any(agent is None for agent in clients):
            raise ValueError("Command workflow requires explicit ScientificReviewerAgent and ExplorationAgent dependencies")
    model_settings = []
    for agent in clients:
        settings = getattr(agent.llm_client, "settings", None)
        model_settings.append({k: v for k, v in asdict(settings).items() if k != "api_key"}
                              if is_dataclass(settings) else None)
    package = Path(__file__).resolve().parents[1]
    return hashlib.sha256(json.dumps({
            "config": asdict(orchestrator.config), "input": case.input.model_dump(mode="json"),
            "registry": orchestrator.deps.capability_registry.snapshot().model_dump(mode="json"),
            "model_settings": model_settings,
            "prompts": {p.name: load_prompt(p.name) for p in sorted((package / "prompts").glob("*.md"))},
            "code": {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(package.rglob("*.py"))},
            "chemistry_environment": {k: v for k, v in os.environ.items()
                                      if k.startswith(("MECHCAL_AMESP_", "MECHCAL_RDKIT_"))},
        }, sort_keys=True, default=str).encode()).hexdigest()


def run_question_loop(orchestrator, case, store, *, resume=False):
    with (store.root / ".question_loop.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This case already has an active question loop") from exc
        fingerprint = configuration_fingerprint(orchestrator, case)
        saved = store.root / "question_checkpoint.json"
        if resume:
            case = CaseRun.model_validate_json(saved.read_text())
            if state(case).get("configuration_hash") != fingerprint:
                raise ValueError("Resume input/configuration/registry differs from checkpoint")
            discovery_path = store.root / "history/tool_discovery.jsonl"
            if state(case)["phase"] != "done" and discovery_path.exists():
                # The full event retains the allowance; the display snapshot only has IDs.
                # Reject a truncated event rather than silently restore an older allowance.
                events = [line for line in discovery_path.read_text().splitlines() if line.strip()]
                discovery = json.loads(events[-1]) if events else None
                if not discovery or discovery.get("case_id") != case.case_id:
                    raise ValueError("Tool discovery checkpoint belongs to another case")
                loaded = LoadedToolState.model_validate(discovery["state"])
                if loaded.case_id != case.case_id or discovery["batches_used"] not in range(3):
                    raise ValueError("Invalid tool discovery resume state")
                case = case.touch(runtime={**case.runtime,
                    "tool_discovery": loaded.model_dump(mode="json"),
                    "tool_discovery_budget": {"case_id": case.case_id,
                                              "round_id": discovery["round_id"],
                                              "decision_revision": discovery.get("decision_revision"),
                                              "batches_used": discovery["batches_used"],
                                              "visible_ids": discovery.get("visible_ids", list(loaded.loaded_tools))},
                })
            if state(case)["phase"] != "done":
                attempt_file = store.root / "planner_request_state.json"
                prior_attempt = json.loads(attempt_file.read_text()) if attempt_file.exists() else None
                generation = state(case).get("resume_generation", 0) + 1
                store.append_jsonl("history/resume_events.jsonl", {
                    "case_id": case.case_id, "phase": state(case)["phase"], "resume_generation": generation,
                    "configuration_hash": fingerprint, "previous_request_state": prior_attempt,
                    "request_retry_policy": "Explicit resume grants a new bounded attempt group; tool discovery allowance is retained.",
                    "timestamp": time.time(),
                })
                runtime = {k: v for k, v in case.runtime.items() if k not in {"failure", "termination"}}
                case = checkpoint(replace_state(case.touch(status="running", runtime=runtime),
                                                blocked_failure=None, resume_generation=generation), store)
        elif saved.exists():
            raise FileExistsError("Do not overwrite an existing question-loop checkpoint")
        else:
            store.write_case_input(case.input)
            store.write_capability_catalog(orchestrator.deps.capability_registry.snapshot())
            store.write_json("ablation_manifest.json", case.runtime["ablation"])
            case = replace_state(case, phase="prepare", revision=0, round_index=1,
                                 configuration_hash=fingerprint, questions={}, results=[], pending_results=[])
            case = checkpoint(case, store)
        try:
            return _drive(orchestrator, case, store)
        except Exception as exc:
            # Preserve the last valid stage; no automatic tool replay on unknown completion.
            latest = CaseRun.model_validate_json(saved.read_text())
            kind = _failure_kind(exc)
            failure = {"type": type(exc).__name__, "kind": kind, "message": str(exc),
                       "phase": state(latest)["phase"], "revision": state(latest)["revision"],
                       "current_round_id": f"R{state(latest)['round_index']:03d}"}
            store.write_json("question_loop_failure.json", failure)
            store.append_jsonl("history/runtime_failures.jsonl", {**failure, "timestamp": time.time()})
            paused = isinstance(exc, ComponentResponseRejected)
            checkpoint(replace_state(latest.touch(status="paused" if paused else "failed", runtime={
                **latest.runtime, "failure": failure,
                "termination": {"reason": kind, "scientific_completion": False, "requires_explicit_resume": True},
            }), blocked_failure=failure), store)
            if paused:
                store.activity("Model response needs correction", failure["current_round_id"],
                               "Accepted evidence and judgment preserved; explicit resume required.")
            raise


def _drive(o, case, store):
    while state(case)["phase"] != "done":
        s = state(case)
        round_id = f"R{s['round_index']:03d}"
        used = len(case.round_ids)
        allowed = o.config.unlimited_exploration or used < o.config.max_exploration_rounds
        case = case.touch(runtime={**case.runtime, "run_limits": {
            **case.runtime["run_limits"], "exploration_rounds_used": used,
            "scientific_tools_allowed": allowed,
        }})
        phase = s["phase"]
        if phase == "prepare":
            store.activity("Prepare structure", round_id)
            case = o._prepare_structure(case, store)
            case = replace_state(case, phase="coverage")
        elif phase == "coverage":
            if o.config.effective_agenda_coverage_reviewer:
                case = o._review_agenda_coverage(case, store, round_id=round_id)
            case = replace_state(case, phase="candidate_expansion"
                                 if o.config.candidate_expansion_mode != "off" else "assimilate")
        elif phase == "candidate_expansion":
            from mechcal.runtime.candidate_expansion import admit, new_observations, proposal_payload

            fresh = new_observations(case)
            if fresh:
                store.activity("Propose evidence-driven candidates", round_id)
                store.write_json(f"rounds/{round_id}/candidate_input.json", proposal_payload(case, fresh))
                proposal = o.deps.mechanism_agenda.propose_candidates(
                    case, fresh, event_sink=store.model_events("candidate_expansion", case_id=case.case_id,
                                                             round_id=round_id),
                )
                case = admit(case, fresh, proposal)
                # Persist admission before rendering derived logs; resume cannot re-admit it.
                case = checkpoint(case, store)
                store.write_json(f"rounds/{round_id}/candidate_expansion.json",
                                 state(case)["candidate_expansion"]["batches"][-1])
            else:
                store.write_json(f"rounds/{round_id}/candidate_expansion.json", {
                    "status": "not_triggered", "reason": "No new non-preparation observations",
                })
                case = replace_state(case, phase="assimilate")
        elif phase == "review":
            from mechcal.runtime.questions import apply_result_review

            store.activity("Audit worker explanations", round_id)
            review = ResultReview(status="disabled", reason="Auditor disabled by configuration")
            if o.config.effective_support_auditor:
                try:
                    review = o.deps.photophysics_arbiter.review_results(
                        case, event_sink=store.model_events("result_review", case_id=case.case_id, round_id=round_id),
                    )
                except Exception as exc:
                    if (api_pause_reason(exc) or not is_review_service_failure(exc)
                            or not o.config.continue_on_arbiter_failure):
                        raise
                    review = ResultReview(status="unavailable", reason=f"{type(exc).__name__}: {exc}")
            store.write_json(f"rounds/{round_id}/result_review.json", review)
            case = apply_result_review(case, review)
        elif phase == "assimilate":
            from mechcal.runtime.predecision import ensure_comparison
            case = ensure_comparison(o, case, store, round_id)
            store.activity("Planner assimilate results", round_id)
            update = o.deps.planner.assimilate(
                case, validation_event_sink=lambda event: store.append_jsonl(
                    "history/planner_assimilation_failures.jsonl", event,
                ),
                request_event_sink=_request_events(case, store),
                **_memory_kwargs(case, store),
            )
            case = apply_assimilation(case, update)
            case = case.touch(runtime={**case.runtime, "last_planner_assessment": {
                "round_id": round_id, **update.decision_assessment.model_dump(mode="json"),
            }})
            store.write_json(f"questions/revision_{update.revision:04d}.json", state(case)["assimilation"])
            if case.runtime.get("comparative_evidence"):
                store.write_json("initial_judgment.json", state(case)["initial_judgment"])
                store.write_json(f"questions/comparison_{update.revision:04d}.json", state(case)["last_comparison"])
            if s.get("argument_review_pending"):
                store.append_jsonl("history/argument_reviews.jsonl", {
                    "event": "planner_response", "round_id": round_id,
                    "revision": update.revision, "audit_responses": update.audit_responses,
                })
                # This records the response basis for deduplication, not audit approval.
                case = replace_state(case, argument_review_pending=False,
                                     argument_review={**s["argument_review"],
                                                      "planner_response_revision": update.revision,
                                                      "planner_responses": update.audit_responses},
                                     argument_review_response_basis=_argument_basis(case))
            elif (o.config.effective_support_auditor
                  and (case.mechanism_support_arguments or case.runtime.get("comparative_evidence"))
                  and s.get("argument_review_round") != round_id
                  and s.get("argument_review_response_basis") != _argument_basis(case)):
                case = replace_state(case, phase="argument_audit")
        elif phase == "argument_audit":
            store.activity("Review scientific arguments", round_id)
            case = case.touch(current_round_id=round_id, photophysics_review=None)
            if case.runtime.get("incremental_judgment"):
                from mechcal.runtime.incremental_audit import audit_incrementally
                case = audit_incrementally(o, case, store)
            else:
                case = o._audit_current_round(case, store)
            review = case.photophysics_review
            feedback = {
                "status": "reviewed" if review is not None else "unavailable",
                "basis_revision": s["revision"],
                "support_ids": [a.support_id for a in case.mechanism_support_arguments],
                "review": review.model_dump(mode="json") if review is not None else None,
                "reason": "" if review is not None else "Internal review unavailable; no approval inferred.",
            }
            store.append_jsonl("history/argument_reviews.jsonl", {
                "event": "review", "round_id": round_id, **feedback,
            })
            if case.runtime.get("incremental_judgment"):
                from mechcal.runtime.incremental_audit import route_feedback
                case = route_feedback(case, feedback)
            else:
                case = replace_state(case, phase="assimilate", argument_review_pending=True,
                                     argument_review=feedback, argument_review_round=round_id)
        elif phase == "act":
            store.activity("Planner choose action", round_id)
            decision = o.deps.planner.plan_initial(case, round_id=round_id,
                                                  request_event_sink=_request_events(case, store),
                                                  **_memory_kwargs(case, store),
                                                  **o._discovery_kwargs(store))
            case = o._retain_tool_discovery(case, decision)
            if decision.action == "finalize" and not decision.failure.is_failure and not s.get("stop_review_done"):
                case = _stop_review(o, case, decision, store)
            else:
                if s.get("stop_review_done"):
                    store.append_jsonl("history/stop_reviews.jsonl", {
                        "event": "response", "round_id": round_id,
                        "decision": decision.model_dump(mode="json"),
                    })
                gate = o.deps.gate.apply(decision, case)
                if gate.status in {"retry_schema", "retry_policy"}:
                    # Accepted actions use the same validator as the gate. Divergence is a runtime defect.
                    raise RuntimeError("Accepted action rejected by execution gate: " + "; ".join(gate.violations))
                case = case.touch(current_round_id=round_id)
                audit_pending = (gate.terminal and gate.status in {"finalize", "stop"}
                                 and not decision.failure.is_failure and o.config.effective_support_auditor)
                delivery_pending = (gate.terminal and gate.status in {"finalize", "stop"}
                                    and not decision.failure.is_failure and case.runtime.get("incremental_judgment"))
                next_phase = ("final_audit" if audit_pending else "final_revision" if delivery_pending
                              else "finish" if gate.terminal else "dispatch")
                if delivery_pending and not audit_pending:
                    case = replace_state(case, final_audit={"status": "disabled", "review": None})
                case = replace_state(case, phase=next_phase,
                                     decision=decision.model_dump(mode="json"), gate=gate.model_dump(mode="json"),
                                     completed_dispatch_ids=[], in_flight=None)
                record = RoundRecord(case_id=case.case_id, round_id=round_id,
                                     round_index=s["round_index"], planner_decision=decision,
                                     gate_result=gate, dispatch_requests=list(gate.dispatch_requests))
                store.write_planner_decision(record)
                store.append_jsonl("history/planner_messages.jsonl", decision.model_dump(mode="json"))
        elif phase == "dispatch":
            case = _dispatch(o, case, store)
        elif phase == "final_audit":
            case = case.touch(photophysics_review=None,
                              runtime={**case.runtime, "final_support_audit_round": round_id})
            case = o._audit_current_round(case, store)
            review = case.photophysics_review
            reviewed = review is not None and review.round_id == round_id
            case = replace_state(case, phase="final_revision", final_audit={
                "status": "reviewed" if reviewed else "unavailable",
                "review": review.model_dump(mode="json") if reviewed else None,
            })
            if reviewed and case.runtime.get("final_delivery_v2"):
                from mechcal.runtime.final_delivery import audit_digest
                case = replace_state(case, final_audit={
                    **state(case)["final_audit"],
                    "content_digest": audit_digest(case, case.mechanism_support_arguments),
                })
        elif phase == "final_revision":
            if case.runtime.get("final_delivery_v2"):
                from mechcal.runtime.final_delivery import advance
                case = advance(o, case, store)
                continue
            store.activity("Planner revise final arguments", round_id)
            revision = o.deps.planner.revise_final_arguments(
                case, validation_event_sink=lambda event: store.append_jsonl(
                    "history/planner_final_revision_failures.jsonl", event,
                ),
                request_event_sink=_request_events(case, store),
                **_memory_kwargs(case, store),
            )
            case, decision = apply_final_revision(case, revision, o.deps.capability_registry)
            gate = o.deps.gate.apply(decision, case)
            if gate.status not in {"finalize", "stop"}:
                raise ValueError("Final revision failed deterministic validation: " + "; ".join(gate.violations))
            store.write_json(f"rounds/{round_id}/planner_final_revision.json", revision)
            store.write_json(f"rounds/{round_id}/planner_final_decision.json", decision)
            store.write_json(f"questions/revision_{revision.update.revision:04d}.json", state(case)["assimilation"])
            if case.runtime.get("comparative_evidence"):
                store.write_json(f"questions/comparison_{revision.update.revision:04d}.json", state(case)["last_comparison"])
            case = case.touch(runtime={**case.runtime, "last_planner_assessment": {
                "round_id": round_id, **revision.update.decision_assessment.model_dump(mode="json"),
            }})
            case = replace_state(case, phase="finish", final_revision_complete=True,
                                 decision=decision.model_dump(mode="json"), gate=gate.model_dump(mode="json"))
        elif phase == "finish":
            from mechcal.schemas import DecisionGateResult

            decision = PlannerDecision.model_validate(s["decision"])
            gate = DecisionGateResult.model_validate(s["gate"])
            record = RoundRecord(case_id=case.case_id, round_id=round_id,
                                 round_index=s["round_index"], planner_decision=decision, gate_result=gate)
            case = o._finish_case(case, decision, record, store, gate,
                                  final_audit_complete=bool(s.get("final_revision_complete")))
            action = decision.raw_response.get("question_action", {})
            case = case.touch(runtime={**case.runtime, "termination": {
                **case.runtime.get("termination", {}), "decision_mode": "question_loop_v1",
                "question_dispositions": action.get("question_dispositions", {}),
                "ranking_justification": action.get("ranking_justification"),
                "ranking_uncertainty_material": action.get("ranking_uncertainty_material"),
                "final_revision_complete": s.get("final_revision_complete", False),
                "final_audit_status": s.get("final_audit", {}).get("status"),
            }})
            case = replace_state(case, phase="done")
        else:
            raise ValueError(f"Unknown question-loop phase: {phase}")
        case = checkpoint(case, store)
    return case


def _argument_basis(case):
    arguments = case.mechanism_support_arguments
    return hashlib.sha256(json.dumps({
        "comparison_order": case.portfolio.model_dump(mode="json")
        if case.runtime.get("comparative_evidence") else None,
        "arguments": [a.model_dump(include={"support_id", "target_label", "finding", "warrant",
                                            "boundary", "observation_refs"}, mode="json") for a in arguments],
        "observations": [e.model_dump(mode="json") for e in relevant_evidence(case, arguments)],
    }, sort_keys=True).encode()).hexdigest()


def _stop_review(o, case, decision, store):
    store.activity("Stopping review", decision.round_id)
    store.append_jsonl("history/stop_reviews.jsonl", {
        "event": "proposal", "round_id": decision.round_id, "decision": decision.model_dump(mode="json"),
    })
    review = StopReview(status="disabled", rationale="Coverage Reviewer disabled by configuration")
    if o.config.effective_agenda_coverage_reviewer:
        try:
            review = o.deps.mechanism_agenda.review_stop(
                case, decision, event_sink=store.model_events("stop_review", case_id=case.case_id,
                                                              round_id=decision.round_id),
            )
        except Exception as exc:
            if api_pause_reason(exc) or not is_review_service_failure(exc):
                raise
            review = StopReview(status="unavailable", rationale=f"{type(exc).__name__}: {exc}")
    store.append_jsonl("history/stop_reviews.jsonl", {
        "event": "review", "round_id": decision.round_id, "review": review.model_dump(mode="json"),
    })
    case = case.touch(runtime={**case.runtime, "stop_review_feedback": review.model_dump(mode="json")})
    # Any ranking change in response to review must pass through assimilation.
    return replace_state(case, phase="assimilate", stop_review_done=True)


def _find_completed(store: RunStore, request: DispatchRequest) -> dict[str, Any] | None:
    saved = store.read_execution_checkpoint(request)
    if saved is not None:
        if saved["stage"] != "reported":
            return None
        # Typed validation is mandatory even when history is absent or damaged.
        from mechcal.schemas import ToolExecutionResult

        report = AgentReport.model_validate(saved["report"])
        result = ToolExecutionResult.model_validate(saved["tool_result"])
        if (report.round_id != request.round_id or report.agent_name != request.agent_name
                or any(getattr(result, key) != getattr(request, key)
                       for key in ("dispatch_id", "round_id", "agent_name", "capability_id"))
                or result.selected_route != request.route):
            raise ValueError("Saved execution result/report identity does not match request")
        # Match the normal dispatch envelope; this is runtime provenance, not
        # a new Worker observation. Older reported checkpoints predate it.
        report = report.model_copy(update={"structured_results": {
            **report.structured_results, "dispatch_request": request.model_dump(mode="json"),
        }})
        return {"event": "agent_report", "dispatch": saved["dispatch"],
                "report": report.model_dump(mode="json"), "tool_result": result.model_dump(mode="json")}
    # Compatibility for old runs predating execution checkpoints only.
    path = store.root / "history/tool_events.jsonl"
    if not path.exists():
        return None
    for line in reversed(path.read_text().splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (row.get("event") == "agent_report"
                and row["dispatch"]["dispatch_id"] == request.dispatch_id
                and row["dispatch"]["round_id"] == request.round_id):
            return row
    return None


def _dispatch(o, case, store):
    s = state(case)
    decision = PlannerDecision.model_validate(s["decision"])
    for request in decision.dispatch_requests:
        if request.dispatch_id in state(case)["completed_dispatch_ids"]:
            continue
        if state(case).get("in_flight"):
            completed = _find_completed(store, request)
            if completed is None:
                if store.read_execution_checkpoint(request) is None:
                    raise RuntimeError("In-flight calculation has unknown completion; reconcile records before resuming")
                o._dispatch_all([request], case, store)
                completed = _find_completed(store, request)
                if completed is None:
                    raise RuntimeError("Recovered tool report is missing from execution history")
        else:
            case = checkpoint(replace_state(case, in_flight=request.dispatch_id), store)
            o._dispatch_all([request], case, store)
            completed = _find_completed(store, request)
            if completed is None:
                raise RuntimeError("Completed tool result is missing from execution history")
        if DispatchRequest.model_validate(completed["dispatch"]).model_dump(mode="json") != request.model_dump(mode="json"):
            raise ValueError("Recovered execution does not match the saved request")
        report = AgentReport.model_validate(completed["report"])
        result = completed["tool_result"]
        source_id = selected_source_id(request, case, o.deps.capability_registry)
        identity = calculation_identity(request, case, o.deps.capability_registry)
        case = o._apply_reports(case, [report])
        packet = {
            "result_id": result["tool_result_id"], "dispatch_id": request.dispatch_id,
            "question_ids": request.question_ids, "capability_id": request.capability_id,
            "expected_observation": request.expected_observation, "status": result["status"],
            "investigation_plan": next((p for p in decision.raw_response.get("question_action", {}).get(
                "investigation_plans", []) if p["dispatch_id"] == request.dispatch_id), None),
            "reuse_result_ids": request.reuse_result_ids,
            "expected_new_information": request.expected_new_information,
            "requested_parameters": request.tool_args, "input_refs": request.input_refs,
            "resolved_source_artifact_id": source_id,
            "structured_results": result["structured_results"], "failure": result["failure"],
            "worker_explanation": report.result_explanation.model_dump(mode="json")
            if report.result_explanation else None,
            "evidence_refs": [e.evidence_id for e in report.evidence_units],
            "signature": request_signature(request, o.deps.capability_registry),
            "calculation_identity": identity,
            "dependencies": o.deps.capability_registry.require(request.capability_id).metadata.get("execution_dependencies", []),
            "dependency_inputs": dependency_inputs(request),
            "failed_dependencies": ["amesp." + result["failure"]["details"]["failed_step"]]
            if result["failure"].get("details", {}).get("failed_step") else [],
        }
        from mechcal.runtime.observation_delivery import delivery_receipt
        packet["required_observables"] = [r.model_dump(mode="json") for r in request.required_observables]
        packet["observation_receipt"] = delivery_receipt(request, packet["structured_results"])
        packet["delivery"] = delivery_summary(packet, state(case)["results"])
        case = checkpoint(replace_state(
            case, in_flight=None,
            completed_dispatch_ids=[*state(case)["completed_dispatch_ids"], request.dispatch_id],
            results=[*state(case)["results"], packet],
            pending_results=[*state(case)["pending_results"], packet],
        ), store)
    from mechcal.schemas import DecisionGateResult

    reports = []
    for request in decision.dispatch_requests:
        completed = _find_completed(store, request)
        if completed is None:
            raise RuntimeError("Cannot record a round without its completed execution")
        reports.append(AgentReport.model_validate(completed["report"]))
    record = RoundRecord(case_id=case.case_id, round_id=decision.round_id,
                         round_index=s["round_index"], planner_decision=decision,
                         dispatch_requests=decision.dispatch_requests, agent_reports=reports,
                         gate_result=DecisionGateResult.model_validate(s["gate"]))
    case = o._record_round(case, record, store)
    case = case.touch(runtime={**case.runtime, "stop_review_feedback": None})
    return replace_state(case, phase="review", round_index=s["round_index"] + 1, stop_review_done=False)
