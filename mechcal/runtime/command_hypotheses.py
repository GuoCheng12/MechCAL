"""Durable provisional beliefs, independent of reviewed arguments and delivery."""

from pydantic import BaseModel, ConfigDict, Field

from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime.command_operations import OperationRejected, check_owner, protocol, put, reference_issues, reject
from mechcal.schemas.commands import ExplorationChange, UpdateHypotheses
from mechcal.runtime import command_followup as followup
from mechcal.runtime import command_investigation as investigation
from mechcal.runtime.command_views import candidate_view
from mechcal.runtime.command_action_policy import QUERY_TURNS, action_policy, read_allowance, science_read_attempt
from mechcal.runtime.command_reads import READ_KINDS, query_refs


class ExplorationState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = 1
    case_id: str
    assessments: int = 0
    entries: dict[str, ExplorationChange] = Field(default_factory=dict)
    observed_results: list[str] = Field(default_factory=list)
    assessed_results: list[str] = Field(default_factory=list)
    query_request_ids: list[str] = Field(default_factory=list)
    note: str = ""
    event: dict | None = None


def state(case):
    check_owner(case)
    saved = protocol(case).get("exploration")
    result = ExplorationState.model_validate(saved) if saved is not None else ExplorationState(case_id=case.case_id)
    if result.case_id != case.case_id or any(label != row.label or label not in MECHANISM_POOL
                                            for label, row in result.entries.items()):
        raise RuntimeError("Exploration record identity or ownership mismatch")
    return result


def sync(case):
    """Only called at a Planner boundary, after the whole compute batch returns."""
    current = state(case)
    ids = list(protocol(case).get("results", {}))
    if not set(current.observed_results).issubset(ids):
        raise RuntimeError("Previously observed results disappeared from the case")
    if ids != current.observed_results:
        current.observed_results = ids
        current.query_request_ids = []
    return put(case, exploration=current.model_dump(mode="json"))


def needs_assessment(current):
    return current.assessments == 0 or current.assessed_results != current.observed_results


def view(case):
    current = state(sync(case))
    known = [row.model_dump(mode="json") for row in current.entries.values() if row.confidence is not None]
    known.sort(key=lambda row: (-row["confidence"], list(MECHANISM_POOL).index(row["label"])))
    # Ties retain the same rank; taxonomy order is only a stable display order.
    for row in known:
        row["rank"] = 1 + sum(other["confidence"] > row["confidence"] for other in known)
    unknown = [current.entries[label].model_dump(mode="json") if label in current.entries
               else {"label": label, "confidence": None} for label in MECHANISM_POOL
               if label not in current.entries or current.entries[label].confidence is None]
    return {"assessment_required": needs_assessment(current), "assessment_count": current.assessments,
            "query_turns_remaining": read_allowance(put(case, exploration=current.model_dump(mode="json"))),
            "new_result_ids": [rid for rid in current.observed_results if rid not in current.assessed_results],
            "confidence_ranking": [candidate_view(case, row) for row in known],
            "unassessed": [candidate_view(case, row) for row in unknown], "last_note": current.note,
            **({"evidence_followup": followup.view(case)} if followup.state(case) is not None else {})}


def record_read_attempt(case, exchange):
    if not science_read_attempt(case, exchange):
        return case
    current = state(sync(case))
    rid = exchange["request_id"]
    if rid not in current.query_request_ids:
        current.query_request_ids.append(rid)
    return put(case, exploration=current.model_dump(mode="json"))


def guard(case, command):
    current = state(sync(case))
    if command.kind in READ_KINDS | {"read_batch"}:
        rid = (protocol(case).get("latest_exchange") or {}).get("request_id")
        if rid not in current.query_request_ids:
            action_policy(case).guard_read(query_refs(command))
    decision = command.kind in {"compute", "compute_batch", "delegate_computation", "revise_judgment", "amend_judgment",
                                "finalize", "submit_final", "amend_final", "review_final"}
    decision = decision or (command.kind == "manage_experiments" and command.action == "retry")
    if decision and needs_assessment(current):
        reject([], "assessment_required", "The initial or newly available results have no provisional assessment yet.",
               "Call update_hypotheses, alone or alongside the next computation. Confidence may remain unchanged; "
               "unassessed mechanisms stay null. No new computation executed.")


def update(case, command: UpdateHypotheses):
    current = state(sync(case))
    seen, issues = set(), []
    for i, change in enumerate(command.changes):
        if change.label in seen:
            reject(["changes", i, "label"], "duplicate_edit", "A candidate is edited twice in one assessment.",
                   "Combine the edits for this label.")
        seen.add(change.label)
        issues += reference_issues(case, change.refs, ["changes", i, "refs"], scientific=True)
        issues += reference_issues(case, change.limitation_refs, ["changes", i, "limitation_refs"],
                                   scientific=True, allow_failed=True)
    if issues:
        raise OperationRejected(issues)
    changes = [row.model_copy(update={"explanation": current.entries[row.label].explanation})
               if row.explanation is None and row.label in current.entries else row
               for row in command.changes]
    changed = [row for row in changes if current.entries.get(row.label) != row]
    newly_assessed = [rid for rid in current.observed_results if rid not in current.assessed_results]
    next_entries = {**current.entries, **{row.label: row for row in changed}}
    updated = current.model_copy(update={"entries": next_entries})
    candidate = followup.apply_update(put(case, exploration=updated.model_dump(mode="json")), command)
    followup_changed = protocol(candidate).get("evidence_followup") != protocol(case).get("evidence_followup")
    candidate = investigation.record_choice(candidate, command)
    choice_changed = protocol(candidate).get("investigation_policy") != protocol(case).get("investigation_policy")
    if not changed and not needs_assessment(current) and command.note == current.note and not followup_changed and not choice_changed:
        return case, {"status": "unchanged", "committed": False, "assessment_count": current.assessments}
    before = {row.label: current.entries[row.label].model_dump(mode="json")
              if row.label in current.entries else None for row in changed}
    current.entries.update({row.label: row for row in changed})
    current.assessments += 1
    current.assessed_results = list(current.observed_results)
    current.note = command.note
    details = {"assessment_count": current.assessments, "new_result_ids": newly_assessed,
               "before": before, "changes": [row.model_dump(mode="json") for row in changed], "note": command.note,
               "scientific_computations": protocol(case).get("rounds", 0)}
    if followup_changed:
        details["impact_responses"] = [row.model_dump(mode="json") for row in getattr(command, "responses", [])]
        details["comparison_questions"] = [row.model_dump(mode="json") for row in getattr(command, "questions", [])]
    if choice_changed:
        details["investigation_choice"] = investigation.current(candidate)
    if investigation.enabled(candidate):
        details["impact_coverage"] = {iid: {"label": item["label"], "disposition": item["disposition"]}
            for iid, item in (protocol(candidate).get("evidence_followup") or {}).get("impacts", {}).items()}
    current.event = {"event_id": f"hypotheses:{current.assessments}", "kind": "hypotheses",
                     "role": "planner", "status": "updated" if changed else "retained", "details": details}
    result = {"status": "hypotheses_updated", "committed": True, "assessment_count": current.assessments,
              "changed_labels": [row.label for row in changed], "assessed_result_ids": newly_assessed}
    return put(candidate, exploration=current.model_dump(mode="json")), result


def complete_step(case):
    """Join native receipts only once the follow-up operation has actually returned."""
    p = protocol(case)
    step = p.get("hypothesis_step")
    if not step or p["phase"] not in {"choose", "done"}:
        return case
    exchange = p.get("latest_exchange") or {}
    if p.get("outcome_request_id") != exchange.get("request_id"):
        return case
    outcome = p["outcome"]
    per_call = dict(outcome.get("call_results", {}))
    if not per_call:
        per_call = {call["id"]: outcome for call in step["followup_calls"]}
    per_call[step["update_call_id"]] = step["update_result"]
    if set(per_call) != {call["id"] for call in exchange["calls"]}:
        raise RuntimeError("Hypothesis/operation receipts do not cover the native call IDs")
    return put(case, hypothesis_step=None, outcome={"status": "hypothesis_step_completed", "call_results": per_call})
