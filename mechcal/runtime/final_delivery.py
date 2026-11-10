"""Resumable final delivery without regenerating accepted scientific decisions."""

import hashlib
import json

from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.public_ids import public_case_id_from_metadata
from mechcal.runtime.final_evidence import validate_final_evidence
from mechcal.runtime.questions import decision_from_action, replace_state, state, validate_action
from mechcal.runtime.result_contract import operational_evidence_refs
from mechcal.runtime.retry import ComponentResponseRejected, RequestValidationError
from mechcal.schemas import PlannerDecision
from mechcal.schemas.final_delivery import FinalDeliveryDraft
from mechcal.schemas.final_evidence import NoSubmittedEvidence, SubmittedEvidence
from mechcal.schemas.question_loop import QuestionAction


MAX_PLANNER_ATTEMPTS = 9
MAX_AUDIT_ATTEMPTS = 3


class FinalDeliveryIncomplete(ComponentResponseRejected):
    failure_kind = "final_delivery_incomplete"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def argument_content(arguments):
    return [a.model_dump(mode="json", exclude={"audit_status", "audit_notes"}) for a in arguments]


def base_digest(case):
    s = state(case)
    return digest({
        "case_id": case.case_id, "input": case.input.model_dump(mode="json"),
        "revision": s["revision"], "decision": s["decision"], "questions": s.get("questions", {}),
        "portfolio": case.portfolio.model_dump(mode="json"),
        "arguments": argument_content(case.mechanism_support_arguments),
        "evidence": case.evidence_ledger.model_dump(mode="json"),
        "artifacts": case.artifact_manifest.model_dump(mode="json"),
        "comparison": s.get("last_comparison"), "stop_review": case.runtime.get("stop_review_feedback"),
        "configuration": s.get("configuration_hash"),
    })


def audit_digest(case, arguments):
    return digest({"base": base_digest(case), "arguments": argument_content(arguments)})


def audit_matches(case, arguments, review):
    ids = [a.support_id for a in review.support_argument_audits]
    return (review.case_id == public_case_id_from_metadata(case.input.metadata)
            and review.round_id == case.current_round_id
            and len(ids) == len(set(ids))
            and set(ids) == {a.support_id for a in arguments})


def open_draft(case):
    saved = state(case).get("final_delivery")
    if saved:
        draft = FinalDeliveryDraft.model_validate(saved)
        if (draft.case_id != case.case_id or draft.base_revision != state(case)["revision"]
                or draft.base_digest != base_digest(case)):
            raise FinalDeliveryIncomplete(["Final-delivery draft belongs to a different or changed accepted state."])
        return draft
    review = state(case).get("final_audit", {}).get("review")
    draft = FinalDeliveryDraft(
        case_id=case.case_id, base_revision=state(case)["revision"], base_digest=base_digest(case),
        arguments=case.mechanism_support_arguments, audit=review,
        reviewed_support_ids=[a.support_id for a in case.mechanism_support_arguments] if review else [],
    )
    # Historical audits lack a content binding. Keep their feedback, but require
    # a content-bound audit before committing a new-format delivery.
    expected = audit_digest(case, draft.arguments)
    if (draft.audit and audit_matches(case, draft.arguments, draft.audit)
            and state(case).get("final_audit", {}).get("content_digest") == expected):
        draft.audit_digest = expected
    return draft


def candidate_case(case, draft):
    return case.touch(mechanism_support_arguments=draft.arguments, photophysics_review=draft.audit)


def preserved_action(case, draft):
    decision = PlannerDecision.model_validate(state(case)["decision"])
    action = QuestionAction.model_validate(decision.raw_response["question_action"])
    return action.model_copy(update={"final_answer_draft": draft.final_answer_draft or action.final_answer_draft})


def manifest(case, draft):
    missing = {item.label: item.reason for item in draft.no_evidence}
    rows, issues = [], []
    for h in case.portfolio.sorted_hypotheses()[:3]:
        ids = [a.support_id for a in draft.arguments if a.target_label == h.name]
        if ids:
            rows.append(SubmittedEvidence(label=h.name, status="submitted", support_ids=ids))
        elif h.name in missing:
            rows.append(NoSubmittedEvidence(label=h.name, status="no_evidence", reason=missing[h.name]))
        else:
            issues.append({"code": "missing_argument", "label": h.name,
                           "message": f"{h.name}: submit an evidence argument, or explicitly explain no_evidence."})
    return rows, issues


def preflight(case, draft, registry):
    rows, issues = manifest(case, draft)
    if len(rows) == 3:
        validate_final_evidence(case.portfolio, draft.arguments, rows)
    if not (draft.final_answer_draft or "").strip():
        issues.append({"code": "missing_final_text", "message": "Write a concise final_answer_draft consistent with the accepted ranking and submitted arguments."})
    try:
        validate_action(preserved_action(case, draft), replace_state(candidate_case(case, draft), phase="act"), registry)
    except ValueError as exc:
        issues.extend(getattr(exc, "issues", [{"code": "stale_stop_decision", "message": str(exc)}]))
    if draft.audit:
        missing = set(draft.reviewed_support_ids) - draft.audit_responses.keys()
        issues.extend({"code": "audit_response", "support_id": sid,
                       "message": f"Respond to the review of {sid}, including if it was withdrawn."} for sid in sorted(missing))
    return issues


def apply_patch(case, draft, patch):
    """Validate all independent record errors before accepting an atomic edit batch."""
    issues = []

    def check(ok, code, message):
        if not ok:
            issues.append({"code": code, "message": message})

    existing = {a.support_id: a for a in draft.arguments}
    ids = [a.support_id for a in patch.arguments]
    withdrawn = patch.withdrawn_support_ids
    check(len(ids) == len(set(ids)), "duplicate_argument", "Argument IDs must be distinct.")
    check(len(withdrawn) == len(set(withdrawn)) and set(withdrawn) <= existing.keys(),
          "unknown_withdrawal", "Withdraw distinct IDs from the current draft only.")
    check(not set(ids) & set(withdrawn), "conflicting_edit", "Do not upsert and withdraw the same argument.")
    scientific = ({e.evidence_id for e in case.evidence_ledger.items} | {"input:smiles"}) - operational_evidence_refs(case)
    for a in patch.arguments:
        check(a.target_label in MECHANISM_POOL, "unknown_label", f"{a.support_id}: unknown mechanism label.")
        check(a.support_id not in existing or a.target_label == existing[a.support_id].target_label,
              "cross_label_id", f"{a.support_id}: cannot reassign another label's argument; write a distinct argument.")
        check(set(a.observation_refs) <= scientific, "invalid_observation",
              f"{a.support_id}: unavailable or operational-only observation refs {sorted(set(a.observation_refs) - scientific)}.")
    updated = {sid: a for sid, a in existing.items() if sid not in withdrawn}
    for a in patch.arguments:
        old = existing.get(a.support_id)
        updated[a.support_id] = (old if old and argument_content([old]) == argument_content([a])
                                else a.model_copy(update={"audit_status": "not_reviewed", "audit_notes": []}))
    empty = {item.label: item for item in draft.no_evidence}
    labels = [item.label for item in patch.no_evidence]
    check(len(labels) == len(set(labels)), "duplicate_no_evidence", "Use distinct no_evidence labels.")
    top3 = {h.name for h in case.portfolio.sorted_hypotheses()[:3]}
    for item in patch.no_evidence:
        check(item.label in top3, "nonfinal_label", "no_evidence must name a final Top-3 label.")
        check(not any(a.target_label == item.label for a in updated.values()), "false_no_evidence",
              f"{item.label}: withdraw current arguments explicitly before declaring no_evidence.")
        empty[item.label] = item
    empty = {label: item for label, item in empty.items() if not any(a.target_label == label for a in updated.values())}
    check(set(patch.audit_responses) <= set(draft.reviewed_support_ids), "unknown_audit_response",
          "Respond only to IDs in the supplied audit, not new unaudited arguments.")
    check(all(text.strip() for text in patch.audit_responses.values()), "empty_audit_response", "Audit responses cannot be blank.")
    for name in ("final_answer_draft", "reconsideration_reason"):
        value = getattr(patch, name)
        check(value is None or bool(value.strip()), "blank_text", f"{name}: use null or nonblank text.")
    if issues:
        raise RequestValidationError(issues)
    return draft.model_copy(update={
        "arguments": list(updated.values()), "no_evidence": list(empty.values()),
        "audit_responses": {**draft.audit_responses, **patch.audit_responses},
        "final_answer_draft": patch.final_answer_draft or draft.final_answer_draft,
        "reconsideration_reason": patch.reconsideration_reason,
        "generation": draft.generation + 1,
    })


def advance(o, case, store):
    """One persisted phase per call; tools are never dispatched from delivery."""
    from mechcal.agents.final_delivery import complete_delivery
    from mechcal.runtime.question_loop import checkpoint, _request_events

    draft = open_draft(case)

    def save(value, event):
        nonlocal case, draft
        draft = value
        case = checkpoint(replace_state(case, final_delivery=value.model_dump(mode="json")), store)
        store.append_jsonl("history/final_delivery.jsonl", {
            "event": event, "case_id": case.case_id, "round_id": case.current_round_id,
            "stage": value.stage, "generation": value.generation,
            "planner_attempts": value.planner_attempts, "audit_attempts": value.audit_attempts,
            "scientific_evidence": False,
        })

    save(draft, "opened")
    if draft.reconsideration_reason:
        raise FinalDeliveryIncomplete(["Planner requests scientific reconsideration: " + draft.reconsideration_reason])
    if draft.stage in {"edit", "respond"}:
        store.activity("Planner complete final evidence", case.current_round_id,
                       f"Draft {draft.generation}; accepted ranking and stopping assessment retained")

        def reserve_attempt():
            if draft.planner_attempts >= MAX_PLANNER_ATTEMPTS:
                raise FinalDeliveryIncomplete(["Final-delivery correction allowance exhausted; accepted draft preserved."])
            save(draft.model_copy(update={"planner_attempts": draft.planner_attempts + 1}), "planner_attempt")

        patched = complete_delivery(o.deps.planner, case, draft, store=store,
                                    reserve_attempt=reserve_attempt, event_sink=_request_events(case, store))
        # Attempt counters advance in the checkpoint while the model sees a stable snapshot.
        patched = patched.model_copy(update={"planner_attempts": draft.planner_attempts})
        next_stage = "audit" if o.config.effective_support_auditor and patched.audit_digest != audit_digest(case, patched.arguments) else "commit"
        save(patched.model_copy(update={"stage": next_stage}), "patch_accepted")
    elif draft.stage == "audit":
        if draft.audit_attempts >= MAX_AUDIT_ATTEMPTS:
            raise FinalDeliveryIncomplete(["Final-delivery audit allowance exhausted; draft preserved."])
        save(draft.model_copy(update={"audit_attempts": draft.audit_attempts + 1}), "audit_attempt")
        # Audit the draft, not the old accepted arguments. Never reuse a stale review.
        audited = o._audit_current_round(candidate_case(case, draft).touch(photophysics_review=None), store, persist=False)
        review = audited.photophysics_review
        if review is None:
            raise FinalDeliveryIncomplete(["Final evidence audit unavailable; no unaudited final output was committed."])
        if not audit_matches(case, draft.arguments, review):
            raise FinalDeliveryIncomplete(["Final audit does not cover the current draft."])
        store.write_json(f"final_delivery/audit_{draft.audit_attempts:02d}.json", review)
        save(draft.model_copy(update={
            "stage": "respond", "audit": review, "audit_digest": audit_digest(case, draft.arguments),
            "audit_responses": {}, "reviewed_support_ids": [a.support_id for a in draft.arguments],
        }), "audit_received")
    else:
        issues = preflight(case, draft, o.deps.capability_registry)
        if issues:
            if any(item["code"] not in {"missing_argument", "missing_final_text", "audit_response"} for item in issues):
                raise FinalDeliveryIncomplete(["Accepted stopping decision needs reconsideration: "
                                               + item["message"] for item in issues])
            save(draft.model_copy(update={"stage": "edit"}), "needs_completion")
            if draft.planner_attempts >= MAX_PLANNER_ATTEMPTS:
                raise FinalDeliveryIncomplete([item["message"] for item in issues])
            return case
        if o.config.effective_support_auditor and draft.audit_digest != audit_digest(case, draft.arguments):
            save(draft.model_copy(update={"stage": "audit"}), "audit_invalidated")
            return case
        updated = replace_state(candidate_case(case, draft), phase="act")
        rows, _ = manifest(case, draft)
        validate_final_evidence(updated.portfolio, updated.mechanism_support_arguments, rows)
        decision = decision_from_action(preserved_action(case, draft), updated, case.current_round_id)
        gate = o.deps.gate.apply(decision, updated)
        if gate.status not in {"finalize", "stop"}:
            raise FinalDeliveryIncomplete(["Final gate rejected preserved decision: " + "; ".join(gate.violations)])
        updated = updated.touch(runtime={**updated.runtime, "final_evidence_submission": [r.model_dump(mode="json") for r in rows]})
        updated = replace_state(updated, phase="finish", final_revision_complete=True,
                                decision=decision.model_dump(mode="json"), gate=gate.model_dump(mode="json"),
                                final_audit={"status": "reviewed" if draft.audit else "disabled",
                                             "review": draft.audit.model_dump(mode="json") if draft.audit else None},
                                final_delivery_receipt={"base_digest": draft.base_digest, "generation": draft.generation,
                                                        "audit_digest": draft.audit_digest, "manifest": updated.runtime["final_evidence_submission"]})
        # This checkpoint is the authoritative atomic commit; projection files can be regenerated.
        case = checkpoint(updated, store)
        store.write_json(f"rounds/{case.current_round_id}/planner_final_decision.json", decision)
        store.write_json(f"rounds/{case.current_round_id}/final_delivery.json", draft)
    return case
