"""Reuse exact audit bases and route explicit Auditor decisions, not inferred approval."""

import hashlib
import json

from mechcal.runtime.retry import ModelContractError
from mechcal.runtime.questions import replace_state, state
from mechcal.schemas.mechanisms import AuditRouting, MechanismSupportAudit, PhotophysicsReview


def validate_routing(review, arguments):
    routing = review.routing
    if routing is None:
        raise ModelContractError("Provide explicit routing: no_change, local_audit, or global.")
    ids = {a.support_id for a in arguments}
    affected = routing.affected_support_ids
    if len(affected) != len(set(affected)) or not set(affected) <= ids:
        raise ModelContractError("Routing must name distinct supplied support IDs.")
    audits = {a.support_id: a for a in review.support_argument_audits}
    if routing.mode == "no_change":
        conflicts = []
        if affected:
            conflicts.append("routing.affected_support_ids must be empty")
        if review.overclaim_warnings:
            conflicts.append("overclaim_warnings must be empty for no_change. Put already acknowledged "
                             "limitations and confirmations in audit_notes or routing.reason; keep actual "
                             "unresolved concerns here and choose the appropriate response routing")
        if review.recommended_next_routes or any(a.recommended_routes for a in review.coverage_axes):
            conflicts.append("recommended_next_routes and coverage_axes[].recommended_routes require global routing")
        if set(audits) != ids:
            conflicts.append(f"support_argument_audits must cover exactly {sorted(ids)}")
        for sid, audit in audits.items():
            if audit.audit_status != "accepted" or audit.boundary_patch:
                conflicts.append(f"{sid}: audit_status must be accepted and boundary_patch must be empty")
        if conflicts:
            raise ModelContractError("no_change requires complete accepted audits with no requested corrections or routes.\n"
                                     + "\n".join(conflicts))
    if routing.mode == "local_audit" and not affected:
        raise ModelContractError("local_audit must identify the arguments to repair.")
    for sid, audit in audits.items():
        assessment = audit.argument_assessment
        serious = assessment and (
            assessment.finding_status == "contradicted"
            or assessment.mechanism_link_status == "unsound"
            or assessment.boundary_effect == "counterevidence"
        )
        if serious and routing.mode != "global":
            raise ModelContractError("Counterevidence or unsound mechanism links require global reconsideration.")
        if routing.mode == "no_change" and assessment and (
            assessment.finding_status in {"unverified", "contradicted"}
            or assessment.mechanism_link_status != "articulated"
        ):
            raise ModelContractError("Unverified findings or missing mechanism links need an explicit Planner response.")
        needs_change = audit.audit_status != "accepted" or bool(audit.boundary_patch)
        if needs_change and routing.mode == "local_audit" and sid not in affected:
            raise ModelContractError("Every argument requiring correction must be included in local routing.")
    if routing.mode == "local_audit" and (review.recommended_next_routes
                                         or any(a.recommended_routes for a in review.coverage_axes)):
        raise ModelContractError("New investigation routes require global reconsideration.")


def audit_basis(case, argument):
    if case.runtime.get("scoped_evidence_updates"):
        from mechcal.runtime.scoped_updates import scoped_audit_basis
        return scoped_audit_basis(case, argument)
    # New observations (including possible counterevidence) and any portfolio
    # change invalidate all entries. Only local argument edits permit partial reuse.
    basis = {
        "version": 1, "case_id": case.case_id, "smiles": case.input.smiles,
        "argument": argument.model_dump(mode="json"),
        "portfolio": case.portfolio.model_dump(mode="json"),
        "observations": [e.model_dump(mode="json") for e in case.evidence_ledger.items],
        "comparison": state(case).get("assimilation", {}).get("ranking_update"),
    }
    return hashlib.sha256(json.dumps(basis, sort_keys=True).encode()).hexdigest()


def audit_incrementally(o, case, store):
    from mechcal.runtime.scoped_updates import comparison_records, current_receipt, digest, new_batch_allows_reuse, observation_versions

    previous = state(case).get("incremental_audit_cache", {})
    scoped_mode = case.runtime.get("scoped_evidence_updates", False)
    comparisons = comparison_records(case)
    old_comparisons = state(case).get("incremental_audit_comparisons", {})
    comparison_versions = {key: {"comparison": digest(row), "observations": observation_versions(case),
                                 "case_id": case.case_id, "smiles": case.input.smiles}
                           for key, row in comparisons.items()} if scoped_mode else {}
    changed_comparisons = {}
    for key, entry in comparison_versions.items():
        previous_entry = old_comparisons.get(key)
        row = comparisons[key]
        labels = [row[f] for f in ("leader", "challenger", "included_label", "excluded_label") if f in row]
        if (not isinstance(previous_entry, dict)
                or any(previous_entry.get(k) != entry[k] for k in ("comparison", "case_id", "smiles"))
                or not new_batch_allows_reuse(case, previous_entry, comparison_id=key, labels=labels)):
            changed_comparisons[key] = row
    arguments = {a.support_id: a for a in case.mechanism_support_arguments}
    bases = {a.support_id: audit_basis(case, a) for a in case.mechanism_support_arguments}
    reused = {
        sid: entry for sid, entry in previous.items()
        if sid in bases and entry.get("basis") == bases[sid]
        and entry.get("routing") == "no_change"
        and (not scoped_mode or new_batch_allows_reuse(
            case, entry, support_id=sid, labels=[arguments[sid].target_label]))
    }
    dirty = [a for a in case.mechanism_support_arguments if a.support_id not in reused]
    receipt = current_receipt(case) if scoped_mode else None
    audit_scope = None
    if scoped_mode:
        labels = {a.target_label for a in dirty}
        selected_comparisons = {key: row for key, row in comparisons.items() if key in changed_comparisons
                                or labels.intersection(row.get(f) for f in
                                    ("leader", "challenger", "included_label", "excluded_label"))}
        comparison_labels = {row[field] for row in selected_comparisons.values()
                             for field in ("leader", "challenger", "included_label", "excluded_label") if field in row}
        # Comparison context does not reopen an unchanged support-evidence audit.
        comparison_support = [a for a in case.mechanism_support_arguments
                              if a.support_id in reused and a.target_label in comparison_labels]
        audit_scope = {
            "reviewed_support_ids": [a.support_id for a in dirty], "reused_support_ids": list(reused),
            "impact_snapshot_id": receipt["review"]["snapshot_id"] if receipt else None,
            "comparisons": selected_comparisons,
            "comparison_support_arguments": [a.model_dump(mode="json") for a in comparison_support],
            "required_evidence_refs": sorted(
                {ref for row in selected_comparisons.values() for ref in row.get("evidence_refs", [])}
                | {ref for a in comparison_support for ref in a.observation_refs}
                | set(receipt["batch_evidence_ids"] if receipt else observation_versions(case))),
            "notice": "Reuse is prior accepted review, not a fresh approval. New evidence impact was checked "
                      "before source-independent claims could be reused. comparison_support_arguments are "
                      "context only; audit comparisons without fresh support audits for those IDs. "
                      "Escalate new concerns through global routing. Final audit remains complete.",
        }
    if dirty or changed_comparisons or not case.mechanism_support_arguments:
        scoped = case.touch(mechanism_support_arguments=dirty)
        if audit_scope:
            scoped = scoped.touch(runtime={**scoped.runtime, "support_audit_scope": audit_scope})
        updated = o._audit_current_round(case, store, audit_case=scoped)
        review = updated.photophysics_review
    else:
        review = PhotophysicsReview(
            review_id=f"{case.case_id}:{case.current_round_id}:reused",
            case_id=case.case_id, round_id=case.current_round_id,
            routing=AuditRouting(mode="no_change", reason="Exact accepted audit bases are unchanged."),
        )
        updated = case.touch(photophysics_review=review)
    if review is None:
        return updated  # Unavailable is never cached as approval.
    all_audits = {sid: MechanismSupportAudit.model_validate(entry["audit"]) for sid, entry in reused.items()}
    all_audits.update({a.support_id: a for a in review.support_argument_audits})
    merged = review.model_copy(update={
        "support_argument_audits": [all_audits[a.support_id] for a in case.mechanism_support_arguments
                                   if a.support_id in all_audits],
    })
    cache = {sid: {**entry, **({"observations": observation_versions(case),
                               "impact_snapshot_id": audit_scope["impact_snapshot_id"]} if scoped_mode else {})}
             for sid, entry in reused.items()}
    if scoped_mode and review.routing and review.routing.mode == "global":
        cache = {}  # A comparison can raise a new concern about context-only support.
    if review.routing and review.routing.mode == "no_change":
        for audit in review.support_argument_audits:
            cache[audit.support_id] = {
                "basis": bases[audit.support_id], "audit": audit.model_dump(mode="json"),
                "routing": "no_change", "source_review_id": review.review_id,
                **({"observations": observation_versions(case),
                    "impact_snapshot_id": audit_scope["impact_snapshot_id"]} if scoped_mode else {}),
            }
    updated = replace_state(updated.touch(photophysics_review=merged), incremental_audit_cache=cache)
    if scoped_mode:
        updated = replace_state(updated, incremental_audit_comparisons=(
            comparison_versions if review.routing and review.routing.mode == "no_change" else {}))
    store.append_jsonl("history/incremental_audits.jsonl", {
        "round_id": case.current_round_id, "reviewed_support_ids": [a.support_id for a in dirty],
        "reused_support_ids": list(reused), "bases": bases,
        "routing": merged.routing.model_dump(mode="json") if merged.routing else None,
        **({"scope": audit_scope, "reused_from": {sid: entry.get("source_review_id") for sid, entry in reused.items()}}
           if scoped_mode else {}),
    })
    store.write_photophysics_review(merged)
    store.write_case_run(updated)
    return updated


def route_feedback(case, feedback):
    review = case.photophysics_review
    routing = review.routing if review is not None else None
    if routing is None:
        return replace_state(case, phase="assimilate", argument_review_pending=True,
                             argument_review=feedback, argument_review_round=case.current_round_id)
    if routing.mode == "no_change":
        return replace_state(case, phase="act", argument_review_pending=False,
                             argument_review=feedback, argument_review_round=case.current_round_id)
    if routing.mode == "local_audit":
        feedback = {**feedback, "support_ids": routing.affected_support_ids,
                    "routing": routing.model_dump(mode="json")}
    return replace_state(case, phase="assimilate", argument_review_pending=True,
                         argument_review=feedback, argument_review_round=case.current_round_id)
