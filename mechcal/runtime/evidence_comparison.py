"""Validate provenance and preserve judgment history without choosing mechanisms."""

from copy import deepcopy

from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime.result_contract import operational_evidence_refs
from mechcal.runtime.retry import RequestValidationError
from mechcal.schemas.evidence_comparison import ArgumentChange
from mechcal.runtime.hypothesis_tests import preserve_tests, validate_comparison, validate_tests


ARGUMENT_FIELDS = ("target_label", "support_level", "finding", "warrant", "boundary", "observation_refs")


def enabled(case):
    return case.runtime.get("comparative_evidence", False)


def order(portfolio):
    return [h.name for h in portfolio.sorted_hypotheses()]


def argument_content(argument):
    return {key: getattr(argument, key) for key in ARGUMENT_FIELDS}


def changed_arguments(case, update):
    current = {a.support_id: a for a in update.mechanism_support_arguments}
    return {a.support_id: a for a in case.mechanism_support_arguments
            if a.support_id not in current or argument_content(a) != argument_content(current[a.support_id])}


def _require(condition, path, message, **details):
    if not condition:
        raise RequestValidationError([{
            "path": path, "code": "comparison_contract", "message": message, **details,
        }])


def argument_change_issues(case, update):
    """Bookkeeping failures always retain a scoped, typed correction path."""
    refs = {e.evidence_id for e in case.evidence_ledger.items} | {"input:smiles"}
    scientific_refs = refs - operational_evidence_refs(case)
    current = {a.support_id: a for a in update.mechanism_support_arguments}
    previous = {a.support_id: a for a in case.mechanism_support_arguments}
    changed = changed_arguments(case, update)
    issues = []

    def issue(message, *, support_id=None):
        issues.append({
            "path": "argument_changes", "code": "argument_change_records",
            "repair_paths": [["argument_changes"]], "message": message,
            "support_id": support_id,
            "changes": [{"support_id": sid, "before": argument_content(old),
                         "after": argument_content(current[sid]) if sid in current else None}
                        for sid, old in changed.items()],
            "allowed_previous_ids": sorted(previous), "current_argument_ids": sorted(current),
            "allowed_refs": sorted(scientific_refs),
            "retained_previous_ids": sorted(previous.keys() & current.keys()),
            "removed_previous_ids": sorted(previous.keys() - current.keys()),
        })

    ids = [c.support_id for c in update.argument_changes]
    if len(ids) != len(set(ids)) or not set(changed) <= set(ids) <= previous.keys():
        issue(f"Account once for each revised or removed argument: {sorted(changed)}. "
              f"Missing records: {sorted(set(changed) - set(ids))}; "
              f"unknown previous IDs: {sorted(set(ids) - previous.keys())}. "
              "Correct only the change records, not the submitted arguments or ranking.")
    for change in update.argument_changes:
        if change.support_id not in previous:
            continue
        if not change.reason.strip():
            issue("Explain the correction, withdrawal or replacement.", support_id=change.support_id)
        if not set(change.evidence_refs) <= scientific_refs:
            issue("Cite existing scientific observations only.", support_id=change.support_id)
        if change.disposition == "revised":
            if change.support_id not in current or change.replacement_ids not in ([], [change.support_id]):
                issue("Revised arguments retain their ID: replacement_ids may be empty or contain only that ID. "
                      "A newly added argument is not a replacement for an argument that remains current. "
                      "Use superseded only when the old ID is absent from current_argument_ids.",
                      support_id=change.support_id)
        else:
            if change.support_id in current:
                issue("Withdrawn/superseded IDs cannot remain current. Use revised for retained IDs.",
                      support_id=change.support_id)
            replacements = change.replacement_ids
            if len(replacements) != len(set(replacements)) or not set(replacements) <= current.keys():
                issue("Replacement IDs must be distinct current arguments.", support_id=change.support_id)
            if bool(replacements) != (change.disposition == "superseded"):
                issue("Superseded requires replacement IDs; withdrawn has none.", support_id=change.support_id)
    return issues


def validate_update(case, update, *, check_comparisons=True):
    if not enabled(case):
        return
    issues = argument_change_issues(case, update)
    if issues:
        raise RequestValidationError(issues)
    s = case.runtime.get("question_loop", {})
    refs = {e.evidence_id for e in case.evidence_ledger.items} | {"input:smiles"}
    scientific_refs = refs - operational_evidence_refs(case)
    current = {a.support_id: a for a in update.mechanism_support_arguments}
    previous = {a.support_id: a for a in case.mechanism_support_arguments}
    for change in update.argument_changes:
        if change.disposition == "revised":
            _require(current[change.support_id].target_label == previous[change.support_id].target_label,
                     f"argument_changes[{change.support_id}]",
                     "A support ID cannot silently change its target mechanism; supersede it explicitly.")

    ranking = update.ranking_update
    _require(ranking is not None, ["ranking_update"], "Explain the comparison even when the order is unchanged.")
    _require(bool(ranking.explanation.strip()), ["ranking_update", "explanation"], "Supply a comparison, not an empty reason.")
    _require(set(ranking.alternatives_considered) <= set(MECHANISM_POOL), ["ranking_update", "alternatives_considered"],
             "Use mechanism-pool labels; alternatives need not remain in the portfolio.",
             allowed_values=list(MECHANISM_POOL), requires_scientific_context=False)
    _require(set(ranking.evidence_refs) <= scientific_refs, ["ranking_update", "evidence_refs"],
             "Tool failures and reviews are not scientific support or counterevidence.")
    initial = "initial_judgment" not in s
    _require((ranking.basis == "initial") == initial, ["ranking_update", "basis"],
             "Use initial only for the first recorded judgment; later updates compare with the saved judgment.")
    before, after = order(case.portfolio), order(update.portfolio)
    common = set(before) & set(after)
    order_changed = before[:3] != after[:3] or (
        [label for label in before if label in common] != [label for label in after if label in common]
    )
    if not initial and order_changed:
        _require(ranking.basis != "unchanged", ["ranking_update", "basis"],
                 "Order or candidate membership changed: explain new evidence or reinterpretation. "
                 "Relative improvement of alternatives may justify a demotion without direct counterevidence.")
    if not initial:
        _require((set(before) ^ set(after)) <= set(ranking.alternatives_considered),
                 ["ranking_update", "alternatives_considered"], "Acknowledge added or removed candidate labels. "
                 "Tail-only coverage changes may leave the existing ranking unchanged.")
    if ranking.basis == "new_evidence":
        _require(bool(ranking.evidence_refs), ["ranking_update", "evidence_refs"], "Identify the observations used.")

    issues = mechanism_effect_issues(case, update, scientific_refs)
    if issues:
        raise RequestValidationError(issues)
    validate_comparison(case, update, check_leading=check_comparisons)
    from mechcal.runtime.boundary_decisions import validate_ranking
    validate_ranking(case, update, check_comparisons=check_comparisons)


def mechanism_effect_issues(case, update, scientific_refs):
    pending = {r["result_id"]: r for r in case.runtime.get("question_loop", {}).get("pending_results", [])}
    issues = []

    def require(condition, path, message, **details):
        if not condition:
            issues.append({"path": path, "code": "comparison_contract", "message": message, **details})

    for disposition_index, disposition in enumerate(update.dispositions):
        result = pending[disposition.result_id]
        plan = result.get("investigation_plan") or {}
        expected_labels = set(plan.get("candidate_labels", []))
        effects = disposition.mechanism_effects
        labels = [e.label for e in effects]
        path = ["dispositions", disposition_index, "mechanism_effects"]
        require(bool(labels) and len(labels) == len(set(labels)) and set(labels) <= set(MECHANISM_POOL),
                 path, "Give one effect per considered mechanism, with valid distinct labels.")
        require(expected_labels <= set(labels), path, "Address every mechanism in this execution's investigation plan.",
                result_id=disposition.result_id, required_labels=sorted(expected_labels),
                missing_labels=sorted(expected_labels - set(labels)))
        for effect in effects:
            require(bool(effect.reason.strip()) and set(effect.evidence_refs) <= set(result["evidence_refs"]),
                     path, "Explain the effect using this execution's observations only.")
            if effect.effect in {"supports", "weakens"}:
                require(disposition.outcome != "no_result" and bool(effect.evidence_refs)
                         and set(effect.evidence_refs) <= scientific_refs, path,
                         "Support/weakening requires a delivered scientific observation, not unavailable output. "
                         "Valid partial observations may still support or weaken a hypothesis.")
    return issues


def validate_plans(case, action):
    if not enabled(case):
        return
    issues = []

    def check(condition, path, message):
        if not condition:
            issues.append({"path": path, "repair_paths": [path],
                           "code": "comparison_contract", "message": message})

    requests = {r.dispatch_id: r for r in action.dispatch_requests}
    ids = [p.dispatch_id for p in action.investigation_plans]
    check(len(ids) == len(set(ids)) and set(ids) == set(requests), ["investigation_plans"],
          "Supply exactly one investigation plan per dispatch; no plans for finalize/stop.")
    for index, plan in enumerate(action.investigation_plans):
        path = ["investigation_plans", index]
        check(len(plan.candidate_labels) == len(set(plan.candidate_labels))
              and set(plan.candidate_labels) <= set(MECHANISM_POOL), [*path, "candidate_labels"],
              "Use distinct mechanism-pool labels.")
        for field in ("decision_question", "usefulness"):
            check(bool(getattr(plan, field).strip()), [*path, field],
                  "Explain the scientific question and expected use, including prerequisites or exploration.")
        for outcome_index, outcome in enumerate(plan.possible_outcomes):
            for field in ("observation", "judgment_effect"):
                check(bool(getattr(outcome, field).strip()), [*path, "possible_outcomes", outcome_index, field],
                      "Each possible observation needs an expected effect on judgment.")
        if plan.purpose == "discrimination":
            check(len(plan.candidate_labels) >= 2 and len(plan.possible_outcomes) >= 2, path,
                  "A comparison needs at least two candidates and two possible outcomes. "
                  "Use preparation/exploration/recovery when appropriate; no forced either/or mechanism.")
            check(len({o.observation.strip() for o in plan.possible_outcomes}) == len(plan.possible_outcomes),
                  [*path, "possible_outcomes"], "Do not duplicate an outcome to fill the comparison.")
    try:
        validate_tests(case, action)
    except RequestValidationError as exc:
        issues.extend(exc.issues)
    if issues:
        raise RequestValidationError(issues)


def preserve_update(case, update):
    """Called only after validation, before replacing the current portfolio."""
    if not enabled(case):
        return {}
    s = case.runtime.get("question_loop", {})
    snapshot = {
        "case_id": case.case_id, "revision": update.revision,
        "portfolio": update.portfolio.model_dump(mode="json"),
        "mechanism_support_arguments": [a.model_dump(mode="json") for a in update.mechanism_support_arguments],
        "decision_assessment": update.decision_assessment.model_dump(mode="json"),
    }
    changed = changed_arguments(case, update)
    previous = {a.support_id: a for a in case.mechanism_support_arguments}
    edits = [{"revision": update.revision, "previous_argument": previous[c.support_id].model_dump(mode="json"),
              "observed_change": c.support_id in changed,
              **c.model_dump(mode="json")} for c in update.argument_changes]
    if case.runtime.get("decision_economy"):
        current = {a.support_id: a for a in update.mechanism_support_arguments}
        for edit in edits:
            edit["provenance"] = "program_diff"
            item = current.get(edit["support_id"])
            edit["current_argument"] = item.model_dump(mode="json") if item else None
        edits.extend({"revision": update.revision, "support_id": sid, "disposition": "added",
                      "provenance": "program_diff", "previous_argument": None,
                      "current_argument": item.model_dump(mode="json"), "observed_change": True,
                      "reason": "Program-recorded new argument.", "replacement_ids": [], "evidence_refs": []}
                     for sid, item in current.items() if sid not in previous)
    return {
        **preserve_tests(case, update),
        "initial_judgment": deepcopy(s.get("initial_judgment", snapshot)),
        "argument_history": [*s.get("argument_history", []), *edits],
        "last_comparison": {
            "revision": update.revision, "before_order": order(case.portfolio) if "initial_judgment" in s else [],
            "after_order": order(update.portfolio), "ranking_update": update.ranking_update.model_dump(mode="json"),
            "added_candidates": sorted(set(order(update.portfolio)) - set(order(case.portfolio))),
            "removed_candidates": sorted(set(order(case.portfolio)) - set(order(update.portfolio))),
            "argument_changes": edits,
        },
    }


def comparison_context(case):
    if not enabled(case):
        return {"enabled": False}
    s = case.runtime.get("question_loop", {})
    latest = deepcopy(s.get("last_comparison"))
    snapshots = []
    if latest:
        edits = latest.get("argument_changes", [])
        snapshots = [{"support_id": e["support_id"], "before": e["previous_argument"],
                      "observed_change": e.get("observed_change", True)} for e in edits]
        latest["argument_changes"] = [
            {k: v for k, v in e.items() if k in ArgumentChange.model_fields or k == "provenance"}
            for e in edits]
    return {"enabled": True, "initial_judgment": s.get("initial_judgment"),
            "latest_update": latest,
            "read_only_audit": {"argument_snapshots": snapshots,
                                "notice": "Program-generated history, never fields of an ArgumentChange."},
            "current_order": order(case.portfolio),
            "changed_argument_fields": list(ARGUMENT_FIELDS)}


def comparison_refs(case):
    if not enabled(case):
        return set()
    context = comparison_context(case)
    initial = context.get("initial_judgment") or {}
    latest = case.runtime.get("question_loop", {}).get("last_comparison") or {}
    refs = {ref for a in initial.get("mechanism_support_arguments", []) for ref in a["observation_refs"]}
    refs.update(ref for edit in latest.get("argument_changes", [])
                for ref in (edit.get("previous_argument") or {}).get("observation_refs", []))
    refs.update(latest.get("ranking_update", {}).get("evidence_refs", []))
    ranking = latest.get("ranking_update", {})
    for basis in ranking.get("candidate_bases", []):
        refs.update(basis["positive_refs"])
        refs.update(basis["counter_refs"])
    for comparison in ranking.get("boundary_comparisons", []):
        refs.update(comparison["evidence_refs"])
    if case.runtime.get("hypothesis_tests"):
        comparison = latest.get("ranking_update", {}).get("leading_comparison") or {}
        refs.update(comparison.get("evidence_refs", []))
        for item in case.runtime.get("question_loop", {}).get("hypothesis_test_history", [])[-6:]:
            refs.update(item["assessment"]["evidence_refs"])
    return refs
