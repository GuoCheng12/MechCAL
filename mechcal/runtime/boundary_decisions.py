"""Check decision completeness and feasibility, never choose scientific labels."""

from mechcal.capabilities.discovery import amesp_enabled
from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime.result_contract import operational_evidence_refs
from mechcal.runtime.retry import RequestValidationError


def require(condition, path, message):
    if not condition:
        raise RequestValidationError([{
            "path": ".".join(map(str, path)), "code": "boundary_decision_contract",
            "repair_paths": [path], "message": message,
        }])


def required_pairs(previous, current):
    """Compare the cutoff to its nearest challenger and displaced Top-3 members."""
    ranked = [h.name for h in current.sorted_hypotheses()]
    if len(ranked) < 4:
        return []
    challengers = [ranked[3]]
    challengers.extend(h.name for h in previous.sorted_hypotheses()[:3] if h.name in ranked[3:])
    return [(ranked[2], label) for label in dict.fromkeys(challengers)]


def validate_ranking(case, update, *, check_comparisons=True):
    if not case.runtime.get("evidence_decision_contracts"):
        return
    issues = []

    def check(condition, path, message):
        if not condition:
            issues.append({"path": path, "repair_paths": [path],
                           "code": "boundary_decision_contract", "message": message})

    ranked = [h.name for h in update.portfolio.sorted_hypotheses()]
    complete = len(ranked) == len(MECHANISM_POOL) and set(ranked) == set(MECHANISM_POOL)
    check(complete, ["portfolio"],
          "Keep the complete shared pool in this mode so an excluded challenger cannot disappear. "
          f"Missing labels: {sorted(set(MECHANISM_POOL) - set(ranked))}. "
          "Each new label needs explicit confidence, differential_priority and rationale, not only a name.")
    ranking = update.ranking_update
    check(ranking is not None, ["ranking_update"], "Record bases and cutoff comparisons for this judgment.")
    if ranking is None:
        raise RequestValidationError(issues)
    path = ["ranking_update", "candidate_bases"]
    labels = [b.label for b in ranking.candidate_bases]
    check(len(labels) == len(MECHANISM_POOL) and set(labels) == set(MECHANISM_POOL), path,
          "Provide one candidate basis per pool label, separating prior rationale from observations and inference gaps. "
          f"Missing labels: {sorted(set(MECHANISM_POOL) - set(labels))}.")
    scientific = ({e.evidence_id for e in case.evidence_ledger.items} | {"input:smiles"}) - operational_evidence_refs(case)
    for index, basis in enumerate(ranking.candidate_bases):
        for field in ("positive_refs", "counter_refs"):
            refs = getattr(basis, field)
            check(len(refs) == len(set(refs)) and set(refs) <= scientific, path + [index, field],
                  "Cite distinct current-case scientific observations; failures and reviews are not evidence. Empty is valid.")
    if not check_comparisons:
        if issues:
            raise RequestValidationError(issues)
        return
    path = ["ranking_update", "boundary_comparisons"]
    comparisons = ranking.boundary_comparisons
    pairs = [(c.included_label, c.excluded_label) for c in comparisons]
    expected = required_pairs(case.portfolio, update.portfolio)
    if not complete:
        check(False, path, "After completing the pool and its explicit priorities, compare the resulting "
              "rank 3 with rank 4 and any displaced previous Top-3 members. Derive positions from "
              "descending differential_priority, not list order. Prior_only or unresolved is valid.")
    elif len(pairs) != len(set(pairs)) or not set(expected) <= set(pairs):
        issues.append({
            "path": path, "repair_paths": [path], "code": "boundary_decision_contract",
            "message": f"Explicitly compare these cutoff pairs: {expected}. "
            "The numeric priorities, not array order or prose, determine the ranking. "
            "Keep these scores and correct the comparison labels and explanation; unresolved is allowed.",
            "submitted_ranking": [
                {"rank": index + 1, "label": hypothesis.name,
                 "differential_priority": hypothesis.differential_priority}
                for index, hypothesis in enumerate(update.portfolio.sorted_hypotheses())
            ],
            "required_pairs": [{"included_label": a, "excluded_label": b} for a, b in expected],
            "submitted_pairs": [{"included_label": a, "excluded_label": b} for a, b in pairs],
        })
    for index, comparison in enumerate(comparisons):
        if complete:
            check(comparison.included_label in ranked[:3] and comparison.excluded_label in ranked[3:],
                  path + [index], "Each pair must cross the submitted Top-3 boundary.")
        check(set(comparison.evidence_refs) <= scientific, path + [index, "evidence_refs"],
              "Only existing scientific observation references are allowed.")
        if comparison.basis in {"observations", "mixed"}:
            check(bool(comparison.evidence_refs), path + [index],
                  "An observation-based comparison needs references; use prior_only/unresolved when appropriate.")
    if issues:
        raise RequestValidationError(issues)


def required_stop_routes(case):
    state = case.runtime.get("question_loop", {})
    ids = {cid for q in state.get("questions", {}).values() if q["answer_state"] != "answered"
           and not (case.runtime.get("decision_economy") and q.get("decision_role") == "limitation")
           for cid in q.get("acceptable_capability_ids", [])}
    review = case.runtime.get("stop_review_feedback") or {}
    ids.update(review.get("capability_ids", []))
    return ids


def validate_continuation(case, action, registry):
    if not case.runtime.get("evidence_decision_contracts"):
        return
    if action.action == "dispatch":
        # Use the same source-specific checks as execution, not backend_enabled.
        issues = []
        for index, request in enumerate(action.dispatch_requests):
            violations = registry.availability_violations(
                request, case.artifact_manifest, amesp_enabled=amesp_enabled(case),
            )
            if violations:
                path = ["dispatch_requests", index]
                issues.append({"path": path, "repair_paths": [path], "code": "boundary_decision_contract",
                               "message": " ".join(violations) +
                               " No calculation ran. Choose applicable preparation, another route, or a bounded ending."})
        if issues:
            raise RequestValidationError(issues)
        return
    if case.runtime.get("run_limits", {}).get("scientific_tools_allowed") is False:
        return  # A runtime-imposed limit is not voluntary scientific convergence.
    assessment = action.continuation_assessment
    path = ["continuation_assessment"]
    issues = []

    def check(condition, item_path, message):
        if not condition:
            issues.append({"path": item_path, "code": "boundary_decision_contract",
                           "repair_paths": [item_path], "message": message})

    check(assessment is not None, path,
            "Assess remaining useful computation for a bounded ranking, not whether experimental proof is obtainable.")
    if assessment is None:
        raise RequestValidationError(issues)
    ids = [r.capability_id for r in assessment.routes]
    check(len(ids) == len(set(ids)) and required_stop_routes(case) <= set(ids), path,
            f"Account for named routes from unresolved questions and stop review: {sorted(required_stop_routes(case))}.")
    check(bool(ids) or not registry.snapshot().capabilities, path,
            "Consider at least one relevant registered route before a voluntary ending; never invent one.")
    for index, route in enumerate(assessment.routes):
        item_path = path + ["routes", index]
        if registry.get(route.capability_id) is None:
            # Preserve a previously named capability gap without making it dispatchable.
            check(route.capability_id in required_stop_routes(case) and route.disposition == "blocked",
                    item_path, "An unregistered route can only be acknowledged as a previously named, blocked capability gap.")
            continue
        # This is a read-only inquiry, not a request or scientific observation.
        try:
            if route.tool_args:
                from mechcal.schemas.tool_arguments import argument_model
                spec = registry.require(route.capability_id)
                argument_model(spec.route, spec.owner_agent).model_validate(route.tool_args)
            readiness = registry.readiness(
                route.capability_id, case.artifact_manifest, amesp_enabled=amesp_enabled(case),
                source_artifact_id=route.tool_args.get("source_artifact_id"), tool_args=route.tool_args or None,
            )
        except ValueError as exc:
            check(False, item_path + ["tool_args"], f"Invalid applicability inquiry for {route.capability_id}: {exc}")
            readiness = None
        incomplete_after_audit = (
            action.action == "stop" and case.runtime.get("question_loop", {}).get("final_audit") is not None
            and action.stopping_assessment.finish_basis is None
            and action.stopping_assessment.judgment_state == "unresolved"
        )
        check(route.disposition != "useful" or incomplete_after_audit, item_path,
                "A useful route conflicts with voluntary completion. Dispatch it or reconsider its value. "
                "After the final audit, when tools cannot reopen, stop incomplete with unresolved judgment and no finish_basis.")
        if route.disposition == "blocked" and readiness is not None:
            check(readiness["status"] != "ready", item_path,
                    f"{route.capability_id} satisfies basic preconditions for the submitted inputs. "
                    "Do not call it blocked; assess usefulness or check exact inputs. "
                    "Ready means callable, not scientifically sufficient. Use not_useful with a reason if further "
                    "calculation would not help; if useful, reconsider completion. The runtime does not decide its scientific value.")
            if readiness["status"] == "missing_input":
                check(route.preparation_assessment is not None, item_path,
                        "Missing inputs alone do not justify stopping. Explain whether available preparation can obtain them.")
    if issues:
        raise RequestValidationError(issues)


def context(case):
    if not case.runtime.get("evidence_decision_contracts"):
        return {"enabled": False}
    ranking = case.runtime.get("question_loop", {}).get("assimilation", {}).get("ranking_update") or {}
    return {
        "enabled": True,
        "current_candidate_bases": ranking.get("candidate_bases", []),
        "current_boundary_comparisons": ranking.get("boundary_comparisons", []),
        "required_stop_route_ids": sorted(required_stop_routes(case)),
        "notice": "Planner judgments, not new evidence. Recompute cutoff pairs from the proposed ranking. No mandated label or confidence change.",
    }
