"""Check explicit decision-value contracts without making scientific judgments."""

from mechcal.capabilities.discovery import case_catalog
from mechcal.runtime.retry import RequestValidationError


def require(condition, path, message):
    if not condition:
        raise RequestValidationError([{"path": path, "repair_paths": [path],
                                       "code": "decision_value_contract", "message": message}])


def validate_value(value, outcomes, *, catalog, path):
    require(value is not None, path, "Supply decision_value: actual deliverable, scope, unresolved extension and novelty.")
    issues = []

    def check(condition, location, message):
        if not condition:
            issues.append({"path": location, "repair_paths": [location],
                           "code": "decision_value_contract", "message": message})

    for field in ("delivered_observable", "observation_scope", "unresolved_extension", "why_not_redundant"):
        check(bool(getattr(value, field).strip()), [*path, field], "Use a substantive description, not whitespace.")
    outcomes_path = [*path[:-1], "possible_outcomes"]
    check(bool(outcomes) and all(o.decision_effect is not None for o in outcomes), outcomes_path,
          "For each possible outcome, declare decision_effect: ranking, support, preparation or no_change.")
    check(any(o.decision_effect != "no_change" for o in outcomes), outcomes_path,
          "All proposed outcomes leave judgment unchanged. Do not dispatch this plan as useful; reconsider or finalize. "
          "Preparation is allowed when it enables a named available downstream route.")
    downstream = value.downstream_capability_id
    if any(o.decision_effect == "preparation" for o in outcomes):
        check(bool(downstream) and bool((value.downstream_question or "").strip()), path,
              "Preparation needs a downstream_capability_id and the scientific question that route can address.")
    if downstream:
        check(downstream in catalog and catalog[downstream]["status"] != "disabled", path,
              "Downstream route must exist and not be disabled; missing inputs may be acquired first.")
    if issues:
        raise RequestValidationError(issues)


def validate_action_value(case, action, registry):
    if not case.runtime.get("decision_value_checks"):
        return
    catalog = {row["capability_id"]: row for row in case_catalog(registry, case)}
    issues = []
    for index, plan in enumerate(action.investigation_plans):
        path = ["investigation_plans", index]
        try:
            validate_value(plan.decision_value, plan.possible_outcomes, catalog=catalog,
                           path=[*path, "decision_value"])
        except RequestValidationError as exc:
            issues.extend(exc.issues)
        if case.runtime.get("decision_economy"):
            from mechcal.runtime.decision_economy import validate_plan_value
            try:
                validate_plan_value(plan, path)
            except RequestValidationError as exc:
                issues.extend(exc.issues)
    if issues:
        raise RequestValidationError(issues)


def validate_stop_opportunity(review, catalog):
    opportunity = review.continuation_opportunity
    if not review.capability_ids and opportunity is None:
        return
    require(opportunity is not None, ["continuation_opportunity"],
            "Recommending another calculation needs an explicit opportunity, not just an untried tool ID.")
    require(review.capability_ids == [opportunity.capability_id], ["capability_ids"],
            "Identify one immediate capability for this opportunity; any later route belongs in downstream_capability_id.")
    require(bool(review.challenge.strip()), ["challenge"], "Explain why this opportunity matters to the proposed ending.")
    require(opportunity.capability_id in catalog and catalog[opportunity.capability_id]["status"] != "disabled",
            ["continuation_opportunity", "capability_id"], "Use a known, enabled capability.")
    validate_value(opportunity.decision_value, opportunity.possible_outcomes, catalog=catalog,
                   path=["continuation_opportunity", "decision_value"])
