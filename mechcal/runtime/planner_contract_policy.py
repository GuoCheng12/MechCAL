"""Project enabled runtime requirements into the Planner's request contract."""

from copy import deepcopy

from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.schemas.contracts import NONBLANK, object_variants
from mechcal.schemas.evidence_comparison import MATERIAL_OUTCOME_CHANGES


def require_value(schema, definition, field):
    node = schema.get("$defs", {}).get(definition, {})
    prop = node.get("properties", {}).get(field)
    if prop is None:
        return
    choices = prop.get("anyOf", [])
    non_null = [choice for choice in choices if choice.get("type") != "null"]
    if choices and len(choices) != len(non_null):
        prop.pop("anyOf")
        prop.update(non_null[0] if len(non_null) == 1 else {"anyOf": non_null})
    prop.pop("default", None)
    node["required"] = list(dict.fromkeys([*node.get("required", []), field]))


def decision_schema(schema, payload):
    result = deepcopy(schema)
    if payload.get("hypothesis_test_context", {}).get("enabled"):
        require_value(result, "InvestigationPlan", "hypothesis_test")
    if payload.get("decision_value_checks"):
        require_value(result, "InvestigationPlan", "decision_value")
        require_value(result, "OutcomeExpectation", "decision_effect")
    definitions = result.get("$defs", {})
    if payload.get("decision_value_checks") and "DecisionValue" in definitions:
        for field in ("delivered_observable", "observation_scope", "unresolved_extension", "why_not_redundant"):
            definitions["DecisionValue"]["properties"][field].update(NONBLANK)
    if payload.get("decision_economy") and payload.get("decision_value_checks") and "OutcomeExpectation" in definitions:
        outcome = definitions["OutcomeExpectation"]
        variants = [{"decision_effect": {"type": "string", "const": effect},
                     "judgment_change": {"type": "string", "enum": list(changes)},
                     "affected_labels": {"type": "array", "minItems": 1, "uniqueItems": True,
                                         "items": {"type": "string", "enum": list(MECHANISM_POOL)}}}
                    for effect, changes in MATERIAL_OUTCOME_CHANGES.items()]
        variants.append({"decision_effect": {"enum": ["preparation", "no_change"]}})
        definitions["OutcomeExpectation"] = object_variants(outcome, variants)
    plan = definitions.get("InvestigationPlan")
    if plan and payload.get("comparative_evidence"):
        for field in ("decision_question", "usefulness"):
            plan["properties"][field].update(NONBLANK)
        for outcome in definitions.get("OutcomeExpectation", {}).get("anyOf", [definitions.get("OutcomeExpectation", {})]):
            for field in ("observation", "judgment_effect"):
                if field in outcome.get("properties", {}):
                    outcome["properties"][field].update(NONBLANK)
        plan["properties"]["candidate_labels"].update(
            uniqueItems=True, items={"type": "string", "enum": list(MECHANISM_POOL)})
        definitions["InvestigationPlan"] = object_variants(plan, [
            {"purpose": {"type": "string", "const": "discrimination"},
             "candidate_labels": {"minItems": 2}, "possible_outcomes": {"minItems": 2}},
            {"purpose": {"type": "string", "enum": ["preparation", "exploration", "recovery"]}},
        ])
    # Refine only live action branches; historical models stay readable.
    if payload.get("stage") == "planner_question_act":
        request = definitions.get("DispatchRequest", {}).get("properties", {})
        if "round_id" in request and payload.get("round_id"):
            request["round_id"]["const"] = payload["round_id"]
        for branch in result.get("anyOf", [result]):
            properties = branch.get("properties", {})
            if "case_id" in properties and payload.get("case_id"):
                properties["case_id"]["const"] = payload["case_id"]
            if "revision" in properties and "revision" in payload:
                properties["revision"]["const"] = payload["revision"]
            action = properties.get("action", {}).get("const")
            if action in {"finalize", "stop"}:
                properties["investigation_plans"]["maxItems"] = 0
            if action == "dispatch" and payload.get("comparative_evidence"):
                properties["investigation_plans"]["minItems"] = 1
            if action == "finalize":
                properties["ranking_justification"] = dict(NONBLANK)
                properties["ranking_uncertainty_material"] = {"type": "boolean"}
    from mechcal.runtime.scoped_contracts import judgment_schema
    return judgment_schema(result, payload)


def context_read_schema(schema, payload):
    result = deepcopy(schema)
    properties = result["properties"]
    memory = payload.get("memory_access", {})
    discovery = payload.get("tool_discovery", {})
    properties["memory_queries"]["maxItems"] = min(4, max(0, memory.get("queries_remaining", 0)))
    if discovery.get("remaining_batches", 0) <= 0:
        properties["describe_capability_ids"]["maxItems"] = 0
    return result


def can_reopen_context(payload):
    access = payload.get("context_access", {})
    return (not access.get("lookup_active", False) and access.get("reopens_remaining", 2) > 0
            and (payload.get("memory_access", {}).get("queries_remaining", 0) > 0
                 or payload.get("tool_discovery", {}).get("remaining_batches", 0) > 0))
