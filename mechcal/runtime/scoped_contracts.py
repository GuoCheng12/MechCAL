"""Live schemas expose known input IDs and structural rules before generation.

Cross-record equality, source-file validity and scientific interpretation still
need runtime validation. No values or scientific judgments are filled in here.
"""

from copy import deepcopy

from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.schemas.contracts import NONBLANK, object_variants


def restrict_items(prop, values, *, exact=False):
    values = list(dict.fromkeys(values))
    prop["items"] = {"type": "string", "enum": values} if values else {"type": "string"}
    prop["uniqueItems"] = True
    if exact:
        prop.update(minItems=len(values), maxItems=len(values))
    elif not values:
        prop["maxItems"] = 0


def judgment_schema(schema, payload):
    """A local argument edit retains ranking; a global edit may replace it."""
    from mechcal.schemas.judgment_commands import LOCAL_AUDIT_RETAINED_FIELDS

    result = deepcopy(schema)
    nodes = [result, *result.get("$defs", {}).values()]
    for node in nodes:
        prop = node.get("properties", {})
        if prop.get("action", {}).get("const") != "update_judgment":
            continue
        pending = payload.get("argument_review_pending")
        ids = list(dict.fromkeys((payload.get("argument_review") or {}).get("support_ids", [])))
        if pending:
            prop["audit_responses"] = {
                "type": "object", "properties": {sid: dict(NONBLANK) for sid in ids},
                "required": ids, "additionalProperties": False,
                "description": "Respond only to these pending support IDs, not the review report ID.",
            }
        local = {"scope": {"type": "string", "const": "local_audit"}}
        for field in LOCAL_AUDIT_RETAINED_FIELDS:
            local[field] = ({"type": "null", "description": "Retain the accepted ranking and comparisons."}
                            if field == "ranking_update" else {"maxItems": 0})
        variants = [{"scope": {"type": "string", "const": "global"}}]
        if pending is None or pending and not payload.get("pending_results"):
            variants.append(local)
        refined = object_variants(node, variants)
        node.clear()
        node.update(refined)
    return result


def worker_schema(schema, request, tool_result):
    result = deepcopy(schema)
    restrict_items(result["properties"]["question_ids"], request.question_ids, exact=True)
    if tool_result.status not in {"success", "partial"}:
        result["properties"]["findings"]["maxItems"] = 0
    finding = result["$defs"]["WorkerFinding"]["properties"]
    restrict_items(finding["evidence_refs"], [e.evidence_id for e in tool_result.evidence_units
                                            if e.status in {"present", "partial"}])
    return result


def result_review_schema(schema, pending):
    result = deepcopy(schema)
    result["properties"]["status"] = {"type": "string", "const": "reviewed"}
    result["properties"]["interpretations"].update(minItems=len(pending), maxItems=len(pending))
    branches = []
    for record in pending:
        branch = deepcopy(result["$defs"]["ResultInterpretation"])
        prop = branch["properties"]
        prop["result_id"]["const"] = record["result_id"]
        restrict_items(prop["question_ids"], record["question_ids"], exact=True)
        restrict_items(prop["evidence_refs"], record.get("evidence_refs", []))
        findings = [f["finding_id"] for f in (record.get("worker_explanation") or {}).get("findings", [])]
        prop["finding_audits"].update(minItems=len(findings), maxItems=len(findings))
        if findings:
            audit = deepcopy(result["$defs"]["FindingAudit"])
            for variant in audit.get("anyOf", [audit]):
                variant["properties"]["finding_id"]["enum"] = findings
            prop["finding_audits"]["items"] = audit
        branches.append(branch)
    if branches:
        result["$defs"]["ResultInterpretation"] = {"anyOf": branches}
    return result


def predecision_schema(schema, payload):
    result = deepcopy(schema)
    prop = result["$defs"]["EvidenceComparison"]["properties"]
    for name in ("candidate_a", "candidate_b"):
        prop[name]["enum"] = list(MECHANISM_POOL)
    refs = {"input:smiles"} | {ref for row in payload["observations"] for ref in row.get("evidence_refs", [])}
    restrict_items(prop["evidence_refs"], sorted(refs))
    return object_variants(result, [
        {"comparisons": {"maxItems": 0}, "unassessable": NONBLANK},
        {"comparisons": {"minItems": 1}},
    ])


def candidate_schema(schema, payload):
    result = deepcopy(schema)
    prop = result["$defs"]["CandidateProposal"]["properties"]
    restrict_items(prop["related_labels"], MECHANISM_POOL)
    refs = {ref for row in payload["computations"] for ref in row["evidence_refs"]}
    restrict_items(prop["evidence_refs"], sorted(refs - set(payload["operational_only_evidence_refs"])))
    return result


def final_delivery_schema(schema, case, draft):
    from mechcal.runtime.result_contract import operational_evidence_refs

    result = deepcopy(schema)
    prop = result["properties"]
    restrict_items(prop["withdrawn_support_ids"], [a.support_id for a in draft.arguments])
    for field in ("final_answer_draft", "reconsideration_reason"):
        prop[field] = {"anyOf": [dict(NONBLANK), {"type": "null"}],
                       "description": prop[field].get("description", "Omit/null to retain; otherwise supply nonblank text.")}
    missing = result["$defs"]["MissingEvidence"]["properties"]
    missing["label"]["enum"] = [h.name for h in case.portfolio.sorted_hypotheses()[:3]]
    missing["reason"].update(NONBLANK)
    argument = result["$defs"]["MechanismSupportArgument"]["properties"]
    argument["target_label"]["enum"] = list(MECHANISM_POOL)
    refs = ({e.evidence_id for e in case.evidence_ledger.items} | {"input:smiles"}) - operational_evidence_refs(case)
    restrict_items(argument["observation_refs"], sorted(refs))
    return result


def stop_review_schema(schema, case, payload):
    from mechcal.runtime.planner_contract_policy import decision_schema

    result = decision_schema(schema, {"decision_value_checks": case.runtime.get("decision_value_checks", False)})
    prop = result["properties"]
    prop["status"] = {"type": "string", "const": "reviewed"}
    restrict_items(prop["evidence_refs"], ["input:smiles", *[e.evidence_id for e in case.evidence_ledger.items]])
    capabilities = [r["capability_id"] for r in payload["capability_cards"] if r["status"] != "disabled"]
    restrict_items(prop["capability_ids"], capabilities)
    if capabilities:
        result["$defs"]["ContinuationOpportunity"]["properties"]["capability_id"]["enum"] = capabilities
    if case.runtime.get("decision_value_checks"):
        prop["capability_ids"]["maxItems"] = min(1, prop["capability_ids"].get("maxItems", 1))
        variants = [{"capability_ids": {"maxItems": 0}, "continuation_opportunity": {"type": "null"}}]
        if capabilities:
            variants.append(
            {"capability_ids": {"minItems": 1, "maxItems": min(1, len(capabilities))},
             "continuation_opportunity": {"$ref": "#/$defs/ContinuationOpportunity"}, "challenge": NONBLANK})
        result = object_variants(result, variants)
    return result
