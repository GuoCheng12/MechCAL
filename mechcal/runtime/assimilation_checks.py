"""Collect independent bookkeeping/reference errors before scientific state changes."""

from mechcal.runtime.evidence_comparison import argument_change_issues, enabled
from mechcal.runtime.result_contract import operational_evidence_refs
from mechcal.runtime.retry import RequestValidationError
from mechcal.mechanism_pool import MECHANISM_POOL


def audit_response_issues(case, responses):
    state = case.runtime.get("question_loop", {})
    if not state.get("argument_review_pending"):
        return []
    expected = set(state["argument_review"]["support_ids"])
    actual = set(responses)
    blank = sorted(k for k, v in responses.items() if not v.strip())
    if actual == expected and not blank:
        return []
    return [{
        "path": ["audit_responses"], "repair_paths": [["audit_responses"]], "code": "bookkeeping",
        "message": "Respond to every reviewed support_id, including withdrawn arguments and unavailable review. "
                   f"Missing IDs: {sorted(expected - actual)}; unexpected IDs: {sorted(actual - expected)}; "
                   f"blank responses: {blank}. Review report IDs are not support IDs.",
        "expected_ids": sorted(expected),
    }]


def validate_bookkeeping(case, update):
    if not enabled(case):
        return
    issues = argument_change_issues(case, update)
    refs = {e.evidence_id for e in case.evidence_ledger.items} | {"input:smiles"}
    scientific = refs - operational_evidence_refs(case)
    state = case.runtime.get("question_loop", {})
    results = {r["result_id"]: r for r in state.get("results", [])}
    pending = {r["result_id"]: r for r in state.get("pending_results", [])}

    def issue(path, message, **details):
        issues.append({"path": path, "code": "bookkeeping", "repair_paths": [path],
                       "message": message, **details})

    if update.case_id != case.case_id:
        issue(["case_id"], "Use the current case_id.", expected=case.case_id)
    if update.revision != state.get("revision", 0) + 1:
        issue(["revision"], "Use the next assimilation revision.", expected=state.get("revision", 0) + 1)
    issues.extend(audit_response_issues(case, update.audit_responses))
    ids = [d.result_id for d in update.dispositions]
    if len(ids) != len(set(ids)) or set(ids) != set(pending):
        issue(["dispositions"], "Every pending execution needs exactly one disposition. "
              "Do not repeat already assimilated executions.", expected_ids=sorted(pending))

    def check(path, values, allowed):
        invalid = sorted(set(values) - set(allowed))
        if invalid:
            issues.append({
                "path": path, "code": "invalid_reference", "repair_paths": [path],
                "invalid_refs": invalid, "allowed_refs": sorted(allowed),
                "message": f"Unavailable or inadmissible records at {path}: {invalid}. "
                           "Choose exact IDs from allowed_refs based on the original result. "
                           "Do not guess or rewrite the scientific argument to conceal a reference error.",
            })

    for i, q in enumerate(update.questions):
        if not q.assessment.strip():
            issue(["questions", i, "assessment"], "Provide a concise assessment of this question.")
        if not q.related_labels or not set(q.related_labels) <= set(MECHANISM_POOL):
            issue(["questions", i, "related_labels"], "Select related labels from the mechanism pool.",
                  allowed_labels=list(MECHANISM_POOL))
        check(["questions", i, "evidence_refs"], q.evidence_refs, refs)
        check(["questions", i, "result_ids"], q.result_ids, results)
    for i, h in enumerate(update.portfolio.hypotheses):
        check(["portfolio", "hypotheses", i, "evidence_refs"], h.evidence_refs, refs)
    for i, a in enumerate(update.mechanism_support_arguments):
        check(["mechanism_support_arguments", i, "observation_refs"], a.observation_refs, scientific)
    check(["decision_assessment", "evidence_refs"], update.decision_assessment.evidence_refs, refs)
    if update.ranking_update:
        check(["ranking_update", "evidence_refs"], update.ranking_update.evidence_refs, scientific)
    for i, d in enumerate(update.dispositions):
        check(["dispositions", i, "result_id"], [d.result_id], pending)
        if d.result_id not in pending:
            continue
        if set(d.question_ids) != set(pending[d.result_id]["question_ids"]):
            issue(["dispositions", i, "question_ids"], "Preserve the original request-question links.",
                  expected_ids=pending[d.result_id]["question_ids"])
        available = set(pending[d.result_id]["evidence_refs"])
        check(["dispositions", i, "evidence_refs"], d.evidence_refs, available)
        for j, effect in enumerate(d.mechanism_effects):
            allowed = available & scientific if effect.effect in {"supports", "weakens"} else available
            check(["dispositions", i, "mechanism_effects", j, "evidence_refs"], effect.evidence_refs, allowed)
    if issues:
        # If a whole record list needs repair, nested issues remain visible but
        # are repaired through that same authorized parent field.
        roots = [p for issue in issues for p in issue["repair_paths"]]
        for issue in issues:
            issue["repair_paths"] = [min((r for r in roots if p[:len(r)] == r), key=len)
                                     for p in issue["repair_paths"]]
        raise RequestValidationError(issues)
