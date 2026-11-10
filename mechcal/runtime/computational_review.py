"""Shared computational opportunities and advisory, version-bound stopping review."""

from mechcal.runtime.command_memory import digest
from mechcal.runtime.command_operations import check_owner, protocol, reject
from mechcal.runtime.command_review_workflow import comparison_options
from mechcal.runtime.command_views import ordered_ranking
from mechcal.mechanism_pool import mechanism_definitions
from mechcal.runtime.command_comparison import comparison_view, question_evidence_view


def context(case, catalog, *, registry=None, amesp_enabled=False):
    check_owner(case)
    p = protocol(case)
    tools = [{k: row[k] for k in ("capability_id", "description", "status", "missing_inputs", "reason")
              if k in row} for row in catalog]
    executions = [{"result_id": rid, "capability_id": r["capability_id"], "status": r["status"]}
                  for rid, r in p.get("results", {}).items()]
    questions = comparison_options(case, catalog, include_closed=True, registry=registry, amesp_enabled=amesp_enabled)
    comparison = comparison_view(case)
    question_evidence = question_evidence_view(case)
    basis = digest({"case_id": case.case_id, "catalog": tools, "results": p.get("results", {}),
                    "definitions": mechanism_definitions(), "candidate_comparison": comparison,
                    "questions": questions, "ranking": [{k: r.get(k) for k in ("label", "confidence", "priority")}
                                                         for r in ordered_ranking(case)],
                    "assessments": (p.get("exploration") or {}).get("entries", {})})
    previous = p.get("computational_review")
    if previous and previous.get("case_id") != case.case_id:
        raise RuntimeError("Computational review belongs to another case")
    current = previous if previous and previous["basis"] == basis else None
    return {"basis": basis, "executions": executions, "comparisons": questions,
            **({"question_evidence": question_evidence} if question_evidence["questions"] else {}),
            "candidate_comparison": comparison,
            "previous": current,
            "policy": "Computations provide preliminary guidance, not experimental proof. "
                      "ready means prerequisites permit an attempt, not success. Check the full tool_catalog, "
                      "not only Planner's chosen routes. Model judgments do not determine backend availability. "
                      "A deferred or untried route is not a failed calculation. A failed route does not exclude a mechanism family."}


def accept_review(case, command, feedback, material, catalog):
    proposals = command.next_computations
    if proposals is None:
        previous = material.get("previous")
        if previous is None:
            reject(["next_computations"], "missing_computational_review",
                   "No current computational review can be reused.",
                   "Recommend useful catalog routes, or use [] and explain why remaining computations would not help.")
        proposals = previous["suggestions"]
    else:
        proposals = [p.model_dump(mode="json") for p in proposals]
    known = {row["capability_id"] for row in catalog}
    questions = {row["id"] for row in material["comparisons"]}
    for i, item in enumerate(proposals):
        cid = item["capability_id"]
        if cid not in known:
            reject(["next_computations", i, "capability_id"], "review_route",
                   "Use a capability ID from the current catalog.", "Correct the route; no calculation was requested.")
        if item["question_id"] is not None and item["question_id"] not in questions:
            reject(["next_computations", i, "question_id"], "review_question",
                   "Question is not in this case.", "Use a displayed ID or null for a new comparison.")
    receipt = {"case_id": case.case_id, "basis": material["basis"], "suggestions": proposals}
    return {**feedback, "computational_review": receipt}


def response_requirements(material):
    """One current capability identity with every comparison it was recommended for."""
    grouped = {}
    for row in (material.get("previous") or {}).get("suggestions", []):
        grouped.setdefault(row["capability_id"], []).append(row["purpose"])
    return [{"capability_id": cid, "purposes": purposes} for cid, purposes in grouped.items()]


def require_response(case, command, material):
    """Check acknowledgment only; never judge whether a scientific reason is correct."""
    current = material["previous"]
    if current is None:
        reject(["draft_id"], "stale_computational_review", "Remaining computational routes have not been reviewed on this basis.",
               "Request final review of the current draft. No scientific result was changed.")
    required = response_requirements(material)
    expected = {r["capability_id"] for r in required}
    responses = getattr(command, "route_responses", [])
    supplied = [r.capability_id for r in responses]
    if len(set(supplied)) != len(supplied) or set(supplied) != expected:
        reject(["route_responses"], "unaddressed_computational_advice",
               "Each distinct recommended capability needs a response covering its suggested comparisons when choosing to stop. "
               f"Missing: {sorted(expected - set(supplied))}; unexpected: {sorted(set(supplied) - expected)}; "
               f"duplicates: {sorted(cid for cid in set(supplied) if supplied.count(cid) > 1)}. "
               f"Current requirements: {required}",
               "Execute a useful test, or submit_final with a reason for each current suggestion. Scientific disagreement is allowed.")
    return [r.model_dump(mode="json") for r in responses]
