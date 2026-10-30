"""Rank-blind evidence comparison; the Planner alone writes the ranking."""

import hashlib
import json

from mechcal.agents.observation_view import observation_view
from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime.retry import RequestValidationError
from mechcal.schemas.predecision import PredecisionReview


def comparison_input(case):
    from mechcal.runtime.result_contract import operational_evidence_refs
    # Deliberately omit hypotheses, priorities, support arguments, planned targets,
    # and prior comparisons. They would tell the reviewer which order to defend.
    results = case.runtime.get("question_loop", {}).get("results", [])
    records = [{key: row[key] for key in (
        "result_id", "capability_id", "status", "structured_results", "worker_explanation",
        "explanation_review", "evidence_refs", "failure", "observation_receipt",
    ) if key in row} for row in results]
    for record in records:
        explanation = record.pop("worker_explanation", None) or {}
        if explanation.get("findings"):
            # Reuse attributed, already audited observations with their conditions.
            # Do not resend full numeric arrays or the task's favored mechanism.
            record.pop("structured_results", None)
            record["findings"] = [{k: v for k, v in finding.items() if k != "mechanism_implications"}
                                  for finding in explanation["findings"]]
            record["missing_observations"] = explanation.get("missing_observations", [])
    return {"case_id": case.case_id, "smiles": case.input.smiles,
            "mechanism_pool": sorted(MECHANISM_POOL),
            "operational_only_evidence_refs": sorted(operational_evidence_refs(case)),
            "scope": "Source-linked Worker observations and their audits, not an independent remeasurement. Raw observations are supplied only when no Worker findings exist.",
            "observations": observation_view(records)}


def input_id(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def validate_review(review, payload):
    refs = {"input:smiles"} | {ref for row in payload["observations"] for ref in row.get("evidence_refs", [])}
    scientific_refs = refs - set(payload.get("operational_only_evidence_refs", []))
    pairs = set()
    issues = []
    for i, item in enumerate(review.comparisons):
        pair = frozenset((item.candidate_a, item.candidate_b))
        if len(pair) != 2 or not pair <= set(MECHANISM_POOL) or pair in pairs:
            issues.append({"path": f"comparisons.{i}", "code": "comparison_pair",
                           "message": "Use distinct pool labels and do not repeat a pair."})
        if not set(item.evidence_refs) <= refs:
            issues.append({"path": f"comparisons.{i}.evidence_refs", "code": "comparison_sources",
                           "message": "Use observations supplied in this case, not invented sources."})
        if item.outcome != "unresolved" and not (set(item.evidence_refs) & scientific_refs):
            issues.append({"path": f"comparisons.{i}.outcome", "code": "operational_not_scientific",
                           "message": "Operational failure alone cannot favor or support a mechanism."})
        pairs.add(pair)
    if not review.comparisons and not review.unassessable.strip():
        issues.append({"path": "unassessable", "code": "comparison_unavailable",
                       "message": "Explain why the observations permit no useful comparison."})
    if issues:
        raise RequestValidationError(issues)


def ensure_comparison(o, case, store, round_id):
    from mechcal.runtime.question_loop import checkpoint
    from mechcal.runtime.questions import replace_state, state

    if not case.runtime.get("predecision_comparison"):
        return case
    payload = comparison_input(case)
    if not payload["observations"]:
        return case
    review_id = input_id(payload)
    prior = state(case).get("predecision_review")
    if prior and prior["review_id"] == review_id:
        return case
    store.activity("Compare evidence before ranking", round_id)
    store.write_json(f"questions/predecision/{review_id}_input.json", payload)
    review = o.deps.mechanism_agenda.compare_evidence(
        payload, event_sink=store.model_events("predecision_comparison", case_id=case.case_id, round_id=round_id),
    )
    review = PredecisionReview.model_validate(review)
    validate_review(review, payload)
    record = {"review_id": review_id, "role": "advisory_not_evidence", **review.model_dump(mode="json")}
    # Persist before the Planner call, so a transport failure cannot repay this review.
    case = checkpoint(replace_state(case, predecision_review=record), store)
    store.write_json(f"questions/predecision/{review_id}.json", record)
    return case


def validate_response(case, response):
    if not case.runtime.get("predecision_comparison"):
        return
    review = case.runtime.get("question_loop", {}).get("predecision_review")
    if review and (response is None or response.review_id != review["review_id"] or not response.response.strip()):
        raise RequestValidationError([{
            "path": "comparison_response", "code": "unaddressed_predecision_review",
            "message": f"Respond to current review {review['review_id']}. You may disagree; the reviewer cannot set your ranking.",
        }])
