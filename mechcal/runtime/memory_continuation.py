"""A self-contained Planner continuation with historical detail behind read handles."""

from copy import deepcopy

from mechcal.runtime.case_memory import serialized
from mechcal.runtime.observation_context import project_observations
from mechcal.runtime.planner_context import referenced_ids, select


def continuation_context(payload, responses):
    """Keep judgments and evidence boundaries; retrieve bulk metrics explicitly."""
    result = deepcopy(payload)
    # Draft-repair requests can wrap the same context instead of exposing it at root.
    if isinstance(result.get("scientific_context"), dict):
        result["scientific_context"] = continuation_context(result["scientific_context"], responses)
    if not payload.get("scoped_evidence_updates"):
        return result
    if "observations" in result:
        result["observations"] = project_observations(result["observations"])
    if payload.get("stage") == "planner_final_revision":
        return result
    # Explicit retrieval and active corrections pin old rows without a relevance heuristic.
    pinned = {rid for row in responses for rid in row.get("request", {}).get("record_ids", [])}
    pinned.update(referenced_ids(payload.get("stop_review_feedback")))
    pinned.update(r["result_id"] for r in payload.get("pending_results", []))
    if payload.get("argument_review_pending"):
        pinned.update(referenced_ids(payload.get("argument_review")))
    pinned.update(e["evidence_id"] for e in payload.get("observations", []))
    retrieved_context = {}
    for response in responses:
        for record in response.get("records", []):
            detail = record.get("record", {})
            if "selected_fields" in detail:
                retrieved_context[record["record_id"]] = detail.get("source_context", {})
    for row in result.get("execution_history", []):
        already_read = retrieved_context.get(row.get("result_id"), {})
        duplicates = [key for key in ("requested_parameters", "calculation_identity")
                      if key in row and key in already_read
                      and serialized(row[key]) == serialized(already_read[key])]
        for key in duplicates:
            row.pop(key)
        if duplicates:
            row["parameters_in_retrieved_record"] = row["result_id"]
        if row.get("result_id") in pinned or pinned.intersection(row.get("evidence_refs", [])):
            continue
        # Status, failure and delivered/missing quantities remain visible to avoid
        # proposing the same failed or already completed work from a bare ID.
        omitted = [key for key in ("requested_parameters", "calculation_identity", "request_signature")
                   if key in row]
        for key in omitted:
            row.pop(key)
        if omitted:
            row["memory_record_id"] = row["result_id"]
            row["read_before_reuse"] = ["/" + key for key in omitted if key != "request_signature"]
    for row in result.get("hypothesis_test_context", {}).get("recent_tests", []):
        if (not row.get("memory_record_id") or row.get("result_id") in pinned
                or row.get("memory_record_id") in pinned):
            continue
        claim = row.get("tested_claim")
        if isinstance(claim, dict):
            # All test outcomes stay visible, including failures to discriminate.
            # The historical test's full wording is readable by memory_record_id.
            row["tested_claim"] = select(claim, {"target_label", "alternative_label"})
    result["memory_continuation"] = {
        "version": "focused_read_v2",
        "notice": "This is a self-contained decision context, not implicit model memory. "
                  "Current judgment, evidence summaries/conditions, open issues and corrections are retained. "
                  "Bulk metrics are behind metrics_memory, not implicit model memory. "
                  "Historical execution parameters are readable by memory_record_id; read them "
                  "before reusing or repeating that calculation. Missing detail is not missing evidence.",
    }
    # A small history can cost less than the retrieval notice; never expand it.
    return result if len(serialized(result)) < len(serialized(payload)) else deepcopy(payload)
