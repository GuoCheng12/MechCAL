"""Read stored scientific findings as whole records, never generated summaries."""


def findings_record(original, *, offset, budget):
    from mechcal.runtime.case_memory import serialized
    from mechcal.runtime.memory_views import selected_record

    explanation = original.get("worker_explanation") or {}
    findings = explanation.get("findings", [])
    review = original.get("explanation_review") or {}
    # Review accompanies the finding: a preserved Worker statement may be wrong.
    context = {k: original[k] for k in ("result_id", "question_ids", "status", "failure") if k in original}
    result = {
        "view": "findings", "complete_record": False, "selected_fields": {},
        "source_context": context, "next_offset": None, "finding_count": len(findings),
        "notice": "Stored Worker observations and interpretations, not new measurements. "
                  "Apply accompanying review corrections; review status is not scientific proof. "
                  "Unreturned findings and raw calculations remain readable, not absent.",
    }
    if not findings:
        return {**result, "error": "no_stored_findings", "notice": "Read an execution record with worker_explanation, "
                "or use auto/full for this record. Missing findings do not mean missing scientific evidence."}
    if offset >= len(findings):
        return {**result, "error": "offset_beyond_findings"}
    for i in range(offset, len(findings)):
        path = f"/worker_explanation/findings/{i}"
        paths = [*result["selected_fields"], path]
        if review:
            paths.append("/explanation_review")
        selected = selected_record(original, list(dict.fromkeys(paths)), focused=False)["selected_fields"]
        candidate = {**result, "selected_fields": selected}
        if len(serialized(candidate)) > budget - 40:
            result["next_offset"] = i
            if not result["selected_fields"]:
                result.update(error="finding_with_review_exceeds_budget", blocked_path=path)
            break
        result = candidate
    return result
