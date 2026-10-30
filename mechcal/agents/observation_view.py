"""Model-facing views only. Canonical evidence and computational inputs stay intact."""

from math import fsum


def observation_view(value, *, preserve_paths=False):
    """Summarize only large orbital-coefficient tables, never arbitrary observations."""
    if isinstance(value, list):
        return [observation_view(item, preserve_paths=preserve_paths) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: observation_view(item, preserve_paths=preserve_paths) for key, item in value.items()}
    rows = value.get("components")
    if ("state_index" not in value or not isinstance(rows, list) or len(rows) <= 8
            or not all(isinstance(row, dict) and {"from_orbital", "to_orbital", "coefficient"} <= row.keys()
                       and type(row["coefficient"]) in (int, float) for row in rows)):
        return result
    selected = sorted(enumerate(rows), key=lambda pair: (-abs(pair[1]["coefficient"]), pair[0]))[:8]
    if preserve_paths:
        # A sampled array must not reuse the original array's JSON-pointer indices.
        result.pop("components")
        result["component_excerpt"] = [{"source_index": i, "value": row} for i, row in selected]
    else:
        result["components"] = [observation_view(row) for _, row in selected]
    result["component_table_view"] = {
        "selection": "8 largest absolute coefficients; original orbital indices and signs preserved",
        "total_rows": len(rows), "omitted_rows": len(rows) - 8,
        "all_coefficients_squared_sum": fsum(row["coefficient"] ** 2 for row in rows),
        "shown_coefficients_squared_sum": fsum(row["coefficient"] ** 2 for _, row in selected),
        "source": "Full table remains in this evidence/result record and its source artifact, at the same state_index.",
        "boundary": "Omitted rows are not zero. This display is not a complete wavefunction and must not be used for density reconstruction. Computational tools use the full original bundle.",
    }
    if preserve_paths:
        result["component_table_view"]["addressing"] = (
            "component_excerpt is a view, not the original components array. "
            "Use source_index to address components/<source_index> in the original record.")
    return result


def repair_payload(context, correction):
    """Keep the candidate and precise repair issues, not the full scientific history."""
    result = {
        **correction,
        "case_id": context["case_id"],
        "evidence_index": context.get("evidence_index", []),
        "execution_history": [
            {key: record[key] for key in ("result_id", "capability_id", "status", "evidence_refs") if key in record}
            for record in context.get("execution_history", [])
        ],
    }
    if context.get("hypothesis_test_context", {}).get("enabled"):
        # A repaired assessment still needs its prospective test and delivered observations.
        result["hypothesis_test_context"] = context["hypothesis_test_context"]
        result["pending_results"] = context.get("pending_results", [])
        pending_refs = {ref for item in result["pending_results"] for ref in item.get("evidence_refs", [])}
        result["observations"] = [item for item in context.get("observations", [])
                                  if item.get("evidence_id") in pending_refs]
        result["result_review"] = context.get("result_review")
        result["operational_only_evidence_refs"] = context.get("operational_only_evidence_refs", [])
    if context.get("candidate_context", {}).get("enabled"):
        result["candidate_context"] = context["candidate_context"]
        result["questions"] = context.get("questions", [])
    if context.get("boundary_decision_context", {}).get("enabled"):
        # New comparison repairs must see their scientific sources, not only IDs.
        for key in ("boundary_decision_context", "mechanism_pool", "portfolio",
                    "observations", "public_input_source", "operational_only_evidence_refs",
                    "questions", "capability_catalog", "stop_review_feedback"):
            result[key] = context.get(key)
    return result
