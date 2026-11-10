"""Exact, case-bound background for interim reviews, not scientific evidence."""

from copy import deepcopy

from mechcal.runtime.command_memory import digest, json_bytes, pointer_child
from mechcal.runtime.command_operations import put


def context_ref(record):
    return "context:reviewer:" + digest({k: v for k, v in record.items() if k != "ref"})


def prepare_review(case, material, receipt, role):
    """Keep claims/facts/conditions/issues inline; defer unrelated case-wide history.

    Final review also decides stopping and retains its full comparison packet.
    The full receipt is unchanged. The new record is metadata, never a citation.
    """
    if role != "reviewer":
        return case, material, None
    value = deepcopy(material)
    shown, literal = value["sections"], receipt["sections"]
    fields = {}
    for name in ("tool_catalog", "candidate_overview", "evidence_followup", "comparison_options"):
        if name in shown and json_bytes(shown[name]) > 1024:
            fields[name] = deepcopy(literal[name])
    compute = shown.get("computational_review", {})
    raw_compute = literal.get("computational_review", {})
    # Do not reinterpret deltas or pointers from prior review receipts.
    if isinstance(compute, dict) and "basis" in compute:
        for name in ("comparisons", "question_evidence", "candidate_comparison"):
            if name in compute and json_bytes(compute[name]) > 1024:
                fields["computational_review." + name] = deepcopy(raw_compute[name])
    if not fields:
        return case, value, None
    record = {"case_id": case.case_id, "role": role,
              "review_basis": digest(literal), "fields": fields}
    record["ref"] = context_ref(record)
    directory = []
    for name, data in fields.items():
        path = pointer_child("/fields", name)
        pointer = {"deferred_task_context": True, "ref": record["ref"], "path": path,
                   "sha256": digest(data), "bytes": json_bytes(data)}
        if name.startswith("computational_review."):
            compute[name.split(".", 1)[1]] = pointer
        else:
            shown[name] = pointer
        directory.append({"field": name, "read": {"ref": record["ref"], "paths": [path],
                                                 "view": "data", "page": None}, "bytes": json_bytes(data)})
    # Preserve the whole candidate set in a small directory, not only the Top-3.
    value["candidate_directory"] = [{k: row[k] for k in
        ("label", "ranked", "confidence", "status") if k in row}
        for row in literal.get("candidate_overview", [])]
    catalog = fields.get("tool_catalog")
    if isinstance(catalog, dict) and "columns" in catalog and "rows" in catalog:
        columns = [k for k in ("capability_id", "status", "missing_inputs", "reason") if k in catalog["columns"]]
        value["route_status_directory"] = {
            "columns": ["source_row", *columns],
            "rows": [[i, *(row[catalog["columns"].index(k)] for k in columns)]
                     for i, row in enumerate(catalog["rows"])],
            "defaults": {k: v for k, v in catalog.get("defaults", {}).items() if k in columns},
            "source_ref": record["ref"], "row_path_prefix": "/fields/tool_catalog/rows/",
            "policy": "Exact current availability, not expected success or evidence. For a relevant route's "
                      "purpose/inputs/outputs, read its source_row at row_path_prefix plus that index; "
                      "column meanings are at /fields/tool_catalog/columns. No full catalog read is needed for status."}
    value["review_background"] = {"fields": directory, "policy":
        "Review current claims, exact citations, execution conditions and unresolved issues first. "
        "Case-wide route/question/candidate history is stored at these exact read selections. "
        "This is context, not absent evidence or an approved review. Read relevant fields (or their child paths) "
        "when a claim needs wider comparison; do not assert route unavailability without checking it. "
        "Do not cite this context record as an observation. Deferred background is not automatically verified. "
        "If remaining reads cannot resolve a question, report that uncertainty without inventing a conclusion."}
    return put(case, review_context=record), value, record["ref"]
