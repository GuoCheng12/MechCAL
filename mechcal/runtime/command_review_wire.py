"""Request-local shared facts and a source-bound, read-only review task.

This is a wire projection, not a second scientific memory. The original review
receipt, query outcomes and source records remain unchanged.
"""

from collections import Counter
from copy import deepcopy

from mechcal.runtime.command_compact import expand_text
from mechcal.runtime.command_memory import digest, json_bytes, pointer_child


POLICY = (
    "same_data_at is a JSON pointer into this request's review_values: substitute that exact value. "
    "It is not a record ID or a scientific citation. Shared values do not imply shared provenance "
    "or independent evidence; each occurrence retains its own source, path and conditions. "
    "Only identical JSON values are shared, without rounding, summarization or scientific judgment."
)


def share_values(material, *, minimum_bytes=512):
    """Intern repeated JSON containers, keeping every occurrence's location.

    Choose parents before children, so an already-shared table does not leave
    a second unused copy of each row. Literal strings use the existing text
    sharing layer instead. A reserved-name collision keeps the original wire.
    """
    if "review_values" in material or "shared_value_policy" in material:
        return deepcopy(material)
    counts, sizes = Counter(), {}

    def inventory(value):
        if isinstance(value, (dict, list)):
            if isinstance(value, dict) and set(value) == {"same_data_at"}:
                raise ValueError("An already packed review must be expanded before projection")
            size = json_bytes(value)
            if size >= minimum_bytes:
                key = digest(value)
                counts[key] += 1
                sizes[key] = size
            for child in value.values() if isinstance(value, dict) else value:
                inventory(child)

    inventory(material)
    shared, names = {}, {}

    def visit(value, *, inline=False):
        if not isinstance(value, (dict, list)):
            return value
        key = digest(value)
        if not inline and counts[key] > 1 and sizes[key] >= minimum_bytes:
            if key not in names:
                names[key] = f"v{len(names) + 1}"
                shared[names[key]] = visit(value, inline=True)
            return {"same_data_at": "/review_values/" + names[key]}
        if isinstance(value, dict):
            return {k: visit(v) for k, v in value.items()}
        return [visit(v) for v in value]

    packed = visit(material, inline=True)
    if not shared:
        return packed
    result = {"shared_value_policy": POLICY, "review_values": shared, **packed}
    # Small packets must not pay more for indirection than their original data.
    return result if json_bytes(result) < json_bytes(material) else deepcopy(material)


def task_binding(case_id, payload, state, role, *, sources=None):
    identity = {"case_id": case_id, "role": role,
                "context_key": state.get("context_key") or digest(payload)}
    if sources is not None:
        identity["source_key"] = digest(sources)
    binding = {**identity, "task_id": "review-task:" + digest(identity)}
    old = state.get("review_task")
    if old is not None and old != binding:
        raise RuntimeError("Review task belongs to different claims, sources, role or case")
    return binding


def method_index(scope):
    """Expose literal method labels even when long method histories are deferred.

    A submitted-input label is not proof of a successful calculation. Keep its
    provenance and result status, and never fill a missing label from a default.
    """
    sources = []
    fields = ("step_id", "source", "requested_method_label", "executed_method_label", "recorded_method_label")
    for ref, record in scope.items():
        rows = record.get("structured_results", {}).get("method_provenance")
        if not isinstance(rows, list):
            continue
        variants = {}
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            labels = {k: row[k] for k in fields if k in row}
            if not any(k in labels for k in fields[2:]):
                continue
            entry = variants.setdefault(digest(labels), {**labels, "row_indices": []})
            entry["row_indices"].append(i)
        if variants:
            sources.append({"ref": ref, "path": "/structured_results/method_provenance",
                            "status": record.get("status"), "variants": list(variants.values())})
    return {"sources": sources, "policy": "Literal labels from the referenced method rows. "
            "row_indices are zero-based. Submitted inputs describe what was requested, not convergence. "
            "Status remains authoritative; missing labels are unknown. Full parameter blocks remain queryable."}


def review_wire(material, *, binding, state, scope):
    """Self-contained continuation: claims + exact facts + one latest result.

    Worker prose is optional interpretation, not a source of execution facts.
    Defer it only when its complete record is actually readable in this task.
    All numerical observations, failure states, conditions and open review
    issues stay in the current request, even with no reads remaining.
    """
    value = expand_text(material)
    methods = method_index(scope)
    if methods["sources"]:
        value["source_method_index"] = methods
    sections = value.get("sections", value)
    interpretations = sections.get("worker_interpretations", [])
    for row in interpretations if isinstance(interpretations, list) else []:
        ref = row.get("full_text_ref")
        if ref not in scope or not isinstance(scope[ref].get("text"), str):
            continue
        row.pop("text_preview", None)
        row.pop("text_truncated", None)
        row["interpretation_read"] = {"ref": ref, "paths": ["/text"], "view": "data", "page": None}
        row["text_bytes"] = json_bytes(scope[ref]["text"])
        row["text_visibility"] = "stored_not_shown"
    value["review_task"] = {**binding,
        "phase": "continue" if state.get("exchange") else "start",
        "completed_query_turns": state.get("queries", 0),
        "read_record_ids": state.get("read_record_ids", []),
        "policy": "Continue this same review, not a new review of the entire case. Current claims, "
                  "unresolved issues, source conditions and retained facts below are authoritative for this request. "
                  "Read means available for inspection, never checked or approved. Worker interpretations are "
                  "optional prose at interpretation_read; their stored observations remain below. "
                  "Report remaining uncertainty if a deferred explanation cannot be read."}
    outcome = state.get("outcome")
    wire_outcome = outcome
    if state.get("exchange") and outcome is not None:
        # Put values in the same document as retained facts, so repeated tables
        # can be shared across the new result and the working context. Native
        # tool outputs retain one acknowledgement per original call ID.
        value["query_result"] = deepcopy(outcome)
        per_call = outcome.get("call_results")
        ids = {call["id"] for call in state["exchange"]["calls"]}
        if per_call is not None and set(per_call) != ids:
            raise RuntimeError("Review query results do not match their tool calls")

        def receipt(row, path):
            return {"status": row.get("status", "returned"), "result_in_current_user_at": path,
                    "policy": "Complete result delivered once in the current user message, not omitted or approved."}

        wire_outcome = ({"call_results": {cid: receipt(row, pointer_child("/query_result/call_results", cid))
                                         for cid, row in per_call.items()}}
                        if per_call is not None else receipt(outcome, "/query_result"))
    return share_values(value), wire_outcome
