"""Scoped proofreading after a source-bound scientific review, without approval reuse."""

from mechcal.runtime.command_memory import digest, json_bytes, resolve_pointer, _read_value
from mechcal.runtime.command_reads import merge_handles
from mechcal.runtime import command_read_handoff as handoff
from mechcal.runtime.command_operations import protocol, record_index
from mechcal.runtime.command_numeric import parent_context as _parent_context
from mechcal.runtime.command_review import review_basis, review_execution_context, review_sections
from mechcal.runtime.command_review_delta import retained_reads

FACT_BYTES = 12000
SCIENCE_FIELDS = {"smiles", "mechanism_pool", "mechanism_definitions", "ranking", "arguments", "candidate_overview",
                  "pending_experiments", "evidence_followup", "run_limits", "tool_catalog"}


def scientific_basis(sections, source_basis):
    # Assembled predictions copy final limitations into each row. Those are
    # delivery text, not independent changes to the accepted arguments/ranking.
    material = {k: v for k, v in sections.items() if k.split(":", 1)[0] in SCIENCE_FIELDS}
    return digest({"sources": source_basis, "claims": material,
                   "computations": sections.get("computational_review", {}).get("basis")})


def fact_packets(case, refs, *, citations=(), index=None):
    """Join selected exact values to their execution; no inferred condition or evidence."""
    index = record_index(case, index)
    refs = set(refs) & index.available.keys()
    inherited, _ = handoff.retained(case, allowed_refs=refs, index=index)
    p = protocol(case)
    # Exact, version-checked facts already read by another role are available
    # for verification. They are not Reviewer endorsements.
    handles = merge_handles(inherited, [h for h in p.get("read_memory", []) if h["ref"] in refs])
    handles = merge_handles(handles, retained_reads(case, "final_reviewer", index, refs))
    handles = merge_handles(handles, [h for h in p.get("planner_read_handles", []) if h["ref"] in refs])
    for row in citations:
        selection = row.get("selection", {})
        if row["ref"] not in refs or selection.get("view") != "data" or "data" not in row:
            continue
        for path, value in (row["data"].items() if selection.get("paths") else [("", row["data"])]):
            handles.append({"ref": row["ref"], "path": path, "page": selection.get("page"),
                            "sha256": digest(value)})
    contexts = review_execution_context(case, refs, index.available)
    execution = {ref: row for row in contexts for ref in row["source_refs"]}
    packets, deferred, seen = [], [], set()
    used = 0
    contexts_seen = {}
    for handle in reversed(handles):
        ref, path, page = (handle[k] for k in ("ref", "path", "page"))
        key = digest({"ref": ref, "path": path, "page": page})
        if key in seen:
            continue
        seen.add(key)
        try:
            value = _read_value(handle, index.available)
        except (KeyError, IndexError, TypeError, ValueError):
            deferred.append({"ref": ref, "path": path, "visibility": "source_changed_or_missing"})
            continue
        selection = {"ref": ref, "paths": [path] if path else [], "view": "data", "page": page}
        packet = {"source": selection, "source_version": digest(index.available[ref]), "value": value,
                  "parent_context": _parent_context(index.available[ref], path),
                  "visibility": "in_request", "execution": execution.get(ref)}
        if ref in contexts_seen:
            packet.pop("execution")
            packet["execution_at_packet"] = contexts_seen[ref]
        size = json_bytes(packet)
        if used + size <= FACT_BYTES:
            contexts_seen.setdefault(ref, len(packets))
            packets.append(packet)
            used += size
        else:
            deferred.append({"read": selection, "bytes": size, "visibility": "stored_not_shown"})
    return {"packets": packets, "deferred": deferred,
            "policy": "Exact stored values with their own source conditions, not new evidence. "
                      "Parent fields are literal siblings, not an inferred assignment. Missing conditions remain unknown. "
                      "Do not transfer a value, ratio, state or geometry between sources. Deferred values are not shown."}


def prepare(case, payload, *, index):
    """Return a proofread view only when the accepted scientific basis is unchanged."""
    p = protocol(case)
    sections = review_sections(payload, granular=True)
    basis = scientific_basis(sections, review_basis(case, index=index))
    binding = {"scientific_basis": basis, "delivery_review_mode": "scientific"}
    previous = p.get("review_memory", {}).get("final_reviewer")
    if not previous or previous.get("feedback", {}).get("status") != "available":
        return payload, binding
    if previous.get("case_id") != case.case_id or index.available.get(previous.get("ref")) != previous:
        raise RuntimeError("Final review receipt is not registered in this case")
    prior_basis = previous.get("scientific_basis") or scientific_basis(previous["sections"], previous["basis"])
    computational = payload.get("computational_review")
    if prior_basis != basis or (computational is not None and computational.get("previous") is None):
        return payload, binding
    anchor = previous.get("scientific_review_ref") or previous["ref"]
    if anchor not in index.available or index.available[anchor].get("case_id") != case.case_id:
        raise RuntimeError("Scientific review anchor is unavailable or belongs to another case")
    issues = payload.get("review_workflow", {}).get("issues", [])
    focused = {ref for issue in issues for ref in issue.get("source_refs", [])}
    focused.update(row["ref"] for row in payload.get("citations", []))
    kept = {k: payload[k] for k in ("stage", "role", "draft_id", "ranking", "review_targets", "review_workflow")}
    answer = payload.get("assembled_answer") or {}
    kept["assembled_answer"] = {k: answer[k] for k in ("answer", "limitations") if k in answer}
    kept["review_facts"] = fact_packets(case, focused, citations=payload.get("citations", []), index=index)
    kept["delivery_task"] = {"mode": "proofread", "scientific_review_ref": anchor,
        "scientific_basis": basis, "current_draft_id": payload["draft_id"],
        "policy": "Check all changed prose and its effects on other claims; keep unresolved findings. "
                  "The source/accepted-claim basis is unchanged. Reuse the existing computational advice, not "
                  "scientific approval. Read the scientific receipt or raw sources when needed; new scientific "
                  "objections remain reportable. This task does not re-propose the entire computation agenda."}
    binding.update(delivery_review_mode="proofread", scientific_review_ref=anchor)
    return kept, binding
