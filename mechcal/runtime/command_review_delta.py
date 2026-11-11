"""Exact review deltas and shared receipts, never semantic approval or evidence."""

from difflib import SequenceMatcher
import re

from mechcal.runtime.command_memory import digest, json_bytes, pointer_child, resolve_pointer
from mechcal.runtime.command_reads import merge_handles


def text_edits(before: str, after: str) -> list[dict]:
    old, new = re.findall(r"\S+\s*|\s+", before), re.findall(r"\S+\s*|\s+", after)
    offsets = [0]
    for token in old:
        offsets.append(offsets[-1] + len(token))
    edits = []
    for kind, a, b, c, d in SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if kind == "equal":
            continue
        start, end = offsets[a], offsets[b]
        edits.append({"start": start, "end": end, "before": before[start:end], "after": "".join(new[c:d]),
                      "left_context": before[max(0, start - 100):start], "right_context": before[end:end + 100]})
    return edits


def changes(before, after, path="") -> list[dict]:
    """All edits are included; lists stay atomic to preserve order and identity."""
    if before == after:
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        rows = []
        for key in sorted(before.keys() | after.keys()):
            child = pointer_child(path, key)
            if key not in after:
                rows.append({"op": "remove", "path": child, "before": before[key]})
            elif key not in before:
                rows.append({"op": "add", "path": child, "value": after[key]})
            else:
                rows.extend(changes(before[key], after[key], child))
        return rows
    replacement = {"op": "replace", "path": path, "before": before, "value": after}
    if isinstance(before, str) and isinstance(after, str) and len(before) > 512:
        delta = {"op": "text_edits", "path": path, "edits": text_edits(before, after)}
        if json_bytes(delta) < json_bytes(replacement):
            return [delta]
    return [replacement]


def receipts(case, role, index):
    memory = case.runtime["command_protocol"].get("review_memory", {})
    roles = [role] + [r for r in ("reviewer", "final_reviewer") if r != role]
    candidates = []
    for name in roles:
        previous = memory.get(name)
        if previous is None:
            continue
        if index.available.get(previous.get("ref")) != previous:
            raise RuntimeError("Review memory is not the registered case-local receipt")
        if previous.get("feedback", {}).get("status") == "available":
            candidates.append(previous)
    return candidates


def retained_reads(case, role, index, scope):
    """Same-case source-versioned reads, shared across ordinary and final review."""
    handles = []
    for receipt in reversed(receipts(case, role, index)):
        for handle in receipt.get("source_reads", []):
            ref = handle["ref"]
            if ref not in scope or receipt.get("read_versions", {}).get(ref) != index.version(ref):
                continue
            value = resolve_pointer(index.available[ref], handle["path"])
            if handle.get("page"):
                page = handle["page"]
                value = value[page["offset"]:page["offset"] + page["limit"]]
            if digest(value) != handle["sha256"]:
                raise RuntimeError("Review read conflicts with its recorded source version")
            handles.append(handle)
    return merge_handles([], handles)


def section_entries(receipt):
    entries = {k: (v, pointer_child("/sections", k)) for k, v in receipt["sections"].items()}
    # Old receipts stored all predictions together. Their original paths remain
    # authoritative; expose per-label views without rewriting historical data.
    old = entries.pop("assembled_answer:mechanism_predictions", None)
    if old is not None:
        predictions, path = old
        for i, row in enumerate(predictions):
            entries[f"assembled_answer:prediction:{row['label']}"] = (row, pointer_child(path, str(i)))
    return entries


def packed_change(prior, path, before, after):
    anchor = {"previous_review_ref": prior["ref"], "path": path}
    if before == after:
        packed = {**anchor, "meaning": "Exact content unchanged; reuse is not scientific approval."}
        if isinstance(after, dict):
            packed.update({k: after[k] for k in ("id", "label", "ref", "status") if k in after})
        return packed
    diff = {**anchor, "view": "exact_delta", "base_sha256": digest(before),
            "current_sha256": digest(after), "changes": changes(before, after)}
    return diff if json_bytes(diff) < json_bytes(after) else None


def section_dependencies(sections, index):
    """Bind content reuse to its explicit sources, not to unrelated new results."""
    from mechcal.runtime.command_operations import citation_source_refs
    from mechcal.runtime.execution_inventory import groups
    inventories = groups(index.case, index=index)
    result_routes = {ref: row.get("capability_id") for ref, row in
                     index.case.runtime["command_protocol"].get("results", {}).items()}
    arguments = {v.get("id"): v for k, v in sections.items() if k.startswith("arguments:")}
    dependencies = {}
    for key, value in sections.items():
        refs, claims = set(), []
        if key.startswith("arguments:"):
            claims = [value]
        elif key.startswith("observations:"):
            refs.add(key.split(":", 1)[1])
        elif key.startswith("citations:"):
            refs.add(value["ref"])
        elif key == "ranking":
            claims = [arguments[aid] for row in value for aid in row.get("argument_ids", []) if aid in arguments]
        elif key.startswith("assembled_answer:prediction:"):
            claims = [a for a in arguments.values() if a.get("label") == value.get("label")]
        elif key in {"assembled_answer:answer", "assembled_answer:limitations"}:
            dependencies[key] = {"execution_versions": {cid: row["version"] for cid, row in inventories.items()}}
            continue
        else:
            continue
        for claim in claims:
            refs.update([*claim.get("refs", []), *claim.get("limitation_refs", [])])
            refs.update(c["ref"] for c in claim.get("citations", []))
        refs.update(citation_source_refs(index.case, refs))
        dependencies[key] = {"claims": [] if key.startswith("arguments:") else [digest(c) for c in claims],
                             "sources": {ref: index.version(ref) if ref in index.available else None
                                         for ref in sorted(refs)}}
        if claims:
            routes = {result_routes[ref] for ref in refs if ref in result_routes}
            dependencies[key]["execution_versions"] = {cid: inventories[cid]["version"]
                for cid in sorted(routes) if cid in inventories}
    return dependencies


def project(case, payload, role, sections, receipt, index, *, task_context=False):
    from mechcal.runtime.command_review_workflow import state
    # Changed public definitions/comparison tasks need a fresh interpretation.
    # Version-checked factual reads remain reusable through retained_reads.
    candidates = [r for r in receipts(case, role, index)
                  if r.get("context_contract") == receipt.get("context_contract")]
    entries = {r["ref"]: section_entries(r) for r in candidates}
    receipt["section_dependencies"] = section_dependencies(sections, index)

    def reusable(prior, key):
        current = receipt["section_dependencies"].get(key)
        if current is None:
            return True
        previous = prior.get("section_dependencies", {}).get(key)
        return previous == current if previous is not None else prior["basis"] == receipt["basis"]
    previous = next((r for r in candidates if r is case.runtime["command_protocol"].get("review_memory", {}).get(role)), None)
    shown, reused, changed, refs, deltas = {}, [], [], set(), []
    execution_rechecks = {}
    eligible = {"arguments", "observations", "execution_context", "citations", "assembled_answer",
                "evidence_followup", "candidate_overview", "pending_experiments", "comparison_options", "worker_interpretations", "execution_inventory"}
    if task_context:
        eligible.update(("tool_catalog", "ranking", "run_limits", "mechanism_pool"))
    focus_refs = set()
    for key, value in sections.items():
        same = next((r for r in candidates if key in entries[r["ref"]] and entries[r["ref"]][key][0] == value
                     and reusable(r, key)), None)
        prior = same or next((r for r in candidates if key in entries[r["ref"]]), None)
        if prior and not reusable(prior, key):
            before_versions = prior.get("section_dependencies", {}).get(key, {}).get("execution_versions", {})
            after_versions = receipt["section_dependencies"].get(key, {}).get("execution_versions", {})
            affected = tuple(sorted(cid for cid in before_versions.keys() | after_versions.keys()
                                    if before_versions.get(cid) != after_versions.get(cid)))
            if affected:
                execution_rechecks.setdefault(affected, []).append(key)
            prior = None
        if same is None:
            changed.append(key)
            if key.startswith("arguments:"):
                focus_refs.update(value.get("refs", []) + value.get("limitation_refs", []))
            elif key.startswith("observations:"):
                focus_refs.add(key.split(":", 1)[1])
        packed = None
        if prior and key.split(":", 1)[0] in eligible and json_bytes(value) > 512:
            before, path = entries[prior["ref"]][key]
            packed = packed_change(prior, path, before, value)
            if same:
                reused.append(key)
            elif packed:
                deltas.append(key)
            if packed:
                refs.add(prior["ref"])
        if prior and key == "computational_review":
            packed = dict(value)
            before, path = entries[prior["ref"]][key]
            # Keep the current stopping basis explicit. Every reused background
            # block is either exact unchanged content or a lossless delta.
            for field in ("executions", "comparisons", "question_evidence", "candidate_comparison", "investigation"):
                if field not in before or field not in value or json_bytes(value[field]) <= 512:
                    continue
                compact = packed_change(prior, pointer_child(path, field), before[field], value[field])
                if compact:
                    packed[field] = compact
                    refs.add(prior["ref"])
        # The literal current content is always retained in the new receipt.
        shown[key] = packed if packed is not None else value
    removed = sorted(set(entries[previous["ref"]]) - set(sections)) if previous else []
    if payload.get("delivery_task", {}).get("mode") == "proofread":
        # Omitted scientific sections belong to the unchanged source-bound
        # review anchor. A smaller task projection did not delete those records.
        removed = []
    current_issues = payload.get("review_workflow", {}).get("issues", [])
    for issue in current_issues:
        focus_refs.update(issue.get("source_refs", []))
    material = {"stage": payload["stage"], "role": role, "sections": shown,
                "reused_sections": reused, "changed_sections": changed, "removed_sections": removed,
                "delta_sections": deltas, "review_receipt_refs": sorted(refs),
                "review_focus_refs": sorted(focus_refs & index.available.keys()),
                "completed_issue_ids": [issue.id for issue in state(case).issues.values()
                    if issue.status == "resolved" and issue.id not in {i["id"] for i in current_issues}],
                "review_contract": "Review every changed passage and its effects on related claims. exact_delta contains "
                    "all changes against the named case-local receipt, not a summary. Text offsets refer to the old text. "
                    "Read unchanged context when needed; it remains available. Reused material is not approval. "
                    "Current ranking and open issues remain authoritative. Do not relist unchanged resolved findings."}
    if execution_rechecks:
        material["execution_rechecks"] = {
            "groups": [{"capability_ids": list(cids), "sections": keys} for cids, keys in execution_rechecks.items()],
            "reason": "These passages depend on execution histories that changed or lack a legacy version binding. "
                      "Recheck their factual absence/count limitations against execution_inventory. "
                      "This is not a finding that the passages are wrong or their mechanism support increased."}
    if previous:
        material["previous_feedback"] = {"ref": previous["ref"], "path": "/feedback",
            "meaning": "Historical prose, not the issue queue. Use current review_workflow issues."}
        refs.add(previous["ref"])
        material["review_receipt_refs"] = sorted(refs)
    if task_context:
        from mechcal.runtime.command_task_context import review_view
        material = review_view(material, sections)
    return material, receipt, previous["ref"] if previous else (next(iter(sorted(refs)), None))
