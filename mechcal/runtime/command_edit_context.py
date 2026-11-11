"""Versioned editing material, separate from the scientific observation cache."""

from mechcal.runtime.command_memory import digest, json_bytes, resolve_pointer, pointer_child
from mechcal.runtime.command_operations import protocol, record_index
from mechcal.runtime.command_query_state import QueryReceipt
from mechcal.runtime.command_views import ANSWER_REF

EDIT_BYTES = 12000


def correction_context(case, *, index=None):
    """Exact editable issue targets; no scientific interpretation or auto-repair."""
    from mechcal.runtime.command_review_workflow import issue_view
    index = record_index(case, index)
    targets, deferred, seen = [], [], set()
    issues = issue_view(case, index=index)
    for issue in issues:
        ref = issue["target"]
        if ref == ANSWER_REF or ref not in index.available:
            continue  # Final prose has its own version-bound edit_context.
        focus = issue.get("focus")
        if focus:
            paths = [focus["path"]]
        elif ref == "followup:state":
            paths = [pointer_child("/questions", key) + "/value"
                     for key in index.available[ref].get("questions", {})]
        else:
            paths = [""]
        for path in paths:
            if (ref, path) in seen:
                continue
            seen.add((ref, path))
            try:
                value = resolve_pointer(index.available[ref], path)
            except (KeyError, IndexError, TypeError, ValueError):
                deferred.append({"ref": ref, "path": path, "visibility": "target_changed_or_missing"})
                continue
            row = {"ref": ref, "path": path, "sha256": digest(value), "data": value}
            if json_bytes(targets + [row]) <= 6000:
                targets.append(row)
            else:
                deferred.append({"ref": ref, "path": path, "bytes": json_bytes(value),
                                 "visibility": "stored_not_shown"})
    result = {"targets": targets, "deferred": deferred,
            "policy": "Current issue targets, not evidence. Use the existing typed edit operations. "
                      "Unrelated history is not a new correction task."}
    # Expose actual argument-to-delivery links, not number matching across prose.
    # Predictions are regenerated from arguments, while final prose is editable.
    p = protocol(case)
    related = []
    for issue in issues:
        target = issue["target"]
        if issue["category"] != "fact" or target not in p.get("arguments", {}):
            continue
        labels = sorted(label for label, row in p.get("ranking", {}).items()
                        if target in row.get("argument_ids", []))
        row = {"issue_id": issue["id"], "argument_id": target, "linked_ranking_labels": labels}
        if labels and p.get("final_draft") and ANSWER_REF in index.available:
            row["prose_to_check"] = {"ref": ANSWER_REF, "paths": ["/answer", "/limitations"],
                                     "draft_id": p["final_draft"]["draft_id"]}
        related.append(row)
    if related:
        result["related_locations"] = related
        result["related_policy"] = (
            "These are stored dependency links, not additional findings or automatic edits. "
            "Corrected arguments regenerate their linked prediction evidence. Check whether final prose "
            "repeats the same claim before amending it; a shared number alone is not identity. "
            "Keep the current review and atomic argument/ranking submission rules.")
    return result


def editable_fields(ref, path, value, draft_id):
    """Normalize only the registered representations of this exact draft."""
    if ref == draft_id:
        if not path:
            value, path = value["command"], ""
        elif path in {"/command", "/content"}:
            path = ""
        elif path.startswith(("/command/", "/content/")):
            path = path.split("/", 2)[2]
            path = "/" + path
        else:
            return
    elif ref != ANSWER_REF:
        return
    if not path:
        for name in ("answer", "limitations"):
            if name in value:
                yield "/" + name, value[name]
    elif path == "/answer" or path == "/limitations" or path.startswith("/limitations/"):
        yield path, value


def delivered_fields(outcome, draft_id):
    if outcome.get("status") == "read_batch":
        for child in outcome["call_results"].values():
            yield from delivered_fields(child, draft_id)
        return
    for row in outcome.get("records", []):
        selection = row.get("selection", {})
        if "data" not in row or selection.get("view", row.get("view")) != "data":
            continue
        values = row["data"].items() if selection.get("paths") else [("", row["data"])]
        for path, value in values:
            for canonical, data in editable_fields(row["ref"], path, value, draft_id):
                yield canonical, selection.get("page"), digest(data)


def edit_context(case, *, index=None, native_replay=False):
    p = protocol(case)
    draft = p.get("final_draft")
    if not draft or ANSWER_REF not in record_index(case, index).available:
        return None
    index = record_index(case, index)
    draft_id = draft["draft_id"]
    candidates = {}
    from mechcal.runtime.command_review_workflow import issue_view
    issues = [i for i in issue_view(case, index=index) if i["target"] == ANSWER_REF]
    # Attach the current repair target, not another retrieval instruction. A
    # broad issue requires both fields; precise issue paths narrow the view.
    paths = set()
    for issue in issues:
        focus = issue.get("focus")
        paths.update([focus["path"]] if focus and (
            focus["path"] == "/answer" or focus["path"].startswith("/limitations")) else ["/answer", "/limitations"])
    for path in sorted(paths):
        try:
            data = resolve_pointer(index.available[ANSWER_REF], path)
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        candidates[(path, digest(None), digest(data))] = {
            "path": path, "page": None, "sha256": digest(data), "bytes": json_bytes(data), "data": data}
    # Earlier checkpoints may have version-bound reads only in the navigation
    # table. Unversioned legacy receipts are never treated as current edit text.
    for raw in p.get("edit_reads", p.get("query_receipts", {})).values():
        receipt = QueryReceipt.model_validate(raw)
        if receipt.case_id != case.case_id:
            raise RuntimeError("Editing receipt belongs to another case")
        query = receipt.accepted_selection
        if (receipt.role != "planner" or receipt.draft_id != draft_id or query is None
                or query.view != "data" or query.ref not in {ANSWER_REF, draft_id}
                or receipt.source_version != index.version(query.ref)):
            continue
        page = query.page.model_dump() if query.page else None
        for path in query.paths or [""]:
            value = resolve_pointer(index.available[query.ref], path)
            if page:
                value = value[page["offset"]:page["offset"] + page["limit"]]
            for canonical, data in editable_fields(query.ref, path, value, draft_id):
                key = (canonical, digest(page), digest(data))
                candidates[key] = {"path": canonical, "page": page, "sha256": digest(data),
                                   "bytes": json_bytes(data), "data": data}
    exchange = p.get("latest_exchange") or {}
    replayed = native_replay and exchange and p.get("outcome_request_id") == exchange.get("request_id")
    delivered = {(path, digest(page), version) for path, page, version in
                 delivered_fields(p.get("outcome") or {}, draft_id)} if replayed else set()
    retained, deferred, whole = [], [], []
    used = 0
    # A parent read covers its children. Equal text in unrelated records does not.
    for key, row in sorted(candidates.items(), key=lambda item: len(item[0][0])):
        if any(row["path"] == path or row["path"].startswith(path + "/") for path in whole):
            continue
        can_retain = used + json_bytes(row) <= EDIT_BYTES
        if not can_retain and key not in delivered:
            deferred.append({k: v for k, v in row.items() if k != "data"})
            continue
        if can_retain:
            used += json_bytes(row)
        if row["page"] is None:
            whole.append(row["path"])
        if key in delivered:
            row = {k: v for k, v in row.items() if k != "data"}
            row["location"] = "latest_tool_result"
            row["retained_for_next_request"] = can_retain
        else:
            row["location"] = "edit_context"
        retained.append(row)
    fields = [{"path": "/" + name, "bytes": json_bytes(draft["command"].get(name, [])),
               "present_in_request": "/" + name in whole}
              for name in ("answer", "limitations")]
    return {"ref": ANSWER_REF, "draft_id": draft_id,
            "write_contract": {"operation": "amend_final", "draft_id": draft_id,
                               "fields": "Use path and complete replacement value; no old_text required."},
            "fields": fields, "retained": retained, "deferred": deferred,
            "read": {"ref": ANSWER_REF, "paths": [f["path"] for f in fields if not f["present_in_request"]],
                     "view": "data", "page": None} if not all(f["present_in_request"] for f in fields) else None,
            "policy": "The only editing location is delivery:answer. draft_id identifies its version, not another "
                      "recommended read location. retained is exact text in this request or its latest tool result. "
                      "deferred is NOT in context. Directory reads do not load text. New draft versions invalidate "
                      "these read receipts; prior existence never implies current visibility. This is prose, not evidence."}
