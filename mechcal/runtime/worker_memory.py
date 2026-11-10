"""Task-local retention of delivered facts, never inferred task completion."""

from mechcal.runtime.command_memory import (
    _read_value, digest, json_bytes, record_layout, resolve_pointer,
    remember_reads, remember_report, working_reads,
)
from mechcal.schemas.commands import ReadRecord


def receipt_handles(outcome):
    """Only values actually delivered in a receipt count as read."""
    if not outcome:
        return []
    handles = []
    if outcome.get("result_report"):
        handles = remember_report(handles, outcome["result_report"])
    for row in outcome.get("records", []):
        if "data" in row and row.get("selection", {}).get("view", "data") == "data":
            selection = ReadRecord(ref=row["ref"], **row["selection"])
            handles = remember_reads(handles, [selection], [row])
    for child in outcome.get("call_results", {}).values():
        handles.extend(receipt_handles(child))
    return handles


def coalesce(entries, available):
    """Keep one exact parent over overlapping children of the same source.

    Paged reads retain their original offsets. Equal values from different
    sources/paths are not aliases. Invalidated handles never cover valid ones.
    """
    valid = []
    for entry in entries:
        try:
            _read_value(entry, available)
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        valid.append(entry)
    retained = {}
    for position, entry in enumerate(valid):
        parents = [e for e in valid if e["ref"] == entry["ref"] and e["page"] is None
                   and (e["path"] == entry["path"] or entry["path"].startswith(e["path"] + "/"))]
        parent = min(parents, key=lambda e: len(e["path"])) if parents else entry
        key = (parent["ref"], parent["path"], str(parent["page"]))
        previous = retained.get(key, (0, {}))[1]
        retained[key] = (position, {**parent, "requested": bool(
            parent.get("requested") or entry.get("requested") or previous.get("requested"))})
    return [item for _, item in sorted(retained.values(), key=lambda pair: pair[0])]


def task_memory(task, available):
    return working_reads(coalesce(task.read_handles, available), available,
        latest_rows=receipt_handles(task.outcome) if task.exchange else (),
        active_refs=task.result_ids, explicit_handles=task.read_handles, context_field_bytes=384)


def remember_directories(previous, outcome, available):
    entries = {(r["ref"], r["path"], r["offset"], r["limit"]): r for r in previous}
    for row in outcome.get("records", []):
        if row.get("selection", {}).get("view") != "directory" or row["ref"] not in available:
            continue
        for path in row.get("data", {}):
            source = resolve_pointer(available[row["ref"]], path)
            page = row["selection"].get("page") or {"offset": 0, "limit": 16}
            entry = {"ref": row["ref"], "path": path, "source_version": digest(source), **page}
            key = (entry["ref"], path, page["offset"], page["limit"])
            entries.pop(key, None)
            entries[key] = entry
    return list(entries.values())


def directory_memory(task, available):
    """Navigation stays separate from evidence and is invalidated on change."""
    rows, omitted = [], 0
    for entry in reversed(task.directory_reads):
        try:
            source = resolve_pointer(available[entry["ref"]], entry["path"])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if digest(source) != entry["source_version"]:
            continue
        layout = record_layout(source, entry["path"], offset=entry["offset"], limit=entry["limit"])
        row = {"ref": entry["ref"], "path": entry["path"],
               "children": [[child["path"], child["type"], child["bytes"]] for child in layout["children"]],
               "offset": layout["offset"], "next_offset": layout["next_offset"],
               "total_children": layout["total_children"]}
        if json_bytes(rows + [row]) <= 3000:
            rows.append(row)
        else:
            omitted += 1
    return {"directories": rows, "omitted_directories": omitted,
            "columns": ["path", "type", "bytes"],
            "policy": "Already inspected directories, not evidence values. Read a listed child directly. "
                      "Only the displayed page is listed; next_offset reaches more children."}


def progress_note(task):
    note = task.progress_note
    if not note or note["request_id"] == (task.exchange or {}).get("request_id"):
        return None  # Already in the latest assistant message.
    text = note["text"]
    return {**note, "text": text[:2000], "truncated": len(text) > 2000,
            "policy": "Your last public work note, not a verified fact or a new instruction. "
                      "It may be revised by later results. It cannot be cited as evidence."}
