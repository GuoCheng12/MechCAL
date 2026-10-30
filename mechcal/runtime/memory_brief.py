"""Exact small-value views. No model summaries or scientific relevance selection."""

from collections import deque


VALUE_CHARS = 256


def represented(record, path, value):
    from mechcal.runtime.case_memory import pointer

    fields = dict(record.get("selected_fields", {}))
    fields.update({k if k.startswith("/") else "/" + k: v for k, v in record.get("source_context", {}).items()})
    for parent, metadata in record.get("field_context", {}).items():
        fields.update({parent + "/" + k: v for k, v in metadata.items()})
    for root, stored in fields.items():
        if path == root and stored == value:
            return True
        if path.startswith(root + "/"):
            try:
                if pointer(stored, path[len(root):]) == value:
                    return True
            except ValueError:
                pass
    return False


def brief_roots(original, paths):
    if paths:
        return list(dict.fromkeys(paths))
    if isinstance(original.get("metrics"), dict):
        return ["/metrics"]
    if isinstance(original.get("structured_results"), dict):
        return ["/structured_results"]
    return [""]


def brief_entries(original, paths):
    from mechcal.runtime.case_memory import pointer, serialized

    queue = deque()
    for path in brief_roots(original, paths):
        try:
            queue.append((path, pointer(original, path) if path else original))
        except ValueError as exc:
            yield path, "error", str(exc)
    while queue:
        path, value = queue.popleft()
        if not path and value == {}:
            continue
        if path and len(serialized(value)) <= VALUE_CHARS:
            yield path, "value", value
        elif isinstance(value, dict) and value:
            queue.extend((path + "/" + str(k).replace("~", "~0").replace("/", "~1"), v)
                         for k, v in sorted(value.items()))
        else:
            # Do not enumerate a large atomic/transition array or slice a long string.
            yield path, "bulk", value


def brief_record(original, paths, *, offset, budget):
    from mechcal.runtime.case_memory import serialized
    from mechcal.runtime.memory_views import distinct_field_metadata, selected_record

    result = {
        "view": "brief", "complete_record": False, "selected_fields": {},
        "source_context": selected_record(original, [], focused=True)["source_context"],
        "roots": brief_roots(original, paths), "bulk_paths": [], "next_offset": None,
        "notice": "Exact small fields only. bulk_paths are UNREAD, not absent. "
                  "Use auto/full with paths for bulk data; offset continues this brief view. "
                  "Conditions are attached to values; omitted data cannot support a claim.",
    }
    if len(serialized(result)) > budget - 40:
        return {"view": "brief", "error": "brief_context_exceeds_budget", "selected_fields": {},
                "context_not_loaded": True, "roots": result["roots"], "next_offset": offset,
                "notice": "Conditions exceed the response allowance. Read fewer records or narrower paths."}
    values = []
    consumed = 0
    entry_count = 0
    for index, (path, kind, value) in enumerate(brief_entries(original, paths)):
        entry_count = index + 1
        if index < offset:
            continue
        candidate = {**result, "bulk_paths": list(result["bulk_paths"])}
        if kind == "bulk":
            if not represented(result, path, value):
                candidate["bulk_paths"].append(path)
        elif kind == "error":
            candidate["field_errors"] = {**result.get("field_errors", {}), path: value}
        else:
            fields = distinct_field_metadata(selected_record(original, [*values, path], focused=True, ancestor_context=True))
            candidate.update({k: v for k, v in fields.items() if k != "notice"})
        if len(serialized(candidate)) > budget - 40:
            result["next_offset"] = index
            if consumed == 0:
                result["error"] = "brief_entry_exceeds_budget"
                result["blocked_path"] = path
                result["notice"] += " Read fewer records or this path directly; the cursor did not advance."
            break
        result = candidate
        consumed += 1
        if kind == "value":
            values.append(path)
    if offset > entry_count:
        result["error"] = "offset_beyond_brief"
        result["entry_count"] = entry_count
    return result
