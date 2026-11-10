"""Bounded views of exact stored data. Directories are not scientific evidence."""

from collections import deque
import json


CONTEXT_FIELDS = (
    "evidence_id", "result_id", "capability_id", "status", "claim", "basis", "context",
    "limits", "limitations", "failure", "delivery", "requested_parameters",
    "calculation_identity", "resolved_source_artifact_id", "input_refs", "artifact_refs",
)
CALCULATION_PATHS = (
    "/structured_results/limitations", "/structured_results/method_provenance",
    "/structured_results/bundle_status", "/structured_results/missing_deliverables",
    "/structured_results/parsed_observables/effective_parameters",
    "/structured_results/parsed_observables/molecular_context",
    "/structured_results/parsed_observables/transition_conventions",
)
FIELD_CONTEXT_KEYS = {
    "unit", "units", "state", "state_index", "root", "method", "basis", "charge", "multiplicity",
    "solvent", "temperature", "geometry_id", "source_artifact_id", "conditions", "limits",
    "limitations", "status", "failure", "effective_parameters", "requested_parameters",
}


def source_context(original):
    from mechcal.runtime.case_memory import pointer
    from mechcal.runtime.observation_context import METRIC_CONTEXT_KEYS

    context = {key: original[key] for key in CONTEXT_FIELDS if key in original}
    for path in CALCULATION_PATHS:
        try:
            context[path] = pointer(original, path)
        except ValueError:
            pass
    metrics = original.get("metrics")
    if isinstance(metrics, dict):
        for key in sorted(METRIC_CONTEXT_KEYS):
            if key in metrics:
                context["/metrics/" + key] = metrics[key]
    return context


def field_rows(value, prefix="", *, max_depth=3):
    """Breadth first: expose nested field names, never enumerate numeric tables."""
    queue = deque([(prefix, value, 0)])
    while queue:
        path, item, depth = queue.popleft()
        if path:
            if isinstance(item, dict):
                row = {"path": path, "type": "object", "fields": len(item)}
            elif isinstance(item, list):
                row = {"path": path, "type": "array", "items": len(item)}
            else:
                row = {"path": path, "type": "value"}
            yield row
        if isinstance(item, dict) and depth < max_depth:
            queue.extend((f"{path}/{str(k).replace('~', '~0').replace('/', '~1')}", v, depth + 1)
                         for k, v in item.items())


def outline_record(original, paths, *, offset, budget, shallow=False):
    from mechcal.runtime.case_memory import pointer, serialized

    roots, errors = [], {}
    for path in paths:
        try:
            roots.append((path, pointer(original, path)))
        except ValueError as exc:
            errors[path] = {"error": str(exc)}
    if not paths:
        roots = [("", original)]
    fields = (row for path, value in roots for row in field_rows(value, path, max_depth=1 if shallow else 3))
    result = {
        "view": "outline", "preview_only": True, "fields": [], "offset": offset,
        "next_offset": None, "source_context": source_context(original),
        "notice": "Field directory, not retrieved values. Use paths to read exact fields. "
                  "For a nested directory use view=outline with paths. For large values use "
                  "view=full and start/next_start; partial fragments are not complete evidence.",
    }
    if errors:
        result["field_errors"] = errors
    if len(serialized(result)) > budget // 2:
        result["context_paths"] = [key if key.startswith("/") else "/" + key
                                   for key in result.pop("source_context")]
        result["context_not_loaded"] = True
    for index, row in enumerate(fields):
        if index < offset:
            continue
        result["fields"].append(row)
        if len(serialized(result)) > budget:
            result["fields"].pop()
            result["next_offset"] = index
            break
    return result


def selected_record(original, paths, *, focused, ancestor_context=False):
    from mechcal.runtime.case_memory import pointer

    selected, field_context = {}, {}
    for path in paths:
        try:
            selected[path] = pointer(original, path)
            parent_path = path.rsplit("/", 1)[0]
            while focused:
                parent = pointer(original, parent_path) if parent_path else original
                if isinstance(parent, dict):
                    metadata = {key: value for key, value in parent.items() if key in FIELD_CONTEXT_KEYS}
                    if metadata:
                        field_context[parent_path] = metadata
                if not ancestor_context or not parent_path:
                    break
                parent_path = parent_path.rsplit("/", 1)[0]
        except ValueError as exc:
            selected[path] = {"error": str(exc)}
    result = {
        "selected_fields": selected,
        "source_context": source_context(original) if focused else {
            k: original[k] for k in ("evidence_id", "result_id", "claim", "status", "limitations")
            if k in original
        },
        "notice": "Selected fields only. Included context is exact stored metadata, not proof of "
                  "comparability. Retrieve additional units, state identity or conditions when needed.",
    }
    for parent, fields in list(field_context.items()):
        field_context[parent] = {k: v for k, v in fields.items()
                                 if f"{parent}/{k}" not in result["source_context"]
                                 or result["source_context"][f"{parent}/{k}"] != v}
        if not field_context[parent]:
            del field_context[parent]
    if field_context:
        result["field_context"] = field_context
    return result


def distinct_field_metadata(record):
    """Keep conditions once when the exact same value is already a selected field."""
    selected = record["selected_fields"]
    def is_selected(path, value):
        return path in selected and selected[path] == value
    result = {**record, "source_context": {
        key: value for key, value in record["source_context"].items()
        if not is_selected(key if key.startswith("/") else "/" + key, value)
    }}
    if "field_context" in record:
        source_paths = {k if k.startswith("/") else "/" + k: v
                        for k, v in result["source_context"].items()}
        result["field_context"] = {
            parent: {key: value for key, value in fields.items()
                     if not is_selected(parent + "/" + key, value)
                     and not (parent + "/" + key in source_paths and source_paths[parent + "/" + key] == value)}
            for parent, fields in record["field_context"].items()
        }
        result["field_context"] = {p: fields for p, fields in result["field_context"].items() if fields}
        if not result["field_context"]:
            result.pop("field_context")
    return result


def bounded_selected_fields(original, paths, budget):
    """Return whole selected fields, not an arbitrary JSON prefix across fields."""
    from mechcal.runtime.case_memory import serialized

    result = selected_record(original, paths, focused=True)
    values = result["selected_fields"]
    result["selected_fields"] = dict(values)
    result["deferred_paths"] = []
    result["deferred_fields"] = []
    result["directory_incomplete"] = False
    result["notice"] += " deferred_paths are NOT loaded. deferred_fields lists narrower paths, not values. "
    result["notice"] += "Read selected child fields, or one full record/field with pagination."
    # Defer whole large fields first. A field's value never affects its priority.
    for path in sorted(values, key=lambda p: len(serialized(values[p])), reverse=True):
        if len(serialized(distinct_field_metadata(result))) <= budget:
            break
        result["selected_fields"].pop(path)
        result["deferred_paths"].append(path)
    result = distinct_field_metadata(result)
    if len(serialized(result)) > budget:
        return outline_record(original, paths, offset=0, budget=budget)
    if not result["deferred_paths"]:
        result.pop("deferred_paths")
        result.pop("deferred_fields")
        result.pop("directory_incomplete")
    else:
        # Expose only structural paths/types, never choose scientifically favorable values.
        for path in result["deferred_paths"]:
            for row in field_rows(values[path], path, max_depth=1):
                result["deferred_fields"].append(row)
                if len(serialized(result)) > budget:
                    result["deferred_fields"].pop()
                    result["directory_incomplete"] = True
                    break
            if result["directory_incomplete"]:
                break
    return result


def allocate_read_budgets(demands, total=12000):
    """Redistribute unused space from small records; never enlarge the batch budget."""
    budgets = [0] * len(demands)
    pending = set(range(len(demands)))
    while pending:
        share = total // len(pending)
        small = {i for i in pending if demands[i] <= share}
        if not small:
            for i in sorted(pending):
                budgets[i] = share
            break
        for i in small:
            budgets[i] = demands[i]
            total -= demands[i]
        pending -= small
    return budgets


def render_record(rid, kind, body, request, *, focused, budget=None, brief_auto=False):
    from mechcal.runtime.case_memory import serialized
    from mechcal.runtime.memory_brief import brief_record

    share = budget if budget is not None else 12000 // len(request.record_ids)
    original = json.loads(body)
    if request.view == "findings":
        from mechcal.runtime.memory_findings import findings_record
        return {"record_id": rid, "kind": kind, "record": findings_record(
            original, offset=request.offset, budget=share)}
    if request.view == "brief":
        return {"record_id": rid, "kind": kind, "record": brief_record(
            original, request.paths, offset=request.offset, budget=share)}
    if request.paths:
        body = serialized(selected_record(original, request.paths, focused=focused))
    if (focused and brief_auto and kind in {"observation", "execution"}
            and request.view == "auto" and not request.start and len(body) > share):
        return {"record_id": rid, "kind": kind, "record": brief_record(
            original, request.paths, offset=request.offset, budget=share)}
    batch = len(request.record_ids) > 1
    if (focused and request.paths and (len(request.paths) > 1 or batch) and not request.start
            and request.view != "outline" and len(body) > share):
        return {"record_id": rid, "kind": kind, "record": bounded_selected_fields(original, request.paths, share)}
    if request.view == "outline" or (
        focused and (request.view == "auto" or batch) and not request.start and len(body) > share
    ):
        return {"record_id": rid, "kind": kind, "record": outline_record(
            original, request.paths, offset=request.offset, budget=share,
            shallow=kind == "context" and not request.paths,
        )}
    if request.start > len(body):
        return {"record_id": rid, "error": "start_beyond_record", "length": len(body)}
    end = min(len(body), request.start + share)
    item = {"record_id": rid, "kind": kind, "total_characters": len(body),
            "start": request.start, "next_start": end if end < len(body) else None}
    if request.start == 0 and end == len(body):
        item["record"] = json.loads(body)
    else:
        item["json_fragment"] = body[request.start:end]
        item["notice"] = "Partial serialized record, not a complete result. Read next_start to recover more."
    return item


def retrieved_working_set(responses):
    """Drop superseded directories/previews, never incomplete or distinct evidence."""
    covered, selected = set(), {}
    kept = []
    for response in reversed(responses):
        if "records" not in response:
            kept.append(response)
            continue
        rows = []
        for row in response["records"]:
            rid = row["record_id"]
            record = row.get("record", {})
            is_directory = record.get("view") == "outline" or row.get("preview_only")
            if not row.get("error") and "json_fragment" not in row:
                if rid in covered or (is_directory and rid in selected):
                    continue
                if "selected_fields" in record:
                    fields = record["selected_fields"]
                    known = selected.setdefault(rid, {})
                    remaining = {key: value for key, value in fields.items()
                                 if key not in known or known[key] != value}
                    known.update({key: value for key, value in fields.items()
                                  if record.get("view") in {"brief", "findings"} or not (isinstance(value, dict) and set(value) == {"error"})})
                    deferred = [path for path in record.get("deferred_paths", []) if path not in known]
                    # A brief projection also carries unread bulk paths and a cursor.
                    if not remaining and not deferred and record.get("view") not in {"brief", "findings"}:
                        continue
                    if len(remaining) != len(fields) or deferred != record.get("deferred_paths", []):
                        record = {**record, "selected_fields": remaining}
                        if deferred:
                            record["deferred_paths"] = deferred
                            if "deferred_fields" in record:
                                record["deferred_fields"] = [r for r in record["deferred_fields"] if any(
                                    r["path"] == p or r["path"].startswith(p + "/") for p in deferred)]
                        else:
                            record.pop("deferred_paths", None)
                            record.pop("deferred_fields", None)
                            record.pop("directory_incomplete", None)
                        row = {**row, "record": record}
                elif "record" in row and not is_directory and record.get("view") not in {"brief", "findings"}:
                    covered.add(rid)
            rows.append(row)
        if rows or response.get("error") or response.get("next_offset") is not None:
            kept.append({**response, "records": rows})
    return list(reversed(kept))


def lookup_memory(responses):
    """Dependency lookup sees the latest reply; earlier values wait for the decision."""
    receipts = []
    for response in responses[:-1]:
        receipt = {"action": response.get("request", {}).get("action")}
        for key in ("error", "status", "next_offset", "total"):
            if key in response:
                receipt[key] = response[key]
        records = []
        for row in response.get("records", []):
            item = {k: row[k] for k in ("record_id", "kind", "error", "start", "next_start", "preview_only") if k in row}
            record = row.get("record", {})
            if "selected_fields" in record:
                item["loaded_paths"] = [p for p, v in record["selected_fields"].items()
                                        if record.get("view") == "brief" or not (isinstance(v, dict) and set(v) == {"error"})]
                errors = {p: v for p, v in record["selected_fields"].items()
                          if record.get("view") != "brief" and isinstance(v, dict) and set(v) == {"error"}}
                if errors:
                    item["field_errors"] = errors
            for key in ("view", "next_offset", "field_errors", "deferred_paths", "bulk_paths", "blocked_path", "error"):
                if key in record:
                    item[key] = record[key]
            if "json_fragment" in row:
                item["partial"] = True
            records.append(item)
        if records:
            receipt["records"] = records
        if "artifacts" in response:
            receipt["artifact_ids"] = [a["artifact_id"] for a in response["artifacts"]]
        receipts.append(receipt)
    return responses[-1:], receipts
