"""Bounded, source-addressable views. Raw observations are never rewritten."""

from copy import deepcopy
import hashlib
import json


READ_BYTES = 24000
REPORT_BYTES = 12000
WORKING_BYTES = 6000
CONTEXT_FIELDS = {"method", "method_provenance", "state", "environment", "limitations", "limits",
                  "requested_parameters", "effective_parameters", "molecular_context", "missing_descriptors",
                  "missing_deliverables", "boundary_only"}
ROUTING_FIELDS = {"selected_route", "tool_args", "result_name", "runtime_status", "failure_kind"}
DERIVATION_FIELDS = {"independent_observation", "derived_from", "calculation_scope"}


def json_bytes(value):
    return len(json.dumps(value).encode())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def pointer_child(parent, key):
    return parent + "/" + str(key).replace("~", "~0").replace("/", "~1")


def read_selection(value, path):
    """A small exact read or metadata navigation, never an implicit data sample."""
    if json_bytes(value) <= 2048:
        return {"paths": [path] if path else [], "view": "data", "page": None}
    if isinstance(value, list) and value and json_bytes(value[0]) <= 2048:
        return {"paths": [path] if path else [], "view": "data", "page": {"offset": 0, "limit": 1}}
    return {"paths": [path] if path else [], "view": "directory", "page": None}


def list_navigation(value, path):
    """Describe the first row's shape without treating it as representative evidence."""
    navigation = {"rows": len(value), "read": read_selection(value, path)}
    if value and isinstance(value[0], dict):
        row_path = pointer_child(path, 0)
        navigation["first_row_fields"] = [
            {"path": pointer_child(row_path, key), "bytes": json_bytes(child),
             "type": type(child).__name__, "read": read_selection(child, pointer_child(row_path, key))}
            for key, child in list(value[0].items())[:8]]
        navigation["row_hint"] = "These paths address row 0 only. Other rows may have different fields; inspect their directory."
    return navigation


def record_layout(value, path="", *, limit=16, offset=0):
    keys = list(value) if isinstance(value, dict) else list(range(len(value))) if isinstance(value, list) else []
    end = min(offset + limit, len(keys))
    shown = keys[offset:end]
    layout = {"path": path, "bytes": json_bytes(value), "type": type(value).__name__,
            "children": [{"path": pointer_child(path, key), "bytes": json_bytes(value[key]),
                          "type": type(value[key]).__name__,
                          "read": read_selection(value[key], pointer_child(path, key))} for key in shown],
            "total_children": len(keys), "offset": offset,
            "next_offset": end if end < len(keys) else None,
            "omitted_children": len(keys) - len(shown)}
    if isinstance(value, list):
        layout["navigation"] = list_navigation(value, path)
    return layout


def resolve_pointer(row, path):
    value = row
    for key in path.split("/")[1:] if path else []:
        decoded = key.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not decoded.isdigit() or str(int(decoded)) != decoded:
                raise ValueError("Array index must be a canonical nonnegative integer")
            value = value[int(decoded)]
        else:
            value = value[decoded]
    return value


def read_directory(value):
    """Discover actual fields without sampling values or guessing their names."""
    layout = record_layout(value, limit=32)
    compact = {"bytes": layout["bytes"], "paths": [child["path"] for child in layout["children"]],
               "next_offset": layout["next_offset"], "containers": [],
               "directory_hint": "Exact root paths; directory reads provide types, sizes and remaining fields."}
    for key in ("structured_results", "metrics", "metadata", "tool_args", "command"):
        child = value.get(key) if isinstance(value, dict) else None
        if not isinstance(child, dict):
            continue
        directory = record_layout(child, pointer_child("", key), limit=16)
        item = {"path": directory["path"], "paths": [entry["path"] for entry in directory["children"]],
                "next_offset": directory["next_offset"]}
        if json_bytes({**compact, "containers": compact["containers"] + [item]}) <= 3000:
            compact["containers"].append(item)
    return compact


def source_provenance(result):
    """Exact field ownership and reported derivation, not a scientific verdict.

    Field lists are complete only for their named container. Missing flags do
    not certify independence; equal numbers never create dependency links.
    """
    structured = result.get("structured_results", {})
    fields, derivation, values_to_show, declared = [], [], [], []
    containers = [("/structured_results", structured)] + [
        ("/structured_results/" + key, structured[key])
        for key in ("metrics", "parsed_observables") if isinstance(structured.get(key), dict)]
    for path, values in containers:
        if path != "/structured_results":
            shown = []
            for name in values:
                if json_bytes(shown + [name]) > 256:
                    break
                shown.append(name)
            fields.append({"path": path, "names": shown, "complete": len(shown) == len(values)})
        for name in sorted(DERIVATION_FIELDS & values.keys()):
            value = values[name]
            item = {"path": pointer_child(path, name), "bytes": json_bytes(value), "deferred": True}
            derivation.append(item)
            values_to_show.append(value)
            if name == "independent_observation":
                declared.append(value)
    independence = ("declared_not_independent" if any(v is False for v in declared) else
                    "declared_independent_unverified" if any(v is True for v in declared) else "not_recorded")
    result = {"field_ownership": fields, "derivation": derivation, "independence": independence}
    for item, value in zip(derivation, values_to_show):
        if json_bytes(value) > 2048:
            continue
        item.pop("deferred")
        item["data"] = value
        if json_bytes(result) > 2048:
            item.pop("data")
            item["deferred"] = True
    return result


def result_report(ref, result):
    """Expose small exact fields once, with explicit pointers for everything else.

    Large tables are not ranked or interpreted here. Selection is by size, with
    method/boundary fields first; only Worker/Planner make scientific judgments.
    """
    report = {"source_result_id": ref, "tool": result.get("capability_id"),
              "status": result.get("status"), "source_bytes": json_bytes(result),
              "fields": [], "deferred": [], "deferred_count": 0,
              "provenance": source_provenance(result),
              "policy": "Exact selected fields, not complete data. Deferred fields are not absent or zero. "
                        "Use read_evidence on source_result_id, view=directory or data. "
                        "Failure diagnostics are not positive mechanism evidence."}
    contextual = CONTEXT_FIELDS

    def candidates(value, path, depth=0):
        if json_bytes(value) <= 2048 or not isinstance(value, dict) or depth >= 4:
            yield path, value
        else:
            for key in sorted(value, key=lambda k: k not in contextual):
                yield from candidates(value[key], pointer_child(path, key), depth + 1)

    sources = [("/" + key, result[key]) for key in ("summary", "failure") if key in result]
    # Evidence units repeat the numeric payload, but may carry unique boundaries.
    sources.extend((f"/evidence_units/{i}/{key}", unit[key])
                   for i, unit in enumerate(result.get("evidence_units", []))
                   for key in ("context", "limits") if unit.get(key))
    structured = result.get("structured_results", {})
    sources.extend((pointer_child("/structured_results", key), structured[key])
                   for key in sorted(structured, key=lambda k: k not in contextual))
    # Reserve space for directory metadata, without truncating any selected value.
    for parent, value in sources:
        for path, field in candidates(value, parent):
            entry = {"path": path, "value": field}
            if (json_bytes(field) <= 2048 and json_bytes(report["fields"] + [entry]) <= 8000
                    and json_bytes({**report, "fields": report["fields"] + [entry]}) <= REPORT_BYTES - 2100):
                report["fields"].append(entry)
            else:
                report["deferred_count"] += 1
                meta = {"path": path, "bytes": json_bytes(field), "type": type(field).__name__,
                        "read": read_selection(field, path)}
                if isinstance(field, list) and field and json_bytes(field[0]) > READ_BYTES:
                    meta["navigation"] = list_navigation(field, path)
                if (json_bytes(report["deferred"] + [meta]) <= 2000
                        and json_bytes({**report, "deferred": report["deferred"] + [meta]}) <= REPORT_BYTES):
                    report["deferred"].append(meta)
    return report


def result_index(ref, result):
    """Show sizes before a read, including the common observable container."""
    structured = result.get("structured_results", {})
    paths = ["/" + key for key in ("summary", "status", "failure") if key in result]
    paths += [pointer_child("/structured_results", key) for key in structured]
    entries = [{"path": path, "bytes": json_bytes(resolve_pointer(result, path))} for path in paths]
    parsed = structured.get("parsed_observables")
    if isinstance(parsed, dict):
        entries += record_layout(parsed, "/structured_results/parsed_observables")["children"]
    for entry in entries:
        value = resolve_pointer(result, entry["path"])
        if isinstance(value, list) and entry["bytes"] > READ_BYTES:
            entry["navigation"] = list_navigation(value, entry["path"])
    return {"ref": ref, "tool": result.get("capability_id"), "status": result.get("status"),
            "bytes": json_bytes(result), "report_ref": "report:" + ref,
            "provenance": source_provenance(result),
            "brief_read": {"ref": "report:" + ref, "paths": [], "view": "data", "page": None},
            "paths": paths, "directory": entries,
            "directory_hint": "Paths above belong to the raw result, not report_ref. "
                              "brief_read returns the report's fields/path/value entries. "
                              "view=directory lists any container; use page.offset for further entries."}


def remember_values(previous, values):
    """Persist source handles and fingerprints, not another copy of the large data."""
    entries = deepcopy(previous)

    def selections(value, path, page):
        if json_bytes(value) > 2048 and isinstance(value, (dict, list)):
            parts = value.items() if isinstance(value, dict) else enumerate(value, start=(page or {}).get("offset", 0))
            for key, child in parts:
                yield from selections(child, pointer_child(path, key), None)
        else:
            yield path, value, page

    for ref, path, value, page in values:
        for child_path, child, child_page in selections(value, path, page):
            prior = next((e for e in entries if (e["ref"], e["path"], e["page"]) ==
                          (ref, child_path, child_page)), {})
            item = {"ref": ref, "path": child_path, "page": child_page,
                    "sha256": digest(child), "bytes": json_bytes(child)}
            if prior.get("requested"):
                item["requested"] = True
            key = (ref, child_path, child_page)
            entries = [e for e in entries if (e["ref"], e["path"], e["page"]) != key]
            entries.append(item)
    return entries


def remember_reads(previous, queries, rows):
    added = remember_values([], (
        (query.ref, path, value, row.get("selection", {}).get("page", query.page.model_dump() if query.page else None))
        for query, row in zip(queries, rows) if query.view == "data"
        for path, value in (row["data"] if query.paths else {"": row["data"]}).items()))
    keys = {(e["ref"], e["path"], json.dumps(e["page"], sort_keys=True)) for e in added}
    return [e for e in previous if (e["ref"], e["path"], json.dumps(e["page"], sort_keys=True)) not in keys] + [
        {**e, "requested": True} for e in added]


def remember_report(previous, report):
    return remember_values(previous, ((report["source_result_id"], item["path"], item["value"], None)
                                      for item in report["fields"]))


def _read_value(entry, available):
    value = resolve_pointer(available[entry["ref"]], entry["path"])
    if entry["page"]:
        if not isinstance(value, list):
            raise ValueError("Paged source is no longer a list")
        offset, limit = entry["page"]["offset"], entry["page"]["limit"]
        value = value[offset:offset + limit]
    if digest(value) != entry["sha256"]:
        raise ValueError("Source changed since this read")
    return value


def _content_key(ref, path, value, page):
    # The structural adapter publishes metrics twice, once under the route name.
    # Collapse only that exact representation alias, never equal numbers or
    # dictionaries at unrelated scientific paths (e.g. two different states).
    if path == "/structured_results/structural_prior" and isinstance(value, dict) and len(value) == 1:
        value = next(iter(value.values()))
        path = "/structured_results/metrics"
    return ref, path, digest(value), json.dumps(page, sort_keys=True)


def _retention_tier(source, path):
    if source.get("status") == "failed":
        return 3
    parts = path.split("/")
    if len(parts) > 2 and parts[1] == "structured_results":
        return 3 if parts[2] in ROUTING_FIELDS else 2 if parts[2] in CONTEXT_FIELDS else 0
    if "source_result_id" in source or path == "/summary":
        return 2  # Interpretations/summary are not another observation.
    if path.startswith(("/failure", "/evidence_units/")):
        return 3
    return 1


def _source_context(source):
    fields = [{"path": pointer_child("/structured_results", key), "data": value}
              for key, value in source.get("structured_results", {}).items() if key in CONTEXT_FIELDS]
    fields += [{"path": f"/evidence_units/{i}/{key}", "data": unit[key]}
               for i, unit in enumerate(source.get("evidence_units", []))
               for key in ("context", "limits") if unit.get(key)]
    return fields


def working_reads(entries, available, *, active_refs=(), latest_rows=(), focus_refs=None, explicit_handles=(),
                  context_field_bytes=384):
    """Keep diverse exact observations with boundaries ahead of logs and prose.

    Selection is by record type, explicit reads and source coverage, not mechanism
    labels or support scores. No scientific inference or source mutation occurs.
    """
    latest = set()
    for entry in latest_rows:
        try:
            latest.add(_content_key(entry["ref"], entry["path"], _read_value(entry, available), entry["page"]))
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    active = set(active_refs)
    for ref in active_refs:
        source = available.get(ref, {})
        active.update(r for r in (source.get("source_report_id"), source.get("source_result_id")) if r)
    candidates = []
    for recency, entry in enumerate(reversed(entries)):
        source = available.get(entry["ref"])
        if not isinstance(source, dict):
            continue
        try:
            value = _read_value(entry, available)
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        key = _content_key(entry["ref"], entry["path"], value, entry["page"])
        if key in latest:
            continue
        candidates.append((entry, source, value, key, recency))
    # Prefer the short canonical path for exact aliases; all aliases stay readable.
    candidates.sort(key=lambda row: (len(row[0]["path"]), row[4]))
    unique = {}
    for row in candidates:
        unique.setdefault(row[3], row)
    ordered = sorted(unique.values(), key=lambda row: (
        _retention_tier(row[1], row[0]["path"]), not row[0].get("requested", False), row[4]))
    depth = {}
    fair = []
    for row in ordered:
        entry, source, _, _, recency = row
        tier = _retention_tier(source, entry["path"])
        group = (tier, entry["ref"])
        slot = depth.get(group, 0)
        depth[group] = slot + 1
        fair.append(((tier, slot, entry["ref"] not in active, not entry.get("requested", False), recency), row))
    focused = set(focus_refs) if focus_refs is not None else None
    explicit = {(h["ref"], h["path"], str(h["page"])) for h in explicit_handles}
    retained, index, contextualized = [], [], set()
    def priority(item):
        order, (entry, _, _, _, recency) = item
        requested = (entry["ref"], entry["path"], str(entry["page"])) in explicit
        return (not requested, recency if requested else 0, order)

    for _, (entry, source, value, _, _) in sorted(fair, key=priority):
        requested_here = (entry["ref"], entry["path"], str(entry["page"])) in explicit
        # Artifact metadata and unrelated historical reads remain queryable.
        # Explicit reads in this agent's current session keep their exact values.
        if focused is not None and not requested_here and (
                entry["ref"] not in focused or "structured_results" not in source):
            index.append({**{k: entry[k] for k in ("ref", "path", "page", "bytes")},
                          "visibility": "stored_not_shown", "reason": "outside_current_focus"})
            continue
        record = {"ref": entry["ref"], "path": entry["path"], "page": entry["page"],
                  "source_status": source.get("status"), "data": value, "visibility": "in_request"}
        if "source_result_id" in source:
            record["source_result_id"] = source["source_result_id"]
        if _retention_tier(source, entry["path"]) == 0 and entry["ref"] not in contextualized:
            record["source_context"] = _source_context(source)
            if context_field_bytes is not None:
                record["source_context"] = [
                    item if json_bytes(item["data"]) <= context_field_bytes else
                    {"path": item["path"], "bytes": json_bytes(item["data"]), "deferred": True,
                     "read": {"ref": entry["ref"], **read_selection(item["data"], item["path"])}}
                    for item in record["source_context"]]
            record["independence"] = source_provenance(source)["independence"]
        if json_bytes(retained + [record]) <= WORKING_BYTES:
            retained.append(record)
            if "source_context" in record:
                contextualized.add(entry["ref"])
        else:
            index.append({**{k: entry[k] for k in ("ref", "path", "page", "bytes")},
                          "visibility": "stored_not_shown", "reason": "working_context_budget"})
    # This index is only a recent-read convenience. The complete record catalog
    # and durable read handles remain available even when the display is bounded.
    return {"findings": retained, "other_reads": index[:12], "other_read_count": len(index),
            "policy": "Source-addressed prior reads, not new evidence. Interpretations remain interpretations. "
                      "Observations and their source_context take priority over logs and repeated explanations. "
                      "Omitted fields remain queryable; omission is not absence or negative evidence."}
