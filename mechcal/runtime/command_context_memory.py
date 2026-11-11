"""Bounded task views over exact, role-local context records.

These records are navigation/background, never observations. The model sees a
small current view; ordinary read_evidence calls recover the stored selections.
"""

from copy import deepcopy

from mechcal.runtime.command_compact import expand_text, table_rows
from mechcal.runtime.command_context import record_table
from mechcal.runtime.command_memory import digest, json_bytes, pointer_child


def context_identity(record):
    return "context:task:" + digest({k: record[k] for k in ("case_id", "role", "fields")})


def read_pointer(ref, field, value):
    path = pointer_child("/fields", field)
    return {"ref": ref, "path": path, "bytes": json_bytes(value),
            "read": {"ref": ref, "paths": [path], "view": "directory", "page": None}}


def brief_rows(rows, *, fields, limit=8, preview=160):
    """Display navigation, not an invented summary or a scientific selection."""
    shown = []
    for i, row in list(enumerate(rows))[-limit:]:
        entry = {"source_row": i, **{k: row[k] for k in fields if k in row}}
        for key in ("question", "text", "explanation", "reason", "description"):
            if isinstance(row.get(key), str):
                entry[key + "_preview"] = row[key][:preview]
                if len(row[key]) > preview:
                    entry[key + "_truncated"] = True
        shown.append(entry)
    return {"items": shown, "total": len(rows), "omitted": max(0, len(rows) - len(shown)),
            "selection": "latest rows for navigation, not scientific priority"}


def compact_catalog(table, *, navigation_only=False):
    """All capabilities stay discoverable; large precondition details are on read."""
    rows = []
    for i, row in enumerate(table_rows(table)):
        keys = (("capability_id", "status") if navigation_only else
                ("capability_id", "status", "missing_inputs", "required_artifact_kinds",
                 "source_artifact_kinds", "outputs"))
        short = {k: row[k] for k in keys if k in row}
        short.update(source_row=i, purpose_preview=(row.get("description") or row.get("purpose_preview") or "")[:120])
        rows.append(short)
    return record_table(rows)


def project_scoped(material, *, case_id, role, focus_refs=(), continuing=False):
    """Pure projection. Keep new results, exact facts and current claims inline."""
    value = expand_text(material)
    sections = value.get("sections", value)
    fields = {}
    for name in ("own_proposals", "open_questions", "question_evidence", "comparison_options",
                 "evidence_followup", "tool_catalog", "shared_tool_guidance"):
        data = sections.get(name)
        if data is None or json_bytes(data) <= 2048:
            continue
        fields[name] = deepcopy(data)
        if name == "tool_catalog" and isinstance(data, dict) and "columns" in data:
            sections[name] = compact_catalog(data, navigation_only=continuing)
        elif isinstance(data, list):
            sections[name] = brief_rows(data, fields=("id", "labels", "status", "origin", "parent_id"))
        else:
            sections[name] = {"stored_not_shown": True}

    directory = value.get("read_scope")
    if isinstance(directory, dict) and "columns" in directory:
        rows = table_rows(directory)
        if len(rows) > 12:
            fields["read_scope"] = deepcopy(directory)
            fields["read_path_sets"] = deepcopy(value.get("read_path_sets", {}))
            focus = set(focus_refs)
            selected = [row for row in rows if row["ref"] in focus]
            selected += [row for row in rows if row["ref"] not in focus][:max(0, 8 - len(selected))]
            # Do not truncate explicitly focused source identities.
            value["read_scope"] = record_table(selected)
            used = {row.get("path_set") for row in selected}
            value["read_path_sets"] = {k: v for k, v in value.get("read_path_sets", {}).items() if k in used}
            value["source_directory_counts"] = {"total": len(rows), "shown": len(selected)}

    if not fields:
        return value, None
    record = {"case_id": case_id, "role": role, "fields": fields}
    record["ref"] = context_identity(record)
    value["task_memory"] = {
        "fields": [dict(field=name, **read_pointer(record["ref"], name, data)) for name, data in fields.items()],
        "policy": "Exact role-local background, not evidence. Previews are incomplete. Use read_evidence "
                  "on a listed field or child path; lists support paging. For read_scope, read its columns/defaults "
                  "and a page of rows together. Readable sources are not limited to the displayed preview. "
                  "Current facts and claims remain in this request. Omission is not absence, rejection or approval. "
                  "On continuation, the tool catalog is navigation only; exact prerequisites and outputs remain "
                  "in this context record or describe_tools. Ready allows an attempt, not guaranteed success."}
    return value, record


def bind_scoped_context(index, scope, state, material, *, role, focus_refs, store):
    value, record = project_scoped(material, case_id=index.case.case_id, role=role, focus_refs=focus_refs,
                                   continuing=bool(state.get("exchange")))
    previous = state.get("task_context_record")
    if previous is not None and (previous.get("case_id") != index.case.case_id or previous.get("role") != role):
        raise RuntimeError("Task context belongs to a different case or role")
    if record is None and previous is None:
        return value, index, scope, state
    # Keep exact task context on disk and in the continuation state, not in the
    # model's input. Original source records are never modified or copied here.
    if record is not None:
        path = "memory/task_contexts/" + record["ref"].split(":")[-1] + ".json"
        if not (store.root / path).exists():
            store.write_json(path, record)
    contexts = [record] if record is not None else []
    # A latest tool exchange may still name the previous immutable view. Keep
    # it for that exchange only, not as a growing collection of context copies.
    if previous is not None and (record is None or previous["ref"] != record["ref"]):
        contexts.append(previous)
    index = index.with_contexts(contexts)
    scope = {**scope, **{r["ref"]: r for r in contexts}}
    state = {**state, "task_context_record": record or previous}
    store.append_jsonl("history/context_views.jsonl", {
        "role": role, "context_ref": (record or previous)["ref"], "original_bytes": json_bytes(material),
        "wire_bytes": json_bytes(value), "deferred_fields": list(record["fields"]) if record else [],
        "scientific_mutation": False})
    return value, index, scope, state
