"""Small role-specific views. Full scientific records remain queryable by ID."""

from mechcal.runtime.command_memory import json_bytes, pointer_child


def record_table(rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    defaults = {key: rows[0].get(key) for key in columns
                if key != "ref" and len(rows) > 1 and all(row.get(key) == rows[0].get(key) for row in rows)}
    columns = [key for key in columns if key not in defaults]
    return {"columns": columns, "defaults": defaults, "rows": [[row.get(key) for key in columns] for row in rows]}


def scoped_directory(cards):
    paths, entries = {}, []
    for card in cards:
        value = card["paths"]
        key = next((key for key, existing in paths.items() if existing == value), None)
        if key is None:
            key = "paths_" + str(len(paths) + 1)
            paths[key] = value
        entries.append({**{k: v for k, v in card.items() if k != "paths"}, "path_set": key})
    return entries, paths


def catalog_table(rows):
    columns = ["capability_id", "description", "required_artifact_kinds", "source_artifact_kinds", "outputs", "status",
               "missing_inputs", "reason", "execution_dependencies", "unimplemented_structured_descriptors"]
    return {"molecular_input": "current case SMILES", "columns": columns,
            "rows": [[row.get(key) for key in columns] for row in rows]}


def result_card(ref, result):
    return {"ref": ref, "tool": result.get("capability_id"), "status": result.get("status"),
            "bytes": json_bytes(result), "report_ref": "report:" + ref,
            "summary_preview": result.get("summary", "")[:160],
            "observable_keys": list(result.get("structured_results", {}).get("parsed_observables", {}))}


def short_identity(identity):
    # The raw ref and its source_result_ref already identify these exact sources.
    return {k: v for k, v in identity.items() if k != "citation_source_refs"
            and not (k == "source_result_ref" and v == identity["ref"])}


def scope_card(identity, row, *, producers=()):
    return {**short_identity(identity), "paths": [pointer_child("", k) for k in row] if isinstance(row, dict) else [""],
            **({"stored_type": row["record_type"]} if isinstance(row, dict) and row.get("record_type") else {}),
            **({"producer_result_refs": list(producers)} if producers else {}),
            "status": row.get("status") if isinstance(row, dict) else None}


def source_card(ref, row):
    """No inferred observations: preview plus an explicit route to exact values."""
    if ref == "input:smiles":
        return row
    return {"ref": ref, **{key: row[key] for key in
            ("status", "capability_id", "source_report_id", "failure") if key in row},
            "summary_preview": row.get("summary", "")[:160]}


def share_text(payload):
    """Represent identical long text once, without merging distinct source records.

    References are JSON pointers within this message, not citations or memory.
    Only strings are shared; observations, source IDs and field positions remain.
    """
    seen = {}

    def visit(value, path):
        if isinstance(value, str) and len(value) >= 160 and any(c.isspace() for c in value):
            if value in seen:
                return {"same_text_at": seen[value]}
            seen[value] = path
        if isinstance(value, dict):
            return {k: visit(v, pointer_child(path, k)) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v, pointer_child(path, i)) for i, v in enumerate(value)]
        return value

    return visit(payload, "")
