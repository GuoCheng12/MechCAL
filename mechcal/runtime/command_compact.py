"""Opt-in wire views. Scientific state and executable contracts stay authoritative."""

from copy import deepcopy

from mechcal.runtime.command_context import record_table, share_text
from mechcal.runtime.command_memory import resolve_pointer


def expand_text(payload):
    def visit(value, active=frozenset()):
        if isinstance(value, dict) and set(value) == {"same_data_at"}:
            path = value["same_data_at"]
            if path in active:
                raise ValueError("Cyclic shared review value")
            return visit(resolve_pointer(payload, path), active | {path})
        if isinstance(value, dict) and set(value) == {"same_text_at"}:
            original = resolve_pointer(payload, value["same_text_at"])
            if not isinstance(original, str):
                raise ValueError("Shared text must resolve to a literal string")
            return original
        if isinstance(value, dict):
            return {k: visit(v, active) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v, active) for v in value]
        return value
    return visit(payload)


def table_rows(table):
    return [{**table.get("defaults", {}), **dict(zip(table["columns"], row))} for row in table["rows"]]


def comparison_rows(payload):
    """Resolve compact question links before moving their targets out of view."""
    questions = payload.get("exploration_state", {}).get("evidence_followup", {}).get("questions", [])
    by_id = {q["id"]: q for q in questions}
    rows = []
    for option in payload.get("comparison_options", []):
        if option.get("question_location"):
            if (option["question_location"] != "exploration_state.evidence_followup.questions"
                    or option["id"] not in by_id):
                raise RuntimeError("Compact comparison points to an unavailable question")
            rows.append({**by_id[option["id"]], **{k: v for k, v in option.items() if k != "question_location"}})
        else:
            rows.append(dict(option))
    return rows


def brief_catalog(table):
    # Dependencies and backend-specific exclusions remain in describe_tools.
    # Keep purpose, outputs, input requirements and actual current status visible.
    fields = {"capability_id", "description", "required_artifact_kinds", "source_artifact_kinds", "outputs", "status", "missing_inputs", "reason"}
    return {"molecular_input": table.get("molecular_input", "current case SMILES"),
            **record_table([{k: v for k, v in row.items() if k in fields} for row in table_rows(table)])}


def planner_view(payload):
    value = expand_text(payload)
    value["context_profile"] = "compact_v1"
    value["tool_catalog"] = brief_catalog(value["tool_catalog"])
    latest = set(value.get("exploration_state", {}).get("new_result_ids", []))
    for field in ("result_index", "evidence_index"):
        rows = table_rows(value[field])
        for row in rows:
            source = row.get("source_result_ref", row["ref"])
            if source not in latest:
                row.pop("summary_preview", None)
                row.pop("observable_keys", None)
        value[field] = record_table(rows)
    # The question text already appears under evidence_followup. Keep the route
    # status and trial metadata here; the original question records stay readable.
    questions = value.get("exploration_state", {}).get("evidence_followup", {}).get("questions", [])
    by_id = {q["id"]: q for q in questions}
    for option in value.get("comparison_options", []):
        question = by_id.get(option["id"])
        if question is None:
            continue
        for field in list(option):
            if field != "id" and field in question and option[field] == question[field]:
                option.pop(field)
        option["question_location"] = "exploration_state.evidence_followup.questions"
    value["context_directory_policy"] = (
        "Historical index rows omit previews, not sources. Every ref remains readable. "
        "Use report_ref for exact bounded results and read_evidence(view=directory) for raw paths. "
        "Tool descriptions retain inputs, outputs and status; describe_tools provides detailed limits. "
        "comparison_options links to the same-ID question, with current route status and trial metadata.")
    return share_text(value)


def commentary_view(material):
    value = deepcopy(material)
    sections = value.get("sections", value)
    if "tool_catalog" in sections and "columns" in sections["tool_catalog"]:
        sections["tool_catalog"] = brief_catalog(sections["tool_catalog"])
    if "reference_ids" in sections:
        allowed = set(sections.pop("reference_ids"))
        sections.pop("reference_contract", None)
        rows = table_rows(value["read_scope"])
        for row in rows:
            row["review_source"] = row["ref"] in allowed
        value["read_scope"] = record_table(rows)
        sections["reference_navigation"] = (
            "read_scope is the sole source-ID directory. review_source=true marks IDs allowed in findings.source_refs. "
            "citation_uses distinguishes scientific support, limitations and context. Readable does not mean supporting evidence.")
    if isinstance(sections.get("review_targets"), dict):
        sections["review_targets"] = list(sections["review_targets"])
    value["context_profile"] = "compact_v1"
    return value


def eligible_tools(tools, models, payload):
    """Remove only operations whose prerequisites cannot hold in this snapshot."""
    if payload.get("context_profile") != "compact_v1":
        return tools, models
    disabled = set()
    if not payload.get("pending_experiments"):
        disabled.add("manage_experiments")
    if not payload.get("final_draft"):
        disabled.update(("amend_final", "submit_final"))
    if not payload.get("argument_index", {}).get("rows"):
        disabled.add("amend_judgment")
    if not payload.get("judgment_status", {}).get("delivery_available"):
        disabled.add("finalize")
    if payload.get("description_batches_remaining") == 0:
        disabled.add("describe_tools")
    allowance = payload.get("computation_allowance", {})
    if not payload.get("tools_allowed", True) or allowance.get("allowed") is False or allowance.get("remaining") == 0:
        disabled.update(tool["name"] for tool in tools if tool["name"].startswith("compute_")
                        or tool["name"] == "delegate_computation")
    return [t for t in tools if t["name"] not in disabled], {k: v for k, v in models.items() if k not in disabled}
