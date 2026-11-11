"""Task-scoped wire views with exact, case-local deferred context.

All stages retain a small current working view and exact, readable background.
Nothing in this module decides claim validity or whether a route is worth testing.
"""

from copy import deepcopy

from mechcal.runtime.command_compact import comparison_rows, expand_text
from mechcal.runtime.command_context import share_text
from mechcal.runtime.command_memory import digest, pointer_child
from mechcal.runtime.command_context_memory import brief_rows, compact_catalog, read_pointer
from mechcal.runtime.command_compact import table_rows
from mechcal.runtime.command_context import record_table
from mechcal.runtime.command_operations import protocol, put, reject
from mechcal.runtime.command_review_workflow import investigation_frontier
from mechcal.runtime import command_investigation as investigation
from mechcal.schemas.commands import ReadEvidence
from mechcal.runtime.prompts import load_prompt


CONTEXT_REF = "context:planner"


def operation_guides(payload):
    """Exact existing manuals remain readable; schemas still define operations."""
    names = ["planner_tools", "numeric_evidence"]
    for field, guides in (
        ("worker_delegation", ["planner_worker_tasks"]),
        ("exploration_state", ["planner_exploration_state"]),
        ("review_workflow", ["planner_review_workflow"]),
        ("computational_review", ["computational_mission"]),
        ("investigation_policy", ["planner_investigation"]),
    ):
        if field in payload:
            names.extend(guides)
    if "evidence_followup" in payload.get("exploration_state", {}):
        names.append("planner_evidence_followup")
    return {name: load_prompt(name + ".md") for name in names}


def read_continuation(case):
    p = protocol(case)
    calls = (p.get("latest_exchange") or {}).get("calls", [])
    return bool(calls) and all(c["name"] in {
        "read_evidence", "reduce_evidence", "compare_evidence", "read_context", "describe_tools",
    } for c in calls)


def task_for(case, payload):
    p = protocol(case)
    if payload.get("judgment_draft"):
        return "repair"
    if investigation.pending_proposals(case):
        return "exploration_decision"
    if payload.get("exploration_state", {}).get("new_result_ids"):
        return "exploration"
    choice = investigation.current(case)
    if (choice and choice["choice"]["decision"] == "investigate"
            and choice["basis"] == investigation.basis(case)):
        return "exploration"
    if p.get("final_draft") and p.get("delivery_rounds") == p.get("rounds"):
        return "delivery"
    if (p.get("arguments") and (p.get("review") or {}).get("status") == "available"
            and payload.get("interim_review_status", {}).get("automatic_review_reason") is None):
        return "revision"
    return "exploration"


def prepare_planner(case, payload):
    """Return a new case plus its view; the caller persists it before dispatch.

    Deferred fields exclude reads/outcomes to prevent recursively storing prior
    context inside itself. Reading this record never makes it scientific evidence.
    """
    value = expand_text(payload)
    draft = protocol(case).get("judgment_draft")
    displayed = value.get("judgment_draft")
    if draft and displayed and draft["draft_id"] == displayed.get("draft_id"):
        # Display indices are command edits, not paths in the persisted record.
        # Supply validated read selections instead of requiring that inference.
        changes = draft["command"]["changes"]
        for row in displayed.get("changes", []):
            i = row.get("change_index")
            if type(i) is int and 0 <= i < len(changes):
                row["read"] = {"ref": "draft:" + draft["draft_id"],
                               "paths": [f"/command/changes/{i}"], "view": "data", "page": None}
    task = task_for(case, value)
    value["context_task"] = task
    value["task_context_version"] = 4
    value["input_phase"] = "query_continuation" if read_continuation(case) else "decision"
    deferred = {"operation_guides": operation_guides(value)}
    value["operation_guide_names"] = list(deferred["operation_guides"])
    if value.get("comparison_options"):
        value["comparison_options"] = comparison_rows(value)
    inventory = value.get("execution_inventory")
    inventory_seen = protocol(case).get("planner_inventory_seen", {})
    if inventory is not None:
        previous = inventory_seen if inventory_seen.get("case_id") == case.case_id else {}
        versions = {cid: row["version"] for cid, row in inventory["by_capability"].items()}
        changed = {cid: row for cid, row in inventory["by_capability"].items()
                   if previous.get("versions", {}).get(cid) != row["version"]}
        if len(changed) < len(versions):
            deferred["execution_inventory"] = deepcopy(inventory)
            value["execution_inventory"] = {**inventory, "by_capability": changed,
                "view": "changed_routes", "unchanged_routes": len(versions) - len(changed),
                "removed_routes": sorted(set(previous.get("versions", {})) - set(versions)),
                "read_all": {"ref": CONTEXT_REF, "paths": ["/fields/execution_inventory"]}}
        inventory_seen = {"case_id": case.case_id, "versions": versions}
    editing = task in {"repair", "delivery", "revision"}
    for name in ("tool_catalog", "comparison_options", "shared_tool_guidance", "action_index",
                 "artifact_index", "explanation_index", "evidence_associations", "untried_routes"):
        if name in value:
            deferred[name] = deepcopy(value.pop(name))
    if deferred.get("comparison_options"):
        focused = value.get("proposal_decision", {}).get("question_ids", [])
        choice = investigation.current(case)
        if choice and choice["choice"].get("question_id"):
            focused = [*focused, choice["choice"]["question_id"]]
        value["investigation_frontier"] = investigation_frontier(deferred["comparison_options"], focus_ids=focused)
    if not editing and "tool_catalog" in deferred:
        value["tool_catalog"] = compact_catalog(deferred["tool_catalog"], navigation_only=read_continuation(case))
        value["tool_catalog_details"] = read_pointer(CONTEXT_REF, "tool_catalog", deferred["tool_catalog"])
        if read_continuation(case):
            value["context_directory_policy"] = (
                "All capability IDs, purposes and live statuses remain listed. Detailed inputs, outputs and "
                "preconditions are at tool_catalog_details or describe_tools, not inferred from the short row. "
                "Source indexes are navigation, not evidence; omitted facts are not absent or negative.")

    feedback = value.get("review_feedback")
    if isinstance(feedback, dict) and len(feedback.get("text", "")) > 480:
        deferred["review_feedback"] = deepcopy(feedback)
        value["review_feedback"] = {k: v for k, v in feedback.items() if k != "text"}
        value["review_feedback"].update(text_preview=feedback["text"][:480], text_truncated=True,
            read={"ref": CONTEXT_REF, "paths": ["/fields/review_feedback/text"], "view": "data", "page": None})

    latest = set(value.get("exploration_state", {}).get("new_result_ids", []))
    latest.update(h.get("ref") for h in protocol(case).get("planner_read_handles", [])[-6:])
    if "artifact_index" in deferred:
        rows = table_rows(deferred["artifact_index"])
        # Keep addresses in the working view; coordinates and historical detail stay deferred.
        preferred = [r for r in rows if r.get("kind") == "initial_structure"]
        preferred += [r for r in rows if r.get("ref") in latest and r not in preferred]
        recent = [r for r in reversed(rows) if r not in preferred][:6]
        shown = (preferred + recent)[:12] if len(rows) > 12 else rows
        value["artifact_index"] = record_table([{k: r[k] for k in ("ref", "kind", "status") if k in r}
                                                for r in shown])
        value["artifact_index_counts"] = {"total": len(rows), "shown": len(shown)}
    for name in ("result_index", "evidence_index"):
        table = value.get(name)
        if table and len(table.get("rows", [])) > 12:
            deferred[name] = deepcopy(table)
            rows = table_rows(table)
            shown = [r for r in rows if r["ref"] in latest or r.get("source_result_ref") in latest]
            shown += [r for r in rows[-6:] if r not in shown]
            value[name] = record_table(shown)
            value[name + "_counts"] = {"total": len(rows), "shown": len(shown)}
    pending = value.get("pending_experiments")
    if pending:
        deferred["pending_experiments"] = deepcopy(pending)
        value["pending_experiments"] = brief_rows(pending, fields=(
            "ref", "experiment_id", "status", "capability_id", "workspace_read", "record_kind", "citation_uses"))["items"]
        value["pending_experiment_count"] = len(pending)
    if "exploration_state" in value:
        exploration = value["exploration_state"]
        deferred["exploration_state"] = deepcopy(exploration)
        view = dict(exploration)
        view["confidence_ranking"] = [
            {**{k: row[k] for k in ("label", "confidence", "rank", "refs", "limitation_refs",
                                   "argument_refs", "formal_matches_assessment") if k in row},
             "explanation_preview": (row.get("explanation") or "")[:160],
             "read": {"ref": CONTEXT_REF, "paths": [f"/fields/exploration_state/confidence_ranking/{i}"]}}
            for i, row in enumerate(exploration.get("confidence_ranking", []))]
        if "evidence_followup" in view:
            followup = exploration["evidence_followup"]
            view["evidence_followup"] = {"read_ref": "followup:state", **{k: v for k, v in
                followup.items() if k in {"pending_impacts", "pending_impact_count", "open_question_count",
                                        "stale_formal_labels", "unreviewed_result_count"}}}
        view.pop("last_note", None)
        value["exploration_state"] = view
    if "ranking" in value:
        deferred["ranking"] = deepcopy(value["ranking"])
        value["ranking"] = [{k: v for k, v in row.items() if k in
                            {"label", "rank", "priority", "confidence", "argument_ids", "support", "claim"}}
                            for row in value["ranking"]]
    snapshot = {"case_id": case.case_id, "fields": deferred}
    reads = protocol(case).get("context_reads", {})
    used = reads.get("count", 0) if reads.get("snapshot") == digest(snapshot) else 0
    value["context_reads_remaining"] = max(0, 2 - used)
    expanded = (reads.get("snapshot") == digest(snapshot)
                and "tool_catalog" in reads.get("requested_fields", []))
    value["planner_interface"] = (task if editing
                                  and used < 2 and not expanded and "tool_catalog" in deferred else "full")
    value["deferred_context"] = [dict(field=name, **read_pointer(CONTEXT_REF, name, data))
                                 for name, data in deferred.items()]
    value["task_context_policy"] = (
        "This is a task view, not a restriction on scientific decisions. Deferred fields are exact current "
        "context, not absent evidence. Use read_context(fields=[...]) to retrieve the named fields. "
        "Reading tool_catalog restores the scientific operation interface, not the full background. "
        "After the context-read allowance, ordinary read_evidence can address these exact fields or child paths "
        "subject to the current action policy. Exhaustion never expands the entire context. "
        "All original source IDs remain readable; context:planner is not evidence. Current ranking, "
        "outstanding findings and computation allowance remain authoritative. "
        "operation_guides stores exact detailed manuals by name; read a needed child path, not the entire manual set. "
        "In query_continuation the catalog lists IDs, purposes and live status; prerequisites and outputs "
        "remain at tool_catalog_details or describe_tools. A truncated review preview is not the full advice.")
    workflow = value.get("review_workflow")
    if workflow:
        workflow["issue_groups"] = issue_groups(workflow.get("issues", []))
    return put(case, planner_context=snapshot, planner_inventory_pending=inventory_seen), share_text(value)


def acknowledge_planner(case):
    """A transport interruption must not acknowledge context that was never returned."""
    pending = protocol(case).get("planner_inventory_pending")
    if pending and pending.get("case_id") == case.case_id:
        return put(case, planner_inventory_seen=pending, planner_inventory_pending=None)
    return case


def context_selection(case, command):
    """Validate a bounded metadata read without spending/resetting evidence reads."""
    p = protocol(case)
    snapshot = p.get("planner_context")
    if not snapshot or snapshot.get("case_id") != case.case_id:
        reject(["fields"], "context_unavailable", "No current task context is registered.", "Use the current full view.")
    revision = digest(snapshot)
    previous = p.get("context_reads", {})
    used = previous.get("count", 0) if previous.get("snapshot") == revision else 0
    if used >= 2:
        reject(["fields"], "context_read_limit", "Two context reads already used for this exact snapshot.",
               "Use the current working view or a permitted exact read_evidence selection; background stays in memory.")
    receipt = {"snapshot": revision, "count": used + 1}
    unknown = set(command.fields) - snapshot["fields"].keys()
    if unknown:
        # Invalid requests count as attempts too; the caller persists this receipt
        # before returning a normal local operation error.
        return None, receipt, sorted(unknown)
    receipt["requested_fields"] = sorted(set(command.fields) | (
        set(previous.get("requested_fields", [])) if previous.get("snapshot") == revision else set()))
    return ReadEvidence(kind="read_evidence", records=[{"ref": CONTEXT_REF,
        "paths": [pointer_child("/fields", name) for name in dict.fromkeys(command.fields)]}]), receipt, []


def planner_task_tools(tools, models, payload):
    """Only stage-relevant contracts; read_context keeps scientific work reachable."""
    task = payload.get("planner_interface", "full")
    if task == "full":
        return tools, models
    common = {"read_evidence", "reduce_evidence", "compare_evidence", "read_context"}
    commands = {
        "repair": {"repair_judgment", "discard_judgment"},
        "revision": {"amend_judgment", "revise_judgment", "update_hypotheses", "finalize"},
        "delivery": {"amend_judgment", "revise_judgment", "update_hypotheses", "finalize",
                     "amend_final", "review_final", "submit_final"},
    }
    if task not in commands:
        raise ValueError("Unknown Planner interface task: " + task)
    # Never restrict an interface without its explicit way back to exploration.
    if "read_context" not in models:
        return tools, models
    names = common | commands[task]
    return ([tool for tool in tools if tool["name"] in names],
            {name: model for name, model in models.items() if name in names})


def issue_groups(issues):
    """Route Reviewer-owned categories; do not reclassify scientific content."""
    return {name: [row["id"] for row in issues if row["category"] == category]
            for name, category in (("factual_corrections", "fact"),
                                   ("scientific_questions", "science"), ("optional_wording", "wording"))}


def review_view(material, sections):
    """Remove duplicated command bodies, retaining exact changes in record views."""
    material = deepcopy(material)
    shown = material["sections"]
    changes = sections.get("changes", [])
    if changes:
        shown["changes"] = [{k: row[k] for k in ("kind", "id", "label") if k in row}
                            for row in changes]
    # Previous argument text may be required on a first review. Defer it only
    # when the exact prior text has a registered, readable review receipt.
    old = sections.get("previous_arguments", [])
    if old:
        rows = []
        for row in old:
            current = shown.get("arguments:" + row["id"], {})
            exact_old = (current.get("base_sha256") == digest(row)
                         or sections.get("arguments:" + row["id"]) == row)
            if current.get("previous_review_ref") and exact_old:
                rows.append({"id": row["id"], "label": row["label"],
                             "see_change": "arguments:" + row["id"]})
            else:
                rows.append(row)
        shown["previous_arguments"] = rows
    workflow = shown.get("review_workflow")
    if workflow:
        workflow["issue_groups"] = issue_groups(workflow.get("issues", []))
    if "ranking" in sections:
        material["ranking_overview"] = [{k: row[k] for k in
            ("label", "rank", "confidence", "priority", "argument_ids", "support", "claim") if k in row}
            for row in sections["ranking"]]
    material["task_context_version"] = 1
    material["review_contract"] += (
        " Changes is an ID directory, not another copy of the arguments. Review the exact record deltas. "
        "Optional wording is not a required correction or a reason to invent extra numerical claims. "
        "Report factual errors separately from scientific uncertainty and optional style.")
    return material
