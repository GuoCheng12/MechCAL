"""Task-scoped wire views with exact, case-local deferred context.

Repair/delivery/revision context is deferred. Scientific action selection retains
the complete compact directory. Nothing in this module decides claim validity.
"""

from copy import deepcopy

from mechcal.runtime.command_compact import expand_text
from mechcal.runtime.command_context import share_text
from mechcal.runtime.command_memory import digest, pointer_child
from mechcal.runtime.command_operations import protocol, put, reject
from mechcal.schemas.commands import ReadEvidence


CONTEXT_REF = "context:planner"


def task_for(case, payload):
    p = protocol(case)
    if payload.get("judgment_draft"):
        return "repair"
    if payload.get("exploration_state", {}).get("new_result_ids"):
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
    value["task_context_version"] = 2
    deferred = {}
    full = deepcopy(value)
    if task in {"repair", "delivery", "revision"}:
        for name in ("tool_catalog", "comparison_options", "shared_tool_guidance",
                     "action_index", "artifact_index", "explanation_index", "evidence_associations"):
            if name in value:
                deferred[name] = value.pop(name)
        # Keep the short source indexes intact: tool identity and failed-result
        # citation permissions are needed even for a wording-only amendment.
        if "exploration_state" in value:
            exploration = value["exploration_state"]
            deferred["exploration_state"] = deepcopy(exploration)
            view = dict(exploration)
            view["confidence_ranking"] = [{k: row[k] for k in
                ("label", "confidence", "rank", "explanation", "refs", "limitation_refs", "argument_refs",
                 "formal_matches_assessment") if k in row}
                for row in exploration.get("confidence_ranking", [])]
            # Presence still advertises the same command contracts. The detailed
            # question state remains explicitly reachable, not silently absent.
            if "evidence_followup" in view:
                view["evidence_followup"] = {"read_ref": "followup:state", **{k: v for k, v in
                    exploration["evidence_followup"].items() if k in {"pending_impacts", "pending_impact_count",
                    "open_question_count", "stale_formal_labels", "unreviewed_result_count"}}}
            view.pop("last_note", None)
            value["exploration_state"] = view
        if "ranking" in value:
            deferred["ranking"] = value["ranking"]
            value["ranking"] = [{k: v for k, v in row.items() if k != "validation_needed"}
                                for row in value["ranking"]]
    snapshot = {"case_id": case.case_id, "fields": deferred}
    reads = protocol(case).get("context_reads", {})
    used = reads.get("count", 0) if reads.get("snapshot") == digest(snapshot) else 0
    value["context_reads_remaining"] = max(0, 2 - used)
    expanded = (reads.get("snapshot") == digest(snapshot)
                and "tool_catalog" in reads.get("requested_fields", []))
    if used >= 2 or expanded:
        value = full
        value["context_reads_remaining"] = max(0, 2 - used)
        value["task_context_fallback"] = ("scientific_interface_requested" if expanded else
                                         "context_read_allowance_exhausted; keep complete current context")
    value["planner_interface"] = (task if task in {"repair", "delivery", "revision"}
                                  and used < 2 and not expanded and "tool_catalog" in deferred else "full")
    value["deferred_context"] = [{"field": name, "ref": CONTEXT_REF,
        "path": pointer_child("/fields", name), "sha256": digest(data)}
        for name, data in deferred.items()] if used < 2 and not expanded else []
    value["task_context_policy"] = (
        "This is a task view, not a restriction on scientific decisions. Deferred fields are exact current "
        "context, not absent evidence. Use read_context(fields=[...]) to retrieve the named fields. "
        "Reading tool_catalog also restores the full scientific operation interface on your next turn. "
        "All original source IDs remain readable; context:planner is not evidence. Current ranking, "
        "outstanding findings and computation allowance remain authoritative.")
    workflow = value.get("review_workflow")
    if workflow:
        workflow["issue_groups"] = issue_groups(workflow.get("issues", []))
    return put(case, planner_context=snapshot), share_text(value)


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
               "The next task view includes complete context. Choose the next operation.")
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
