"""Version-bound review issues and local edits, separate from scientific verdicts."""

from pydantic import BaseModel, ConfigDict, Field

from mechcal.runtime.command_memory import digest, resolve_pointer
from mechcal.runtime.command_operations import OperationRejected, check_owner, protocol, put, record_index, reject
from mechcal.runtime.command_views import enabled
from mechcal.schemas.commands import RepairJudgment, ReviseJudgment, ReviewFinding


class ReviewIssue(ReviewFinding):
    id: str
    binding: dict[str, str] = Field(default_factory=dict)
    history: list[dict] = Field(default_factory=list)


class ReviewWorkflow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    version: int = 1
    issues: dict[str, ReviewIssue] = Field(default_factory=dict)


def state(case):
    check_owner(case)
    raw = protocol(case).get("review_workflow")
    current = ReviewWorkflow.model_validate(raw) if raw is not None else None
    if current is not None and any(k != v.id for k, v in current.issues.items()):
        raise RuntimeError("Review issue identity mismatch")
    return current


def enable(case):
    return case if enabled(case) else put(case, review_workflow=ReviewWorkflow(case_id=case.case_id).model_dump(mode="json"))


def focused_value(value, focus):
    if focus is None:
        return value
    value = resolve_pointer(value, focus.path)
    if focus.quote is not None:
        if not isinstance(value, str) or value.count(focus.quote) != 1:
            raise ValueError("The issue quote must occur exactly once at its target path")
        value = focus.quote
    return {"focus": focus.model_dump(mode="json"), "value": value}


def binding(target, sources, available, focus=None):
    bound = {ref: digest(available[ref]) for ref in dict.fromkeys([target, *sources]) if ref in available}
    if target in available and focus is not None:
        try:
            bound[target] = digest(focused_value(available[target], focus))
        except (KeyError, IndexError, TypeError, ValueError):
            bound[target] = "focus_changed_or_missing"
    return bound


def issue_view(case, *, index=None):
    current = state(case)
    if current is None:
        return []
    index = record_index(case, index)
    rows = []
    for item in current.issues.values():
        current_binding = binding(item.target, item.source_refs, index.available, item.focus)
        changed = item.binding != current_binding
        source_changed = any(item.binding.get(ref) != current_binding.get(ref) for ref in item.source_refs)
        # A broad legacy target changing is a new text-review task, not evidence
        # that every closed issue in that text has recurred. The exact final gate
        # still requires review of the changed content. Precise scopes and source
        # changes do re-enter the issue queue.
        broad_edit = (item.focus is None and item.target in index.available and not source_changed)
        if item.status == "resolved" and (not changed or broad_edit):
            continue
        rows.append({"id": item.id, "target": item.target, "category": item.category,
                     "status": "needs_recheck" if changed else item.status, "text": item.text,
                     "source_refs": item.source_refs,
                     "focus": item.focus.model_dump(mode="json") if item.focus else None,
                     "target_present": item.target in index.available})
    return rows


def prior_resolutions(case, *, index=None):
    """Navigation to prior objections, not a queue requiring repeated closure."""
    current = state(case)
    if current is None:
        return []
    index = record_index(case, index)
    active = {row["id"] for row in issue_view(case, index=index)}
    return [{"id": item.id, "target": item.target, "category": item.category,
             "read_ref": "review:issues", "path": "/issues/" + item.id.replace("~", "~0").replace("/", "~1")}
            for item in current.issues.values()
            if item.status == "resolved" and item.id not in active
            and item.binding != binding(item.target, item.source_refs, index.available, item.focus)]


def review_targets(case, *, index=None):
    index = record_index(case, index)
    refs = [*protocol(case).get("arguments", {}), "judgment:ranking", "followup:state", "delivery:answer"]
    return {ref: digest(index.available[ref]) for ref in refs if ref in index.available}


def validate_report(case, command, *, targets, source_refs, index=None):
    """Only structure/ownership checks. The Reviewer, not code, resolves science."""
    current = state(case)
    if current is None:
        raise RuntimeError("Review workflow is disabled")
    index = record_index(case, index)
    findings, seen, issues, retained_resolutions = [], set(), [], 0
    scope_warnings = []

    def note(path, code, message, remedy, *, record_id=None):
        issues.append(dict(path=path, code=code, message=message, remedy=remedy, record_id=record_id))

    for i, row in enumerate(command.findings):
        if row.target not in targets or row.target not in index.available:
            note(["findings", i, "target"], "review_target", "Target was not offered for this review.",
                 "Use an ID from review_targets; advice may also remain in the prose feedback.", record_id=row.target)
        elif digest(index.available[row.target]) != targets[row.target]:
            raise RuntimeError("Review target changed during the read session")
        unknown = set(row.source_refs) - set(source_refs)
        for ref in sorted(unknown):
            note(["findings", i, "source_refs"], "review_source", "Source was not in the review scope.",
                 "Use exact supplied source IDs or omit an unverified reference.", record_id=ref)
        previous = current.issues.get(row.id) if row.id else None
        if row.id and previous is None:
            note(["findings", i, "id"], "unknown_review_issue", "Issue ID does not belong to this case.",
                 "Use an existing issue ID, or null for a new finding.", record_id=row.id)
        if previous and (previous.target != row.target or previous.category != row.category):
            note(["findings", i], "review_identity", "An existing issue cannot be moved to another target/category.",
                 "Update the existing issue in place; report a distinct finding separately.", record_id=row.id)
        if row.focus is not None and row.target in index.available:
            try:
                focused_value(index.available[row.target], row.focus)
            except (KeyError, IndexError, TypeError, ValueError):
                # Focus is an optional reuse hint, not scientific content. A bad
                # hint cannot justify narrow reuse, but need not discard a valid
                # review or spend another model call regenerating its prose.
                scope_warnings.append({"finding_index": i, "target": row.target,
                    "code": "broad_issue_scope", "requested_focus": row.focus.model_dump(mode="json"),
                    "message": "Focus did not match exactly; whole-target scope retained. No text or source was inferred."})
                row = row.model_copy(update={"focus": None})
        if not row.id and row.status == "resolved":
            note(["findings", i, "status"], "new_resolved_issue", "No existing issue was provided to resolve.",
                 "Omit already-correct material; use the existing issue ID to close a finding.")
        identity = row.id or "issue:" + digest({"case": case.case_id, "target": row.target,
                                                "category": row.category, "text": row.text})[:24]
        if identity in seen:
            note(["findings", i, "id"], "duplicate_review_issue", "An issue is edited twice.", "Combine the updates.")
        seen.add(identity)
        bound = binding(row.target, row.source_refs, index.available, row.focus)
        if previous and previous.status == row.status == "resolved" and previous.binding == bound:
            retained_resolutions += 1
        else:
            findings.append({**row.model_dump(mode="json"), "id": identity, "binding": bound})
    if issues:
        raise OperationRejected(issues)
    return {"status": "available", "text": command.text, "findings": findings,
            "retained_resolutions": retained_resolutions, "scope_warnings": scope_warnings}


def accept_report(case, feedback):
    if feedback.get("status") != "available" or "findings" not in feedback:
        return case
    current = state(case)
    index = record_index(case)
    for row in feedback["findings"]:
        finding = ReviewFinding.model_validate({k: v for k, v in row.items() if k in ReviewFinding.model_fields})
        if row["binding"] != binding(row["target"], row["source_refs"], index.available, finding.focus):
            raise RuntimeError("Review finding belongs to a stale target/source version")
        previous = current.issues.get(row["id"])
        history = list(previous.history) if previous else []
        value = finding.model_dump(mode="json")
        if previous and previous.model_dump(mode="json", exclude={"history"}) == {**value, "binding": row["binding"]}:
            continue
        if (previous and previous.status == finding.status == "resolved" and previous.binding == row["binding"]):
            continue  # Rephrasing a closed, unchanged finding is not a new issue revision.
        if previous:
            history.append(previous.model_dump(mode="json", exclude={"history"}))
        current.issues[row["id"]] = ReviewIssue(**value, binding=row["binding"], history=history)
    return put(case, review_workflow=current.model_dump(mode="json"))


def _legacy_changes(case, changes):
    ranking = protocol(case).get("ranking", {})
    return [{**row, "rationale": ranking.get(row["label"], {}).get("rationale", "")}
            if row["kind"] == "ranking" and "rationale" not in row else row for row in changes]


def internal_judgment(case, command):
    """Preserve legacy prose for old readers; new ranking views use argument links."""
    if not enabled(case):
        return command
    values = command.model_dump(mode="json")
    field = "additional_changes" if command.kind == "repair_judgment" else "changes"
    values[field] = _legacy_changes(case, values[field])
    model = RepairJudgment if command.kind == "repair_judgment" else ReviseJudgment
    return model.model_validate(values)


def amended_judgment(case, command):
    """Expand only specified fields; the existing atomic transaction validates all links."""
    arguments = protocol(case).get("arguments", {})
    changes, seen = [], set()
    for i, edit in enumerate(command.edits):
        if edit.id not in arguments or edit.id in seen:
            reject(["edits", i, "id"], "argument_edit_target", "Use distinct accepted argument IDs from this case.",
                   "Read the current argument; add a new argument through revise_judgment.", record_id=edit.id)
        seen.add(edit.id)
        values = edit.model_dump(mode="json", exclude_none=True, exclude={"id"})
        if values:
            changes.append({**arguments[edit.id], **values})
    changes.extend(_legacy_changes(case, [row.model_dump(mode="json") for row in command.rankings]))
    if not changes:
        reject([], "empty_amendment", "No argument or ranking field was edited.", "Provide a local edit or choose the next action.")
    return ReviseJudgment(kind="revise_judgment", changes=changes,
                          note=command.note, request_review=command.request_review)


def validate_routes(case, update, catalog):
    if not enabled(case):
        return
    available = record_index(case).available
    known = {row["capability_id"] for row in catalog}
    for i, q in enumerate(getattr(update, "questions", [])):
        route = getattr(q, "route", None)
        if route is None:
            continue
        if route.capability_id not in known:
            reject(["questions", i, "route", "capability_id"], "unknown_tool", "Comparison route is not in the enabled catalog.",
                   "Use a catalog ID or leave route null for an external/unspecified test.")
        for ref in route.input_refs:
            if ref not in available:
                reject(["questions", i, "route", "input_refs"], "unknown_reference", "Comparison input is not in this case.",
                       "Use an existing input record or leave inputs empty until preparation succeeds.", record_id=ref)


def comparison_options(case, catalog, *, include_closed=False, registry=None, amesp_enabled=False):
    from mechcal.runtime.command_followup import state as followup_state, question_rows
    from mechcal.runtime.command_route_status import route_status
    current = followup_state(case)
    if current is None:
        return []
    tools = {row["capability_id"]: row for row in catalog}
    results = protocol(case).get("results", {})
    options = []
    index = record_index(case)
    for row in question_rows(current, include_closed=include_closed):
        qid = row["id"]
        record = current.questions[qid]
        q = record.value
        route = getattr(q, "route", None)
        tool = tools.get(route.capability_id) if route else None
        options.append({**row,
                        "assessment": getattr(q, "assessment", None),
                        "question": q.question, "outcomes": q.outcomes, "next_test": q.next_test,
                        "disposition_reason": getattr(q, "disposition_reason", ""),
                        "route": route.model_dump(mode="json") if route else None,
                        "route_status": route_status(case, route, tool, registry=registry,
                            amesp_enabled=amesp_enabled, index=index),
                        "route_attempts_for_question": sum(t.get("capability_id") == route.capability_id
                                                           for t in record.trials) if route else 0,
                        "trials": [{"result_id": t.get("result_id"), "question_revision": t["revision"],
                                    "status": results.get(t.get("result_id"), {}).get("status", "not_executed")}
                                   for t in record.trials[-2:]]})
    return options
