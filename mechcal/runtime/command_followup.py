"""Evidence-to-candidate work items. Associations are advice, never evidence."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mechcal.runtime.command_memory import digest
from mechcal.runtime.command_operations import check_owner, protocol, put, reject, reference_issues, OperationRejected
from mechcal.schemas.commands import ComparisonQuestion, ComparisonQuestionWithRoute, ImpactSuggestion


class ImpactItem(ImpactSuggestion):
    disposition: Literal["pending", "reviewed", "deferred", "dismissed"] = "pending"
    response: str = ""
    judgment_basis: str | None = None


class QuestionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: ComparisonQuestion | ComparisonQuestionWithRoute
    revision: int = 1
    trials: list[dict] = Field(default_factory=list)


class FollowupState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    version: int = 1
    reviews: dict[str, dict] = Field(default_factory=dict)
    impacts: dict[str, ImpactItem] = Field(default_factory=dict)
    questions: dict[str, QuestionRecord] = Field(default_factory=dict)
    formal_basis: dict[str, str | None] = Field(default_factory=dict)


def state(case):
    check_owner(case)
    saved = protocol(case).get("evidence_followup")
    return FollowupState.model_validate(saved) if saved is not None else None


def enable(case):
    if state(case) is not None:
        return case
    return put(case, evidence_followup=FollowupState(case_id=case.case_id).model_dump(mode="json"))


def save_state(case, current):
    return put(case, evidence_followup=current.model_dump(mode="json"))


def judgment_basis(case, label):
    entry = (protocol(case).get("exploration") or {}).get("entries", {}).get(label)
    return digest(entry) if entry is not None else None


def unreviewed_results(case):
    current = state(case)
    if current is None:
        return []
    attempted = {rid for review in current.reviews.values() for rid in review["result_ids"]}
    return [rid for rid in protocol(case).get("results", {}) if rid not in attempted]


def accept_review(case, result_ids, feedback):
    current = state(case)
    if current is None:
        raise RuntimeError("Evidence follow-up is not enabled")
    if not set(result_ids).issubset(protocol(case).get("results", {})):
        raise RuntimeError("Impact review refers to missing results")
    key = digest(result_ids)
    receipt = {"result_ids": result_ids, "feedback": feedback}
    if key in current.reviews:
        if current.reviews[key] != receipt:
            raise RuntimeError("Impact review receipt cannot be overwritten")
        return case
    current.reviews[key] = receipt
    if feedback["status"] == "available":
        for suggestion in feedback.get("suggestions", []):
            row = ImpactSuggestion.model_validate(suggestion)
            iid = "impact:" + digest({"case": case.case_id, "group": key, "label": row.label})[:24]
            current.impacts[iid] = ImpactItem(**row.model_dump())
    # Failed optional commentary is recorded once, not converted into clearance.
    return save_state(case, current)


def apply_update(case, command):
    current = state(case)
    responses = getattr(command, "responses", [])
    questions = getattr(command, "questions", [])
    if current is None:
        if responses or questions:
            reject([], "operation_disabled", "Evidence follow-up is not enabled.", "Use the advertised operations.")
        return case
    seen = set()
    for i, response in enumerate(responses):
        if response.impact_id not in current.impacts or response.impact_id in seen:
            reject(["responses", i, "impact_id"], "impact_reference", "Use a distinct impact ID from this case.",
                   "Inspect evidence_followup or read followup:state; omit unchanged responses.", record_id=response.impact_id)
        seen.add(response.impact_id)
        item = current.impacts[response.impact_id]
        item.disposition, item.response = response.disposition, response.note
        item.judgment_basis = judgment_basis(case, item.label)
    seen.clear()
    # Validate the proposed graph as one transaction, including forward references.
    proposed = dict(current.questions)
    for i, question in enumerate(questions):
        if question.id in seen or len(set(question.labels)) != len(question.labels):
            reject(["questions", i], "duplicate_edit", "Question IDs and candidate labels must be distinct.",
                   "Combine repeated edits; no scientific update committed.")
        seen.add(question.id)
        previous = current.questions.get(question.id)
        retained = {name: getattr(previous.value, name, None) for name in
                    ("scope", "parent_id", "relation", "contrast")
                    if previous is not None and getattr(question, name) is None}
        if question.parent_id == "":
            retained["parent_id"] = None
        question = question.model_copy(update=retained)
        if question.contrast:
            issues = reference_issues(case, question.contrast.baseline_refs,
                                      ["questions", i, "contrast", "baseline_refs"], scientific=False)
            if issues:
                raise OperationRejected(issues)
        if previous is None:
            proposed[question.id] = QuestionRecord(value=question)
        elif previous.value != question:
            proposed[question.id] = previous.model_copy(update={"value": question, "revision": previous.revision + 1})
    for i, question in enumerate(questions):
        visited = {question.id}
        parent = proposed[question.id].value.parent_id
        while parent:
            if parent not in proposed or parent in visited:
                reject(["questions", i, "parent_id"], "question_link",
                       "Parent must name a question in this case without creating a cycle.",
                       "Use an existing or same-update question ID, or detach with an empty string.", record_id=parent)
            visited.add(parent)
            parent = proposed[parent].value.parent_id
    current.questions = proposed
    return save_state(case, current)


def unresolved(question):
    return question.status != "resolved" or getattr(question, "assessment", None) != "distinguished"


def question_rows(current, *, include_closed=False):
    """A test result never propagates scientific closure to a broader question."""
    children = {qid: [] for qid in current.questions}
    active = {qid for qid, row in current.questions.items() if unresolved(row.value)}
    for qid, row in current.questions.items():
        parent = row.value.parent_id
        if parent:
            if parent not in children:
                raise RuntimeError("Saved question parent is missing")
            children[parent].append(qid)
        visited = {qid}
        while parent:
            if parent not in current.questions:
                raise RuntimeError("Saved question parent is missing")
            if parent in visited:
                raise RuntimeError("Saved question graph contains a cycle")
            visited.add(parent)
            if unresolved(row.value):
                active.add(parent)
            parent = current.questions[parent].value.parent_id
    rows = []
    for qid, row in current.questions.items():
        if not include_closed and qid not in active:
            continue
        value = {**row.value.model_dump(mode="json"), "revision": row.revision,
                 "recent_results": [{"result_id": t["result_id"], "question_revision": t["revision"]}
                                    for t in row.trials[-3:]]}
        if children[qid]:
            value["subquestions"] = [{"id": child, "status": current.questions[child].value.status,
                "assessment": getattr(current.questions[child].value, "assessment", None),
                "result_ids": [t["result_id"] for t in current.questions[child].trials[-2:]]}
                for child in children[qid]]
            value["unresolved_descendants"] = any(child in active for child in children[qid])
        rows.append(value)
    # Keep broader questions visible first; this order never chooses the next test.
    return sorted(rows, key=lambda q: (q["parent_id"] is not None, q["scope"] != "mechanism"))


def question_snapshot(case, question_id):
    if question_id is None:
        return None
    current = state(case)
    row = current.questions.get(question_id) if current else None
    if row is None:
        reject(["question_id"], "unknown_question", "Comparison question is not in this case.",
               "Save the question in update_hypotheses, or omit question_id for an exploratory calculation.",
               record_id=question_id)
    return {"question_id": question_id, "revision": row.revision, "question": row.value.model_dump(mode="json")}


def record_trial(case, question, *, capability_id, purpose, dispatch_id=None, result_id=None):
    if question is None:
        return case
    current = state(case)
    row = current.questions[question["question_id"]]
    trial = {**question, "capability_id": capability_id, "purpose": purpose,
             "dispatch_id": dispatch_id, "result_id": result_id}
    if trial not in row.trials:
        row.trials.append(trial)
    return save_state(case, current)


def bind_formal(case, changes):
    """Record the basis of changed formal claims; unchanged claims keep their age."""
    current = state(case)
    if current is None:
        return case
    for label in {c.label for c in changes if c.kind in {"argument", "ranking"}}:
        current.formal_basis[label] = judgment_basis(case, label)
    return save_state(case, current)


def view(case):
    current = state(case)
    if current is None:
        return None
    pending = [{"id": iid, **item.model_dump(mode="json")} for iid, item in reversed(current.impacts.items())
               if item.disposition in {"pending", "deferred"}]
    pending.sort(key=lambda row: row["disposition"] == "deferred")
    active = question_rows(current)
    stale = [label for label, basis in current.formal_basis.items() if basis != judgment_basis(case, label)]
    return {"pending_impacts": pending[:12], "pending_impact_count": len(pending),
            "questions": active[:8], "open_question_count": len(active),
            "stale_formal_labels": stale, "state_ref": "followup:state",
            "state_paths": ["/impacts", "/questions", "/reviews", "/formal_basis"],
            "unavailable_reviews": sum(r["feedback"]["status"] != "available" for r in current.reviews.values()),
            "unreviewed_result_count": len(unreviewed_results(case)),
            "policy": "Unresolved work is not negative evidence and does not veto a next action or final answer. "
                      "Only an explicit Planner response resolves an impact; an overall assessment does not. "
                      "Child tests and parent mechanism questions have independent dispositions. "
                      "A completed child does not settle the broader explanation; contrast is planned, not verified."}
