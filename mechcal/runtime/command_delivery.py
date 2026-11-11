"""Content-bound final review; scientific approval is not a program decision."""

from mechcal.runtime.command_memory import digest
from mechcal.runtime.command_operations import finish, observation_records, protocol, reject
from mechcal.runtime.command_views import enabled
from mechcal.schemas.commands import Finalize


def final_draft(case, command):
    candidate = finish(case, command)
    content = candidate.final_answer.model_dump(mode="json")
    revision = protocol(case)["revision"]
    source = {"source_basis": digest(observation_records(case))} if enabled(case) else {}
    return candidate, {"draft_id": "final:" + digest({"revision": revision, "answer": content, **source}),
                       **source,
                       "revision": revision, "command": command.model_dump(mode="json"),
                       "content": content}


def saved_final(case, draft_id):
    draft = protocol(case).get("final_draft")
    if not draft or draft["draft_id"] != draft_id:
        reject(["draft_id"], "unknown_final_draft", "Final draft ID is not current.",
               "Use the displayed final_draft ID, or propose a new answer with finalize.")
    candidate, current = final_draft(case, Finalize.model_validate(draft["command"]))
    if current != draft:
        reject(["draft_id"], "stale_final_draft", "Accepted records have changed since this draft was assembled.",
               "Propose a current final answer for review. Nothing was submitted.")
    return candidate, draft


def reviewed(case, draft):
    review = protocol(case).get("final_review") or {}
    return review.get("draft_id") == draft["draft_id"] and review.get("status") == "available"


def delivery_status(case, *, index=None):
    """Report transaction state, never convert advisory feedback into approval."""
    from mechcal.runtime.command_review_workflow import issue_view
    p = protocol(case)
    draft = p.get("final_draft")
    issues = issue_view(case, index=index) if enabled(case) else []
    return {"draft_id": draft["draft_id"] if draft else None,
            "exact_draft_reviewed": bool(draft and reviewed(case, draft)),
            "open_fact_issues": [i["id"] for i in issues if i["category"] == "fact"],
            "scientific_advice": [i["id"] for i in issues if i["category"] == "science"],
            "optional_wording": [i["id"] for i in issues if i["category"] == "wording"],
            "policy": "Review availability is not scientific approval. Correct factual errors; "
                      "explain unresolved scientific limits. Optional wording alone need not restart review. "
                      "Only submit_final delivers; the runtime never auto-submits."}


def amended_final(case, command):
    draft = protocol(case).get("final_draft")
    if not draft or draft["draft_id"] != command.draft_id:
        reject(["draft_id"], "unknown_final_draft", "Final draft ID is not current.",
               "Use the displayed final_draft ID; no text was changed.")
    answer = draft["command"]["answer"]
    paths = [f.path for f in command.fields]
    if len(paths) != len(set(paths)) or set(paths) & {r.path for r in command.replacements}:
        reject(["fields"], "conflicting_edit_targets", "A field is replaced more than once.",
               "Choose one field edit or exact-text replacement per path. Nothing was changed.")
    if command.limitations is not None and any(r.path.startswith("/limitations/") for r in [*command.replacements, *command.fields]):
        reject(["limitations"], "conflicting_edit_targets", "Cannot replace the limitations list and edit its entries together.",
               "Use either indexed replacements or the full limitations list. Nothing was changed.")
    limitations = list(draft["command"].get("limitations", []) if command.limitations is None else command.limitations)
    for i, edit in enumerate(command.fields):
        if edit.path == "/answer":
            answer = edit.value
        else:
            item = int(edit.path.rsplit("/", 1)[1])
            if item >= len(limitations):
                reject(["fields", i, "path"], "unknown_edit_target", "Limitation index does not exist in this draft.",
                       "Use a current edit_context path. Nothing was changed.")
            limitations[item] = edit.value
    for i, replacement in enumerate(command.replacements):
        item = int(replacement.path.rsplit("/", 1)[1]) if replacement.path.startswith("/limitations/") else None
        if item is not None and item >= len(limitations):
            reject(["replacements", i, "path"], "unknown_edit_target", "The requested limitation entry does not exist.",
                   "Read delivery:answer at /limitations and use a current zero-based index. Nothing was changed.")
        text = answer if item is None else limitations[item]
        if text.count(replacement.old_text) != 1:
            reject(["replacements", i, "old_text"], "text_match",
                   "Replacement must match exactly one passage inside " + replacement.path + ".",
                   "Use a unique exact passage and the correct field path from edit_context. "
                   "Limitations require path=/limitations/<index>, not /answer. If missing, "
                   "use read_evidence for that field from delivery:answer. "
                   "This working-record read does not spend scientific query allowance. No replacements were committed.")
        updated = text.replace(replacement.old_text, replacement.new_text, 1)
        if item is None:
            answer = updated
        else:
            limitations[item] = updated
    if not answer.strip():
        reject(["replacements"], "empty_answer", "The final answer cannot be empty.", "Keep a complete scientific answer.")
    return Finalize(kind="finalize", answer=answer, limitations=limitations)
