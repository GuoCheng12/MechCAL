"""Repair review associations once, without regenerating scientific prose."""

from copy import deepcopy

from mechcal.runtime.command_operations import reject


LINK_ERRORS = {"unknown_review_issue", "review_target", "review_source", "review_identity", "duplicate_review_issue"}


def repairable(command, issues):
    return (command is not None and command.kind == "record_review" and bool(issues)
            and all(issue["code"] in LINK_ERRORS for issue in issues))


def repair_material(payload, repair):
    sections = payload.get("sections", payload)
    return {"stage": "review_link_repair", "proposal": repair["proposal"], "issues": repair["issues"],
            "valid_targets": list(sections.get("review_targets", {})),
            "valid_sources": sections.get("reference_ids", []),
            "existing_issues": [{k: row[k] for k in ("id", "target", "category", "text", "source_refs") if k in row}
                                for row in sections.get("review_workflow", {}).get("issues", [])],
            "attempts_remaining": 1}


def apply_links(model, proposal, command):
    updated, seen = deepcopy(proposal), set()
    for i, edit in enumerate(command.edits):
        key = (edit.finding_index, edit.field)
        if edit.finding_index >= len(updated["findings"]) or key in seen:
            reject(["edits", i], "review_link_edit", "Use a distinct field of an existing finding.",
                   "The original review remains uncommitted.")
        seen.add(key)
        updated["findings"][edit.finding_index][edit.field] = edit.value
    return model.model_validate(updated)
