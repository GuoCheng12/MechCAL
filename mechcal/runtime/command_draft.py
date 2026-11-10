"""Durable, uncommitted judgment edits. No repair changes accepted science alone."""

from copy import deepcopy

from pydantic import ValidationError

from mechcal.runtime.command_memory import digest, json_bytes
from mechcal.runtime.command_operations import check_owner, protocol, reject
from mechcal.schemas.commands import ReviseJudgment


def make_draft(case, command, issues):
    data = {"case_id": case.case_id, "base_revision": protocol(case).get("revision", 0),
            "command": command.model_dump(mode="json"), "issues": deepcopy(issues)}
    return {"draft_id": digest(data), **data}


def require_draft(case, draft_id, *, current_revision=True):
    check_owner(case)
    draft = protocol(case).get("judgment_draft")
    if not draft or draft["draft_id"] != draft_id:
        reject(["draft_id"], "stale_draft", "No current draft matches this ID.",
               "Use the displayed draft ID; an accepted or discarded draft cannot be edited again.")
    if digest({k: v for k, v in draft.items() if k != "draft_id"}) != draft_id:
        raise RuntimeError("Judgment draft content does not match its stored identity.")
    if current_revision and draft["base_revision"] != protocol(case).get("revision", 0):
        reject(["draft_id"], "stale_revision", "Accepted judgment changed after this draft was created.",
               "Discard the old draft and propose changes against current accepted records.")
    return draft


def repair(case, command):
    draft = require_draft(case, command.draft_id)
    data = deepcopy(draft["command"])
    changes, removed, seen = data["changes"], set(), set()
    for i, edit in enumerate(command.edits):
        index = edit.change_index
        if index >= len(changes) or index in seen:
            reject(["edits", i, "change_index"], "invalid_edit_target",
                   "Use a distinct change_index from the current draft.",
                   "Combine field edits for each record into one patch. Indices refer to the unmodified draft.")
        seen.add(index)
        if edit.kind == "remove_change":
            removed.add(index)
            continue
        if changes[index]["kind"] != edit.kind.removesuffix("_patch"):
            reject(["edits", i, "kind"], "edit_kind", "Patch kind does not match the draft record.",
                   "Use argument_patch, ranking_patch or candidate_patch for its matching record.")
        fields = edit.model_dump(exclude={"kind", "change_index"}, exclude_none=True)
        if not fields:
            reject(["edits", i], "empty_edit", "This patch changes no fields.",
                   "Supply changed values; null keeps a field, while [] clears a list.")
        changes[index].update(fields)
    data["changes"] = [row for i, row in enumerate(changes) if i not in removed]
    data["changes"].extend(c.model_dump(mode="json") for c in command.additional_changes)
    try:
        return ReviseJudgment.model_validate(data)
    except ValidationError as exc:
        reject(["edits"], "invalid_repaired_command", "Repaired transaction must contain 1 to 48 valid changes.",
               "Adjust additions/removals, or discard the draft explicitly.",
               details={"errors": [dict(path=list(e["loc"]), code=e["type"], message=e["msg"])
                                    for e in exc.errors(include_input=False, include_url=False)]})


def draft_view(case):
    draft = protocol(case).get("judgment_draft")
    if not draft:
        return None
    check_owner(case)
    rows = draft["command"]["changes"]
    issues = []
    for issue in draft["issues"]:
        value = draft["command"]
        try:
            for part in issue["path"]:
                value = value[part]
        except (KeyError, IndexError, TypeError):
            value = None
        issues.append({**issue, **({"current_value": value} if json_bytes(value) <= 2048 else {})})
    return {"draft_id": draft["draft_id"], "ref": "draft:" + draft["draft_id"],
            "base_revision": draft["base_revision"], "accepted": False,
            "changes": [{"change_index": i, **{k: row[k] for k in ("kind", "id", "label") if k in row}}
                        for i, row in enumerate(rows)], "issues": issues,
            "policy": "Only this draft is editable. Read /command/changes/<index> for its full text. "
                      "repair_judgment patches fields and revalidates the entire transaction. "
                      "Null patch fields stay unchanged; [] clears a list. "
                      "revise_judgment explicitly replaces the draft; discard_judgment abandons it. "
                      "No draft content is accepted evidence or an accepted ranking."}
