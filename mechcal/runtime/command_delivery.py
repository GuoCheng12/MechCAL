"""Content-bound final review; scientific approval is not a program decision."""

from mechcal.runtime.command_memory import digest
from mechcal.runtime.command_operations import finish, protocol, reject
from mechcal.schemas.commands import Finalize


def final_draft(case, command):
    candidate = finish(case, command)
    content = candidate.final_answer.model_dump(mode="json")
    revision = protocol(case)["revision"]
    return candidate, {"draft_id": "final:" + digest({"revision": revision, "answer": content}),
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
