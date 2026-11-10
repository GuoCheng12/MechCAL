"""Derived judgment views. No independent copy of scientific prose or ranking."""

from mechcal.runtime.command_memory import digest

ANSWER_REF = "delivery:answer"


def edit_draft_id(case, ref):
    draft = case.runtime.get("command_protocol", {}).get("final_draft")
    return draft["draft_id"] if draft and ref in {ANSWER_REF, draft["draft_id"]} else None


def enabled(case):
    return case.runtime.get("command_protocol", {}).get("review_workflow") is not None


def ordered_ranking(case):
    ranking = case.runtime.get("command_protocol", {}).get("ranking", {})
    rows = [{**ranking[h.name], "rank": i + 1} for i, h in enumerate(case.portfolio.sorted_hypotheses())
            if h.name in ranking]
    if enabled(case):
        for row in rows:
            row.pop("rationale", None)
            row["argument_refs"] = [link for link in argument_links(case, row["label"])
                                    if link["ref"] in row["argument_ids"]]
    return rows


def argument_links(case, label):
    arguments = case.runtime.get("command_protocol", {}).get("arguments", {})
    return [{"ref": aid, "version": digest(row)} for aid, row in arguments.items() if row["label"] == label]


def workflow_records(case):
    if not enabled(case):
        return {}
    p = case.runtime["command_protocol"]
    records = {"judgment:ranking": {"entries": ordered_ranking(case)},
               "judgment:assessments": p.get("exploration", {}).get("entries", {}),
               "review:issues": p["review_workflow"]}
    if p.get("evidence_followup") is not None:
        records["followup:state"] = p["evidence_followup"]
    if p.get("investigation_policy") is not None:
        records["investigation:state"] = p["investigation_policy"]
    if p.get("review") is not None:
        records["review:feedback"] = p["review"]
    if p.get("final_draft"):
        records[ANSWER_REF] = p["final_draft"]["command"]
    return records


def candidate_view(case, row):
    """Formal arguments are current text; assessment notes retain their own age."""
    if not enabled(case):
        return row
    links = argument_links(case, row["label"])
    if not links:
        return row
    p = case.runtime["command_protocol"]
    current = p.get("exploration", {}).get("entries", {}).get(row["label"])
    bound = p.get("evidence_followup", {}).get("formal_basis", {}).get(row["label"])
    aligned = bound is not None and bound == digest(current)
    view = {**row, "argument_refs": links, "formal_matches_assessment": aligned}
    if aligned:
        view.pop("note", None)
        view["assessment_note_location"] = {"ref": "judgment:assessments", "path": "/" + row["label"] + "/note"}
    return view


def computation_allowance(case):
    limits = case.runtime.get("run_limits", {})
    used = case.runtime.get("command_protocol", {}).get("rounds", 0)
    maximum = limits.get("max_exploration_rounds")
    if maximum is None:
        maximum = limits.get("max_rounds")
    if limits.get("unlimited_exploration"):
        maximum = None
    return {"used": used, "maximum": maximum,
            "remaining": max(0, maximum - used) if maximum is not None else None,
            "allowed": limits.get("scientific_tools_allowed", True),
            "unit": "scientific computations; reads, reviews and edits do not consume this allowance"}
