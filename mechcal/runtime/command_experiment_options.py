"""Observation-led experiment options from the existing impact Reviewer."""

from mechcal.runtime.command_memory import digest
from mechcal.runtime.command_operations import protocol, record_index, reference_issues, OperationRejected, reject
from mechcal.schemas.commands import ExperimentSuggestion, ComparisonQuestionWithRoute


def enabled(case):
    return bool((protocol(case).get("investigation_policy") or {}).get("offer_experiments"))


def validate(case, command, catalog, source_ids):
    rows = getattr(command, "experiments", [])
    if not rows:
        return []
    if not enabled(case):
        reject(["experiments"], "operation_disabled", "Experiment proposals are not enabled.",
               "Use the advertised review fields.")
    return validate_rows(case, rows, catalog, source_ids)


def validate_rows(case, rows, catalog, source_ids):
    index = record_index(case)
    tools = {row["capability_id"] for row in catalog}
    for i, row in enumerate(rows):
        for field, refs in ((["source_refs"], row.source_refs),
                            (["route", "input_refs"], row.route.input_refs if row.route else []),
                            (["contrast", "baseline_refs"], row.contrast.baseline_refs if row.contrast else [])):
            path = ["experiments", i, *field]
            # A failed attempt can motivate a new experiment, but never support
            # its predicted outcome. Execution checks its inputs independently.
            issues = reference_issues(case, refs, path, scientific=False, allow_failed=True, index=index)
            if issues:
                raise OperationRejected(issues)
            if not set(refs).issubset(source_ids):
                reject(path, "proposal_source", "Proposal references were not in the review scope.",
                       "Use supplied or readable source IDs; omit unavailable inputs until prepared.")
        if row.route and row.route.capability_id not in tools:
            reject(["experiments", i, "route", "capability_id"], "unknown_tool",
                   "Route is not in this review's enabled catalog.", "Choose an advertised route or null.")
        if len(set(row.labels)) != len(row.labels):
            reject(["experiments", i, "labels"], "duplicate_edit", "Candidate labels must be distinct.",
                   "List each affected candidate once.")
    return [row.model_dump(mode="json") for row in rows]


def accept_options(case, current, rows):
    if not enabled(case):
        raise RuntimeError("Cannot store experiment options while disabled")
    return store_options(case, current, rows, origin="impact_reviewer")


def option_id(case, row, origin):
    data = row.model_dump(mode="json", exclude={"source_refs"})
    prefix = "option_" if origin == "impact_reviewer" else "explore_"
    return prefix + digest({"case": case.case_id, **data})[:24]


def store_options(case, current, rows, *, origin):
    from mechcal.runtime.command_followup import QuestionRecord
    for raw in rows:
        row = ExperimentSuggestion.model_validate(raw)
        data = row.model_dump(mode="json", exclude={"source_refs"})
        qid = option_id(case, row, origin)
        previous = current.questions.get(qid)
        if previous is not None:
            # An identical suggestion cannot reopen a Planner disposition or
            # overwrite its subsequent scientific edits and linked executions.
            continue
        current.questions[qid] = QuestionRecord(value=ComparisonQuestionWithRoute(
            id=qid, status="open", scope="test", **data), origin=origin,
            source_refs=row.source_refs)
    return current


def review_context(case, catalog):
    from mechcal.runtime.command_followup import state
    return {"tool_catalog": catalog,
        "previous_options": [{"id": qid, "question": row.value.question, "status": row.value.status,
                              "route": row.value.route.model_dump(mode="json") if row.value.route else None}
                             for qid, row in state(case).questions.items() if row.origin == "impact_reviewer"],
        "policy": "Optional experiment proposals are advisory questions, not scientific evidence or required work. "
                  "Derive useful comparisons from unexplained observations, competing explanations and actual "
                  "method limits across the mechanism pool, not just confidence leaders. Existing questions may "
                  "already suffice. Consider computable perturbations/controls, including prerequisites and scope "
                  "limits. Do not assume absence of wet-lab conditions makes every computational proxy useless. "
                  "Do not repeat unchanged previous options, invent tool capabilities, force a label or exhaust "
                  "the pool. Up to three options; an empty list is valid. Planner may choose, revise, defer or decline."}
