"""Materialize one model-authored command without mutating accepted case data."""

from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime.boundary_decisions import require
from mechcal.runtime.questions import state
from mechcal.runtime.question_changes import apply_question_changes
from mechcal.runtime.retry import RequestValidationError
from mechcal.schemas import CaseRun
from mechcal.schemas.boundary_decisions import CandidateBasis
from mechcal.schemas.evidence import HypothesisEntry, HypothesisPortfolio
from mechcal.schemas.evidence_comparison import RankingUpdate
from mechcal.schemas.judgment_commands import JudgmentCommand, LOCAL_AUDIT_RETAINED_FIELDS
from mechcal.schemas.question_loop import QuestionAssimilation

Record = TypeVar("Record", bound=BaseModel)


def validate_command_scope(case: CaseRun, command: JudgmentCommand) -> None:
    """Report independent request errors together, before materializing a snapshot."""
    from mechcal.runtime.assimilation_checks import audit_response_issues

    issues = audit_response_issues(case, command.audit_responses)
    current = state(case)
    if command.scope == "local_audit":
        def issue(paths, message):
            issues.append({"path": paths[0], "repair_paths": paths,
                           "code": "judgment_scope_contract", "message": message})

        if not current.get("argument_review_pending") or current.get("pending_results"):
            issue([["scope"]], "Local audit updates require pending audit feedback, not new results. Use scope=global.")
        for field in LOCAL_AUDIT_RETAINED_FIELDS:
            value = getattr(command, field)
            supplied = value is not None if field == "ranking_update" else bool(value)
            if supplied:
                empty = "null" if field == "ranking_update" else "[]"
                issue([[field], ["scope"]],
                      f"For local_audit, {field} must be {empty} to retain the accepted state. "
                      "Only an intended ranking or investigation change requires scope=global; "
                      "do not widen scope just to repeat an unchanged ranking.")
        allowed = set((current.get("argument_review") or {}).get("support_ids", []))
        for field, ids in (
            ("mechanism_support_arguments", {a.support_id for a in command.mechanism_support_arguments}),
            ("withdrawn_support_ids", set(command.withdrawn_support_ids)),
        ):
            if ids - allowed:
                issue([[field], ["scope"]], "Changes outside the reviewed arguments require scope=global. "
                      f"Local support IDs: {sorted(allowed)}; unexpected IDs: {sorted(ids - allowed)}.")
    if issues:
        raise RequestValidationError(issues)


def snapshot_feedback(error: RequestValidationError, snapshot: BaseModel) -> RequestValidationError:
    """Attach stable record keys; snapshot indexes are not sparse-command indexes."""
    data = snapshot.model_dump(mode="json")
    body = data.get("update", data)
    portfolio = HypothesisPortfolio.model_validate(body["portfolio"]) if "portfolio" in body else None
    issues = []
    for original in error.issues:
        issue = {k: v for k, v in original.items() if k not in {"path", "repair_paths"}}
        path = (original.get("repair_paths") or [original.get("path", [])])[0]
        issue["snapshot_path"] = path
        root = original.get("response_field", "update") if "update" in data else None
        record = data.get(root, data) if root else data
        if root:
            issue["command_field"] = root
        if isinstance(path, (list, tuple)):
            for position, part in enumerate(path):
                is_record = isinstance(record, list)
                if isinstance(record, dict):
                    record = record.get(part)
                elif isinstance(record, list) and str(part).isdigit() and int(part) < len(record):
                    record = record[int(part)]
                else:
                    break
                if isinstance(record, dict):
                    keys = ("support_id", "question_id", "result_id", "name", "label")
                    selector = {key: record[key] for key in keys if key in record}
                    if selector and is_record:
                        issue["record_selector"] = selector
                        issue.setdefault("record_selectors", []).append({
                            "collection_path": list(path[:position]), "selector": selector,
                        })
        issue["feedback_scope"] = "materialized_snapshot"
        if portfolio is not None:
            issue["proposed_ranking"] = [
                {"rank": i + 1, "label": h.name, "differential_priority": h.differential_priority}
                for i, h in enumerate(portfolio.sorted_hypotheses())]
        issues.append(issue)
    return RequestValidationError(issues)


def _upsert(
    previous: Sequence[Record], submitted: Sequence[Record], key: str, path: list[str]
) -> list[Record]:
    ids = [getattr(record, key) for record in submitted]
    require(len(ids) == len(set(ids)), path, f"Submit each {key} once in {path[0]}.")
    records = {getattr(record, key): record.model_copy(deep=True) for record in previous}
    records.update({getattr(record, key): record.model_copy(deep=True) for record in submitted})
    return list(records.values())


def materialize_judgment(case: CaseRun, command: JudgmentCommand) -> QuestionAssimilation:
    """Bind case/revision in the runtime; only the model supplies scientific values."""
    require(
        case.runtime.get("incremental_judgment"),
        ["action"],
        "Judgment commands require incremental mode.",
    )
    validate_command_scope(case, command)
    current = state(case)
    revision = current.get("revision", 0)
    old = {h.name: h.model_copy(deep=True) for h in case.portfolio.hypotheses} if revision else {}
    names = [h.name for h in command.hypotheses]
    require(
        len(names) == len(set(names)), ["hypotheses"], "Submit each mechanism once in hypotheses."
    )
    require(
        set(names) <= set(MECHANISM_POOL),
        ["hypotheses"],
        "Use names from the supplied mechanism pool.",
    )
    issues = []
    for index, change in enumerate(command.hypotheses):
        fields = change.model_dump(exclude={"name"}, exclude_none=True)
        base = old.get(change.name)
        message = None
        if base is None:
            required = {"confidence", "differential_priority", "rationale"}
            if missing := sorted(required - fields.keys()):
                message = f"New mechanism {change.name} needs explicit fields: {missing}."
        elif not fields:
            message = "Supply changed fields, or omit this existing mechanism."
        if message:
            issues.append({
                "path": ["hypotheses", index], "repair_paths": [["hypotheses", index]],
                "code": "hypothesis_change_contract", "message": message,
            })
            continue
        values = base.model_dump() if base is not None else {"name": change.name}
        old[change.name] = HypothesisEntry.model_validate({**values, **fields})
    if issues:
        raise RequestValidationError(issues)
    require(bool(old), ["hypotheses"], "The initial judgment needs explicit mechanism candidates.")
    hypotheses = list(old.values())
    # This derives display metadata from Planner scores, not a new scientific order.
    ordered = HypothesisPortfolio(hypotheses=hypotheses).sorted_hypotheses()
    portfolio = HypothesisPortfolio(
        hypotheses=hypotheses,
        current=ordered[0].name,
        runner_up=ordered[1].name if len(ordered) > 1 else None,
    )
    previous_support = {a.support_id: a for a in case.mechanism_support_arguments}
    removed = command.withdrawn_support_ids
    require(
        len(removed) == len(set(removed)) and set(removed) <= previous_support.keys(),
        ["withdrawn_support_ids"],
        "Withdraw distinct existing support IDs only.",
    )
    require(
        not set(removed) & {a.support_id for a in command.mechanism_support_arguments},
        ["withdrawn_support_ids"],
        "Do not withdraw and update the same support record.",
    )
    arguments = _upsert(
        [a for sid, a in previous_support.items() if sid not in removed],
        command.mechanism_support_arguments,
        "support_id",
        ["mechanism_support_arguments"],
    )
    questions = apply_question_changes(current.get("questions", {}), command.questions)
    require(bool(questions), ["questions"], "Keep at least one explicit investigation question.")
    saved = current.get("assimilation", {}).get("ranking_update")
    ranking = (
        RankingUpdate.model_validate(command.ranking_update.model_dump(mode="json"))
        if command.ranking_update is not None else None
    )
    if ranking is None and saved is not None:
        ranking = RankingUpdate.model_validate(
            {**saved, "basis": "unchanged", "explanation": command.reason}
        )
    elif ranking is not None:
        ranking.candidate_bases = _upsert(
            [CandidateBasis.model_validate(b) for b in (saved or {}).get("candidate_bases", [])],
            ranking.candidate_bases,
            "label",
            ["ranking_update", "candidate_bases"],
        )
    return QuestionAssimilation(
        case_id=case.case_id,
        revision=revision + 1,
        portfolio=portfolio,
        questions=questions,
        mechanism_support_arguments=arguments,
        ranking_update=ranking,
        comparison_response=command.comparison_response,
        dispositions=[d.model_copy(deep=True) for d in command.dispositions],
        candidate_responses=[c.model_copy(deep=True) for c in command.candidate_responses],
        decision_assessment=command.decision_assessment.model_copy(deep=True),
        coverage_response=command.coverage_response,
        audit_responses=dict(command.audit_responses),
    )
