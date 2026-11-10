"""Question state and provenance checks. Scientific judgements remain with Planner."""

import hashlib
import json

from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.schemas import CaseRun, PlannerDecision
from mechcal.schemas.question_loop import QuestionAction, QuestionAssimilation, ResultReview
from mechcal.schemas.tool_arguments import argument_model, planner_argument_model
from mechcal.schemas.mechanisms import INPUT_SMILES_REF
from mechcal.runtime.retry import ModelContractError, RequestValidationError
from mechcal.runtime.result_contract import disposition_contract, operational_evidence_refs
from mechcal.runtime.evidence_comparison import (
    comparison_context, comparison_refs, preserve_update, validate_plans, validate_update,
)


def state(case: CaseRun) -> dict:
    return case.runtime.get("question_loop", {})


def replace_state(case: CaseRun, **updates) -> CaseRun:
    return case.touch(runtime={**case.runtime, "question_loop": {**state(case), **updates}})


def request_signature(request, registry) -> str:
    spec = registry.get(request.capability_id)
    if spec is None:
        raise RequestValidationError([{"path": "capability_id", "code": "unknown_capability",
                                       "message": f"Unknown capability_id: {request.capability_id}"}])
    args = argument_model(spec.route, spec.owner_agent).model_validate(request.tool_args)
    return hashlib.sha256(json.dumps({
        "capability_id": request.capability_id, "input_refs": sorted(request.input_refs),
        "arguments": args.model_dump(mode="json"),
    }, sort_keys=True).encode()).hexdigest()


def dependency_inputs(request) -> dict:
    # Descriptive only. Shared dependency names cannot establish identical work.
    return {"input_refs": sorted(request.input_refs), "parameters": dict(request.tool_args)}


def calculation_identity(request, case, registry) -> dict:
    spec = registry.require(request.capability_id)
    args = argument_model(spec.route, spec.owner_agent).model_validate(request.tool_args).model_dump(mode="json")
    source_id = selected_source_id(request, case, registry)
    args.pop("source_artifact_id", None)
    return {"capability_id": request.capability_id, "definition_version": registry.definition_version(request.capability_id),
            "resolved_source_artifact_id": source_id, "arguments": args,
            "input_refs": sorted(set(request.input_refs) - {source_id})}


def selected_source_id(request, case, registry) -> str | None:
    explicit = request.tool_args.get("source_artifact_id")
    if explicit:
        return explicit
    kinds = registry.require(request.capability_id).input_artifact_kinds
    candidates = [a.artifact_id for a in case.artifact_manifest.artifacts
                  if a.kind in kinds and a.status in {"available", "partial"}]
    return candidates[-1] if candidates else None


def delivery_summary(result: dict, previous: list[dict]) -> dict:
    """Compare delivered values, not tool names or claimed scientific novelty."""
    parsed = result.get("structured_results", {}).get("parsed_observables", {})
    fields = ("final_energy_hartree", "mulliken_charges", "excited_states", "bright_state",
              "route_records", "parsed_states", "charge_schemes", "fragment_partition",
              "frontier_orbitals", "fragment_definition", "solvent_records",
              "dominant_transitions", "ground_to_excited_transition_dipoles",
              "excited_to_excited_transition_dipoles")
    values = {k: parsed[k] for k in fields if parsed.get(k) is not None and parsed[k] not in ([], {})}
    matches = {}
    for key, value in values.items():
        matches[key] = [p["result_id"] for p in previous
                        if p.get("status") in {"success", "partial"}
                        and p.get("structured_results", {}).get("parsed_observables", {}).get(key) == value]
    return {
        "requested": parsed.get("requested_descriptors", []),
        "available": parsed.get("available_descriptors", []),
        "missing": list(dict.fromkeys([*parsed.get("missing_descriptors", []),
                                      *result.get("structured_results", {}).get("missing_deliverables", [])])),
        "returned_value_fields": list(values),
        "identical_value_matches": {k: ids for k, ids in matches.items() if ids},
        "comparison_scope": "Exact returned values only, not proof of identical calculation or scientific equivalence.",
    }


def validate_review(review: ResultReview, case: CaseRun) -> None:
    from mechcal.runtime.scoped_updates import validate_impact_review

    validate_impact_review(review, case)
    pending = {r["result_id"]: r for r in state(case).get("pending_results", [])}
    if review.status != "reviewed":
        if review.interpretations:
            raise ModelContractError("Unavailable review cannot contain interpretations.")
        return
    ids = [i.result_id for i in review.interpretations]
    if len(ids) != len(set(ids)) or set(ids) != set(pending):
        raise ModelContractError("Review must account for every pending result exactly once.")
    for item in review.interpretations:
        result = pending[item.result_id]
        if set(item.question_ids) != set(result["question_ids"]):
            raise ModelContractError("Review question links must match the original request.")
        if not set(item.evidence_refs).issubset(result["evidence_refs"]):
            raise ModelContractError("Review must cite evidence from that execution, not another result.")
        explanation = result.get("worker_explanation") or {}
        expected = {f["finding_id"] for f in explanation.get("findings", [])}
        audited = [a.finding_id for a in item.finding_audits]
        if len(audited) != len(set(audited)) or set(audited) != expected:
            raise ModelContractError("Audit each worker finding exactly once; do not add findings.")


def apply_result_review(case: CaseRun, review: ResultReview) -> CaseRun:
    validate_review(review, case)
    reviewed = {i.result_id: i.model_dump(mode="json") for i in review.interpretations}
    pending_ids = {r["result_id"] for r in state(case).get("pending_results", [])}

    def attach(result):
        if result["result_id"] not in pending_ids or not result.get("worker_explanation"):
            return result
        return {**result, "explanation_review": {
            "status": review.status, "reason": review.reason,
            "assessment": reviewed.get(result["result_id"]),
        }}

    receipt = {}
    if case.runtime.get("scoped_evidence_updates"):
        from mechcal.runtime.scoped_updates import impact_receipt
        receipt["evidence_impact_receipt"] = impact_receipt(case, review)
    return replace_state(
        case, phase="coverage", result_review=review.model_dump(mode="json"),
        results=[attach(r) for r in state(case).get("results", [])],
        pending_results=[attach(r) for r in state(case).get("pending_results", [])],
        **receipt,
    )


def validate_assimilation(update: QuestionAssimilation, case: CaseRun, *, check_comparisons=True) -> None:
    from mechcal.runtime.predecision import validate_response
    from mechcal.runtime.observation_delivery import validate_question_delivery
    validate_response(case, update.comparison_response)
    validate_question_delivery(case, update)
    s = state(case)
    if s.get("phase") != "assimilate":
        raise ValueError("Judgment updates are allowed only during assimilation.")
    from mechcal.runtime.decision_economy import expand_retained, record_argument_changes
    expand_retained(case, update)
    record_argument_changes(case, update)
    if not update.questions:
        raise ModelContractError("Keep at least one explicit investigation question.")
    if case.runtime.get("decision_economy"):
        issues = []
        for index, question in enumerate(update.questions):
            for field, valid in (("decision_role", question.decision_role is not None),
                                 ("assessment", bool(question.assessment.strip()))):
                if not valid:
                    path = ["questions", index, field]
                    issues.append({
                        "path": path, "repair_paths": [path], "type": "missing",
                        "record_selector": {"question_id": question.question_id},
                        "code": "question_decision_role",
                        "message": f"Provide {field}: classify the question as material or limitation and explain why.",
                    })
        if issues:
            raise RequestValidationError(issues)
    from mechcal.runtime.assimilation_checks import validate_bookkeeping
    from mechcal.runtime.candidate_expansion import validate_responses
    from mechcal.runtime.evidence_impact import validate_impact_responses
    validate_bookkeeping(case, update)
    validate_responses(case, update)
    validate_impact_responses(case, update)
    if update.case_id != case.case_id or update.revision != s.get("revision", 0) + 1:
        raise ModelContractError("Wrong case or stale assimilation revision.")
    if s.get("argument_review_pending"):
        expected = set(s["argument_review"]["support_ids"])
        if set(update.audit_responses) != expected or any(not r.strip() for r in update.audit_responses.values()):
            raise ModelContractError("Respond to every reviewed support_id, including withdrawn arguments and unavailable review.")
    questions = {q.question_id: q for q in update.questions}
    if len(questions) != len(update.questions):
        raise ModelContractError("Question IDs must be distinct.")
    if not set(s.get("questions", {})).issubset(questions):
        raise ModelContractError("Do not delete previous questions; explicitly defer or revise them.")
    refs = {e.evidence_id for e in case.evidence_ledger.items} | {INPUT_SMILES_REF}
    results = {r["result_id"]: r for r in s.get("results", [])}
    pending = {r["result_id"]: r for r in s.get("pending_results", [])}
    ids = [d.result_id for d in update.dispositions]
    if len(ids) != len(set(ids)) or set(ids) != set(pending):
        duplicates = sorted({rid for rid in ids if ids.count(rid) > 1})
        raise ModelContractError(
            "Every pending execution, including failures, needs exactly one disposition; "
            "do not resubmit previously assimilated executions. "
            f"Required result_ids={sorted(pending)}; "
            f"missing={sorted(set(pending) - set(ids))}; "
            f"unexpected={sorted(set(ids) - set(pending))}; duplicates={duplicates}. "
            + ("No executions are pending: return dispositions=[]. " if not pending else "")
            + "Existing results remain usable in question result_ids and support argument observation_refs. "
            "Respond to argument review in audit_responses, not dispositions."
        )
    for index, d in enumerate(update.dispositions):
        r = pending[d.result_id]
        if set(d.question_ids) != set(r["question_ids"]):
            raise ModelContractError(
                f"Disposition {d.result_id} must preserve request-question links: "
                f"expected {r['question_ids']}, got {d.question_ids}. "
                "To reuse this result, cite it in the other question's result_ids."
            )
        if not set(d.evidence_refs).issubset(r["evidence_refs"]):
            raise ModelContractError(
                f"Disposition {d.result_id} cites evidence outside its execution: "
                f"{sorted(set(d.evidence_refs) - set(r['evidence_refs']))}."
            )
        contract = disposition_contract(r, case)
        if d.outcome not in contract["allowed_outcomes"]:
            raise RequestValidationError([{
                "path": f"dispositions[{index}].outcome", "code": "invalid_result_outcome",
                "result_id": d.result_id, "execution_status": r["status"],
                "actual": d.outcome, "allowed": contract["allowed_outcomes"],
                "message": f"Failed execution or boundary-only result {d.result_id}: "
                f"dispositions[{index}].outcome={d.outcome!r}; allowed={contract['allowed_outcomes']}. "
                + contract["reason"] + " This does not close the question or forbid a feasible recovery action.",
            }])
        if not set(d.question_ids).issubset(questions):
            raise ModelContractError("Disposition references an unknown question.")
    for q in update.questions:
        if not q.assessment.strip() or not q.related_labels or not set(q.related_labels).issubset(MECHANISM_POOL):
            raise ModelContractError(f"Question {q.question_id} needs a concise assessment and valid related_labels.")
        missing_refs = sorted(set(q.evidence_refs) - refs)
        missing_results = sorted(set(q.result_ids) - results.keys())
        if missing_refs or missing_results:
            raise ModelContractError(
                f"Question {q.question_id} cites unavailable records in case {case.case_id}: "
                f"evidence_refs={missing_refs}, result_ids={missing_results}. "
                "Use exact IDs from this case's evidence_index and execution_history."
            )
        # Questions may reuse existing results; original dispatch links stay in results/dispositions.
    labels = [h.name for h in update.portfolio.hypotheses]
    argument_ids = [a.support_id for a in update.mechanism_support_arguments]
    if len(argument_ids) != len(set(argument_ids)) or any(
        a.target_label not in MECHANISM_POOL for a in update.mechanism_support_arguments
    ):
        raise ModelContractError("Support arguments need distinct support IDs and mechanism-pool labels.")
    if not labels or len(labels) != len(set(labels)) or not set(labels).issubset(MECHANISM_POOL):
        raise ModelContractError("Return distinct mechanism-pool candidates, never invented labels.")
    ordered = update.portfolio.sorted_hypotheses()
    if update.portfolio.current != ordered[0].name or (
        len(ordered) > 1 and update.portfolio.runner_up != ordered[1].name
    ):
        raise ModelContractError("Portfolio current/runner_up must match Planner's priorities.")
    used = set(update.decision_assessment.evidence_refs)
    used.update(ref for h in update.portfolio.hypotheses for ref in h.evidence_refs)
    used.update(ref for a in update.mechanism_support_arguments for ref in a.observation_refs)
    if not used.issubset(refs):
        raise ModelContractError("Judgement must cite existing observations, never review IDs.")
    operational = operational_evidence_refs(case)
    issues = []
    for index, argument in enumerate(update.mechanism_support_arguments):
        invalid = sorted(set(argument.observation_refs) & operational)
        if invalid:
            issues.append({
                "path": f"mechanism_support_arguments[{index}].observation_refs",
                "code": "operational_record_as_observation", "support_id": argument.support_id,
                "invalid_refs": invalid,
                "message": f"Argument {argument.support_id} cites operational-only records {invalid}. "
                "Keep failures in question evidence_refs/result_ids or dispositions, not scientific support "
                "or counterevidence. Retain other valid observations; do not invent replacements.",
            })
    if issues:
        raise RequestValidationError(issues)

    validate_update(case, update, check_comparisons=check_comparisons)


def apply_assimilation(case: CaseRun, update: QuestionAssimilation) -> CaseRun:
    from mechcal.runtime.candidate_expansion import preserve_responses

    validate_assimilation(update, case)
    comparison_state = preserve_update(case, update)
    candidate_state = preserve_responses(case, update)
    previous = {a.support_id: a for a in case.mechanism_support_arguments}
    checked = []
    for item in update.mechanism_support_arguments:
        old = previous.get(item.support_id)
        fields = ("target_label", "support_level", "finding", "warrant", "boundary", "observation_refs")
        unchanged = old and all(getattr(old, key) == getattr(item, key) for key in fields)
        checked.append(item.model_copy(update={
            "audit_status": old.audit_status if unchanged else "not_reviewed",
            "audit_notes": old.audit_notes if unchanged else [],
        }))
    update = update.model_copy(update={"mechanism_support_arguments": checked})
    case = case.touch(portfolio=update.portfolio,
                      mechanism_support_arguments=update.mechanism_support_arguments)
    applied = replace_state(
        case, phase="act", revision=update.revision, pending_results=[],
        questions={q.question_id: q.model_dump(mode="json") for q in update.questions},
        assimilation=update.model_dump(mode="json"),
        **comparison_state,
        **candidate_state,
    )
    if case.runtime.get("scoped_evidence_updates"):
        from mechcal.runtime.scoped_updates import remember_comparisons
        applied = replace_state(applied, comparison_dependency_cache=remember_comparisons(applied))
    return applied


def apply_final_revision(case: CaseRun, revision, registry) -> tuple[CaseRun, PlannerDecision]:
    if state(case).get("phase") != "final_revision":
        raise ValueError("Final revision is allowed only after the final audit.")
    original = PlannerDecision.model_validate(state(case)["decision"])
    incomplete = (case.runtime.get("evidence_decision_contracts")
                  and original.action == "finalize" and revision.final_action.action == "stop"
                  and revision.final_action.stopping_assessment.finish_basis is None
                  and revision.final_action.stopping_assessment.judgment_state == "unresolved")
    if (revision.final_action.action != original.action and not incomplete) or revision.final_action.dispatch_requests:
        raise ModelContractError("Final revision cannot reopen exploration or dispatch tools.")
    arguments = {a.support_id: a for a in case.mechanism_support_arguments}
    if set(revision.audit_responses) != set(arguments) or any(
        not response.strip() for response in revision.audit_responses.values()
    ):
        raise ModelContractError("Respond to every pre-revision argument, even if withdrawn.")
    if not revision.final_action.final_answer_draft or not revision.final_action.final_answer_draft.strip():
        raise ModelContractError("Final revision requires a consistent final answer draft.")
    update = revision.update
    from mechcal.runtime.decision_economy import expand_retained
    expand_retained(case, update)
    if revision.final_action.action == "finalize" and len(update.portfolio.hypotheses) < 3:
        raise ModelContractError("Final revision must retain at least three mechanism-pool candidates.")
    labels = [a.target_label for a in update.mechanism_support_arguments]
    ids = [a.support_id for a in update.mechanism_support_arguments]
    if not set(labels).issubset(MECHANISM_POOL) or len(ids) != len(set(ids)):
        raise ModelContractError("Final support arguments require valid labels and distinct IDs.")
    updated = apply_assimilation(replace_state(case, phase="assimilate"), update)
    if revision.evidence_submission is not None:
        from mechcal.runtime.final_evidence import validate_final_evidence
        validate_final_evidence(updated.portfolio, updated.mechanism_support_arguments, revision.evidence_submission)
        updated = updated.touch(runtime={**updated.runtime, "final_evidence_submission": [
            next(item for item in revision.evidence_submission if item.label == h.name).model_dump(mode="json")
            for h in updated.portfolio.sorted_hypotheses()[:3]
        ]})
    try:
        validate_action(revision.final_action, updated, registry)
    except RequestValidationError as exc:
        raise RequestValidationError([
            {**issue, "response_field": "final_action"} for issue in exc.issues
        ]) from exc
    decision = decision_from_action(revision.final_action, updated, original.round_id)
    return updated, decision


def validate_action(action: QuestionAction, case: CaseRun, registry) -> None:
    s = state(case)
    if s.get("phase") != "act" or s.get("pending_results"):
        raise ValueError("Assimilate all results before acting.")
    if s.get("argument_review_pending"):
        raise ValueError("Respond to the pending argument review before acting.")
    if action.case_id != case.case_id or action.revision != s.get("revision"):
        raise ModelContractError("Wrong case or stale action revision.")
    from mechcal.runtime.decision_value import validate_action_value
    issues = []
    # These read-only checks are independent. Report the whole invalid plan,
    # rather than spend one model correction on each pre-existing error.
    for validate, args in ((validate_plans, (case, action)),
                           (validate_action_value, (case, action, registry))):
        try:
            validate(*args)
        except RequestValidationError as exc:
            issues.extend(exc.issues)
    questions = s.get("questions", {})
    for index, request in enumerate(action.dispatch_requests):
        # The current validated plan states decision value. A previous question
        # classification must not veto a new, explicit scientific investigation.
        path = ["dispatch_requests", index]
        spec = registry.get(request.capability_id)
        if spec is None:
            issues.append({"path": [*path, "capability_id"], "code": "unknown_capability",
                           "message": f"Unknown capability_id: {request.capability_id}"})
            continue
        if not request.question_ids or not set(request.question_ids).issubset(questions):
            issues.append({"path": [*path, "question_ids"], "code": "unknown_question",
                           "message": "Every request must link existing questions."})
        if len(request.question_ids) != len(set(request.question_ids)) or not request.expected_observation.strip():
            issues.append({"path": path, "code": "missing_observation",
                           "message": "Distinct question IDs and expected_observation are required."})
        try:
            signature = request_signature(request, registry)
            identity = calculation_identity(request, case, registry)
            selection = planner_argument_model(spec.route)
            if selection is not None:
                selection.model_validate(request.tool_args)
        except ValueError as exc:
            issues.append({"path": [*path, "tool_args"], "code": "invalid_arguments",
                           "message": f"{request.capability_id} requires valid arguments and an explicit measurement selection: {exc}. "
                           "Use the selected structure's atom map (list_rotatable_dihedrals exposes it) "
                           "when atom selections are missing, or choose a different useful action. No computation has run."})
            continue
        known_results = {r["result_id"] for r in s.get("results", [])}
        if not set(request.reuse_result_ids).issubset(known_results):
            issues.append({"path": [*path, "reuse_result_ids"], "code": "unknown_result",
                           "message": "Reuse must cite existing same-case result IDs: " +
                           ", ".join(sorted(set(request.reuse_result_ids) - known_results))})
        repeats = []
        for result in s.get("results", []):
            if ("resolved_source_artifact_id" in result and
                    result["resolved_source_artifact_id"] != selected_source_id(request, case, registry)):
                continue
            repeat = (identity == result["calculation_identity"] if "calculation_identity" in result
                      else signature == result.get("signature"))
            if repeat:
                repeats.append(result["result_id"])
        if repeats and not request.retry_reason.strip():
            issues.append({"path": [*path, "retry_reason"], "code": "repeated_request",
                           "message": f"Repeated work needs an explicit retry_reason. Matching executions: {repeats}"})
    from mechcal.runtime.boundary_decisions import validate_continuation
    try:
        validate_continuation(case, action, registry)
    except RequestValidationError as exc:
        issues.extend(exc.issues)
    def final_check(condition, path, message, alternative=None):
        if not condition:
            issues.append({"path": path, "code": "finalization_contract", "message": message,
                           "repair_paths": [path, alternative] if alternative else [path]})

    if action.action == "finalize":
        final_check(action.ranking_uncertainty_material is not None, ["ranking_uncertainty_material"],
                    "Declare whether remaining uncertainty materially prevents the bounded ranking.")
        final_check(not (action.ranking_uncertainty_material and
                         action.stopping_assessment.finish_basis == "judgment_sufficient"),
                    ["stopping_assessment"], "Material unresolved ranking uncertainty cannot be called judgment_sufficient.",
                    ["ranking_uncertainty_material"])
        final_check(set(action.question_dispositions) == set(questions) and all(
            reason.strip() for reason in action.question_dispositions.values()), ["question_dispositions"],
            f"Finalization must account for every retained question: {sorted(questions)}.")
        final_check(bool(action.ranking_justification.strip()), ["ranking_justification"],
                    "Explain the leading candidates versus important alternatives.")
        unresolved = [q for q in questions.values() if q["answer_state"] != "answered"]
        final_check(not unresolved or bool(action.unresolved_gaps), ["unresolved_gaps"],
                    "Unanswered questions must remain explicit in the final limitations.")
    final_check(not case.runtime.get("stop_review_feedback") or bool((action.reviewer_response or "").strip()),
                ["reviewer_response"], "Respond to the stop review, including unavailable review.")
    if issues:
        raise RequestValidationError(issues)


def decision_from_action(action: QuestionAction, case: CaseRun, round_id: str) -> PlannerDecision:
    assimilation = QuestionAssimilation.model_validate(state(case)["assimilation"])
    return PlannerDecision(
        decision_id=f"{round_id}:question_action", round_id=round_id,
        portfolio=case.portfolio, current_hypothesis=case.portfolio.current,
        runner_up_hypothesis=case.portfolio.runner_up,
        diagnosis=assimilation.decision_assessment.current_best,
        decision_assessment=assimilation.decision_assessment,
        stopping_assessment=action.stopping_assessment,
        action=action.action, dispatch_requests=action.dispatch_requests,
        mechanism_support_arguments=case.mechanism_support_arguments,
        unresolved_gaps=action.unresolved_gaps, final_answer_draft=action.final_answer_draft,
        rationale=action.rationale, reviewer_response=action.reviewer_response,
        raw_response={"question_action": action.model_dump(mode="json")},
    )


def validate_decision(decision: PlannerDecision, case: CaseRun, registry) -> None:
    action = QuestionAction.model_validate(decision.raw_response.get("question_action"))
    issues = []
    try:
        validate_action(action, case, registry)
    except RequestValidationError as exc:
        issues.extend(exc.issues)
    expected = decision_from_action(action, case, decision.round_id)
    for field in (
        "action", "dispatch_requests", "portfolio", "mechanism_support_arguments",
        "stopping_assessment", "final_answer_draft", "unresolved_gaps",
    ):
        if getattr(decision, field) != getattr(expected, field):
            raise ModelContractError(f"Action selection cannot override saved judgement/contract: {field}")
    from mechcal.capabilities.discovery import LoadedToolState, amesp_enabled, discovery_dispatch_violations

    for index, request in enumerate(decision.dispatch_requests):
        violations = registry.validate_dispatch(request, case.artifact_manifest)
        violations += registry.availability_violations(request, case.artifact_manifest,
                                                       amesp_enabled=amesp_enabled(case))
        issues.extend({"path": ["dispatch_requests", index], "code": "capability_policy", "message": text}
                      for text in dict.fromkeys(violations))
    if case.runtime.get("planner_tool_loading") == "on_demand":
        loaded = LoadedToolState.model_validate(case.runtime.get("tool_discovery", {"case_id": case.case_id}))
        for index, request in enumerate(decision.dispatch_requests):
            issues.extend({"path": ["dispatch_requests", index], "code": "tool_discovery", "message": text}
                          for text in discovery_dispatch_violations(registry, case, loaded, [request]))
    limits = case.runtime.get("run_limits", {})
    if decision.action == "dispatch" and limits.get("scientific_tools_allowed") is False:
        issues.append({"path": "action", "code": "tools_prohibited", "message": "Scientific tools are prohibited during final synthesis."})
    if decision.action == "finalize":
        if len(decision.portfolio.hypotheses) < 3:
            issues.append({"path": "portfolio", "code": "missing_candidates", "message": "Finalization requires at least three distinct mechanism_pool labels."})
        if limits.get("scientific_tools_allowed") and decision.stopping_assessment.finish_basis is None:
            issues.append({"path": "stopping_assessment.finish_basis", "code": "missing_finish_basis",
                           "message": "Voluntary finalization requires a finish_basis."})
    if issues:
        # A rejected source must be repairable using current registry facts,
        # without guessing a replacement or replaying the full scientific plan.
        informed = set()
        for issue in issues:
            path = issue.get("path", [])
            if not isinstance(path, list) or len(path) < 2 or path[0] != "dispatch_requests":
                continue
            index = path[1]
            if index in informed:
                continue
            request = decision.dispatch_requests[index]
            spec = registry.get(request.capability_id)
            if spec is None:
                continue
            informed.add(index)
            issue["execution_contract"] = registry.describe_tools(
                [request.capability_id], case.artifact_manifest, amesp_enabled=amesp_enabled(case))[0]
            issue["source_candidates"] = [{
                "artifact_id": artifact.artifact_id, "kind": artifact.kind,
                **registry.readiness(request.capability_id, case.artifact_manifest,
                    amesp_enabled=amesp_enabled(case), source_artifact_id=artifact.artifact_id,
                    tool_args={**request.tool_args, "source_artifact_id": artifact.artifact_id}),
            } for artifact in case.artifact_manifest.artifacts
                if artifact.kind in spec.input_artifact_kinds and artifact.status in {"available", "partial"}]
        raise RequestValidationError(issues)


def public_payload(case: CaseRun, registry, *, stage: str) -> dict:
    from mechcal.runtime.boundary_decisions import context as boundary_context
    from mechcal.runtime.hypothesis_tests import test_context
    from mechcal.runtime.candidate_expansion import candidate_context
    from mechcal.agents.observation_view import observation_view
    from mechcal.agents.evidence_context import source_checks
    from mechcal.capabilities.discovery import case_catalog
    from mechcal.runtime.question_changes import public_questions
    from mechcal.runtime.evidence_impact import impact_context

    s = state(case)
    cited = {ref for q in s.get("questions", {}).values() for ref in q["evidence_refs"]}
    cited.update(comparison_refs(case))
    cited.update(ref for h in case.portfolio.hypotheses for ref in h.evidence_refs)
    cited.update(ref for a in case.mechanism_support_arguments for ref in a.observation_refs)
    cited.update(ref for r in s.get("pending_results", []) for ref in r["evidence_refs"])
    cited_results = {rid for q in s.get("questions", {}).values() for rid in q.get("result_ids", [])}
    cited.update(ref for r in s.get("results", []) if r["result_id"] in cited_results
                 for ref in r["evidence_refs"])
    observations = [e for e in case.evidence_ledger.items if e.evidence_id in cited]
    return {
        "stage": stage, "case_id": case.case_id, "smiles": case.input.smiles,
        "scoped_evidence_updates": case.runtime.get("scoped_evidence_updates", False),
        "decision_value_checks": case.runtime.get("decision_value_checks", False),
        "decision_economy": case.runtime.get("decision_economy", False),
        "comparative_evidence": case.runtime.get("comparative_evidence", False),
        "comparison_context": comparison_context(case),
        "hypothesis_test_context": test_context(case),
        "boundary_decision_context": boundary_context(case),
        "candidate_context": candidate_context(case),
        "public_input_source": {"evidence_id": INPUT_SMILES_REF, "smiles": case.input.smiles,
                                "scope": "Molecular graph only, not a computation or experimental observation."},
        "task": case.input.user_query, "mechanism_pool": list(MECHANISM_POOL),
        "revision": s.get("revision", 0), "run_limits": case.runtime.get("run_limits", {}),
        "questions": list(public_questions(case).values()),
        "portfolio": case.portfolio.model_dump(mode="json"),
        "previous_support_arguments": [a.model_dump(mode="json") for a in case.mechanism_support_arguments],
        "pending_results": observation_view(s.get("pending_results", [])),
        "evidence_impact_context": impact_context(case),
        "disposition_requirements": [disposition_contract(r, case) for r in s.get("pending_results", [])],
        "operational_only_evidence_refs": sorted(operational_evidence_refs(case)),
        "result_review": s.get("result_review"),
        "predecision_review": s.get("predecision_review"),
        "predecision_comparison": case.runtime.get("predecision_comparison", False),
        "argument_review": s.get("argument_review"),
        "argument_review_pending": s.get("argument_review_pending", False),
        "previous_assessment": s.get("assimilation", {}).get("decision_assessment"),
        "coverage_feedback": case.mechanism_agenda_coverage.model_dump(mode="json")
        if case.mechanism_agenda_coverage else None,
        "stop_review_feedback": case.runtime.get("stop_review_feedback"),
        "evidence_index": [{"evidence_id": e.evidence_id, "claim": e.claim, "status": e.status}
                           for e in case.evidence_ledger.items],
        "observations": [observation_view(e.model_dump(mode="json")) for e in observations],
        "source_checks": source_checks(case, observations),
        "execution_history": [{**{k: v for k, v in r.items()
                                  if k not in {"structured_results", "worker_explanation", "explanation_review"}},
                               "has_worker_explanation": r.get("worker_explanation") is not None,
                               "delivery": delivery_summary(r, s.get("results", [])[:i])}
                              for i, r in enumerate(s.get("results", []))],
        "previous_worker_explanations": [{
            "result_id": r["result_id"], "worker_explanation": r["worker_explanation"],
            "explanation_review": r.get("explanation_review"),
        } for r in s.get("results", []) if r.get("worker_explanation")
            and r["result_id"] not in {p["result_id"] for p in s.get("pending_results", [])}],
        "artifacts": [{"artifact_id": a.artifact_id, "kind": a.kind, "status": a.status}
                      for a in case.artifact_manifest.artifacts],
        "capability_catalog": case_catalog(registry, case),
    }
