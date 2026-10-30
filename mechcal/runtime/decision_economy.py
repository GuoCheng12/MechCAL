"""Mechanical reuse and change tracking; scientific decisions remain model-authored."""

from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime.boundary_decisions import require
from mechcal.runtime.retry import RequestValidationError
from mechcal.schemas.boundary_decisions import CandidateBasis
from mechcal.schemas.evidence_comparison import ArgumentChange
from mechcal.schemas.mechanisms import EvidenceQuestion


def expand_retained(case, update):
    if update.delta is not None:
        from mechcal.runtime.incremental_judgment import expand_delta
        expand_delta(case, update)
    retained = update.retained_judgment
    if retained is None:
        return
    require(case.runtime.get("decision_economy"), ["retained_judgment"],
            "Explicit reuse is available only in decision_economy mode.")
    state = case.runtime.get("question_loop", {})
    saved = state.get("assimilation", {}).get("ranking_update") or {}
    specs = [
        ("hypothesis_labels", update.portfolio.hypotheses, case.portfolio.hypotheses, "name"),
        ("support_ids", update.mechanism_support_arguments, case.mechanism_support_arguments, "support_id"),
        ("question_ids", update.questions,
         [EvidenceQuestion.model_validate(q) for q in state.get("questions", {}).values()], "question_id"),
        ("candidate_basis_labels", update.ranking_update.candidate_bases if update.ranking_update else [],
         [CandidateBasis.model_validate(b) for b in saved.get("candidate_bases", [])], "label"),
    ]
    additions = []
    paths = {
        "hypothesis_labels": ["portfolio", "hypotheses"],
        "support_ids": ["mechanism_support_arguments"],
        "question_ids": ["questions"],
        "candidate_basis_labels": ["ranking_update", "candidate_bases"],
    }
    for field, target, previous, key in specs:
        ids = getattr(retained, field)
        old = {getattr(item, key): item for item in previous}
        submitted = {getattr(item, key) for item in target}
        require(len(submitted) == len(target), paths[field],
                "Submitted records must also have distinct IDs; reuse cannot hide duplicates.")
        require(len(ids) == len(set(ids)) and set(ids) <= old.keys() | submitted,
                ["retained_judgment", field],
                f"Base IDs must be distinct and resolve to previous or explicitly submitted records. "
                f"Unknown IDs: {sorted(set(ids) - (old.keys() | submitted))}. "
                "A submitted record replaces the base record of the same ID.")
        if field == "candidate_basis_labels":
            require(not ids or update.ranking_update is not None, ["ranking_update"],
                    "Provide the current comparison when reusing candidate bases.")
        # One explicit upsert rule: supplied content wins, never synthesized or
        # replaced by an older scientific judgment. New records need full content.
        combined = {sid: old[sid].model_copy(deep=True) for sid in ids if sid in old}
        combined.update({getattr(item, key): item for item in target})
        order = [sid for sid in old if sid in combined]
        order.extend(sid for sid in combined if sid not in old)
        additions.append((target, [combined[sid] for sid in order] if ids else list(target)))
    # Apply only after all reuse references pass; a rejected update cannot partly expand.
    for target, records in additions:
        target[:] = records
    update.retained_judgment = None


def record_argument_changes(case, update):
    if not case.runtime.get("decision_economy"):
        return
    from mechcal.runtime.evidence_comparison import changed_arguments
    current = {a.support_id: a for a in update.mechanism_support_arguments}
    changes = []
    for sid, old in changed_arguments(case, update).items():
        new = current.get(sid)
        require(new is None or old.target_label == new.target_label,
                ["mechanism_support_arguments"], "Keep support IDs bound to their mechanism; use a new ID for a new target.")
        changes.append(ArgumentChange(
            support_id=sid, disposition="revised" if new is not None else "withdrawn",
            reason="Program-recorded content difference; scientific rationale remains in the submitted argument and judgment.",
            evidence_refs=[], replacement_ids=[],
        ))
    update.argument_changes = changes


def validate_material_outcomes(outcomes, labels, path, *, candidate_path=None):
    from mechcal.schemas.evidence_comparison import MATERIAL_OUTCOME_CHANGES
    issues = []

    def check(condition, location, message, **details):
        if not condition:
            issues.append({"path": location, "repair_paths": [location],
                           "code": "boundary_decision_contract", "message": message, **details})

    useful = [o for o in outcomes if o.decision_effect in {"ranking", "support"}]
    check(bool(useful), path,
          "Name a possible change to ranking or evidence support. More detail or experimental proof alone is not a decision benefit.")
    for index, outcome in enumerate(outcomes):
        if outcome.decision_effect not in MATERIAL_OUTCOME_CHANGES:
            continue
        location = [*path, index] if isinstance(path, list) else [path, index]
        label_path = [*location, "affected_labels"]
        check(bool(outcome.affected_labels) and len(set(outcome.affected_labels)) == len(outcome.affected_labels)
              and set(outcome.affected_labels) <= set(labels) <= set(MECHANISM_POOL), label_path,
              f"affected_labels must be nonempty, distinct, and drawn from this plan's candidate_labels {labels}; "
              f"submitted {outcome.affected_labels}. Correct the affected labels or explicitly revise the plan's "
              "candidate_labels if the scientific comparison includes another mechanism. No labels were changed.",
              repair_paths=[label_path, candidate_path] if candidate_path else [label_path],
              candidate_labels=list(labels), submitted_affected_labels=list(outcome.affected_labels))
        allowed = MATERIAL_OUTCOME_CHANGES[outcome.decision_effect]
        check(outcome.judgment_change in allowed, [*location, "judgment_change"],
              "Specify promote/demote for ranking or strengthen/weaken for support, with the scientific reason in judgment_effect.")
    if issues:
        raise RequestValidationError(issues)


def validate_plan_value(plan, path=None):
    path = path if path is not None else ["investigation_plans"]
    outcomes = plan.possible_outcomes
    issues = []
    groups = []
    if any(o.decision_effect == "preparation" for o in outcomes):
        if plan.decision_value is None or not plan.decision_value.downstream_outcomes:
            target = [*path, "decision_value"]
            issues.append({"path": target, "repair_paths": [target], "code": "boundary_decision_contract",
                           "message": "Preparation must lead to a concrete downstream decision benefit, not another open-ended preparation chain."})
        else:
            groups.append((plan.decision_value.downstream_outcomes, [*path, "decision_value", "downstream_outcomes"]))
    if plan.purpose not in {"preparation", "recovery"} or any(
            o.decision_effect in {"ranking", "support"} for o in outcomes):
        groups.append((outcomes, [*path, "possible_outcomes"]))
    else:
        if not any(o.decision_effect == "preparation" for o in outcomes):
            target = [*path, "possible_outcomes"]
            issues.append({"path": target, "repair_paths": [target], "code": "boundary_decision_contract",
                           "message": "Preparation/recovery needs a named downstream route and decision benefit."})
    for group, location in groups:
        try:
            validate_material_outcomes(group, plan.candidate_labels, location,
                                       candidate_path=[*path, "candidate_labels"])
        except RequestValidationError as exc:
            issues.extend(exc.issues)
    if issues:
        raise RequestValidationError(issues)
