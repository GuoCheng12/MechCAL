"""Validate test provenance and explicit comparisons, never set a ranking."""

from copy import deepcopy

from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime.result_contract import disposition_contract, operational_evidence_refs
from mechcal.runtime.retry import RequestValidationError


def require(condition, path, message):
    if not condition:
        raise RequestValidationError([{
            "path": ".".join(map(str, path)), "code": "hypothesis_test_contract",
            "repair_paths": [path], "message": message,
        }])


def validate_leading(case, update):
    if not case.runtime.get("hypothesis_tests"):
        return
    path = ["ranking_update", "leading_comparison"]
    comparison = update.ranking_update.leading_comparison
    require(comparison is not None, path, "Explain the current leader and a concrete alternative.")
    require(comparison.leader == update.portfolio.current, path,
            "Use the submitted current leader; do not rewrite the ranking to repair this record.")
    require(comparison.challenger in MECHANISM_POOL and comparison.challenger != comparison.leader,
            path, "Choose a different pool label as an important challenger, not necessarily rank two.")
    refs = {e.evidence_id for e in case.evidence_ledger.items} | {"input:smiles"}
    refs -= operational_evidence_refs(case)
    require(set(comparison.evidence_refs) <= refs, path, "Cite this case's scientific observations only.")
    if comparison.preference_status == "evidence_favors_leader":
        require(bool(comparison.evidence_refs), path, "An evidence-based preference needs a source.")


def validate_comparison(case, update, *, check_leading=True):
    if not case.runtime.get("hypothesis_tests"):
        return
    if check_leading:
        validate_leading(case, update)
    refs = {e.evidence_id for e in case.evidence_ledger.items} | {"input:smiles"}
    refs -= operational_evidence_refs(case)
    pending = {r["result_id"]: r for r in case.runtime["question_loop"].get("pending_results", [])}
    for index, disposition in enumerate(update.dispositions):
        result = pending[disposition.result_id]
        plan = result.get("investigation_plan") or {}
        path = ["dispositions", index, "test_assessment"]
        assessment = disposition.test_assessment
        require(assessment is not None, path, "Compare this result with its original planned test.")
        require(plan.get("hypothesis_test") is not None, path,
                "This execution lacks a prospective hypothesis test; do not invent one after the result.")
        indices = assessment.matched_outcome_indices
        require(len(indices) == len(set(indices)) and all(0 <= i < len(plan["possible_outcomes"]) for i in indices),
                path, "Use distinct zero-based indices from the original possible_outcomes, or [] for unexpected results.")
        contract = disposition_contract(result, case)
        scientific_refs = set(contract["observation_refs"]) & refs
        operational_refs = set(contract["operational_refs"])
        require(set(assessment.evidence_refs) <= scientific_refs, path,
                f"test_assessment.evidence_refs must use scientific observations from this execution: "
                f"{sorted(scientific_refs)}. Use [] when none were delivered. "
                f"Record failures or missing output in operational_refs using {sorted(operational_refs)}, "
                "not as scientific evidence. Preserve the unavailable/unresolved assessment for no_result.")
        require(set(assessment.operational_refs) <= operational_refs, path,
                f"test_assessment.operational_refs must use operational-only records from this execution: "
                f"{sorted(operational_refs)}. Do not cite other executions, reviews, or scientific observations here.")
        directional = assessment.comparison_effect in ("favors_target", "favors_alternative")
        changed = assessment.basis_effect in ("strengthened", "weakened")
        if directional or changed:
            require(disposition.outcome != "no_result" and bool(assessment.evidence_refs), path,
                    "Changing a scientific basis requires delivered observations, not missing output.")
        if disposition.outcome == "no_result":
            require(assessment.comparison_effect == "unavailable" and assessment.basis_effect == "unresolved",
                    path, "No result leaves the test unavailable and its basis unresolved, not disproven.")
        if disposition.outcome == "non_discriminating":
            require(not directional, path,
                    "A non-discriminating result cannot itself favor one explanation over the other.")
        if plan.get("purpose") == "preparation":
            require(not directional and not changed, path,
                    "Preparation enables a test; it does not itself strengthen or weaken its scientific hypothesis.")


def validate_tests(case, action):
    if not case.runtime.get("hypothesis_tests"):
        return
    issues = []

    def check(condition, paths, message):
        if not condition:
            issues.append({"path": ".".join(map(str, paths[0])), "code": "hypothesis_test_contract",
                           "repair_paths": paths, "message": message})

    for index, plan in enumerate(action.investigation_plans):
        path = ["investigation_plans", index, "hypothesis_test"]
        test = plan.hypothesis_test
        check(test is not None, [path], "Identify the concrete claim tested or enabled by this request.")
        if test is None:
            continue
        labels = {test.target_label, test.alternative_label}
        check(len(labels) == 2 and labels <= set(plan.candidate_labels) & set(MECHANISM_POOL),
              [["investigation_plans", index, "candidate_labels"], path],
              "The target and alternative must be distinct pool labels already considered by this plan. "
              f"Submitted pair: {sorted(labels)}; candidate_labels: {plan.candidate_labels}. "
              "Correct the candidate list or the test pair to match the intended test; keep unrelated plans unchanged.")
        check(test.tested_claim not in MECHANISM_POOL, [[*path, "tested_claim"]],
              "Name a molecule-specific physical claim, not just a mechanism label.")
    if issues:
        raise RequestValidationError(issues)


def preserve_tests(case, update):
    if not case.runtime.get("hypothesis_tests"):
        return {}
    state = case.runtime["question_loop"]
    pending = {r["result_id"]: r for r in state.get("pending_results", [])}
    records = [{"revision": update.revision, "round": state["round_index"],
                "result_id": d.result_id,
                "plan": deepcopy(pending[d.result_id]["investigation_plan"]),
                "assessment": d.test_assessment.model_dump(mode="json")}
               for d in update.dispositions]
    return {"hypothesis_test_history": [*state.get("hypothesis_test_history", []), *records]}


def test_context(case):
    if not case.runtime.get("hypothesis_tests"):
        return {"enabled": False}
    state = case.runtime.get("question_loop", {})
    history = state.get("hypothesis_test_history", [])
    latest = state.get("last_comparison") or {}
    ranking = latest.get("ranking_update") or {}
    counts = {}
    for item in history:
        effect = item["assessment"]["comparison_effect"]
        counts[effect] = counts.get(effect, 0) + 1
    return {"enabled": True, "current_comparison": deepcopy(ranking.get("leading_comparison")),
            "recent_tests": deepcopy(history[-6:]), "comparison_outcomes": counts,
            "notice": "These are past Planner judgments, not independent evidence. No numeric score or forced ranking change is prescribed."}
