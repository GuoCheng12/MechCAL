"""Offline dependency contracts; synthetic reviews are not scientific approvals."""

from copy import deepcopy

import pytest

from mechcal.agents.evidence_context import relevant_evidence
from mechcal.runtime import scoped_updates
from mechcal.runtime.incremental_audit import audit_basis, audit_incrementally, route_feedback, validate_routing
from mechcal.runtime.questions import apply_result_review, state
from mechcal.runtime.ranking_explanations import RankingExplanationDraft, retain_comparisons
from mechcal.schemas import CaseRun
from mechcal.schemas.question_loop import ResultDisposition, ResultReview
from test_argument_assessment import Client, review as agent_review
from test_audit_wire_contract import wire
from test_evidence_comparison import argument
from test_incremental_judgment import review as support_review
from test_scoped_ranking import observe, review_batch, scoped_case, update_for
from test_scoped_updates import auditor, impact as impact_review, new_batch


def dependency_case():
    case = scoped_case().touch(current_round_id="R002")
    ranked = case.portfolio.sorted_hypotheses()
    case.mechanism_support_arguments = [argument(f"S-{i}", h.name) for i, h in enumerate(ranked)]
    state(case).pop("comparison_dependency_cache")
    state(case)["comparison_dependency_cache"] = scoped_updates.remember_comparisons(case)
    return case


@pytest.mark.parametrize("change", ["priority", "tail_order", "confidence"])
def test_unrelated_candidate_does_not_invalidate_pair_dependencies(change):
    case = dependency_case()
    ranked = case.portfolio.sorted_hypotheses()
    field = "confidence" if change == "confidence" else "differential_priority"
    value = (ranked[4].differential_priority + ranked[5].differential_priority) / 2 if change == "tail_order" else 0
    update = update_for(case, [dict(name=ranked[-1].name, **{field: value})])
    before = case.model_dump_json()
    draft = RankingExplanationDraft(case, update)
    assert not draft.leading_required and not draft.pairs
    assert draft.retained_comparisons == scoped_updates.comparison_records(case)
    assert retain_comparisons(case, update)
    assert case.model_dump_json() == before


@pytest.mark.parametrize("index", [0, 2])
def test_pair_priority_change_invalidates_only_its_comparison(index):
    case = dependency_case()
    h = case.portfolio.sorted_hypotheses()[index]
    update = update_for(case, [dict(name=h.name, differential_priority=h.differential_priority - .001)])
    draft = RankingExplanationDraft(case, update)
    assert draft.leading_required == (index == 0)
    assert bool(draft.pairs) == (index == 2)


@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_relevant_role_changes_invalidate_without_changing_old_pair_facts(position):
    case = dependency_case()
    ranked = case.portfolio.sorted_hypotheses()
    priority = 1 if position == 0 else (
        ranked[position - 1].differential_priority + ranked[position].differential_priority) / 2
    update = update_for(case, [dict(name=ranked[-1].name, differential_priority=priority)])
    records = scoped_updates.comparison_records(case)
    assert scoped_updates.comparison_reusable(case, update, "leading") == (position >= 2)
    boundary = next(key for key in records if key.startswith("boundary:"))
    assert not scoped_updates.comparison_reusable(case, update, boundary)


@pytest.mark.parametrize("kind", ["leading", "boundary"])
def test_rewritten_comparison_audits_context_without_reauditing_retained_support(tmp_path, kind):
    case = dependency_case()
    source = observe(case, "E:retained-source")
    for a in case.mechanism_support_arguments:
        a.observation_refs.append(source.evidence_id)
    o, store, calls = auditor(tmp_path, case)
    initial = audit_incrementally(o, case, store)
    changed = initial.model_copy(deep=True)
    records = scoped_updates.comparison_records(changed)
    key = "leading" if kind == "leading" else next(k for k in records if k.startswith("boundary:"))
    field = "basis_for_preference" if kind == "leading" else "reason_for_order"
    records[key][field] += " Rewritten relative-ranking explanation."
    fresh = new_batch(changed, "E:new")
    for result in [*state(fresh)["results"], *state(fresh)["pending_results"]]:
        result["status"] = "ok"
    changed = apply_result_review(fresh, impact_review(fresh))
    before = changed.model_dump_json()
    result = audit_incrementally(o, changed, store)
    assert len(calls) == 2
    audit_case = calls[-1]
    assert audit_case.mechanism_support_arguments == []
    scope = audit_case.runtime["support_audit_scope"]
    assert scope["reviewed_support_ids"] == []
    assert scope["comparisons"] == {key: records[key]}
    labels = set(scoped_updates._comparison_labels(records[key]))
    expected = [a.model_dump(mode="json") for a in changed.mechanism_support_arguments if a.target_label in labels]
    assert scope["comparison_support_arguments"] == expected
    assert {source.evidence_id, "E:new"} <= {
        e.evidence_id for e in relevant_evidence(audit_case, [])}
    assert result.photophysics_review.support_argument_audits == initial.photophysics_review.support_argument_audits
    assert {entry["source_review_id"] for entry in state(result)["incremental_audit_cache"].values()} == {"offline-review"}
    assert "support_audit_scope" not in result.runtime
    assert changed.model_dump_json() == before
    audit_incrementally(o, result, store)
    assert len(calls) == 2

    # Exercise the real strict audit interface with an offline scripted response.
    audit_case.runtime["planner_strict_outputs"] = True
    client = Client([wire(audit_case)])
    checked = agent_review(audit_case, client)
    payload = client.calls[0]["payload"]
    assert payload["allowed_support_ids"] == []
    assert checked.support_argument_audits == []
    assert payload["audit_scope"]["comparison_support_arguments"]


@pytest.mark.parametrize("change", ["source", "finding", "warrant", "boundary"])
def test_actual_source_or_claim_change_still_requires_support_and_pair_review(tmp_path, change):
    case = dependency_case()
    source = observe(case, "E:cited")
    case.mechanism_support_arguments[0].observation_refs.append(source.evidence_id)
    state(case).pop("comparison_dependency_cache")
    state(case)["comparison_dependency_cache"] = scoped_updates.remember_comparisons(case)
    o, store, calls = auditor(tmp_path, case)
    changed = audit_incrementally(o, case, store).model_copy(deep=True)
    if change == "source":
        changed.evidence_ledger.items[-1].summary += " Corrected source fact."
        review_batch(changed, [source.evidence_id])
    else:
        a = changed.mechanism_support_arguments[0]
        setattr(a, change, getattr(a, change) + " Changed claim content.")
    result = audit_incrementally(o, changed, store)
    assert [a.support_id for a in calls[-1].mechanism_support_arguments] == ["S-0"]
    assert "leading" in calls[-1].runtime["support_audit_scope"]["comparisons"]
    assert len(result.photophysics_review.support_argument_audits) == len(case.mechanism_support_arguments)
    assert RankingExplanationDraft(changed, update_for(changed)).leading_required


@pytest.mark.parametrize("impact", ["support", "label", "worker", "planner", "global", "missing", "stale"])
def test_counterevidence_and_uncertain_receipts_cannot_reuse_support(tmp_path, impact):
    case = dependency_case()
    o, store, calls = auditor(tmp_path, case)
    fresh = audit_incrementally(o, case, store).model_copy(deep=True)
    unit = observe(fresh)
    label = fresh.mechanism_support_arguments[0].target_label
    if impact != "missing":
        review_batch(fresh, [unit.evidence_id], affected_support_ids=["S-0"] if impact == "support" else [],
                     affected_labels=[label] if impact == "label" else [], global_review=impact == "global")
    if impact == "worker":
        state(fresh)["results"][-1]["worker_explanation"] = {
            "findings": [{"mechanism_implications": [{"label": label, "reason": "Synthetic counterevidence."}]}]}
        checked = ResultReview.model_validate(dict(status="reviewed", impact_review={
            **state(fresh)["evidence_impact_receipt"]["review"],
            "snapshot_id": scoped_updates.impact_context(fresh)["snapshot_id"],
        }))
        scoped_updates.validate_impact_review(checked, fresh)
        state(fresh)["evidence_impact_receipt"] = scoped_updates.impact_receipt(fresh, checked)
    elif impact == "planner":
        state(fresh)["assimilation"]["dispositions"] = [dict(
            result_id="X:new", mechanism_effects=[dict(label=label, effect="weakens")])]
    elif impact == "stale":
        unit.summary += " Changed after impact review."
    audit_incrementally(o, fresh, store)
    dirty = {a.support_id for a in calls[-1].mechanism_support_arguments}
    assert "S-0" in dirty
    if impact in {"global", "missing", "stale"}:
        assert dirty == {a.support_id for a in fresh.mechanism_support_arguments}
    assert unit.evidence_id in calls[-1].runtime["support_audit_scope"]["required_evidence_refs"]


def test_no_impact_batches_and_checkpoint_reuse_without_fresh_approval(tmp_path):
    case = dependency_case()
    o, store, calls = auditor(tmp_path, case)
    current = audit_incrementally(o, case, store)
    audits = deepcopy(current.photophysics_review.support_argument_audits)
    for index in range(5):
        fresh = new_batch(current, f"E:batch-{index}", f"X:batch-{index}")
        current = apply_result_review(fresh, impact_review(fresh))
        current = audit_incrementally(o, CaseRun.model_validate_json(current.model_dump_json()), store)
    assert len(calls) == 1
    assert current.photophysics_review.support_argument_audits == audits


@pytest.mark.parametrize("enabled", [False, None])
def test_legacy_comparison_change_still_invalidates_all_support(tmp_path, enabled):
    case = dependency_case()
    if enabled is None:
        case.runtime.pop("scoped_evidence_updates")
    else:
        case.runtime["scoped_evidence_updates"] = enabled
    o, store, calls = auditor(tmp_path, case)
    initial = audit_incrementally(o, case, store)
    scoped_updates.comparison_records(initial)["leading"]["basis_for_preference"] += " Changed comparison."
    audit_incrementally(o, initial, store)
    assert len(calls[-1].mechanism_support_arguments) == len(case.mechanism_support_arguments)
    assert "support_audit_scope" not in calls[-1].runtime


def test_cross_case_cache_and_receipt_never_authorize_reuse(tmp_path):
    case = dependency_case()
    o, store, calls = auditor(tmp_path, case)
    foreign = audit_incrementally(o, case, store).model_copy(deep=True)
    unit = observe(foreign)
    review_batch(foreign, [unit.evidence_id])
    foreign.case_id = "another-case"
    audit_incrementally(o, foreign, store)
    assert len(calls[-1].mechanism_support_arguments) == len(case.mechanism_support_arguments)
    assert not scoped_updates.current_receipt(foreign)
    draft = RankingExplanationDraft(foreign, update_for(foreign))
    assert draft.leading_required and draft.pairs


def test_comparison_only_global_concern_reopens_previously_reused_support(tmp_path):
    case = dependency_case()
    o, store, calls = auditor(tmp_path, case)
    current = audit_incrementally(o, case, store)
    scoped_updates.comparison_records(current)["leading"]["basis_for_preference"] += " Changed comparison."
    normal_audit = o._audit_current_round

    def global_audit(case, store, audit_case=None):
        calls.append(audit_case)
        assert audit_case.mechanism_support_arguments == []
        review = support_review(audit_case, mode="global")
        review.overclaim_warnings = ["New comparison concern requires reconsidering retained S-0."]
        validate_routing(review, audit_case.mechanism_support_arguments)
        return case.touch(photophysics_review=review)

    o._audit_current_round = global_audit
    current = audit_incrementally(o, current, store)
    assert current.photophysics_review.routing.mode == "global"
    assert current.photophysics_review.overclaim_warnings
    assert not state(current)["incremental_audit_cache"]
    assert not state(current)["incremental_audit_comparisons"]
    routed = route_feedback(current, {"support_ids": ["S-0"]})
    assert state(routed)["argument_review_pending"]
    o._audit_current_round = normal_audit
    audit_incrementally(o, routed, store)
    assert len(calls) == 3
    assert calls[-1].mechanism_support_arguments == current.mechanism_support_arguments


@pytest.mark.parametrize("effect", ["supports", "weakens", "unavailable", None])
def test_non_neutral_or_missing_planner_effect_stays_conservative(tmp_path, effect):
    case = dependency_case()
    o, store, calls = auditor(tmp_path, case)
    current = audit_incrementally(o, case, store)
    unit = observe(current)
    review_batch(current, [unit.evidence_id])
    label = current.mechanism_support_arguments[0].target_label
    state(current)["assimilation"]["dispositions"] = [dict(
        result_id="X:new", mechanism_effects=[dict(label=label, effect=effect)])]
    audit_incrementally(o, current, store)
    assert "S-0" in {a.support_id for a in calls[-1].mechanism_support_arguments}
    assert RankingExplanationDraft(current, update_for(current)).leading_required


@pytest.mark.parametrize("receipt", [True, False])
def test_explicit_non_discriminating_effect_needs_receipt_for_comparison_reuse(receipt):
    case = dependency_case()
    unit = observe(case)
    if receipt:
        review_batch(case, [unit.evidence_id])
    disposition = ResultDisposition(
        result_id="X:new", question_ids=["Q1"], outcome="non_discriminating",
        interpretation="Synthetic preparation only.", reviewer_response="No support change.",
        evidence_refs=[unit.evidence_id], mechanism_effects=[dict(
            label=case.portfolio.current, effect="non_discriminating", reason="Preparation only.")])
    state(case)["revision"] += 1
    state(case)["assimilation"]["dispositions"] = [disposition.model_dump(mode="json")]
    update = update_for(case)
    update.dispositions = [disposition]
    draft = RankingExplanationDraft(case, update)
    assert draft.leading_required == (not receipt)
    assert bool(draft.pairs) == (not receipt)
    remembered = scoped_updates.remember_comparisons(case)
    assert all(bool(entry.get("invalidated")) == (not receipt) for entry in remembered.values())


def test_unavailable_comparison_only_review_is_not_cached_as_approval(tmp_path):
    case = dependency_case()
    o, store, calls = auditor(tmp_path, case)
    current = audit_incrementally(o, case, store)
    accepted = deepcopy(state(current)["incremental_audit_comparisons"])
    scoped_updates.comparison_records(current)["leading"]["basis_for_preference"] += " Changed comparison."
    normal_audit = o._audit_current_round
    o._audit_current_round = lambda case, store, audit_case=None: case.touch(photophysics_review=None)
    unavailable = audit_incrementally(o, current, store)
    assert unavailable.photophysics_review is None
    assert state(unavailable)["incremental_audit_comparisons"] == accepted
    o._audit_current_round = normal_audit
    audit_incrementally(o, unavailable, store)
    assert len(calls) == 2
    assert calls[-1].mechanism_support_arguments == []
    assert "leading" in calls[-1].runtime["support_audit_scope"]["comparisons"]


def test_logged_r002_no_impact_preparation_reuses_all_five_support_audits(tmp_path):
    # Reduced from Case 0003's 2026-10-25 five-round smoke R001/R002 logs.
    # IDs, pair changes and typed decisions are retained; prose/evidence are synthetic.
    # The log had five identical bases but reaudited METAL, AGG, PACK and RIM.
    case = dependency_case().touch(current_round_id="R001")
    bindings = {
        "S_METAL_01": "TRIPLET_METAL_ENERGY_TRANSFER",
        "S_AGG_01": "AGGREGATE_EXCITON_EXCIMER",
        "S_PACK_01": "PACKING_HOST_MATRIX_CONFINEMENT",
        "S_RIM_01": "RIM_RIR_RIV",
        "S_RATE_01": "RADIATIVE_RATE_STATE_BALANCE",
    }
    case.mechanism_support_arguments = [argument(sid, label) for sid, label in bindings.items()]
    ranking = state(case)["assimilation"]["ranking_update"]
    ranking["leading_comparison"].update(leader=bindings["S_METAL_01"], challenger=bindings["S_AGG_01"])
    ranking["boundary_comparisons"][0].update(included_label=bindings["S_PACK_01"], excluded_label=bindings["S_RIM_01"])
    before = {a.support_id: audit_basis(case, a) for a in case.mechanism_support_arguments}
    o, store, calls = auditor(tmp_path, case)
    initial = audit_incrementally(o, case, store)
    rid = "execution:dbf115f70ac63b72f07f"
    fresh = new_batch(initial.touch(current_round_id="R002"), f"{rid}:E001", rid)
    current = apply_result_review(fresh, impact_review(fresh))
    state(current)["assimilation"]["dispositions"] = [dict(
        result_id=rid, outcome="non_discriminating", mechanism_effects=[
            dict(label=bindings[sid], effect="non_discriminating", reason="Preparation only; no support change.")
            for sid in ("S_AGG_01", "S_PACK_01")],
        test_assessment=dict(basis_effect="unchanged", comparison_effect="non_discriminating"))]
    ranking = state(current)["assimilation"]["ranking_update"]
    ranking["leading_comparison"]["challenger"] = bindings["S_RIM_01"]
    for row in scoped_updates.comparison_records(current).values():
        row["evidence_refs"] = ["input:smiles", f"{rid}:E001"]
    assert {a.support_id: audit_basis(current, a) for a in current.mechanism_support_arguments} == before
    result = audit_incrementally(o, current, store)
    assert len(calls) == 2
    scope = calls[-1].runtime["support_audit_scope"]
    assert scope["reviewed_support_ids"] == []
    assert set(scope["reused_support_ids"]) == set(bindings)
    assert len(scope["comparisons"]) == 2
    assert {a["support_id"] for a in scope["comparison_support_arguments"]} == {
        "S_METAL_01", "S_PACK_01", "S_RIM_01"}
    assert result.photophysics_review.support_argument_audits == initial.photophysics_review.support_argument_audits
