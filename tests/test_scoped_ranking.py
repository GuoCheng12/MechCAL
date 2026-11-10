"""Offline scoped explanation reuse; synthetic records are not scientific evidence."""

from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from mechcal.agents.ranking_explanations import explain_ranking
from mechcal.runtime import scoped_updates
from mechcal.runtime.boundary_decisions import required_pairs, validate_ranking
from mechcal.runtime.hypothesis_tests import validate_leading
from mechcal.runtime.judgment_commands import materialize_judgment
from mechcal.runtime.questions import apply_assimilation, replace_state, state, validate_assimilation
from mechcal.runtime.ranking_explanations import RankingExplanationDraft, retain_comparisons
from mechcal.runtime.retry import RequestValidationError
from mechcal.runtime.structured_output import PlannerOutputContract
from mechcal.schemas import CaseRun
from mechcal.schemas.judgment_commands import JudgmentProposal
from mechcal.schemas.question_loop import ResultReview
from mechcal.schemas.ranking_explanations import RankingExplanations
from test_judgment_commands import explanation_response, planner
from test_question_loop import NOTE
from test_ranking_explanation_transaction import prepared_case, proposal
from test_runtime import _evidence_unit


def scoped_case():
    case = prepared_case()
    case.runtime["scoped_evidence_updates"] = True
    records = scoped_updates.comparison_records(case)
    records["leading"]["basis_for_preference"] = "Exact accepted leading explanation."
    for key, row in records.items():
        if key != "leading":
            row["reason_for_order"] = "Exact accepted boundary explanation."
    state(case)["comparison_dependency_cache"] = scoped_updates.remember_comparisons(case)
    return case


def update_for(case, changes=()):
    return materialize_judgment(case, JudgmentProposal.model_validate(proposal(case, changes)))


def fresh_response(draft):
    payload = draft.payload()
    response = explanation_response(payload)
    if response["leading"]:
        response["leading"]["basis_for_preference"] = "Fresh leading explanation."
    for row in response["boundaries"]:
        row["reason_for_order"] = "Fresh boundary explanation."
    Draft202012Validator(payload["output_contract"]).validate(response)
    strict = PlannerOutputContract(payload)
    assert strict.decode(strict.encode(response)) == response
    return RankingExplanations.model_validate(response)


def observe(case, evidence_id="E:new"):
    unit = _evidence_unit(
        evidence_id=evidence_id, round_id="R002", source_report_id="synthetic-report",
        agent_name="macro", family="torsion_sensitivity", summary="Synthetic new observation.")
    case.evidence_ledger.items.append(unit)
    return unit


def review_batch(case, evidence_ids, **impact):
    result = dict(result_id="X:new", evidence_refs=evidence_ids)
    state(case)["pending_results"] = [result]
    state(case)["results"] = [r for r in state(case).get("results", []) if r["result_id"] != "X:new"] + [result]
    context = scoped_updates.impact_context(case)
    checked = dict(
        record_type="EvidenceImpactReview",
        snapshot_id=context["snapshot_id"], result_ids=["X:new"],
        checked_support_ids=[a.support_id for a in case.mechanism_support_arguments],
        affected_support_ids=[], affected_labels=[], affected_comparison_ids=[],
        global_review=False, reason=NOTE)
    review = ResultReview.model_validate(dict(status="reviewed", impact_review={**checked, **impact}))
    scoped_updates.validate_impact_review(review, case)
    state(case)["evidence_impact_receipt"] = scoped_updates.impact_receipt(case, review)


@pytest.mark.parametrize("memory", [False, True])
@pytest.mark.parametrize("changed", ["leading", "boundary"])
def test_partial_reuse_requests_only_changed_comparison_and_restores_exact_records(memory, changed):
    case = scoped_case()
    case.runtime["planner_case_memory"] = memory
    update = update_for(case)
    ranked = update.portfolio.sorted_hypotheses()
    label = ranked[0 if changed == "leading" else 2].name
    basis = next(b for b in update.ranking_update.candidate_bases if b.label == label)
    basis.inference_gap = "A changed comparison basis without a changed score."
    update.ranking_update.basis = "reinterpretation"
    old = deepcopy(scoped_updates.comparison_records(case))
    before = case.model_dump_json(), update.model_dump_json()
    draft = RankingExplanationDraft(case, update)
    payload = draft.payload()
    assert bool(payload["leading_target"]) == (changed == "leading")
    assert bool(payload["boundary_targets"]) == (changed == "boundary")
    if changed == "boundary":
        assert payload["output_contract"]["properties"]["leading"] == {"type": "null"}
    else:
        assert payload["output_contract"]["properties"]["boundaries"]["maxItems"] == 0
    assert not retain_comparisons(case, update)
    result = draft.apply(fresh_response(draft))
    if changed == "boundary":
        assert result.ranking_update.leading_comparison.model_dump(mode="json") == old["leading"]
        assert result.ranking_update.boundary_comparisons[0].reason_for_order == "Fresh boundary explanation."
    else:
        assert result.ranking_update.leading_comparison.basis_for_preference == "Fresh leading explanation."
        assert result.ranking_update.boundary_comparisons[0].model_dump(mode="json") == next(
            row for key, row in old.items() if key != "leading")
    validate_assimilation(result, case)
    assert result.portfolio == update.portfolio
    assert (case.model_dump_json(), update.model_dump_json()) == before


def test_all_cached_comparisons_skip_explanation_request_and_validate_full_contract():
    case = scoped_case()
    update = update_for(case)
    old = deepcopy(scoped_updates.comparison_records(case))
    update.ranking_update.leading_comparison = None
    update.ranking_update.boundary_comparisons = []
    before = case.model_dump_json()
    result = explain_ranking(None, case, update, references=None, memory=None, cache={})
    assert result is update
    assert result.ranking_update.leading_comparison.model_dump(mode="json") == old["leading"]
    assert result.ranking_update.boundary_comparisons[0].model_dump(mode="json") == next(
        row for key, row in old.items() if key != "leading")
    validate_assimilation(result, case)
    assert case.model_dump_json() == before


def test_all_cached_comparisons_do_not_bypass_full_comparison_validation():
    case = scoped_case()
    update = update_for(case)
    update.ranking_update.candidate_bases.pop()
    before = update.model_dump_json()
    assert not retain_comparisons(case, update)
    assert update.model_dump_json() == before


def test_planner_only_requests_changed_comparison():
    case = scoped_case()
    leader = case.portfolio.current
    data = proposal(case, [dict(name=leader, rationale="Changed leader rationale, unchanged order.")])
    agent, client = planner(case, [data])
    result = agent.assimilate(case)
    assert len(client.calls) == 2
    payload = client.calls[-1]["payload"]
    assert payload["leading_target"] == {"leader": leader}
    assert payload["boundary_targets"] == []
    validate_assimilation(result, case)


@pytest.mark.parametrize("change", ["leader", "top3", "two_displaced"])
def test_changed_order_and_new_top3_request_current_roles(change):
    case = scoped_case()
    ranked = case.portfolio.sorted_hypotheses()
    if change == "two_displaced":
        changes = [dict(name=h.name, differential_priority=0) for h in ranked[:2]]
    else:
        priority = 1.0 if change == "leader" else (
            ranked[1].differential_priority + ranked[2].differential_priority) / 2
        changes = [dict(name=ranked[-1].name, differential_priority=priority)]
    update = update_for(case, changes)
    draft = RankingExplanationDraft(case, update)
    assert not retain_comparisons(case, update)
    assert draft.payload()["leading_target"] == (None if change == "top3" else {"leader": update.portfolio.current})
    assert list(draft.pairs.values()) == required_pairs(case.portfolio, update.portfolio)
    assert draft.retained_comparisons == ({"leading": scoped_updates.comparison_records(case)["leading"]}
                                           if change == "top3" else {})
    result = draft.apply(fresh_response(draft))
    validate_assimilation(result, case)


def test_sparse_boundary_targets_keep_original_ids_and_pair_binding(monkeypatch):
    case = scoped_case()
    ranked = case.portfolio.sorted_hypotheses()
    update = update_for(case, [dict(name=ranked[0].name, differential_priority=0),
                               dict(name=ranked[-1].name, differential_priority=1)])
    # Isolate binding from the helper's conservative all-priority-change invalidation.
    cached_key = f"boundary:{ranked[2].name}|{ranked[3].name}"
    monkeypatch.setattr(scoped_updates, "comparison_reusable", lambda case, update, key: key == cached_key)
    draft = RankingExplanationDraft(case, update)
    assert list(draft.required_pairs) == ["cutoff:1", "cutoff:2"]
    assert list(draft.pairs) == ["cutoff:2"]
    schema = draft.payload()["output_contract"]
    assert schema["$defs"]["BoundaryExplanation"]["properties"]["target_id"]["enum"] == ["cutoff:2"]
    response = fresh_response(draft)
    result = draft.apply(response)
    assert result.ranking_update.boundary_comparisons[0].model_dump(mode="json") == (
        scoped_updates.comparison_records(case)[cached_key])
    assert [(c.included_label, c.excluded_label) for c in result.ranking_update.boundary_comparisons] == (
        list(draft.required_pairs.values()))
    validate_assimilation(result, case)
    response.boundaries[0].target_id = "cutoff:1"
    with pytest.raises(RequestValidationError):
        draft.apply(response)


@pytest.mark.parametrize("fault", ["missing_cache", "new_without_receipt", "stale_receipt"])
def test_missing_or_stale_dependency_authorization_is_conservative(fault):
    case = scoped_case()
    if fault == "missing_cache":
        state(case).pop("comparison_dependency_cache")
    else:
        unit = observe(case)
        if fault == "stale_receipt":
            review_batch(case, [unit.evidence_id])
            unit.summary = "Source content changed after the impact review."
    update = update_for(case)
    draft = RankingExplanationDraft(case, update)
    assert draft.leading_required and draft.pairs
    assert not draft.retained_comparisons
    before = update.model_dump_json()
    assert not retain_comparisons(case, update)
    assert update.model_dump_json() == before


@pytest.mark.parametrize("affected", [None, "leading", "boundary"])
def test_current_impact_receipt_scopes_new_observation_reuse(affected):
    case = scoped_case()
    old = deepcopy(scoped_updates.comparison_records(case))
    key = next(key for key in old if key.startswith("boundary:")) if affected == "boundary" else affected
    unit = observe(case)
    review_batch(case, [unit.evidence_id], affected_comparison_ids=[key] if key else [])
    update = update_for(case)
    draft = RankingExplanationDraft(case, update)
    assert draft.leading_required == (affected == "leading")
    assert bool(draft.pairs) == (affected == "boundary")
    result = draft.apply(fresh_response(draft))
    validate_leading(case, result)
    validate_ranking(case, result)
    assert retain_comparisons(case, update) == (affected is None)


@pytest.mark.parametrize("field", ["case_id", "smiles", "cited_source"])
def test_changed_input_or_cited_source_cannot_reuse_under_an_unaffected_receipt(field):
    case = scoped_case()
    if field == "cited_source":
        unit = observe(case, "E:cited")
        scoped_updates.comparison_records(case)["leading"]["evidence_refs"] = [unit.evidence_id]
        state(case).pop("comparison_dependency_cache")
        state(case)["comparison_dependency_cache"] = scoped_updates.remember_comparisons(case)
        unit.summary = "A corrected source fact under the same observation ID."
        review_batch(case, [unit.evidence_id])
    elif field == "smiles":
        case.input.smiles = "CCO"
    else:
        case.case_id = "another-case"
    draft = RankingExplanationDraft(case, update_for(case))
    assert draft.leading_required


def test_changed_source_result_facts_invalidate_cited_comparison():
    case = scoped_case()
    unit = observe(case, "E:source")
    scoped_updates.comparison_records(case)["leading"]["evidence_refs"] = [unit.evidence_id]
    state(case)["results"] = [dict(
        result_id="X:source", evidence_refs=[unit.evidence_id], structured_results={"energy": 1.2})]
    state(case).pop("comparison_dependency_cache")
    state(case)["comparison_dependency_cache"] = scoped_updates.remember_comparisons(case)
    state(case)["results"][0]["structured_results"]["energy"] = 2.4
    assert RankingExplanationDraft(case, update_for(case)).leading_required


def test_confidence_fallback_reordering_cannot_retain_old_challenger():
    case = scoped_case()
    ranked = case.portfolio.sorted_hypotheses()
    for index, hypothesis in enumerate(ranked):
        hypothesis.differential_priority = None
        hypothesis.confidence = .9 - .02 * index
    state(case).pop("comparison_dependency_cache")
    state(case)["comparison_dependency_cache"] = scoped_updates.remember_comparisons(case)
    update = update_for(case, [dict(name=ranked[-1].name, confidence=.89)])
    assert update.portfolio.current == case.portfolio.current
    assert update.portfolio.sorted_hypotheses()[1].name == ranked[-1].name
    assert RankingExplanationDraft(case, update).leading_required


@pytest.mark.parametrize("comparison", ["leading", "boundary"])
def test_retained_state_is_frozen_and_part_of_draft_identity(comparison):
    case = scoped_case()
    update = update_for(case)
    update.ranking_update.leading_comparison = None
    update.ranking_update.boundary_comparisons = []
    draft = RankingExplanationDraft(case, update)
    response = fresh_response(draft)
    old = deepcopy(scoped_updates.comparison_records(case))
    records = scoped_updates.comparison_records(case)
    key = "leading" if comparison == "leading" else next(k for k in records if k.startswith("boundary:"))
    field = "basis_for_preference" if comparison == "leading" else "reason_for_order"
    records[key][field] = "A different accepted explanation."
    state(case).pop("comparison_dependency_cache")
    state(case)["comparison_dependency_cache"] = scoped_updates.remember_comparisons(case)
    changed = RankingExplanationDraft(case, update)
    assert changed.draft_id != draft.draft_id
    with pytest.raises(RequestValidationError):
        changed.apply(response)
    result = draft.apply(response)
    assert result.ranking_update.leading_comparison.model_dump(mode="json") == old["leading"]
    assert result.ranking_update.boundary_comparisons[0].model_dump(mode="json") == next(
        row for key, row in old.items() if key != "leading")
    result.ranking_update.leading_comparison.evidence_refs.append("mutated-result-only")
    assert draft.apply(response).ranking_update.leading_comparison.model_dump(mode="json") == old["leading"]


@pytest.mark.parametrize("fault", ["leading", "boundary", "draft_id", "leader", "scores"])
def test_response_cannot_overwrite_retained_or_frozen_fields(fault):
    case = scoped_case()
    draft = RankingExplanationDraft(case, update_for(case))
    response = fresh_response(draft).model_dump(mode="json")
    uncached = case.model_copy(deep=True)
    uncached.runtime["scoped_evidence_updates"] = False
    full = explanation_response(RankingExplanationDraft(uncached, update_for(uncached)).payload())
    if fault in {"leading", "boundary"}:
        field = "boundaries" if fault == "boundary" else fault
        response[field] = full[field]
    elif fault == "draft_id":
        response["draft_id"] = "wrong-draft"
    else:
        response[fault] = "model cannot mutate frozen ranking"
    before = case.model_dump_json(), draft.update.model_dump_json()
    with pytest.raises((RequestValidationError, ValidationError)):
        draft.apply(RankingExplanations.model_validate(response))
    assert (case.model_dump_json(), draft.update.model_dump_json()) == before


def test_new_unranked_challenger_remains_available_when_leading_needs_refresh():
    case = scoped_case()
    case.runtime.update(evidence_decision_contracts=False, planner_case_memory=True)
    update = update_for(case)
    removed = update.portfolio.hypotheses.pop()
    draft = RankingExplanationDraft(case, update)
    payload = draft.payload()
    assert draft.leading_required
    assert removed.name in payload["mechanism_pool"]
    response = explanation_response(payload)
    response["leading"]["challenger"] = removed.name
    Draft202012Validator(payload["output_contract"]).validate(response)
    result = draft.apply(RankingExplanations.model_validate(response))
    assert result.ranking_update.leading_comparison.challenger == removed.name
    validate_leading(case, result)


@pytest.mark.parametrize("enabled", [False, None])
def test_disabled_flag_preserves_legacy_reuse_and_full_explanation_requests(enabled):
    case = scoped_case()
    if enabled is None:
        case.runtime.pop("scoped_evidence_updates")
    else:
        case.runtime["scoped_evidence_updates"] = enabled
    state(case).pop("comparison_dependency_cache")
    update = update_for(case)
    assert retain_comparisons(case, update)
    draft = RankingExplanationDraft(case, update)
    assert draft.leading_required and draft.pairs
    assert not draft.retained_comparisons
    update.ranking_update.candidate_bases[0].inference_gap = "A changed basis."
    assert not retain_comparisons(case, update)


@pytest.mark.parametrize("leading,boundary", [(False, False), (False, True), (True, False), (True, True)])
def test_scoped_reuse_respects_comparison_feature_flags(leading, boundary):
    case = scoped_case()
    case.runtime.update(hypothesis_tests=leading, evidence_decision_contracts=boundary)
    update = update_for(case)
    draft = RankingExplanationDraft(case, update)
    result = draft.apply(fresh_response(draft))
    assert (result.ranking_update.leading_comparison is not None) == leading
    assert bool(result.ranking_update.boundary_comparisons) == boundary
    validate_leading(case, result)
    validate_ranking(case, result)


@pytest.mark.parametrize("prompt", ["planner_explain_ranking.md", "result_impact.md"])
def test_prompt_changes_invalidate_comparisons_and_frozen_draft_identity(monkeypatch, prompt):
    case = scoped_case()
    update = update_for(case)
    original = RankingExplanationDraft(case, update)
    load_prompt = scoped_updates.load_prompt
    monkeypatch.setattr(scoped_updates, "load_prompt", lambda name: load_prompt(name) + (
        "\nChanged instruction." if name == prompt else ""))
    changed = RankingExplanationDraft(case, update)
    assert changed.draft_id != original.draft_id
    assert changed.leading_required and changed.pairs


def test_source_fact_hashes_ignore_review_attachment_but_track_corrected_facts():
    case = scoped_case()
    unit = observe(case)
    result = dict(result_id="X:source", evidence_refs=[unit.evidence_id], structured_results={"energy": 1.2})
    state(case).update(results=[deepcopy(result)], pending_results=[deepcopy(result)])
    before = scoped_updates.observation_versions(case)
    state(case)["pending_results"][0]["explanation_review"] = {"status": "reviewed"}
    assert scoped_updates.observation_versions(case) == before
    state(case)["pending_results"] = []
    assert scoped_updates.observation_versions(case) == before
    state(case)["results"][0]["structured_results"]["energy"] = 2.4
    assert scoped_updates.observation_versions(case) != before


@pytest.mark.parametrize("fault", ["observations", "source", "basis", "case_id", "prompt"])
def test_remembering_does_not_clear_invalidation_without_successful_explanation(monkeypatch, fault):
    case = scoped_case()
    unit = observe(case, "E:cited")
    records = scoped_updates.comparison_records(case)
    records["leading"]["evidence_refs"] = [unit.evidence_id]
    state(case)["results"] = [dict(result_id="X:source", evidence_refs=[unit.evidence_id], value=1)]
    state(case).pop("comparison_dependency_cache")
    state(case)["comparison_dependency_cache"] = scoped_updates.remember_comparisons(case)
    before = deepcopy(state(case)["comparison_dependency_cache"])
    if fault == "observations":
        observe(case)
    elif fault == "source":
        state(case)["results"][0]["value"] = 2
    elif fault == "basis":
        state(case)["assimilation"]["ranking_update"]["candidate_bases"][0]["inference_gap"] = "Changed basis."
    elif fault == "case_id":
        case.case_id = "another-case"
    else:
        load_prompt = scoped_updates.load_prompt
        monkeypatch.setattr(scoped_updates, "load_prompt", lambda name: load_prompt(name) + "\nChanged.")
    assert RankingExplanationDraft(case, update_for(case)).leading_required
    for _ in range(2):
        state(case)["comparison_dependency_cache"] = scoped_updates.remember_comparisons(case)
        assert state(case)["comparison_dependency_cache"]["leading"]["invalidated"]
        assert state(case)["comparison_dependency_cache"]["leading"]["observations"] == before["leading"]["observations"]
        case = CaseRun.model_validate_json(case.model_dump_json())
        assert RankingExplanationDraft(case, update_for(case)).leading_required


def test_successful_identical_explanation_refreshes_invalidated_cache_after_acceptance():
    case = scoped_case()
    old = deepcopy(scoped_updates.comparison_records(case))
    observe(case)
    # A structurally valid assimilation alone is not an explanation receipt.
    case = replace_state(apply_assimilation(case, update_for(case)), phase="assimilate")
    assert state(case)["comparison_dependency_cache"]["leading"]["invalidated"]
    draft = RankingExplanationDraft(case, update_for(case))
    response = explanation_response(draft.payload())
    response["leading"] = {key: old["leading"][key] for key in response["leading"]}
    for boundary in response["boundaries"]:
        included, excluded = draft.pairs[boundary["target_id"]]
        row = old[f"boundary:{included}|{excluded}"]
        boundary.update({key: row[key] for key in boundary if key != "target_id"})
    result = draft.apply(RankingExplanations.model_validate(response))
    assert "_scoped_comparison_receipt" not in result.portfolio.model_dump()
    case = apply_assimilation(case, result)
    assert scoped_updates.comparison_records(case) == old
    assert not state(case)["comparison_dependency_cache"]["leading"].get("invalidated")
    restored = replace_state(CaseRun.model_validate_json(case.model_dump_json()), phase="assimilate")
    assert not hasattr(restored.portfolio, "_scoped_comparison_receipt")
    assert retain_comparisons(restored, update_for(restored))


def test_explanation_for_old_sources_cannot_authorize_new_dependencies():
    case = scoped_case()
    observe(case)
    draft = RankingExplanationDraft(case, update_for(case))
    response = fresh_response(draft)
    observe(case, "E:after-draft")
    case = apply_assimilation(case, draft.apply(response))
    assert state(case)["comparison_dependency_cache"]["leading"]["invalidated"]
    case = replace_state(case, phase="assimilate")
    assert RankingExplanationDraft(case, update_for(case)).leading_required


def test_remember_is_idempotent_and_old_receipt_cannot_clear_later_invalidation():
    case = scoped_case()
    unit = observe(case)
    draft = RankingExplanationDraft(case, update_for(case))
    case = apply_assimilation(case, draft.apply(fresh_response(draft)))
    before = deepcopy(state(case)["comparison_dependency_cache"])
    assert scoped_updates.remember_comparisons(case) == before
    summary = unit.summary
    unit.summary = "Source fact corrected after the accepted explanation."
    state(case)["comparison_dependency_cache"] = scoped_updates.remember_comparisons(case)
    assert state(case)["comparison_dependency_cache"]["leading"]["invalidated"]
    unit.summary = summary
    state(case)["comparison_dependency_cache"] = scoped_updates.remember_comparisons(case)
    assert state(case)["comparison_dependency_cache"]["leading"]["invalidated"]
    assert RankingExplanationDraft(case, update_for(case)).leading_required


def test_restored_cached_argument_requires_impact_review_of_that_exact_content():
    from test_evidence_comparison import argument

    case = scoped_case()
    original = argument("S-same-id")
    case.mechanism_support_arguments = [original]
    cached = {"observations": scoped_updates.observation_versions(case)}
    original_basis = scoped_updates.scoped_audit_basis(case, original)
    case.mechanism_support_arguments = [original.model_copy(update={"warrant": "Different scientific argument B."})]
    unit = observe(case)
    review_batch(case, [unit.evidence_id])
    case.mechanism_support_arguments = [original]
    assert scoped_updates.scoped_audit_basis(case, original) == original_basis
    assert scoped_updates.current_receipt(case) is not None
    assert not scoped_updates.new_batch_allows_reuse(
        case, cached, support_id=original.support_id, labels=[original.target_label])


@pytest.mark.parametrize("mutation", ["audit_metadata", "missing_argument", "missing_receipt_hash"])
def test_impact_argument_binding_uses_scientific_content_not_audit_metadata(mutation):
    from mechcal.runtime.evidence_comparison import argument_content
    from test_evidence_comparison import argument

    case = scoped_case()
    original = argument("S-checked")
    case.mechanism_support_arguments = [original]
    cached = {"observations": scoped_updates.observation_versions(case)}
    unit = observe(case)
    review_batch(case, [unit.evidence_id])
    receipt = state(case)["evidence_impact_receipt"]
    assert receipt["arguments"][original.support_id] == scoped_updates.digest(argument_content(original))
    if mutation == "audit_metadata":
        original.audit_status = "accepted"
        original.audit_notes = ["Changed review metadata, unchanged scientific content."]
    elif mutation == "missing_argument":
        case.mechanism_support_arguments = []
    else:
        receipt["arguments"].pop(original.support_id)
    assert scoped_updates.new_batch_allows_reuse(
        case, cached, support_id=original.support_id, labels=[original.target_label]) == (mutation == "audit_metadata")
