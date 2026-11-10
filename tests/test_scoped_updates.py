"""Dependency/impact contracts using synthetic records, never scientific outcomes."""

from types import SimpleNamespace

import pytest

from mechcal.agents.evidence_context import relevant_evidence
from mechcal.runtime.incremental_audit import audit_basis, audit_incrementally
from mechcal.runtime.persistence import RunStore
from mechcal.runtime.questions import apply_result_review, replace_state, state
from mechcal.runtime.retry import ModelContractError, complete_recorded_json
from mechcal.runtime.scoped_updates import impact_context, validate_impact_review
from mechcal.schemas import CaseRun
from mechcal.schemas.impact_review import EvidenceImpactReview
from mechcal.schemas.question_loop import ResultReview
from test_evidence_comparison import argument
from test_incremental_judgment import prepared, review
from test_runtime import _evidence_unit


def scoped():
    c = prepared().touch(current_round_id="R002")
    return c.touch(runtime={**c.runtime, "scoped_evidence_updates": True},
                   mechanism_support_arguments=[argument("S-motion"), argument("S-ct", "ICT_TICT_CT")])


def new_batch(c, eid="E-synthetic", rid="X-synthetic"):
    c = c.model_copy(deep=True)
    c.evidence_ledger.items.append(_evidence_unit(
        evidence_id=eid, round_id="R002", source_report_id="synthetic", agent_name="microscopic",
        family="charge_localization", summary="Synthetic observation for dependency testing, not a real calculation."))
    result = {"result_id": rid, "question_ids": ["Q1"], "evidence_refs": [eid]}
    return replace_state(c, pending_results=[result], results=[*state(c).get("results", []), result])


def impact(c, *, affected=(), labels=(), global_review=False):
    context = impact_context(c)
    return ResultReview(
        interpretations=[{"result_id": r["result_id"], "question_ids": r["question_ids"],
                          "delivered": "Synthetic test observation.", "missing": "Not evaluated in this fixture.",
                          "boundary": "Not a scientific result.", "evidence_refs": r["evidence_refs"]}
                         for r in state(c).get("pending_results", [])],
        impact_review=EvidenceImpactReview(
            snapshot_id=context["snapshot_id"], result_ids=context["result_ids"],
            checked_support_ids=[a.support_id for a in c.mechanism_support_arguments],
            affected_support_ids=list(affected), affected_labels=list(labels), affected_comparison_ids=[],
            global_review=global_review, reason="Synthetic impact decision for contract testing only."))


def auditor(tmp_path, c):
    store = RunStore(tmp_path, c.case_id)
    store.initialize()
    calls = []

    def audit(case, store, audit_case=None):
        calls.append(audit_case)
        return case.touch(photophysics_review=review(audit_case))

    return SimpleNamespace(_audit_current_round=audit), store, calls


def test_uncited_new_result_reopens_affected_claim_only(tmp_path):
    c = scoped()
    o, store, calls = auditor(tmp_path, c)
    initial = audit_incrementally(o, c, store)
    fresh = new_batch(initial)
    updated = apply_result_review(fresh, impact(fresh, affected=["S-ct"]))
    result = audit_incrementally(o, updated, store)
    assert [a.support_id for a in calls[-1].mechanism_support_arguments] == ["S-ct"]
    assert "E-synthetic" in calls[-1].runtime["support_audit_scope"]["required_evidence_refs"]
    assert len(result.photophysics_review.support_argument_audits) == 2
    assert state(result)["incremental_audit_cache"]["S-motion"]["source_review_id"] == "offline-review"
    assert result.mechanism_support_arguments == initial.mechanism_support_arguments


@pytest.mark.parametrize("fault", ["missing", "stale", "unchecked", "unknown", "duplicate", "global"])
def test_invalid_impact_never_implies_no_change(fault):
    c = new_batch(scoped())
    r = impact(c)
    if fault == "missing":
        r.impact_review = None
    elif fault == "stale":
        r.impact_review.snapshot_id = "old-snapshot"
    elif fault == "unchecked":
        r.impact_review.checked_support_ids.pop()
    elif fault == "unknown":
        r.impact_review.affected_support_ids = ["other-case"]
    elif fault == "duplicate":
        r.impact_review.result_ids *= 2
    else:
        r.status = "unavailable"
    before = c.model_dump()
    with pytest.raises(ModelContractError):
        validate_impact_review(r, c)
    assert c.model_dump() == before


@pytest.mark.parametrize("mode", ["no_receipt", "global", "source_change", "argument_change", "foreign"])
def test_uncertain_or_changed_dependencies_are_reaudited(tmp_path, mode):
    c = scoped()
    o, store, calls = auditor(tmp_path, c)
    initial = audit_incrementally(o, c, store)
    fresh = new_batch(initial)
    if mode != "no_receipt":
        fresh = apply_result_review(fresh, impact(fresh, global_review=mode == "global"))
    if mode == "source_change":
        fresh.input.smiles = "CCC"
    elif mode == "argument_change":
        for a in fresh.mechanism_support_arguments:
            a.warrant += " Changed scientific explanation."
    elif mode == "foreign":
        fresh.case_id = "another-case"
    audit_incrementally(o, fresh, store)
    assert {a.support_id for a in calls[-1].mechanism_support_arguments} == {"S-ct", "S-motion"}
    assert "E-synthetic" in calls[-1].runtime["support_audit_scope"]["required_evidence_refs"]


def test_reuse_persists_across_batches_and_checkpoint(tmp_path):
    c = scoped()
    o, store, calls = auditor(tmp_path, c)
    current = audit_incrementally(o, c, store)
    for n in (1, 2):
        fresh = new_batch(current, f"E-{n}", f"X-{n}")
        current = apply_result_review(fresh, impact(fresh))
        current = CaseRun.model_validate_json(current.model_dump_json())
        current = audit_incrementally(o, current, store)
    assert len(calls) == 1
    assert set(state(current)["incremental_audit_cache"]["S-ct"]["observations"]) == {"E-1", "E-2"}
    assert "support_audit_scope" not in current.runtime  # Final audit is not restricted by an intermediate scope.


def test_scoped_audit_ignores_unrelated_confidence_not_cited_source(tmp_path):
    c = new_batch(scoped())
    c.mechanism_support_arguments[0].observation_refs = ["E-synthetic"]
    before = audit_basis(c, c.mechanism_support_arguments[0])
    changed = c.model_copy(deep=True)
    changed.portfolio.hypotheses[0].confidence = 0.123
    assert audit_basis(changed, changed.mechanism_support_arguments[0]) == before
    changed.evidence_ledger.items[0].summary += " Changed source finding."
    assert audit_basis(changed, changed.mechanism_support_arguments[0]) != before


def test_scoped_selection_includes_uncited_counterevidence(tmp_path):
    c = scoped()
    o, store, calls = auditor(tmp_path, c)
    old = audit_incrementally(o, c, store)
    fresh = new_batch(old)
    audit_incrementally(o, fresh, store)
    selected = relevant_evidence(calls[-1], calls[-1].mechanism_support_arguments)
    assert any(e.evidence_id == "E-synthetic" for e in selected)


def test_impact_on_unchanged_comparison_is_not_skipped(tmp_path):
    from test_scoped_ranking import scoped_case
    c = scoped_case()
    o, store, calls = auditor(tmp_path, c)
    initial = audit_incrementally(o, c, store)
    fresh = new_batch(initial)
    r = impact(fresh)
    r.impact_review.affected_comparison_ids = ["leading"]
    updated = apply_result_review(fresh, r)
    audit_incrementally(o, updated, store)
    assert len(calls) == 2
    assert "leading" in calls[-1].runtime["support_audit_scope"]["comparisons"]


def test_full_request_trace_records_before_exception_without_client_secrets():
    class Failing:
        api_key = "private-sentinel"
        def complete_json(self, **kwargs):
            raise TimeoutError("fixture timeout")
    events = []
    payload = {"scientific_data": [1, 2]}
    with pytest.raises(TimeoutError):
        complete_recorded_json(Failing(), event_sink=events.append, system_prompt="fixture",
                               payload=payload, schema_feedback=None)
    assert len(events) == 1 and events[0]["kind"] == "model_request"
    assert events[0]["request"]["payload"] == payload
    payload["scientific_data"].append(3)
    assert events[0]["request"]["payload"] != payload
    assert "private-sentinel" not in str(events)


def test_scoped_review_wire_contract_and_bounded_correction():
    from jsonschema import Draft202012Validator
    from mechcal.agents.photophysics_arbiter import PhotophysicsArbiterAgent
    from test_result_review_schema import Client
    from test_worker_explanation import case_with_explanation, review_payload

    c = case_with_explanation()
    c.runtime.update(scoped_evidence_updates=True, planner_strict_outputs=True)
    r = ResultReview.model_validate(review_payload())
    r.impact_review = impact(c).impact_review
    valid = r.model_dump(mode="json")
    invalid = {k: v for k, v in valid.items() if k != "impact_review"}
    client = Client([invalid, valid])
    before = c.model_dump_json()
    result = PhotophysicsArbiterAgent(llm_client=client).review_results(c)
    assert result == r and len(client.calls) == 2
    assert c.model_dump_json() == before
    schema = client.calls[0]["response_format"]["json_schema"]["schema"]
    Draft202012Validator(schema).validate(valid)
    assert "impact_review" in client.calls[1]["schema_feedback"]
