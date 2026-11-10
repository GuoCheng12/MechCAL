"""Offline contracts; fixtures are not scientific measurements."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from mechcal.runtime import OrchestratorConfig
from mechcal.runtime.case_memory import CaseMemoryStore
from mechcal.runtime.decision_economy import expand_retained
from mechcal.runtime.incremental_audit import audit_basis, audit_incrementally, route_feedback, validate_routing
from mechcal.runtime.incremental_judgment import add_incremental_context
from mechcal.runtime.persistence import RunStore
from mechcal.runtime.planner_context import build_context
from mechcal.runtime.questions import apply_assimilation, public_payload, replace_state, state, validate_assimilation
from mechcal.runtime.retry import ModelContractError, RequestValidationError
from mechcal.schemas.mechanisms import AuditRouting, MechanismSupportAudit, PhotophysicsReview
from mechcal.schemas.memory import MemoryQuery
from mechcal.schemas.question_loop import JudgmentDelta, HypothesisScoreUpdate
from test_decision_economy import economy_start, next_update
from test_evidence_comparison import argument
from test_question_loop import REGISTRY


def prepared():
    c, u = economy_start()
    c = c.touch(runtime={**c.runtime, "incremental_judgment": True, "planner_case_memory": True})
    return replace_state(apply_assimilation(c, u), phase="assimilate")


def delta(c):
    u = next_update(c)
    u.delta = JudgmentDelta(base_revision=state(c)["revision"], reason="Retain unchanged scientific judgment.")
    u.portfolio = c.portfolio.model_copy(deep=True)
    u.portfolio.hypotheses = []
    u.questions = []
    u.mechanism_support_arguments = []
    u.ranking_update = None
    return u


def review(c, mode="no_change", ids=None):
    return PhotophysicsReview(
        review_id="offline-review", case_id=c.case_id, round_id="R001",
        support_argument_audits=[MechanismSupportAudit(support_id=a.support_id, audit_status="accepted")
                                for a in c.mechanism_support_arguments],
        routing=AuditRouting(mode=mode, affected_support_ids=ids or [], reason="Fixture routing."),
    )


def test_delta_preserves_records_is_idempotent_and_versioned():
    c = prepared()
    u = delta(c)
    before = deepcopy(c.model_dump())
    wire_size = len(u.model_dump_json())
    validate_assimilation(u, c)
    validate_assimilation(u, c)
    assert wire_size < len(u.model_dump_json()) * 0.6
    assert c.model_dump() == before
    assert u.portfolio == c.portfolio
    assert u.mechanism_support_arguments == c.mechanism_support_arguments
    applied = apply_assimilation(c, u)
    with pytest.raises(ModelContractError):
        apply_assimilation(replace_state(applied, phase="assimilate"), u)


@pytest.mark.parametrize("fault", ["stale", "foreign", "withdraw_overlap", "duplicate_score", "score_overlap", "disabled"])
def test_delta_bad_inputs_do_not_mutate_state(fault):
    c = prepared()
    u = delta(c)
    label = c.portfolio.hypotheses[0].name
    score = HypothesisScoreUpdate(name=label, confidence=0.4, differential_priority=0.8)
    if fault == "stale":
        u.delta.base_revision += 1
    elif fault == "foreign":
        u.delta.withdrawn_support_ids = ["foreign"]
    elif fault == "withdraw_overlap":
        u.delta.withdrawn_support_ids = [c.mechanism_support_arguments[0].support_id]
        u.mechanism_support_arguments = [c.mechanism_support_arguments[0].model_copy(deep=True)]
    elif fault == "duplicate_score":
        u.delta.hypothesis_scores = [score, score]
    elif fault == "score_overlap":
        u.delta.hypothesis_scores = [score]
        u.portfolio.hypotheses = [c.portfolio.hypotheses[0]]
    else:
        c.runtime["incremental_judgment"] = False
    original = deepcopy(u.model_dump())
    with pytest.raises(RequestValidationError):
        expand_retained(c, u)
    assert u.model_dump() == original


def test_scores_and_argument_upserts_do_not_rewrite_other_records():
    c = prepared()
    u = delta(c)
    first = c.portfolio.hypotheses[0]
    u.delta.hypothesis_scores = [HypothesisScoreUpdate(
        name=first.name, confidence=0.31, differential_priority=0.91)]
    changed = c.mechanism_support_arguments[0].model_copy(update={"boundary": "Revised scope."})
    u.mechanism_support_arguments = [changed]
    expand_retained(c, u)
    assert u.portfolio.hypotheses[0].confidence == 0.31
    assert u.portfolio.hypotheses[0].rationale == first.rationale
    assert u.portfolio.hypotheses[1:] == c.portfolio.hypotheses[1:]
    assert u.mechanism_support_arguments == [changed]


def test_explicit_withdrawal():
    c = prepared()
    u = delta(c)
    u.delta.withdrawn_support_ids = [a.support_id for a in c.mechanism_support_arguments]
    expand_retained(c, u)
    assert u.mechanism_support_arguments == []
    assert len(u.questions) == len(state(c)["questions"])


def test_duplicate_upsert_does_not_partly_expand_delta():
    c = prepared()
    u = delta(c)
    u.mechanism_support_arguments = [argument(), argument()]
    before = u.model_dump()
    with pytest.raises(RequestValidationError):
        expand_retained(c, u)
    assert u.model_dump() == before


def test_local_repair_and_explicit_escalation():
    c = prepared()
    sid = c.mechanism_support_arguments[0].support_id
    c = replace_state(c, argument_review_pending=True, argument_review={"support_ids": [sid]})
    u = delta(c)
    u.delta.scope = "local_audit"
    u.audit_responses = {sid: "Clarified the boundary."}
    u.mechanism_support_arguments = [c.mechanism_support_arguments[0].model_copy(update={"boundary": "Clarified scope."})]
    validate_assimilation(u, c)
    assert u.portfolio == c.portfolio
    invalid = delta(c)
    invalid.delta.scope = "local_audit"
    invalid.portfolio.hypotheses = [c.portfolio.hypotheses[0]]
    with pytest.raises(RequestValidationError, match="scope=global"):
        expand_retained(c, invalid)
    invalid.delta.scope = "global"
    expand_retained(c, invalid)


def test_routing_never_infers_approval_from_missing_review():
    c = prepared()
    c = c.touch(current_round_id="R001", photophysics_review=review(c))
    feedback = {"support_ids": [a.support_id for a in c.mechanism_support_arguments]}
    assert state(route_feedback(c, feedback))["phase"] == "act"
    unavailable = c.touch(photophysics_review=None)
    assert state(route_feedback(unavailable, feedback))["argument_review_pending"]
    sid = c.mechanism_support_arguments[0].support_id
    local = c.touch(photophysics_review=review(c, "local_audit", [sid]))
    routed = route_feedback(local, feedback)
    assert state(routed)["argument_review"]["support_ids"] == [sid]
    assert state(routed)["argument_review"]["routing"]["mode"] == "local_audit"


@pytest.mark.parametrize("fault", ["missing", "unknown", "ignored_patch", "ignored_warning", "ignored_route"])
def test_no_change_must_be_explicit_and_consistent(fault):
    c = prepared()
    r = review(c)
    if fault == "missing":
        r.routing = None
    elif fault == "unknown":
        r.routing.affected_support_ids = ["foreign"]
    elif fault == "ignored_patch":
        r.support_argument_audits[0].boundary_patch = "Correction required."
    elif fault == "ignored_warning":
        r.overclaim_warnings = ["Overclaim"]
    else:
        r.recommended_next_routes = ["route"]
    with pytest.raises(ModelContractError):
        validate_routing(r, c.mechanism_support_arguments)


def test_audit_reuse_is_exact_and_invalidated_by_new_context(tmp_path):
    c = prepared().touch(current_round_id="R001")
    store = RunStore(tmp_path, c.case_id)
    store.initialize()
    calls = []

    def audit(case, store, audit_case=None):
        calls.append([a.support_id for a in audit_case.mechanism_support_arguments])
        return case.touch(photophysics_review=review(audit_case))

    o = SimpleNamespace(_audit_current_round=audit)
    once = audit_incrementally(o, c, store)
    twice = audit_incrementally(o, once, store)
    assert len(calls) == 1
    assert twice.mechanism_support_arguments == c.mechanism_support_arguments
    changed = twice.model_copy(deep=True)
    changed.portfolio.hypotheses[0].confidence = 0.123
    audit_incrementally(o, changed, store)
    assert len(calls) == 2
    assert audit_basis(changed, changed.mechanism_support_arguments[0]) != audit_basis(c, c.mechanism_support_arguments[0])


def test_unavailable_audit_not_cached(tmp_path):
    c = prepared().touch(current_round_id="R001")
    store = RunStore(tmp_path, c.case_id)
    store.initialize()
    o = SimpleNamespace(_audit_current_round=lambda case, store, audit_case=None: case.touch(photophysics_review=None))
    updated = audit_incrementally(o, c, store)
    assert not state(updated).get("incremental_audit_cache")


def test_local_argument_change_reuses_other_audit_but_new_evidence_invalidates(tmp_path):
    c = prepared().touch(current_round_id="R001")
    c = c.touch(mechanism_support_arguments=[*c.mechanism_support_arguments, argument("S-other")])
    store = RunStore(tmp_path, c.case_id)
    store.initialize()
    calls = []

    def audit(case, store, audit_case=None):
        calls.append([a.support_id for a in audit_case.mechanism_support_arguments])
        return case.touch(photophysics_review=review(audit_case))

    o = SimpleNamespace(_audit_current_round=audit)
    once = audit_incrementally(o, c, store)
    changed = once.model_copy(deep=True)
    changed.mechanism_support_arguments[0].boundary += " Scope clarified."
    updated = audit_incrementally(o, changed, store)
    assert calls[-1] == [changed.mechanism_support_arguments[0].support_id]
    assert len(updated.photophysics_review.support_argument_audits) == 2
    assert len(updated.mechanism_support_arguments) == 2
    # Structural input is also an evidence dependency.
    new_source = updated.model_copy(deep=True)
    new_source.input.smiles = "CCC"
    audit_incrementally(o, new_source, store)
    assert len(calls[-1]) == 2


def test_contradictory_assessment_cannot_skip_global_reconsideration():
    from mechcal.schemas.mechanisms import MechanismArgumentAssessment
    c = prepared()
    r = review(c, "local_audit", [c.mechanism_support_arguments[0].support_id])
    r.support_argument_audits[0].argument_assessment = MechanismArgumentAssessment(
        finding_status="contradicted", finding_reason="Fixture counterevidence.",
        mechanism_link_status="articulated", mechanism_link_quote="fixture",
        mechanism_link_reason="Fixture link.", boundary_effect="counterevidence",
        boundary_reason="Fixture counterevidence.",
    )
    with pytest.raises(ModelContractError, match="global"):
        validate_routing(r, c.mechanism_support_arguments)
    r.routing.mode = "global"
    validate_routing(r, c.mechanism_support_arguments)


def test_argument_records_retrievable_and_context_still_shows_boundaries(tmp_path):
    c = prepared()
    store = RunStore(tmp_path, c.case_id)
    store.initialize()
    payload = public_payload(c, REGISTRY, stage="planner_assimilate")
    add_incremental_context(c, payload)
    projected = build_context(payload)
    original = c.mechanism_support_arguments[0]
    entry = projected["previous_support_arguments"][0]
    assert entry["boundary"] == original.boundary and "warrant" not in entry
    read = CaseMemoryStore(store, c).read(MemoryQuery(
        action="read_case_records", record_ids=[entry["memory_record_id"]]))
    assert original.warrant in str(read)


def test_feature_requires_existing_memory_and_economy():
    with pytest.raises(ValueError, match="incremental_judgment"):
        OrchestratorConfig(run_base_dir="unused", incremental_judgment=True)


def test_incremental_wire_contract_removes_legacy_choice():
    from mechcal.runtime.incremental_judgment import planner_prompt
    from mechcal.schemas.question_loop import QuestionAssimilation, FinalArgumentRevision
    c = prepared()
    for model in (QuestionAssimilation, FinalArgumentRevision):
        payload = {"output_contract": model.model_json_schema()}
        add_incremental_context(c, payload)
        contract = payload["output_contract"]
        target = contract.get("$defs", {}).get("QuestionAssimilation", contract)
        assert target["properties"]["retained_judgment"]["type"] == "null"
        assert target["properties"]["delta"] == {"$ref": "#/$defs/JudgmentDelta"}
        assert "delta" in target["required"]
    prompt = planner_prompt(c, "planner_assimilate.md")
    assert "# Decision economy" not in prompt
    assert "Return retained_judgment=null" in prompt
