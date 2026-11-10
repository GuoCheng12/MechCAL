"""Final-delivery contracts. Synthetic text is not a scientific result."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from mechcal.runtime.final_delivery import (
    MAX_PLANNER_ATTEMPTS, FinalDeliveryIncomplete, advance, apply_patch, audit_digest,
    candidate_case, manifest, open_draft, preflight, preserved_action,
)
from mechcal.runtime.persistence import RunStore
from mechcal.runtime.questions import replace_state, state
from mechcal.runtime.retry import PlannerCommandRejected, RequestValidationError
from mechcal.runtime.structured_output import PlannerOutputContract
from mechcal.public_ids import public_case_id_from_metadata
from mechcal.schemas import CaseRun, MechanismSupportAudit, PhotophysicsReview
from mechcal.schemas.final_delivery import FinalDeliveryPatch, MissingEvidence
from test_delivery_and_final_revision import final_fixture
from test_question_loop import REGISTRY, NOTE


def fixture():
    case, _ = final_fixture()
    return replace_state(case.touch(current_round_id="R002"), round_index=2)


def completion(case, draft):
    return FinalDeliveryPatch(
        no_evidence=[MissingEvidence(label=h.name, reason=NOTE) for h in case.portfolio.sorted_hypotheses()[:3]
                     if not any(a.target_label == h.name for a in draft.arguments)],
        audit_responses={sid: NOTE for sid in draft.reviewed_support_ids}, final_answer_draft=NOTE,
    )


def saved(case, draft):
    return replace_state(case, final_delivery=draft.model_dump(mode="json"))


def test_small_contract_does_not_ask_for_runtime_linkage():
    schema = FinalDeliveryPatch.model_json_schema()
    assert not {"portfolio", "final_action", "evidence_submission", "continuation_assessment"} & schema["properties"].keys()
    patch = completion(fixture(), open_draft(fixture()))
    wire = PlannerOutputContract({"output_contract": schema})
    assert wire.decode(wire.encode(patch.model_dump(mode="json"))) == patch.model_dump(mode="json")


def test_accepted_decision_and_evidence_are_not_rewritten():
    case = fixture()
    before = case.model_dump_json()
    draft = apply_patch(case, open_draft(case), completion(case, open_draft(case)))
    rows, errors = manifest(case, draft)
    assert not errors and len(rows) == 3
    assert rows[0].support_ids == ["S1"]
    assert not preflight(case, draft, REGISTRY)
    assert preserved_action(case, draft).continuation_assessment == preserved_action(case, open_draft(case)).continuation_assessment
    assert case.model_dump_json() == before


@pytest.mark.parametrize("fault", ["case", "revision", "ranking", "evidence", "questions", "configuration"])
def test_draft_rejects_changed_context(fault):
    case = fixture()
    draft = open_draft(case)
    changed = case.model_copy(deep=True)
    if fault == "case":
        changed.case_id = "other"
    elif fault == "revision":
        changed.runtime["question_loop"]["revision"] += 1
    elif fault == "ranking":
        changed.portfolio.hypotheses[0].confidence = 0.001
    elif fault == "evidence":
        changed.evidence_ledger.items[0].summary = "Changed observation"
    elif fault == "questions":
        changed.runtime["question_loop"]["questions"] = {}
    else:
        changed.runtime["question_loop"]["configuration_hash"] = "changed"
    with pytest.raises(FinalDeliveryIncomplete, match="changed accepted state"):
        open_draft(saved(changed, draft))


def test_edit_invalidates_audit_but_not_the_saved_decision():
    case = fixture()
    draft = open_draft(case)
    original = audit_digest(case, draft.arguments)
    revised = draft.arguments[0].model_copy(update={"warrant": "New explicit physical link."})
    patched = apply_patch(case, draft, FinalDeliveryPatch(arguments=[revised]))
    assert audit_digest(case, patched.arguments) != original
    assert patched.arguments[0].audit_status == "not_reviewed"
    assert preserved_action(case, patched) == preserved_action(case, draft)


def test_unchanged_argument_reuses_audit_not_model_supplied_approval():
    case = fixture()
    draft = open_draft(case)
    forged = draft.arguments[0].model_copy(update={"audit_status": "rejected", "audit_notes": ["forged"]})
    patched = apply_patch(case, draft, FinalDeliveryPatch(arguments=[forged]))
    assert patched.arguments == draft.arguments
    assert audit_digest(case, patched.arguments) == audit_digest(case, draft.arguments)


def test_cross_label_and_foreign_refs_are_reported_together():
    case = fixture()
    draft = open_draft(case)
    bad = draft.arguments[0].model_copy(update={"target_label": "ICT_TICT_CT", "observation_refs": ["other-case"]})
    with pytest.raises(RequestValidationError) as error:
        apply_patch(case, draft, FinalDeliveryPatch(arguments=[bad], withdrawn_support_ids=["foreign"]))
    assert {i["code"] for i in error.value.issues} == {"cross_label_id", "invalid_observation", "unknown_withdrawal"}
    assert draft.arguments[0].target_label == "RIM_RIR_RIV"


def test_new_label_argument_uses_distinct_id_even_when_sharing_observation():
    case = fixture()
    draft = open_draft(case)
    new = draft.arguments[0].model_copy(update={"support_id": "S2", "target_label": "ICT_TICT_CT"})
    patched = apply_patch(case, draft, FinalDeliveryPatch(arguments=[new]))
    rows, _ = manifest(case, patched)
    assert {r.label: r.support_ids for r in rows} == {"RIM_RIR_RIV": ["S1"], "ICT_TICT_CT": ["S2"]}


def test_withdrawal_requires_explicit_no_evidence_not_fabricated_support():
    case = fixture()
    draft = open_draft(case)
    no = MissingEvidence(label="RIM_RIR_RIV", reason="The proposed link is unjustified.")
    with pytest.raises(RequestValidationError, match="withdraw"):
        apply_patch(case, draft, FinalDeliveryPatch(no_evidence=[no]))
    patched = apply_patch(case, draft, FinalDeliveryPatch(withdrawn_support_ids=["S1"], no_evidence=[no]))
    assert not patched.arguments
    rows, _ = manifest(case, patched)
    assert rows[0].status == "no_evidence"


def test_historical_review_keeps_feedback_but_requires_new_bound_audit():
    case = fixture()
    review = PhotophysicsReview(review_id="old", case_id=case.case_id, round_id=case.current_round_id)
    case = replace_state(case, final_audit={"status": "reviewed", "review": review.model_dump(mode="json")})
    draft = open_draft(case)
    assert draft.audit == review and draft.audit_digest is None




def runtime_fixture(tmp_path, monkeypatch, *, audit=True):
    from test_judgment_commands import command_loop
    o, _ = command_loop(tmp_path, monkeypatch)
    o.config = replace(o.config, final_delivery_v2=True, enable_support_auditor=audit)
    case = fixture()
    case = case.touch(runtime={**case.runtime, "final_delivery_v2": True})
    store = RunStore(tmp_path / "runs", case.case_id)
    def patch(planner, current, draft, *, reserve_attempt, **kwargs):
        reserve_attempt()
        return apply_patch(current, draft, completion(current, draft))
    monkeypatch.setattr("mechcal.agents.final_delivery.complete_delivery", patch)
    audits = []
    def review(current, **kwargs):
        audits.append(current)
        return PhotophysicsReview(review_id="new", case_id=public_case_id_from_metadata(current.input.metadata), round_id=current.current_round_id,
            support_argument_audits=[MechanismSupportAudit(support_id=a.support_id, audit_status="accepted")
                                    for a in current.mechanism_support_arguments])
    monkeypatch.setattr(o.deps.photophysics_arbiter, "review", review)
    return o, case, store, audits


@pytest.mark.parametrize("audit", [True, False])
def test_delivery_state_machine_and_atomic_resume(tmp_path, monkeypatch, audit):
    o, case, store, audits = runtime_fixture(tmp_path, monkeypatch, audit=audit)
    initial = case.model_dump(mode="json")
    steps = []
    for _ in range(8):
        case = advance(o, case, store)
        steps.append(state(case)["final_delivery"]["stage"])
        if state(case)["phase"] == "finish":
            break
        # Reconstruct on every phase boundary, as after a process interruption.
        case = CaseRun.model_validate_json((store.root / "question_checkpoint.json").read_text())
        assert case.mechanism_support_arguments == fixture().mechanism_support_arguments
    assert state(case)["phase"] == "finish"
    assert len(audits) == int(audit)
    assert len(case.runtime["final_evidence_submission"]) == 3
    assert state(case)["revision"] == initial["runtime"]["question_loop"]["revision"]
    assert case.portfolio.model_dump(mode="json") == initial["portfolio"]
    assert case.evidence_ledger.model_dump(mode="json") == initial["evidence_ledger"]
    assert "audit" in steps if audit else "audit" not in steps


def test_saved_valid_edit_survives_later_transport_failure(tmp_path, monkeypatch):
    o, case, store, _ = runtime_fixture(tmp_path, monkeypatch)
    draft = open_draft(case)
    revised = draft.arguments[0].model_copy(update={"warrant": "Corrected before transport failure."})
    draft = apply_patch(case, draft, FinalDeliveryPatch(arguments=[revised]))
    case = saved(case, draft)
    case = advance(o, case, store)
    monkeypatch.setattr(o.deps.photophysics_arbiter, "review", lambda *a, **kw: (_ for _ in ()).throw(TimeoutError("injected")))
    with pytest.raises(TimeoutError):
        advance(o, case, store)
    recovered = CaseRun.model_validate_json((store.root / "question_checkpoint.json").read_text())
    assert open_draft(recovered).arguments[0].warrant == revised.warrant
    assert recovered.mechanism_support_arguments[0].warrant != revised.warrant
    assert open_draft(recovered).audit_attempts == 1


def test_budget_cannot_reset_through_resume(tmp_path, monkeypatch):
    o, case, store, _ = runtime_fixture(tmp_path, monkeypatch)
    draft = open_draft(case).model_copy(update={"planner_attempts": MAX_PLANNER_ATTEMPTS})
    case = saved(case, draft)
    with pytest.raises(FinalDeliveryIncomplete, match="allowance exhausted"):
        advance(o, case, store)
    recovered = CaseRun.model_validate_json((store.root / "question_checkpoint.json").read_text())
    assert open_draft(recovered).planner_attempts == MAX_PLANNER_ATTEMPTS


def test_reconsideration_pauses_without_fake_final_or_new_tools(tmp_path, monkeypatch):
    o, case, store, _ = runtime_fixture(tmp_path, monkeypatch)
    draft = open_draft(case).model_copy(update={"reconsideration_reason": "Need a new calculation."})
    case = saved(case, draft)
    with pytest.raises(FinalDeliveryIncomplete, match="scientific reconsideration"):
        advance(o, case, store)
    assert "final_evidence_submission" not in case.runtime


def test_model_format_retries_use_same_total_allowance(tmp_path, monkeypatch):
    from mechcal.agents.final_delivery import complete_delivery
    class BadClient:
        def complete_json(self, **kwargs):
            return {"nonsense": True}
    o, _, _, _ = runtime_fixture(tmp_path, monkeypatch)
    p = o.deps.planner
    p.llm_client = BadClient()
    case = fixture()
    used = []
    with pytest.raises(PlannerCommandRejected):
        complete_delivery(p, case, open_draft(case), store=RunStore(tmp_path, case.case_id),
                          reserve_attempt=lambda: used.append(True), event_sink=lambda _: None)
    assert len(used) == p.runtime.retry_runner.policy.max_attempts


def test_complete_offline_loop_new_delivery_and_resume_no_extra_calls(tmp_path, monkeypatch):
    from test_judgment_commands import command_loop
    o, client = command_loop(tmp_path, monkeypatch)
    o.config = replace(o.config, final_delivery_v2=True)
    original = client.complete_json
    def respond(**kwargs):
        payload = kwargs["payload"]
        if payload["stage"] != "planner_complete_delivery":
            return original(**kwargs)
        client.calls.append(payload)
        details = payload["final_delivery"]
        return FinalDeliveryPatch(
            no_evidence=[MissingEvidence(label=label, reason=NOTE) for label in details["top3"]
                         if not any(a["target_label"] == label for a in payload["previous_support_arguments"])],
            audit_responses={sid: NOTE for sid in details["reviewed_support_ids"]},
            final_answer_draft=NOTE,
        ).model_dump(mode="json")
    monkeypatch.setattr(client, "complete_json", respond)
    result = o.run(smiles="CCO", user_query=NOTE, case_id="command-loop")
    assert result.status == "finalized"
    assert len(state(result)["results"]) == 1
    assert state(result)["final_delivery_receipt"]["manifest"] == result.runtime["final_evidence_submission"]
    assert not any(p["stage"] == "planner_final_revision" for p in client.calls)
    count = len(client.calls)
    assert o.run(smiles="CCO", user_query=NOTE, case_id="command-loop", resume=True) == result
    assert len(client.calls) == count


def test_new_argument_after_audit_gets_reaudited(tmp_path, monkeypatch):
    o, case, store, audits = runtime_fixture(tmp_path, monkeypatch)
    case = advance(o, case, store)
    case = advance(o, case, store)
    assert len(audits) == 1
    draft = open_draft(case)
    new = draft.arguments[0].model_copy(update={"support_id": "new", "warrant": "Additional bounded argument."})
    patched = apply_patch(case, draft, FinalDeliveryPatch(arguments=[new],
        audit_responses={sid: NOTE for sid in draft.reviewed_support_ids}))
    case = saved(case, patched)
    case = advance(o, case, store)
    assert open_draft(case).stage == "audit"
    case = advance(o, case, store)
    assert len(audits) == 2
    assert {a.support_id for a in audits[-1].mechanism_support_arguments} == {"S1", "new"}


def test_new_bound_review_is_reused_when_no_argument_changes():
    case = fixture()
    review = PhotophysicsReview(review_id="old", case_id=public_case_id_from_metadata(case.input.metadata), round_id=case.current_round_id,
        support_argument_audits=[MechanismSupportAudit(support_id="S1", audit_status="accepted")])
    case = replace_state(case, final_audit={"status": "reviewed", "review": review.model_dump(mode="json"),
                                          "content_digest": audit_digest(case, case.mechanism_support_arguments)})
    draft = open_draft(case)
    assert draft.audit_digest == audit_digest(case, draft.arguments)


def test_audit_identity_uses_existing_public_id_contract():
    from mechcal.runtime.final_delivery import audit_matches
    case = fixture()
    case.input.metadata["public_case_id"] = "anonymous-example"
    review = PhotophysicsReview(review_id="public", case_id="anonymous-example", round_id=case.current_round_id,
        support_argument_audits=[MechanismSupportAudit(support_id="S1", audit_status="accepted")])
    assert audit_matches(case, case.mechanism_support_arguments, review)
    for changes in ({"case_id": case.case_id}, {"round_id": "R999"}, {"support_argument_audits": []},
                    {"support_argument_audits": review.support_argument_audits * 2}):
        assert not audit_matches(case, case.mechanism_support_arguments, review.model_copy(update=changes))


def test_production_auditor_output_binds_to_private_delivery_case():
    from mechcal.runtime.final_delivery import audit_matches
    from test_argument_assessment import Client, argument_case, review
    from test_auditor_contract_recovery import source_response
    case = argument_case().touch(current_round_id="R002")
    case.input.metadata["public_case_id"] = "public-case-for-audit"
    result = review(case, Client([source_response(case)]))
    assert result.case_id != case.case_id
    assert audit_matches(case, case.mechanism_support_arguments, result)
