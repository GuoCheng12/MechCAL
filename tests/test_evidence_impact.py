"""Source routing and atomic updates only; scripted text is not scientific validation."""

from copy import deepcopy

import pytest

from mechcal.agents.worker_explanation import explain_result
from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime.case_memory import CaseMemoryStore
from mechcal.runtime.evidence_impact import impact_context, validate_impact_responses
from mechcal.runtime.judgment_commands import materialize_judgment
from mechcal.runtime.persistence import RunStore
from mechcal.runtime.planner_context import build_context
from mechcal.runtime.questions import apply_assimilation, public_payload, replace_state, state
from mechcal.runtime.retry import RequestValidationError
from mechcal.runtime.structured_output import PlannerOutputContract
from mechcal.schemas import CaseRun
from mechcal.schemas.evidence_comparison import MechanismEffect
from mechcal.schemas.judgment_commands import HypothesisChange, JudgmentProposal
from mechcal.schemas.memory import MemoryQuery
from test_judgment_commands import command, planner, repair_fields
from test_question_loop import NOTE, REGISTRY, assimilation, pending_case
from test_worker_explanation import ExplanationClient, fixture_result

LABEL = "ICT_TICT_CT"


def implications(data):
    for finding in data["findings"]:
        finding["mechanism_implications"] = [{"label": LABEL, "reason": NOTE}]


def impact_case():
    case = pending_case("success")
    case.runtime["incremental_judgment"] = True
    req, result = fixture_result()
    explanation = explain_result(ExplanationClient(implications), req, case, result,
                                 REGISTRY.require(req.capability_id), None)
    packet = {**state(case)["results"][0], "evidence_refs": ["X1:E1"],
              "worker_explanation": explanation.model_dump(mode="json")}
    case.evidence_ledger.items.extend(result.evidence_units)
    return replace_state(case, results=[packet], pending_results=[packet])


def effect(response="Retain current priority: the observation does not distinguish the candidates."):
    return dict(label=LABEL, effect="non_discriminating", reason=NOTE,
                evidence_refs=["X1:E1"], judgment_response=response)


def request(case, effects=None):
    data = assimilation(public_payload(case, REGISTRY, stage="planner_assimilate"))
    data["dispositions"][0]["mechanism_effects"] = effects if effects is not None else [effect()]
    return command(case, dispositions=data["dispositions"])


def test_worker_can_flag_outside_assigned_question_without_seeing_ranking():
    req, result = fixture_result()
    case = pending_case("success")
    case.input.metadata["hidden_reference"] = "PRIVATE_SENTINEL"
    client = ExplanationClient(implications)
    before = result.model_dump()
    explanation = explain_result(client, req, case, result, REGISTRY.require(req.capability_id), None)
    payload = client.calls[0]
    assert payload["mechanism_pool"] == list(MECHANISM_POOL)
    assert "portfolio" not in payload and "PRIVATE_SENTINEL" not in str(payload)
    assert explanation.findings[0].mechanism_implications[0].label == LABEL
    assert result.model_dump() == before


@pytest.mark.parametrize("labels", [["invented"], [LABEL, LABEL]])
def test_worker_unknown_and_duplicate_labels_rejected(labels):
    def change(data):
        data["findings"][0]["mechanism_implications"] = [dict(label=x, reason=NOTE) for x in labels]
    req, result = fixture_result()
    with pytest.raises(ValueError, match="distinct mechanism-pool"):
        explain_result(ExplanationClient(change), req, pending_case("success"), result,
                       REGISTRY.require(req.capability_id), None)


def test_context_exposes_old_judgment_without_requiring_old_citation():
    case = impact_case()
    hypothesis = next(h for h in case.portfolio.hypotheses if h.name == LABEL)
    hypothesis.evidence_refs = []
    hypothesis.rationale = "No computation was available in the previous revision."
    before = deepcopy(case.model_dump())
    payload = public_payload(case, REGISTRY, stage="planner_assimilate")
    context = build_context(payload)
    target = context["evidence_impact_context"]["targets"][0]
    from mechcal.runtime.case_memory import pointer
    assert pointer(context, target["current_hypothesis_context_path"])["rationale"] == hypothesis.rationale
    assert target["worker_suggestions"][0]["evidence_refs"] == ["X1:E1"]
    assert context["pending_results"][0]["worker_explanation"]["findings"]
    assert case.model_dump() == before


def test_repeated_findings_group_by_result_label_without_dropping_reasons():
    case = impact_case()
    packet = state(case)["pending_results"][0]
    finding = deepcopy(packet["worker_explanation"]["findings"][0])
    finding["finding_id"] = "F2"
    packet["worker_explanation"]["findings"].append(finding)
    targets = impact_context(case)["targets"]
    assert len(targets) == 1 and len(targets[0]["worker_suggestions"]) == 2


@pytest.mark.parametrize("effects", [[], [effect(None)], [effect(" ")], [effect(), effect()]])
def test_missing_or_duplicate_response_rejected_without_state_change(effects):
    case = impact_case()
    before = deepcopy(case.model_dump())
    update = materialize_judgment(case, request(case, effects))
    with pytest.raises(RequestValidationError) as error:
        validate_impact_responses(case, update)
    assert all(i["repair_paths"] for i in error.value.issues)
    assert case.model_dump() == before


def test_disagreement_does_not_force_score_change_and_survives_persistence(tmp_path):
    case = impact_case()
    update = materialize_judgment(case, request(case))
    accepted = apply_assimilation(case, update)
    assert accepted.portfolio == case.portfolio
    assert accepted.evidence_ledger == case.evidence_ledger
    assert CaseRun.model_validate_json(accepted.model_dump_json()) == accepted
    store = RunStore(tmp_path, case.case_id)
    store.initialize()
    store.write_json(f"questions/revision_{update.revision:04d}.json", state(accepted)["assimilation"])
    memory = CaseMemoryStore(store, accepted)
    record = memory.read(MemoryQuery(action="read_case_records", record_ids=[f"judgment:{update.revision}"]))
    assert "judgment_response" in str(record)
    assert impact_context(accepted)["targets"] == []


def test_fact_can_be_updated_without_changing_scores_or_unrelated_labels():
    case = impact_case()
    old = next(h for h in case.portfolio.hypotheses if h.name == LABEL)
    wire = request(case)
    wire.hypotheses = [HypothesisChange(name=LABEL, rationale="New recorded computation is now available.",
                                      evidence_refs=["X1:E1"])]
    accepted = apply_assimilation(case, materialize_judgment(case, wire))
    new = next(h for h in accepted.portfolio.hypotheses if h.name == LABEL)
    assert new.rationale != old.rationale and new.confidence == old.confidence
    assert new.differential_priority == old.differential_priority
    assert [h for h in accepted.portfolio.hypotheses if h.name != LABEL] == [
        h for h in case.portfolio.hypotheses if h.name != LABEL]


def test_legacy_no_implications_and_case_isolation():
    first = impact_case()
    second = pending_case("success")
    assert impact_context(second)["targets"] == []
    assert impact_context(first)["targets"]
    legacy = MechanismEffect.model_validate(dict(label=LABEL, effect="unavailable", reason=NOTE))
    assert legacy.judgment_response is None


def test_missing_response_uses_scoped_planner_repair():
    case = impact_case()
    bad = request(case, []).model_dump(mode="json")
    agent, client = planner(case, [bad, repair_fields(mechanism_effects=[effect()])])
    update = agent.assimilate(case)
    assert len(client.calls) == 2
    assert client.calls[1]["payload"]["targets"][0]["field"] == "mechanism_effects"
    assert update.portfolio == case.portfolio
    assert update.dispositions[0].mechanism_effects[0].judgment_response


def test_strict_wire_roundtrip_keeps_response_and_null_legacy_field():
    case = impact_case()
    proposal = JudgmentProposal.model_validate(request(case).model_dump()).model_dump(mode="json")
    contract = PlannerOutputContract({"output_contract": JudgmentProposal.model_json_schema()})
    assert contract.decode(contract.encode(proposal)) == proposal
    proposal["dispositions"][0]["mechanism_effects"][0]["judgment_response"] = None
    assert contract.decode(contract.encode(proposal)) == proposal


def test_new_label_without_previous_judgment_is_discoverable():
    case = impact_case()
    case.portfolio.hypotheses = [h for h in case.portfolio.hypotheses if h.name != LABEL]
    assert impact_context(case)["targets"][0]["current_hypothesis"] is None


def test_affected_argument_warrant_is_pinned_despite_memory_compaction():
    from test_incremental_judgment import prepared
    case = prepared()
    label = case.mechanism_support_arguments[0].target_label
    packet = deepcopy(state(impact_case())["pending_results"][0])
    packet["worker_explanation"]["findings"][0]["mechanism_implications"][0]["label"] = label
    case = replace_state(case, results=[packet], pending_results=[packet])
    payload = public_payload(case, REGISTRY, stage="planner_assimilate")
    payload["incremental_update"] = {"enabled": True}
    compact = build_context(payload)
    assert compact["previous_support_arguments"][0]["warrant"] == case.mechanism_support_arguments[0].warrant


@pytest.mark.parametrize("status", ["failed", "precondition_missing", "unsupported"])
def test_failed_calculation_cannot_create_implications(status):
    req, result = fixture_result()
    result.status = status
    explanation = explain_result(ExplanationClient(implications), req, pending_case(status), result,
                                 REGISTRY.require(req.capability_id), None)
    assert explanation.findings == []
