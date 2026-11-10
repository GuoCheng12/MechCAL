"""Source arithmetic and workflow tests, not validation of chemical conclusions."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from mechcal.agents.evidence_context import source_checks
from mechcal.agents.planner import PlannerAgent
from mechcal.chem.solvent_comparisons import solvent_comparisons
from mechcal.runtime.orchestrator import _apply_support_audits
from mechcal.runtime.output_assembler import OutputAssembler
from mechcal.runtime.questions import apply_assimilation, public_payload, replace_state, state, validate_action
from mechcal.runtime.retry import SchemaRetryError
from mechcal.schemas import MechanismSupportAudit, PhotophysicsReview
from mechcal.schemas.question_loop import QuestionAssimilation, ResultReview
from mechcal.tools.amesp_microscopic import MICROSCOPIC_ROUTE_PROFILES, _solvation_proxy_payload
from test_argument_assessment import Client, argument_case, audit_response, review
from test_question_loop import NOTE, REGISTRY, QuestionClient, action, assimilation, loop, pending_case
from test_runtime import _evidence_unit


def solvent_records():
    return [dict(solvent=name, first_state=dict(state_index=1, excitation_energy_ev=value),
                 bright_state=dict(state_index=index + 1, oscillator_strength=value / 10))
            for index, (name, value) in enumerate(zip(("gas", "water", "ethanol"), (2.6688, 2.5567, 2.5522)))]


def test_pairwise_difference_is_not_full_range_and_preserves_conditions():
    records = solvent_records()
    before = deepcopy(records)
    result = _solvation_proxy_payload(MICROSCOPIC_ROUTE_PROFILES["run_solvation_polarity_proxy"], records, [])
    row = result["solvent_comparisons"]["first_excitation_energy_shift_ev"]
    assert result["first_excitation_energy_shift_ev"] == row["first_pair_delta"] == .1121
    assert row["available_value_range"] == .1166
    assert [p["solvent"] for p in row["points"]] == ["gas", "water", "ethanol"]
    assert row["unit"] == "eV"
    assert row["complete_for_returned_records"] is True
    assert records == before


@pytest.mark.parametrize("bad", [None, float("nan"), True, "2.6688"])
def test_missing_invalid_numbers_do_not_become_zero_or_change_comparison_pair(bad):
    records = solvent_records()
    records[0]["first_state"]["excitation_energy_ev"] = bad
    result = solvent_comparisons(records)["first_excitation_energy_shift_ev"]
    assert result["first_pair_delta"] is None
    assert result["available_value_range"] == .0045
    assert result["complete_for_returned_records"] is False
    assert solvent_comparisons(records[:1])["first_excitation_energy_shift_ev"]["available_value_range"] is None


def test_arithmetic_is_recomputed_for_old_records_and_reaches_both_agents():
    case = argument_case()
    evidence = _evidence_unit(evidence_id="source", round_id="R001", source_report_id="fixture",
                             agent_name="microscopic", family="state_ordering_brightness", summary=NOTE)
    evidence.metrics = {"computed_observables": {"solvent_records": solvent_records(),
                                                "first_excitation_energy_shift_ev": .9}}
    case = case.touch(evidence_ledger=case.evidence_ledger.model_copy(update={"items": [evidence]}),
                      mechanism_support_arguments=[case.mechanism_support_arguments[0].model_copy(
                          update={"observation_refs": ["source"]})])
    before = case.model_dump_json()
    checks = source_checks(case, [evidence])
    assert checks[0]["solvent_comparisons"]["first_excitation_energy_shift_ev"]["reported_delta_check"] == "mismatch"
    assert public_payload(case, REGISTRY, stage="planner_assimilate")["source_checks"] == checks
    client = Client([audit_response(case)])
    review(case, client)
    delivered = client.calls[0]["payload"]["source_checks"][0]
    assert delivered["solvent_comparisons"] == checks[0]["solvent_comparisons"]
    assert delivered["evidence_id"] != "source"  # request-local reference alias
    assert case.model_dump_json() == before


def test_advisory_review_cannot_change_levels_boundaries_ranking_or_export():
    case = argument_case()
    before = OutputAssembler().build_mechanism_predictions(case)
    audit = MechanismSupportAudit(support_id="S1", audit_status="unsupported",
                                  recommended_support_level="unsupported", boundary_patch="Changed")
    updated = _apply_support_audits(case, [audit])
    assert updated == case
    assert OutputAssembler().build_mechanism_predictions(updated) == before


def test_pending_objection_requires_response_but_allows_withdrawal_or_disagreement():
    case = replace_state(argument_case(), phase="assimilate", argument_review_pending=True,
                         argument_review=dict(status="reviewed", support_ids=["S1"]))
    raw = assimilation(public_payload(case, REGISTRY, stage="planner_assimilate"))
    raw["audit_responses"] = {}
    with pytest.raises(ValueError, match="every reviewed support_id"):
        apply_assimilation(case, QuestionAssimilation.model_validate(raw))
    raw["audit_responses"] = {"S1": "Withdrawn because the fixture does not support the proposed link."}
    withdrawn = apply_assimilation(case, QuestionAssimilation.model_validate(raw))
    assert withdrawn.mechanism_support_arguments == []
    raw["audit_responses"] = {"S1": "Disagree for the explicit source-grounded reason in the fixture."}
    raw["mechanism_support_arguments"] = [a.model_dump(mode="json") for a in case.mechanism_support_arguments]
    retained = apply_assimilation(case, QuestionAssimilation.model_validate(raw))
    assert retained.mechanism_support_arguments == case.mechanism_support_arguments


def test_action_cannot_bypass_a_pending_review():
    case = replace_state(argument_case(), argument_review_pending=True)
    with pytest.raises(ValueError, match="pending argument review"):
        validate_action(action(case.case_id, state(case)["revision"]), case, REGISTRY)


def audited_result_case(count):
    case = argument_case()
    results = [dict(state(pending_case("success"))["results"][0], result_id=f"X{i + 1}")
               for i in range(count)]
    case = replace_state(case, phase="assimilate", results=results, pending_results=results,
                         result_review=ResultReview(status="unavailable", reason=NOTE).model_dump(mode="json"))
    raw = assimilation(public_payload(case, REGISTRY, stage="planner_assimilate"))
    raw["questions"][0]["result_ids"] = [r["result_id"] for r in results]
    raw["mechanism_support_arguments"] = [a.model_dump(mode="json") for a in case.mechanism_support_arguments]
    case = apply_assimilation(case, QuestionAssimilation.model_validate(raw))
    return replace_state(case, phase="assimilate", argument_review_pending=True,
                         argument_review=dict(status="reviewed", support_ids=["S1"]))


@pytest.mark.parametrize("count", [1, 2])
def test_audit_response_rejects_replayed_dispositions_with_actionable_feedback(count):
    case = audited_result_case(count)
    before = case.model_dump_json()
    good = assimilation(public_payload(case, REGISTRY, stage="planner_assimilate"))
    good["mechanism_support_arguments"] = [a.model_dump(mode="json") for a in case.mechanism_support_arguments]
    bad = {**good, "dispositions": deepcopy(state(case)["assimilation"]["dispositions"])}
    client = Client([bad, good])
    events = []
    planner = PlannerAgent(llm_client=client, enable_llm_decisions=True, capability_registry=REGISTRY)
    update = planner.assimilate(case, validation_event_sink=events.append)
    assert len(client.calls) == 2
    feedback = client.calls[1]["schema_feedback"]
    assert "Required result_ids=[]" in feedback
    assert "return dispositions=[]" in feedback
    assert "unexpected=" in feedback and all(f"X{i + 1}" in feedback for i in range(count))
    assert "audit_responses" in feedback
    payload = client.calls[0]["payload"]
    assert payload["assimilation_context"] == dict(
        purpose="respond_to_argument_review", required_disposition_result_ids=[], result_review_is_current=False)
    contract = payload["output_contract"]["properties"]["dispositions"]
    assert contract["minItems"] == contract["maxItems"] == 0
    assert len(payload["execution_history"]) == count
    assert payload["result_review"] == state(case)["result_review"]
    assert payload["observations"] == public_payload(case, REGISTRY, stage="planner_assimilate")["observations"]
    assert events[0]["response"]["dispositions"] == bad["dispositions"]  # no silent dropping
    assert update.dispositions == []
    updated = apply_assimilation(case, update)
    assert updated.portfolio == case.portfolio
    assert updated.mechanism_support_arguments == case.mechanism_support_arguments
    assert updated.evidence_ledger == case.evidence_ledger
    assert state(updated)["results"] == state(case)["results"]
    assert state(updated)["questions"]["Q1"]["result_ids"] == [f"X{i + 1}" for i in range(count)]
    assert case.model_dump_json() == before


def test_persistently_replayed_dispositions_still_fail_after_three_attempts():
    case = audited_result_case(1)
    raw = assimilation(public_payload(case, REGISTRY, stage="planner_assimilate"))
    raw["dispositions"] = state(case)["assimilation"]["dispositions"]
    client = Client([raw])
    planner = PlannerAgent(llm_client=client, enable_llm_decisions=True, capability_registry=REGISTRY)
    before = case.model_dump_json()
    with pytest.raises(SchemaRetryError, match="return dispositions"):
        planner.assimilate(case)
    assert len(client.calls) == 3
    assert case.model_dump_json() == before


def test_assimilation_contract_distinguishes_new_results_and_does_not_leak_between_calls():
    pending = pending_case("success")
    pending = replace_state(pending, result_review=ResultReview(status="unavailable", reason=NOTE).model_dump(mode="json"))
    no_results = replace_state(argument_case(), phase="assimilate")
    responses = [assimilation(public_payload(c, REGISTRY, stage="planner_assimilate"))
                 for c in (pending, no_results)]
    client = Client(responses)
    planner = PlannerAgent(llm_client=client, enable_llm_decisions=True, capability_registry=REGISTRY)
    planner.assimilate(pending)
    planner.assimilate(no_results)
    first, second = [c["payload"] for c in client.calls]
    assert first["assimilation_context"] == dict(
        purpose="assimilate_results", required_disposition_result_ids=["X1"], result_review_is_current=True)
    assert first["output_contract"]["properties"]["dispositions"]["maxItems"] == 1
    branches = first["output_contract"]["$defs"]["ResultDisposition"]["anyOf"]
    assert [b["properties"]["result_id"]["const"] for b in branches] == ["X1"]
    assert second["assimilation_context"] == dict(
        purpose="update_investigation", required_disposition_result_ids=[], result_review_is_current=False)
    assert second["output_contract"]["properties"]["dispositions"]["maxItems"] == 0
    assert "enum" not in second["output_contract"]["$defs"]["ResultDisposition"]["properties"]["result_id"]
    assert "maxItems" not in QuestionAssimilation.model_json_schema()["properties"]["dispositions"]


def test_planner_cannot_award_itself_audit_approval():
    case = replace_state(argument_case(), phase="assimilate")
    raw = assimilation(public_payload(case, REGISTRY, stage="planner_assimilate"))
    arg = case.mechanism_support_arguments[0].model_dump(mode="json")
    arg.update(warrant="Changed offline argument.", audit_status="accepted", audit_notes=["Self approved"])
    raw["mechanism_support_arguments"] = [arg]
    result = apply_assimilation(case, QuestionAssimilation.model_validate(raw))
    assert result.mechanism_support_arguments[0].audit_status == "not_reviewed"
    assert result.mechanism_support_arguments[0].audit_notes == []


class ArgumentClient(QuestionClient):
    def complete_json(self, *, payload, system_prompt, schema_feedback=None):
        result = super().complete_json(payload=payload, system_prompt=system_prompt, schema_feedback=schema_feedback)
        if payload["stage"] == "planner_assimilate":
            result["mechanism_support_arguments"] = [dict(
                support_id="S1", target_label="RIM_RIR_RIV", support_level="weak",
                finding=NOTE, warrant=NOTE, boundary=NOTE, observation_refs=["input:smiles"],
            )]
        elif payload["stage"] == "planner_final_revision":
            result["update"]["mechanism_support_arguments"] = payload["previous_support_arguments"]
        return result


def argument_loop(tmp_path, monkeypatch, *, fail=False):
    o, _ = loop(tmp_path, rounds=2, unlimited=True)
    o.config = replace(o.config, enable_support_auditor=True, continue_on_arbiter_failure=True)
    client = ArgumentClient(rounds=2)
    o.deps.planner.llm_client = client
    audits = []
    def audit(case, **kwargs):
        audits.append((state(case)["phase"], case.current_round_id))
        if fail:
            raise TimeoutError("Offline unavailable review")
        return PhotophysicsReview(review_id="fixture", case_id=case.case_id,
            round_id=case.current_round_id, support_argument_audits=[dict(
                support_id="S1", audit_status="unsupported", audit_notes=["Offline inference objection."])])
    monkeypatch.setattr(o.deps.photophysics_arbiter, "review", audit)
    monkeypatch.setattr(o.deps.photophysics_arbiter, "review_results",
                        lambda _, **kw: ResultReview(status="unavailable", reason=NOTE))
    return o, client, audits


@pytest.mark.parametrize("fail", [False, True])
def test_argument_review_precedes_action_is_bounded_and_never_forces_tools(tmp_path, monkeypatch, fail):
    o, client, audits = argument_loop(tmp_path, monkeypatch, fail=fail)
    result = o.run(smiles="CCO", user_query=NOTE, case_id="advisory")
    assert result.status == "finalized"
    assert len(state(result)["results"]) == 2
    assert result.round_ids == ["R001", "R002", "R003"]
    argument_rounds = [rid for phase, rid in audits if phase == "argument_audit"]
    assert argument_rounds == ["R001", "R002", "R003"]
    assert len(audits) == 4  # three new source contexts, then the existing final audit
    for index, payload in enumerate(client.calls):
        if payload.get("argument_review_pending") and payload["stage"] == "planner_assimilate":
            assert payload["argument_review"]["status"] == ("unavailable" if fail else "reviewed")
            assert payload["pending_results"] == []
            assert client.calls[index + 1]["stage"] == "planner_question_act"
            assert client.calls[index + 1]["argument_review"]["planner_responses"] == {"S1": NOTE}
    # Unresolved feedback is not converted into a forced rank/grade or extra execution.
    assert result.portfolio.current == "RIM_RIR_RIV"
    assert result.mechanism_support_arguments[0].support_level == "weak"
    assert state(result)["argument_review_pending"] is False


def test_audit_response_retry_reaches_next_action_without_rerunning_tools(tmp_path, monkeypatch):
    o, _, audits = argument_loop(tmp_path, monkeypatch)

    class ReplayDispositionClient(ArgumentClient):
        def __init__(self):
            super().__init__(rounds=2)
            self.feedback = []

        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            self.feedback.append(schema_feedback)
            if schema_feedback is not None:
                assert "return dispositions=[]" in schema_feedback
            result = super().complete_json(payload=payload, system_prompt=system_prompt)
            if (payload["stage"] == "planner_assimilate" and payload["argument_review_pending"]
                    and payload["execution_history"] and schema_feedback is None):
                result["dispositions"] = assimilation({
                    **payload, "pending_results": payload["execution_history"],
                })["dispositions"]
            return result

    client = ReplayDispositionClient()
    o.deps.planner.llm_client = client
    result = o.run(smiles="CCO", user_query=NOTE, case_id="audit-response-retry")
    assert result.status == "finalized"
    assert len(state(result)["results"]) == 2
    assert result.round_ids == ["R001", "R002", "R003"]
    assert len(audits) == 4
    assert len([f for f in client.feedback if f]) == 2
    for i, feedback in enumerate(client.feedback):
        if feedback:
            assert client.calls[i + 1]["stage"] == "planner_question_act"
    failures = [json.loads(line) for line in
                (tmp_path / "audit-response-retry/history/planner_assimilation_failures.jsonl").read_text().splitlines()]
    assert len(failures) == 2
    assert all(f["attempt"] == 1 and f["response"]["dispositions"] for f in failures)


def test_resume_after_audit_does_not_repeat_review_or_execute_a_tool(tmp_path, monkeypatch):
    o, client, audits = argument_loop(tmp_path, monkeypatch)
    original = o.deps.planner.assimilate
    def interrupt(case, **kwargs):
        if state(case).get("argument_review_pending"):
            raise KeyboardInterrupt("After persisted audit, before Planner response")
        return original(case, **kwargs)
    monkeypatch.setattr(o.deps.planner, "assimilate", interrupt)
    with pytest.raises(KeyboardInterrupt):
        o.run(smiles="CCO", user_query=NOTE, case_id="resume-review")
    saved = json.loads((tmp_path / "resume-review/question_checkpoint.json").read_text())
    assert saved["runtime"]["question_loop"]["phase"] == "assimilate"
    assert saved["runtime"]["question_loop"]["argument_review_pending"]
    assert saved["runtime"]["question_loop"]["results"] == []
    monkeypatch.setattr(o.deps.planner, "assimilate", original)
    result = o.run(smiles="CCO", user_query=NOTE, case_id="resume-review", resume=True)
    assert audits.count(("argument_audit", "R001")) == 1
    assert len(state(result)["results"]) == 2
    before = deepcopy(client.calls)
    assert o.run(smiles="CCO", user_query=NOTE, case_id="resume-review", resume=True) == result
    assert client.calls == before


def test_failed_final_audit_does_not_reuse_earlier_same_round_review(tmp_path, monkeypatch):
    o, client, audits = argument_loop(tmp_path, monkeypatch)
    original = o.deps.photophysics_arbiter.review
    def review_with_final_failure(case, **kwargs):
        if state(case)["phase"] == "final_audit":
            raise TimeoutError("Offline final-only failure")
        return original(case, **kwargs)
    monkeypatch.setattr(o.deps.photophysics_arbiter, "review", review_with_final_failure)
    result = o.run(smiles="CCO", user_query=NOTE, case_id="final-only-failure")
    assert audits[-1] == ("argument_audit", "R003")
    assert state(result)["argument_review"]["status"] == "reviewed"
    assert state(result)["final_audit"] == {"status": "unavailable", "review": None}
    final = next(p for p in client.calls if p["stage"] == "planner_final_revision")
    assert final["final_internal_audit"]["status"] == "unavailable"


def test_private_evaluation_metadata_is_not_forwarded_to_either_agent():
    case = argument_case()
    private = {"rubric": "SECRET_RUBRIC_MARKER", "hidden_reference": "SECRET_REFERENCE_MARKER",
               "judge_feedback": "SECRET_JUDGE_MARKER"}
    case = case.touch(input=case.input.model_copy(update={"metadata": {**case.input.metadata, **private}}),
                      runtime={**case.runtime, **private})
    planner = public_payload(case, REGISTRY, stage="planner_assimilate")
    client = Client([audit_response(case)])
    review(case, client)
    for marker in private.values():
        assert marker not in json.dumps(planner)
        assert marker not in json.dumps(client.calls)
    assert client.calls[0]["payload"]["advisory_only"] is True
