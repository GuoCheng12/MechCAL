"""Offline audit contracts, not claims about chemical correctness."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from mechcal.agents.photophysics_arbiter import (
    DisabledPhotophysicsWebSearch,
    PhotophysicsArbiterAgent,
    _safe_evidence_payload,
)
from mechcal.runtime.incremental_audit import validate_routing
from mechcal.runtime.question_loop import _failure_kind
from mechcal.runtime.questions import state
from mechcal.runtime.retry import AuditorReviewRejected, ModelContractError
from mechcal.schemas import CaseRun, EvidenceUnit
from mechcal.schemas.mechanisms import PhotophysicsReview
from test_argument_assessment import Client, argument_case, audit_response, review
from test_judgment_commands import command_loop


def source_response(case, field="warrant"):
    response = audit_response(case)
    assessment = response["support_argument_audits"][0]["argument_assessment"]
    assessment.pop("mechanism_link_quote")
    assessment["mechanism_link_source_field"] = field
    return response


@pytest.mark.parametrize("field", ["finding", "warrant", "boundary"])
def test_source_field_resolves_original_text_and_preserves_assessment(field):
    case = argument_case()
    before = case.model_dump_json()
    response = source_response(case, field)
    client = Client([response])
    result = review(case, client)
    assessment = result.support_argument_audits[0].argument_assessment
    assert assessment.mechanism_link_quote == getattr(case.mechanism_support_arguments[0], field)
    assert assessment.mechanism_link_source_field == field
    assert assessment.mechanism_link_reason == response["support_argument_audits"][0]["argument_assessment"]["mechanism_link_reason"]
    contract = client.calls[0]["payload"]["argument_assessment_contract"]
    assert all("mechanism_link_quote" not in branch["properties"] for branch in contract["anyOf"])
    assert all("mechanism_link_source_field" in branch["required"] for branch in contract["anyOf"])
    assert PhotophysicsReview.model_validate_json(result.model_dump_json()) == result
    assert before == case.model_dump_json()


@pytest.mark.parametrize("fault", ["foreign_field", "foreign_id", "duplicate_id", "wrong_quote", "foreign_quote", "empty_link"])
def test_invalid_source_reference_is_not_guessed_or_approved(fault):
    case = argument_case()
    response = source_response(case)
    row = response["support_argument_audits"][0]
    a = row["argument_assessment"]
    if fault == "foreign_field":
        a["mechanism_link_source_field"] = "another_argument.warrant"
    elif fault == "foreign_id":
        row["support_id"] = "S_OTHER"
    elif fault == "duplicate_id":
        response["support_argument_audits"].append(deepcopy(row))
    elif fault == "wrong_quote":
        a["mechanism_link_quote"] = case.mechanism_support_arguments[0].finding
    elif fault == "foreign_quote":
        a["mechanism_link_quote"] = "Invented text."
    else:
        a["mechanism_link_source_field"] = None
    client = Client([response])
    before = case.model_dump_json()
    with pytest.raises(AuditorReviewRejected):
        review(case, client, attempts=2)
    assert len(client.calls) == 2
    assert before == case.model_dump_json()


def test_missing_link_remains_missing_with_no_runtime_invention():
    case = argument_case()
    response = source_response(case, None)
    response["support_argument_audits"][0]["argument_assessment"]["mechanism_link_status"] = "missing"
    result = review(case, Client([response]))
    assert result.support_argument_audits[0].argument_assessment.mechanism_link_quote == ""


def test_correction_feedback_retains_previous_response_and_all_errors():
    case = argument_case()
    first = source_response(case, "unknown")
    second = source_response(case)
    second["support_argument_audits"][0]["recommended_support_level"] = "strong"
    client = Client([first, second, source_response(case)])
    result = review(case, client, attempts=3)
    feedback = json.loads(client.calls[2]["schema_feedback"])
    assert len(feedback["validation_history"]) == 2
    assert "mechanism_link_source_field" in feedback["validation_history"][0]
    assert "advisory-only" in feedback["validation_history"][1]
    assert feedback["rejected_response"] == second
    assert result.support_argument_audits[0].recommended_support_level is None


@pytest.mark.parametrize("metric_key", ["computed_observables", "parsed_observables", "tool_args", "modeled_systems", "new_tool_observables"])
def test_scientific_values_are_never_truncated_by_list_or_dictionary_position(metric_key):
    points = [dict(
        point_index=i, status="success", energy_hartree=-100.0 - i,
        requested_distance_angstrom=2.0 + i, description="A full condition " * 20,
        interaction_energy_ev=-0.1 * i, actual_distance_angstrom=2.0 + i,
        contacts_under_3_5_angstrom=i, fragment_atom_indices=list(range(56)),
        limitations=["Fixture limitation " + str(n) for n in range(7)],
    ) for i in range(6)]
    unit = EvidenceUnit(
        evidence_id="E_SCAN", agent_name="microscopic", capability_id="microscopic.scan_intermolecular_distance",
        round_id="R001", source_report_id="report", context="Fixture conditions.", summary="Fixture only.",
        claim="Long scientific claim " * 20, basis="computed", family="raw_artifact_inspection",
        metrics={metric_key: {"points": points}, "prepared_file_paths": ["private/path"], "smiles": "CCO"},
    )
    original = unit.model_dump_json()
    payload = _safe_evidence_payload(unit)
    assert payload["metrics"][metric_key]["points"] == points
    assert payload["claim"] == unit.claim
    assert "prepared_file_paths" not in payload["metrics"] and "smiles" not in payload["metrics"]
    assert unit.model_dump_json() == original


def test_large_orbital_projection_remains_explicit_not_silent_truncation():
    unit = EvidenceUnit(
        evidence_id="E_ORBITAL", agent_name="microscopic", capability_id="fixture",
        round_id="R001", source_report_id="report", context="Fixture conditions.", summary="Fixture only.",
        claim="Fixture only.", basis="computed", family="raw_artifact_inspection",
        metrics={"parsed_observables": {"state_index": 1, "components": [
            dict(from_orbital=i, to_orbital=i + 1, coefficient=0.1) for i in range(12)
        ]}},
    )
    projected = _safe_evidence_payload(unit)["metrics"]["parsed_observables"]
    assert len(projected["components"]) == 8
    assert projected["component_table_view"]["omitted_rows"] == 4
    assert len(unit.metrics["parsed_observables"]["components"]) == 12


@pytest.mark.parametrize("mode", ["local_audit", "no_change", "global"])
def test_mixed_boundary_does_not_override_auditor_routing(mode):
    case = argument_case()
    response = source_response(case)
    row = response["support_argument_audits"][0]
    row["argument_assessment"]["boundary_effect"] = "mixed"
    response["routing"] = dict(mode=mode, affected_support_ids=["S1"] if mode == "local_audit" else [],
                               reason="Both directions already bounded; no new ranking concern.")
    case.runtime["incremental_judgment"] = True
    before = case.model_dump_json()
    client = Client([response])
    result = review(case, client)
    assert result.routing.mode == mode
    assert "include routing matching routing_contract" in client.calls[0]["payload"]["output_contract"]
    assert before == case.model_dump_json()


@pytest.mark.parametrize("change", [
    {"finding_status": "contradicted"}, {"mechanism_link_status": "unsound"},
    {"boundary_effect": "counterevidence"},
])
def test_actual_contradiction_still_requires_global_response(change):
    case = argument_case()
    response = audit_response(case)
    response["support_argument_audits"][0]["argument_assessment"].update(change)
    r = PhotophysicsReview(review_id="test", case_id=case.case_id, round_id="R001", **response,
                          routing=dict(mode="local_audit", affected_support_ids=["S1"], reason="Fixture."))
    with pytest.raises(ModelContractError, match="global"):
        validate_routing(r, case.mechanism_support_arguments)
    r.routing.mode = "global"
    validate_routing(r, case.mechanism_support_arguments)


@pytest.mark.parametrize("error", [TimeoutError("offline"), OSError("offline"), TypeError("offline program error")])
def test_non_contract_failures_are_not_retried_or_relabelled_as_audit_rejections(error):
    class Broken(Client):
        def complete_json(self, **kwargs):
            self.calls.append(kwargs)
            raise error

    client = Broken([])
    with pytest.raises(RuntimeError) as raised:
        review(argument_case(), client, attempts=3)
    assert not isinstance(raised.value, AuditorReviewRejected)
    assert raised.value.__cause__ is error
    assert len(client.calls) == 1


@pytest.mark.parametrize("continue_on_failure", [False, True])
def test_auditor_pause_and_explicit_resume_preserve_completed_tools(tmp_path, monkeypatch, continue_on_failure):
    o, _ = command_loop(tmp_path, monkeypatch)
    o.config = replace(o.config, continue_on_arbiter_failure=continue_on_failure)
    original = o.deps.photophysics_arbiter.review
    bad_client = Client([{"routing": {"mode": "invalid"}}])
    rejected = PhotophysicsArbiterAgent(llm_client=bad_client, web_search=DisabledPhotophysicsWebSearch(), max_attempts=2)
    blocked = True

    def audit(case, **kwargs):
        if blocked and state(case).get("results"):
            return rejected.review(case, **kwargs)
        return original(case, **kwargs)

    monkeypatch.setattr(o.deps.photophysics_arbiter, "review", audit)
    from test_question_loop import NOTE

    with pytest.raises(AuditorReviewRejected) as error:
        o.run(smiles="CCO", user_query=NOTE, case_id="audit-pause")
    assert _failure_kind(error.value) == "support_audit_rejected"
    assert len(bad_client.calls) == 2
    root = tmp_path / "audit-pause"
    saved = CaseRun.model_validate_json((root / "question_checkpoint.json").read_text())
    assert saved.status == "paused" and state(saved)["phase"] == "argument_audit"
    assert len(state(saved)["results"]) == 1 and not state(saved)["pending_results"]
    assert not saved.runtime["termination"]["scientific_completion"]
    assert saved.runtime["termination"]["requires_explicit_resume"]
    prior_tools = (root / "history/tool_events.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        o.run(smiles="CCO", user_query=NOTE, case_id="audit-pause")
    blocked = False
    result = o.run(smiles="CCO", user_query=NOTE, case_id="audit-pause", resume=True)
    assert result.status == "finalized"
    assert result.evidence_ledger == saved.evidence_ledger
    assert (root / "history/tool_events.jsonl").read_bytes() == prior_tools
    assert result.runtime["worker_call_counts"]["macro"] == 1


def test_wrapped_rejection_cannot_be_treated_as_service_unavailability():
    from mechcal.runtime.api_failures import is_review_service_failure

    error = AuditorReviewRejected(["Invalid audit fixture."])
    wrapped = RuntimeError("Outer operation")
    wrapped.__cause__ = error
    assert not is_review_service_failure(error)
    assert not is_review_service_failure(wrapped)
