"""Contract tests only; scripted assessments are not chemical ground truth."""

from copy import deepcopy

import pytest

from mechcal.agents.photophysics_arbiter import DisabledPhotophysicsWebSearch, PhotophysicsArbiterAgent
from mechcal.eval.prompts import load_eval_prompt
from mechcal.eval.zero_shot_v3 import PROMPT_NAME, ZeroShotV3Judge, compose_v3_judge_prompt
from mechcal.schemas.mechanisms import MechanismSupportArgument, MechanismSupportAudit
from test_question_loop import ready_case


def argument_case():
    case = ready_case()
    argument = MechanismSupportArgument(
        support_id="S1", target_label="RIM_RIR_RIV", support_level="partial",
        finding="An offline fixture observation.",
        warrant="A conditional mechanism connection, supplied only as a contract fixture.",
        boundary="Occurrence remains unconfirmed.", observation_refs=["input:smiles"],
    )
    return case.touch(mechanism_support_arguments=[argument],
                      runtime={**case.runtime, "final_support_audit_round": case.current_round_id})


def assessment(case):
    return dict(
        finding_status="verified", finding_reason="Offline fixture source assessment.",
        mechanism_link_status="articulated", mechanism_link_quote=case.mechanism_support_arguments[0].warrant,
        mechanism_link_reason="Offline fixture link assessment.",
        boundary_effect="scope_limit", boundary_reason="Uncertainty, not contradictory data.",
    )


def audit_response(case):
    return dict(support_argument_audits=[dict(
        support_id="S1", audit_status="accepted",
        audit_notes=["Offline fixture."], argument_assessment=assessment(case),
    )])


class Client:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def is_configured(self):
        return True

    def complete_json(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return deepcopy(self.responses[min(len(self.calls) - 1, len(self.responses) - 1)])


def review(case, client, attempts=1):
    return PhotophysicsArbiterAgent(llm_client=client, web_search=DisabledPhotophysicsWebSearch(),
                                  max_attempts=attempts).review(case)


def test_three_assessments_survive_normalization_and_persistence_without_mutating_case():
    case = argument_case()
    before = case.model_dump_json()
    client = Client([audit_response(case)])
    result = review(case, client)
    recorded = result.support_argument_audits[0]
    assert recorded.argument_assessment.mechanism_link_quote == case.mechanism_support_arguments[0].warrant
    assert recorded.argument_assessment.boundary_effect == "scope_limit"
    assert MechanismSupportAudit.model_validate_json(recorded.model_dump_json()) == recorded
    assert client.calls[0]["payload"]["require_argument_assessments"]
    contract = client.calls[0]["payload"]["argument_assessment_contract"]
    assert all("finding_status" in branch["properties"] for branch in contract["anyOf"])
    assert before == case.model_dump_json()


@pytest.mark.parametrize("change,expected", [
    ({"mechanism_link_quote": "Invented explanation supplied by the auditor"}, "must come"),
    ({"mechanism_link_quote": ""}, "exact submitted quote"),
    ({"mechanism_link_reason": "  "}, "mechanism_link_reason"),
])
def test_inconsistent_audit_gets_feedback_not_an_automatic_scientific_repair(change, expected):
    case = argument_case()
    bad = audit_response(case)
    bad["support_argument_audits"][0]["argument_assessment"].update(change)
    client = Client([bad, audit_response(case)])
    result = review(case, client, attempts=2)
    assert len(client.calls) == 2
    assert expected in client.calls[1]["schema_feedback"]
    assert result.support_argument_audits[0].recommended_support_level is None


def test_missing_assessment_retries_are_bounded_and_do_not_silently_accept():
    case = argument_case()
    bad = audit_response(case)
    bad["support_argument_audits"][0].pop("argument_assessment")
    client = Client([bad])
    with pytest.raises(RuntimeError, match="argument_assessment is required"):
        review(case, client, attempts=2)
    assert len(client.calls) == 2


def test_missing_link_can_be_reported_without_fabricating_one_or_failing():
    case = argument_case()
    raw = audit_response(case)
    row = raw["support_argument_audits"][0]
    row.update(audit_status="unsupported")
    row["argument_assessment"].update(mechanism_link_status="missing", mechanism_link_quote="")
    assert review(case, Client([raw])).support_argument_audits[0].argument_assessment.mechanism_link_status == "missing"
    assert case.mechanism_support_arguments[0].support_level == "partial"


def test_partial_finding_and_mixed_boundary_are_not_automatically_zeroed():
    case = argument_case()
    raw = audit_response(case)
    raw["support_argument_audits"][0]["argument_assessment"].update(
        finding_status="partly_verified", boundary_effect="mixed")
    assert review(case, Client([raw])).support_argument_audits[0].recommended_support_level is None


@pytest.mark.parametrize("changes", [dict(recommended_support_level="partial"), dict(boundary_patch="New text")])
def test_advisory_auditor_cannot_assign_grades_or_rewrite_planner(changes):
    case = argument_case()
    raw = audit_response(case)
    raw["support_argument_audits"][0].update(changes)
    client = Client([raw, audit_response(case)])
    review(case, client, attempts=2)
    assert len(client.calls) == 2
    assert "advisory-only" in client.calls[1]["schema_feedback"]


@pytest.mark.parametrize("status", ["contradicted", "unverified"])
def test_factual_objection_is_allowed_without_a_grade(status):
    case = argument_case()
    raw = audit_response(case)
    raw["support_argument_audits"][0]["argument_assessment"]["finding_status"] = status
    result = review(case, Client([raw]))
    assert result.support_argument_audits[0].argument_assessment.finding_status == status
    assert case.mechanism_support_arguments[0].support_level == "partial"


def test_historical_audits_still_load_without_invented_assessments():
    old = MechanismSupportAudit.model_validate(dict(support_id="old", audit_status="accepted"))
    assert old.argument_assessment is None


def test_judge_sources_share_the_same_interpretation_policy_without_rewriting_inputs():
    policy = load_eval_prompt("evidence_interpretation_policy.md")
    assert ZeroShotV3Judge(None).prompt == load_eval_prompt(PROMPT_NAME)
    assert compose_v3_judge_prompt("Tool-source base prompt.\n") == "Tool-source base prompt.\n"
    assert ZeroShotV3Judge(None, semantic_scope=True).prompt.endswith(policy)
    assert compose_v3_judge_prompt("Tool-source base prompt.", semantic_scope=True).endswith(policy)
    assert "Do not invent that connection" in policy
    assert "same evidential scope" in policy
    assert "contradicts" in policy
    assert "No score is prescribed" in policy
