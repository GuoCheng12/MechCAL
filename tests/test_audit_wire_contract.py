"""Audit interface regressions; no scientific correctness claims."""

from copy import deepcopy
import json

import pytest

from mechcal.agents.photophysics_arbiter import _argument_assessment_contract
from mechcal.runtime.audit_output import audit_output_schema, validate_audit_output
from mechcal.runtime.incremental_audit import validate_routing
from mechcal.runtime.retry import AuditorReviewRejected, ModelContractError
from mechcal.schemas.mechanisms import PhotophysicsReview
from test_argument_assessment import Client, argument_case, review
from test_auditor_contract_recovery import source_response


def strict_case():
    case = argument_case().touch(current_round_id="R001")
    case.runtime.update(planner_strict_outputs=True, incremental_judgment=True)
    return case


def wire(case):
    raw = source_response(case) if case.mechanism_support_arguments else {"support_argument_audits": []}
    raw.update(review_id="fixture", case_id=case.case_id, round_id=case.current_round_id,
               routing=dict(mode="no_change", affected_support_ids=[], reason="Accepted as written."))
    raw = PhotophysicsReview.model_validate(raw).model_dump(mode="json")
    raw.pop("context_notes")
    for audit in raw["support_argument_audits"]:
        audit["argument_assessment"].pop("mechanism_link_quote")
    return raw


def schema(ids=("S1",)):
    return audit_output_schema(dict(
        require_argument_assessments=True, require_complete_support_audit=True,
        require_explicit_audit_routing=True, allowed_support_ids=list(ids),
        argument_assessment_contract=_argument_assessment_contract(),
    ))


def test_live_schema_is_sent_and_no_change_remains_consistent_after_persistence():
    case = strict_case()
    before = case.model_dump_json()
    client = Client([wire(case)])
    result = review(case, client)
    spec = client.calls[0]["response_format"]["json_schema"]
    assert spec["strict"] and spec["name"] == "support_audit_v1"
    assert spec["schema"] == client.calls[0]["payload"]["output_contract"]
    assert not result.overclaim_warnings
    assert result.context_notes
    validate_routing(result, case.mechanism_support_arguments)
    assert PhotophysicsReview.model_validate_json(result.model_dump_json()) == result
    assert before == case.model_dump_json()


@pytest.mark.parametrize("fault", ["missing_assessment", "null_assessment", "missing_routing",
                                  "unknown_support", "extra", "grade", "boundary", "missing_source"])
def test_wire_rejects_incomplete_or_unauthorized_fields(fault):
    raw = wire(strict_case())
    row = raw["support_argument_audits"][0]
    if fault == "missing_assessment":
        row.pop("argument_assessment")
    elif fault == "null_assessment":
        row["argument_assessment"] = None
    elif fault == "missing_routing":
        raw.pop("routing")
    elif fault == "unknown_support":
        row["support_id"] = "OTHER"
    elif fault == "extra":
        raw["invented"] = "ignored?"
    elif fault == "grade":
        row["recommended_support_level"] = "strong"
    elif fault == "boundary":
        row["boundary_patch"] = "Replacement science."
    else:
        row["argument_assessment"].pop("mechanism_link_source_field")
    with pytest.raises(ModelContractError):
        validate_audit_output(raw, schema())


def test_multiple_schema_errors_are_reported_together_and_model_repairs_them():
    case = strict_case()
    bad = wire(case)
    bad.pop("routing")
    bad["support_argument_audits"][0].pop("argument_assessment")
    client = Client([bad, wire(case)])
    result = review(case, client, attempts=2)
    feedback = json.loads(client.calls[1]["schema_feedback"])
    assert "routing" in feedback["validation_history"][0]
    assert "argument_assessment" in feedback["validation_history"][0]
    assert result.routing.mode == "no_change"


@pytest.mark.parametrize("mode", ["no_change", "local_audit"])
def test_axis_route_cannot_bypass_global_routing(mode):
    case = strict_case()
    raw = wire(case)
    raw["routing"].update(mode=mode, affected_support_ids=["S1"] if mode == "local_audit" else [])
    raw["coverage_axes"] = [dict(axis_id="test", label="Test", status="missing",
                                  rationale="A new investigation.", recommended_routes=["some.route"])]
    with pytest.raises(ModelContractError, match="global"):
        validate_routing(PhotophysicsReview.model_validate(raw), case.mechanism_support_arguments)


def test_warning_conflict_names_field_and_never_discards_the_warning():
    case = strict_case()
    raw = wire(case)
    raw["overclaim_warnings"] = ["Unresolved comparison concern."]
    before = deepcopy(raw)
    client = Client([raw])
    with pytest.raises(AuditorReviewRejected, match="overclaim_warnings must be empty"):
        review(case, client, attempts=2)
    assert raw == before
    repaired = deepcopy(raw)
    repaired["routing"].update(mode="global", reason="Planner should reconsider the comparison.")
    result = review(case, Client([repaired]))
    assert result.overclaim_warnings == raw["overclaim_warnings"]


def test_empty_argument_scope_is_supported_without_fabricated_audits():
    case = strict_case().touch(mechanism_support_arguments=[])
    raw = wire(case)
    raw["support_argument_audits"] = []
    validate_audit_output(raw, schema(ids=()))
    result = review(case, Client([raw]))
    assert result.support_argument_audits == []
