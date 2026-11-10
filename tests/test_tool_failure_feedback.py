"""Failure classifications are operational feedback, never scientific evidence."""

import copy
import pytest

from mechcal.schemas.failures import FailureReport
from mechcal.tools.failure_feedback import member_failures, tool_failure


@pytest.mark.parametrize("code,kind,recoverable", [
    ("not_converged", "runtime_failed", True),
    ("numerical_validation_failed", "runtime_failed", True),
    ("timeout", "runtime_failed", True),
    ("parse_failed", "runtime_failed", False),
    ("invalid_args", "tool_args_invalid", True),
    ("tool_args_invalid", "tool_args_invalid", True),
    ("missing_input", "precondition_missing", True),
    ("precondition_missing", "precondition_missing", True),
    ("unsupported", "capability_unsupported", False),
    ("capability_unsupported", "capability_unsupported", False),
    ("partial_states", "partial_evidence", True),
])
def test_classification_and_legacy_serialization(code, kind, recoverable):
    report = tool_failure(code, "Exact reason", details={"workdir": "original", "failed_checks": ["example"]})
    assert (report.kind, report.recoverable) == (kind, recoverable)
    assert report.details["code"] == code
    assert report.details["failed_checks"] == ["example"]
    assert report.details["workdir"] == "original"
    assert report.details["remedy"] and not report.details["automatic_retry"]
    assert FailureReport.model_validate_json(report.model_dump_json()) == report
    assert FailureReport.model_validate({"kind": "runtime_failed", "message": "historical"}).is_failure


@pytest.mark.parametrize("status", ["failed", "partial"])
def test_member_convergence_is_recoverable(status):
    raw = {"search_attempts": [{"search_index": 1, "status": "failed",
            "failure_code": "not_converged", "reason": "GFN2 geometry optimization did not converge."}],
           "candidates": [{"member_id": "retained"}] if status == "partial" else []}
    before = copy.deepcopy(raw)
    failure = member_failures(raw, status)
    assert raw == before
    assert failure.recoverable
    assert failure.details["code"] == "not_converged"
    assert failure.details["member_failures"][0]["search_index"] == 1
    assert failure.details["successful_results_preserved"] == (status == "partial")
    assert failure.kind == ("partial_evidence" if status == "partial" else "runtime_failed")


def test_mixed_members_do_not_hide_unsupported_or_timeout():
    raw = {"points": [
        {"member_id": "a", "status": "success"},
        {"member_id": "b", "status": "failed", "failure_code": "unsupported", "reason": "Unsupported input"},
        {"member_id": "c", "status": "not_attempted", "failure_code": "timeout", "reason": "Route deadline"},
    ]}
    failure = member_failures(raw, "partial")
    assert failure.details["code"] == "mixed_failures"
    assert [f["recoverable"] for f in failure.details["member_failures"]] == [False, True]
    assert failure.recoverable
    assert not member_failures({"points": raw["points"][1:2]}, "failed").recoverable


def test_dftb_point_details_and_success_are_preserved():
    failed = member_failures({"points": [{"coordinate_value": 30., "converged": False,
        "failure_code": "timeout", "failure_message": "Saved backend reason"}]}, "partial")
    assert failed.message == "Saved backend reason"
    assert failed.details["member_failures"][0]["coordinate_value"] == 30.
    assert member_failures({}, "success") == FailureReport()
