"""Validate that the final delivery matches Planner-authored arguments."""

from collections.abc import Sequence

from mechcal.runtime.retry import RequestValidationError
from mechcal.schemas.evidence import HypothesisPortfolio
from mechcal.schemas.final_evidence import FinalEvidenceSubmission
from mechcal.schemas.mechanisms import MechanismSupportArgument


def validate_final_evidence(
    portfolio: HypothesisPortfolio,
    arguments: Sequence[MechanismSupportArgument],
    submissions: Sequence[FinalEvidenceSubmission],
) -> None:
    ordered = portfolio.sorted_hypotheses()[:3]
    expected = [h.name for h in ordered]
    issues = []
    labels = [s.label for s in submissions]
    if len(labels) != len(set(labels)) or set(labels) != set(expected):
        issues.append({"path": ["evidence_submission"], "code": "final_evidence_labels",
                       "message": f"Submit one evidence decision per final label: {expected}. Array order is irrelevant."})
    for i, submission in enumerate(submissions):
        ids = {a.support_id for a in arguments if a.target_label == submission.label}
        if submission.status == "submitted":
            provided = submission.support_ids
            if len(set(provided)) != len(provided) or set(provided) != ids:
                message = (f"{submission.label}: support_ids must match its current arguments "
                           f"{sorted(ids)} without duplicates. Add missing finding/warrant/boundary records "
                           "in update.mechanism_support_arguments, or explicitly choose no_evidence.")
                issues.append({"path": ["evidence_submission", i], "code": "final_evidence_arguments",
                               "message": message})
                if set(provided) - ids:
                    issues.append({"path": ["update", "mechanism_support_arguments"],
                                   "code": "final_evidence_arguments", "message": message})
        elif ids:
            message = (f"{submission.label}: no_evidence conflicts with current arguments "
                       f"{sorted(ids)}. Submit them or explicitly withdraw unjustified arguments.")
            for path in (["evidence_submission", i], ["update", "withdrawn_support_ids"]):
                issues.append({"path": path, "code": "final_evidence_conflict", "message": message})
    if issues:
        raise RequestValidationError(issues)
