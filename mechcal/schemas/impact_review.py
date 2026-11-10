"""A review of possible impact, never a mechanism decision or approval."""

from pydantic import Field

from mechcal.schemas.base import MechCALModel


class EvidenceImpactReview(MechCALModel):
    record_type: str = "EvidenceImpactReview"
    snapshot_id: str = Field(min_length=1)
    result_ids: list[str]
    checked_support_ids: list[str]
    affected_support_ids: list[str]
    affected_labels: list[str]
    affected_comparison_ids: list[str]
    global_review: bool
    reason: str = Field(min_length=1)
