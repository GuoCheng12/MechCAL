"""Small final-delivery edits; accepted science and linkage remain runtime-owned."""

from typing import Literal

from pydantic import Field, field_validator

from mechcal.schemas.base import MechCALModel
from mechcal.schemas.mechanisms import MechanismSupportArgument, PhotophysicsReview


class MissingEvidence(MechCALModel):
    record_type: str = "MissingEvidence"
    label: str
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Explain the evidence gap.")
        return value.strip()


class FinalDeliveryPatch(MechCALModel):
    record_type: str = "FinalDeliveryPatch"
    action: Literal["complete_delivery"] = "complete_delivery"
    arguments: list[MechanismSupportArgument] = Field(default_factory=list)
    withdrawn_support_ids: list[str] = Field(default_factory=list)
    no_evidence: list[MissingEvidence] = Field(default_factory=list)
    audit_responses: dict[str, str] = Field(default_factory=dict)
    final_answer_draft: str | None = None
    reconsideration_reason: str | None = Field(
        default=None, description="Explain if this ranking or stopping decision needs scientific reconsideration. "
        "The runtime pauses delivery rather than silently retaining an invalid decision or calling tools here.",
    )


class FinalDeliveryDraft(MechCALModel):
    record_type: str = "FinalDeliveryDraft"
    case_id: str
    base_revision: int
    base_digest: str
    stage: Literal["edit", "audit", "respond", "commit"] = "edit"
    arguments: list[MechanismSupportArgument]
    no_evidence: list[MissingEvidence] = Field(default_factory=list)
    final_answer_draft: str | None = None
    reconsideration_reason: str | None = None
    audit: PhotophysicsReview | None = None
    audit_digest: str | None = None
    audit_responses: dict[str, str] = Field(default_factory=dict)
    reviewed_support_ids: list[str] = Field(default_factory=list)
    planner_attempts: int = 0
    audit_attempts: int = 0
    generation: int = 0
