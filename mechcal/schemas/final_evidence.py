"""Explicit final evidence delivery, not an evidence grade."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class SubmittedEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(min_length=1)
    status: Literal["submitted"]
    support_ids: list[str] = Field(min_length=1, description="All current argument IDs for this final label.")


class NoSubmittedEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(min_length=1)
    status: Literal["no_evidence"]
    reason: str = Field(min_length=1, pattern=r"\S", description=(
        "Why no argument is justified for this label. This is not evidence or a support score."
    ))


FinalEvidenceSubmission = Annotated[
    SubmittedEvidence | NoSubmittedEvidence, Field(discriminator="status")
]
