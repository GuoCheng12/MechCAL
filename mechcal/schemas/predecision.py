"""Advisory comparisons made before a ranking is proposed."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvidenceComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_a: str
    candidate_b: str
    outcome: Literal["favors_a", "favors_b", "unresolved", "compatible"]
    observation: str = Field(min_length=1)
    interpretation: str = Field(min_length=1)
    remaining_gap: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)


class PredecisionReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    comparisons: list[EvidenceComparison] = Field(max_length=6)
    unassessable: str = ""


class ComparisonResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    review_id: str
    response: str = Field(min_length=1, description="How the advisory comparison affects judgment, or why it is rejected. Agreement is not required.")
