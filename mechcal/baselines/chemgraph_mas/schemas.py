from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mechcal.mechanism_pool import MECHANISM_POOL

ClaimStatus = Literal[
    "supported",
    "proxy_supported",
    "candidate_requires_validation",
    "underdetermined_candidate",
    "weakened",
]
SupportStrength = Literal["strong", "partial", "weak_or_proxy", "unsupported"]


class MechanismPredictionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    rank: int = Field(ge=1, le=3)
    confidence: float = Field(ge=0.0, le=1.0)
    claim_status: ClaimStatus
    support_strength: SupportStrength
    evidence: list[str] = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)

    @field_validator("label")
    @classmethod
    def _known_label(cls, value: str) -> str:
        normalized = value.strip()
        if normalized not in MECHANISM_POOL:
            raise ValueError(f"Unknown mechanism label: {normalized}")
        return normalized

    @field_validator("evidence", "limitations")
    @classmethod
    def _non_empty_items(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value if item.strip()]
        if not normalized and value:
            raise ValueError("Text lists cannot contain only empty items.")
        return normalized


class PhotoMechOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_summary: str = Field(min_length=1)
    mechanism_predictions: list[MechanismPredictionOutput] = Field(
        min_length=3,
        max_length=3,
    )

    @model_validator(mode="after")
    def _ranked_distinct_predictions(self) -> PhotoMechOutput:
        ranks = [item.rank for item in self.mechanism_predictions]
        labels = [item.label for item in self.mechanism_predictions]
        if ranks != [1, 2, 3]:
            raise ValueError("Predictions must be sorted with ranks 1, 2, and 3.")
        if len(set(labels)) != 3:
            raise ValueError("Predicted mechanism labels must be distinct.")
        return self
