"""Model-facing commands, separate from persisted investigation snapshots."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.schemas.decisions import PlannerAssessment
from mechcal.schemas.evidence import EvidenceBasis, HypothesisStatus
from mechcal.schemas.evidence_comparison import RankingUpdate
from mechcal.schemas.mechanisms import MechanismSupportArgument
from mechcal.schemas.final_evidence import FinalEvidenceSubmission
from mechcal.schemas.predecision import ComparisonResponse
from mechcal.schemas.boundary_decisions import CandidateBasis
from mechcal.schemas.question_loop import CandidateResponse, QuestionAction, ResultDisposition


LOCAL_AUDIT_RETAINED_FIELDS = ("hypotheses", "questions", "candidate_responses", "dispositions", "ranking_update")


class CommandModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HypothesisChange(CommandModel):
    """One write path: omitted/null fields retain the accepted value."""

    name: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    differential_priority: float | None = Field(default=None, ge=0, le=1)
    evidence_support: str | None = None
    claim_status: str | None = None
    status: HypothesisStatus | None = None
    rationale: str | None = None
    evidence_refs: list[str] | None = None
    validation_needed: list[str] | None = None


class QuestionChange(CommandModel):
    """Scientific question edits, not a copy of persisted runtime state."""

    operation: Literal["create", "update"]
    question_id: str = Field(min_length=1)
    mechanism_id: str | None = None
    question: str | None = None
    observable: str | None = None
    expected_basis: EvidenceBasis | None = None
    acceptable_capability_ids: list[str] | None = None
    missing_or_unresolved: list[str] | None = None
    related_labels: list[str] | None = None
    answer_state: Literal["open", "partial", "answered", "deferred"] | None = None
    assessment: str | None = None
    decision_role: Literal["material", "limitation"] | None = None
    evidence_refs: list[str] | None = None
    result_ids: list[str] | None = None


class QuestionCreate(QuestionChange):
    """Creation is complete; only updates can inherit an accepted value."""

    operation: Literal["create"]
    mechanism_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    observable: str = Field(min_length=1)
    expected_basis: EvidenceBasis
    related_labels: list[str]
    answer_state: Literal["open", "partial", "answered", "deferred"]
    assessment: str = Field(min_length=1)
    decision_role: Literal["material", "limitation"]


class QuestionUpdate(QuestionChange):
    operation: Literal["update"]


class JudgmentCommand(CommandModel):
    action: Literal["update_judgment"] = "update_judgment"
    reason: str = Field(min_length=1)
    scope: Literal["local_audit", "global"] = "global"
    hypotheses: list[HypothesisChange] = Field(default_factory=list)
    questions: list[Annotated[QuestionCreate | QuestionUpdate, Field(discriminator="operation")]] = Field(default_factory=list)
    mechanism_support_arguments: list[MechanismSupportArgument] = Field(default_factory=list)
    withdrawn_support_ids: list[str] = Field(default_factory=list)
    ranking_update: RankingUpdate | None = None
    dispositions: list[ResultDisposition] = Field(default_factory=list)
    candidate_responses: list[CandidateResponse] = Field(default_factory=list)
    decision_assessment: PlannerAssessment
    coverage_response: str = Field(min_length=1)
    audit_responses: dict[str, str] = Field(default_factory=dict)
    comparison_response: ComparisonResponse | None = None


class FinalJudgmentCommand(CommandModel):
    action: Literal["revise_final"] = "revise_final"
    update: JudgmentCommand
    final_action: QuestionAction
    audit_responses: dict[str, str]
    evidence_submission: list[FinalEvidenceSubmission] = Field(
        min_length=3, max_length=3,
        description="One explicit evidence submission per final Top-3 label, keyed by label, in any array order. "
        "Submit actual argument IDs or explain no evidence. Hypothesis prose alone is not submitted evidence.",
    )


class RankingRationale(CommandModel):
    """Scientific interpretation only; rank-bound explanations are a later read."""

    basis: Literal["initial", "new_evidence", "reinterpretation", "unchanged"]
    explanation: str = Field(min_length=1)
    alternatives_considered: list[str] = Field(
        min_length=1,
        description="Exact mechanism-pool labels only; put the comparison reasoning in explanation.",
        json_schema_extra={"items": {"type": "string", "enum": list(MECHANISM_POOL)}},
    )
    evidence_refs: list[str] = Field(default_factory=list)
    candidate_bases: list[CandidateBasis] = Field(default_factory=list)


class JudgmentProposal(JudgmentCommand):
    """Live write interface. JudgmentCommand remains readable for old records."""

    ranking_update: RankingRationale | None = None


class FinalJudgmentProposal(FinalJudgmentCommand):
    update: JudgmentProposal


class CommandReceipt(CommandModel):
    """Operational feedback, never a scientific observation or audit approval."""

    action: Literal["update_judgment", "revise_final"]
    status: Literal["rejected", "validated"]
    case_id: str
    base_revision: int
    attempt: int
    message: str
    issues: list[dict] = Field(default_factory=list)
