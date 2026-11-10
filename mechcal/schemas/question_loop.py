"""Contracts for staged question-driven decisions; no scientific scoring rules."""

from copy import deepcopy
from typing import ClassVar, Literal

from jsonschema import Draft202012Validator
from pydantic import Field, ValidationError, model_validator

from mechcal.schemas.base import MechCALModel
from mechcal.schemas.final_evidence import FinalEvidenceSubmission
from mechcal.schemas.boundary_decisions import ContinuationAssessment
from mechcal.schemas.hypothesis_tests import TestAssessment
from mechcal.schemas.candidate_expansion import CandidateResponse
from mechcal.schemas.worker_explanation import FindingAudit
from mechcal.schemas.impact_review import EvidenceImpactReview
from mechcal.schemas.predecision import ComparisonResponse
from mechcal.schemas.decisions import DispatchRequest, PlannerAssessment, StoppingAssessment
from mechcal.schemas.evidence import HypothesisPortfolio
from mechcal.schemas.mechanisms import EvidenceQuestion, MechanismSupportArgument
from mechcal.schemas.evidence_comparison import (
    ArgumentChange, InvestigationPlan, MechanismEffect, RankingUpdate,
)


class ResultInterpretation(MechCALModel):
    record_type: str = "ResultInterpretation"
    result_id: str
    question_ids: list[str] = Field(min_length=1)
    delivered: str = Field(min_length=1)
    missing: str = Field(min_length=1)
    boundary: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)
    finding_audits: list[FindingAudit] = Field(default_factory=list)


class ResultReview(MechCALModel):
    record_type: str = "ResultReview"
    status: Literal["reviewed", "unavailable", "disabled"] = "reviewed"
    interpretations: list[ResultInterpretation] = Field(default_factory=list)
    reason: str = ""
    impact_review: EvidenceImpactReview | None = None


class ResultDisposition(MechCALModel):
    record_type: str = "ResultDisposition"
    result_id: str
    question_ids: list[str] = Field(
        min_length=1,
        description="Preserve this execution's original question_ids. Later reuse belongs in EvidenceQuestion.result_ids.",
    )
    outcome: Literal["informative", "non_discriminating", "incomplete", "no_result"]
    interpretation: str = Field(min_length=1)
    reviewer_response: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)
    mechanism_effects: list[MechanismEffect] = Field(default_factory=list)
    test_assessment: TestAssessment | None = None


class RetainedJudgment(MechCALModel):
    """Select base records; complete submitted records override matching IDs."""

    record_type: str = "RetainedJudgment"
    hypothesis_labels: list[str] = Field(default_factory=list)
    support_ids: list[str] = Field(default_factory=list)
    question_ids: list[str] = Field(default_factory=list)
    candidate_basis_labels: list[str] = Field(default_factory=list)


class HypothesisScoreUpdate(MechCALModel):
    record_type: str = "HypothesisScoreUpdate"
    name: str
    confidence: float = Field(ge=0, le=1)
    differential_priority: float = Field(ge=0, le=1)


class JudgmentDelta(MechCALModel):
    """Versioned upserts; omission reuses prior records, not deletion."""

    record_type: str = "JudgmentDelta"
    base_revision: int = Field(ge=1)
    withdrawn_support_ids: list[str] = Field(default_factory=list)
    hypothesis_scores: list[HypothesisScoreUpdate] = Field(default_factory=list)
    scope: Literal["local_audit", "global"] = "global"
    reason: str = Field(min_length=1)


class QuestionAssimilation(MechCALModel):
    record_type: str = "QuestionAssimilation"
    candidate_responses: list[CandidateResponse] = Field(default_factory=list)
    case_id: str
    revision: int = Field(ge=1)
    questions: list[EvidenceQuestion] = Field(default_factory=list)
    delta: JudgmentDelta | None = None
    retained_judgment: RetainedJudgment | None = Field(default=None, description=(
        "Select current-case base records by ID. Supplied full records replace matching base IDs; "
        "base-only IDs reuse prior contents. Newly submitted IDs may also be listed. "
        "An ID with neither a prior nor a supplied record is invalid. Omitted, unselected records are not retained."
    ))
    dispositions: list[ResultDisposition] = Field(
        default_factory=list,
        description="Exactly one entry per pending result, not execution history. Return [] when pending_results is empty, including audit-only responses. Historical results can still be cited in questions and arguments.",
    )
    portfolio: HypothesisPortfolio
    mechanism_support_arguments: list[MechanismSupportArgument] = Field(
        default_factory=list,
        description="Current findings, mechanism-specific warrants and boundaries. Cite runtime observations or input:smiles for explicit structural findings. Return [] when no argument is submitted.",
    )
    decision_assessment: PlannerAssessment
    coverage_response: str = Field(min_length=1)
    argument_changes: list[ArgumentChange] = Field(default_factory=list)
    ranking_update: RankingUpdate | None = None
    comparison_response: ComparisonResponse | None = None
    audit_responses: dict[str, str] = Field(
        default_factory=dict,
        description="When argument_review_pending is true, respond to each reviewed support_id, including withdrawn arguments. Explain agreement, correction, justified disagreement, or unresolved uncertainty; never infer external scores.",
    )


class QuestionAction(MechCALModel):
    # One structural contract drives both the published schema and local checks.
    action_fields: ClassVar[dict] = {
        "dispatch": {"dispatch_requests": {"minItems": 1}},
        "finalize": {"dispatch_requests": {"maxItems": 0},
                     "final_answer_draft": {"type": "string", "minLength": 1, "pattern": r"\S"}},
        "stop": {"dispatch_requests": {"maxItems": 0}},
    }
    record_type: str = "QuestionAction"
    case_id: str
    revision: int = Field(ge=1)
    action: Literal["dispatch", "finalize", "stop"]
    dispatch_requests: list[DispatchRequest] = Field(default_factory=list, max_length=6)
    investigation_plans: list[InvestigationPlan] = Field(default_factory=list)
    stopping_assessment: StoppingAssessment
    # Every question gets a final disposition, including irrelevant/deferred ones.
    question_dispositions: dict[str, str] = Field(default_factory=dict)
    ranking_justification: str = ""
    ranking_uncertainty_material: bool | None = Field(
        default=None,
        description="For finalize, does remaining uncertainty prevent a defensible bounded ranking? Planner's judgment, not a tool-success flag.",
    )
    unresolved_gaps: list[str] = Field(default_factory=list)
    final_answer_draft: str | None = None
    rationale: str = Field(min_length=1)
    reviewer_response: str | None = None
    continuation_assessment: ContinuationAssessment | None = None

    @model_validator(mode="after")
    def action_shape(self):
        issues = []
        for name, constraint in self.action_fields[self.action].items():
            value = getattr(self, name)
            for error in Draft202012Validator(constraint).iter_errors(value):
                issues.append({"type": "value_error", "loc": (name,), "input": value,
                               "ctx": {"error": ValueError(f"For action={self.action}: {error.message}")}})
        if issues:
            raise ValidationError.from_exception_data(type(self).__name__, issues)
        return self

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        base = handler.resolve_ref_schema(handler(core_schema))
        variants = []
        for action, fields in cls.action_fields.items():
            branch = deepcopy(base)
            branch["properties"]["action"] = {"type": "string", "const": action}
            for name, constraint in fields.items():
                prop = branch["properties"][name]
                if "type" in constraint:
                    prop.pop("anyOf", None)
                    prop.pop("default", None)
                prop.update(constraint)
            branch["required"] = list(dict.fromkeys([*branch.get("required", []), *fields]))
            variants.append(branch)
        return {"anyOf": variants}


class FinalArgumentRevision(MechCALModel):
    """One evidence-only response to the final internal audit, never a new dispatch."""

    record_type: str = "FinalArgumentRevision"
    update: QuestionAssimilation
    # Absent in historical snapshots; current command interface requires it.
    evidence_submission: list[FinalEvidenceSubmission] | None = None
    final_action: QuestionAction
    audit_responses: dict[str, str] = Field(
        description="Respond to each pre-revision support_id, including retained or withdrawn arguments.",
    )
