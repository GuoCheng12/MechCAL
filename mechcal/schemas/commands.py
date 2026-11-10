"""Model-facing operations. Identity, revisions and execution status stay in runtime."""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.schemas.tool_arguments import Atom, HoleElectronArgs, argument_model, planner_argument_model

MechanismLabel = Literal[tuple(MECHANISM_POOL)]


def unchanged(default=None, **kwargs):
    """Explicit delta semantics, not permission to infer missing scientific data."""
    options = {"json_schema_extra": {"x-mechcal-no-change": True}, **kwargs}
    return Field(**options) if "default_factory" in kwargs else Field(default=default, **options)


class CommandModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ReadPage(CommandModel):
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=16, ge=1, le=32)


class ReadRecord(CommandModel):
    ref: str
    paths: list[Annotated[str, Field(pattern=r"^(?:/.*)?$")]] = Field(
        default_factory=list, max_length=8,
        description="Exact JSON pointers from the record root, e.g. /structured_results/parsed_observables. "
                    "Use advertised paths, not bare field names. An empty list reads the whole record.")
    view: Literal["data", "directory"] = Field(default="data",
        description="data returns exact values; directory returns child paths and byte sizes only.")
    page: ReadPage | None = Field(default=None,
        description="For directory, page through children. For data, select one list path and page its rows. "
                    "Null preserves complete reads. Responses are limited to 24000 bytes.")


class ReadEvidence(CommandModel):
    kind: Literal["read_evidence"]
    records: list[ReadRecord] = Field(min_length=1, max_length=6)


class NumericOperand(CommandModel):
    ref: str
    path: str = Field(pattern=r"^(?:/.*)?$",
        description="Exact source JSON pointer. For array reductions, select the original array for the intended condition, not a displayed page.")
    reduction: Literal["scalar", "min", "max", "sum"] = "scalar"
    value_path: str = Field(default="", pattern=r"^(?:/.*)?$",
        description="Relative JSON pointer to the numeric field, e.g. /oscillator_strength in each selected array row. "
                    "Empty for numbers. Missing/non-numeric values are rejected, never skipped.")
    row_indices: list[Annotated[int, Field(strict=True, ge=0)]] | None = Field(default=None,
        min_length=1, max_length=512,
        description="Optional distinct zero-based ARRAY ROW indices for min/max/sum. Null selects every row. "
                    "These are not atom IDs. Verify each row's atom_index or state identity before selecting a fragment or state subset.")

    @model_validator(mode="after")
    def explicit_subset(self):
        if self.row_indices is not None:
            if self.reduction == "scalar":
                raise ValueError("row_indices applies only to min/max/sum; scalar uses an exact path.")
            if len(set(self.row_indices)) != len(self.row_indices):
                raise ValueError("row_indices must be distinct; duplicate rows would double-count observations.")
        return self


class ReduceEvidence(CommandModel):
    kind: Literal["reduce_evidence"]
    source: NumericOperand


class CompareEvidence(CommandModel):
    kind: Literal["compare_evidence"]
    operation: Literal["ratio", "difference"]
    left: NumericOperand
    right: NumericOperand


class ReadContext(CommandModel):
    """Read already materialized task context, never raw scientific observations."""

    kind: Literal["read_context"]
    fields: list[Literal["tool_catalog", "comparison_options", "shared_tool_guidance", "action_index",
                         "artifact_index", "explanation_index", "result_index", "evidence_index",
                         "exploration_state", "ranking"]] = Field(min_length=1, max_length=3)


class ReadCall(CommandModel):
    """Internal receipt input, never an extra model-facing command."""

    call_id: str
    command: ReadEvidence | ReduceEvidence | CompareEvidence | ReadContext | None = None
    issues: list[dict] = Field(default_factory=list)


class ReadBatch(CommandModel):
    kind: Literal["read_batch"] = "read_batch"
    calls: list[ReadCall] = Field(min_length=1, max_length=4)


class ComputeCall(CommandModel):
    """Internal batch receipt; each command is validated by its registered model."""

    call_id: str
    command: dict | None = None
    issues: list[dict] = Field(default_factory=list)
    original_call: dict = Field(default_factory=dict)


class ComputeBatch(CommandModel):
    kind: Literal["compute_batch"] = "compute_batch"
    calls: list[ComputeCall] = Field(min_length=1, max_length=6)


class ManageExperiments(CommandModel):
    kind: Literal["manage_experiments"]
    experiment_ids: list[str] = Field(min_length=1, max_length=6)
    action: Literal["retry", "cancel"]
    reason: str = ""


class DescribeTools(CommandModel):
    kind: Literal["describe_tools"]
    capability_ids: list[str] = Field(min_length=1, max_length=6)


class ArgumentChange(CommandModel):
    kind: Literal["argument"]
    id: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    label: MechanismLabel
    text: str = Field(min_length=1, pattern=r"\S")
    refs: list[str] = Field(default_factory=list, description="Exact successful observation IDs, not ID/path combinations.")
    limitation_refs: list[str] = Field(default_factory=list,
        description="Source or operation IDs for limitations, including failed or unexecuted experiments. Not positive evidence.")
    citations: list[ReadRecord] = Field(default_factory=list, max_length=6,
        description="Optional exact data selections for quantitative claims. Use a cited record or its advertised "
                    "citation_source_refs (registered parent result). Keep each source's own paths; "
                    "these same source values will be supplied to Reviewer. Do not copy values into this field.")


class CandidateChange(CommandModel):
    kind: Literal["candidate"]
    label: MechanismLabel
    status: Literal["unexamined", "considering", "deprioritized", "unavailable"]
    note: str
    refs: list[str] = Field(default_factory=list)
    limitation_refs: list[str] = Field(default_factory=list)


class ExplorationChange(CommandModel):
    label: MechanismLabel
    confidence: float | None = Field(ge=0, le=1,
        description="Current subjective confidence, not calibrated probability. Null means unassessed, not zero.")
    note: str = Field(min_length=1, pattern=r"\S")
    explanation: str | None = Field(default=None, description=
        "Current case-specific physical explanation for this family, separate from confidence-change notes. "
        "Null or omission retains the previous explanation; empty string clears it. This is interpretation, not evidence.")
    refs: list[str] = Field(default_factory=list)
    limitation_refs: list[str] = Field(default_factory=list)


class ImpactSuggestion(CommandModel):
    label: MechanismLabel
    refs: list[str] = Field(min_length=1, max_length=6)
    reason: str = Field(min_length=1, pattern=r"\S")


class ReportEvidenceImpacts(CommandModel):
    kind: Literal["report_evidence_impacts"]
    suggestions: list[ImpactSuggestion] = Field(default_factory=list, max_length=12)
    text: str = Field(min_length=1, pattern=r"\S")


class ImpactResponse(CommandModel):
    impact_id: str
    disposition: Literal["reviewed", "deferred", "dismissed"]
    note: str = Field(min_length=1, pattern=r"\S")


class ContrastDesign(CommandModel):
    varied: str = ""
    held_fixed: str = ""
    readout: str = ""
    baseline_refs: list[str] = Field(default_factory=list, max_length=6,
        description="Existing case-local inputs or results for comparison, not copied measurements.")


class ComparisonQuestion(CommandModel):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    labels: list[MechanismLabel] = Field(min_length=1, max_length=12)
    question: str = Field(min_length=1, pattern=r"\S")
    next_test: str = ""
    outcomes: str = Field(default="", description="How possible results would affect the candidates, including an inconclusive result.")
    status: Literal["open", "deferred", "resolved"] = "open"
    scope: Literal["mechanism", "test", "unspecified"] | None = unchanged(default=None,
        description="Scope of the question, not its answer. Null/omission retains the saved value; legacy scope is unknown.")
    parent_id: str | None = unchanged(default=None,
        description="Optional broader question ID in this case. Null/omission retains it; empty string detaches. "
                    "Answering this question never answers its parent automatically.")
    relation: Literal["alternatives", "complementary", "unknown"] | None = unchanged(default=None,
        description="Planner's interpretation of the named candidates for this question, not a taxonomy rule. "
                    "Null/omission retains the saved value.")
    contrast: ContrastDesign | None = unchanged(default=None,
        description="Optional small comparison design. Null/omission retains it; an empty object clears the design. "
                    "This is intent, not a claim that conditions matched or a computation request.")


class ComparisonRoute(CommandModel):
    capability_id: str
    input_refs: list[str] = Field(default_factory=list, max_length=6)


class ComparisonQuestionWithRoute(ComparisonQuestion):
    assessment: Literal["distinguished", "inconclusive"] | None = Field(default=None,
        description="Scientific answer to this question, separate from whether a calculation finished. "
                    "Null means not assessed, including legacy records. Execution success never implies distinguished.")
    route: ComparisonRoute | None = Field(default=None,
        description="Optional exact available route for this comparison; not an executable request. "
                    "Parameters are supplied to the computation tool separately.")
    disposition_reason: str | None = Field(default=None, description="Why a test is deferred or the comparison is resolved, if applicable.")


class UpdateHypotheses(CommandModel):
    kind: Literal["update_hypotheses"]
    changes: list[ExplorationChange] = unchanged(default_factory=list, max_length=12)
    note: str = Field(min_length=1, pattern=r"\S",
        description="Brief assessment of the available new results, or why confidence remains unchanged or unresolved.")


class UpdateHypothesesWithFollowup(UpdateHypotheses):
    responses: list[ImpactResponse] = unchanged(default_factory=list, max_length=24)
    questions: list[ComparisonQuestion] = unchanged(default_factory=list, max_length=8,
        description="Upsert only changed comparison questions. These are Planner decisions, not scientific evidence.")


class UpdateHypothesesWithWorkflow(UpdateHypothesesWithFollowup):
    questions: list[ComparisonQuestionWithRoute] = unchanged(default_factory=list, max_length=8)


class HypothesisStep(CommandModel):
    """Internal native-call transaction; never advertised as a model schema."""

    kind: Literal["hypothesis_step"] = "hypothesis_step"
    update: UpdateHypothesesWithWorkflow | UpdateHypothesesWithFollowup | UpdateHypotheses
    update_call_id: str
    followup_calls: list[dict] = Field(min_length=1, max_length=6)


class WithdrawArgument(CommandModel):
    kind: Literal["withdraw_argument"]
    id: str


class RankingChange(CommandModel):
    kind: Literal["ranking"]
    label: MechanismLabel
    confidence: float = Field(ge=0, le=1)
    priority: float = Field(ge=0, le=1)
    rationale: str
    argument_ids: list[str] = Field(default_factory=list)
    support: Literal["strong", "partial", "unsupported"] = "unsupported"
    claim: Literal["supported_claim", "partially_supported_candidate",
                   "candidate_requires_validation", "underdetermined"] = "candidate_requires_validation"
    validation_needed: list[str] = Field(default_factory=list)


Change = Annotated[Union[ArgumentChange, WithdrawArgument, RankingChange, CandidateChange], Field(discriminator="kind")]


class ReviseJudgment(CommandModel):
    kind: Literal["revise_judgment"]
    changes: list[Change] = Field(min_length=1, max_length=48)
    note: str = ""
    request_review: bool = Field(default=False,
        description="Request another scientific review even when observations and mechanism scope are unchanged. "
                    "Not needed for ordinary wording corrections.")


class AcceptedArgumentEdit(CommandModel):
    id: str
    text: str | None = unchanged(min_length=1, pattern=r"\S")
    refs: list[str] | None = unchanged()
    limitation_refs: list[str] | None = unchanged()
    citations: list[ReadRecord] | None = unchanged(max_length=6)


class AmendJudgment(CommandModel):
    kind: Literal["amend_judgment"]
    edits: list[AcceptedArgumentEdit] = unchanged(default_factory=list, max_length=12)
    rankings: list[RankingChange] = unchanged(default_factory=list, max_length=12)
    note: str = ""
    request_review: bool = False


class ReviewFocus(CommandModel):
    path: str = Field(pattern=r"^(|/.*)$", description="Exact JSON pointer within the review target, e.g. /text or /answer.")
    quote: str | None = Field(default=None, min_length=1,
        description="Optional unique exact passage at path. Null binds the whole field. Include relevant qualifications.")


class ReviewFinding(CommandModel):
    id: str | None = Field(default=None, description="Existing issue ID to update; null creates a new issue.")
    target: str = Field(description="An exact review target ID. Versions are recorded by runtime.")
    category: Literal["fact", "science", "wording"]
    status: Literal["open", "resolved", "disagreement"] = "open"
    text: str = Field(min_length=1, pattern=r"\S")
    source_refs: list[str] = Field(default_factory=list, max_length=6)
    focus: ReviewFocus | None = Field(default=None,
        description="Optional precise scope of this issue. Null conservatively binds the whole target.")


class RecordReview(CommandModel):
    kind: Literal["record_review"]
    text: str = Field(min_length=1, pattern=r"\S", description="Concise natural-language scientific feedback.")
    findings: list[ReviewFinding] = unchanged(default_factory=list, max_length=12,
        description="Only new or updated issues. Omitted issues retain their prior state; no need to restate them.")


class ReviewIdEdit(CommandModel):
    finding_index: int = Field(ge=0)
    field: Literal["id"]
    value: str | None


class ReviewTargetEdit(CommandModel):
    finding_index: int = Field(ge=0)
    field: Literal["target"]
    value: str


class ReviewSourcesEdit(CommandModel):
    finding_index: int = Field(ge=0)
    field: Literal["source_refs"]
    value: list[str] = Field(max_length=6)


class RepairReviewLinks(CommandModel):
    kind: Literal["repair_review_links"]
    edits: list[ReviewIdEdit | ReviewTargetEdit | ReviewSourcesEdit] = Field(min_length=1, max_length=36)


class NextComputation(CommandModel):
    capability_id: str
    question_id: str | None = None
    purpose: str = Field(min_length=1, pattern=r"\S",
        description="Which competing explanations this calculation could distinguish and why it adds information.")


class RecordComputationalReview(RecordReview):
    next_computations: list[NextComputation] | None = Field(default=None, max_length=3,
        description="Up to three useful remaining routes from the live catalog. [] means none recommended, "
                    "with the reason in text. Null reuses a displayed current computational review, not a new clearance.")


class RecordFreshComputationalReview(RecordComputationalReview):
    next_computations: list[NextComputation] = Field(max_length=3,
        description="No current computational review is reusable. Supply up to three useful catalog routes, "
                    "or [] with the reason in text. This is advice, not execution.")


class ArgumentPatch(CommandModel):
    kind: Literal["argument_patch"]
    change_index: int = Field(ge=0)
    id: str | None = unchanged(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    label: MechanismLabel | None = unchanged()
    text: str | None = unchanged(min_length=1, pattern=r"\S")
    refs: list[str] | None = unchanged()
    limitation_refs: list[str] | None = unchanged()
    citations: list[ReadRecord] | None = unchanged(max_length=6)


class CandidatePatch(CommandModel):
    kind: Literal["candidate_patch"]
    change_index: int = Field(ge=0)
    label: MechanismLabel | None = unchanged()
    status: Literal["unexamined", "considering", "deprioritized", "unavailable"] | None = unchanged()
    note: str | None = unchanged()
    refs: list[str] | None = unchanged()
    limitation_refs: list[str] | None = unchanged()


class RankingPatch(CommandModel):
    kind: Literal["ranking_patch"]
    change_index: int = Field(ge=0)
    label: MechanismLabel | None = unchanged()
    confidence: float | None = unchanged(ge=0, le=1)
    priority: float | None = unchanged(ge=0, le=1)
    rationale: str | None = unchanged()
    argument_ids: list[str] | None = unchanged()
    support: Literal["strong", "partial", "unsupported"] | None = unchanged()
    claim: Literal["supported_claim", "partially_supported_candidate",
                   "candidate_requires_validation", "underdetermined"] | None = unchanged()
    validation_needed: list[str] | None = unchanged()


class RemoveDraftChange(CommandModel):
    kind: Literal["remove_change"]
    change_index: int = Field(ge=0)


class RepairJudgment(CommandModel):
    kind: Literal["repair_judgment"]
    draft_id: str
    edits: list[Annotated[Union[ArgumentPatch, RankingPatch, CandidatePatch, RemoveDraftChange], Field(discriminator="kind")]] = unchanged(
        default_factory=list, max_length=48,
        description="Indices refer to the displayed draft. Null patch fields are unchanged; [] clears a list.")
    additional_changes: list[Change] = unchanged(default_factory=list, max_length=48)

    @model_validator(mode="after")
    def require_edits(self):
        if not self.edits and not self.additional_changes:
            raise ValueError("Supply a local edit or an additional change.")
        return self


class DiscardJudgment(CommandModel):
    kind: Literal["discard_judgment"]
    draft_id: str


# The opt-in wire links rankings to authoritative arguments instead of accepting
# a second editable copy of the scientific explanation. Legacy storage is read.
LinkedRankingChange = create_model(
    "LinkedRankingChange", __base__=CommandModel,
    **{name: (field.annotation, field) for name, field in RankingChange.model_fields.items() if name != "rationale"},
)
LinkedRankingPatch = create_model(
    "LinkedRankingPatch", __base__=CommandModel,
    **{name: (field.annotation, field) for name, field in RankingPatch.model_fields.items() if name != "rationale"},
)
LinkedChange = Annotated[Union[ArgumentChange, WithdrawArgument, LinkedRankingChange, CandidateChange], Field(discriminator="kind")]


class LinkedReviseJudgment(ReviseJudgment):
    changes: list[LinkedChange] = Field(min_length=1, max_length=48)


class LinkedAmendJudgment(AmendJudgment):
    rankings: list[LinkedRankingChange] = unchanged(default_factory=list, max_length=12)


class LinkedRepairJudgment(RepairJudgment):
    edits: list[Annotated[Union[ArgumentPatch, LinkedRankingPatch, CandidatePatch, RemoveDraftChange], Field(discriminator="kind")]] = unchanged(default_factory=list, max_length=48)
    additional_changes: list[LinkedChange] = unchanged(default_factory=list, max_length=48)


class Finalize(CommandModel):
    kind: Literal["finalize"]
    answer: str = Field(min_length=1, pattern=r"\S")
    limitations: list[str] = Field(default_factory=list)


class SubmitFinal(CommandModel):
    kind: Literal["submit_final"]
    draft_id: str = Field(min_length=1, description="Exact saved final-draft ID; no narrative rewrite required.")


class ReviewFinal(CommandModel):
    kind: Literal["review_final"]
    draft_id: str = Field(min_length=1, description="Review this saved draft without regenerating its contents.")


class RouteResponse(CommandModel):
    capability_id: str
    reason: str = Field(min_length=1, pattern=r"\S",
        description="Why this recommended calculation is not pursued before delivery. Scientific disagreement is allowed.")


class SubmitFinalAfterReview(SubmitFinal):
    route_responses: list[RouteResponse] = Field(default_factory=list, max_length=3,
        description="Respond once per distinct capability in next_computations, covering all its suggested comparisons when stopping. "
                    "Use [] if none; compute requests remain separate operations.")


def reviewed_submission_model(requirements):
    """Bind both advertised schema and local parser to the current review."""
    ids = tuple(row["capability_id"] for row in requirements)
    response = (create_model("CurrentRouteResponse", __base__=RouteResponse,
                            capability_id=(Literal[ids], ...)) if ids else RouteResponse)
    return create_model("CurrentSubmitFinal", __base__=SubmitFinalAfterReview,
        route_responses=(list[response], Field(min_length=len(ids), max_length=len(ids),
            description="Respond once to each listed capability, covering its comparisons. Explain why stopping is "
                        "appropriate; executing the suggestions is not mandatory. Current requirements: " + str(requirements))))


class TextReplacement(CommandModel):
    path: str = Field(default="/answer", pattern=r"^/(answer|limitations/(0|[1-9][0-9]*))$",
        description="Editable field in delivery:answer: /answer or /limitations/<zero-based index>. "
                    "Old text is matched only inside this field. Defaults to /answer for older commands.")
    old_text: str = Field(min_length=1)
    new_text: str


class FinalFieldEdit(CommandModel):
    path: str = Field(pattern=r"^/(answer|limitations/(0|[1-9][0-9]*))$",
        description="Exact editable field, not a quote or arbitrary JSON patch.")
    value: str = Field(min_length=1, pattern=r"\S")


class AmendFinal(CommandModel):
    kind: Literal["amend_final"]
    draft_id: str
    replacements: list[TextReplacement] = unchanged(default_factory=list, max_length=8,
        description="Atomic exact-text edits, each scoped to its path. Specify /limitations/<index> for a limitation.")
    limitations: list[str] | None = unchanged(description="Replace the entire limitations list, or omit/null to retain it. "
        "Do not combine a list replacement with indexed limitation edits in replacements.")
    fields: list[FinalFieldEdit] = unchanged(default_factory=list, max_length=8,
        description="Replace only these complete text fields, bound to draft_id. No old_text is required. "
                    "Do not also edit the same field through replacements or limitations.")


class AmendFinalFields(CommandModel):
    """Canonical model-facing edit. Legacy storage remains readable separately."""

    kind: Literal["amend_final"]
    draft_id: str
    fields: list[FinalFieldEdit] = Field(min_length=1, max_length=8,
        description="Copy the exact field path from edit_context and provide its complete replacement text. "
                    "Only listed fields change. All other fields remain unchanged.")


class DeferCommandRepair(CommandModel):
    reason: str = Field(min_length=1, pattern=r"\S")


class CommentaryText(CommandModel):
    """Runtime-owned prose container, never sent as a model output schema."""

    text: str = Field(min_length=1, pattern=r"\S")


class Commentary(CommentaryText):
    """Historical JSON review envelope, retained for offline contract replays."""

    refs: list[str] = Field(default_factory=list)


def commentary_model(reference_ids):
    """Bind review citations to this request, without constraining scientific prose."""
    ids = tuple(sorted(set(reference_ids)))
    annotation = list[Literal[ids]] if ids else list[str]
    constraints = {} if ids else {"max_length": 0}
    return create_model(
        "ScopedCommentary", __base__=Commentary,
        refs=(annotation, Field(default_factory=list, **constraints)),
    )


class AtomGroup(CommandModel):
    name: str = Field(min_length=1)
    atoms: list[Atom] = Field(min_length=1)


# The only open parameter map in the registry is named atom groups. The wire
# uses explicit objects; conversion to the calculator's map is a single adapter.
HoleElectronCommandArgs = create_model(
    "HoleElectronCommandArgs",
    __config__=HoleElectronArgs.model_config,
    **{name: (field.annotation, field) for name, field in HoleElectronArgs.model_fields.items()
       if name != "fragments"},
    fragments=(list[AtomGroup] | None, Field(default=None, max_length=12)),
)


def tool_argument_model(spec):
    model = planner_argument_model(spec.route) or argument_model(spec.route, spec.owner_agent)
    return HoleElectronCommandArgs if model is HoleElectronArgs else model


def computation_model(spec, *, with_question=False):
    return create_model(
        "Compute_" + spec.capability_id.replace(".", "_"),
        __base__=CommandModel,
        kind=(Literal["compute"], ...),
        capability_id=(Literal[spec.capability_id], ...),
        arguments=(tool_argument_model(spec), ...),
        purpose=(str, Field(min_length=1, pattern=r"\S")),
        retry_of=(str | None, Field(default=None, min_length=1,
            json_schema_extra={"x-mechcal-optional": True},
            description="Failed or partial result ID to retry once with the same tool and parameters. "
                        "Omit or null for normal cached reuse. Repeating the same retry returns its saved attempt.")),
        **({"question_id": (str | None, Field(default=None,
            description="Optional ID of a saved comparison question this computation addresses. Not a tool parameter."))}
           if with_question else {}),
    )


def envelope_model(registry, capability_ids, *, hypothesis_tracking=False, evidence_followup=False, review_workflow=False,
                   computational_review=False, task_context=False, worker_tasks=False):
    commands = [ReadEvidence, ReduceEvidence, CompareEvidence, DescribeTools, ManageExperiments,
                LinkedReviseJudgment if review_workflow else ReviseJudgment,
                LinkedRepairJudgment if review_workflow else RepairJudgment,
                DiscardJudgment, Finalize, ReviewFinal, SubmitFinalAfterReview if computational_review else SubmitFinal]
    if task_context:
        commands.append(ReadContext)
    if worker_tasks:
        from mechcal.schemas.worker_tasks import DelegateComputation
        commands.append(DelegateComputation)
    if hypothesis_tracking:
        commands.append(UpdateHypothesesWithWorkflow if review_workflow else
                        UpdateHypothesesWithFollowup if evidence_followup else UpdateHypotheses)
    if review_workflow:
        commands.extend([LinkedAmendJudgment, AmendFinal])
    commands.extend(computation_model(registry.require(cid), with_question=evidence_followup) for cid in capability_ids)
    return create_model("PlannerCommand", __base__=CommandModel,
                        command=(Union[tuple(commands)], ...))


def internal_arguments(command, spec):
    args = command.arguments.model_dump(exclude_none=True)
    if spec.route == "analyze_hole_electron_from_bundle" and args.get("fragments") is not None:
        groups = args["fragments"]
        if len({g["name"] for g in groups}) != len(groups):
            raise ValueError("Fragment names must be unique.")
        args["fragments"] = {g["name"]: g["atoms"] for g in groups}
    model = planner_argument_model(spec.route) or argument_model(spec.route, spec.owner_agent)
    return model.model_validate(args).model_dump(exclude_none=True)
