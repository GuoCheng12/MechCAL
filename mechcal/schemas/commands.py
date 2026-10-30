"""Model-facing operations. Identity, revisions and execution status stay in runtime."""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.schemas.tool_arguments import Atom, HoleElectronArgs, argument_model, planner_argument_model

MechanismLabel = Literal[tuple(MECHANISM_POOL)]


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


class ReadCall(CommandModel):
    """Internal receipt input, never an extra model-facing command."""

    call_id: str
    command: ReadEvidence | None = None
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
    refs: list[str] = Field(default_factory=list)
    limitation_refs: list[str] = Field(default_factory=list)


class UpdateHypotheses(CommandModel):
    kind: Literal["update_hypotheses"]
    changes: list[ExplorationChange] = Field(default_factory=list, max_length=12)
    note: str = Field(min_length=1, pattern=r"\S",
        description="Brief assessment of the available new results, or why confidence remains unchanged or unresolved.")


class HypothesisStep(CommandModel):
    """Internal native-call transaction; never advertised as a model schema."""

    kind: Literal["hypothesis_step"] = "hypothesis_step"
    update: UpdateHypotheses
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
        description="Request scientific review of a changed interpretation even when sources and ranking are unchanged. "
                    "Not needed for ordinary wording corrections.")


class ArgumentPatch(CommandModel):
    kind: Literal["argument_patch"]
    change_index: int = Field(ge=0)
    id: str | None = Field(default=None, min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    label: MechanismLabel | None = None
    text: str | None = Field(default=None, min_length=1, pattern=r"\S")
    refs: list[str] | None = None
    limitation_refs: list[str] | None = None
    citations: list[ReadRecord] | None = Field(default=None, max_length=6)


class CandidatePatch(CommandModel):
    kind: Literal["candidate_patch"]
    change_index: int = Field(ge=0)
    label: MechanismLabel | None = None
    status: Literal["unexamined", "considering", "deprioritized", "unavailable"] | None = None
    note: str | None = None
    refs: list[str] | None = None
    limitation_refs: list[str] | None = None


class RankingPatch(CommandModel):
    kind: Literal["ranking_patch"]
    change_index: int = Field(ge=0)
    label: MechanismLabel | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    priority: float | None = Field(default=None, ge=0, le=1)
    rationale: str | None = None
    argument_ids: list[str] | None = None
    support: Literal["strong", "partial", "unsupported"] | None = None
    claim: Literal["supported_claim", "partially_supported_candidate",
                   "candidate_requires_validation", "underdetermined"] | None = None
    validation_needed: list[str] | None = None


class RemoveDraftChange(CommandModel):
    kind: Literal["remove_change"]
    change_index: int = Field(ge=0)


class RepairJudgment(CommandModel):
    kind: Literal["repair_judgment"]
    draft_id: str
    edits: list[Annotated[Union[ArgumentPatch, RankingPatch, CandidatePatch, RemoveDraftChange], Field(discriminator="kind")]] = Field(
        default_factory=list, max_length=48,
        description="Indices refer to the displayed draft. Null patch fields are unchanged; [] clears a list.")
    additional_changes: list[Change] = Field(default_factory=list, max_length=48)

    @model_validator(mode="after")
    def require_edits(self):
        if not self.edits and not self.additional_changes:
            raise ValueError("Supply a local edit or an additional change.")
        return self


class DiscardJudgment(CommandModel):
    kind: Literal["discard_judgment"]
    draft_id: str


class Finalize(CommandModel):
    kind: Literal["finalize"]
    answer: str = Field(min_length=1, pattern=r"\S")
    limitations: list[str] = Field(default_factory=list)


class SubmitFinal(CommandModel):
    kind: Literal["submit_final"]
    draft_id: str = Field(min_length=1, description="Exact saved final-draft ID; no narrative rewrite required.")


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


def computation_model(spec):
    return create_model(
        "Compute_" + spec.capability_id.replace(".", "_"),
        __base__=CommandModel,
        kind=(Literal["compute"], ...),
        capability_id=(Literal[spec.capability_id], ...),
        arguments=(tool_argument_model(spec), ...),
        purpose=(str, Field(min_length=1, pattern=r"\S")),
    )


def envelope_model(registry, capability_ids, *, hypothesis_tracking=False):
    commands = [ReadEvidence, DescribeTools, ManageExperiments, ReviseJudgment, RepairJudgment, DiscardJudgment, Finalize, SubmitFinal]
    if hypothesis_tracking:
        commands.append(UpdateHypotheses)
    commands.extend(computation_model(registry.require(cid)) for cid in capability_ids)
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
