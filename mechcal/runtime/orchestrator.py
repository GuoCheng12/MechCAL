from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

from mechcal.agents import (
    ConclusionLedgerAgent,
    MacroAgent,
    MechanismAgendaAgent,
    MechanismCriticAgent,
    MicroscopicAgent,
    PhotophysicsArbiterAgent,
    PlannerAgent,
    WorkerAgent,
)
from mechcal.agents.mechanism_agenda import program_from_coverage
from mechcal.agents.photophysics_arbiter import DisabledPhotophysicsWebSearch
from mechcal.capabilities import CapabilityRegistry, default_capability_registry
from mechcal.public_ids import normalize_public_case_id
from mechcal.runtime.ablations import (
    ABLATION_MODES,
    AblationMode,
    AblationPolicy,
    ablation_policy,
)
from mechcal.runtime.claims import (
    annotate_evidence_units,
    empty_claim_ledger,
    reduce_claim_ledger,
)
from mechcal.runtime.context import build_planner_context
from mechcal.runtime.gates import (
    DISCRIMINATING_FOLLOWUP_CAPABILITY_IDS,
    DeterministicGate,
    GatePolicy,
)
from mechcal.runtime.llm import OpenAIJsonClient
from mechcal.runtime.mechanism_program import build_diagnosis_units
from mechcal.runtime.output_assembler import OutputAssembler
from mechcal.runtime.persistence import RunStore
from mechcal.schemas import (
    AgentReport,
    ArtifactManifest,
    CaseInput,
    CaseRun,
    DecisionGateResult,
    DifferentialMechanismPortfolio,
    DifferentialMechanismPortfolioRow,
    DispatchRequest,
    EvidenceLedger,
    FinalAnswer,
    HypothesisEntry,
    HypothesisPortfolio,
    MechanismPriorityDelta,
    MechanismSupportArgument,
    MechanismSupportAudit,
    RoundRecord,
)
from mechcal.tools import (
    LocalMacroTool,
    LocalMicroscopicTool,
    LocalStructureTool,
    McpStructureTool,
    McpTransport,
    McpWorkerTool,
    StructureToolClient,
)
from mechcal.tools.mcp_adapter import AieMcpToolSpecs


@dataclass(frozen=True)
class OrchestratorConfig:
    run_base_dir: Path
    max_rounds: int = 4
    ablation_mode: AblationMode = "full_mechcal"
    enable_llm_planner: bool = False
    enable_macro_llm_planning: bool = False
    enable_microscopic_llm_planning: bool = False
    enable_llm_worker_planning: bool = False
    enable_llm_worker_reports: bool = False
    enable_amesp: bool = False
    enable_legacy_conclusion_ledger: bool = False
    enable_support_auditor: bool = True
    enable_agenda_coverage_reviewer: bool = True
    enable_incremental_mechanism_portfolio: bool = True
    enable_support_auditor_web_search: bool = False
    continue_on_arbiter_failure: bool = False
    continue_on_agenda_coverage_failure: bool = True
    continue_on_planner_update_failure: bool = False
    continue_on_mechanism_critic_failure: bool = True
    verifier_failure_stop_threshold: int = 0
    convergence_min_rounds: int = 6
    convergence_window: int = 3
    defer_review_until_portfolio: bool = False

    def __post_init__(self) -> None:
        if self.ablation_mode not in ABLATION_MODES:
            raise ValueError(f"Unknown ablation_mode: {self.ablation_mode}")

    @property
    def ablation_policy(self) -> AblationPolicy:
        return ablation_policy(self.ablation_mode)

    @property
    def effective_support_auditor(self) -> bool:
        return (
            self.enable_support_auditor
            and self.ablation_policy.enable_support_auditor
        )

    @property
    def effective_agenda_coverage_reviewer(self) -> bool:
        return (
            self.enable_agenda_coverage_reviewer
            and self.ablation_policy.enable_agenda_coverage_reviewer
        )


@dataclass
class OrchestratorDeps:
    planner: PlannerAgent
    mechanism_agenda: MechanismAgendaAgent
    photophysics_arbiter: PhotophysicsArbiterAgent
    mechanism_critic: MechanismCriticAgent
    conclusion_ledger: ConclusionLedgerAgent
    output_assembler: OutputAssembler
    structure: StructureToolClient
    workers: dict[str, WorkerAgent]
    gate: DeterministicGate
    capability_registry: CapabilityRegistry


def _agent_llm_client(
    base_client: OpenAIJsonClient,
    *,
    max_retries: int | None = None,
    min_max_tokens: int | None = None,
) -> OpenAIJsonClient:
    settings = base_client.settings
    max_tokens = settings.max_tokens
    if min_max_tokens is not None:
        max_tokens = max(min_max_tokens, max_tokens or 0)
    return OpenAIJsonClient(
        replace(
            settings,
            max_retries=settings.max_retries if max_retries is None else max_retries,
            max_tokens=max_tokens,
        )
    )


def default_deps(config: OrchestratorConfig) -> OrchestratorDeps:
    capability_registry = default_capability_registry().for_owner_agents(
        config.ablation_policy.enabled_worker_agents
    )
    llm_client = OpenAIJsonClient()
    planner_llm_client = _agent_llm_client(llm_client, min_max_tokens=2400)
    reviewer_llm_client = _agent_llm_client(llm_client, max_retries=0, min_max_tokens=2400)
    workers: dict[str, WorkerAgent] = {
        "macro": MacroAgent(
            LocalMacroTool(),
            capability_registry=capability_registry,
            llm_client=llm_client,
            enable_llm_planning=(
                config.enable_macro_llm_planning
                or config.enable_llm_worker_planning
            ),
            enable_llm_reports=config.enable_llm_worker_reports,
        ),
        "microscopic": MicroscopicAgent(
            LocalMicroscopicTool(enable_amesp=config.enable_amesp),
            capability_registry=capability_registry,
            llm_client=llm_client,
            enable_llm_planning=(
                config.enable_microscopic_llm_planning
                or config.enable_llm_worker_planning
            ),
            enable_llm_reports=config.enable_llm_worker_reports,
        ),
    }
    workers = {
        name: worker
        for name, worker in workers.items()
        if name in config.ablation_policy.enabled_worker_agents
    }
    return OrchestratorDeps(
        planner=PlannerAgent(
            capability_registry=capability_registry,
            llm_client=planner_llm_client,
            enable_llm_decisions=config.enable_llm_planner,
        ),
        mechanism_agenda=MechanismAgendaAgent(
            capability_registry=capability_registry,
            llm_client=reviewer_llm_client,
        ),
        photophysics_arbiter=PhotophysicsArbiterAgent(
            llm_client=reviewer_llm_client,
            web_search=None
            if config.enable_support_auditor_web_search
            else DisabledPhotophysicsWebSearch(),
            max_attempts=1,
        ),
        mechanism_critic=MechanismCriticAgent(
            llm_client=reviewer_llm_client,
            max_attempts=3,
        ),
        conclusion_ledger=ConclusionLedgerAgent(llm_client=reviewer_llm_client),
        output_assembler=OutputAssembler(),
        structure=LocalStructureTool(),
        workers=workers,
        gate=DeterministicGate(
            GatePolicy(
                max_rounds=config.max_rounds,
            ),
            capability_registry=capability_registry,
        ),
        capability_registry=capability_registry,
    )


def mcp_deps(config: OrchestratorConfig, transport: McpTransport) -> OrchestratorDeps:
    specs = AieMcpToolSpecs()
    capability_registry = specs.discover_capability_registry(transport).for_owner_agents(
        config.ablation_policy.enabled_worker_agents
    )
    llm_client = OpenAIJsonClient()
    planner_llm_client = _agent_llm_client(llm_client, min_max_tokens=2400)
    reviewer_llm_client = _agent_llm_client(llm_client, max_retries=0, min_max_tokens=2400)
    workers: dict[str, WorkerAgent] = {
        "macro": MacroAgent(
            McpWorkerTool(transport, specs.macro),
            capability_registry=capability_registry,
            llm_client=llm_client,
            enable_llm_planning=(
                config.enable_macro_llm_planning
                or config.enable_llm_worker_planning
            ),
            enable_llm_reports=config.enable_llm_worker_reports,
        ),
        "microscopic": MicroscopicAgent(
            McpWorkerTool(transport, specs.microscopic),
            capability_registry=capability_registry,
            llm_client=llm_client,
            enable_llm_planning=(
                config.enable_microscopic_llm_planning
                or config.enable_llm_worker_planning
            ),
            enable_llm_reports=config.enable_llm_worker_reports,
        ),
    }
    workers = {
        name: worker
        for name, worker in workers.items()
        if name in config.ablation_policy.enabled_worker_agents
    }
    return OrchestratorDeps(
        planner=PlannerAgent(
            capability_registry=capability_registry,
            llm_client=planner_llm_client,
            enable_llm_decisions=config.enable_llm_planner,
        ),
        mechanism_agenda=MechanismAgendaAgent(
            capability_registry=capability_registry,
            llm_client=reviewer_llm_client,
        ),
        photophysics_arbiter=PhotophysicsArbiterAgent(
            llm_client=reviewer_llm_client,
            web_search=None
            if config.enable_support_auditor_web_search
            else DisabledPhotophysicsWebSearch(),
            max_attempts=1,
        ),
        mechanism_critic=MechanismCriticAgent(
            llm_client=reviewer_llm_client,
            max_attempts=3,
        ),
        conclusion_ledger=ConclusionLedgerAgent(llm_client=reviewer_llm_client),
        output_assembler=OutputAssembler(),
        structure=McpStructureTool(transport, specs.structure),
        workers=workers,
        gate=DeterministicGate(
            GatePolicy(
                max_rounds=config.max_rounds,
            ),
            capability_registry=capability_registry,
        ),
        capability_registry=capability_registry,
    )


class MechCALOrchestrator:
    def __init__(
        self,
        config: OrchestratorConfig,
        deps: OrchestratorDeps | None = None,
    ) -> None:
        self.config = config
        self.deps = deps or default_deps(config)

    def run(
        self,
        *,
        smiles: str,
        user_query: str,
        case_id: str | None = None,
        public_case_id: str | None = None,
    ) -> CaseRun:
        case_id = case_id or uuid.uuid4().hex[:12]
        public_case_id = normalize_public_case_id(public_case_id)
        store = RunStore(self.config.run_base_dir, case_id)
        store.initialize()
        store.write_capability_catalog(self.deps.capability_registry.snapshot())

        case_input = CaseInput(
            case_id=case_id,
            smiles=smiles,
            user_query=user_query,
            metadata={"public_case_id": public_case_id},
        )
        policy = self.config.ablation_policy
        runtime = {
            "orchestrator": "mechcal",
            "langgraph": False,
            "ablation": {
                "mode": policy.mode,
                "workflow": policy.workflow,
                "enabled_agents": _enabled_agent_names(policy),
                "disabled_agents": _disabled_agent_names(policy),
                "enabled_capability_owners": list(policy.enabled_worker_agents),
                "disabled_capability_owners": list(policy.disabled_worker_agents),
                "worker_calls_per_agent_limit": policy.worker_calls_per_agent_limit,
                "max_rounds": self.config.max_rounds,
                "incremental_mechanism_portfolio": (
                    self.config.enable_incremental_mechanism_portfolio
                ),
                "subject_model": _configured_model_name(self.deps.planner),
                "prompt_files": _ablation_prompt_files(policy),
                "private_answer_key_access": "none",
                "case_specific_rules": "none",
            },
            "worker_call_counts": {
                agent_name: 0
                for agent_name in policy.enabled_worker_agents
            },
        }
        if not self.config.effective_support_auditor:
            runtime["support_auditor"] = {
                "enabled": False,
                "agent": "PhotophysicsArbiterAgent",
                "policy": _disabled_component_reason(
                    configured=self.config.enable_support_auditor,
                    mode=policy.mode,
                ),
            }
        if not self.config.effective_agenda_coverage_reviewer:
            runtime["agenda_coverage_reviewer"] = {
                "enabled": False,
                "agent": "MechanismAgendaAgent",
                "policy": _disabled_component_reason(
                    configured=self.config.enable_agenda_coverage_reviewer,
                    mode=policy.mode,
                ),
            }
        if not self.config.enable_incremental_mechanism_portfolio:
            runtime["incremental_mechanism_portfolio"] = {
                "enabled": False,
                "policy": "stateless_ranking_with_feedback_retained",
            }
        case_run = CaseRun(
            case_id=case_id,
            input=case_input,
            status="running",
            evidence_ledger=EvidenceLedger(case_id=case_id),
            claim_ledger=empty_claim_ledger(case_id),
            artifact_manifest=ArtifactManifest(case_id=case_id),
            runtime=runtime,
        )
        store.write_case_input(case_input)
        store.write_json("ablation_manifest.json", runtime["ablation"])
        try:
            case_run = self._prepare_structure(case_run, store)
            if policy.workflow == "parallel_summary":
                return self._run_parallel_worker_summary(case_run, store)
            if policy.workflow == "one_pass":
                return self._run_one_pass_mechcal(case_run, store)

            for round_index in range(1, self.config.max_rounds + 1):
                round_id = f"R{round_index:03d}"
                if self._should_review_agenda_coverage(case_run, round_index=round_index):
                    case_run = self._review_agenda_coverage(
                        case_run,
                        store,
                        round_id=round_id,
                    )
                try:
                    decision = (
                        self.deps.planner.plan_initial(case_run, round_id=round_id)
                        if round_index == 1
                        else self.deps.planner.plan_update(
                            build_planner_context(
                                case_run,
                                round_index=round_index,
                                max_rounds=self.config.max_rounds,
                            ),
                            self._planner_case_view(case_run),
                            round_id=round_id,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    if not self._can_stop_after_transient_planner_failure(
                        round_index=round_index,
                        case_run=case_run,
                        exc=exc,
                    ):
                        raise
                    case_run = self._record_planner_failure(
                        case_run,
                        store,
                        exc,
                        attempted_round_id=round_id,
                    )
                    return self._stop_after_transient_planner_failure(
                        case_run,
                        store,
                        attempted_round_id=round_id,
                    )
                case_run = case_run.touch(
                    current_round_id=round_id,
                    portfolio=decision.portfolio,
                    mechanism_priority_deltas=_merge_priority_deltas(
                        case_run.mechanism_priority_deltas,
                        decision.priority_deltas,
                    ),
                    mechanism_support_arguments=_merge_support_arguments(
                        case_run.mechanism_support_arguments,
                        decision.mechanism_support_arguments,
                    ),
                )
                if decision.mechanism_support_arguments:
                    store.write_mechanism_support_arguments(
                        case_run.mechanism_support_arguments,
                        round_id=round_id,
                    )
                if decision.priority_deltas:
                    store.write_mechanism_priority_deltas(
                        case_run.mechanism_priority_deltas,
                        round_id=round_id,
                    )
                gate_result = self.deps.gate.apply(decision, case_run)
                round_record = RoundRecord(
                    case_id=case_id,
                    round_id=round_id,
                    round_index=round_index,
                    planner_decision=decision,
                    dispatch_requests=list(gate_result.dispatch_requests),
                    gate_result=gate_result,
                )
                store.append_jsonl(
                    "history/planner_messages.jsonl",
                    decision.model_dump(mode="json"),
                )
                store.write_planner_decision(round_record)

                if gate_result.terminal:
                    return self._finish_case(
                        case_run,
                        decision,
                        round_record,
                        store,
                        gate_result,
                    )

                reports = self._dispatch_all(
                    gate_result.dispatch_requests,
                    case_run,
                    store,
                )
                case_run = self._apply_reports(case_run, reports)
                round_record = round_record.model_copy(update={"agent_reports": reports})
                case_run = self._record_round(case_run, round_record, store)
                if self._can_finalize_on_evidence_sufficiency_after_round(
                    case_run,
                    store=store,
                    gate_result=gate_result,
                    calibrated_after_dispatch=False,
                ):
                    return self._finish_converged_case(case_run, store)
                if self._should_run_post_dispatch_review(case_run):
                    case_run = self._audit_current_round(case_run, store)
                    case_run = self._update_mechanism_reviewer_portfolio(case_run, store)
                calibrated_after_dispatch = False
                if self._should_calibrate_portfolio_midrun(
                    case_run,
                    round_index=round_index,
                ):
                    case_run = self._calibrate_final_planner_portfolio(
                        case_run,
                        store,
                        force=True,
                    )
                    case_run = self._update_differential_portfolio(
                        case_run,
                        store,
                        force=True,
                    )
                    calibrated_after_dispatch = True
                if self._should_stop_on_verifier_failures(case_run):
                    return self._stop_after_repeated_verifier_failures(
                        case_run,
                        store,
                    )
                if self._should_finalize_on_convergence(case_run, store):
                    return self._finish_converged_case(case_run, store)
                if self._can_finalize_on_evidence_sufficiency_after_round(
                    case_run,
                    store=store,
                    gate_result=gate_result,
                    calibrated_after_dispatch=calibrated_after_dispatch,
                ):
                    return self._finish_converged_case(case_run, store)

            return self._stop_without_terminal_decision(case_run, store)
        except Exception as exc:  # noqa: BLE001
            self._record_runtime_failure(case_run, store, exc)
            raise

    def _run_parallel_worker_summary(
        self,
        case_run: CaseRun,
        store: RunStore,
    ) -> CaseRun:
        round_id = "R001"
        dispatches = [
            _fixed_generic_dispatch(
                self.deps.capability_registry,
                round_id=round_id,
                capability_id="macro.macro_structure_scan",
                task=(
                    "Perform one fixed broad structure-level analysis covering "
                    "rigidity, rotatable bonds, donor-acceptor organization, steric "
                    "constraints, and packing/confinement cues. Return observations "
                    "only and do not rank mechanisms."
                ),
            ),
            _fixed_generic_dispatch(
                self.deps.capability_registry,
                round_id=round_id,
                capability_id="microscopic.run_baseline_bundle",
                task=(
                    "Perform one fixed broad baseline microscopic analysis covering "
                    "available state-ordering, brightness, and electronic-state "
                    "proxies. Return observations only and do not rank mechanisms."
                ),
            ),
        ]
        gate_result = DecisionGateResult(
            gate_id=f"{round_id}:gate:parallel_worker_summary",
            round_id=round_id,
            status="allow",
            dispatch_requests=dispatches,
            notes=["Fixed generic Macro and Microscopic tasks; no Planner was used."],
        )
        case_run = case_run.touch(current_round_id=round_id)
        reports = self._dispatch_all(dispatches, case_run, store)
        case_run = self._apply_reports(case_run, reports)
        round_record = RoundRecord(
            case_id=case_run.case_id,
            round_id=round_id,
            round_index=1,
            dispatch_requests=dispatches,
            agent_reports=reports,
            gate_result=gate_result,
            notes=[
                "Parallel-Worker Summary ablation.",
                "No Planner, Coverage Reviewer, Support Auditor, feedback, or revisit.",
            ],
        )
        case_run = self._record_round(case_run, round_record, store)
        differential_portfolio = (
            self.deps.mechanism_critic.synthesize_parallel_worker_summary(
                case_run,
                worker_reports=reports,
            )
        )
        case_run = case_run.touch(
            differential_mechanism_portfolio=differential_portfolio,
            portfolio=_hypothesis_portfolio_from_differential(
                differential_portfolio
            ),
        )
        store.write_differential_mechanism_portfolio(differential_portfolio)
        case_run = self._assemble_current_planner_predictions(case_run, store)
        top_prediction = (
            case_run.mechanism_predictions[0]
            if case_run.mechanism_predictions
            else None
        )
        final_answer = FinalAnswer(
            case_id=case_run.case_id,
            current_hypothesis=(
                top_prediction.label if top_prediction is not None else "unknown"
            ),
            confidence=(
                top_prediction.confidence if top_prediction is not None else 0.0
            ),
            answer=(
                "A one-shot synthesis ranked mechanisms from one fixed Macro "
                "report and one fixed Microscopic report."
            ),
            evidence_refs=[
                item.evidence_id for item in case_run.evidence_ledger.items
            ],
            round_refs=list(case_run.round_ids),
            artifact_refs=case_run.artifact_manifest.artifact_ids(),
            limitations=[
                (
                    "No Planner, Coverage Reviewer, Support Auditor, feedback "
                    "loop, or worker revisit was used."
                ),
                "No hidden reference, source paper, or case-specific search was used.",
            ],
            mechanism_predictions=case_run.mechanism_predictions,
        )
        updated = case_run.touch(
            status="finalized",
            final_answer=final_answer,
            runtime=_runtime_with_termination(
                case_run,
                reason="parallel_worker_summary_complete",
            ),
        )
        store.write_final_answer(final_answer)
        store.write_case_run(updated)
        return updated

    def _run_one_pass_mechcal(
        self,
        case_run: CaseRun,
        store: RunStore,
    ) -> CaseRun:
        round_id = "R001"
        decision = self.deps.planner.plan_initial(case_run, round_id=round_id)
        decision = _limit_dispatches_per_worker(
            decision,
            limit=self.config.ablation_policy.worker_calls_per_agent_limit or 1,
        )
        gate_result = self.deps.gate.apply(decision, case_run)
        round_record = RoundRecord(
            case_id=case_run.case_id,
            round_id=round_id,
            round_index=1,
            planner_decision=decision,
            dispatch_requests=list(gate_result.dispatch_requests),
            gate_result=gate_result,
            notes=[
                "One-Pass MechCAL ablation.",
                "Each selected worker may be called at most once.",
            ],
        )
        store.append_jsonl(
            "history/planner_messages.jsonl",
            decision.model_dump(mode="json"),
        )
        store.write_planner_decision(round_record)
        if gate_result.terminal:
            return self._finish_case(
                case_run.touch(current_round_id=round_id),
                decision,
                round_record,
                store,
                gate_result,
            )

        case_run = case_run.touch(
            current_round_id=round_id,
            portfolio=decision.portfolio,
            mechanism_priority_deltas=list(decision.priority_deltas),
            mechanism_support_arguments=list(
                decision.mechanism_support_arguments
            ),
        )
        reports = self._dispatch_all(
            gate_result.dispatch_requests,
            case_run,
            store,
        )
        case_run = self._apply_reports(case_run, reports)
        round_record = round_record.model_copy(update={"agent_reports": reports})
        case_run = self._record_round(case_run, round_record, store)

        calibration = self.deps.planner.calibrate_final_portfolio(
            self._planner_case_view(case_run),
            round_id=round_id,
        )
        case_run = case_run.touch(
            portfolio=calibration.portfolio,
            mechanism_priority_deltas=_merge_priority_deltas(
                case_run.mechanism_priority_deltas,
                calibration.priority_deltas,
            ),
            mechanism_support_arguments=_merge_support_arguments(
                case_run.mechanism_support_arguments,
                calibration.mechanism_support_arguments,
            ),
        )
        store.write_json(
            f"rounds/{round_id}/planner_final_calibration.json",
            calibration,
        )
        store.append_jsonl(
            "history/planner_final_calibrations.jsonl",
            calibration,
        )
        if case_run.mechanism_priority_deltas:
            store.write_mechanism_priority_deltas(
                case_run.mechanism_priority_deltas,
                round_id=round_id,
            )
        if case_run.mechanism_support_arguments:
            store.write_mechanism_support_arguments(
                case_run.mechanism_support_arguments,
                round_id=round_id,
            )
        case_run = self._assemble_current_planner_predictions(case_run, store)
        diagnosis_units = build_diagnosis_units(case_run)
        case_run = case_run.touch(diagnosis_units=diagnosis_units)
        planner_final_answer = self.deps.planner.build_final_answer(
            calibration,
            case_run,
        )
        final_answer = planner_final_answer.model_copy(
            update={
                "limitations": [
                    *planner_final_answer.limitations,
                    (
                        "One-Pass MechCAL stopped after first-round targeted "
                        "evidence and one final portfolio synthesis."
                    ),
                    (
                        "Coverage Reviewer, Support Auditor, replanning, and "
                        "worker revisit were disabled."
                    ),
                ],
            }
        )
        updated = case_run.touch(
            status="finalized",
            final_answer=final_answer,
            runtime=_runtime_with_termination(
                case_run,
                reason="one_pass_complete",
            ),
        )
        store.write_final_answer(final_answer)
        store.write_case_run(updated)
        return updated

    def _prepare_structure(self, case_run: CaseRun, store: RunStore) -> CaseRun:
        artifact, failure = self.deps.structure.prepare(
            case_run.input,
            round_id="R000",
            workspace=store.root,
        )
        manifest = case_run.artifact_manifest.with_records([artifact])
        store.write_artifact_manifest(manifest)
        store.append_jsonl(
            "history/tool_events.jsonl",
            {
                "event": "prepare_structure",
                "artifact": artifact.model_dump(mode="json"),
                "failure": failure.model_dump(mode="json"),
            },
        )
        return case_run.touch(artifact_manifest=manifest)

    def _should_review_agenda_coverage(
        self,
        case_run: CaseRun,
        *,
        round_index: int,
    ) -> bool:
        if not self.config.effective_agenda_coverage_reviewer:
            return False
        if round_index == 1:
            return True
        if len(case_run.evidence_ledger.items) < 2:
            return False
        if _needs_followup_coverage_review(case_run):
            return True
        if round_index % 3 == 0:
            return True
        return _has_low_margin_portfolio(case_run)

    def _planner_case_view(self, case_run: CaseRun) -> CaseRun:
        if self.config.enable_incremental_mechanism_portfolio:
            return case_run
        return _stateless_planner_case_view(case_run)

    def _should_run_post_dispatch_review(self, case_run: CaseRun) -> bool:
        if not self.config.defer_review_until_portfolio:
            return True
        return _has_concrete_planner_portfolio(case_run)

    def _review_agenda_coverage(
        self,
        case_run: CaseRun,
        store: RunStore,
        *,
        round_id: str,
    ) -> CaseRun:
        previous_action = _previous_planner_action(store)
        try:
            coverage = self.deps.mechanism_agenda.review_coverage(
                case_run,
                round_id=round_id,
                previous_planner_action=previous_action,
            )
        except Exception as exc:  # noqa: BLE001
            if not self._can_continue_after_agenda_coverage_failure(exc):
                raise
            return self._record_agenda_coverage_failure(case_run, store, exc)
        program = program_from_coverage(
            coverage,
            capability_registry=self.deps.capability_registry,
        )
        updated = case_run.touch(
            mechanism_agenda_coverage=coverage,
            mechanism_program=program,
        )
        store.write_mechanism_agenda_coverage(coverage)
        store.write_mechanism_program(program)
        store.write_case_run(updated)
        return updated

    def _can_continue_after_agenda_coverage_failure(self, exc: Exception) -> bool:
        return (
            self.config.continue_on_agenda_coverage_failure
            and _is_transient_llm_failure(exc)
        )

    def _record_agenda_coverage_failure(
        self,
        case_run: CaseRun,
        store: RunStore,
        exc: Exception,
    ) -> CaseRun:
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "current_round_id": case_run.current_round_id,
            "agent": "MechanismAgendaAgent",
            "policy": "continue_with_previous_agenda_coverage_without_scientific_fallback",
        }
        previous = list(case_run.runtime.get("agenda_coverage_failures", []))
        updated = case_run.touch(
            runtime={
                **case_run.runtime,
                "agenda_coverage_failures": [*previous, failure][-8:],
            }
        )
        store.append_jsonl("history/agenda_coverage_failures.jsonl", failure)
        store.write_case_run(updated)
        return updated

    def _dispatch_all(
        self,
        dispatches: list[DispatchRequest],
        case_run: CaseRun,
        store: RunStore,
    ) -> list[AgentReport]:
        reports: list[AgentReport] = []
        for request in dispatches:
            worker = self.deps.workers[request.agent_name]
            plan = worker.plan(request, case_run)
            store.write_execution_plan(plan)
            store.append_jsonl(
                "history/tool_events.jsonl",
                {
                    "event": "agent_execution_plan",
                    "dispatch": request.model_dump(mode="json"),
                    "plan": plan.model_dump(mode="json"),
                },
            )
            tool_result, report = worker.run_with_trace(
                request,
                case_run,
                execution_plan=plan,
            )
            reports.append(report)
            store.write_tool_result(tool_result)
            store.write_agent_report(report)
            store.write_round_evidence_units(
                report.round_id,
                report.agent_name,
                report.evidence_units,
                route=request.route,
            )
            store.append_jsonl(
                "history/tool_events.jsonl",
                {
                    "event": "agent_report",
                    "dispatch": request.model_dump(mode="json"),
                    "tool_result": tool_result.model_dump(mode="json"),
                    "report": report.model_dump(mode="json"),
                },
            )
        return reports

    def _apply_reports(self, case_run: CaseRun, reports: list[AgentReport]) -> CaseRun:
        evidence_units = annotate_evidence_units(
            [item for report in reports for item in report.evidence_units]
        )
        artifacts = [artifact for report in reports for artifact in report.artifact_updates]
        operational_notes = [
            report.operational_note
            for report in reports
            if report.operational_note is not None
        ]
        evidence_ledger = case_run.evidence_ledger.with_items(evidence_units)
        worker_call_counts = dict(case_run.runtime.get("worker_call_counts", {}))
        for report in reports:
            worker_call_counts[report.agent_name] = (
                int(worker_call_counts.get(report.agent_name, 0)) + 1
            )
        return case_run.touch(
            evidence_ledger=evidence_ledger,
            claim_ledger=reduce_claim_ledger(case_run.case_id, evidence_ledger),
            artifact_manifest=case_run.artifact_manifest.with_records(artifacts),
            operational_notes=[*case_run.operational_notes, *operational_notes],
            runtime={
                **case_run.runtime,
                "worker_call_counts": worker_call_counts,
            },
        )

    def _record_round(
        self,
        case_run: CaseRun,
        round_record: RoundRecord,
        store: RunStore,
    ) -> CaseRun:
        round_ids = [*case_run.round_ids, round_record.round_id]
        updated = case_run.touch(round_ids=round_ids)
        store.write_json(
            f"rounds/{round_record.round_id}/evidence_ledger.json",
            updated.evidence_ledger,
        )
        if updated.claim_ledger is not None:
            store.write_json(
                f"rounds/{round_record.round_id}/claim_ledger.json",
                updated.claim_ledger,
            )
        store.write_round_record(round_record)
        store.write_case_run(updated)
        return updated

    def _update_differential_portfolio(
        self,
        case_run: CaseRun,
        store: RunStore,
        *,
        force: bool = False,
    ) -> CaseRun:
        if (
            not force
            and case_run.differential_mechanism_portfolio is not None
            and _round_index(case_run.current_round_id) > 2
        ):
            return case_run
        try:
            portfolio = self.deps.mechanism_critic.review(case_run)
        except Exception as exc:  # noqa: BLE001
            if not self._can_continue_after_mechanism_critic_failure(case_run, exc):
                raise
            return self._record_mechanism_critic_failure(case_run, store, exc)
        portfolio = _apply_planner_priorities_to_differential_portfolio(
            case_run,
            portfolio,
        )
        updated = case_run.touch(differential_mechanism_portfolio=portfolio)
        store.write_differential_mechanism_portfolio(portfolio)
        store.write_case_run(updated)
        return updated

    def _update_mechanism_reviewer_portfolio(
        self,
        case_run: CaseRun,
        store: RunStore,
    ) -> CaseRun:
        return self._update_differential_portfolio(case_run, store, force=True)

    def _should_calibrate_portfolio_midrun(
        self,
        case_run: CaseRun,
        *,
        round_index: int,
    ) -> bool:
        if round_index < 4:
            return False
        if len(case_run.evidence_ledger.items) < 5:
            return False
        if case_run.runtime.get("planner_failures"):
            return False
        if case_run.runtime.get("mechanism_critic_failures"):
            return False
        if round_index % 4 == 0:
            return True
        return _portfolio_top3_has_stalled(case_run)

    def _should_finalize_on_convergence(self, case_run: CaseRun, store: RunStore) -> bool:
        if len(case_run.round_ids) < max(1, self.config.convergence_min_rounds):
            return False
        if (
            self.config.effective_support_auditor
            and case_run.photophysics_review is None
        ):
            return False
        recent_rankings = self._recent_rankings(store, self.config.convergence_window)
        if len(recent_rankings) < max(1, self.config.convergence_window):
            return False
        latest = recent_rankings[-1]
        if not latest or latest[0] == "unknown":
            return False
        return all(ranking == latest for ranking in recent_rankings) or (
            _rankings_stable_under_low_margin(recent_rankings)
        )

    def _should_finalize_on_evidence_sufficiency(self, case_run: CaseRun) -> bool:
        if len(case_run.round_ids) < 2:
            return False
        if len(case_run.evidence_ledger.items) < 3:
            return False
        if _needs_followup_coverage_review(case_run):
            return False
        if case_run.runtime.get("planner_failures"):
            return False
        ranked = [
            item
            for item in case_run.portfolio.sorted_hypotheses()
            if item.name and item.name != "unknown"
        ]
        if len(ranked) < 2:
            return False
        top, runner_up = ranked[0], ranked[1]
        top_priority = _hypothesis_priority(top)
        runner_up_priority = _hypothesis_priority(runner_up)
        if top_priority <= 0.0:
            return False
        if top.evidence_support not in {"weak_or_proxy", "partial", "strong"}:
            return False
        if not top.evidence_refs:
            return False
        supported_labels = {
            argument.target_label
            for argument in case_run.mechanism_support_arguments
            if argument.support_level != "unsupported"
            and argument.audit_status not in {"unsupported", "invalid_ref", "overclaim"}
        }
        if top.name not in supported_labels:
            return False
        if not _top_candidate_has_specific_support(case_run, top.name):
            return False
        margin = top_priority - runner_up_priority
        if margin >= 0.08:
            return True
        return True

    def _can_finalize_on_evidence_sufficiency_after_round(
        self,
        case_run: CaseRun,
        *,
        store: RunStore,
        gate_result: DecisionGateResult,
        calibrated_after_dispatch: bool,
    ) -> bool:
        del store
        if gate_result.dispatch_requests and not calibrated_after_dispatch:
            return False
        return self._should_finalize_on_evidence_sufficiency(case_run)

    def _recent_rankings(self, store: RunStore, window: int) -> list[tuple[str, ...]]:
        rankings: list[tuple[str, ...]] = []
        round_dirs = sorted((store.root / "rounds").glob("R*"))
        for round_dir in round_dirs[-max(1, window):]:
            path = round_dir / "planner_final_calibration.json"
            if not path.is_file():
                path = round_dir / "planner_decision.json"
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            rankings.append(_ranking_from_decision_payload(payload))
        return rankings

    def _audit_current_round(self, case_run: CaseRun, store: RunStore) -> CaseRun:
        if not self.config.effective_support_auditor:
            return self._record_support_auditor_skipped(case_run, store)
        try:
            review = self.deps.photophysics_arbiter.review(case_run)
        except Exception as exc:  # noqa: BLE001
            if not self.config.continue_on_arbiter_failure:
                raise
            return self._record_verifier_failure(case_run, store, exc)
        updated = _apply_support_audits(
            case_run.touch(photophysics_review=review),
            review.support_argument_audits,
        )
        store.write_photophysics_review(review)
        if updated.mechanism_support_arguments:
            store.write_mechanism_support_arguments(
                updated.mechanism_support_arguments,
                round_id=review.round_id,
            )
        store.write_case_run(updated)
        return updated

    def _record_support_auditor_skipped(
        self,
        case_run: CaseRun,
        store: RunStore,
    ) -> CaseRun:
        round_id = case_run.current_round_id or (
            case_run.round_ids[-1] if case_run.round_ids else "FINAL"
        )
        event = {
            "round_id": round_id,
            "agent": "PhotophysicsArbiterAgent",
            "policy": "ablation_disabled_without_scientific_fallback",
        }
        previous = list(case_run.runtime.get("support_auditor_skipped_rounds", []))
        skipped_rounds = [*previous, round_id]
        updated = case_run.touch(
            runtime={
                **case_run.runtime,
                "support_auditor": {
                    "enabled": False,
                    "agent": "PhotophysicsArbiterAgent",
                    "policy": "ablation_disabled_without_scientific_fallback",
                },
                "support_auditor_skipped_rounds": list(dict.fromkeys(skipped_rounds))[-16:],
            }
        )
        store.append_jsonl("history/support_auditor_skipped.jsonl", event)
        store.write_case_run(updated)
        return updated

    def _record_verifier_failure(
        self,
        case_run: CaseRun,
        store: RunStore,
        exc: Exception,
    ) -> CaseRun:
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "current_round_id": case_run.current_round_id,
            "agent": "PhotophysicsArbiterAgent",
            "policy": "continue_without_scientific_fallback",
        }
        previous = list(case_run.runtime.get("verifier_failures", []))
        updated = case_run.touch(
            runtime={
                **case_run.runtime,
                "verifier_failures": [*previous, failure][-8:],
            }
        )
        store.append_jsonl("history/verifier_failures.jsonl", failure)
        store.write_case_run(updated)
        return updated

    def _should_stop_on_verifier_failures(self, case_run: CaseRun) -> bool:
        threshold = self.config.verifier_failure_stop_threshold
        if threshold <= 0:
            return False
        return len(case_run.runtime.get("verifier_failures", [])) >= threshold

    def _stop_after_repeated_verifier_failures(
        self,
        case_run: CaseRun,
        store: RunStore,
    ) -> CaseRun:
        case_for_final = self._assemble_current_planner_predictions(case_run, store)
        diagnosis_units = case_for_final.diagnosis_units or build_diagnosis_units(
            case_for_final
        )
        failure_count = len(case_for_final.runtime.get("verifier_failures", []))
        final_answer = FinalAnswer(
            case_id=case_for_final.case_id,
            current_hypothesis=case_for_final.portfolio.current,
            confidence=_portfolio_current_confidence(case_for_final),
            answer=(
                "The MechCAL run stopped because repeated PhotophysicsArbiterAgent "
                "LLM/API failures made further audited rounds unreliable. "
                "Existing Planner state and accepted evidence were preserved; "
                "no synthetic verifier review or scientific fallback was generated."
            ),
            evidence_refs=[item.evidence_id for item in case_for_final.evidence_ledger.items],
            round_refs=list(case_for_final.round_ids),
            artifact_refs=case_for_final.artifact_manifest.artifact_ids(),
            limitations=[
                (
                    "Stopped after "
                    f"{failure_count} verifier failures "
                    f"(threshold={self.config.verifier_failure_stop_threshold})."
                ),
                "No hidden reference, source paper, or case-specific web search was used.",
            ],
            diagnosis_units=diagnosis_units,
            mechanism_predictions=case_for_final.mechanism_predictions,
        )
        updated = case_for_final.touch(
            status="stopped",
            final_answer=final_answer,
            diagnosis_units=diagnosis_units,
            runtime=_runtime_with_termination(
                case_for_final,
                reason="verifier_failure_threshold",
            ),
        )
        store.write_final_answer(final_answer)
        store.write_case_run(updated)
        return updated

    def _can_continue_after_mechanism_critic_failure(
        self,
        case_run: CaseRun,
        exc: Exception,
    ) -> bool:
        return (
            self.config.continue_on_mechanism_critic_failure
            and _is_transient_llm_failure(exc)
        )

    def _record_mechanism_critic_failure(
        self,
        case_run: CaseRun,
        store: RunStore,
        exc: Exception,
    ) -> CaseRun:
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "current_round_id": case_run.current_round_id,
            "agent": "MechanismCriticAgent",
            "policy": "continue_with_previous_differential_portfolio_without_scientific_fallback",
        }
        previous = list(case_run.runtime.get("mechanism_critic_failures", []))
        updated = case_run.touch(
            runtime={
                **case_run.runtime,
                "mechanism_critic_failures": [*previous, failure][-8:],
            }
        )
        store.append_jsonl("history/mechanism_critic_failures.jsonl", failure)
        store.write_case_run(updated)
        return updated

    def _can_stop_after_transient_planner_failure(
        self,
        *,
        round_index: int,
        case_run: CaseRun,
        exc: Exception,
    ) -> bool:
        if not self.config.continue_on_planner_update_failure:
            return False
        if round_index <= 1 or not case_run.round_ids:
            return False
        return _is_transient_llm_failure(exc)

    def _record_planner_failure(
        self,
        case_run: CaseRun,
        store: RunStore,
        exc: Exception,
        *,
        attempted_round_id: str,
    ) -> CaseRun:
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "current_round_id": case_run.current_round_id,
            "attempted_round_id": attempted_round_id,
            "agent": "PlannerAgent",
            "policy": "stop_without_scientific_fallback",
        }
        previous = list(case_run.runtime.get("planner_failures", []))
        updated = case_run.touch(
            runtime={
                **case_run.runtime,
                "planner_failures": [*previous, failure][-8:],
            }
        )
        store.append_jsonl("history/planner_failures.jsonl", failure)
        store.write_case_run(updated)
        return updated

    def _stop_after_transient_planner_failure(
        self,
        case_run: CaseRun,
        store: RunStore,
        *,
        attempted_round_id: str,
    ) -> CaseRun:
        case_for_final = self._assemble_current_planner_predictions(case_run, store)
        diagnosis_units = case_for_final.diagnosis_units or build_diagnosis_units(
            case_for_final
        )
        failure = (case_for_final.runtime.get("planner_failures") or [{}])[-1]
        final_answer = FinalAnswer(
            case_id=case_for_final.case_id,
            current_hypothesis=case_for_final.portfolio.current,
            confidence=_portfolio_current_confidence(case_for_final),
            answer=(
                "The MechCAL run stopped after a transient Planner LLM failure while "
                f"updating {attempted_round_id}. Existing Planner state and "
                "accepted evidence were preserved; no synthetic Planner decision "
                "or scientific fallback was generated."
            ),
            evidence_refs=[item.evidence_id for item in case_for_final.evidence_ledger.items],
            round_refs=list(case_for_final.round_ids),
            artifact_refs=case_for_final.artifact_manifest.artifact_ids(),
            limitations=[
                "Stopped because a transient Planner LLM/API failure interrupted an update round.",
                "No hidden reference, source paper, or case-specific web search was used.",
                f"Failure: {failure.get('type', 'unknown')}: {failure.get('message', '')}",
            ],
            diagnosis_units=diagnosis_units,
            mechanism_predictions=case_for_final.mechanism_predictions,
        )
        updated = case_for_final.touch(
            status="stopped",
            final_answer=final_answer,
            diagnosis_units=diagnosis_units,
            runtime=_runtime_with_termination(
                case_for_final,
                reason="planner_update_failure",
            ),
        )
        store.write_final_answer(final_answer)
        store.write_case_run(updated)
        return updated

    def _finish_converged_case(self, case_run: CaseRun, store: RunStore) -> CaseRun:
        case_for_final = self._assemble_current_planner_predictions(case_run, store)
        diagnosis_units = case_for_final.diagnosis_units or build_diagnosis_units(
            case_for_final
        )
        final_answer = FinalAnswer(
            case_id=case_for_final.case_id,
            current_hypothesis=case_for_final.portfolio.current,
            confidence=_portfolio_current_confidence(case_for_final),
            answer=(
                "The MAS finalized because the Planner-owned mechanism ranking "
                f"stabilized over recent {self._round_review_descriptor()} rounds. "
                "The final mechanism "
                "predictions were assembled deterministically from the Planner "
                "portfolio and Planner support arguments."
            ),
            evidence_refs=[item.evidence_id for item in case_for_final.evidence_ledger.items],
            round_refs=list(case_for_final.round_ids),
            artifact_refs=case_for_final.artifact_manifest.artifact_ids(),
            limitations=[
                "Finalized by ranking-convergence stop before the max_rounds cap.",
                "No hidden reference, source paper, or case-specific web search was used.",
                *self._support_auditor_limitation(),
            ],
            diagnosis_units=diagnosis_units,
            mechanism_predictions=case_for_final.mechanism_predictions,
        )
        updated = case_for_final.touch(
            status="finalized",
            final_answer=final_answer,
            diagnosis_units=diagnosis_units,
            runtime=_runtime_with_termination(
                case_for_final,
                reason="ranking_convergence",
            ),
        )
        store.write_final_answer(final_answer)
        store.write_case_run(updated)
        return updated

    def _round_review_descriptor(self) -> str:
        return (
            "audited"
            if self.config.effective_support_auditor
            else "non-audited"
        )

    def _support_auditor_limitation(self) -> list[str]:
        if self.config.effective_support_auditor:
            return []
        return [
            "SupportAuditor was disabled for ablation; no PhotophysicsArbiterAgent "
            "review was generated."
        ]

    def _assemble_current_planner_predictions(
        self,
        case_run: CaseRun,
        store: RunStore,
    ) -> CaseRun:
        mechanism_predictions = self.deps.output_assembler.build_mechanism_predictions(
            case_run
        )
        updated = case_run.touch(mechanism_predictions=mechanism_predictions)
        store.write_mechanism_predictions(mechanism_predictions)
        return updated

    def _record_runtime_failure(
        self,
        case_run: CaseRun,
        store: RunStore,
        exc: Exception,
    ) -> CaseRun:
        latest_case_run = self._latest_persisted_case_run(store) or case_run
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "current_round_id": latest_case_run.current_round_id,
        }
        updated = latest_case_run.touch(
            status="failed",
            runtime={
                **_runtime_with_termination(
                    latest_case_run,
                    reason="runtime_failure",
                ),
                "failure": failure,
            },
        )
        store.append_jsonl("history/runtime_failures.jsonl", failure)
        store.write_case_run(updated)
        return updated

    def _latest_persisted_case_run(self, store: RunStore) -> CaseRun | None:
        path = store.root / "case_run.json"
        if not path.is_file():
            return None
        try:
            return CaseRun.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

    def _finish_case(
        self,
        case_run: CaseRun,
        decision,
        round_record: RoundRecord,
        store: RunStore,
        gate_result: DecisionGateResult,
    ) -> CaseRun:
        status = "finalized" if gate_result.status == "finalize" else "stopped"
        round_ids = [*case_run.round_ids, round_record.round_id]
        case_for_final = self._assemble_current_planner_predictions(
            case_run.touch(round_ids=round_ids),
            store,
        )
        final_answer = (
            self.deps.planner.build_final_answer(decision, case_for_final)
            if gate_result.status == "finalize"
            else self._gate_stop_answer(case_for_final, decision, gate_result)
        )
        updated = case_for_final.touch(
            status=status,
            final_answer=final_answer,
            runtime=_runtime_with_termination(
                case_for_final,
                reason=gate_result.terminal_reason or gate_result.status,
            ),
        )
        if final_answer.diagnosis_units:
            updated = updated.touch(diagnosis_units=final_answer.diagnosis_units)
        store.write_final_answer(final_answer)
        store.write_round_record(round_record)
        store.write_case_run(updated)
        return updated

    def _with_planner_scientific_evidence(
        self,
        case_run: CaseRun,
        *,
        store: RunStore,
        audit_current_round: bool = True,
    ) -> CaseRun:
        if audit_current_round:
            case_run = self._audit_current_round(case_run, store)
        case_run = self._update_differential_portfolio(case_run, store, force=True)
        calibrated_case_run = self._calibrate_final_planner_portfolio(case_run, store)
        if calibrated_case_run is not case_run:
            case_run = self._update_differential_portfolio(
                calibrated_case_run,
                store,
                force=True,
            )
        mechanism_predictions = self.deps.output_assembler.build_mechanism_predictions(
            case_run
        )
        case_run = case_run.touch(mechanism_predictions=mechanism_predictions)
        store.write_mechanism_predictions(mechanism_predictions)
        if self.config.enable_legacy_conclusion_ledger:
            conclusion_ledger = self.deps.conclusion_ledger.build(case_run)
            case_run = case_run.touch(conclusion_ledger=conclusion_ledger)
            store.write_case_run(case_run)
        diagnosis_units = build_diagnosis_units(case_run)
        return case_run.touch(diagnosis_units=diagnosis_units)

    def _calibrate_final_planner_portfolio(
        self,
        case_run: CaseRun,
        store: RunStore,
        *,
        force: bool = False,
    ) -> CaseRun:
        if case_run.runtime.get("planner_failures"):
            return case_run
        if case_run.runtime.get("mechanism_critic_failures"):
            return case_run
        if not force and len(case_run.evidence_ledger.items) < 3:
            return case_run
        calibrate = getattr(self.deps.planner, "calibrate_final_portfolio", None)
        if not callable(calibrate):
            return case_run
        round_id = case_run.current_round_id or (
            case_run.round_ids[-1] if case_run.round_ids else "FINAL"
        )
        decision = calibrate(self._planner_case_view(case_run), round_id=round_id)
        updated = case_run.touch(
            portfolio=decision.portfolio,
            mechanism_support_arguments=_merge_support_arguments(
                case_run.mechanism_support_arguments,
                decision.mechanism_support_arguments,
            ),
            mechanism_priority_deltas=_merge_priority_deltas(
                case_run.mechanism_priority_deltas,
                decision.priority_deltas,
            ),
        )
        store.write_json(f"rounds/{round_id}/planner_final_calibration.json", decision)
        store.append_jsonl("history/planner_final_calibrations.jsonl", decision)
        if decision.mechanism_support_arguments:
            store.write_mechanism_support_arguments(
                updated.mechanism_support_arguments,
                round_id=round_id,
            )
        if decision.priority_deltas:
            store.write_mechanism_priority_deltas(
                updated.mechanism_priority_deltas,
                round_id=round_id,
            )
        store.write_case_run(updated)
        return updated

    def _gate_stop_answer(
        self,
        case_run: CaseRun,
        decision,
        gate_result: DecisionGateResult,
    ) -> FinalAnswer:
        reason = gate_result.terminal_reason or gate_result.status
        limitations = list(gate_result.violations) or [reason]
        diagnosis_units = build_diagnosis_units(case_run)
        return FinalAnswer(
            case_id=case_run.case_id,
            current_hypothesis=decision.current_hypothesis,
            confidence=0.0,
            answer=f"The MechCAL run stopped before final synthesis because {reason}.",
            evidence_refs=[item.evidence_id for item in case_run.evidence_ledger.items],
            round_refs=list(case_run.round_ids),
            artifact_refs=case_run.artifact_manifest.artifact_ids(),
            limitations=limitations,
            diagnosis_units=diagnosis_units,
            mechanism_predictions=case_run.mechanism_predictions,
        )

    def _stop_without_terminal_decision(self, case_run: CaseRun, store: RunStore) -> CaseRun:
        case_run = self._assemble_current_planner_predictions(case_run, store)
        diagnosis_units = case_run.diagnosis_units or build_diagnosis_units(case_run)
        final_answer = FinalAnswer(
            case_id=case_run.case_id,
            current_hypothesis=case_run.portfolio.current,
            confidence=0.0,
            answer="The MechCAL run stopped after exhausting the configured round budget.",
            evidence_refs=[item.evidence_id for item in case_run.evidence_ledger.items],
            round_refs=list(case_run.round_ids),
            artifact_refs=case_run.artifact_manifest.artifact_ids(),
            limitations=["No terminal Planner decision was reached before max_rounds."],
            diagnosis_units=diagnosis_units,
            mechanism_predictions=case_run.mechanism_predictions,
        )
        updated = case_run.touch(
            status="stopped",
            final_answer=final_answer,
            diagnosis_units=diagnosis_units,
            runtime=_runtime_with_termination(
                case_run,
                reason="max_rounds_exhausted",
            ),
        )
        store.write_final_answer(final_answer)
        store.write_case_run(updated)
        return updated


def _fixed_generic_dispatch(
    capability_registry: CapabilityRegistry,
    *,
    round_id: str,
    capability_id: str,
    task: str,
) -> DispatchRequest:
    capability = capability_registry.require(capability_id)
    return DispatchRequest(
        dispatch_id=f"{round_id}:ablation:{capability_id.replace('.', ':')}",
        round_id=round_id,
        agent_name=capability.owner_agent,
        capability_id=capability.capability_id,
        task=task,
        objective=task,
        evidence_goal_family=capability.evidence_family,
        route=capability.route,
        constraints={
            "ablation_fixed_generic_task": True,
            "mechanism_ranking_forbidden": True,
        },
    )


def _limit_dispatches_per_worker(
    decision,
    *,
    limit: int,
):
    if decision.action != "dispatch" or limit < 1:
        return decision
    selected: list[DispatchRequest] = []
    counts: dict[str, int] = {}
    for request in decision.dispatch_requests:
        count = counts.get(request.agent_name, 0)
        if count >= limit:
            continue
        selected.append(request)
        counts[request.agent_name] = count + 1
    if not selected:
        return decision
    return decision.model_copy(update={"dispatch_requests": selected})


def _hypothesis_portfolio_from_differential(
    portfolio: DifferentialMechanismPortfolio,
) -> HypothesisPortfolio:
    rows = sorted(
        portfolio.rows,
        key=lambda row: row.differential_priority,
        reverse=True,
    )
    hypotheses = [
        HypothesisEntry(
            name=row.label,
            confidence=row.differential_priority,
            differential_priority=row.differential_priority,
            evidence_support=(
                "weak_or_proxy"
                if row.support_strength == "weak_proxy"
                else row.support_strength
            ),
            claim_status=row.claim_status,
            status=(
                "plausible"
                if row.trigger_status in {"triggered", "weak_trigger"}
                else "screened"
            ),
            rationale=row.rationale,
            evidence_refs=list(row.positive_evidence_refs),
            validation_needed=list(row.missing_validation),
        )
        for row in rows
    ]
    current = hypotheses[0].name if hypotheses else "unknown"
    runner_up = hypotheses[1].name if len(hypotheses) > 1 else None
    return HypothesisPortfolio(
        hypotheses=hypotheses,
        current=current,
        runner_up=runner_up,
    )


def _runtime_with_termination(
    case_run: CaseRun,
    *,
    reason: str,
) -> dict[str, object]:
    return {
        **case_run.runtime,
        "termination": {
            "reason": reason,
            "round_count": len(case_run.round_ids),
            "worker_call_counts": dict(
                case_run.runtime.get("worker_call_counts", {})
            ),
        },
    }


def _enabled_agent_names(policy: AblationPolicy) -> list[str]:
    if policy.workflow == "parallel_summary":
        return ["Macro Worker", "Microscopic Worker", "One-Shot Synthesis LLM"]
    if policy.workflow == "one_pass":
        return ["Initial Planner", *list(policy.enabled_worker_agents)]
    names = ["Initial Planner", *list(policy.enabled_worker_agents)]
    if policy.enable_agenda_coverage_reviewer:
        names.append("Agenda & Coverage Reviewer")
    if policy.enable_support_auditor:
        names.append("Support Auditor")
    names.append("Mechanism Evidence Reviewer")
    return names


def _disabled_agent_names(policy: AblationPolicy) -> list[str]:
    names = list(policy.disabled_worker_agents)
    if not policy.enable_agenda_coverage_reviewer:
        names.append("Agenda & Coverage Reviewer")
    if not policy.enable_support_auditor:
        names.append("Support Auditor")
    if policy.workflow == "parallel_summary":
        names.append("Planner")
    if policy.workflow in {"parallel_summary", "one_pass"}:
        names.append("Mechanism Evidence Reviewer")
    return names


def _ablation_prompt_files(policy: AblationPolicy) -> list[str]:
    if policy.workflow == "parallel_summary":
        return ["parallel_worker_summary_synthesis.md"]
    if policy.workflow == "one_pass":
        return [
            "planner_decision_compact.md",
            "planner_initial_evidence_ranking.md",
        ]
    return [
        "planner_decision_compact.md",
        "planner_initial_evidence_ranking.md",
        "mechanism_agenda_coverage.md",
        "photophysics_arbiter.md",
        "mechanism_critic.md",
    ]


def _configured_model_name(agent: object) -> str:
    client = getattr(agent, "llm_client", None)
    settings = getattr(client, "settings", None)
    model = getattr(settings, "model", None)
    return str(model or "unknown")


def _disabled_component_reason(*, configured: bool, mode: str) -> str:
    if not configured:
        return "explicitly_disabled_without_scientific_fallback"
    return f"disabled_by_ablation_mode:{mode}"


def _ranking_from_decision_payload(payload: dict[str, object]) -> tuple[str, ...]:
    portfolio = payload.get("portfolio")
    if not isinstance(portfolio, dict):
        return ()
    hypotheses = portfolio.get("hypotheses")
    if not isinstance(hypotheses, list):
        return ()
    ranked: list[tuple[float, str]] = []
    for item in hypotheses:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        confidence = item.get("confidence")
        ranked.append((float(confidence) if isinstance(confidence, int | float) else 0.0, name))
    ranked.sort(reverse=True)
    return tuple(name for _, name in ranked[:3])


def _stateless_planner_case_view(case_run: CaseRun) -> CaseRun:
    """Return a case-local view that hides prior ranking memory.

    The Planner still sees public evidence, support audits, coverage feedback,
    and the differential evidence review. Only the previous Planner ranking,
    deltas, and exported predictions are removed, so feedback remains available
    without carrying forward old priority state.
    """

    prior_planner_ranking_present = (
        case_run.portfolio.current not in {"", "unknown"}
        and bool(case_run.portfolio.hypotheses)
    )
    return case_run.touch(
        portfolio=HypothesisPortfolio(),
        mechanism_priority_deltas=[],
        mechanism_predictions=[],
        diagnosis_units=[],
        final_answer=None,
        runtime={
            **case_run.runtime,
            "incremental_mechanism_portfolio": {
                "enabled": False,
                "policy": "stateless_ranking_with_feedback_retained",
                "prior_planner_ranking_present": prior_planner_ranking_present,
            },
        },
    )


def _rankings_stable_under_low_margin(rankings: list[tuple[str, ...]]) -> bool:
    if not rankings:
        return False
    latest = rankings[-1]
    if len(latest) < 3 or latest[0] == "unknown":
        return False
    latest_top1 = latest[0]
    latest_top3 = set(latest[:3])
    if len(latest_top3) < 3:
        return False
    for ranking in rankings:
        if len(ranking) < 3:
            return False
        if ranking[0] != latest_top1:
            return False
        if set(ranking[:3]) != latest_top3:
            return False
    return True


def _previous_planner_action(store: RunStore) -> str | None:
    round_dirs = sorted((store.root / "rounds").glob("R*"))
    for round_dir in reversed(round_dirs):
        path = round_dir / "planner_decision.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        action = payload.get("action")
        if not isinstance(action, str) or not action.strip():
            return None
        dispatches = payload.get("dispatch_requests")
        if isinstance(dispatches, list):
            routes = [
                str(item.get("capability_id") or item.get("route") or "").strip()
                for item in dispatches
                if isinstance(item, dict)
            ]
            routes = [item for item in routes if item]
            if routes:
                return f"{action}: " + ", ".join(routes[:3])
        return action.strip()
    return None


def _portfolio_top3_has_stalled(case_run: CaseRun) -> bool:
    top_items = [
        item
        for item in case_run.portfolio.sorted_hypotheses()[:3]
        if item.name and item.name != "unknown"
    ]
    if len(top_items) < 3:
        return False
    if not any(item.evidence_support in {"unsupported", "weak_or_proxy"} for item in top_items):
        return False
    top_labels = {item.name for item in top_items}
    round_ids = sorted({delta.round_id for delta in case_run.mechanism_priority_deltas})
    if len(round_ids) < 2:
        return False
    recent_rounds = set(round_ids[-2:])
    recent_top_deltas = [
        delta
        for delta in case_run.mechanism_priority_deltas
        if delta.round_id in recent_rounds and delta.label in top_labels
    ]
    if not recent_top_deltas:
        return False
    return all(abs(delta.priority_delta) < 0.02 for delta in recent_top_deltas)


def _top_candidate_has_specific_support(case_run: CaseRun, label: str) -> bool:
    refs = {
        ref
        for item in case_run.portfolio.hypotheses
        if item.name == label
        for ref in item.evidence_refs
    }
    if not refs:
        return False
    evidence_by_id = {item.evidence_id: item for item in case_run.evidence_ledger.items}
    return any(
        evidence_by_id.get(ref) is not None
        and evidence_by_id[ref].capability_id in DISCRIMINATING_FOLLOWUP_CAPABILITY_IDS
        for ref in refs
    )


def _hypothesis_priority(item) -> float:  # noqa: ANN001
    value = item.differential_priority
    if value is None:
        value = item.confidence
    return float(value or 0.0)


def _merge_support_arguments(
    existing: list[MechanismSupportArgument],
    new_items: list[MechanismSupportArgument],
) -> list[MechanismSupportArgument]:
    by_id = {item.support_id: item for item in existing}
    by_id.update({item.support_id: item for item in new_items})
    return list(by_id.values())


def _merge_priority_deltas(
    existing: list[MechanismPriorityDelta],
    new_items: list[MechanismPriorityDelta],
) -> list[MechanismPriorityDelta]:
    by_key = {(item.round_id, item.label): item for item in existing}
    by_key.update({(item.round_id, item.label): item for item in new_items})
    return list(by_key.values())


def _apply_support_audits(
    case_run: CaseRun,
    audits: list[MechanismSupportAudit],
) -> CaseRun:
    if not audits or not case_run.mechanism_support_arguments:
        return case_run
    audits_by_id = {item.support_id: item for item in audits}
    updated_arguments: list[MechanismSupportArgument] = []
    for argument in case_run.mechanism_support_arguments:
        audit = audits_by_id.get(argument.support_id)
        if audit is None:
            updated_arguments.append(argument)
            continue
        boundary = argument.boundary
        if audit.boundary_patch:
            boundary = f"{boundary} Auditor boundary: {audit.boundary_patch}"
        support_level = audit.recommended_support_level or argument.support_level
        updated_arguments.append(
            argument.model_copy(
                update={
                    "support_level": support_level,
                    "boundary": boundary,
                    "audit_status": audit.audit_status,
                    "audit_notes": [
                        *argument.audit_notes,
                        *audit.audit_notes,
                    ],
                }
            )
        )
    return case_run.touch(mechanism_support_arguments=updated_arguments)


def _apply_planner_priorities_to_differential_portfolio(
    case_run: CaseRun,
    portfolio: DifferentialMechanismPortfolio,
) -> DifferentialMechanismPortfolio:
    priorities = {
        item.name: (
            item.differential_priority
            if item.differential_priority is not None
            else item.confidence
        )
        for item in case_run.portfolio.hypotheses
        if item.name and item.name != "unknown"
    }
    if not priorities:
        return portfolio
    planner_is_uninformative = _planner_portfolio_is_uninformative(case_run)
    rows = []
    for row in portfolio.rows:
        if row.label in priorities:
            planner_priority = priorities[row.label]
            adjusted_priority = _reviewer_feedback_priority_floor(
                row,
                planner_priority=planner_priority,
                max_existing_priority=max(priorities.values(), default=0.0),
                planner_is_uninformative=planner_is_uninformative,
            )
            rows.append(
                row.model_copy(
                    update={"differential_priority": adjusted_priority}
                )
            )
        else:
            rows.append(
                row.model_copy(
                    update={
                        "differential_priority": min(
                            row.differential_priority,
                            0.05,
                        )
                    }
                )
            )
    return portfolio.model_copy(update={"rows": rows})


def _reviewer_feedback_priority_floor(
    row: DifferentialMechanismPortfolioRow,
    *,
    planner_priority: float,
    max_existing_priority: float,
    planner_is_uninformative: bool,
) -> float:
    if not row.positive_evidence_refs:
        return planner_priority
    if row.support_strength == "unsupported":
        return planner_priority
    if row.trigger_status not in {"triggered", "weak_trigger"}:
        return planner_priority
    if planner_is_uninformative:
        return max(planner_priority, min(row.differential_priority, 0.28))
    # Keep Planner as the ranking owner, but do not erase source-grounded reviewer
    # distinctions among low-margin candidates before the next Planner update.
    reviewer_floor = min(max(row.differential_priority, 0.12), max_existing_priority + 0.02, 0.28)
    return max(planner_priority, reviewer_floor)


def _planner_portfolio_is_uninformative(case_run: CaseRun) -> bool:
    hypotheses = [
        item
        for item in case_run.portfolio.hypotheses
        if item.name and item.name != "unknown"
    ]
    if not hypotheses:
        return True
    if any(item.evidence_refs for item in hypotheses):
        return False
    if any(item.evidence_support != "unsupported" for item in hypotheses):
        return False
    return max((_hypothesis_priority(item) for item in hypotheses), default=0.0) <= 0.02


def _portfolio_current_confidence(case_run: CaseRun) -> float:
    current = case_run.portfolio.current
    for hypothesis in case_run.portfolio.hypotheses:
        if hypothesis.name == current:
            return hypothesis.confidence
    return 0.0


def _has_low_margin_portfolio(case_run: CaseRun, *, threshold: float = 0.05) -> bool:
    ranked = [
        item
        for item in case_run.portfolio.sorted_hypotheses()[:4]
        if item.name and item.name != "unknown"
    ]
    if len(ranked) < 2:
        return False
    for current, next_item in zip(ranked, ranked[1:], strict=False):
        current_priority = (
            current.differential_priority
            if current.differential_priority is not None
            else current.confidence
        )
        next_priority = (
            next_item.differential_priority
            if next_item.differential_priority is not None
            else next_item.confidence
        )
        if current_priority <= 0.0 and next_priority <= 0.0:
            continue
        if abs(float(current_priority) - float(next_priority)) < threshold:
            return True
    return False


def _needs_followup_coverage_review(case_run: CaseRun) -> bool:
    executed = {
        str(item.capability_id)
        for item in case_run.evidence_ledger.items
        if str(item.capability_id or "").strip()
    }
    if not executed:
        return False
    packing_triggered = bool(
        executed
        & {
            "macro.screen_aggregation_prone_scaffold",
            "macro.run_solid_state_emission_proxy",
        }
    )
    packing_followups = {
        "macro.run_dimer_packing_proxy",
        "macro.run_aggregate_contact_proxy",
        "macro.run_crystal_restriction_checklist",
    }
    if packing_triggered and not packing_followups <= executed:
        return True
    state_triggered = "microscopic.run_baseline_bundle" in executed
    if state_triggered and "microscopic.run_bright_dark_state_ordering" not in executed:
        return True
    return False


def _has_concrete_planner_portfolio(case_run: CaseRun) -> bool:
    return any(
        item.name
        and item.name != "unknown"
        and (
            (item.differential_priority is not None and item.differential_priority > 0.0)
            or item.confidence > 0.0
            or bool(item.evidence_refs)
        )
        for item in case_run.portfolio.hypotheses
    )


def _round_index(round_id: str | None) -> int:
    if not round_id or not round_id.startswith("R"):
        return 0
    try:
        return int(round_id[1:])
    except ValueError:
        return 0


def _is_transient_llm_failure(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    non_transient_terms = (
        "insufficient funds",
        "quota exhausted",
        "billing",
        "invalid api key",
        "authentication",
        "permission denied",
    )
    if any(term in text for term in non_transient_terms):
        return False
    transient_terms = (
        "apitimeouterror",
        "timeout",
        "api connection",
        "apiconnectionerror",
        "rate limit",
        "ratelimiterror",
        "readtimeout",
        "connecttimeout",
        "remoteprotocolerror",
        "connection reset",
        "failed to produce a valid",
        "schema validation failed",
        "rows must cover mechanism_pool exactly",
        "did not return any usable",
        "unterminated json object",
        "did not contain a json object",
        "temporarily unavailable",
        "server error",
        "503",
        "502",
        "504",
    )
    return any(term in text for term in transient_terms)
