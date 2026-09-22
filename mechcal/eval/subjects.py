from __future__ import annotations

import json
import re
import signal
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from mechcal.capabilities import CapabilityRegistry, default_capability_registry
from mechcal.eval.llm import OpenAITextClient
from mechcal.eval.prompts import load_eval_prompt
from mechcal.eval.schemas import EvalCase, NormalizedPrediction, SubjectRawOutput
from mechcal.public_ids import normalize_public_case_id
from mechcal.runtime.llm import OpenAICompatibleSettings, OpenAIJsonClient
from mechcal.schemas import (
    AgentExecutionPlan,
    ArtifactManifest,
    CaseInput,
    CaseRun,
    DispatchRequest,
    EvidenceLedger,
    ToolExecutionResult,
)
from mechcal.tools import LocalMacroTool, LocalMicroscopicTool, LocalStructureTool

ZERO_SHOT_PROMPT = "zero_shot_llm.md"
ZERO_SHOT_PROMPT_VERSION = "2026-06-01"
STRUCTURE_LLM_PROMPT = "structure_llm.md"
STRUCTURE_LLM_PROMPT_VERSION = "2026-06-01"
REACT_TOOL_LLM_PROMPT = "react_tool_llm.md"
REACT_TOOL_LLM_PROMPT_VERSION = "2026-06-03"
REACT_TOOL_LLM_FULL_PROMPT = "react_tool_llm_full.md"
REACT_TOOL_LLM_FULL_PROMPT_VERSION = "2026-06-03"
REACT_TOOL_LLM_FULL_EVIDENCE_FORMATTED_PROMPT = (
    "react_tool_llm_full_evidence_formatted.md"
)
REACT_TOOL_LLM_FULL_EVIDENCE_FORMATTED_PROMPT_VERSION = "2026-07-07"

DEFAULT_REACT_TOOL_IDS = (
    "macro.macro_structure_scan",
    "macro.screen_donor_acceptor_architecture",
    "macro.screen_lipophilicity_proxy",
    "macro.screen_rotor_rim_prior",
    "macro.screen_esipt_structural_motif",
    "macro.screen_aggregation_prone_scaffold",
    "macro.run_solid_state_emission_proxy",
    "macro.run_crystal_restriction_checklist",
    "macro.run_water_fraction_aggregation_checklist",
    "macro.screen_pi_stacking_prone_geometry",
    "microscopic.run_baseline_bundle",
    "microscopic.run_conformer_bundle",
    "microscopic.run_torsion_snapshots",
    "microscopic.unsupported_excited_state_relaxation",
)

DEFAULT_REACT_FULL_TOOL_IDS = (
    "macro.macro_structure_scan",
    "macro.screen_donor_acceptor_architecture",
    "macro.screen_rotor_rim_prior",
    "macro.screen_esipt_structural_motif",
    "macro.screen_aggregation_prone_scaffold",
    "macro.screen_pi_stacking_prone_geometry",
    "macro.run_solid_state_emission_proxy",
    "macro.run_crystal_restriction_checklist",
    "macro.run_water_fraction_aggregation_checklist",
    "macro.screen_lipophilicity_proxy",
    "microscopic.unsupported_excited_state_relaxation",
)


class BenchmarkSubject(Protocol):
    subject_id: str

    def run(self, case: EvalCase) -> SubjectRawOutput | NormalizedPrediction:
        ...


class ZeroShotLLMBaselineSubject:
    subject_id = "zero_shot_llm"

    def __init__(
        self,
        *,
        client: OpenAITextClient | None = None,
        settings: OpenAICompatibleSettings | None = None,
    ) -> None:
        self.client = client or OpenAITextClient(settings)

    @property
    def model(self) -> str:
        return self.client.settings.model

    def run(self, case: EvalCase) -> SubjectRawOutput:
        raw_text = self.client.complete_text(
            system_prompt=load_eval_prompt(ZERO_SHOT_PROMPT),
            payload={
                "case_id": _prompt_case_id(case),
                "smiles": case.smiles,
                "user_query": case.user_query,
            },
        )
        return SubjectRawOutput(
            raw_output_id=f"{case.case_id}:zero_shot_llm:raw",
            case_id=case.case_id,
            subject_id=self.subject_id,
            subject_kind="zero_shot_llm",
            model=self.model,
            prompt_version=ZERO_SHOT_PROMPT_VERSION,
            raw_text=raw_text,
        )


class StructureLLMBaselineSubject:
    subject_id = "structure_llm"

    def __init__(
        self,
        *,
        client: OpenAIJsonClient | None = None,
        settings: OpenAICompatibleSettings | None = None,
    ) -> None:
        self.client = client or OpenAIJsonClient(settings)

    @property
    def model(self) -> str:
        return self.client.settings.model

    def run(self, case: EvalCase) -> SubjectRawOutput:
        response = self.client.complete_json(
            system_prompt=load_eval_prompt(STRUCTURE_LLM_PROMPT),
            payload={
                "case_id": _prompt_case_id(case),
                "smiles": case.smiles,
                "user_query": case.user_query,
            },
        )
        raw_output = SubjectRawOutput(
            raw_output_id=f"{case.case_id}:structure_llm:raw",
            case_id=case.case_id,
            subject_id=self.subject_id,
            subject_kind="structure_llm",
            model=self.model,
            prompt_version=STRUCTURE_LLM_PROMPT_VERSION,
            raw_text="",
            raw_json=response,
        )
        return raw_output


class ReactStepDecision(BaseModel):
    thought: str
    action: str
    tool_name: str | None = None
    action_input: dict[str, object] = Field(default_factory=dict)
    tool_args: dict[str, object] = Field(default_factory=dict)
    final_answer: str | None = None


class ReactToolLLMBaselineSubject:
    subject_id = "react_tool_llm"
    subject_kind = "react_tool_llm"
    prompt_name = REACT_TOOL_LLM_PROMPT
    prompt_version = REACT_TOOL_LLM_PROMPT_VERSION
    separate_action_input = False
    fallback_coverage_tool_ids: tuple[str, ...] = ("macro.macro_structure_scan",)
    default_finalization_label = "ReAct baseline"
    finalize_on_redundant_after_coverage = False
    finalize_after_public_coverage = False

    def __init__(
        self,
        *,
        client: OpenAIJsonClient | None = None,
        settings: OpenAICompatibleSettings | None = None,
        capability_registry: CapabilityRegistry | None = None,
        run_base_dir: Path = Path("outputs/evaluation/react_tool_runs"),
        max_steps: int = 4,
        tool_ids: tuple[str, ...] = DEFAULT_REACT_TOOL_IDS,
        enable_amesp: bool = False,
        decision_timeout_seconds: float = 20.0,
        finalization_timeout_seconds: float | None = None,
    ) -> None:
        self.client = client or OpenAIJsonClient(settings)
        self.capability_registry = capability_registry or default_capability_registry()
        self.run_base_dir = run_base_dir
        self.max_steps = max(1, max_steps)
        self.tool_ids = tool_ids
        self.decision_timeout_seconds = max(1.0, decision_timeout_seconds)
        if finalization_timeout_seconds is None:
            finalization_timeout_seconds = self.decision_timeout_seconds
        self.finalization_timeout_seconds = max(1.0, finalization_timeout_seconds)
        self.structure_tool = LocalStructureTool()
        self.macro_tool = LocalMacroTool()
        self.microscopic_tool = LocalMicroscopicTool(enable_amesp=enable_amesp)

    @property
    def model(self) -> str:
        return self.client.settings.model

    def run(self, case: EvalCase) -> SubjectRawOutput:
        run_id = uuid.uuid4().hex[:8]
        run_dir = self.run_base_dir.expanduser().resolve() / case.case_id / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        case_run = self._initial_case_run(case)
        case_run, structure_observation = self._prepare_structure(case_run, run_dir)
        transcript = [structure_observation]
        tool_cards = self._tool_cards()
        final_answer = ""

        for step_index in range(1, self.max_steps + 1):
            decision = self._decide(
                case=case,
                transcript=transcript,
                tool_cards=tool_cards,
                step_index=step_index,
                force_final=False,
            )
            transcript.append(f"Thought {step_index}: {decision.thought}")
            if decision.action == "final":
                final_answer = (decision.final_answer or "").strip()
                transcript.append(f"Final Answer: {final_answer}")
                break
            if decision.action != "tool":
                transcript.append(
                    f"Observation {step_index}: invalid action {decision.action!r}; "
                    "valid actions are tool or final."
                )
                continue
            if self._should_finalize_redundant_tool(decision, transcript):
                tool_name = str(decision.tool_name or "<missing>").strip()
                transcript.append(
                    f"Observation {step_index}: skipped redundant tool request "
                    f"{tool_name!r} after public coverage was complete; finalizing "
                    "from transcript observations."
                )
                final_answer = self._fallback_final_answer(transcript, error=None)
                transcript.append(
                    "Thought final: Generate a transcript-only final diagnosis after "
                    "public coverage is complete and the requested tool is redundant."
                )
                transcript.append(f"Final Answer: {final_answer}")
                break
            observation, case_run = self._execute_tool_decision(
                decision,
                case_run,
                step_index=step_index,
            )
            transcript.append(observation)
            if self._should_finalize_after_public_coverage(transcript):
                final_answer = self._fallback_final_answer(transcript, error=None)
                transcript.append(
                    "Thought final: Generate a transcript-only final diagnosis after "
                    "all public coverage tools have returned observations."
                )
                transcript.append(f"Final Answer: {final_answer}")
                break

        if not final_answer:
            decision = self._decide(
                case=case,
                transcript=transcript,
                tool_cards=tool_cards,
                step_index=self.max_steps + 1,
                force_final=True,
            )
            final_answer = (decision.final_answer or "").strip()
            transcript.append(f"Thought final: {decision.thought}")
            transcript.append(f"Final Answer: {final_answer}")

        raw_text = "\n".join(transcript)
        raw_json = {
            "react_policy": {
                "state_management": "transcript_only_no_mas_hypothesis_or_evidence_ledger",
                "max_steps": self.max_steps,
                "tool_ids": list(self.tool_ids),
                "fallback_coverage_tool_ids": list(self.fallback_coverage_tool_ids),
                "decision_timeout_seconds": self.decision_timeout_seconds,
                "finalization_timeout_seconds": self.finalization_timeout_seconds,
            },
            "transcript": transcript,
            "final_answer": final_answer,
        }
        return SubjectRawOutput(
            raw_output_id=f"{case.case_id}:{self.subject_id}:{run_id}:raw",
            case_id=case.case_id,
            subject_id=self.subject_id,
            subject_kind=self.subject_kind,  # type: ignore[arg-type]
            model=self.model,
            prompt_version=self.prompt_version,
            raw_text=raw_text,
            raw_json=raw_json,
            metadata={
                "run_dir": str(run_dir),
                "adapter_required": "baseline_common_adapter",
                "hidden_reference_access": False,
                "public_case_id": _prompt_case_id(case),
            },
        )

    def _initial_case_run(self, case: EvalCase) -> CaseRun:
        return CaseRun(
            case_id=case.case_id,
            input=CaseInput(
                case_id=case.case_id,
                smiles=case.smiles,
                user_query=case.user_query,
                metadata={"public_case_id": _prompt_case_id(case)},
            ),
            status="running",
            evidence_ledger=EvidenceLedger(case_id=case.case_id),
            artifact_manifest=ArtifactManifest(case_id=case.case_id),
            runtime={
                "subject": self.subject_id,
                "state_management": "transcript_only",
            },
        )

    def _prepare_structure(self, case_run: CaseRun, run_dir: Path) -> tuple[CaseRun, str]:
        artifact, failure = self.structure_tool.prepare(
            case_run.input,
            round_id="R000",
            workspace=run_dir,
        )
        updated = case_run.touch(
            artifact_manifest=case_run.artifact_manifest.with_records([artifact])
        )
        metadata = artifact.metadata
        fields = [
            "tool: structure.prepare",
            f"status: {artifact.status}",
            f"artifact_id: {artifact.artifact_id}",
            f"artifact_kind: {artifact.kind}",
            f"failure_kind: {failure.kind}",
            f"canonical_smiles: {metadata.get('canonical_smiles')}",
            f"heavy_atom_count: {metadata.get('heavy_atom_count')}",
            f"atom_count: {metadata.get('atom_count')}",
            f"conformer_count: {metadata.get('conformer_count')}",
            f"force_field: {metadata.get('force_field')}",
        ]
        return (
            updated,
            "Observation 0: " + "; ".join(item for item in fields if not item.endswith(": None")),
        )

    def _decide(
        self,
        *,
        case: EvalCase,
        transcript: list[str],
        tool_cards: list[dict[str, object]],
        step_index: int,
        force_final: bool,
    ) -> ReactStepDecision:
        try:
            timeout_seconds = (
                self.finalization_timeout_seconds
                if force_final
                else self.decision_timeout_seconds
            )
            with _react_decision_deadline(timeout_seconds):
                payload = self.client.complete_json(
                    system_prompt=load_eval_prompt(self.prompt_name),
                    payload={
                        "case_id": _prompt_case_id(case),
                        "smiles": case.smiles,
                        "user_query": case.user_query,
                        "step_index": step_index,
                        "max_steps": self.max_steps,
                        "force_final": force_final,
                        "available_tools": tool_cards,
                        "transcript": transcript,
                        "coverage_status": self._coverage_status(transcript),
                    },
                )
            decision = ReactStepDecision.model_validate(payload)
            if (
                decision.action == "final"
                and not force_final
                and self._next_fallback_tool(transcript) is not None
            ):
                return self._coverage_fallback_decision(
                    transcript,
                    reason="Model attempted to finalize before completing public coverage.",
                )
            return decision
        except Exception as exc:  # noqa: BLE001
            if force_final:
                return ReactStepDecision(
                    thought=f"Finalization fallback after invalid ReAct response: {exc}",
                    action="final",
                    final_answer=self._fallback_final_answer(transcript, error=exc),
                )
            next_tool = self._next_fallback_tool(transcript)
            if next_tool is not None:
                return self._coverage_fallback_decision(transcript, reason=str(exc))
            if _has_tool_observation(transcript):
                return ReactStepDecision(
                    thought=f"Finalize after invalid ReAct response: {exc}",
                    action="final",
                    final_answer=self._fallback_final_answer(transcript, error=exc),
                )
            return ReactStepDecision(
                thought=f"Invalid ReAct response: {exc}",
                action="tool",
                tool_name="macro.macro_structure_scan",
            )

    def _coverage_status(self, transcript: list[str]) -> dict[str, object]:
        called = sorted(_called_tool_ids(transcript))
        remaining = [
            tool_id for tool_id in self.fallback_coverage_tool_ids if tool_id not in called
        ]
        return {
            "called_tool_ids": called,
            "remaining_public_coverage_tool_ids": remaining,
        }

    def _next_fallback_tool(self, transcript: list[str]) -> str | None:
        called = _called_tool_ids(transcript)
        for tool_id in self.fallback_coverage_tool_ids:
            if tool_id in self.tool_ids and tool_id not in called:
                return tool_id
        return None

    def _coverage_fallback_decision(
        self,
        transcript: list[str],
        *,
        reason: str,
    ) -> ReactStepDecision:
        tool_id = self._next_fallback_tool(transcript)
        if tool_id is None:
            return ReactStepDecision(
                thought=f"Coverage fallback has no remaining tool after: {reason}",
                action="final",
                final_answer=self._fallback_final_answer(transcript, error=None),
            )
        return ReactStepDecision(
            thought=(
                f"Coverage fallback selects {tool_id}. Reason: {reason}. "
                "This is a public ReAct checklist step, not a MAS evidence ledger."
            ),
            action="tool",
            tool_name=tool_id,
            action_input={},
            tool_args={},
        )

    def _fallback_final_answer(
        self,
        transcript: list[str],
        *,
        error: Exception | None,
    ) -> str:
        del transcript
        suffix = f" Fallback reason: {error}" if error is not None else ""
        return (
            f"The {self.default_finalization_label} could not produce a valid final JSON "
            "response. Use the transcript observations as bounded tool evidence."
            f"{suffix}"
        )

    def _should_finalize_redundant_tool(
        self,
        decision: ReactStepDecision,
        transcript: list[str],
    ) -> bool:
        if not self.finalize_on_redundant_after_coverage:
            return False
        if self._next_fallback_tool(transcript) is not None:
            return False
        tool_name = str(decision.tool_name or "").strip()
        return bool(tool_name and tool_name in _called_tool_ids(transcript))

    def _should_finalize_after_public_coverage(self, transcript: list[str]) -> bool:
        return self.finalize_after_public_coverage and self._next_fallback_tool(transcript) is None

    def _tool_cards(self) -> list[dict[str, object]]:
        cards_by_id = {
            str(item["capability_id"]): item for item in self.capability_registry.cards()
        }
        return [
            _compact_tool_card(cards_by_id[tool_id])
            for tool_id in self.tool_ids
            if tool_id in cards_by_id
        ]

    def _execute_tool_decision(
        self,
        decision: ReactStepDecision,
        case_run: CaseRun,
        *,
        step_index: int,
    ) -> tuple[str, CaseRun]:
        tool_name = str(decision.tool_name or "").strip()
        if tool_name not in self.tool_ids:
            return (
                f"Action {step_index}: {tool_name or '<missing>'}\n"
                f"Observation {step_index}: unknown or disallowed tool.",
                case_run,
            )
        capability = self.capability_registry.require(tool_name)
        dispatch_id = f"react:{step_index:03d}:{capability.capability_id}"
        round_id = f"R{step_index:03d}"
        request = DispatchRequest(
            dispatch_id=dispatch_id,
            round_id=round_id,
            agent_name=capability.owner_agent,
            capability_id=capability.capability_id,
            task=f"ReAct baseline tool call {step_index}: {capability.description}",
            objective="Collect bounded atomic observations for a baseline final answer.",
            evidence_goal_family=capability.evidence_family,
            route=capability.route,
            input_refs=case_run.artifact_manifest.artifact_ids(),
            constraints={"react_baseline": True},
        )
        plan = AgentExecutionPlan(
            plan_id=f"{dispatch_id}:plan",
            round_id=round_id,
            agent_name=capability.owner_agent,
            dispatch_id=dispatch_id,
            capability_id=capability.capability_id,
            selected_route=capability.route,
            tool_steps=[f"execute {capability.route}"],
            tool_args=_decision_tool_args(decision),
            tool_arg_rationale="Arguments supplied by ReAct baseline action JSON.",
            parameter_adjustment_hint="No MAS planner state is available.",
            artifact_refs=case_run.artifact_manifest.artifact_ids(),
            planning_mode="llm",
        )
        try:
            result = self._run_tool(request, case_run, plan)
        except Exception as exc:  # noqa: BLE001
            action_text = self._render_action(step_index, tool_name, _decision_tool_args(decision))
            return (
                f"{action_text}\n"
                f"Observation {step_index}: status: failed; capability_id: {tool_name}; "
                f"failure_kind: runtime_failed; failure_message: {exc}",
                case_run,
            )
        if result.artifact_updates:
            case_run = case_run.touch(
                artifact_manifest=case_run.artifact_manifest.with_records(
                    list(result.artifact_updates)
                )
            )
        return (
            f"{self._render_action(step_index, tool_name, _decision_tool_args(decision))}\n"
            f"Observation {step_index}: {self._render_tool_result(result)}",
            case_run,
        )

    def _render_action(
        self,
        step_index: int,
        tool_name: str,
        tool_args: dict[str, object],
    ) -> str:
        if self.separate_action_input:
            return (
                f"Action {step_index}: {tool_name}\n"
                f"Action Input {step_index}: "
                f"{json.dumps(tool_args, ensure_ascii=False, sort_keys=True)}"
            )
        action = {
            "tool_name": tool_name,
            "tool_args": tool_args,
        }
        return f"Action {step_index}: {json.dumps(action, ensure_ascii=False, sort_keys=True)}"

    def _run_tool(
        self,
        request: DispatchRequest,
        case_run: CaseRun,
        plan: AgentExecutionPlan,
    ) -> ToolExecutionResult:
        if request.agent_name == "macro":
            return self.macro_tool.run(request, case_run, execution_plan=plan)
        return self.microscopic_tool.run(request, case_run, execution_plan=plan)

    def _render_tool_result(self, result: ToolExecutionResult) -> str:
        fields = [
            f"status: {result.status}",
            f"capability_id: {result.capability_id}",
            f"selected_route: {result.selected_route}",
            f"summary: {result.summary}",
            f"failure_kind: {result.failure.kind}",
        ]
        fields.extend(_flatten_structured_results(result.structured_results))
        for index, item in enumerate(result.evidence_units, start=1):
            fields.append(
                "evidence_{index}: claim: {claim}; basis: {basis}; support: {support}; "
                "status: {status}; summary: {summary}; limits: {limits}; tags: {tags}".format(
                    index=index,
                    claim=item.claim,
                    basis=item.basis,
                    support=item.support,
                    status=item.status,
                    summary=item.summary,
                    limits=" | ".join(item.limits),
                    tags=", ".join(item.observable_tags),
                )
            )
        return "; ".join(fields)


class ReactToolLLMFullBaselineSubject(ReactToolLLMBaselineSubject):
    subject_id = "react_tool_llm_full"
    subject_kind = "react_tool_llm_full"
    prompt_name = REACT_TOOL_LLM_FULL_PROMPT
    prompt_version = REACT_TOOL_LLM_FULL_PROMPT_VERSION
    separate_action_input = True
    fallback_coverage_tool_ids = DEFAULT_REACT_FULL_TOOL_IDS
    default_finalization_label = "full ReAct baseline"
    finalize_on_redundant_after_coverage = True
    finalize_after_public_coverage = True

    def __init__(
        self,
        *,
        client: OpenAIJsonClient | None = None,
        settings: OpenAICompatibleSettings | None = None,
        capability_registry: CapabilityRegistry | None = None,
        run_base_dir: Path = Path("outputs/evaluation/react_tool_full_runs"),
        max_steps: int = 15,
        tool_ids: tuple[str, ...] = DEFAULT_REACT_FULL_TOOL_IDS,
        enable_amesp: bool = False,
        decision_timeout_seconds: float = 20.0,
        finalization_timeout_seconds: float = 60.0,
    ) -> None:
        super().__init__(
            client=client,
            settings=settings,
            capability_registry=capability_registry,
            run_base_dir=run_base_dir,
            max_steps=max_steps,
            tool_ids=tool_ids,
            enable_amesp=enable_amesp,
            decision_timeout_seconds=decision_timeout_seconds,
            finalization_timeout_seconds=finalization_timeout_seconds,
        )

    def _fallback_final_answer(
        self,
        transcript: list[str],
        *,
        error: Exception | None,
    ) -> str:
        text = "\n".join(transcript)
        covered: list[str] = []
        if "macro.macro_structure_scan" in text:
            covered.append(
                "The broad structure scan reported scaffold, aromaticity, heteroatom, "
                "conjugation, and flexibility proxy descriptors."
            )
        if "macro.screen_donor_acceptor_architecture" in text:
            covered.append(
                "The donor-acceptor architecture screen reported a bounded ICT/D-A "
                "structural proxy."
            )
        if "macro.screen_rotor_rim_prior" in text:
            covered.append(
                "The rotor/RIM screen reported rotatable-bond or torsion-topology proxy "
                "features."
            )
        if "macro.screen_esipt_structural_motif" in text:
            covered.append("The ESIPT screen reported whether a proton-transfer motif is present.")
        if "macro.screen_aggregation_prone_scaffold" in text or "pi_stacking" in text:
            covered.append(
                "Aggregation and pi-stacking proxy screens reported scaffold-level packing "
                "or contact tendencies."
            )
        if "macro.run_solid_state_emission_proxy" in text:
            covered.append("The solid-state emission proxy screen reported a checklist boundary.")
        if "macro.run_crystal_restriction_checklist" in text:
            covered.append(
                "The crystal restriction checklist identified crystal/wet-lab boundaries."
            )
        if "macro.run_water_fraction_aggregation_checklist" in text:
            covered.append(
                "The water-fraction aggregation checklist identified aggregation experiment "
                "boundaries."
            )
        if "microscopic.unsupported_excited_state_relaxation" in text:
            covered.append(
                "The excited-state relaxation route reported that CI, trajectory, and "
                "nonadiabatic dynamics deliverables are unavailable in this baseline scope."
            )
        if not covered:
            covered.append(
                "Only structure preparation was completed; no mechanism-specific tool evidence "
                "was available."
            )
        diagnoses = self._fallback_mechanism_diagnoses(text)
        reason = f" Fallback reason: {error}" if error is not None else ""
        return (
            "Full ReAct Final Answer. Evidence summary: "
            + " ".join(covered)
            + " Mechanism diagnosis: "
            + " ".join(diagnoses)
            + " Scope limits: these are SMILES/tool-derived proxy observations, not hidden "
            "benchmark facts, paper-specific wet-lab results, DLS/microscopy data, crystal "
            "structures, or high-level excited-state computations unless explicitly reported "
            "by a tool observation."
            + reason
        )

    def _fallback_mechanism_diagnoses(self, transcript_text: str) -> list[str]:
        sulfur_scaffold_obs = _observation_reference(
            transcript_text,
            "macro.screen_aggregation_prone_scaffold",
            ("Sulfur-containing aromatic topology was detected",),
        )
        rim_obs = _observation_reference(
            transcript_text,
            "macro.screen_rotor_rim_prior",
            ("metrics.rim_prior: True", "metrics.rotatable_bond_count: 0"),
        )
        if "metrics.rim_prior: True" in rim_obs:
            scaffold_clause = (
                f" {sulfur_scaffold_obs} anchors this rotor prior to a "
                "sulfur-containing aromatic scaffold."
                if "Sulfur-containing aromatic topology" in sulfur_scaffold_obs
                else ""
            )
            rim = (
                f"RIM/RIR is proxy-supported: {rim_obs} and the rotor/torsion screen "
                "therefore supports a structure-level restriction-of-motion prior."
                f"{scaffold_clause}"
            )
        elif "metrics.rotatable_bond_count: 0" in rim_obs:
            rim = (
                f"RIM/RIR is weakened: {rim_obs}, so the current transcript gives little "
                "rotor-based RIM support."
            )
        else:
            rim = (
                f"RIM/RIR is underdetermined: {rim_obs} and no direct solid-state "
                "restriction measurement is available."
            )

        esipt_obs = _observation_reference(
            transcript_text,
            "macro.screen_esipt_structural_motif",
            ("metrics.esipt_motif_proxy: True", "metrics.esipt_motif_proxy: False"),
        )
        if "metrics.esipt_motif_proxy: True" in esipt_obs:
            esipt = (
                f"ESIPT is proxy-supported: {esipt_obs}, but this remains a "
                "SMILES-level proton-transfer motif proxy."
            )
        elif "metrics.esipt_motif_proxy: False" in esipt_obs:
            esipt = (
                f"ESIPT is weakened: {esipt_obs}, so the transcript weakens an "
                "ESIPT-first diagnosis."
            )
        else:
            esipt = (
                f"ESIPT is underdetermined: {esipt_obs} and the transcript lacks a "
                "clear proton-transfer motif observation."
            )

        ict_obs = _observation_reference(
            transcript_text,
            "macro.screen_donor_acceptor_architecture",
            (
                "metrics.architecture_proxy_status: donor_acceptor_proxy_present",
                "metrics.donor_acceptor_proxy: 0",
            ),
        )
        if "donor_acceptor_proxy_present" in ict_obs:
            ict = (
                f"ICT / D-A / TICT is proxy-supported: {ict_obs}, which supports only "
                "a structural charge-transfer tendency."
            )
        elif "metrics.donor_acceptor_proxy: 0" in ict_obs:
            ict = (
                f"ICT / D-A / TICT is weakened: {ict_obs}, so the transcript does not "
                "support a strong donor-acceptor prior."
            )
        else:
            ict = (
                f"ICT / D-A / TICT is underdetermined: {ict_obs} and no excited-state "
                "charge-localization calculation was produced."
            )

        packing_obs = _observation_reference(
            transcript_text,
            "macro.screen_pi_stacking_prone_geometry",
            (
                "metrics.pi_stacking_prone_proxy: True",
                "metrics.pi_stacking_prone_proxy: False",
            ),
        )
        aggregation_obs = _observation_reference(
            transcript_text,
            "macro.screen_aggregation_prone_scaffold",
            (
                "Sulfur-containing aromatic topology was detected",
                "metrics.aggregation_prone_proxy: True",
                "metrics.aggregation_prone_proxy: False",
            ),
        )
        if "Sulfur-containing aromatic topology" in aggregation_obs:
            aggregation = (
                "Aggregation / packing / pi-stacking is proxy-supported: "
                f"{aggregation_obs}; {packing_obs} gives a sulfur-containing aromatic "
                "scaffold and packing proxy rather than direct aggregate PL evidence."
            )
        elif "metrics.pi_stacking_prone_proxy: True" in packing_obs:
            aggregation = (
                "Aggregation / packing / pi-stacking is proxy-supported: "
                f"{packing_obs}; {aggregation_obs} gives a bounded aggregation-scaffold "
                "proxy rather than direct aggregate PL evidence."
            )
        elif "metrics.aggregation_prone_proxy: True" in aggregation_obs:
            aggregation = (
                "Aggregation / packing / pi-stacking is proxy-supported: "
                f"{aggregation_obs}; {packing_obs} remains a structural packing proxy."
            )
        elif "False" in packing_obs or "False" in aggregation_obs:
            aggregation = (
                "Aggregation / packing / pi-stacking is weakened: "
                f"{packing_obs}; {aggregation_obs}, so direct aggregation support is absent."
            )
        else:
            aggregation = (
                "Aggregation / packing / pi-stacking is underdetermined: "
                f"{packing_obs}; {aggregation_obs}, with no DLS or aggregate PL series."
            )

        solid_obs = _observation_reference(
            transcript_text,
            "macro.run_solid_state_emission_proxy",
            ("metrics.solid_state_emission_claim_status: proxy_only",),
        )
        if solid_obs.startswith("no transcript observation"):
            solid = (
                f"Solid-state restriction is underdetermined: {solid_obs}, so the "
                "transcript lacks a solid-state restriction proxy or measurement."
            )
        else:
            solid = (
                f"Solid-state restriction is boundary-only: {solid_obs}; the transcript "
                "keeps solid-state emission at proxy/checklist level."
            )

        wet_lab_obs = _observation_reference(
            transcript_text,
            "macro.run_water_fraction_aggregation_checklist",
            ("metrics.required_evidence: water_fraction_PL_series",),
        )
        if wet_lab_obs.startswith("no transcript observation"):
            wet_lab = (
                f"Wet-lab boundary is underdetermined: {wet_lab_obs}, so no wet-lab "
                "checklist observation was produced in the current transcript."
            )
        else:
            wet_lab = (
                f"Wet-lab boundary is boundary-only: {wet_lab_obs}, so water-fraction PL, "
                "DLS, and scattering controls are required rather than observed."
            )

        compute_obs = _observation_reference(
            transcript_text,
            "microscopic.unsupported_excited_state_relaxation",
            ("failure_kind: capability_unsupported", "runtime_status: unsupported"),
        )
        if compute_obs.startswith("no transcript observation"):
            compute = (
                "High-level excited-state computation / nonadiabatic dynamics boundary is "
                f"underdetermined: {compute_obs}, so the transcript does not contain a "
                "computed or failed high-level excited-state route."
            )
        else:
            compute = (
                "High-level excited-state computation / nonadiabatic dynamics boundary is "
                f"boundary-only: {compute_obs}, so CI, trajectory surface hopping, and "
                "nonadiabatic dynamics are not computed."
            )

        return [rim, esipt, ict, aggregation, solid, wet_lab, compute]


class EvidenceFormattedReactToolLLMFullBaselineSubject(ReactToolLLMFullBaselineSubject):
    subject_id = "evidence_formatted_react_tool_llm_full"
    subject_kind = "evidence_formatted_react_tool_llm_full"
    prompt_name = REACT_TOOL_LLM_FULL_EVIDENCE_FORMATTED_PROMPT
    prompt_version = REACT_TOOL_LLM_FULL_EVIDENCE_FORMATTED_PROMPT_VERSION
    default_finalization_label = "evidence-formatted full ReAct baseline"
    finalize_after_public_coverage = False

    def _fallback_mechanism_diagnoses(self, transcript_text: str) -> list[str]:
        return [
            _format_fwb_diagnosis(item)
            for item in super()._fallback_mechanism_diagnoses(transcript_text)
        ]


def _has_tool_observation(transcript: list[str]) -> bool:
    return any(
        line.startswith("Observation ") and not line.startswith("Observation 0:")
        for line in transcript
    )


def _compact_tool_card(card: dict[str, object]) -> dict[str, object]:
    return {
        "capability_id": card.get("capability_id"),
        "owner_agent": card.get("owner_agent"),
        "evidence_family": card.get("evidence_family"),
        "route": card.get("route"),
        "description": card.get("description"),
        "required_artifact_kinds": card.get("required_artifact_kinds", []),
        "outputs": card.get("outputs", []),
        "cost": card.get("cost"),
        "failure_modes": card.get("failure_modes", []),
    }


def _flatten_structured_results(payload: dict[str, object]) -> list[str]:
    fields: list[str] = []
    for key, value in payload.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if _is_plain_observation_value(nested_value):
                    fields.append(f"{key}.{nested_key}: {_plain_observation_value(nested_value)}")
            continue
        if _is_plain_observation_value(value):
            fields.append(f"{key}: {_plain_observation_value(value)}")
    return fields


def _is_plain_observation_value(value: object) -> bool:
    if isinstance(value, str | int | float | bool) or value is None:
        return True
    return isinstance(value, list) and all(
        isinstance(item, str | int | float | bool) or item is None for item in value
    )


def _plain_observation_value(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _decision_tool_args(decision: ReactStepDecision) -> dict[str, object]:
    return dict(decision.tool_args or decision.action_input or {})


def _prompt_case_id(case: EvalCase) -> str:
    return normalize_public_case_id(case.public_case_id)


def _called_tool_ids(transcript: list[str]) -> set[str]:
    joined = "\n".join(transcript)
    called: set[str] = set()
    for match in re.finditer(r'"tool_name"\s*:\s*"([^"]+)"', joined):
        called.add(match.group(1))
    for match in re.finditer(r"Action\s+\d+:\s+([A-Za-z0-9_.-]+)", joined):
        called.add(match.group(1))
    for match in re.finditer(r"capability_id:\s+([A-Za-z0-9_.-]+)", joined):
        called.add(match.group(1))
    return called


def _observation_reference(
    transcript_text: str,
    capability_id: str,
    preferred_phrases: tuple[str, ...],
) -> str:
    for match in re.finditer(
        r"(Observation\s+(\d+):\s+.*?)(?=\nThought|\nAction|\nFinal Answer|$)",
        transcript_text,
        flags=re.S,
    ):
        block = match.group(1)
        observation_id = match.group(2)
        if capability_id not in block:
            continue
        for phrase in preferred_phrases:
            if phrase in block:
                return f"Observation {observation_id} reports {phrase}"
        return f"Observation {observation_id} reports capability_id: {capability_id}"
    return f"no transcript observation for {capability_id}"


def _format_fwb_diagnosis(diagnosis: str) -> str:
    match = re.match(
        r"(?P<name>.+?) is "
        r"(?P<status>proxy-supported|weakened|underdetermined|boundary-only): "
        r"(?P<body>.*)",
        diagnosis,
        flags=re.S,
    )
    if match is None:
        return (
            "Mechanism candidate is underdetermined. Finding: "
            f"{diagnosis} Warrant: the transcript states this as a bounded "
            "candidate. Boundary: no stronger support is added beyond the raw "
            "ReAct transcript."
        )
    name = match.group("name").strip()
    status = match.group("status").strip()
    body = match.group("body").strip()
    return (
        f"{name} is {status}. Finding: {body} Warrant: "
        f"{_mechanism_warrant(name)} Boundary: {_mechanism_boundary(name, status)}"
    )


def _mechanism_warrant(name: str) -> str:
    lowered = name.lower()
    if "rim" in lowered or "rir" in lowered:
        return (
            "rotor, torsion, or restricted-motion proxy observations are relevant "
            "to restriction-of-intramolecular-motion candidates."
        )
    if "esipt" in lowered:
        return (
            "a proton donor/acceptor motif observation is mechanistically relevant "
            "to excited-state proton-transfer candidates."
        )
    if "ict" in lowered or "d-a" in lowered or "tict" in lowered:
        return (
            "donor-acceptor structural observations are relevant to charge-transfer "
            "or twisted charge-transfer candidates."
        )
    if "aggregation" in lowered or "packing" in lowered or "pi-stacking" in lowered:
        return (
            "aggregation, pi-stacking, or scaffold packing observations are relevant "
            "to aggregate-state or packing-mediated photophysical candidates."
        )
    if "solid-state" in lowered:
        return (
            "solid-state checklist observations are relevant to packing or matrix "
            "restriction boundaries."
        )
    if "wet-lab" in lowered:
        return (
            "the transcript identifies experiments needed to distinguish structural "
            "proxies from measured aggregation or emission behavior."
        )
    if "excited-state" in lowered or "nonadiabatic" in lowered:
        return (
            "the transcript identifies high-level excited-state routes needed to test "
            "CI, state-ordering, or nonadiabatic hypotheses."
        )
    return "the observation is relevant only at the mechanism-candidate level."


def _mechanism_boundary(name: str, status: str) -> str:
    lowered = name.lower()
    if status == "weakened":
        return "this is counterevidence or missing-feature evidence, not a confirmed absence."
    if status == "underdetermined":
        return "the current transcript lacks direct case-specific validation."
    if status == "boundary-only":
        return (
            "the transcript reports a validation boundary rather than a positive "
            "mechanism result."
        )
    if "rim" in lowered or "packing" in lowered or "aggregation" in lowered:
        return (
            "SMILES/tool proxy evidence does not replace aggregate PL, crystal "
            "packing, DLS, microscopy, pressure, or viscosity validation."
        )
    if "ict" in lowered or "d-a" in lowered or "tict" in lowered:
        return (
            "SMILES/tool proxy evidence does not establish excited-state charge "
            "transfer without NTO, solvatochromism, or related validation."
        )
    if "esipt" in lowered:
        return (
            "SMILES/tool proxy evidence does not establish proton transfer without "
            "hydrogen-bond geometry, spectroscopy, isotope, or excited-state PES "
            "validation."
        )
    return "no direct wet-lab or high-level excited-state validation is added."


@contextmanager
def _react_decision_deadline(seconds: float) -> Iterator[None]:
    def _raise_timeout(signum, frame) -> None:  # noqa: ANN001
        del signum, frame
        raise TimeoutError(f"ReAct LLM decision exceeded {seconds:.1f} seconds.")

    try:
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, seconds)
    except ValueError:
        yield
        return
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])
