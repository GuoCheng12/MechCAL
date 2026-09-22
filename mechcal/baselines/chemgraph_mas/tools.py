from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from mechcal.baselines.chemgraph_mas.budget import SharedBudget
from mechcal.capabilities import CapabilityRegistry, default_capability_registry
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


class AnalysisToolInput(BaseModel):
    capability_id: str = Field(
        description=(
            "Exact allowed capability ID assigned by the Planner, including the "
            "macro. or microscopic. owner prefix."
        )
    )
    tool_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional bounded arguments for the selected capability.",
    )


class ChemGraphToolHarness:
    """Expose MechCAL's tool implementations without its evidence-state workflow."""

    def __init__(
        self,
        *,
        case_id: str,
        public_case_id: str,
        smiles: str,
        task: str,
        workspace: Path,
        budget: SharedBudget,
        enable_amesp: bool,
        capability_registry: CapabilityRegistry | None = None,
    ) -> None:
        self.registry = capability_registry or default_capability_registry()
        self.budget = budget
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._step_index = 0
        self.macro_tool = LocalMacroTool()
        self.microscopic_tool = LocalMicroscopicTool(enable_amesp=enable_amesp)
        self.allowed_ids = {
            owner: tuple(
                str(card["capability_id"])
                for card in self.registry.cards()
                if card["owner_agent"] == owner
            )
            for owner in ("macro", "microscopic")
        }

        case_input = CaseInput(
            case_id=case_id,
            smiles=smiles,
            user_query=task,
            metadata={"public_case_id": public_case_id},
        )
        artifact, prep_failure = LocalStructureTool().prepare(
            case_input,
            round_id="R000",
            workspace=self.workspace,
        )
        self.structure_preparation = {
            "artifact_id": artifact.artifact_id,
            "status": artifact.status,
            "failure": prep_failure.model_dump(mode="json"),
        }
        self.case_run = CaseRun(
            case_id=case_id,
            input=case_input,
            status="running",
            evidence_ledger=EvidenceLedger(case_id=case_id),
            artifact_manifest=ArtifactManifest(case_id=case_id).with_records([artifact]),
            runtime={
                "baseline": "chemgraph_mas_adapted",
                "state_management": "executor_transcript_and_artifacts_only",
            },
        )

    def langchain_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self.run_macro_analysis,
                name="run_macro_analysis",
                description=(
                    "Run one allowed low-cost Macro structural analysis for the verified "
                    "SMILES. Use only capability IDs supplied in the task and preserve "
                    "the full macro. prefix."
                ),
                args_schema=AnalysisToolInput,
            ),
            StructuredTool.from_function(
                func=self.run_microscopic_analysis,
                name="run_microscopic_analysis",
                description=(
                    "Run one allowed Microscopic analysis backed by the same local Amesp "
                    "tool layer as MechCAL. Use only capability IDs supplied in the task "
                    "and preserve the full microscopic. prefix."
                ),
                args_schema=AnalysisToolInput,
            ),
        ]

    def run_macro_analysis(
        self,
        capability_id: str,
        tool_args: dict[str, Any] | None = None,
    ) -> str:
        return self._execute("macro", capability_id, tool_args or {})

    def run_microscopic_analysis(
        self,
        capability_id: str,
        tool_args: dict[str, Any] | None = None,
    ) -> str:
        return self._execute("microscopic", capability_id, tool_args or {})

    def compact_catalog(self) -> str:
        lines = []
        cards = {
            str(card["capability_id"]): card for card in self.registry.cards()
        }
        for owner in ("macro", "microscopic"):
            lines.append(f"{owner.upper()} CAPABILITIES")
            for capability_id in self.allowed_ids[owner]:
                card = cards[capability_id]
                lines.append(
                    f"- {capability_id}: {str(card['description']).strip()}"
                )
        return "\n".join(lines)

    def policy_snapshot(self) -> dict[str, Any]:
        return {
            "exposed_langchain_tools": [
                tool.name for tool in self.langchain_tools()
            ],
            "allowed_capability_ids": {
                owner: list(ids) for owner, ids in self.allowed_ids.items()
            },
            "forbidden_sources": [
                "PubChem",
                "molecule_name_resolution",
                "papers",
                "web",
                "external_databases",
            ],
            "evidence_ledger_shared_with_agents": False,
            "evidence_ledger_unit_count": len(self.case_run.evidence_ledger.items),
            "structure_preparation": self.structure_preparation,
        }

    def _execute(
        self,
        owner: str,
        capability_id: str,
        tool_args: dict[str, Any],
    ) -> str:
        with self._lock:
            if capability_id not in self.allowed_ids[owner]:
                return json.dumps(
                    {
                        "status": "rejected",
                        "failure_kind": "capability_not_allowed",
                        "capability_id": capability_id,
                        "owner": owner,
                    },
                    ensure_ascii=False,
                )
            self.budget.reserve_tool_call(
                owner=owner,
                capability_id=capability_id,
            )
            self._step_index += 1
            capability = self.registry.require(capability_id)
            round_id = f"CG{self._step_index:03d}"
            dispatch_id = f"chemgraph:{self._step_index:03d}:{capability_id}"
            request = DispatchRequest(
                dispatch_id=dispatch_id,
                round_id=round_id,
                agent_name=owner,  # type: ignore[arg-type]
                capability_id=capability_id,
                task=f"ChemGraph executor call: {capability.description}",
                objective="Return bounded observations for final mechanism ranking.",
                evidence_goal_family=capability.evidence_family,
                route=capability.route,
                input_refs=self.case_run.artifact_manifest.artifact_ids(),
                constraints={
                    "baseline": "chemgraph_mas_adapted",
                    "hidden_reference_access": False,
                },
            )
            plan = AgentExecutionPlan(
                plan_id=f"{dispatch_id}:plan",
                round_id=round_id,
                agent_name=owner,  # type: ignore[arg-type]
                dispatch_id=dispatch_id,
                capability_id=capability_id,
                selected_route=capability.route,
                tool_steps=[f"execute {capability.route}"],
                tool_args=tool_args,
                tool_arg_rationale="Arguments supplied by the ChemGraph executor.",
                parameter_adjustment_hint="No MechCAL planner state is available.",
                artifact_refs=self.case_run.artifact_manifest.artifact_ids(),
                planning_mode="llm",
            )
            result = self._run_tool(owner, request, plan)
            if result.artifact_updates:
                self.case_run = self.case_run.touch(
                    artifact_manifest=self.case_run.artifact_manifest.with_records(
                        list(result.artifact_updates)
                    )
                )
            return json.dumps(_compact_tool_result(result), ensure_ascii=False)

    def _run_tool(
        self,
        owner: str,
        request: DispatchRequest,
        plan: AgentExecutionPlan,
    ) -> ToolExecutionResult:
        if owner == "macro":
            return self.macro_tool.run(
                request,
                self.case_run,
                execution_plan=plan,
            )
        return self.microscopic_tool.run(
            request,
            self.case_run,
            execution_plan=plan,
        )


def _compact_tool_result(result: ToolExecutionResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "capability_id": result.capability_id,
        "selected_route": result.selected_route,
        "summary": result.summary,
        "structured_results": result.structured_results,
        "evidence": [
            {
                "claim": item.claim,
                "basis": item.basis,
                "support": item.support,
                "status": item.status,
                "summary": item.summary,
                "limits": item.limits,
                "observable_tags": item.observable_tags,
            }
            for item in result.evidence_units
        ],
        "artifact_ids": [item.artifact_id for item in result.artifact_updates],
        "failure": result.failure.model_dump(mode="json"),
    }
