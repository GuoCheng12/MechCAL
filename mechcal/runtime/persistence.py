from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from mechcal.capabilities import CapabilityCatalog
from mechcal.schemas import (
    AgentExecutionPlan,
    AgentReport,
    ArtifactManifest,
    CaseInput,
    CaseRun,
    ClaimLedger,
    FinalAnswer,
    MechanismPrediction,
    MechanismPriorityDelta,
    MechanismSupportArgument,
    RoundRecord,
    ToolExecutionResult,
)
from mechcal.schemas.evidence import EvidenceLedger, EvidenceUnit
from mechcal.schemas.mechanisms import (
    ConclusionLedger,
    DiagnosisUnit,
    DifferentialMechanismPortfolio,
    MechanismAgendaCoverage,
    MechanismProgram,
    PhotophysicsReview,
)


def _jsonable(payload: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json")
    return payload


class RunStore:
    def __init__(self, base_dir: Path, case_id: str) -> None:
        self.root = base_dir.expanduser().resolve() / case_id
        self.case_id = case_id

    def initialize(self) -> None:
        for relative in ("history", "artifacts", "claims", "rounds"):
            (self.root / relative).mkdir(parents=True, exist_ok=True)

    def write_json(self, relative_path: str, payload: BaseModel | dict[str, Any]) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def append_jsonl(self, relative_path: str, payload: BaseModel | dict[str, Any]) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(payload), ensure_ascii=False) + "\n")
        return path

    def write_case_input(self, case_input: CaseInput) -> Path:
        return self.write_json("input.json", case_input)

    def write_capability_catalog(self, catalog: CapabilityCatalog) -> Path:
        return self.write_json("capabilities.json", catalog)

    def write_case_run(self, case_run: CaseRun) -> Path:
        self.write_json("artifacts/manifest.json", case_run.artifact_manifest)
        self.write_evidence_ledger(case_run.evidence_ledger)
        if case_run.mechanism_program is not None:
            self.write_mechanism_program(case_run.mechanism_program)
        if case_run.mechanism_agenda_coverage is not None:
            self.write_mechanism_agenda_coverage(case_run.mechanism_agenda_coverage)
        if case_run.photophysics_review is not None:
            self.write_photophysics_review(case_run.photophysics_review)
        if case_run.conclusion_ledger is not None:
            self.write_conclusion_ledger(case_run.conclusion_ledger)
        if case_run.differential_mechanism_portfolio is not None:
            self.write_differential_mechanism_portfolio(
                case_run.differential_mechanism_portfolio
            )
        if case_run.mechanism_priority_deltas:
            self.write_mechanism_priority_deltas(
                case_run.mechanism_priority_deltas,
                round_id=case_run.current_round_id,
            )
        if case_run.mechanism_support_arguments:
            self.write_mechanism_support_arguments(
                case_run.mechanism_support_arguments,
                round_id=case_run.current_round_id,
            )
        if case_run.mechanism_predictions:
            self.write_mechanism_predictions(case_run.mechanism_predictions)
        if case_run.diagnosis_units:
            self.write_diagnosis_units(case_run.diagnosis_units)
        if case_run.claim_ledger is not None:
            self.write_claim_ledger(case_run.claim_ledger)
        return self.write_json("case_run.json", case_run)

    def write_artifact_manifest(self, manifest: ArtifactManifest) -> Path:
        return self.write_json("artifacts/manifest.json", manifest)

    def write_claim_ledger(self, ledger: ClaimLedger) -> Path:
        return self.write_json("claims/claim_ledger.json", ledger)

    def write_evidence_ledger(self, ledger: EvidenceLedger) -> Path:
        return self.write_json("evidence_units.json", ledger)

    def write_round_evidence_units(
        self,
        round_id: str,
        agent_name: str,
        evidence_units: list[EvidenceUnit],
        route: str | None = None,
    ) -> Path:
        payload = {
            "case_id": self.case_id,
            "round_id": round_id,
            "agent_name": agent_name,
            "evidence_units": [item.model_dump(mode="json") for item in evidence_units],
        }
        if route:
            self.write_json(
                f"rounds/{round_id}/{agent_name}_{_path_token(route)}_evidence_units.json",
                payload,
            )
        return self.write_json(f"rounds/{round_id}/{agent_name}_evidence_units.json", payload)

    def write_mechanism_program(self, program: MechanismProgram) -> Path:
        self.write_json(f"rounds/{program.round_id}/mechanism_program.json", program)
        return self.write_json("mechanism_program.json", program)

    def write_mechanism_agenda_coverage(
        self,
        coverage: MechanismAgendaCoverage,
    ) -> Path:
        self.write_json(
            f"rounds/{coverage.round_id}/mechanism_agenda_coverage.json",
            coverage,
        )
        return self.write_json("mechanism_agenda_coverage.json", coverage)

    def write_diagnosis_units(self, diagnosis_units: list[DiagnosisUnit]) -> Path:
        payload = {
            "case_id": self.case_id,
            "diagnosis_units": [item.model_dump(mode="json") for item in diagnosis_units],
        }
        return self.write_json("diagnosis_units.json", payload)

    def write_mechanism_predictions(
        self,
        predictions: list[MechanismPrediction],
    ) -> Path:
        payload = {
            "case_id": self.case_id,
            "mechanism_predictions": [
                item.model_dump(mode="json") for item in predictions
            ],
        }
        return self.write_json("mechanism_predictions.json", payload)

    def write_mechanism_support_arguments(
        self,
        arguments: list[MechanismSupportArgument],
        *,
        round_id: str | None = None,
    ) -> Path:
        payload = {
            "case_id": self.case_id,
            "mechanism_support_arguments": [
                item.model_dump(mode="json") for item in arguments
            ],
        }
        if round_id:
            round_arguments = [
                item
                for item in arguments
                if item.support_id.startswith(f"{round_id}:")
            ]
            self.write_json(
                f"rounds/{round_id}/mechanism_support_arguments.json",
                {
                    "case_id": self.case_id,
                    "round_id": round_id,
                    "mechanism_support_arguments": [
                        item.model_dump(mode="json") for item in round_arguments
                    ],
                },
            )
        return self.write_json("mechanism_support_arguments.json", payload)

    def write_mechanism_priority_deltas(
        self,
        deltas: list[MechanismPriorityDelta],
        *,
        round_id: str | None = None,
    ) -> Path:
        payload = {
            "case_id": self.case_id,
            "mechanism_priority_deltas": [
                item.model_dump(mode="json") for item in deltas
            ],
        }
        if round_id:
            round_deltas = [
                item
                for item in deltas
                if item.round_id == round_id
            ]
            self.write_json(
                f"rounds/{round_id}/mechanism_priority_deltas.json",
                {
                    "case_id": self.case_id,
                    "round_id": round_id,
                    "mechanism_priority_deltas": [
                        item.model_dump(mode="json") for item in round_deltas
                    ],
                },
            )
        return self.write_json("mechanism_priority_deltas.json", payload)

    def write_photophysics_review(self, review: PhotophysicsReview) -> Path:
        self.write_json(f"rounds/{review.round_id}/photophysics_review.json", review)
        self.write_json(
            f"rounds/{review.round_id}/support_audit.json",
            {
                "case_id": self.case_id,
                "round_id": review.round_id,
                "support_argument_audits": [
                    item.model_dump(mode="json")
                    for item in review.support_argument_audits
                ],
            },
        )
        return self.write_json("photophysics_review.json", review)

    def write_conclusion_ledger(self, ledger: ConclusionLedger) -> Path:
        self.write_json(f"rounds/{ledger.round_id}/conclusion_ledger.json", ledger)
        return self.write_json("conclusion_ledger.json", ledger)

    def write_differential_mechanism_portfolio(
        self,
        portfolio: DifferentialMechanismPortfolio,
    ) -> Path:
        self.write_json(
            f"rounds/{portfolio.round_id}/differential_mechanism_portfolio.json",
            portfolio,
        )
        return self.write_json("differential_mechanism_portfolio.json", portfolio)

    def write_round_record(self, record: RoundRecord) -> Path:
        return self.write_json(f"rounds/{record.round_id}/round.json", record)

    def write_planner_decision(self, record: RoundRecord) -> Path | None:
        if record.planner_decision is None:
            return None
        return self.write_json(
            f"rounds/{record.round_id}/planner_decision.json",
            record.planner_decision,
        )

    def write_agent_report(self, report: AgentReport) -> Path:
        route = _report_route(report)
        if route:
            self.write_json(
                f"rounds/{report.round_id}/{report.agent_name}_{_path_token(route)}_report.json",
                report,
            )
        return self.write_json(
            f"rounds/{report.round_id}/{report.agent_name}_report.json",
            report,
        )

    def write_execution_plan(self, plan: AgentExecutionPlan) -> Path:
        self.write_json(
            (
                f"rounds/{plan.round_id}/{plan.agent_name}_"
                f"{_path_token(plan.selected_route)}_execution_plan.json"
            ),
            plan,
        )
        return self.write_json(
            f"rounds/{plan.round_id}/{plan.agent_name}_execution_plan.json",
            plan,
        )

    def write_tool_result(self, result: ToolExecutionResult) -> Path:
        self.write_json(
            (
                f"rounds/{result.round_id}/{result.agent_name}_"
                f"{_path_token(result.selected_route)}_tool_result.json"
            ),
            result,
        )
        return self.write_json(
            f"rounds/{result.round_id}/{result.agent_name}_tool_result.json",
            result,
        )

    def write_final_answer(self, final_answer: FinalAnswer) -> Path:
        if final_answer.diagnosis_units:
            self.write_diagnosis_units(final_answer.diagnosis_units)
        if final_answer.mechanism_predictions:
            self.write_mechanism_predictions(final_answer.mechanism_predictions)
        return self.write_json("final.json", final_answer)


def _report_route(report: AgentReport) -> str | None:
    if report.operational_note is not None:
        return report.operational_note.selected_route
    for item in report.evidence_units:
        route = item.capability_id.split(".", maxsplit=1)[-1]
        if route:
            return route
    return None


def _path_token(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in value
    )
