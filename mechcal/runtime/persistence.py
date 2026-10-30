from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from mechcal.capabilities import CapabilityCatalog
from mechcal.runtime.progress import atomic_json, write_activity
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
from mechcal.schemas.decisions import DispatchRequest


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

    def activity(self, stage: str, round_id: str | None = None, detail: str = "") -> None:
        write_activity(self.root, stage, round_id, detail)

    def model_events(self, stage: str, **context):
        """Per-request audit sink; no stateful client/global logging configuration."""
        attempt = 0

        def record(event):
            nonlocal attempt
            attempt = event.get("attempt", attempt)
            self.append_jsonl("history/component_requests.jsonl", {
                **context, "stage": stage, "timestamp": time.time(), "attempt": attempt, **event,
            })
        return record

    def write_json(self, relative_path: str, payload: BaseModel | dict[str, Any]) -> Path:
        path = self.root / relative_path
        atomic_json(path, _jsonable(payload))
        return path

    def append_jsonl(self, relative_path: str, payload: BaseModel | dict[str, Any]) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        # A previous interrupted append may lack its final newline. Keep those
        # bytes for diagnosis, but do not concatenate the next complete record.
        with path.open("ab+") as handle:
            handle.seek(0, 2)
            if handle.tell():
                handle.seek(-1, 2)
                if handle.read(1) != b"\n":
                    handle.write(b"\n")
            handle.write((json.dumps(_jsonable(payload), ensure_ascii=False) + "\n").encode("utf-8"))
        return path

    def execution_checkpoint_path(self, request: DispatchRequest) -> Path:
        token = hashlib.sha256(json.dumps([request.round_id, request.dispatch_id]).encode()).hexdigest()[:24]
        return self.root / "executions" / f"{token}.json"

    def read_execution_checkpoint(self, request: DispatchRequest) -> dict[str, Any] | None:
        path = self.execution_checkpoint_path(request)
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        if record.get("case_id") != self.case_id or record.get("dispatch") != request.model_dump(mode="json"):
            raise ValueError("Execution checkpoint does not match case and saved request")
        if record.get("stage") not in {"planning", "planned", "executing", "tool_returned", "reported"}:
            raise ValueError("Unknown execution checkpoint stage")
        return record

    def write_execution_checkpoint(
        self, request: DispatchRequest, *, stage: str, plan: AgentExecutionPlan | None = None,
        result: ToolExecutionResult | None = None, report: AgentReport | None = None,
    ) -> None:
        atomic_json(self.execution_checkpoint_path(request), {
            "case_id": self.case_id, "dispatch": request.model_dump(mode="json"), "stage": stage,
            "plan": plan.model_dump(mode="json") if plan is not None else None,
            "tool_result": result.model_dump(mode="json") if result is not None else None,
            "report": report.model_dump(mode="json") if report is not None else None,
        })

    def write_case_input(self, case_input: CaseInput) -> Path:
        return self.write_json("input.json", case_input)

    def write_tool_discovery_event(self, event: dict[str, Any]) -> None:
        self.append_jsonl("history/tool_discovery.jsonl", event)
        self.write_json("tool_discovery.json", event["state"])
        querying = event.get("kind") == "tool_description_query" or (
            event.get("kind") == "tool_discovery_waiting" and event["batches_used"] > 0
            and event.get("planner_stage") not in {"planner_stop_review_response", "planner_final_synthesis"}
        )
        planner_stage = {
            "planner_stop_review_response": "Planner response to stop review",
            "planner_final_synthesis": "Planner final synthesis",
            "planner_question_act": "Planner choose action",
        }.get(event.get("planner_stage"), "Planner")
        self.activity(
            "Query tool descriptions" if querying else planner_stage,
            event.get("round_id"),
            str(event.get("error") or ", ".join(event.get("returned_ids", [])) or
                "Documentation loaded; awaiting Planner decision") if querying else "",
        )

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
        # Route/agent filenames below are latest-result aliases for older readers.
        # Keep a separate record for every execution, including repeated routes.
        execution_id = hashlib.sha256(json.dumps(
            [result.dispatch_id, result.tool_result_id],
        ).encode()).hexdigest()[:16]
        self.write_json(
            f"rounds/{result.round_id}/execution_{execution_id}_tool_result.json", result,
        )
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
