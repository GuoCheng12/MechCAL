from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mechcal.compat import write_legacy_report
from mechcal.runtime import MechCALOrchestrator, OrchestratorConfig, mcp_deps
from mechcal.schemas import CaseRun
from mechcal.tools import StdioMcpTransport

BenchmarkBackend = Literal["local", "mcp-stdio"]

SMILES_COLUMNS = ("smiles", "SMILES", "canonical_smiles", "CanonicalSMILES")
CASE_ID_COLUMNS = ("case_id", "id", "name", "molecule_id")
QUERY_COLUMNS = ("user_query", "query", "prompt")
DEFAULT_QUERY = "Assess the likely AIE mechanism for this molecule."


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    smiles: str
    user_query: str = DEFAULT_QUERY


@dataclass(frozen=True)
class BenchmarkResult:
    summary_path: Path
    jsonl_path: Path
    case_count: int
    finalized_count: int


def build_evidence_alignment_payload(
    case_run: CaseRun,
    hidden_reference: dict[str, object],
) -> dict[str, object]:
    return {
        "case_id": case_run.case_id,
        "payload_type": "evidence_alignment",
        "evidence_units": [
            item.model_dump(mode="json") for item in case_run.evidence_ledger.items
        ],
        "semantic_evidence_targets": _semantic_targets(
            hidden_reference,
            "semantic_evidence_targets",
        ),
    }


def build_diagnosis_alignment_payload(
    case_run: CaseRun,
    hidden_reference: dict[str, object],
) -> dict[str, object]:
    diagnosis_units = (
        case_run.diagnosis_units
        or (case_run.final_answer.diagnosis_units if case_run.final_answer else [])
    )
    return {
        "case_id": case_run.case_id,
        "payload_type": "diagnosis_alignment",
        "diagnosis_units": [item.model_dump(mode="json") for item in diagnosis_units],
        "semantic_diagnosis_targets": _semantic_targets(
            hidden_reference,
            "semantic_diagnosis_targets",
        ),
    }


def load_benchmark_cases(dataset_path: Path, *, limit: int | None = None) -> list[BenchmarkCase]:
    with dataset_path.expanduser().resolve().open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected_rows = rows[:limit] if limit is not None else rows
    return [_case_from_row(row, index) for index, row in enumerate(selected_rows, start=1)]


def run_benchmark(
    *,
    dataset_path: Path,
    output_dir: Path,
    backend: BenchmarkBackend = "local",
    limit: int | None = None,
    max_rounds: int = 4,
    export_legacy: bool = True,
) -> BenchmarkResult:
    cases = load_benchmark_cases(dataset_path, limit=limit)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = OrchestratorConfig(run_base_dir=output_dir, max_rounds=max_rounds)
    deps_by_backend = {
        "local": lambda: None,
        "mcp-stdio": lambda: mcp_deps(config, StdioMcpTransport()),
    }
    orchestrator = MechCALOrchestrator(config, deps=deps_by_backend[backend]())
    summaries = []

    for case in cases:
        result = orchestrator.run(
            smiles=case.smiles,
            user_query=case.user_query,
            case_id=case.case_id,
        )
        run_dir = output_dir / result.case_id
        if export_legacy:
            write_legacy_report(result, run_dir / "legacy_report.json")
            write_legacy_report(result, run_dir / "legacy_report.md")
        summaries.append(
            {
                "case_id": result.case_id,
                "smiles": result.input.smiles,
                "status": result.status,
                "run_dir": str(run_dir),
                "round_count": len(result.round_ids),
                "evidence_families": result.evidence_ledger.covered_families(),
                "final_confidence": (
                    result.final_answer.confidence if result.final_answer else None
                ),
            }
        )

    jsonl_path = output_dir / "benchmark_summary.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in summaries:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = output_dir / "benchmark_summary.json"
    aggregate = {
        "dataset_path": str(dataset_path.expanduser().resolve()),
        "backend": backend,
        "case_count": len(summaries),
        "finalized_count": sum(1 for row in summaries if row["status"] == "finalized"),
        "cases": summaries,
    }
    summary_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n")
    return BenchmarkResult(
        summary_path=summary_path,
        jsonl_path=jsonl_path,
        case_count=aggregate["case_count"],
        finalized_count=aggregate["finalized_count"],
    )


def _case_from_row(row: dict[str, str], index: int) -> BenchmarkCase:
    smiles = _first_value(row, SMILES_COLUMNS)
    if smiles is None:
        raise ValueError(f"Dataset row {index} does not contain a SMILES column.")
    return BenchmarkCase(
        case_id=_first_value(row, CASE_ID_COLUMNS) or f"case_{index:04d}",
        smiles=smiles,
        user_query=_first_value(row, QUERY_COLUMNS) or DEFAULT_QUERY,
    )


def _first_value(row: dict[str, str], columns: tuple[str, ...]) -> str | None:
    values = [row[column].strip() for column in columns if column in row and row[column].strip()]
    return values[0] if values else None


def _semantic_targets(
    hidden_reference: dict[str, object],
    key: str,
) -> list[dict[str, object]]:
    raw = hidden_reference.get(key)
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]
