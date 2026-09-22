from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from mechcal.schemas import CaseRun, FinalAnswer

LegacyReportFormat = Literal["json", "markdown"]


def load_case_run(run_dir: Path) -> CaseRun:
    return CaseRun.model_validate(
        json.loads((run_dir.expanduser().resolve() / "case_run.json").read_text(encoding="utf-8"))
    )


def legacy_payload_from_case_run(case_run: CaseRun) -> dict[str, Any]:
    final = case_run.final_answer or _fallback_final(case_run)
    status_counts = _count_by_status(case_run)
    return {
        "schema_version": "legacy_report_compat_v1",
        "source_schema_version": case_run.schema_version,
        "case_id": case_run.case_id,
        "status": case_run.status,
        "smiles": case_run.input.smiles,
        "query": case_run.input.user_query,
        "current_hypothesis": final.current_hypothesis,
        "confidence": final.confidence,
        "answer": final.answer,
        "evidence_families": case_run.evidence_ledger.covered_families(),
        "coverage_debt_hypotheses": (
            case_run.claim_ledger.coverage_debt_hypotheses()
            if case_run.claim_ledger is not None
            else []
        ),
        "evidence_refs": final.evidence_refs,
        "round_refs": final.round_refs,
        "artifact_refs": final.artifact_refs,
        "limitations": final.limitations,
        "portfolio": [
            item.model_dump(mode="json") for item in case_run.portfolio.sorted_hypotheses()
        ],
        "evidence": [item.model_dump(mode="json") for item in case_run.evidence_ledger.items],
        "claims": (
            [item.model_dump(mode="json") for item in case_run.claim_ledger.assessments]
            if case_run.claim_ledger is not None
            else []
        ),
        "artifacts": [
            item.model_dump(mode="json") for item in case_run.artifact_manifest.artifacts
        ],
        "status_counts": status_counts,
        "runtime": case_run.runtime,
    }


def legacy_markdown_from_case_run(case_run: CaseRun) -> str:
    payload = legacy_payload_from_case_run(case_run)
    lines = [
        f"# AIE-MAS Report: {payload['case_id']}",
        "",
        f"- Status: {payload['status']}",
        f"- SMILES: `{payload['smiles']}`",
        f"- Current hypothesis: {payload['current_hypothesis']}",
        f"- Confidence: {payload['confidence']:.2f}",
        f"- Evidence families: {', '.join(payload['evidence_families']) or 'none'}",
        f"- Coverage debt hypotheses: {', '.join(payload['coverage_debt_hypotheses']) or 'none'}",
        "",
        "## Answer",
        "",
        str(payload["answer"]),
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["limitations"] or ["None recorded."])
    lines.extend(["", "## Evidence"])
    lines.extend(
        f"- `{item['evidence_id']}` [{item['family']}/{item['status']}]: {item['summary']}"
        for item in payload["evidence"]
    )
    lines.extend(["", "## Claims"])
    lines.extend(
        f"- `{item['claim_id']}` [{item['coverage_status']}]: {item['summary']}"
        for item in payload["claims"]
    )
    lines.extend(["", "## Artifacts"])
    lines.extend(
        f"- `{item['artifact_id']}` [{item['kind']}/{item['status']}]"
        for item in payload["artifacts"]
    )
    return "\n".join(lines) + "\n"


def write_legacy_report(
    case_run: CaseRun,
    output_path: Path,
    *,
    report_format: LegacyReportFormat | None = None,
) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_format = report_format or _format_from_suffix(output_path)
    content_by_format = {
        "json": lambda: json.dumps(
            legacy_payload_from_case_run(case_run),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        "markdown": lambda: legacy_markdown_from_case_run(case_run),
    }
    output_path.write_text(content_by_format[selected_format](), encoding="utf-8")
    return output_path


def _fallback_final(case_run: CaseRun) -> FinalAnswer:
    return FinalAnswer(
        case_id=case_run.case_id,
        current_hypothesis=case_run.portfolio.current,
        confidence=0.0,
        answer="No final answer was recorded.",
        evidence_refs=[item.evidence_id for item in case_run.evidence_ledger.items],
        round_refs=list(case_run.round_ids),
        artifact_refs=case_run.artifact_manifest.artifact_ids(),
        limitations=["Compatibility export used a fallback final answer."],
    )


def _count_by_status(case_run: CaseRun) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in case_run.evidence_ledger.items:
        counts[item.status] = counts.get(item.status, 0) + 1
    return counts


def _format_from_suffix(path: Path) -> LegacyReportFormat:
    suffix_map: dict[str, LegacyReportFormat] = {
        ".md": "markdown",
        ".markdown": "markdown",
    }
    return suffix_map.get(path.suffix.lower(), "json")
