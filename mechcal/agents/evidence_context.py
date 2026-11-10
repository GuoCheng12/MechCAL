"""Public evidence context shared by the Planner and its reviewers."""

from __future__ import annotations

from math import isclose

from mechcal.chem.solvent_comparisons import solvent_comparisons
from mechcal.schemas import CaseRun, EvidenceUnit, MechanismSupportArgument
from mechcal.schemas.reports import AgentReport


def source_checks(case_run: CaseRun, items: list[EvidenceUnit]) -> list[dict]:
    """Expose source status and reproducible arithmetic without grading free text."""
    results = case_run.runtime.get("question_loop", {}).get("results", [])
    checks = []
    for item in items:
        computed = item.metrics.get("computed_observables", {})
        computed = computed if isinstance(computed, dict) else {}
        row = {
            "evidence_id": item.evidence_id, "evidence_status": item.status,
            "executions": [{"result_id": r["result_id"], "status": r["status"],
                            "requested_parameters": r.get("requested_parameters", {})}
                           for r in results if item.evidence_id in r.get("evidence_refs", [])],
            "scope": "Checks on recorded sources, not verification of the Planner's prose or mechanism.",
        }
        records = computed.get("solvent_records")
        if isinstance(records, list) and all(isinstance(r, dict) for r in records):
            comparisons = solvent_comparisons(records)
            for name, comparison in comparisons.items():
                reported = computed.get(name)
                expected = comparison["first_pair_delta"]
                comparison["reported_delta"] = reported
                comparison["reported_delta_check"] = (
                    "not_checkable" if type(reported) not in (int, float) or expected is None
                    else "matches" if isclose(reported, expected, rel_tol=0, abs_tol=1e-6)
                    else "mismatch"
                )
            row["solvent_comparisons"] = comparisons
        checks.append(row)
    return checks


def calculation_record(report: AgentReport) -> dict[str, object]:
    request = report.structured_results.get("dispatch_request", {})
    computed = report.structured_results.get("parsed_observables", {})
    return {
        "report_id": report.report_id,
        "round_id": report.round_id,
        "capability_id": request.get("capability_id"),
        "dispatch_id": request.get("dispatch_id"),
        "objective": request.get("objective"),
        "requested_parameters": request.get("tool_args", {}),
        "input_refs": request.get("input_refs", []),
        "effective_parameters": computed.get("effective_parameters"),
        "sampled_conditions": sampled_conditions(computed),
        "available_descriptors": computed.get("available_descriptors", []),
        "missing_descriptors": computed.get("missing_descriptors", []),
        "status": report.status,
        "failure": report.failure.model_dump(mode="json"),
        "evidence_refs": [item.evidence_id for item in report.evidence_units],
        "artifact_refs": [item.artifact_id for item in report.artifact_updates],
    }


def current_support_arguments(
    case_run: CaseRun, *, limit: int = 6,
) -> list[MechanismSupportArgument]:
    """Keep the latest argument per label, prioritizing current leading claims."""
    if case_run.runtime.get("question_loop"):
        # Each assimilation replaces this list; dropping same-label arguments loses evidence.
        return list(case_run.mechanism_support_arguments)
    latest = {item.target_label: item for item in case_run.mechanism_support_arguments}
    labels = [item.name for item in case_run.portfolio.sorted_hypotheses()]
    labels.extend(item.target_label for item in reversed(case_run.mechanism_support_arguments))
    return [latest[label] for label in dict.fromkeys(labels) if label in latest][:limit]


def relevant_evidence(
    case_run: CaseRun,
    arguments: list[MechanismSupportArgument],
    *,
    limit: int = 12,
) -> list[EvidenceUnit]:
    """The soft limit never drops cited evidence or the latest evidence batch."""
    items = case_run.evidence_ledger.items
    refs = {ref for item in arguments for ref in item.observation_refs}
    scope = case_run.runtime.get("support_audit_scope")
    if case_run.runtime.get("scoped_evidence_updates") and scope:
        refs.update(scope["required_evidence_refs"])
        return [item for item in items if item.evidence_id in refs]
    from mechcal.runtime.evidence_comparison import comparison_refs

    refs.update(comparison_refs(case_run))
    refs.update(
        ref for hypothesis in case_run.portfolio.sorted_hypotheses()[:3]
        for ref in hypothesis.evidence_refs
    )
    latest_round = items[-1].round_id if items else None
    refs.update(item.evidence_id for item in items if item.round_id == latest_round)
    for item in reversed(items):
        if len(refs) >= limit:
            break
        refs.add(item.evidence_id)
    return [item for item in items if item.evidence_id in refs]


def evidence_index(case_run: CaseRun) -> list[dict[str, object]]:
    """Index all rounds without inferring mechanism support from route names."""
    return [
        {
            "evidence_id": item.evidence_id,
            "round_id": item.round_id,
            "capability_id": item.capability_id,
            "status": item.status,
            "claim": item.claim,
            "limits": item.limits,
            "artifact_refs": item.artifact_refs,
        }
        for item in case_run.evidence_ledger.items
    ]


def calculation_history(case_run: CaseRun) -> list[dict[str, object]]:
    """Return recorded conditions; legacy checkpoints expose only known facts."""
    records = list(case_run.runtime.get("calculation_history", []))
    known_refs = {ref for row in records for ref in row.get("evidence_refs", [])}
    for item in case_run.evidence_ledger.items:
        if item.evidence_id in known_refs:
            continue
        computed = item.metrics.get("computed_observables", {})
        computed = computed if isinstance(computed, dict) else {}
        records.append({
            "round_id": item.round_id,
            "capability_id": item.capability_id,
            "status": item.status,
            "requested_parameters": computed.get("requested_parameters"),
            "effective_parameters": computed.get("effective_parameters"),
            "available_descriptors": computed.get("available_descriptors", []),
            "missing_descriptors": computed.get("missing_descriptors", []),
            "sampled_conditions": sampled_conditions(computed),
            "evidence_refs": [item.evidence_id],
            "artifact_refs": item.artifact_refs,
            "record_source": "evidence_ledger; unrecorded parameters remain unknown",
        })
    return records


def sampled_conditions(computed: dict[str, object]) -> list[dict[str, object]]:
    keys = {
        "member_id", "dihedral_id", "dihedral_atoms", "target_angle_deg",
        "conformer_id", "rdkit_conformer_id", "solvent", "state_root",
        "source_artifact_id", "coordinate", "target_value",
    }
    return [
        {key: value for key, value in row.items() if key in keys}
        for row in computed.get("route_records", []) if isinstance(row, dict)
    ]
