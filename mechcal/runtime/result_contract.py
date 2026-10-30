"""Execution bookkeeping, not scientific support grading or automatic recovery."""

from copy import deepcopy
from typing import get_args

from mechcal.schemas import CaseRun
from mechcal.schemas.question_loop import ResultDisposition


def _boundary_flag(value: object) -> bool:
    return isinstance(value, dict) and value.get("boundary_only") is True


def disposition_contract(result: dict, case: CaseRun) -> dict:
    units = {e.evidence_id: e for e in case.evidence_ledger.items}
    refs = list(dict.fromkeys(result.get("evidence_refs", [])))
    structured = result.get("structured_results", {})
    boundary_only = _boundary_flag(structured) or _boundary_flag(structured.get("parsed_observables"))
    boundary_refs = {ref for ref in refs if boundary_only or (
        ref in units and _boundary_flag(units[ref].metrics.get("computed_observables"))
    )}
    observations = [ref for ref in refs if ref in units
                    and units[ref].status in {"present", "partial"} and ref not in boundary_refs]
    operational = [ref for ref in refs if ref in boundary_refs or
                   (ref in units and units[ref].status in {"failed", "unsupported", "missing"})]
    # Route failure need not invalidate surviving, individually typed subresults.
    no_observations = not observations and (
        boundary_only or result.get("status") in {"failed", "unsupported", "precondition_missing"}
        or (bool(refs) and set(refs) == set(operational))
    )
    return {
        "result_id": result["result_id"], "execution_status": result.get("status"),
        "required_mechanism_labels": list((result.get("investigation_plan") or {}).get("candidate_labels", [])),
        "status_scope": "whole execution; individual observations may survive failed substeps",
        "allowed_outcomes": ["no_result"] if no_observations
        else list(get_args(ResultDisposition.model_fields["outcome"].annotation)),
        "observation_refs": observations, "operational_refs": operational,
        "unclassified_refs": [ref for ref in refs if ref not in observations and ref not in operational],
        "reason": "No delivered scientific observation; record the limitation using no_result."
        if no_observations else "Interpret retained observations within their scope; missing outputs remain unresolved.",
    }


def operational_evidence_refs(case: CaseRun) -> set[str]:
    refs = {e.evidence_id for e in case.evidence_ledger.items
            if e.status in {"failed", "unsupported", "missing"}}
    state = case.runtime.get("question_loop", {})
    for result in [*state.get("results", []), *state.get("pending_results", [])]:
        refs.update(disposition_contract(result, case)["operational_refs"])
    return refs


def constrain_disposition_schema(schema: dict, contracts: list[dict]) -> None:
    """Refine the public output schema using the validator's exact per-result rule."""
    schema["properties"]["dispositions"].update(minItems=len(contracts), maxItems=len(contracts))
    if not contracts:
        return
    definition = schema["$defs"]["ResultDisposition"]
    # Complete alternatives preserve object shape in both local validation and
    # strict API schemas. Property-only intersections are not wire objects.
    branches = []
    for contract in contracts:
        branch = deepcopy(definition)
        branch["properties"]["result_id"]["const"] = contract["result_id"]
        branch["properties"]["outcome"]["enum"] = contract["allowed_outcomes"]
        branches.append(branch)
    schema["$defs"]["ResultDisposition"] = {"anyOf": branches}
