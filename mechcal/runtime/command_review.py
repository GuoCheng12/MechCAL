"""Attach scoped execution facts to reviewed sources, not inferred conditions."""

from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.agents.observation_view import observation_view
from mechcal.runtime.command_memory import digest, json_bytes, pointer_child, result_report, source_provenance
from mechcal.runtime.command_operations import protocol, read_records
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import ReadEvidence


def candidate_overview(case: CaseRun) -> list[dict]:
    p = protocol(case)
    return [{"label": label, "ranked": label in p.get("ranking", {}),
             **p.get("candidates", {}).get(label, {
                 "status": "considering" if label in p.get("ranking", {}) else "unexamined",
                 "note": "No separate comparison note recorded.", "refs": [], "limitation_refs": []})}
            for label in MECHANISM_POOL]


def review_basis(case: CaseRun) -> str:
    """Compare recorded decisions/provenance, not prose meaning or scientific merit."""
    p = protocol(case)
    material = {
        "ranking": {label: {k: v.get(k) for k in ("priority", "confidence", "support", "claim", "argument_ids")}
                    for label, v in p.get("ranking", {}).items()},
        "arguments": {aid: {k: v.get(k, []) for k in ("label", "refs", "limitation_refs", "citations")}
                      for aid, v in p.get("arguments", {}).items()},
        "candidates": {label: {k: v.get(k) for k in ("status", "refs", "limitation_refs")}
                       for label, v in p.get("candidates", {}).items()},
    }
    return digest(material)


def review_citations(case: CaseRun, arguments: list[dict]) -> list[dict]:
    """Resolve the exact Planner-selected fields; never regenerate numbers with an LLM."""
    rows, seen = [], set()
    for argument in arguments:
        selections = argument.get("citations", [])
        if not selections:
            continue
        for row in read_records(case, ReadEvidence(kind="read_evidence", records=selections)):
            identity = digest({"ref": row["ref"], "selection": row["selection"]})
            if identity not in seen:
                rows.append({**row, "values_sha256": digest(row["data"])})
                seen.add(identity)
    return rows


def review_observations(case, refs, available):
    """Exact bounded views, with full records available via scoped read_evidence."""
    shown = {}
    results = protocol(case).get("results", {})
    for ref in sorted(refs):
        row = available[ref]
        if json_bytes(row) <= 2048:
            shown[ref] = observation_view(row, preserve_paths=True)
        elif ref in results:
            shown[ref] = result_report(ref, row)
        else:
            fields, deferred = {}, []
            for name, value in row.items():
                if json_bytes(value) <= 1024 and json_bytes({**fields, name: value}) <= 2048:
                    fields[name] = value
                else:
                    deferred.append({"path": pointer_child("", name), "bytes": json_bytes(value)})
            shown[ref] = {"fields": fields, "deferred": deferred,
                          "policy": "Selected exact fields; deferred values remain readable at the original ref/path."}
    return shown


def reviewer_scope(case, refs, available):
    """Cited records and their direct execution/input/output records, not all history."""
    scoped = set(refs)
    for context in review_execution_context(case, refs, available):
        rid = context["result_id"]
        result = protocol(case)["results"][rid]
        scoped.add(rid)
        scoped.update(e["evidence_id"] for e in result.get("evidence_units", []) if e.get("evidence_id"))
        scoped.update(a["artifact_id"] for a in result.get("artifact_updates", []))
        request = context.get("request")
        if request:
            scoped.add(request["dispatch_id"])
            scoped.update(request["input_refs"])
    return {ref: available[ref] for ref in sorted(scoped) if ref in available}


def _parameters(ref: str, path: str, values: dict | None) -> dict | None:
    if values is None:
        return None
    shown, deferred = {}, []
    for key, value in values.items():
        if json_bytes(value) <= 2048 and json_bytes({**shown, key: value}) <= 4096:
            shown[key] = value
        else:
            deferred.append({"path": pointer_child(path, key), "bytes": json_bytes(value)})
    return {"source_ref": ref, "path": path, "values": shown, "deferred": deferred}


def review_execution_context(case: CaseRun, refs: set[str], available: dict) -> list[dict]:
    """Join only cited sources to exact stored results and dispatches.

    `available` is the ownership-checked records index. Input references identify
    inputs; they do not pull those inputs' contents or upstream execution history.
    """
    p = protocol(case)
    contexts = []
    for result_id, result in p.get("results", {}).items():
        if not isinstance(result, dict):
            continue
        source_ids = {result_id}
        source_ids.update(ref for ref in refs if isinstance(available[ref], dict)
                          and available[ref].get("source_report_id") == result_id)
        source_ids.update(a["artifact_id"] for a in result.get("artifact_updates", []))
        sources = sorted(refs & source_ids)
        if not sources:
            continue
        dispatch_id = result.get("dispatch_id")
        action = p.get("actions", {}).get(dispatch_id)
        request = None
        if action is not None:
            if (action.get("dispatch_id", dispatch_id) != dispatch_id
                    or action.get("case_id", case.case_id) != case.case_id
                    or any(action[key] != result[key] for key in ("capability_id", "round_id", "agent_name")
                           if key in action and key in result)):
                raise RuntimeError(f"Execution request identity or ownership mismatch: {dispatch_id}")
            request = {"dispatch_id": dispatch_id, "input_refs": action.get("input_refs", []),
                       "parameters": _parameters(dispatch_id, "/tool_args", action.get("tool_args"))}
        computed = result.get("structured_results", {}).get("parsed_observables", {})
        contexts.append({
            "source_refs": sources, "result_id": result_id,
            "capability_id": result.get("capability_id"), "round_id": result.get("round_id"),
            "status": result.get("status"), "failure": result.get("failure"),
            "provenance": source_provenance(result),
            "request": request,
            "effective_parameters": _parameters(result_id,
                "/structured_results/parsed_observables/effective_parameters",
                computed.get("effective_parameters")),
        })
    return contexts
