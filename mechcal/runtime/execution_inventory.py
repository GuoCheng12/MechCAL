"""Derived execution facts, never mechanism support or a second evidence store."""

from collections import Counter

from mechcal.runtime.command_memory import digest
from mechcal.runtime.command_operations import protocol, record_index


def result_fact(ref, result):
    structured = result.get("structured_results", {})
    parsed = structured.get("parsed_observables", {})
    fact = {"result_id": ref, "status": result.get("status", "unknown")}
    if result.get("round_id"):
        fact["round_id"] = result["round_id"]
    # Counts concern returned members of one execution, not distinct structures
    # across runs or independent pieces of scientific evidence.
    members = {}
    for field in ("candidates", "points", "excited_states", "states"):
        rows = parsed.get(field)
        if isinstance(rows, list):
            members[field] = {"path": "/structured_results/parsed_observables/" + field,
                              "returned": len(rows)}
            statuses = Counter(row["status"] for row in rows if isinstance(row, dict) and "status" in row)
            if statuses:
                members[field]["statuses"] = dict(statuses)
            geometry = Counter(str(row["geometry_converged"]).lower() for row in rows
                               if isinstance(row, dict) and "geometry_converged" in row)
            if geometry:
                members[field]["geometry_converged_counts"] = dict(geometry)
    if members:
        fact["returned_members"] = members
    failure = result.get("failure") or {}
    if failure.get("kind", "none") != "none":
        details = failure.get("details") or {}
        fact["failure"] = {"code": details.get("code", failure.get("kind")),
            "details_path": "/failure/details",
            **{key: details[key] for key in ("stage", "completed_stages", "max_wall_seconds", "elapsed_seconds")
               if key in details}}
    args = structured.get("tool_args") or (failure.get("details") or {}).get("tool_args") or {}
    for name in ("source_artifact_id", "state_root"):
        if name in args:
            fact[name] = args[name]
    return fact


def groups(case, *, index=None):
    index = record_index(case, index)

    def build():
        grouped = {}
        for ref, result in protocol(case).get("results", {}).items():
            cid = result.get("capability_id")
            if cid:
                grouped.setdefault(cid, []).append((ref, result))
        summary = {cid: {"version": digest({ref: index.version(ref) for ref, _ in rows}),
                      "attempts": len(rows),
                      "statuses": dict(Counter(row.get("status", "unknown") for _, row in rows)),
                      "recent": [result_fact(ref, row) for ref, row in rows[-3:]],
                      "omitted_attempts": max(0, len(rows) - 3)}
                for cid, rows in grouped.items()}
        for cid, rows in grouped.items():
            usable = [(ref, row) for ref, row in rows if row.get("status") in {"success", "partial"}]
            if usable and usable[-1][0] not in {ref for ref, _ in rows[-3:]}:
                summary[cid]["earlier_usable_result"] = result_fact(*usable[-1])
        return summary

    return index.memo("execution_inventory", build)


def view(case, *, index=None, refs=None):
    rows = groups(case, index=index)
    if refs is not None:
        related = versions_for_refs(case, refs, index=index)
        rows = {cid: row for cid, row in rows.items() if "execution_scope:" + cid in related}
    return {"by_capability": rows,
        "policy": "Execution facts only, not support scores. Counts are per execution, not unique conformers "
                  "or independent evidence. Earlier failures remain recorded after recovery; later failure does not "
                  "erase success. A partial execution can contain successfully converged members; member status "
                  "and geometry_converged_counts concern individual returned results. Recheck historical absence/count limitations against these current records. "
                  "Use result IDs and listed paths to inspect original observations. Unknown is not absent."}


def route_history(case, capability_id, *, index=None):
    row = groups(case, index=index).get(capability_id)
    return {"scope": "capability across this case, not necessarily this question or these parameters",
            **(row if row else {"attempts": 0, "statuses": {}, "recent": [], "omitted_attempts": 0})}


def versions_for_refs(case, refs, *, index=None):
    from mechcal.runtime.command_operations import citation_source_refs
    index = record_index(case, index)
    results = protocol(case).get("results", {})
    refs = set(refs) | set(citation_source_refs(case, refs))
    cids = {results[ref].get("capability_id") for ref in refs if ref in results}
    inventory = groups(case, index=index)
    return {"execution_scope:" + cid: inventory[cid]["version"] for cid in sorted(c for c in cids if c)
            if cid in inventory}


def opportunities(case, catalog, *, index=None):
    attempted = groups(case, index=index)
    return [{"capability_id": row["capability_id"], "status": row["status"],
             "missing_inputs": row.get("missing_inputs", [])}
            for row in catalog if row.get("status") in {"ready", "missing_input"}
            and row["capability_id"] not in attempted]
