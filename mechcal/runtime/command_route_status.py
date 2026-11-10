"""Read-only prerequisites for explicitly referenced comparison inputs."""

from mechcal.runtime.command_operations import protocol, record_index, record_identity


def route_status(case, route, tool, *, registry=None, amesp_enabled=False, index=None):
    if tool is None:
        return {"status": "unspecified" if route is None else "not_enabled"}
    general = {k: tool.get(k) for k in ("status", "missing_inputs", "reason")}
    if not route.input_refs:
        return general
    if registry is None:
        return {"status": "not_checked", "reason": "Referenced inputs require registry preflight."}
    spec = registry.get(route.capability_id)
    if spec is None:
        return {"status": "not_enabled"}
    if spec.disabled_reason or not spec.backend_enabled or (spec.requires_amesp and not amesp_enabled):
        return registry.readiness(route.capability_id, case.artifact_manifest, amesp_enabled=amesp_enabled)
    index = record_index(case, index)
    artifacts = {a.artifact_id: a for a in case.artifact_manifest.artifacts}
    results = protocol(case).get("results", {})
    candidates, unresolved = set(), []
    for ref in route.input_refs:
        if ref not in index.available:
            unresolved.append(ref)
            continue
        if ref in artifacts:
            candidates.add(ref)
            continue
        identity = record_identity(case, ref, index=index)
        rid = identity.get("source_result_ref") or identity.get("result_ref")
        candidates.update(a["artifact_id"] for a in results.get(rid, {}).get("artifact_updates", [])
                          if a.get("artifact_id") in artifacts)
    supplied = sorted(candidates)
    candidates = [ref for ref in supplied if artifacts[ref].kind in spec.input_artifact_kinds]
    if unresolved or (spec.input_artifact_kinds and not candidates):
        preparation = []
        if not unresolved:
            for producer in registry.snapshot().capabilities:
                if not set(producer.outputs).intersection(spec.input_artifact_kinds):
                    continue
                for ref in supplied:
                    if artifacts[ref].kind not in producer.input_artifact_kinds:
                        continue
                    check = registry.readiness(producer.capability_id, case.artifact_manifest,
                                               amesp_enabled=amesp_enabled, source_artifact_id=ref)
                    if check["status"] == "ready":
                        preparation.append({"capability_id": producer.capability_id, "source_artifact_id": ref})
        return {"status": "missing_input", "missing_inputs": unresolved or spec.input_artifact_kinds,
                "preparation_options": preparation,
                "reason": "Referenced records do not provide a usable input of the required kind. No unrelated input was substituted. "
                          "Preparation options match input/output kinds only; the Planner chooses parameters and scientific purpose."}
    if not spec.input_artifact_kinds:
        return general
    checks = [{"source_artifact_id": ref, **registry.readiness(
        route.capability_id, case.artifact_manifest, amesp_enabled=amesp_enabled,
        source_artifact_id=ref)} for ref in candidates]
    ready = [r for r in checks if r["status"] == "ready"]
    status = "ready" if ready else "disabled" if all(r["status"] == "disabled" for r in checks) else "missing_input"
    return {"status": status,
            "source_checks": checks, "selection_required": len(candidates) > 1,
            "scope": "Basic prerequisites only. Exact parameters are checked at execution; no calculation or scientific conclusion is implied."}
