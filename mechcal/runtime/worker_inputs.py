"""Metadata-only input reachability; not scientific approval or source selection."""

from mechcal.capabilities.registry import CapabilityRegistry
from mechcal.capabilities.schemas import CapabilitySpec
from mechcal.schemas.artifacts import ArtifactManifest


def missing_kinds(spec: CapabilitySpec, kinds: set[str]) -> dict:
    return {
        "required": sorted(set(spec.required_artifact_kinds) - kinds),
        "one_of": (list(spec.source_artifact_kinds)
                   if spec.source_artifact_kinds and not kinds.intersection(spec.source_artifact_kinds) else []),
    }


def input_paths(registry: CapabilityRegistry, capability_ids: list[str],
                artifacts: ArtifactManifest, *, amesp_enabled: bool) -> dict:
    specs = [registry.require(cid) for cid in capability_ids]
    disabled = {s.capability_id: s.disabled_reason or "Backend disabled in this run."
                for s in specs if s.disabled_reason or not s.backend_enabled or (s.requires_amesp and not amesp_enabled)}
    usable = [s for s in specs if s.capability_id not in disabled]
    initial = {a.kind for a in artifacts.artifacts if a.status in {"available", "partial"}}
    kinds, reachable = set(initial), set()
    ready = [s.capability_id for s in usable if not any(missing_kinds(s, kinds).values())]
    # Declared products allow planning a chain, but do not create observations.
    while True:
        new = [s for s in usable if s.capability_id not in reachable
               and not any(missing_kinds(s, kinds).values())]
        if not new:
            break
        for spec in new:
            reachable.add(spec.capability_id)
            kinds.update(spec.outputs)
    return {"input_ready": ready, "potentially_reachable": sorted(reachable),
            "unreachable": {s.capability_id: missing_kinds(s, kinds)
                            for s in usable if s.capability_id not in reachable},
            "disabled": disabled}


def gap_report(paths: dict, artifacts: ArtifactManifest) -> dict:
    needed = {kind for gap in paths["unreachable"].values() for values in gap.values() for kind in values}
    candidates = [{"ref": a.artifact_id, "kind": a.kind, "status": a.status}
                  for a in artifacts.artifacts if a.kind in needed and a.status in {"available", "partial"}]
    return {**paths, "candidate_inputs": candidates[:6], "candidate_input_count": len(candidates),
            "directory": {"ref": "context:planner", "paths": ["/fields/artifact_index"]},
            "policy": "Candidates are not selected or authorized. Planner must explicitly delegate inputs or "
                      "a route that can produce them. Reachability checks kinds only, not calculation success."}
