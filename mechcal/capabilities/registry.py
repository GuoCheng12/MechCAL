from __future__ import annotations

import hashlib
import json

from mechcal.capabilities.schemas import CapabilityCatalog, CapabilitySpec
from mechcal.schemas.artifacts import ArtifactManifest
from mechcal.schemas.decisions import DispatchRequest
from mechcal.schemas.tool_arguments import argument_model, planner_argument_model, supplied_argument_schema, validate_tool_args


class CapabilityRegistry:
    def __init__(self, capabilities: list[CapabilitySpec]) -> None:
        self._capabilities = self._index(capabilities)

    @staticmethod
    def _index(capabilities: list[CapabilitySpec]) -> dict[str, CapabilitySpec]:
        indexed: dict[str, CapabilitySpec] = {}
        for capability in capabilities:
            if capability.capability_id in indexed:
                raise ValueError(f"Duplicate capability_id: {capability.capability_id}")
            indexed[capability.capability_id] = capability
        return indexed

    def get(self, capability_id: str) -> CapabilitySpec | None:
        return self._capabilities.get(capability_id)

    def require(self, capability_id: str) -> CapabilitySpec:
        capability = self.get(capability_id)
        if capability is None:
            raise KeyError(f"Unknown capability_id: {capability_id}")
        return capability

    def snapshot(self) -> CapabilityCatalog:
        return CapabilityCatalog(capabilities=list(self._capabilities.values()))

    def cards(self) -> list[dict[str, object]]:
        return [
            {
                "capability_id": capability.capability_id,
                "owner_agent": capability.owner_agent,
                "evidence_family": capability.evidence_family,
                "route": capability.route,
                "description": capability.description,
                "required_artifact_kinds": list(capability.required_artifact_kinds),
                "outputs": list(capability.outputs),
                "cost": capability.cost,
                "failure_modes": list(capability.failure_modes),
                "tool_name": capability.tool_binding.tool_name,
                "tool_arg_schema": supplied_argument_schema(argument_model(capability.route, capability.owner_agent)),
                "question_loop_arg_schema": supplied_argument_schema(planner_argument_model(capability.route))
                if planner_argument_model(capability.route) else None,
                "requires_amesp": capability.requires_amesp,
                "backend_enabled": capability.backend_enabled,
                "disabled_reason": capability.disabled_reason,
                "allowed_argument_values": capability.allowed_argument_values,
                "usage_hints": capability.metadata.get("tool_arg_hints", {}),
                "limitations": capability.metadata.get("limitations", []),
                "execution_dependencies": capability.metadata.get("execution_dependencies", []),
                "delivery_contract": capability.metadata.get("delivery_contract", {}),
            }
            for capability in self._capabilities.values()
        ]

    def with_amesp_backend(self, enabled: bool) -> CapabilityRegistry:
        return CapabilityRegistry([
            item.model_copy(update={"backend_enabled": enabled and item.backend_enabled})
            if item.requires_amesp else item
            for item in self._capabilities.values()
        ])

    def readiness(
        self, capability_id: str, artifacts: ArtifactManifest, *, amesp_enabled: bool,
        source_artifact_id: str | None = None,
        tool_args: dict | None = None,
    ) -> dict[str, object]:
        spec = self.require(capability_id)
        available = {a.kind for a in artifacts.artifacts if a.status in {"available", "partial"}}
        missing = sorted(set(spec.required_artifact_kinds) - available)
        reason = spec.disabled_reason
        if not reason and spec.requires_amesp and (not amesp_enabled or not spec.backend_enabled):
            reason = "Amesp execution is disabled in the current backend or run configuration."
        if not reason and not missing and spec.metadata.get("external_backend"):
            from mechcal.tools.external_qc import source_preflight, contact_scan_preflight
            try:
                source, atoms = source_preflight(artifacts, source_artifact_id, spec.metadata["external_backend"])
                if spec.route in {"compare_singlet_triplet_states", "evaluate_spin_orbit_couplings"}:
                    from mechcal.tools.dftb_spin import spin_parameters
                    from mechcal.tools.qc_process import QCConfig
                    spin_parameters(QCConfig(**spec.metadata["spin_parameter_config"]), atoms.get_chemical_symbols())
                if spec.route == "search_dimer_contacts" and tool_args:
                    _, partner = source_preflight(artifacts, tool_args.get("partner_artifact_id"), spec.metadata["external_backend"])
                    if len(atoms) + len(partner) > 150:
                        raise ValueError("Dimer calculations support at most 150 atoms in total.")
                if spec.route == "scan_intermolecular_distance":
                    from mechcal.schemas.external_qc import ContactScanArgs
                    scan_args = ContactScanArgs.model_validate(tool_args) if tool_args else None
                    contact_scan_preflight(source, atoms, scan_args)
            except ValueError as exc:
                missing.append(str(exc))
        elif not reason and not missing and spec.owner_agent == "microscopic":
            from mechcal.tools.amesp_microscopic import AmespMicroscopicError, microscopic_preflight

            candidates = [a for a in artifacts.artifacts if a.kind in spec.required_artifact_kinds
                          and a.status in {"available", "partial"}]
            if source_artifact_id:
                candidates = [a for a in candidates if a.artifact_id == source_artifact_id]
            if source_artifact_id and not candidates:
                missing.append("usable source_artifact_id: " + source_artifact_id)
            elif candidates:
                # Same default source selection as the local microscopic adapter.
                try:
                    if spec.route == "select_computed_geometry":
                        from mechcal.tools.geometry_reuse import geometry_selection_input
                        geometry_selection_input(artifacts, candidates[-1].artifact_id,
                                                 (tool_args or {}).get("member_id"))
                    else:
                        microscopic_preflight(spec.route, candidates[-1], tool_args=tool_args)
                except AmespMicroscopicError as exc:
                    if exc.code == "capability_unsupported":
                        reason = exc.message
                    else:
                        missing.append(exc.message)
        return {
            "status": "disabled" if reason else "missing_input" if missing else "ready",
            "missing_inputs": missing,
            "reason": reason,
        }

    def compact_catalog(
        self, artifacts: ArtifactManifest, *, amesp_enabled: bool,
    ) -> list[dict[str, object]]:
        return [{
            "capability_id": spec.capability_id,
            "description": spec.description.split(". ", 1)[0],
            "molecular_input": "current case SMILES",
            "required_artifact_kinds": list(spec.required_artifact_kinds),
            "outputs": list(spec.outputs),
            "execution_dependencies": spec.metadata.get("execution_dependencies", []),
            "unimplemented_structured_descriptors": spec.metadata.get("delivery_contract", {}).get(
                "unimplemented_structured_descriptors", []),
            **self.readiness(spec.capability_id, artifacts, amesp_enabled=amesp_enabled),
        } for spec in self._capabilities.values()]

    def describe_tools(
        self, capability_ids: list[str], artifacts: ArtifactManifest, *, amesp_enabled: bool,
    ) -> list[dict[str, object]]:
        unknown = set(capability_ids) - self._capabilities.keys()
        if unknown:
            raise ValueError("Unknown or disallowed capability IDs: " + ", ".join(sorted(unknown)))
        cards = {str(card["capability_id"]): card for card in self.cards()}
        return [{
            **cards[capability_id],
            "definition_version": self.definition_version(capability_id),
            **self.readiness(capability_id, artifacts, amesp_enabled=amesp_enabled),
        } for capability_id in dict.fromkeys(capability_ids)]

    def definition_version(self, capability_id: str) -> str:
        spec = self.require(capability_id)
        definition = {
            **spec.model_dump(mode="json"),
            "tool_arg_schema": supplied_argument_schema(argument_model(spec.route, spec.owner_agent)),
            "question_loop_arg_schema": supplied_argument_schema(planner_argument_model(spec.route))
            if planner_argument_model(spec.route) else None,
        }
        return hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()

    def availability_violations(
        self, request: DispatchRequest, artifacts: ArtifactManifest, *, amesp_enabled: bool,
    ) -> list[str]:
        if self.get(request.capability_id) is None:
            return [f"Unknown capability_id: {request.capability_id}"]
        state = self.readiness(request.capability_id, artifacts, amesp_enabled=amesp_enabled,
                               source_artifact_id=request.tool_args.get("source_artifact_id"),
                               tool_args=request.tool_args)
        violations = []
        if state["status"] != "ready":
            violations.append(f"{request.capability_id} is {state['status']}: "
                              f"{state['reason'] or state['missing_inputs']}")
        spec = self.require(request.capability_id)
        args = argument_model(spec.route, spec.owner_agent)
        for name, allowed in spec.allowed_argument_values.items():
            value = request.tool_args.get(name, args.model_fields[name].default)
            if value not in allowed:
                violations.append(f"{request.capability_id}: {name} must be one of {allowed}; "
                                  "other values are disabled pending validation.")
        return violations

    def for_owner_agents(self, owner_agents: tuple[str, ...]) -> CapabilityRegistry:
        allowed = set(owner_agents)
        return CapabilityRegistry(
            [
                capability
                for capability in self._capabilities.values()
                if capability.owner_agent in allowed
            ]
        )

    def validate_dispatch(
        self,
        request: DispatchRequest,
        artifact_manifest: ArtifactManifest,
    ) -> list[str]:
        capability = self.get(request.capability_id)
        if capability is None:
            return [f"Unknown capability_id: {request.capability_id}"]

        violations = [
            message
            for condition, message in (
                (
                    request.agent_name != capability.owner_agent,
                    (
                        f"Capability {request.capability_id} belongs to "
                        f"{capability.owner_agent}, got {request.agent_name}."
                    ),
                ),
                (
                    request.route != capability.route,
                    (
                        f"Capability {request.capability_id} requires route "
                        f"{capability.route}, got {request.route}."
                    ),
                ),
                (
                    request.evidence_goal_family != capability.evidence_family,
                    (
                        f"Capability {request.capability_id} produces "
                        f"{capability.evidence_family}, got "
                        f"{request.evidence_goal_family}."
                    ),
                ),
            )
            if condition
        ]
        violations.extend(self._artifact_violations(capability, artifact_manifest))
        from mechcal.runtime.observation_delivery import requirement_violations
        try:
            validate_tool_args(request.route, request.tool_args, request.agent_name)
            violations.extend(requirement_violations(capability, request))
        except ValueError as exc:
            violations.append(f"Invalid tool_args: {exc}")
        return violations

    def _artifact_violations(
        self,
        capability: CapabilitySpec,
        artifact_manifest: ArtifactManifest,
    ) -> list[str]:
        available_kinds = {
            artifact.kind
            for artifact in artifact_manifest.artifacts
            if artifact.status in {"available", "partial"}
        }
        return [
            (
                f"Capability {capability.capability_id} requires artifact kind "
                f"{kind}, but no available artifact of that kind is present."
            )
            for kind in capability.required_artifact_kinds
            if kind not in available_kinds
        ]
