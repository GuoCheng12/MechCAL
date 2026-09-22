from __future__ import annotations

from mechcal.capabilities.schemas import CapabilityCatalog, CapabilitySpec
from mechcal.schemas.artifacts import ArtifactManifest
from mechcal.schemas.decisions import DispatchRequest


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
            }
            for capability in self._capabilities.values()
        ]

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
