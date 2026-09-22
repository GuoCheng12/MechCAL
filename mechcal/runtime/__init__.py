from mechcal.runtime.ablations import (
    ABLATION_MODES,
    AblationMode,
    AblationPolicy,
    ablation_policy,
)
from mechcal.runtime.agent_runtime import BaseAgentRuntime
from mechcal.runtime.context import PlannerContextView, build_planner_context
from mechcal.runtime.gates import DeterministicGate, GatePolicy
from mechcal.runtime.output_assembler import OutputAssembler
from mechcal.runtime.persistence import RunStore
from mechcal.runtime.retry import RetryPolicy, SchemaRetryError, SchemaRetryRunner

__all__ = [
    "DeterministicGate",
    "ABLATION_MODES",
    "AblationMode",
    "AblationPolicy",
    "BaseAgentRuntime",
    "GatePolicy",
    "MechCALOrchestrator",
    "OrchestratorConfig",
    "OrchestratorDeps",
    "OutputAssembler",
    "PlannerContextView",
    "RetryPolicy",
    "RunStore",
    "SchemaRetryError",
    "SchemaRetryRunner",
    "build_planner_context",
    "ablation_policy",
    "default_deps",
    "mcp_deps",
]


def __getattr__(name: str):
    if name in {
        "MechCALOrchestrator",
        "OrchestratorConfig",
        "OrchestratorDeps",
        "default_deps",
        "mcp_deps",
    }:
        from mechcal.runtime import orchestrator

        return getattr(orchestrator, name)
    raise AttributeError(name)
