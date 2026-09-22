from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AblationMode = Literal[
    "full_mechcal",
    "parallel_worker_summary",
    "one_pass_mechcal",
    "mechcal_wo_macro",
    "mechcal_wo_microscopic",
]

ABLATION_MODES: tuple[AblationMode, ...] = (
    "full_mechcal",
    "parallel_worker_summary",
    "one_pass_mechcal",
    "mechcal_wo_macro",
    "mechcal_wo_microscopic",
)


@dataclass(frozen=True)
class AblationPolicy:
    mode: AblationMode
    enabled_worker_agents: tuple[str, ...]
    workflow: Literal["iterative", "parallel_summary", "one_pass"]
    enable_agenda_coverage_reviewer: bool
    enable_support_auditor: bool
    worker_calls_per_agent_limit: int | None = None

    @property
    def disabled_worker_agents(self) -> tuple[str, ...]:
        return tuple(
            agent
            for agent in ("macro", "microscopic")
            if agent not in self.enabled_worker_agents
        )


def ablation_policy(mode: AblationMode) -> AblationPolicy:
    policies: dict[AblationMode, AblationPolicy] = {
        "full_mechcal": AblationPolicy(
            mode="full_mechcal",
            enabled_worker_agents=("macro", "microscopic"),
            workflow="iterative",
            enable_agenda_coverage_reviewer=True,
            enable_support_auditor=True,
        ),
        "parallel_worker_summary": AblationPolicy(
            mode="parallel_worker_summary",
            enabled_worker_agents=("macro", "microscopic"),
            workflow="parallel_summary",
            enable_agenda_coverage_reviewer=False,
            enable_support_auditor=False,
            worker_calls_per_agent_limit=1,
        ),
        "one_pass_mechcal": AblationPolicy(
            mode="one_pass_mechcal",
            enabled_worker_agents=("macro", "microscopic"),
            workflow="one_pass",
            enable_agenda_coverage_reviewer=False,
            enable_support_auditor=False,
            worker_calls_per_agent_limit=1,
        ),
        "mechcal_wo_macro": AblationPolicy(
            mode="mechcal_wo_macro",
            enabled_worker_agents=("microscopic",),
            workflow="iterative",
            enable_agenda_coverage_reviewer=True,
            enable_support_auditor=True,
        ),
        "mechcal_wo_microscopic": AblationPolicy(
            mode="mechcal_wo_microscopic",
            enabled_worker_agents=("macro",),
            workflow="iterative",
            enable_agenda_coverage_reviewer=True,
            enable_support_auditor=True,
        ),
    }
    return policies[mode]
