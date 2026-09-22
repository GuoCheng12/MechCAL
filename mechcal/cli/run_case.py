from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Literal

import typer

from mechcal.runtime import MechCALOrchestrator, OrchestratorConfig, mcp_deps
from mechcal.tools import StdioMcpServerConfig, StdioMcpTransport

SMILES_OPTION = typer.Option(..., "--smiles", help="Input molecule SMILES.")
USER_QUERY_OPTION = typer.Option(
    "Assess the likely AIE mechanism for this molecule.",
    "--user-query",
    help="User mechanism-investigation request.",
)
OUTPUT_DIR_OPTION = typer.Option(
    Path("outputs/runs"),
    "--output-dir",
    help="Directory where typed run records are written.",
)
CASE_ID_OPTION = typer.Option(None, "--case-id", help="Optional stable case id.")
MAX_ROUNDS_OPTION = typer.Option(30, "--max-rounds", min=1, help="Maximum evidence rounds.")
TOOL_BACKEND_OPTION = typer.Option(
    "local",
    "--tool-backend",
    help="Tool backend: local Python adapters or stdio MCP adapters.",
)
LLM_PLANNER_OPTION = typer.Option(
    None,
    "--llm-planner/--no-llm-planner",
    help="Enable or disable OpenAI-compatible PlannerDecision generation.",
)
LLM_WORKERS_OPTION = typer.Option(
    None,
    "--llm-workers/--no-llm-workers",
    help="Enable or disable OpenAI-compatible worker execution planning.",
)
LLM_MACRO_OPTION = typer.Option(
    None,
    "--llm-macro/--no-llm-macro",
    help="Enable or disable OpenAI-compatible Macro execution planning only.",
)
LLM_MICROSCOPIC_OPTION = typer.Option(
    None,
    "--llm-microscopic/--no-llm-microscopic",
    help="Enable or disable OpenAI-compatible Microscopic execution planning only.",
)
LLM_WORKER_REPORTS_OPTION = typer.Option(
    None,
    "--llm-worker-reports/--no-llm-worker-reports",
    help="Enable or disable OpenAI-compatible worker report writing.",
)
AMESP_OPTION = typer.Option(
    None,
    "--amesp/--no-amesp",
    help="Enable or disable the local Amesp microscopic baseline adapter.",
)


def run(
    smiles: str = SMILES_OPTION,
    user_query: str = USER_QUERY_OPTION,
    output_dir: Path = OUTPUT_DIR_OPTION,
    case_id: str | None = CASE_ID_OPTION,
    max_rounds: int = MAX_ROUNDS_OPTION,
    tool_backend: Literal["local", "mcp-stdio"] = TOOL_BACKEND_OPTION,
    llm_planner: bool | None = LLM_PLANNER_OPTION,
    llm_workers: bool | None = LLM_WORKERS_OPTION,
    llm_macro: bool | None = LLM_MACRO_OPTION,
    llm_microscopic: bool | None = LLM_MICROSCOPIC_OPTION,
    llm_worker_reports: bool | None = LLM_WORKER_REPORTS_OPTION,
    amesp: bool | None = AMESP_OPTION,
) -> None:
    config = OrchestratorConfig(
        run_base_dir=output_dir,
        max_rounds=max_rounds,
        enable_llm_planner=_flag_or_env(llm_planner, "MECHCAL_ENABLE_LLM_PLANNER"),
        enable_macro_llm_planning=_flag_or_env(
            llm_macro,
            "MECHCAL_ENABLE_LLM_MACRO",
        ),
        enable_microscopic_llm_planning=_flag_or_env(
            llm_microscopic,
            "MECHCAL_ENABLE_LLM_MICROSCOPIC",
        ),
        enable_llm_worker_planning=_flag_or_env(
            llm_workers,
            "MECHCAL_ENABLE_LLM_WORKERS",
        ),
        enable_llm_worker_reports=_flag_or_env(
            llm_worker_reports,
            "MECHCAL_ENABLE_LLM_WORKER_REPORTS",
        ),
        enable_amesp=_flag_or_env(amesp, "MECHCAL_ENABLE_AMESP"),
        enable_incremental_mechanism_portfolio=False,
        defer_review_until_portfolio=True,
    )
    deps_by_backend = {
        "local": lambda: None,
        "mcp-stdio": lambda: mcp_deps(config, StdioMcpTransport(_mcp_server_config(config))),
    }
    orchestrator = MechCALOrchestrator(config, deps=deps_by_backend[tool_backend]())
    result = orchestrator.run(smiles=smiles, user_query=user_query, case_id=case_id)
    summary = {
        "case_id": result.case_id,
        "status": result.status,
        "run_dir": str((output_dir / result.case_id).resolve()),
        "final_answer": (
            result.final_answer.model_dump(mode="json")
            if result.final_answer is not None
            else None
        ),
    }
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


def cli() -> None:
    typer.run(run)


def _flag_or_env(value: bool | None, env_name: str) -> bool:
    if value is not None:
        return value
    raw = os.environ.get(env_name, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _mcp_server_config(config: OrchestratorConfig) -> StdioMcpServerConfig:
    server_config = StdioMcpServerConfig.current_python_server()
    return replace(
        server_config,
        env={"MECHCAL_ENABLE_AMESP": "true" if config.enable_amesp else "false"},
    )


if __name__ == "__main__":
    cli()
