"""Portable current command loop. No private dataset, credential store or judge."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from mechcal.agents.command_tools import NativeCommandClient
from mechcal.runtime import MechCALOrchestrator, OrchestratorConfig, default_deps, mcp_deps
from mechcal.runtime.llm import OpenAICompatibleSettings
from mechcal.tools import StdioMcpServerConfig, StdioMcpTransport


def build_config(args):
    return OrchestratorConfig(
        run_base_dir=args.output_dir, max_exploration_rounds=args.max_computations,
        decision_mode="question_loop_v1", planner_command_protocol=True,
        command_native_tools=True, command_hypothesis_tracking=True,
        enable_llm_planner=True, enable_llm_worker_reports=True,
        enable_amesp=args.amesp, planner_tool_loading="on_demand",
        enable_support_auditor=True, enable_agenda_coverage_reviewer=True,
        enable_legacy_conclusion_ledger=False, enable_incremental_mechanism_portfolio=False,
        defer_review_until_portfolio=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--case-id", default="example-001")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/command-loop"))
    parser.add_argument("--max-computations", type=int, default=5)
    parser.add_argument("--amesp", action="store_true")
    parser.add_argument("--tool-backend", choices=("local", "mcp-stdio"), default="local")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = build_config(args)
    if args.dry_run:
        print(json.dumps({"case_id": args.case_id, "max_computations": config.max_exploration_rounds,
                          "command_native_tools": config.command_native_tools,
                          "confidence_tracking": config.command_hypothesis_tracking,
                          "tool_backend": args.tool_backend, "amesp": config.enable_amesp,
                          "cumulative_token_cap": None, "model_call_cap": None,
                          "model_calls": 0, "judge_calls": 0}, indent=2))
        return
    settings = OpenAICompatibleSettings.from_env()
    if settings.missing_fields():
        parser.error("Missing model configuration: " + ", ".join(settings.missing_fields()))
    if args.tool_backend == "mcp-stdio":
        server = replace(StdioMcpServerConfig.current_python_server(), timeout_seconds=600,
                         env={"MECHCAL_ENABLE_AMESP": "true" if config.enable_amesp else "false"})
        deps = mcp_deps(config, StdioMcpTransport(server))
    else:
        deps = default_deps(config)
    client = NativeCommandClient(settings)
    deps.planner.llm_client = client
    deps.mechanism_agenda.llm_client = client
    deps.photophysics_arbiter.llm_client = client
    for worker in deps.workers.values():
        worker.report_writer.llm_client = client
    result = MechCALOrchestrator(config, deps).run(
        smiles=args.smiles, case_id=args.case_id, resume=args.resume,
        user_query="Rank three candidate photophysical mechanisms from the fixed pool and provide supporting evidence.",
    )
    print(json.dumps({"case_id": result.case_id, "status": result.status,
                      "run_dir": str((args.output_dir / result.case_id).resolve()),
                      "final_answer": result.final_answer.model_dump(mode="json") if result.final_answer else None},
                     indent=2))


if __name__ == "__main__":
    main()
