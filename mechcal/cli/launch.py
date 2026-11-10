"""Installed, workspace-independent MechCAL command workflow."""

import argparse
from dataclasses import replace
from importlib import metadata, resources, util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import uuid

from pydantic import ValidationError

from mechcal.agents.command_tools import NativeCommandClient
from mechcal.cli.deployment import load_deployment, environment, scoped_environment, tool_checks
from mechcal.runtime import MechCALOrchestrator, mcp_deps
from mechcal.runtime.progress import atomic_json
from mechcal.tools import StdioMcpServerConfig, StdioMcpTransport
from mechcal.tools.mcp_adapter import AieMcpToolSpecs


def transport():
    return StdioMcpTransport(replace(StdioMcpServerConfig.current_python_server(), timeout_seconds=600))


def doctor(config, *, mcp=False):
    runtime = config.orchestrator_config(Path("unused"))
    packages = {name: util.find_spec(name) is not None for name in ("rdkit", "ase", "numpy", "fastmcp", "mcp")}
    checks = tool_checks(config, runtime)
    report = {"packages": packages, "tools": checks,
              "credential_present": bool(os.environ.get(config.model.api_key_env, "").strip()),
              "model": config.model.model, "model_calls": 0,
              "ready": all(packages.values()) and not any(c["error"] for c in checks)}
    if mcp and report["ready"]:
        with tempfile.TemporaryDirectory(prefix="mechcal-doctor-") as tmp:
            path = Path(tmp) / "qc.json"
            atomic_json(path, config.qc.model_dump())
            with scoped_environment(environment(config, runtime, path)):
                registry = AieMcpToolSpecs().discover_capability_registry(transport())
                report["mcp_capabilities"] = len(registry.snapshot().capabilities)
    return report


def create_orchestrator(runtime, settings):
    deps = mcp_deps(runtime, transport())
    client = NativeCommandClient(settings)
    for agent in (deps.planner, deps.mechanism_agenda, deps.photophysics_arbiter,
                  deps.mechanism_critic, deps.conclusion_ledger):
        agent.llm_client = client
    for worker in deps.workers.values():
        worker.report_writer.llm_client = client
        worker.execution_planner.llm_client = client
    return MechCALOrchestrator(runtime, deps)


def run(config, *, smiles: str, output: Path, case_id: str):
    from rdkit import Chem
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", case_id):
        raise ValueError("case-id must be a filename-safe identifier, not a path")
    if Chem.MolFromSmiles(smiles) is None:
        raise ValueError("Invalid SMILES; nothing was started")
    runtime = config.orchestrator_config(output)
    settings = config.model.settings()
    report = doctor(config)
    if not report["ready"]:
        raise ValueError("Dependency check failed; run the doctor command first")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    started = time.monotonic()
    qc_path = output / "qc_config.json"
    atomic_json(qc_path, config.qc.model_dump())
    task = resources.files("mechcal").joinpath("prompts/run_task.md").read_text().strip()
    atomic_json(output / "manifest.json", {
        "case_id": case_id, "smiles": smiles, "task": task, "profile": "command_v1",
        "package_version": metadata.version("mechcal"),
        "configuration": config.model_dump(), "judge_called": False,
    })
    env = environment(config, runtime, qc_path)
    env["MECHCAL_LLM_TELEMETRY_PATH"] = str(output / "api_telemetry.jsonl")
    atomic_json(output / "status.json", {"status": "running", "case_id": case_id})
    try:
        with scoped_environment(env):
            result = create_orchestrator(runtime, settings).run(
                smiles=smiles, user_query=task, case_id=case_id, public_case_id="CASE_001")
        summary = {"case_id": case_id, "status": result.status,
                   "elapsed_seconds": round(time.monotonic() - started, 3),
                   "run_dir": str(output / "runs" / case_id), "judge_called": False,
                   "final_answer": result.final_answer.model_dump(mode="json") if result.final_answer else None}
        atomic_json(output / "summary.json", summary)
        atomic_json(output / "status.json", {k: v for k, v in summary.items() if k != "final_answer"})
        return summary
    except BaseException as exc:
        atomic_json(output / "status.json", {
            "status": "interrupted", "case_id": case_id, "error_type": type(exc).__name__,
            "error": str(exc).replace(settings.api_key, "[REDACTED]"),
            "elapsed_seconds": round(time.monotonic() - started, 3), "judge_called": False,
        })
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Local TOML configuration; no inline API key")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("doctor", help="Check local dependencies; no paid API calls")
    check.add_argument("--mcp", action="store_true", help="Also start local MCP and inspect its registry")
    commands.add_parser("probe", help="Make one small, explicitly requested model connectivity call")
    start = commands.add_parser("run", help="Run the current command-based MechCAL workflow")
    start.add_argument("--smiles", required=True)
    start.add_argument("--output", required=True, type=Path, help="New private result directory")
    start.add_argument("--case-id", default=None)
    args = parser.parse_args()
    config = None
    try:
        config = load_deployment(args.config)
        if args.command == "doctor":
            result = doctor(config, mcp=args.mcp)
            code = 0 if result["ready"] else 1
        elif args.command == "probe":
            settings = replace(config.model.settings(), max_tokens=2048, max_retries=0)
            response = NativeCommandClient(settings).complete_text_result(
                system_prompt="Reply briefly.", payload={"request": "Reply with the single word OK."})
            if response.response_error or not response.raw_text.strip():
                raise RuntimeError("Connectivity probe did not return complete text")
            result = {"text": response.raw_text, "usage": response.usage, "model": settings.model,
                      "probe_output_limit": 2048}
            code = 0
        else:
            result = run(config, smiles=args.smiles, output=args.output.expanduser().resolve(),
                         case_id=args.case_id or uuid.uuid4().hex[:12])
            code = 0 if result["status"] == "finalized" else 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return code
    except KeyboardInterrupt:
        print("Interrupted; existing records were retained.", file=sys.stderr)
        return 130
    except Exception as exc:
        message = (json.dumps([{"field": e["loc"], "message": e["msg"]}
                               for e in exc.errors(include_input=False, include_url=False)])
                   if isinstance(exc, ValidationError) else str(exc))
        if config is not None:
            key = os.environ.get(config.model.api_key_env, "").strip()
            if key:
                message = message.replace(key, "[REDACTED]")
        print(f"{type(exc).__name__}: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
