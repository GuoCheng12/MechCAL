"""Tool environment for agent runs, derived from the same TOML deployment config as ``mechcal run``."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def tool_environment(config_path: Path, run_dir: Path):
    """Export Amesp/Multiwfn/QC settings for the duration of a run; yields the parsed deployment."""
    from mechcal.cli.deployment import environment, load_deployment, scoped_environment

    config = load_deployment(Path(config_path))
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    qc_path = run_dir / "qc_config.json"
    qc_path.write_text(json.dumps(config.qc.model_dump(), indent=1), encoding="utf-8")
    runtime = config.orchestrator_config(run_dir)
    env = environment(config, runtime, qc_path)
    env["MECHCAL_LLM_TELEMETRY_PATH"] = str(run_dir / "api_telemetry.jsonl")
    with scoped_environment(env):
        yield config
