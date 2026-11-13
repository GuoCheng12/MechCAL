"""Run the MechCAL agent (release 0.3.0) on one case.

    python -m mechcal.agent.cli --case-id AIE_DDX_MECH_V4_0003 --out outputs/agent/0003-smoke

The SMILES is read from the public benchmark inputs when --smiles is omitted. Model settings
come from the environment (env/deepseek.env.sh); tool paths from mechcal.local.toml.
Inference only: no hidden reference, rubric or judge is loaded here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from mechcal.agent.environment import tool_environment
from mechcal.agent.loop import AGENT_RELEASE, AgentConfig, AgentRun
from mechcal.eval.llm_settings import backbone_settings, describe
from mechcal.runtime.llm import OpenAIJsonClient

from mechcal.eval.dataset import PUBLIC_INPUTS, public_smiles  # noqa: F401  (structure-corrected inputs)

ROOT = Path(__file__).resolve().parents[2]


def _commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--smiles")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "mechcal.local.toml")
    parser.add_argument("--allow-not-runnable", action="store_true",
                        help="run a case whose structure is marked as not yet preparable (see structure_verification)")
    parser.add_argument("--allow-test-case", action="store_true",
                        help="run a held-out test case (only after the system is frozen)")
    for name, default in asdict(AgentConfig()).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(flag, type=type(default), default=default)
    args = parser.parse_args(argv)
    from mechcal.eval.splits import HELD_OUT, split_of
    if split_of(args.case_id) == "test" and not args.allow_test_case:
        raise SystemExit(HELD_OUT.format(case=args.case_id, flag="--allow-test-case"))
    from mechcal.eval.dataset import NOT_RUNNABLE, runnable
    ok, why = runnable(args.case_id)
    if not ok and not args.allow_not_runnable and not args.smiles:
        raise SystemExit(NOT_RUNNABLE.format(case=args.case_id, reason=why, flag="--allow-not-runnable"))
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; use a new output directory")
    smiles = args.smiles or public_smiles(args.case_id)
    if args.cost_log_path in {"", None}:          # default: shared cost log of agent runs ("off" disables)
        args.cost_log_path = str(ROOT / "outputs" / "agent" / "cost_log.jsonl")
    elif args.cost_log_path == "off":
        args.cost_log_path = ""
    config = AgentConfig(**{k: getattr(args, k) for k in asdict(AgentConfig())})
    settings = backbone_settings()
    if not settings.api_key:
        raise SystemExit("No API key in the environment; source ~/.config/mechcal/secrets.env and env/deepseek.env.sh")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps({
        "release": AGENT_RELEASE, "commit": _commit(), "case_id": args.case_id, "smiles": smiles,
        "agent_config": asdict(config), "model": describe(settings), "inference_only": True,
    }, indent=1))
    with tool_environment(args.config, args.out) as deployment:
        run = AgentRun(OpenAIJsonClient(settings), case_id=args.case_id, smiles=smiles, run_dir=args.out,
                       config=config, enable_amesp=bool(deployment.runtime.get("enable_amesp", True)))
        result = run.run()
    print(json.dumps({k: result[k] for k in ("case_id", "status", "stop_reason", "seconds", "usage", "tool_cost")},
                     indent=1))
    print("top-3:", [p["label"] for p in result["submission"]["mechanism_predictions"]])
    return result


if __name__ == "__main__":
    main()
