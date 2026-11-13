"""Run one baseline (Direct or ReAct) on one case; output is graded like an agent run.

    python -m mechcal.eval.baseline_cli --subject react --case-id AIE_DDX_MECH_V4_0047 --out outputs/baselines/react/0047

ReAct uses the same recorded gateway, capability cards (tools, text, applicability, cost) and
tool wall-time budget as the MechCAL agent. Writes result.json (with "submission", possibly null),
artifacts.json, transcript.json and manifest.json. Held-out test cases are refused unless
--allow-test-case is given.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> dict:
    from mechcal.agent.cli import _commit, public_smiles
    from mechcal.agent.environment import tool_environment
    from mechcal.eval.grounded_subjects import DirectSubject, ReactSubject
    from mechcal.eval.llm_settings import backbone_settings, describe
    from mechcal.eval.splits import HELD_OUT, split_of
    from mechcal.eval.tool_gateway import ToolGateway
    from mechcal.runtime.llm import OpenAIJsonClient

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject", choices=["direct", "react"], required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "mechcal.local.toml")
    parser.add_argument("--max-model-calls", type=int, default=24)
    parser.add_argument("--max-tool-calls", type=int, default=12)
    parser.add_argument("--max-tool-wall-seconds", type=float, default=3600.0)
    parser.add_argument("--allow-test-case", action="store_true")
    parser.add_argument("--allow-not-runnable", action="store_true")
    args = parser.parse_args(argv)
    if split_of(args.case_id) == "test" and not args.allow_test_case:
        raise SystemExit(HELD_OUT.format(case=args.case_id, flag="--allow-test-case"))
    from mechcal.eval.dataset import NOT_RUNNABLE, runnable
    ok, why = runnable(args.case_id)
    if not ok and not args.allow_not_runnable:
        raise SystemExit(NOT_RUNNABLE.format(case=args.case_id, reason=why, flag="--allow-not-runnable"))
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; use a new output directory")
    settings = backbone_settings()
    if not settings.api_key:
        raise SystemExit("No API key in the environment; source ~/.config/mechcal/secrets.env and env/deepseek.env.sh")
    smiles = public_smiles(args.case_id)
    args.out.mkdir(parents=True, exist_ok=True)
    budgets = {"max_model_calls": args.max_model_calls, "max_tool_calls": args.max_tool_calls,
               "max_tool_wall_seconds": args.max_tool_wall_seconds}
    (args.out / "manifest.json").write_text(json.dumps({
        "subject": args.subject, "commit": _commit(), "case_id": args.case_id, "smiles": smiles,
        "budgets": budgets if args.subject == "react" else {"max_model_calls": 2}, "model": describe(settings)}, indent=1))
    client = OpenAIJsonClient(settings)
    with tool_environment(args.config, args.out) as deployment:
        if args.subject == "direct":
            result = DirectSubject(client).run(args.case_id, smiles)
        else:
            def factory(case_id, smi):
                return ToolGateway(case_id=case_id, smiles=smi, run_dir=args.out / "tools",
                                   enable_amesp=bool(deployment.runtime.get("enable_amesp", True)),
                                   max_wall_seconds=args.max_tool_wall_seconds)
            result = ReactSubject(client, factory, max_model_calls=args.max_model_calls,
                                  max_tool_calls=args.max_tool_calls).run(args.case_id, smiles)
    out = {"case_id": args.case_id, "subject": args.subject, "status": result.status, "error": result.error,
           "submission": result.submission.model_dump() if result.submission else None,
           "usage": result.usage, "tool_cost": result.tool_cost, "budgets": budgets}
    (args.out / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    (args.out / "artifacts.json").write_text(result.store.model_dump_json())
    (args.out / "transcript.json").write_text(json.dumps(result.transcript, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("case_id", "subject", "status", "usage", "tool_cost")}, indent=1, default=str))
    return out


if __name__ == "__main__":
    main()
