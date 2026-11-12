"""Run the MechCAL agent on several cases in parallel; rerunning the command completes only what is missing.

    python -m mechcal.agent.batch --cases 0047,0020,0068 --out outputs/agent/dev3 --parallel 6 --grade
    python -m mechcal.agent.batch --cases-file dev_cases.txt --out ... -- --max-computations 10

A case whose directory already holds result.json is skipped. A directory without result.json (killed
or crashed run) is moved aside as <case>-incomplete-<time> and the case runs again. Arguments after
"--" go to every ``mechcal.agent.cli`` call. Each case logs to <out>/<case>.log.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "AIE_DDX_MECH_V4_"


def _unlimited_stack() -> None:      # Amesp needs a large stack (same as `ulimit -s unlimited`)
    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))


def case_ids(args) -> list[str]:
    raw = args.cases.split(",") if args.cases else Path(args.cases_file).read_text().split()
    return [c if c.startswith(PREFIX) else PREFIX + c.zfill(4) for c in (x.strip() for x in raw) if c]


def run_case(case_id: str, out: Path, extra: list[str], grade: bool) -> dict:
    short = case_id.removeprefix(PREFIX)
    run_dir = out / short
    if (run_dir / "result.json").is_file():
        return {"case": short, "state": "skipped (already finished)"}
    if run_dir.exists():
        run_dir.rename(out / f"{short}-incomplete-{time.strftime('%Y%m%d-%H%M%S')}")
    env = dict(os.environ, KMP_STACKSIZE=os.environ.get("KMP_STACKSIZE", "4G"))
    started = time.monotonic()
    with (out / f"{short}.log").open("a", encoding="utf-8") as log:
        code = subprocess.run([sys.executable, "-m", "mechcal.agent.cli", "--case-id", case_id, "--out", str(run_dir), *extra],
                              cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, preexec_fn=_unlimited_stack).returncode
    state = "finished" if (run_dir / "result.json").is_file() else f"failed (exit {code})"
    row = {"case": short, "state": state, "minutes": round((time.monotonic() - started) / 60, 1)}
    if grade and state == "finished":
        with (out / f"{short}.grade.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, str(ROOT / "examples/grade_agent_run.py"), str(run_dir)], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT)
        if (run_dir / "grade.json").is_file():
            m = json.loads((run_dir / "grade.json").read_text())["metrics"]
            row.update(ndcg=round(m["ndcg_at_3"], 3), es_ndcg=round(m["es_ndcg_at_3"], 3))
    print(json.dumps(row), flush=True)
    return row


def main(argv: list[str] | None = None) -> list[dict]:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases")
    parser.add_argument("--cases-file")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument("--grade", action="store_true", help="grade each finished run (calls the judge model)")
    args = parser.parse_args(argv)
    if not (args.cases or args.cases_file):
        parser.error("give --cases or --cases-file")
    if not os.environ.get("MECHCAL_OPENAI_API_KEY") and not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("No API key in the environment; source ~/.config/mechcal/secrets.env and env/deepseek.env.sh")
    args.out.mkdir(parents=True, exist_ok=True)
    cases = case_ids(args)
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        rows = list(pool.map(lambda c: run_case(c, args.out, extra, args.grade), cases))
    print(f"{sum(r['state'] == 'finished' for r in rows)} finished, "
          f"{sum(r['state'].startswith('skipped') for r in rows)} skipped, "
          f"{sum(r['state'].startswith('failed') for r in rows)} failed")
    return rows


if __name__ == "__main__":
    main()
