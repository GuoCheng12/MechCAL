"""Token use and cost of runs, from their API telemetry (evaluated systems and judge separately).

    python examples/token_usage.py outputs/agent/dev4a outputs/baselines/react_dev ...

Reads <run>/api_telemetry.jsonl (the agent or baseline) and <run>/judge_telemetry.jsonl or
<dir>/*.judge_telemetry.jsonl (grading, perturbations) under each directory. Cost uses the
deepseek-flash prices and the peak/off-peak window of each call's timestamp
(mechcal.agent.schedule); prices per 1M tokens: cache hit, cache miss, output (peak; off-peak half).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from mechcal.agent.schedule import is_peak

PEAK_PRICE = {"hit": 0.006, "miss": 0.30, "out": 1.20}      # deepseek-flash, USD per 1M tokens, read 2026-11-13


def tally(paths) -> Counter:
    c = Counter()
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            u, ts = row.get("usage") or {}, row.get("timestamp_utc")
            hit = u.get("cached_tokens", 0)
            miss, out = u.get("prompt_tokens", 0) - hit, u.get("completion_tokens", 0)
            peak = bool(ts) and is_peak(datetime.fromisoformat(ts.replace("Z", "+00:00")))
            f = 1.0 if peak else 0.5
            c["calls"] += 1
            c["tokens"] += u.get("total_tokens", 0)
            c["peak_tokens"] += u.get("total_tokens", 0) if peak else 0
            c["usd"] += f * (hit * PEAK_PRICE["hit"] + miss * PEAK_PRICE["miss"] + out * PEAK_PRICE["out"]) / 1e6
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    args = ap.parse_args()
    for d in args.dirs:
        system = tally(sorted(d.glob("*/api_telemetry.jsonl")) + sorted(d.glob("api_telemetry.jsonl")))
        judge = tally(sorted(d.glob("*/judge_telemetry.jsonl")) + sorted(d.glob("*.judge_telemetry.jsonl")))
        for name, c in (("system", system), ("judge", judge)):
            if c["calls"]:
                print(f"{str(d):48s} {name:6s} calls {c['calls']:5d}  tokens {c['tokens'] / 1e6:7.2f} M"
                      f"  peak {c['peak_tokens'] / max(c['tokens'], 1):4.0%}  ${c['usd']:.2f}")


if __name__ == "__main__":
    main()
