from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.text import Text

from mechcal.runtime.progress import ExperimentProgress, read_snapshot, render_progress, write_activity


def demo(output_dir: Path, *, mode: str = "auto", delay: float = 0.25) -> None:
    ids = [f"DEMO_{index:02}" for index in range(1, 11)]
    with ExperimentProgress(output_dir, ids, method="DEMO ONLY - no API or tools",
                            model="simulated", judge="simulated", max_rounds=30, mode=mode) as progress:
        for index, case_id in enumerate(ids):
            run_dir = output_dir / "demo_runs" / case_id
            progress.start(case_id, run_dir=run_dir)
            for round_id, stage in [("R001", "Planner"), ("R001", "Macro"), ("R002", "Microscopic")]:
                write_activity(run_dir, stage, round_id)
                time.sleep(delay)
            progress.update(case_id, "judging", detail="DEMO")
            time.sleep(delay)
            status = "failed" if index == 7 else "cached" if index == 8 else "done"
            progress.finish(case_id, status=status, detail="DEMO - not an experimental result")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only terminal experiment monitor.")
    parser.add_argument("output_dir", type=Path, nargs="?")
    parser.add_argument("--once", action="store_true", help="Print one snapshot without terminal control codes.")
    parser.add_argument("--demo", action="store_true", help="Run a short simulation; no model or tool calls.")
    args = parser.parse_args()
    if args.demo:
        if args.output_dir:
            parser.error("--demo uses its own temporary directory; omit output_dir")
        directory = Path(tempfile.mkdtemp(prefix="mechcal-progress-demo-"))
        demo(directory)
        print(f"Demo checkpoint: {directory / 'progress.json'}")
        return
    if not args.output_dir:
        parser.error("output_dir is required unless --demo is used")
    path = args.output_dir.expanduser().resolve() / "progress.json"
    console = Console()
    try:
        snapshot = read_snapshot(path)
    except (OSError, ValueError) as exc:
        parser.error(f"Cannot read {path}: {type(exc).__name__}. Start an instrumented run first.")
    if args.once or not console.is_terminal:
        console.print(render_progress(snapshot, width=console.width))
        return
    try:
        with Live(console=console, screen=True, refresh_per_second=2) as live:
            while True:
                try:
                    snapshot = read_snapshot(path)
                    live.update(render_progress(snapshot, width=console.width))
                except (OSError, ValueError):
                    live.update(Text("Checkpoint temporarily unreadable; waiting...", style="yellow"))
                time.sleep(0.5)
    except KeyboardInterrupt:
        console.print("Monitor closed. The experiment process was not stopped.")


if __name__ == "__main__":
    main()
