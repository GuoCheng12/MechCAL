from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import typer

from mechcal.compat import load_case_run, write_legacy_report

RUN_DIR_OPTION = typer.Option(..., "--run-dir", help="A MechCAL case run directory.")
OUTPUT_OPTION = typer.Option(..., "--output", help="Compatibility report path.")
FORMAT_OPTION = typer.Option(
    None,
    "--format",
    help="Report format. Defaults from output suffix.",
)


def run(
    run_dir: Path = RUN_DIR_OPTION,
    output: Path = OUTPUT_OPTION,
    report_format: Literal["json", "markdown"] | None = FORMAT_OPTION,
) -> None:
    path = write_legacy_report(load_case_run(run_dir), output, report_format=report_format)
    typer.echo(json.dumps({"output": str(path)}, ensure_ascii=False, indent=2))


def cli() -> None:
    typer.run(run)


if __name__ == "__main__":
    cli()
