from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import typer

from mechcal.benchmark import run_benchmark

DATASET_OPTION = typer.Option(..., "--dataset", help="CSV dataset with a SMILES column.")
OUTPUT_DIR_OPTION = typer.Option(
    Path("outputs/benchmarks"),
    "--output-dir",
    help="Directory for MechCAL benchmark run records.",
)
BACKEND_OPTION = typer.Option(
    "local",
    "--backend",
    help="Tool backend used by benchmark cases.",
)
LIMIT_OPTION = typer.Option(None, "--limit", min=1, help="Optional row limit.")
MAX_ROUNDS_OPTION = typer.Option(4, "--max-rounds", min=1, help="Maximum rounds per case.")
EXPORT_LEGACY_OPTION = typer.Option(
    True,
    "--export-legacy/--no-export-legacy",
    help="Write legacy compatibility JSON and Markdown per case.",
)


def run(
    dataset: Path = DATASET_OPTION,
    output_dir: Path = OUTPUT_DIR_OPTION,
    backend: Literal["local", "mcp-stdio"] = BACKEND_OPTION,
    limit: int | None = LIMIT_OPTION,
    max_rounds: int = MAX_ROUNDS_OPTION,
    export_legacy: bool = EXPORT_LEGACY_OPTION,
) -> None:
    result = run_benchmark(
        dataset_path=dataset,
        output_dir=output_dir,
        backend=backend,
        limit=limit,
        max_rounds=max_rounds,
        export_legacy=export_legacy,
    )
    typer.echo(
        json.dumps(
            {
                "summary_path": str(result.summary_path),
                "jsonl_path": str(result.jsonl_path),
                "case_count": result.case_count,
                "finalized_count": result.finalized_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cli() -> None:
    typer.run(run)


if __name__ == "__main__":
    cli()
