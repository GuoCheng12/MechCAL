from __future__ import annotations

import json
from pathlib import Path

import typer

from mechcal.eval.case_correction import (
    DEFAULT_MAX_DIAGNOSIS_TARGETS,
    DEFAULT_MAX_EVIDENCE_TARGETS,
    correct_case_path,
)

INPUT_PATH_OPTION = typer.Option(
    ...,
    "--input-path",
    help="Case JSON file, directory, or ZIP archive to correct.",
)
OUTPUT_DIR_OPTION = typer.Option(
    Path("var/corrected_cases"),
    "--output-dir",
    help="Directory for corrected JSON files.",
)
MAX_EVIDENCE_TARGETS_OPTION = typer.Option(
    DEFAULT_MAX_EVIDENCE_TARGETS,
    "--max-evidence-targets",
    min=1,
    help="Maximum MAS-fair semantic evidence targets per case.",
)
MAX_DIAGNOSIS_TARGETS_OPTION = typer.Option(
    DEFAULT_MAX_DIAGNOSIS_TARGETS,
    "--max-diagnosis-targets",
    min=1,
    help="Maximum MAS-fair semantic diagnosis targets per case.",
)


def run(
    input_path: Path = INPUT_PATH_OPTION,
    output_dir: Path = OUTPUT_DIR_OPTION,
    max_evidence_targets: int = MAX_EVIDENCE_TARGETS_OPTION,
    max_diagnosis_targets: int = MAX_DIAGNOSIS_TARGETS_OPTION,
) -> None:
    result = correct_case_path(
        input_path,
        output_dir,
        max_evidence_targets=max_evidence_targets,
        max_diagnosis_targets=max_diagnosis_targets,
    )
    typer.echo(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "case_count": result.case_count,
                "summary_path": str(result.summary_path),
                "audit_path": str(result.audit_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cli() -> None:
    typer.run(run)


if __name__ == "__main__":
    cli()
