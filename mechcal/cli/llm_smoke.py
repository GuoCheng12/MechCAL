from __future__ import annotations

import json

import typer

from mechcal.runtime.llm import OpenAICompatibleSettings, run_llm_smoke


def run() -> None:
    settings = OpenAICompatibleSettings.from_env()
    missing = settings.missing_fields()
    if missing:
        typer.echo(
            json.dumps(
                {
                    "status": "missing_configuration",
                    "missing": missing,
                    "hint": (
                        "Fill mechcal/env/llm.local.sh or export the missing "
                        "variables."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise typer.Exit(2)

    reply = run_llm_smoke(settings)
    typer.echo(
        json.dumps(
            {
                "status": "ok",
                "base_url": settings.base_url,
                "model": settings.model,
                "reply": reply,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cli() -> None:
    typer.run(run)


if __name__ == "__main__":
    cli()
