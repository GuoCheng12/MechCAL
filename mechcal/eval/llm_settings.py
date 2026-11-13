"""Model settings for evaluation: backbone vs. evidence judge, read from the environment.

The backbone uses the standard ``MECHCAL_OPENAI_*`` variables. The judge has its own
``MECHCAL_JUDGE_*`` variables so that it can run at a different effort now and move to a
different model family later without touching code. See ``env/deepseek.env.sh``.
"""

from __future__ import annotations

import os
from dataclasses import replace

from mechcal.runtime.llm import OpenAICompatibleSettings


def backbone_settings() -> OpenAICompatibleSettings:
    return OpenAICompatibleSettings.from_env()


def record_judge_usage(path) -> None:
    """Write the judge's per-call API telemetry (tokens, latency) to ``path`` (JSON lines), so that
    grading and perturbation costs are counted like the evaluated systems' own calls."""
    from mechcal.runtime.llm import TELEMETRY_ENV
    os.environ[TELEMETRY_ENV] = str(path)


def judge_settings() -> OpenAICompatibleSettings:
    base = backbone_settings()
    env = os.environ.get
    effort = (env("MECHCAL_JUDGE_REASONING_EFFORT") or "max").strip()
    return replace(
        base,
        base_url=(env("MECHCAL_JUDGE_BASE_URL") or base.base_url).strip(),
        model=(env("MECHCAL_JUDGE_MODEL") or base.model).strip(),
        api_key=(env("MECHCAL_JUDGE_API_KEY") or base.api_key).strip(),
        reasoning_effort=None if effort.lower() in {"", "none", "omit"} else effort,
        timeout_seconds=float(env("MECHCAL_JUDGE_TIMEOUT") or 600),
    )


def describe(settings: OpenAICompatibleSettings) -> dict:
    """Non-secret summary for run manifests."""
    return {"base_url": settings.base_url, "model": settings.model, "api_protocol": settings.api_protocol,
            "reasoning_effort": settings.reasoning_effort, "timeout_seconds": settings.timeout_seconds,
            "max_retries": settings.max_retries, "api_key_present": bool(settings.api_key)}
