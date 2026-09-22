from importlib import resources
from pathlib import Path
import json
import os
import subprocess
import sys

from mechcal.runtime.llm import OpenAICompatibleSettings


def test_packaged_resources_are_available():
    package = resources.files("mechcal")
    assert (package / "prompts/planner_decision.md").read_text()
    assert (package / "eval/prompts/mechanism_evidence_support_judge.md").read_text()
    rubric = json.loads(
        (package / "eval/rubrics/mechanism_evidence_rubric_v2.json").read_text()
    )
    assert rubric["mechanism_evidence_rubric_version"] == "v2"


def test_model_settings_use_public_namespace(monkeypatch):
    monkeypatch.setenv("MECHCAL_OPENAI_MODEL", "example-model")
    monkeypatch.setenv("MECHCAL_OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("MECHCAL_OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("MECHCAL_OPENAI_TEMPERATURE", "omit")
    settings = OpenAICompatibleSettings.from_env()
    assert settings.model == "example-model"
    assert settings.temperature is None
    assert not settings.missing_fields()


def test_installed_entrypoints_work_outside_repository(tmp_path):
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("MECHCAL_", "AIE_MAS_")) and key != "PYTHONPATH"
    }
    commands = (
        "mechcal-run-case", "mechcal-evaluate", "mechcal-direct",
        "mechcal-tool-llm", "mechcal-common-prior", "mechcal-rescore",
        "mechcal-shards", "mechcal-export-report", "mechcal-llm-smoke",
    )
    for name in commands:
        command = Path(sys.executable).parent / name
        result = subprocess.run(
            [str(command), "--help"], cwd=tmp_path, env=env,
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"{name}: {result.stderr}"
