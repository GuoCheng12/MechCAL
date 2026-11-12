from dataclasses import fields
from importlib import resources
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mechcal.cli.deployment import Deployment, ModelConfig, load_deployment, scoped_environment
from mechcal.cli import launch
from mechcal.runtime import OrchestratorConfig


def minimal_config():
    return Deployment(model=ModelConfig(base_url="https://example.test", model="test-model"),
                      runtime={"enable_amesp": False})


def test_profile_preserves_current_command_workflow():
    profile = json.loads(resources.files("mechcal").joinpath("profiles/command_v1.json").read_text())
    assert set(profile) == {f.name for f in fields(OrchestratorConfig)} - {"run_base_dir"}
    config = minimal_config().orchestrator_config(Path("out"))
    assert config.command_native_tools and config.command_task_context and config.command_worker_tasks
    assert config.command_evidence_followup and config.planner_case_memory
    assert config.unlimited_exploration and config.command_delivery_max_model_turns == 30


def test_relative_tools_resolve_against_config_not_cwd(tmp_path, monkeypatch):
    root = tmp_path / "configuration"
    root.mkdir()
    file = root / "config.toml"
    file.write_text('[model]\nbase_url="https://example.test"\nmodel="test"\n'
                    '[tools]\namesp_binary="../tools/amesp"\n'
                    '[qc]\nparameter_dir="../params"\n')
    monkeypatch.chdir(tmp_path)
    settings = load_deployment(file)
    assert settings.tools.amesp_binary == str(tmp_path / "tools/amesp")
    assert settings.qc.parameter_dir == str(tmp_path / "params")


def test_unknown_runtime_option_is_not_silently_ignored():
    config = minimal_config()
    config.runtime["typo"] = True
    with pytest.raises(ValueError, match="Unknown runtime"):
        config.orchestrator_config(Path("out"))


def test_config_cannot_override_output_directory():
    config = minimal_config()
    config.runtime["run_base_dir"] = "elsewhere"
    with pytest.raises(ValueError, match="run_base_dir"):
        config.orchestrator_config(Path("out"))


def test_inline_credentials_and_credential_urls_are_rejected():
    with pytest.raises(ValueError):
        ModelConfig(base_url="https://example.test", model="test", api_key="test-only")
    with pytest.raises(ValueError):
        ModelConfig(base_url="https://user:password@example.test", model="test")


def test_scoped_environment_restores_previous_values(monkeypatch):
    monkeypatch.setenv("MECHCAL_QC_CONFIG", "old")
    with pytest.raises(RuntimeError), scoped_environment({"MECHCAL_QC_CONFIG": "new"}):
        raise RuntimeError("test")
    import os
    assert os.environ["MECHCAL_QC_CONFIG"] == "old"


def test_doctor_makes_no_model_calls_and_missing_tools_are_explicit(monkeypatch):
    monkeypatch.setattr(launch, "NativeCommandClient", lambda *_: pytest.fail("No model calls allowed"))
    config = minimal_config()
    report = launch.doctor(config)
    assert report["ready"] and report["model_calls"] == 0
    config.runtime["enable_amesp"] = True
    assert not launch.doctor(config)["ready"]


def test_bounded_example_and_disabled_scientific_gates():
    path = Path(__file__).resolve().parents[1] / "configs/deepseek.example.toml"
    config = load_deployment(path)
    runtime = config.orchestrator_config(Path("out"))
    assert runtime.max_exploration_rounds == 30 and not runtime.unlimited_exploration
    assert not config.qc.allow_amesp_soc and not config.qc.allow_excited_scan


def test_run_uses_supplied_smiles_not_benchmark_and_does_not_judge(tmp_path, monkeypatch):
    key = "fake-key-for-test-only"
    monkeypatch.setenv("DEEPSEEK_API_KEY", key)
    seen = {}

    class FakeRun:
        def run(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(status="finalized", final_answer=None)

    def factory(runtime, settings):
        assert settings.api_key == key
        assert runtime.run_base_dir == tmp_path / "result/runs"
        return FakeRun()

    monkeypatch.setattr(launch, "create_orchestrator", factory)
    result = launch.run(minimal_config(), smiles="CCO", output=tmp_path / "result", case_id="demo")
    assert seen["smiles"] == "CCO" and seen["case_id"] == "demo"
    assert not result["judge_called"]
    assert key not in (tmp_path / "result/manifest.json").read_text()
    assert result["status"] == "finalized"
    with pytest.raises(FileExistsError):
        launch.run(minimal_config(), smiles="CCO", output=tmp_path / "result", case_id="demo")


def test_failures_are_recorded_without_secret_and_not_scored(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret")
    def fail(*_):
        raise RuntimeError("failed using test-secret")
    monkeypatch.setattr(launch, "create_orchestrator", fail)
    with pytest.raises(RuntimeError):
        launch.run(minimal_config(), smiles="CCO", output=tmp_path / "result", case_id="demo")
    status = (tmp_path / "result/status.json").read_text()
    assert "test-secret" not in status
    assert json.loads(status)["status"] == "interrupted"
    assert not (tmp_path / "result/summary.json").exists()


@pytest.mark.parametrize("case_id", ["../escape", "/absolute", "a/b", "..", ""])
def test_unsafe_case_ids_are_rejected_before_output(tmp_path, case_id):
    with pytest.raises(ValueError, match="case-id"):
        launch.run(minimal_config(), smiles="CCO", output=tmp_path / "out", case_id=case_id)
    assert not (tmp_path / "out").exists()


def test_invalid_config_does_not_echo_inline_secret(tmp_path, monkeypatch, capsys):
    config = tmp_path / "bad.toml"
    config.write_text('[model]\nbase_url="https://example.test"\nmodel="test"\n'
                      'api_key="must-not-be-printed"\n')
    monkeypatch.setattr("sys.argv", ["mechcal", "--config", str(config), "doctor"])
    assert launch.main() == 1
    assert "must-not-be-printed" not in capsys.readouterr().err


def test_postprocessing_and_spin_route_threads_are_separate_settings(tmp_path, monkeypatch):
    from mechcal.cli.deployment import environment
    from mechcal.tools.hole_electron import SpatialAnalysisError, run_spatial
    config = minimal_config()
    config.tools.multiwfn_threads = 16
    config.qc.pyscf_states_threads, config.qc.pyscf_states_max_memory_mb = 32, 16000
    env = environment(config, config.orchestrator_config(tmp_path), tmp_path / "qc.json")
    assert env["MECHCAL_MULTIWFN_THREADS"] == "16" and config.qc.threads == 4   # [qc] threads untouched
    monkeypatch.setenv("MECHCAL_MULTIWFN_THREADS", "0")
    with pytest.raises(SpatialAnalysisError) as exc:
        run_spatial(None, tmp_path / "out")
    assert exc.value.code == "capability_unsupported"
