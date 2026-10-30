import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from examples.run_command_case import build_config, main


def test_dry_run_needs_no_credentials_and_makes_no_output_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["example", "--smiles", "CCO", "--dry-run",
                                     "--output-dir", str(tmp_path / "output")])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["model_calls"] == result["judge_calls"] == 0
    assert result["command_native_tools"] and result["confidence_tracking"]
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("amesp", [False, True])
def test_configuration_uses_requested_limits_and_backend(amesp):
    config = build_config(SimpleNamespace(output_dir=Path("outputs/test"), max_computations=20, amesp=amesp))
    assert config.max_exploration_rounds == 20 and not config.unlimited_exploration
    assert config.command_native_tools and config.command_hypothesis_tracking
    assert config.enable_amesp == amesp


def test_nonpositive_computation_limit_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        build_config(SimpleNamespace(output_dir=Path("outputs/test"), max_computations=0, amesp=False))
