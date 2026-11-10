"""Offline applicability and backend-diagnostic checks; no scientific computation."""

import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from mechcal.capabilities import default_capability_registry
from mechcal.schemas import ArtifactRecord, MicroscopicArtifactBundle
from mechcal.schemas.tool_arguments import validate_tool_args
from mechcal.tools.amesp_baseline import AmespBaselineError, AmespBaselineRunner
from mechcal.tools.amesp_diagnostics import backend_stop, basis_precondition
from mechcal.tools.amesp_microscopic import (
    MICROSCOPIC_ROUTE_PROFILES, AmespMicroscopicRunner, MicroscopicRunOptions,
    microscopic_delivery_contract,
)
from mechcal.tools.local_micro import LocalMicroscopicTool
from test_runtime_failure_paths import case_with_ranking
from test_tool_parameters import request


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No external process or network in these tests")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.mark.parametrize("symbol", ["Cs", "Ba", "Pt", "Au", "U"])
def test_basis_guard_is_an_element_range_not_a_case_rule(symbol):
    error = basis_precondition(["B3LYP", "STO-3G"], ["C", "H", symbol])
    assert error["code"] == "capability_unsupported"
    assert error["details"]["unsupported_elements"] == [symbol]
    assert error["details"]["retry_same_configuration"] is False


@pytest.mark.parametrize("symbols", [["H"], ["C", "N", "O"], ["I", "Xe"]])
def test_known_range_not_rejected(symbols):
    assert basis_precondition(["b3lyp", "sto-3g"], symbols) is None


def test_unrelated_methods_are_not_declared_unsupported_or_validated():
    assert basis_precondition(["aTB1", "force"], ["Au"]) is None
    assert basis_precondition(["b3lyp", "unknown-basis"], ["Au"]) is None


def step_args(tmp_path, symbols=None):
    symbols = symbols or ["H"]
    return dict(step_id="fixture_step", label="fixture", workdir=tmp_path,
                keywords=["b3lyp", "sto-3g"], block_lines=[], charge=0, multiplicity=1,
                symbols=symbols, coordinates=[[0., 0., float(i)] for i in range(len(symbols))])


def test_preflight_never_starts_backend_or_creates_fake_outputs(tmp_path):
    runner = AmespBaselineRunner(amesp_bin=Path("/offline/amesp"), npara=4)
    with pytest.raises(AmespBaselineError) as caught:
        runner._run_step(**step_args(tmp_path, ["Ba"]))
    assert caught.value.code == "capability_unsupported"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("channel", ["aop", "stdout", "stderr"])
def test_exit_zero_stop_retains_reason_and_raw_files(tmp_path, monkeypatch, channel):
    text = "  Stop : Only support H--Xe for sto-3g\n"
    calls = []
    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if channel == "aop":
            (Path(kwargs["cwd"]) / cmd[2]).write_text(text)
        return SimpleNamespace(returncode=0, stdout=text if channel == "stdout" else "",
                               stderr=text if channel == "stderr" else "")
    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = AmespBaselineRunner(amesp_bin=Path("/offline/amesp"), npara=4)
    with pytest.raises(AmespBaselineError) as caught:
        runner._run_step(**step_args(tmp_path))
    error = caught.value
    assert error.code == "capability_unsupported" and len(calls) == 1
    assert error.details["exit_code"] == 0 and error.details["diagnostic_source"] == channel
    assert error.details["backend_stop_reason"] == "Only support H--Xe for sto-3g"
    assert Path(error.file_paths[channel]).read_text() == text
    assert "! b3lyp sto-3g" in Path(error.file_paths["aip"]).read_text()


def test_generic_stop_is_not_invented_as_basis_failure():
    error = backend_stop({"aop": "Stop : SCF did not converge\n"})
    assert error["code"] == "runtime_failed"
    assert error["details"]["reason"] == "backend_stop"


def test_explicit_basis_stop_is_not_retried_as_native_crash(tmp_path, monkeypatch):
    calls = []
    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=-11, stdout="Stop : Only support H--Xe for sto-3g\n", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = AmespBaselineRunner(amesp_bin=Path("/offline/amesp"), npara=4)
    with pytest.raises(AmespBaselineError) as caught:
        runner._run_step(**step_args(tmp_path))
    assert caught.value.code == "capability_unsupported" and len(calls) == 1


@pytest.mark.parametrize("stop", [False, True])
def test_stop_dominates_normal_marker_but_normal_success_unchanged(tmp_path, monkeypatch, stop):
    def fake_run(cmd, **kwargs):
        (tmp_path / cmd[2]).write_text("Normal termination of Amesp!\n")
        return SimpleNamespace(returncode=0, stdout="Stop : Invalid option\n" if stop else "", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = AmespBaselineRunner(amesp_bin=Path("/offline/amesp"))
    if stop:
        with pytest.raises(AmespBaselineError, match="Invalid option"):
            runner._run_step(**step_args(tmp_path))
    else:
        assert runner._run_step(**step_args(tmp_path)).terminated_normally


def case_with_bundle(tmp_path, symbol):
    case = case_with_ranking()
    xyz = tmp_path / "fixture.xyz"
    xyz.write_text(f"1\nSynthetic applicability fixture only\n{symbol} 0 0 0\n")
    bundle = MicroscopicArtifactBundle(bundle_id="fixture", capability_name="run_baseline_bundle",
                                      status="available", geometry_source="synthetic_fixture",
                                      file_paths={"analysis_geometry_xyz": str(xyz)})
    artifact = ArtifactRecord(artifact_id="bundle:fixture", kind="amesp_baseline_bundle",
                              created_by_round="R001", created_by_agent="microscopic",
                              metadata={"bundle": bundle.model_dump(mode="json")})
    case.artifact_manifest.artifacts.append(artifact)
    return case


@pytest.mark.parametrize("route", ["run_targeted_localized_orbital_analysis",
                                   "run_targeted_natural_orbital_analysis", "run_ris_state_characterization"])
def test_discovery_and_execution_share_unsupported_condition(tmp_path, route):
    case = case_with_bundle(tmp_path, "Ba")
    registry = default_capability_registry()
    readiness = registry.readiness("microscopic." + route, case.artifact_manifest, amesp_enabled=True)
    assert readiness["status"] == "disabled" and "Ba" in readiness["reason"]
    result = LocalMicroscopicTool(enable_amesp=True).run(request(route, {}), case)
    assert result.status == "unsupported" and result.failure.kind == "capability_unsupported"
    assert not result.failure.recoverable
    assert result.failure.details["unsupported_elements"] == ["Ba"]
    assert result.structured_results["boundary_only"] is True


@pytest.mark.parametrize("missing", ["geometry", "manifest"])
def test_missing_source_is_missing_input_not_basis_failure(tmp_path, missing):
    route = "run_targeted_localized_orbital_analysis"
    case = case_with_bundle(tmp_path, "H")
    if missing == "geometry":
        (tmp_path / "fixture.xyz").unlink()
    else:
        artifact = case.artifact_manifest.artifacts[0]
        artifact.metadata = {}
        artifact.file_paths = {"manifest": str(tmp_path / "missing_manifest.json")}
    registry = default_capability_registry()
    assert registry.readiness("microscopic." + route, case.artifact_manifest,
                              amesp_enabled=True)["status"] == "missing_input"
    result = LocalMicroscopicTool(enable_amesp=True).run(request(route, {}), case)
    assert result.status == "precondition_missing"


def test_manifest_permission_failure_still_propagates(tmp_path, monkeypatch):
    case = case_with_bundle(tmp_path, "H")
    artifact = case.artifact_manifest.artifacts[0]
    artifact.metadata = {}
    artifact.file_paths = {"manifest": str(tmp_path / "protected_manifest.json")}
    def denied(*args, **kwargs):
        raise PermissionError("Offline manifest permission fixture")
    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(PermissionError):
        default_capability_registry().readiness(
            "microscopic.run_targeted_localized_orbital_analysis", case.artifact_manifest,
            amesp_enabled=True)


def test_basis_override_still_rejected_and_documented():
    route = "run_targeted_localized_orbital_analysis"
    with pytest.raises(ValueError):
        validate_tool_args(route, {"basis": "replacement"})
    assert "no basis or method override" in " ".join(microscopic_delivery_contract(route)["backend_constraints"])


def test_unsupported_vertical_does_not_trigger_method_fallback(tmp_path, monkeypatch):
    runner = AmespMicroscopicRunner()
    calls = []
    def unsupported(**kwargs):
        calls.append(kwargs["mode"])
        raise AmespBaselineError("capability_unsupported", "Backend basis boundary")
    monkeypatch.setattr(runner, "_run_vertical", unsupported)
    with pytest.raises(AmespBaselineError):
        runner._optional_vertical(MICROSCOPIC_ROUTE_PROFILES["run_ris_state_characterization"],
                                  label="fixture", workdir=tmp_path, charge=0, multiplicity=1,
                                  symbols=["Ba"], coordinates=[[0, 0, 0]], run_options=MicroscopicRunOptions())
    assert calls == ["tda_ris"]


@pytest.mark.parametrize("code,status,kind,recoverable", [
    ("capability_unsupported", "unsupported", "capability_unsupported", False),
    ("precondition_missing", "precondition_missing", "precondition_missing", True),
    ("runtime_failed", "failed", "runtime_failed", True),
])
def test_backend_error_preserves_category_in_planner_result(tmp_path, code, status, kind, recoverable):
    class FailedRunner:
        def run(self, **kwargs):
            raise AmespBaselineError(code, "Exact backend reason", file_paths={"aop": "original.aop"},
                                     details={"backend_stop_reason": "Exact backend reason"})
    result = LocalMicroscopicTool(enable_amesp=True, amesp_runner=FailedRunner()).run(
        request("run_targeted_localized_orbital_analysis", {}), case_with_bundle(tmp_path, "H"))
    assert (result.status, result.failure.kind, result.failure.recoverable) == (status, kind, recoverable)
    assert result.failure.details["file_paths"] == {"aop": "original.aop"}
    assert result.failure.details["backend_stop_reason"] == "Exact backend reason"
