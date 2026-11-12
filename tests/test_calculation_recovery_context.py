import json
from types import SimpleNamespace

import pytest
from ase import Atoms
from pydantic import ValidationError

from mechcal.schemas.external_qc import BroadSpinArgs
from mechcal.schemas.tool_arguments import HoleElectronArgs
from mechcal.tools import pyscf_states, hole_electron
from mechcal.tools.qc_process import QCError
from test_external_qc import config, case, dispatch


def test_spin_timeout_returns_actual_stage_and_not_computed_triplet(tmp_path):
    class Job:
        workdir = tmp_path
        config = SimpleNamespace(threads=2, pyscf_python="python", pyscf_states_threads=32,
                                 pyscf_states_max_memory_mb=16000)
        max_wall_seconds = 420

        def timing(self):
            return {"total_wall_seconds": 420.1}

        def execute(self, argv, directory, threads=None):
            assert threads == 32                        # route-specific threads, not [qc] threads
            request = json.loads((directory / "request.json").read_text())
            assert request["threads"] == 32 and request["max_memory_mb"] == 16000
            (directory / "result.json").write_text(json.dumps({
                "schema_version": "pyscf_spin_states_v1", "request": request, "states": [],
                "scf_converged": True, "scf_seconds": 247.1, "stability_check_status": "not_completed"}))
            (directory / "result.log").write_text("SCF converged. Starting TDA, no states completed.")
            raise QCError("timeout", "Process killed.")

    with pytest.raises(QCError) as exc:
        pyscf_states.evaluate(Job(), Atoms("H2", positions=[[0, 0, 0], [0, 0, .74]]),
                              BroadSpinArgs(source_artifact_id="source", max_wall_seconds=420), 0)
    d = exc.value.details
    assert exc.value.code == "timeout"
    assert d["stage"] == "excited_states" and d["completed_stages"] == ["ground_state_scf"]
    assert d["completed_state_count"] == 0 and d["scientific_evidence"] is False
    assert d["scf_seconds"] == 247.1 and d["max_wall_seconds"] == 420
    assert "Starting TDA" in d["calculation_diagnostic"]["text"]


def test_spatial_explicit_longer_deadline_has_same_numerical_contract():
    args = HoleElectronArgs(max_wall_seconds=900)
    assert args.max_wall_seconds == 900
    assert HoleElectronArgs().max_wall_seconds == 300
    assert "tolerance" not in HoleElectronArgs.model_fields
    for seconds in [9, 1801]:
        with pytest.raises(ValidationError):
            HoleElectronArgs(max_wall_seconds=seconds)


def test_adapter_preserves_failure_diagnostics_without_computed_evidence(config, case, monkeypatch):
    import sys
    from mechcal.tools.external_qc import execute_external

    def fail(*args):
        raise QCError("timeout", "No states completed", details={"stage":"excited_states",
            "completed_stages":["ground_state_scf"], "completed_state_count":0, "scientific_evidence":False})
    monkeypatch.setattr(pyscf_states, "evaluate", fail)
    result = execute_external(dispatch("compare_spin_states_pyscf", max_wall_seconds=420), case,
        config=config.model_copy(update={"pyscf_python":sys.executable}))
    assert result.status == "failed"
    assert result.failure.details["stage"] == "excited_states"
    assert result.failure.details["max_wall_seconds"] == 420
    assert result.failure.details["tool_args"]["source_artifact_id"] == "prepared"
    assert not result.artifact_updates
    assert all(e.basis != "computed" for e in result.evidence_units)


def test_spatial_deployment_limit_is_explicit_and_never_silently_clamped(tmp_path, monkeypatch):
    monkeypatch.setenv("MECHCAL_SPATIAL_MAX_WALL_SECONDS", "600")
    monkeypatch.setattr(hole_electron, "backend_paths", lambda: pytest.fail("Must not execute"))
    with pytest.raises(hole_electron.SpatialAnalysisError) as exc:
        hole_electron.run_spatial(None, tmp_path, max_wall_seconds=900)
    assert exc.value.details == {"stage": "preflight", "requested_wall_seconds": 900,
                                 "effective_max_wall_seconds": 600}


@pytest.mark.parametrize("value", ["not-a-number", "0", "1801"])
def test_invalid_spatial_config_cannot_start_process(tmp_path, monkeypatch, value):
    monkeypatch.setenv("MECHCAL_SPATIAL_MAX_WALL_SECONDS", value)
    monkeypatch.setattr(hole_electron, "backend_paths", lambda: pytest.fail("Must not execute"))
    with pytest.raises(hole_electron.SpatialAnalysisError, match="configuration"):
        hole_electron.run_spatial(None, tmp_path)


def test_mcp_wait_honors_validated_spatial_deadline(config):
    from mechcal.tools import StdioMcpTransport
    transport = StdioMcpTransport()
    req = dispatch("analyze_hole_electron_from_bundle", max_wall_seconds=900)
    args = {"dispatch_request":req.model_dump(mode="json")}
    assert transport._call_timeout("run_microscopic", args) == 930
    args["dispatch_request"]["tool_args"]["max_wall_seconds"] = 1801
    assert transport._call_timeout("run_microscopic", args) == 120
