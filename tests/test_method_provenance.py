"""Synthetic process fixtures test provenance, not molecular predictions."""

import hashlib
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from mechcal.capabilities import default_capability_registry
from mechcal.runtime.questions import public_payload
from mechcal.schemas import ArtifactRecord, MicroscopicArtifactBundle, MicroscopicStepRecord
from mechcal.tools.amesp_baseline import AmespBaselineError, AmespBaselineRunner, AmespStepResult
from mechcal.tools.amesp_microscopic import (
    AmespMicroscopicRunner, MicroscopicRunOptions, MICROSCOPIC_ROUTE_PROFILES,
    PreparedGeometryMember, _artifact_bundle_members, _recorded_method_fallback, _step_record,
)
from mechcal.tools.local_micro import LocalMicroscopicTool
from test_amesp_basis_guard import case_with_bundle
from test_tool_parameters import request


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No live computation or network in method provenance tests")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def runner_with_fixture(tmp_path, monkeypatch, fail_keywords=()):
    calls = []
    def fake_run(cmd, **kwargs):
        aip = Path(kwargs["cwd"]) / cmd[1]
        text = aip.read_text()
        keywords = next(line[1:].strip() for line in text.splitlines() if line.startswith("!"))
        calls.append((aip, keywords))
        body = ("Stop : Synthetic failure for " + keywords + "\n" if keywords in fail_keywords
                else "Final Energy: -1.000000\nNormal termination of Amesp!\n")
        (Path(kwargs["cwd"]) / cmd[2]).write_text(body)
        return SimpleNamespace(returncode=0, stdout=body, stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    binary = tmp_path / "never_executed_binary"
    binary.touch()
    return AmespMicroscopicRunner(AmespBaselineRunner(amesp_bin=binary)), calls


def vertical_kwargs(tmp_path, label="misleading_atb_fallback_name"):
    return dict(label=label, workdir=tmp_path, charge=0, multiplicity=1,
                symbols=["C", "H"], coordinates=[[0., 0., 0.], [0., 0., 1.]],
                run_options=MicroscopicRunOptions())


@pytest.mark.parametrize("mode,keywords", [
    ("tda_ris", ["b3lyp", "sto-3g", "tda-ris"]),
    ("tda_b3lyp", ["b3lyp", "sto-3g", "tda"]),
    ("tda_atb_excdip", ["aTB1", "tda"]),
])
def test_executed_input_not_filename_or_requested_label(tmp_path, monkeypatch, mode, keywords):
    runner, calls = runner_with_fixture(tmp_path, monkeypatch)
    result = runner._run_vertical(mode=mode, **vertical_kwargs(tmp_path))
    record = _step_record(result, "incorrect-request-only-label")
    provenance = record.method_provenance
    assert provenance.source == "submitted_input"
    assert provenance.requested_method_label == mode
    assert record.method_label == provenance.executed_method_label == " ".join(keywords)
    assert provenance.input_keywords == keywords
    assert provenance.input_sha256 == hashlib.sha256(result.aip_path.read_bytes()).hexdigest()
    assert provenance.input_blocks and not provenance.fallbacks and len(calls) == 1
    assert result.to_json()["method_provenance"]["input_sha256"] == provenance.input_sha256


def test_successful_fallback_keeps_both_methods_reason_and_failed_input(tmp_path, monkeypatch):
    runner, calls = runner_with_fixture(tmp_path, monkeypatch, {"b3lyp sto-3g tda-ris"})
    result = runner._optional_vertical(MICROSCOPIC_ROUTE_PROFILES["run_ris_state_characterization"],
                                      **vertical_kwargs(tmp_path, label="opaque"))
    record = _step_record(result, "tda_ris")
    provenance = record.method_provenance
    assert record.method_label == provenance.executed_method_label == "aTB1 tda"
    assert provenance.requested_method_label == "tda_ris"
    assert [words for _, words in calls] == ["b3lyp sto-3g tda-ris", "aTB1 tda"]
    previous = provenance.fallbacks[0]
    assert previous["failure_code"] == "runtime_failed" and "Synthetic failure" in previous["reason"]
    failed_aip = Path(previous["failed_file_paths"]["aip"])
    assert failed_aip.exists() and failed_aip != result.aip_path
    assert previous["failure_details"]["method_provenance"]["input_sha256"] == hashlib.sha256(failed_aip.read_bytes()).hexdigest()
    assert "Stop :" in Path(previous["failed_file_paths"]["aop"]).read_text()


def test_failed_fallback_does_not_erase_original_attempt(tmp_path, monkeypatch):
    runner, calls = runner_with_fixture(tmp_path, monkeypatch, {"b3lyp sto-3g tda-ris", "aTB1 tda"})
    with pytest.raises(AmespBaselineError) as caught:
        runner._optional_vertical(MICROSCOPIC_ROUTE_PROFILES["run_ris_state_characterization"],
                                  **vertical_kwargs(tmp_path))
    failure = caught.value
    assert failure.details["method_provenance"]["executed_method_label"] == "aTB1 tda"
    assert failure.details["method_fallbacks"][0]["requested_method_label"] == "tda_ris"
    first = failure.details["method_fallbacks"][0]
    assert first["failure_details"]["method_provenance"]["input_keywords"] == ["b3lyp", "sto-3g", "tda-ris"]
    assert failure.details["method_provenance"]["input_keywords"] == ["aTB1", "tda"]
    for provenance, paths in ((first["failure_details"]["method_provenance"], first["failed_file_paths"]),
                              (failure.details["method_provenance"], failure.file_paths)):
        assert provenance["input_sha256"] == hashlib.sha256(Path(paths["aip"]).read_bytes()).hexdigest()
        assert "Synthetic failure" in Path(paths["aop"]).read_text()
    assert len(calls) == 2
    assert all(aip.exists() for aip, _ in calls)


def test_timeout_retains_submitted_configuration(tmp_path, monkeypatch):
    def timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"], output="synthetic partial stdout")
    monkeypatch.setattr(subprocess, "run", timeout)
    runner = AmespMicroscopicRunner(AmespBaselineRunner(amesp_bin=Path("/unused")))
    with pytest.raises(AmespBaselineError) as caught:
        runner._run_vertical(mode="tda_ris", **vertical_kwargs(tmp_path))
    assert caught.value.details["method_provenance"]["input_keywords"] == ["b3lyp", "sto-3g", "tda-ris"]
    assert Path(caught.value.file_paths["stdout"]).read_text() == "synthetic partial stdout"


def legacy_result(tmp_path):
    return AmespStepResult("legacy", tmp_path / "tda_ris.aip", tmp_path / "aop",
                           tmp_path / "stdout", tmp_path / "stderr", 0, True, 0.)


def test_unknown_actual_method_never_filled_from_request_or_name(tmp_path):
    record = _step_record(legacy_result(tmp_path), "tda_ris")
    assert record.method_label == record.method_provenance.executed_method_label == "unknown"
    assert record.method_provenance.source == "unknown"
    assert record.method_provenance.requested_method_label == "tda_ris"


def test_fallback_without_execution_metadata_stays_unknown(tmp_path):
    result = legacy_result(tmp_path)
    wrapped = _recorded_method_fallback(lambda: result, requested="tda_ris",
                                        failure=AmespBaselineError("runtime_failed", "Synthetic stop"))
    assert wrapped.method_provenance.executed_method_label == "unknown"
    assert wrapped.method_provenance.fallbacks[0]["reason"] == "Synthetic stop"
    assert result.method_provenance.fallbacks == []


def bundle_with_step(step):
    return MicroscopicArtifactBundle(bundle_id="fixture", capability_name="run_ris_state_characterization",
                                      status="partial", geometry_source="fixture",
                                      step_records=[step], parsed_observables={"fixture_value": 1.25})


def test_legacy_schema_reads_without_promoting_old_label(tmp_path):
    record = _step_record(legacy_result(tmp_path), "tda_ris").model_dump(mode="json")
    record.pop("method_provenance")
    record["method_label"] = "tda_ris"
    parsed = MicroscopicStepRecord.model_validate(record)
    assert parsed.method_label == "tda_ris"
    bundle = bundle_with_step(parsed)
    assert bundle.method_sources()[0]["executed_method_label"] == "unknown"
    members, _, _ = _artifact_bundle_members(None, bundle)
    assert members["members"][0]["method_label"] == "unknown"
    assert members["members"][0]["recorded_method_label"] == "tda_ris"


def test_tool_result_and_planner_observations_receive_method_sources(tmp_path, monkeypatch):
    runner, _ = runner_with_fixture(tmp_path, monkeypatch, {"b3lyp sto-3g tda-ris"})
    step = runner._optional_vertical(MICROSCOPIC_ROUTE_PROFILES["run_ris_state_characterization"],
                                    **vertical_kwargs(tmp_path))
    bundle = bundle_with_step(_step_record(step, "tda_ris"))
    class ReturnedBundle:
        def run(self, **kwargs):
            return SimpleNamespace(bundle=bundle, manifest_path=tmp_path / "fixture_manifest.json")
    case = case_with_bundle(tmp_path, "H")
    result = LocalMicroscopicTool(enable_amesp=True, amesp_runner=ReturnedBundle()).run(
        request("run_ris_state_characterization", {}), case)
    case.evidence_ledger.items.extend(result.evidence_units)
    packet = {"result_id": result.tool_result_id, "status": result.status,
              "evidence_refs": [e.evidence_id for e in result.evidence_units],
              "structured_results": result.structured_results, "failure": result.failure.model_dump(mode="json")}
    case.runtime["question_loop"] = {"pending_results": [packet]}
    payload = public_payload(case, default_capability_registry(), stage="assimilate")
    sources = payload["pending_results"][0]["structured_results"]["method_provenance"]
    assert sources[0]["requested_method_label"] == "tda_ris"
    assert sources[0]["executed_method_label"] == "aTB1 tda"
    assert sources[0]["fallbacks"]
    assert payload["observations"][0]["metrics"]["method_provenance"] == sources
    assert result.structured_results["parsed_observables"] == {"fixture_value": 1.25}
    assert len(result.evidence_units) == 1
    assert "method_provenance" not in result.evidence_units[0].metrics["computed_observables"]


def prepared(tmp_path):
    xyz = tmp_path / "source.xyz"
    xyz.write_text("1\nFixture\nC 0 0 0\n")
    return ArtifactRecord(artifact_id="fixture", kind="prepared_structure", created_by_round="R001",
                          created_by_agent="macro", file_paths={"xyz": str(xyz)})


def test_baseline_s0_fallback_and_s1_actual_method_labels(tmp_path, monkeypatch):
    runner, calls = runner_with_fixture(tmp_path, monkeypatch, {"aTB1 opt force"})
    result = runner._run_baseline(capability_name="run_baseline_bundle",
                                  profile=MICROSCOPIC_ROUTE_PROFILES["run_baseline_bundle"],
                                  source_artifact=prepared(tmp_path), case_id="fixture", round_id="R002",
                                  run_options=MicroscopicRunOptions())
    s0, s1 = result.bundle.step_records
    assert s0.method_label == "aTB1 force"
    assert s0.method_provenance.requested_method_label == "aTB1-opt-force"
    assert s0.method_provenance.fallbacks
    assert s1.method_label == "aTB1 tda" and not s1.method_provenance.fallbacks
    assert len(calls) == 3


def test_member_series_uses_executed_method_not_profile_or_member_label(tmp_path, monkeypatch):
    runner, _ = runner_with_fixture(tmp_path, monkeypatch)
    artifact = prepared(tmp_path)
    member = PreparedGeometryMember("m1", "tda_ris_misleading", ["C"], [[0., 0., 0.]],
                                    Path(artifact.file_paths["xyz"]), {})
    result = runner._run_geometry_member_series(
        capability_name="run_conformer_bundle", profile=MICROSCOPIC_ROUTE_PROFILES["run_conformer_bundle"],
        source_artifact=artifact, round_id="R002", label="fixture", workdir=tmp_path,
        charge=0, multiplicity=1, members=[member], source_xyz_path=member.xyz_path,
        generation_limitations=[], run_options=MicroscopicRunOptions())
    assert [s.method_label for s in result.bundle.step_records] == ["aTB1 force", "aTB1 tda"]


def test_targeted_route_records_actual_fallback_method(tmp_path, monkeypatch):
    runner, _ = runner_with_fixture(tmp_path, monkeypatch, {"b3lyp sto-3g tda-ris"})
    artifact = case_with_bundle(tmp_path, "H").artifact_manifest.artifacts[0]
    artifact.file_paths["workdir"] = str(tmp_path)
    result = runner._run_targeted(capability_name="run_ris_state_characterization",
                                  profile=MICROSCOPIC_ROUTE_PROFILES["run_ris_state_characterization"],
                                  source_artifact=artifact, case_id="fixture", round_id="R002",
                                  run_options=MicroscopicRunOptions())
    assert result.bundle.step_records[1].method_label == "aTB1 tda"
    assert result.bundle.step_records[1].method_provenance.requested_method_label == "tda_ris"
    assert result.bundle.step_records[1].method_provenance.fallbacks
