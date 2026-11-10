"""Both execution paths retain producer facts, nested failure logs and source scope."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import sys

import pytest
from jsonschema import Draft202012Validator

from mechcal.agents.command_tools import tool_contracts
from mechcal.runtime.command_loop import _apply, _comment, _execute
from mechcal.runtime.command_memory import resolve_pointer
from mechcal.runtime.command_operations import OperationRejected, protocol, put, record_index
from mechcal.runtime.command_review import review_execution_context
from mechcal.runtime import worker_tasks as tasks
from mechcal.schemas import CaseRun
from mechcal.schemas.tool_arguments import PlannedTorsionArgs, TorsionArgs
from mechcal.schemas.worker_tasks import ReadWorkerLog
from mechcal.tools.qc_logs import DIAGNOSTIC_BYTES, failure_diagnostics, registered_failure_logs
from mechcal.tools.qc_process import QCConfig, QCError, QCJob
from test_command_protocol import ScriptClient, command, computed, orchestrator, store
from test_reviewer_execution_context import execution_case, capture_review, OTHER
from test_worker_tasks import setup, start


def nested_case():
    c = execution_case()
    results = deepcopy(protocol(c)["results"])
    results[OTHER]["artifact_updates"] = [{"artifact_id": "bundle:actual", "kind": "baseline_bundle",
        "status": "available", "metadata": {"bundle": {"schema_version": "2.0.0",
            "record_type": "MicroscopicArtifactBundle", "geometry_source": "s0_optimized_geometry"}}}]
    return put(c, results=results)


def test_nested_bundle_is_read_from_its_actual_pointer_not_inferred_from_input():
    c = nested_case()
    before = c.model_dump_json()
    context = review_execution_context(c, {OTHER}, record_index(c).available)[0]
    geometry = context["reported_geometry"]
    assert geometry["path"] == "/artifact_updates/0/metadata/bundle"
    assert geometry["values"] == {"geometry_source": "s0_optimized_geometry"}
    assert geometry["values"].items() <= resolve_pointer(protocol(c)["results"][OTHER], geometry["path"]).items()
    assert context["geometry_records"] == [geometry]
    assert c.model_dump_json() == before
    restored = CaseRun.model_validate_json(before)
    assert review_execution_context(restored, {OTHER}, record_index(restored).available) == [context]


def test_multiple_or_conflicting_output_geometries_are_not_silently_merged():
    c = nested_case()
    results = deepcopy(protocol(c)["results"])
    results[OTHER]["structured_results"]["geometry_source"] = "prepared_geometry"
    c = put(c, results=results)
    context = review_execution_context(c, {OTHER}, record_index(c).available)[0]
    assert context["reported_geometry"] is None
    assert [r["values"]["geometry_source"] for r in context["geometry_records"]] == [
        "prepared_geometry", "s0_optimized_geometry"]


@pytest.mark.parametrize("native", [False, True])
def test_direct_worker_and_parent_receive_same_receipt_without_scientific_mutation(tmp_path, native):
    from test_native_commands import NativeFake
    o, c = orchestrator(tmp_path), computed()
    o.config = replace(o.config, command_native_tools=native)
    s = store(tmp_path, c)
    c = _execute(o, _apply(o, c, command(), s), s)
    p = protocol(c)
    rid = p["pending_result"]
    results = deepcopy(p["results"])
    results[rid]["artifact_updates"] = protocol(nested_case())["results"][OTHER]["artifact_updates"]
    c = put(c, results=results)
    expected = review_execution_context(c, {rid}, record_index(c).available)
    writer = o.deps.workers["macro"].report_writer
    writer.enable_llm = True
    writer.llm_client = (NativeFake([{"request_id": "prose", "complete": True, "calls": [],
                                    "text": "Interpretation, not a new calculation."}]) if native
                         else ScriptClient(["Interpretation, not a new calculation."]))
    after = _comment(o, c, s)
    assert protocol(after)["outcome"]["execution_context"] == expected
    from mechcal.runtime.command_compact import expand_text
    payload = (writer.llm_client.requests[0]["payload"] if native else writer.llm_client.calls[0]["payload"])
    assert expand_text(payload)["execution_context"] == expected
    for key in ("results", "actions", "ranking", "rounds", "revision"):
        assert protocol(after)[key] == protocol(c)[key]


def test_reviewer_sees_nested_geometry_even_with_legacy_wrong_worker_prose(tmp_path):
    c = put(nested_case(), explanations={OTHER: {"status": "available", "text": "Uses prepared geometry.", "refs": [OTHER]}})
    payload = capture_review(tmp_path, c, refs=[OTHER])
    assert payload["execution_context"][0]["reported_geometry"]["values"]["geometry_source"] == "s0_optimized_geometry"
    assert "prepared geometry" in payload["worker_interpretations"][0]["text_preview"]
    assert not payload["worker_interpretations"][0]["scientific_approval"]


def test_disabled_explanation_does_not_drop_runtime_execution_receipt(tmp_path):
    o, c = orchestrator(tmp_path), computed()
    s = store(tmp_path, c)
    c = _execute(o, _apply(o, c, command(), s), s)
    before = protocol(c)["outcome"]["execution_context"]
    assert before and not o.deps.workers["macro"].report_writer.enable_llm
    c = _comment(o, c, s)
    assert protocol(c)["outcome"]["execution_context"] == before


def failed_result(root, log, **step):
    return {"status": "failed", "failure": {"details": {"workdir": str(root)}},
            "raw_results": {"steps": [{"log": str(log), "returncode": 1, **step}]}}


def test_real_subprocess_failure_saves_bounded_diagnostic_and_does_not_retry(tmp_path):
    job = QCJob(tmp_path / "job", QCConfig(), 10)
    message = "Increase the number of excited states to solve for (NrOfExcitations)."
    with pytest.raises(QCError, match="Exit 1"):
        job.execute([sys.executable, "-c", "print('x' * 8000); print(" + repr(message) + "); raise SystemExit(1)"],
                    job.workdir / "soc_source" / "spectra")
    assert len(job.records) == 1
    record = job.records[0]
    excerpt = record["diagnostic"]
    assert excerpt["truncated"] and message in excerpt["text"]
    assert len(excerpt["text"].encode()) <= DIAGNOSTIC_BYTES
    raw = (job.workdir / "soc_source/spectra/process.log").read_bytes()
    assert hashlib.sha256(raw[excerpt["offset"]:]).hexdigest() == excerpt["chunk_sha256"]
    result = failed_result(job.workdir, record["log"], diagnostic=excerpt)
    assert failure_diagnostics(result)[0]["saved_in_record"]
    assert not failure_diagnostics(result)[0]["scientific_evidence"]
    # The persisted diagnostic survives deletion/unavailability of the raw file.
    (job.workdir / "soc_source/spectra/process.log").unlink()
    assert failure_diagnostics(result)[0]["excerpt"] == excerpt


def test_nested_registered_log_is_readable_but_unregistered_neighbors_are_not(tmp_path):
    o, c, s, contract = setup(tmp_path)
    root = tmp_path / "job"
    nested = root / "soc_source/spectra"
    nested.mkdir(parents=True)
    log = nested / "process.log"
    log.write_text("SCC converged. Excited-state solver reached maximum iterations.")
    (nested / "private.txt").write_text("Not registered")
    c = put(c, results={"failure": failed_result(root, log)})
    contract.input_refs.append("failure")
    c = start(o, c, s, contract)
    task = tasks.state(c)
    assert tasks.log_files(c, task)["failure"] == {"soc_source/spectra/process.log": log}
    read = tasks.read_log(c, task, ReadWorkerLog(record_id="failure", file_key="soc_source/spectra/process.log"))
    assert read["text"] == log.read_text()
    for ref, key in [("failure", "soc_source/spectra/private.txt"), ("another-case", "soc_source/spectra/process.log")]:
        with pytest.raises(OperationRejected):
            tasks.read_log(c, task, ReadWorkerLog(record_id=ref, file_key=key))
    available = record_index(c).available
    context = review_execution_context(c, {"failure"}, available)[0]
    assert "SCC converged" in context["diagnostics"][0]["excerpt"]["text"]
    assert not context["diagnostics"][0]["saved_in_record"]


def test_log_paths_cannot_escape_job_or_follow_links(tmp_path):
    root = tmp_path / "job"
    root.mkdir()
    outside = tmp_path / "secret.log"
    outside.write_text("Secret")
    link = root / "link.log"
    link.symlink_to(outside)
    for path in (outside, root / "../secret.log", link):
        result = failed_result(root, path)
        assert registered_failure_logs(result) == {}
        assert failure_diagnostics(result) == []


def test_partial_routes_keep_registered_logs_and_bound_diagnostic_delivery(tmp_path):
    o, c, s, contract = setup(tmp_path)
    log = tmp_path / "partial" / "step.log"
    log.parent.mkdir()
    log.write_text("Failed member; successful members retained.")
    result = {"status": "partial", "failure": {"kind": "partial_evidence", "details": {}},
              "raw_results": {"workdir": str(log.parent), "steps": [{"log": str(log), "returncode": 1}] * 4}}
    c = put(c, results={"partial": result})
    contract.input_refs.append("partial")
    c = start(o, c, s, contract)
    assert tasks.log_files(c, tasks.state(c))["partial"]["step.log"] == log
    diagnostics = failure_diagnostics(result)
    assert len(diagnostics) == 2
    assert all(d["omitted_failed_steps"] == 2 for d in diagnostics)
    assert protocol(c)["results"]["partial"]["status"] == "partial"


def test_planner_scan_wire_preserves_executable_angle_constraint():
    from mechcal.capabilities.catalog import default_capability_registry
    registry = default_capability_registry()
    ids = ["microscopic.run_torsion_snapshots", "microscopic.run_torsion_brightness_coupling_scan"]
    assert ids
    for capability in ids:
        tools, _ = tool_contracts(registry, [capability])
        tool = next(t for t in tools if t["name"] == "compute_" + capability.replace(".", "_"))
        schema = tool["parameters"]
        Draft202012Validator.check_schema(schema)
        assert TorsionArgs.model_fields["angles"].description in str(schema)
    assert PlannedTorsionArgs.model_fields["angles"].description == TorsionArgs.model_fields["angles"].description
