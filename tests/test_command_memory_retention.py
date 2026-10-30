"""Retention policies on synthetic observations, not benchmark-specific labels."""

from copy import deepcopy

from mechcal.runtime import command_memory as memory
from mechcal.runtime.command_loop import _apply, planner_payload
from mechcal.runtime.command_memory import remember_reads, remember_report, remember_values, result_report, working_reads
from mechcal.runtime.command_operations import protocol, put, read_records, records
from mechcal.schemas.commands import ReadEvidence
from test_command_protocol import case, orchestrator, store


def observation(value):
    metrics = {"value": value, "scope": "synthetic proxy, not experimental validation"}
    return {"status": "success", "capability_id": "fixture.tool", "structured_results": {
        "metrics": metrics, "structural_prior": {"fixture": metrics}},
        "evidence_units": [{"context": "fixture gas phase", "limits": ["Proxy only."]}]}


def remembered(sources):
    handles = []
    for ref, raw in sources.items():
        handles = remember_report(handles, result_report(ref, raw))
    return handles


def test_failure_log_and_explanation_cannot_evict_six_sources():
    sources = {f"r{i}": observation(i) for i in range(6)}
    handles = remembered(sources)
    sources["failed"] = {"status": "failed", "summary": "Failure." * 100,
        "failure": {str(i): "diagnostic " * 100 for i in range(15)},
        "structured_results": {"runtime_status": "failed"}}
    handles = remember_report(handles, result_report("failed", sources["failed"]))
    sources["explanation:failed"] = {"source_result_id": "failed", "status": "available", "text": "Comment." * 250}
    handles = remember_values(handles, [("explanation:failed", "/text", sources["explanation:failed"]["text"], None)])
    before = deepcopy((sources, handles))
    kept = working_reads(handles, sources)["findings"]
    observations = [f for f in kept if f["path"] == "/structured_results/metrics"]
    assert {f["ref"] for f in observations} == {f"r{i}" for i in range(6)}
    assert not any(f["path"] == "/structured_results/structural_prior" for f in kept)
    assert all(f["source_context"] == [
        {"path": "/evidence_units/0/context", "data": "fixture gas phase"},
        {"path": "/evidence_units/0/limits", "data": ["Proxy only."]}] for f in observations)
    assert memory.json_bytes(kept) <= memory.WORKING_BYTES
    assert (sources, handles) == before


def test_many_fields_in_one_result_do_not_displace_other_sources(monkeypatch):
    monkeypatch.setattr(memory, "WORKING_BYTES", 1800)
    sources = {"old": observation(7), "new": {
        "status": "success", "structured_results": {f"measurement_{i}": {"value": i} for i in range(30)}}}
    kept = working_reads(remembered(sources), sources)["findings"]
    assert {f["ref"] for f in kept[:2]} == {"old", "new"}


def test_explicit_read_priority_survives_later_report_delivery():
    raw = {"status": "success", "structured_results": {"older": 1, "newer": 2}}
    query = ReadEvidence(kind="read_evidence", records=[{"ref": "r", "paths": ["/structured_results/older"]}])
    handles = remember_reads([], query.records, [{"data": {"/structured_results/older": 1}}])
    handles = remember_report(handles, result_report("r", raw))
    assert working_reads(handles, {"r": raw})["findings"][0]["path"] == "/structured_results/older"


def test_equal_values_at_different_paths_or_sources_are_not_deduplicated():
    raw = {"status": "success", "structured_results": {
        "state1": {"energy": 1}, "state2": {"energy": 1}, "a": 0, "b": 0}}
    sources = {"r1": raw, "r2": deepcopy(raw)}
    kept = working_reads(remembered(sources), sources)["findings"]
    assert len(kept) == 8


def test_latest_metric_delivery_also_excludes_its_exact_wrapper():
    raw = observation(1)
    handles = remembered({"r": raw})
    latest = [e for e in handles if e["path"] == "/structured_results/metrics"]
    kept = working_reads(handles, {"r": raw}, latest_rows=latest)["findings"]
    assert not any(f["path"] in {"/structured_results/metrics", "/structured_results/structural_prior"} for f in kept)


def test_context_too_large_defers_values_together_not_naked_numbers(monkeypatch):
    monkeypatch.setattr(memory, "WORKING_BYTES", 700)
    raw = observation(1)
    raw["structured_results"]["method_provenance"] = "method details " * 100
    result = working_reads(remembered({"r": raw}), {"r": raw})
    assert not any(f["path"] == "/structured_results/metrics" for f in result["findings"])
    assert any(f["path"] == "/structured_results/metrics" for f in result["other_reads"])


def test_active_evidence_unit_resolves_to_its_source_without_hiding_alternatives(monkeypatch):
    monkeypatch.setattr(memory, "WORKING_BYTES", 650)
    sources = {"older": observation(1), "newer": observation(2)}
    handles = remembered(sources)
    sources["older:E001"] = {"source_report_id": "older"}
    result = working_reads(handles, sources, active_refs=["older:E001"])
    assert result["findings"][0]["ref"] == "older"
    assert any(e["ref"] == "newer" for e in result["other_reads"])


def test_candidate_and_limitation_references_participate_in_retention(tmp_path, monkeypatch):
    seen = {}
    from mechcal.runtime import command_loop
    original = command_loop.working_reads

    def capture(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(command_loop, "working_reads", capture)
    c = put(case(), candidates={"RIM_RIR_RIV": {
        "status": "unexamined", "refs": ["r"], "limitation_refs": ["failed"], "note": "fixture"}})
    planner_payload(orchestrator(tmp_path), c)
    assert seen["active_refs"] == {"r", "failed"}


def test_retrieval_remains_allowed_and_checkpoint_science_is_unchanged(tmp_path):
    c = put(case(), results={"r": observation(3)}, read_memory=remembered({"r": observation(3)}))
    o, s = orchestrator(tmp_path), store(tmp_path, c)
    query = ReadEvidence(kind="read_evidence", records=[{"ref": "r", "paths": ["/structured_results/metrics"]}])
    for _ in range(3):
        c = _apply(o, c, query, s)
        assert protocol(c)["outcome"]["status"] == "read"
    assert read_records(c, query)[0]["data"]["/structured_results/metrics"]["value"] == 3
    assert protocol(c)["rounds"] == protocol(c)["revision"] == 0
    assert protocol(c)["results"] == {"r": observation(3)}
    assert not working_reads(protocol(c)["read_memory"], {})["findings"]
    assert working_reads(protocol(c)["read_memory"], records(c))["findings"]
