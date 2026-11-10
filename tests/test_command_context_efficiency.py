"""Context savings must not alter sources, review validity, or record ownership."""

from copy import deepcopy
from dataclasses import replace

import pytest

from mechcal.runtime import command_operations as ops
from mechcal.runtime.command_context import catalog_table, record_table, share_text, scoped_directory, scope_card
from mechcal.runtime.command_loop import _apply, _comment, planner_payload, save
from mechcal.runtime.command_memory import resolve_pointer
from mechcal.runtime.command_reads import read_selection_results
from mechcal.runtime.command_review import review_projection
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import Finalize, ReadEvidence
from test_command_protocol import case, judgment, orchestrator, store
from test_command_handoff import prose, unchanged_science
from test_native_commands import NativeFake


def expand(payload):
    def visit(value):
        if isinstance(value, dict) and set(value) == {"same_text_at"}:
            return resolve_pointer(payload, value["same_text_at"])
        if isinstance(value, dict):
            return {k: visit(v) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v) for v in value]
        return value
    return visit(payload)


def test_shared_index_built_once_for_planner_and_read_batch(tmp_path, monkeypatch):
    c = ops.revise(case(), judgment())
    calls, original = [], ops.observation_records
    monkeypatch.setattr(ops, "observation_records", lambda c: (calls.append(c), original(c))[1])
    planner_payload(orchestrator(tmp_path), c, native=True)
    assert len(calls) == 1
    calls.clear()
    read_selection_results(c, ReadEvidence(kind="read_evidence", records=[
        {"ref": "A1", "paths": ["/text", "/refs", "/label"]}, {"ref": "input:smiles"}]))
    assert len(calls) == 1


def test_index_rejects_other_case_and_new_snapshot():
    c = case()
    index = ops.RecordIndex(c)
    for other in (case(), ops.put(c, revision=1)):
        with pytest.raises(RuntimeError, match="snapshot"):
            ops.record_identity(other, "input:smiles", index=index)


def test_shared_index_preserves_exact_id_and_failure_rules():
    c = ops.put(case(), results={"failed:R1": {"status": "failed"}})
    index = ops.RecordIndex(c)
    assert ops.record_identity(c, "failed:R1", index=index)["citation_uses"] == ["limitation"]
    with pytest.raises(ops.OperationRejected, match="not scientific"):
        ops.check_refs(c, ["failed:R1"], ["refs"], scientific=True, index=index)
    with pytest.raises(ops.OperationRejected, match="not in this case"):
        ops.check_refs(c, ["other-case:R1"], ["refs"], index=index)


def test_text_sharing_is_lossless_and_does_not_merge_sources():
    text = "Exact claim with a stated condition and limitation. " * 10
    payload = {"first": {"ref": "R1", "text": text}, "second": {"ref": "R2", "text": text},
               "ids": ["long-id:" + "a" * 200] * 2, "numbers": [1.25, 1.25]}
    packed = share_text(payload)
    assert packed["second"]["text"] == {"same_text_at": "/first/text"}
    assert packed["second"]["ref"] == "R2"
    assert packed["ids"] == payload["ids"] and expand(packed) == payload


def test_table_defaults_and_shared_paths_preserve_records():
    rows = [{"ref": "R1", "status": "failed", "paths": ["/status", "/value"]},
            {"ref": "R2", "status": "failed", "paths": ["/status", "/value"]}]
    table = record_table(rows)
    assert [{**table["defaults"], **dict(zip(table["columns"], row))} for row in table["rows"]] == rows
    directory, paths = scoped_directory(rows)
    assert [{**{k: v for k, v in row.items() if k != "path_set"}, "paths": paths[row["path_set"]]}
            for row in directory] == rows
    assert len(paths) == 1


def test_artifact_directory_names_producer_results_not_input_users():
    c = ops.put(case(), results={
        "R1": {"status": "success", "artifact_updates": [{"artifact_id": "A"}]},
        "R2": {"status": "success", "artifact_refs": ["A"]},
        "R3": {"status": "partial", "artifact_updates": [{"artifact_id": "A"}]}})
    index = ops.RecordIndex(c)
    assert index.artifact_producers == {"A": ["R1", "R3"]}
    card = scope_card({"ref": "A", "record_kind": "observation"},
                      {"record_type": "ArtifactRecord", "metadata": {"state": 1}},
                      producers=index.artifact_producers["A"])
    assert card["stored_type"] == "ArtifactRecord"
    assert card["producer_result_refs"] == ["R1", "R3"]
    assert "/metadata" in card["paths"] and "/structured_results" not in card["paths"]


def test_tool_catalog_table_preserves_all_capabilities_and_limits(tmp_path):
    o, c = orchestrator(tmp_path), case()
    before = o.deps.capability_registry.compact_catalog(c.artifact_manifest, amesp_enabled=o.config.enable_amesp)
    compact = catalog_table(before)
    restored = [dict(zip(compact["columns"], row)) for row in compact["rows"]]
    assert len(before) == len(restored)
    for original, row in zip(before, restored):
        for key in compact["columns"]:
            assert row[key] == original.get(key)


def review_material():
    return {"stage": "command_reviewer", "role": "reviewer",
            "arguments": [{"id": "A1", "label": "RIM_RIR_RIV", "refs": ["input:smiles"],
                           "text": "Only a structural hypothesis, not confirmation. " * 30}],
            "ranking": [{"label": "RIM_RIR_RIV", "priority": .5}],
            "observations": {"input:smiles": {"smiles": case().input.smiles}}}


def with_review(c, payload, feedback=None):
    material, receipt, _ = review_projection(c, payload, "reviewer", index=ops.RecordIndex(c))
    receipt["feedback"] = feedback or {"status": "available", "text": "Source attribution remains unresolved."}
    return ops.put(c, review_memory={"reviewer": receipt}, review_receipts={receipt["ref"]: receipt}), receipt, material


def test_unchanged_sections_reusable_with_prior_objections_and_exact_read():
    payload = review_material()
    c, receipt, _ = with_review(case(), payload)
    material, _, ref = review_projection(c, payload, "reviewer", index=ops.RecordIndex(c))
    assert "arguments:A1" in material["reused_sections"]
    assert material["previous_feedback"]["text"] == "Source attribution remains unresolved."
    selection = material["sections"]["arguments:A1"]
    rows = ops.read_records(c, ReadEvidence(kind="read_evidence", records=[{
        "ref": ref, "paths": [selection["path"]]}]))
    assert rows[0]["data"][selection["path"]] == payload["arguments"][0]
    assert ops.record_identity(c, receipt["ref"])["citation_uses"] == []


@pytest.mark.parametrize("mutation", ["wording", "citation", "source", "new_result", "request", "unavailable"])
def test_reuse_invalidation(mutation):
    payload = review_material()
    c = ops.put(case(), results={"R1": {"status": "success", "value": 1.2}}, actions={"D1": {"tool_args": {"state": 1}}})
    c, _, _ = with_review(c, payload, {"status": "unavailable"} if mutation == "unavailable" else None)
    payload = deepcopy(payload)
    if mutation == "wording":
        payload["arguments"][0]["text"] += " New assertion."
    elif mutation == "citation":
        payload["arguments"][0]["refs"].append("R1")
    elif mutation == "source":
        c = ops.put(c, results={"R1": {"status": "success", "value": 9.9}})
    elif mutation == "new_result":
        c = ops.put(c, results={**ops.protocol(c)["results"], "R2": {"status": "failed"}})
    elif mutation == "request":
        c = ops.put(c, actions={"D1": {"tool_args": {"state": 2}}})
    material, _, _ = review_projection(c, payload, "reviewer", index=ops.RecordIndex(c))
    assert "arguments:A1" not in material["reused_sections"]


def test_legacy_judgment_without_interim_receipt_gets_one_review(tmp_path):
    c, o = ops.revise(case(), judgment()), orchestrator(tmp_path, reviews=True)
    update = judgment().model_copy(deep=True)
    update.request_review = False
    update.changes[0].text = "This now asserts something stronger."
    new = _apply(o, c, update, store(tmp_path, c))
    assert ops.protocol(new)["phase"] == "review"


def test_native_final_review_resume_retains_receipts_and_science(tmp_path):
    o, c = orchestrator(tmp_path, reviews=True), ops.revise(case(), judgment())
    o.config = replace(o.config, command_native_tools=True)
    client = NativeFake([prose("A limitation remains unresolved."), prose("The limitation still remains.")])
    o.deps.mechanism_agenda.llm_client = client
    s = store(tmp_path, c)
    c = _apply(o, c, Finalize(kind="finalize", answer="A bounded answer. " * 100), s)
    first = _comment(o, c, s)
    unchanged_science(c, first)
    save(first, s)
    resumed = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    resumed = ops.put(resumed, phase="review")
    second = _comment(o, resumed, s)
    assert expand(client.requests[1]["payload"])["reused_sections"]
    assert ops.protocol(second)["final_review"]["draft_id"] == ops.protocol(second)["final_draft"]["draft_id"]
    for ref in ops.protocol(second)["review_receipts"]:
        assert ref in ops.records(second)
    for ref, receipt in ops.protocol(first)["review_receipts"].items():
        assert ops.protocol(second)["review_receipts"][ref] == receipt


def test_foreign_review_receipt_is_rejected():
    c, receipt, _ = with_review(case(), review_material())
    other = {**receipt, "case_id": "foreign-case"}
    c = ops.put(c, review_receipts={receipt["ref"]: other})
    with pytest.raises(RuntimeError, match="ownership"):
        ops.records(c)


def test_detached_review_memory_cannot_be_reused():
    c, _, _ = with_review(case(), review_material())
    c = ops.put(c, review_receipts={})
    with pytest.raises(RuntimeError, match="registered"):
        review_projection(c, review_material(), "reviewer", index=ops.RecordIndex(c))
