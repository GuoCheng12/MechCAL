"""Execution facts are runtime-owned; Worker prose remains reviewable inference."""

from copy import deepcopy

import pytest

from mechcal.runtime import worker_tasks as tasks
from mechcal.runtime.command_impact_review import review_material
from mechcal.runtime.command_loop import _execute
from mechcal.runtime.command_operations import protocol, put, record_index
from mechcal.runtime.command_review import review_execution_context, review_worker_interpretations, reviewer_scope
from mechcal.schemas import CaseRun
from mechcal.schemas.artifacts import ArtifactRecord
from test_reviewer_execution_context import execution_case, capture_review, OTHER, OTHER_DISPATCH, RESULT
from test_worker_tasks import setup, start, load, step, compute_reply
from test_native_commands import reply


def geometry_case():
    c = execution_case()
    p = deepcopy(protocol(c))
    geometry = ArtifactRecord(artifact_id="geometry:selected", kind="prepared_structure",
        created_by_round="R001", created_by_agent="microscopic")
    p["results"][OTHER]["artifact_updates"] = [geometry.model_dump()]
    p["results"][OTHER]["structured_results"].update(
        parent_artifact_id="fixture:scan", parent_member_id="member:6",
        topology_artifact_id="structure:prepared", geometry_validation="fixed_geometry_series_member",
        atom_index_base=0)
    p["actions"][OTHER_DISPATCH]["tool_args"]["source_artifact_id"] = "fixture:scan"
    return c.touch(runtime={**c.runtime, "command_protocol": p},
                   artifact_manifest=c.artifact_manifest.with_records([geometry]))


def test_receipt_retains_exact_roles_not_an_inferred_validation():
    c = geometry_case()
    original = c.model_dump_json()
    available = record_index(c).available
    row = review_execution_context(c, {OTHER}, available)[0]
    assert row["request"]["parameters"]["values"]["source_artifact_id"] == "fixture:scan"
    assert row["outputs"] == [{"artifact_id": "geometry:selected", "kind": "prepared_structure", "status": "available"}]
    checks = row["reported_geometry"]["values"]
    assert checks["topology_artifact_id"] == "structure:prepared"
    assert checks["parent_member_id"] == "member:6"
    assert checks["geometry_validation"] == "fixed_geometry_series_member"
    assert "atom_map_verified" not in checks
    assert row["reported_geometry"]["source_ref"] == OTHER
    assert row["reported_geometry"]["path"] == "/structured_results"
    assert c.model_dump_json() == original


def test_old_missing_geometry_checks_do_not_become_verified():
    c = execution_case()
    rows = review_execution_context(c, {RESULT, OTHER}, record_index(c).available)
    assert all(r["reported_geometry"] is None and r["outputs"] == [] for r in rows)
    assert next(r for r in rows if r["result_id"] == RESULT)["status"] == "failed"


def test_restricted_directory_does_not_leak_producer_request_from_artifact_only():
    c = geometry_case()
    available = record_index(c).available
    restricted = {"geometry:selected": available["geometry:selected"]}
    assert review_execution_context(c, set(restricted), restricted) == []


def test_worker_and_parent_receive_same_facts_and_checkpoint_retains_them(tmp_path):
    from mechcal.runtime.command_read_handoff import retained
    o, c, s, contract = setup(tmp_path)
    c = load(o, start(o, c, s, contract), s)
    c, _ = step(o, c, s, compute_reply())
    c = _execute(o, c, s)
    task = tasks.state(c)
    rid = task.result_ids[0]
    before = deepcopy(protocol(c)["results"])
    payload = tasks.payload(o, c, task)
    for row in payload["execution_context"]:
        request = row["request"]
        if request:
            assert request["parameters"]["source_ref"] in tasks.scope(c, task)
    c, _ = step(o, c, s, reply("complete_worker_task", {
        "status": "complete", "refs": [rid], "summary": "Interpretation of the observations, not a new execution record."}))
    report = protocol(c)["outcome"]
    assert report["execution_context"] == payload["execution_context"]
    assert report["source_observations"] == payload["working_memory"]
    assert report["summary_role"] == "worker_interpretation"
    assert report["scientific_approval"] is False
    assert protocol(c)["results"] == before
    restored = CaseRun.model_validate_json(c.model_dump_json())
    assert protocol(restored)["outcome"]["execution_context"] == report["execution_context"]
    views = review_worker_interpretations(restored, [{"result_id": rid}])
    assert views[0]["source_observations"] == report["source_observations"]
    handles, issues = retained(restored, result_ids=[rid])
    assert not issues
    assert {(h["ref"], h["path"]) for h in handles} == {
        (row["ref"], row["path"]) for row in report["source_observations"]["findings"]}
    assert protocol(c)["rounds"] == 1 and protocol(c)["revision"] == 0


def test_foreign_worker_delivery_cannot_supply_review_facts():
    c = with_interpretation(geometry_case(), "Unverified prose")
    c = put(c, worker_task_reports={"worker-task:fixture": {
        "case_id": "another-case", "source_observations": {"findings": []}}})
    with pytest.raises(RuntimeError, match="another case"):
        review_worker_interpretations(c, [{"result_id": OTHER}])


def with_interpretation(c, text):
    explanation = {"status": "available", "text": text, "refs": [OTHER, RESULT],
                   "source_result_ids": [OTHER, RESULT], "task_id": "worker-task:fixture",
                   "scientific_approval": False, "scope": "task_interpretation"}
    return put(c, explanations={OTHER: explanation, RESULT: deepcopy(explanation)})


def test_task_comparison_is_not_duplicated_as_independent_explanations():
    c = with_interpretation(geometry_case(), "A comparison, not two independent observations.")
    context = review_execution_context(c, {OTHER, RESULT}, record_index(c).available)
    prose = review_worker_interpretations(c, context)
    assert len(prose) == 1
    assert prose[0]["scope"] == "task_interpretation"
    assert prose[0]["scientific_approval"] is False
    assert set(prose[0]["source_result_ids"]) == {OTHER, RESULT}
    assert len(review_material(c, [OTHER, RESULT])["worker_explanations"]) == 1


def test_existing_reviewer_receives_wrong_prose_and_correct_facts_without_execution(tmp_path):
    text = "The scan output is structure:prepared and no change occurred."
    c = with_interpretation(geometry_case(), text)
    before = c.model_dump_json()
    payload = capture_review(tmp_path, c, refs=[OTHER])
    assert payload["worker_interpretations"][0]["text_preview"] == text
    assert payload["worker_interpretations"][0]["scientific_approval"] is False
    assert payload["execution_context"][0]["outputs"][0]["artifact_id"] == "geometry:selected"
    assert set(payload["worker_interpretations"][0]["refs"]) <= set(payload["reference_ids"])
    assert payload["worker_interpretations"][0]["full_text_ref"] in payload["reference_ids"]
    assert c.model_dump_json() == before


def test_scoped_review_can_read_full_comparison_and_its_original_sources():
    c = with_interpretation(geometry_case(), "word " * 2000)
    available = record_index(c).available
    contexts = review_execution_context(c, {OTHER}, available)
    prose = review_worker_interpretations(c, contexts)[0]
    assert prose["text_truncated"] and len(prose["text_preview"]) == 1600
    scoped = reviewer_scope(c, {OTHER}, available)
    assert scoped[prose["full_text_ref"]]["text"] == "word " * 2000
    assert {OTHER, RESULT, "geometry:selected"} <= scoped.keys()
    assert review_worker_interpretations(c, []) == []
    assert not review_worker_interpretations(execution_case(), contexts)


def test_short_report_tail_is_not_hidden_from_reviewer():
    text = "word " * 500 + "Incorrect opposite direction appears in the final sentence."
    c = with_interpretation(geometry_case(), text)
    row = review_worker_interpretations(c, [{"result_id": OTHER}])[0]
    assert row["text_preview"] == text and not row["text_truncated"]


@pytest.mark.parametrize("incremental", [False, True])
def test_unchanged_interpretation_reuses_exact_review_receipt(incremental):
    from mechcal.runtime import command_review_workflow as workflow
    from mechcal.runtime.command_review import review_projection
    from test_command_context_efficiency import review_material as material, with_review
    from test_command_protocol import case
    c = workflow.enable(case()) if incremental else case()
    payload = material()
    payload["worker_interpretations"] = [{"text_preview": "Prior interpretation. " * 150,
                                          "scientific_approval": False}]
    c, receipt, _ = with_review(c, payload)
    shown, current, _ = review_projection(c, payload, "reviewer", index=record_index(c))
    assert "worker_interpretations" in shown["reused_sections"]
    assert shown["sections"]["worker_interpretations"]["previous_review_ref"] == receipt["ref"]
    assert current["sections"]["worker_interpretations"] == payload["worker_interpretations"]
    changed = deepcopy(payload)
    changed["worker_interpretations"][0]["text_preview"] = "A new interpretation with different meaning."
    shown, current, _ = review_projection(c, changed, "reviewer", index=record_index(c))
    assert "worker_interpretations" not in shown["reused_sections"]
    assert current["sections"]["worker_interpretations"] == changed["worker_interpretations"]


@pytest.mark.parametrize("status", ["failed", "partial"])
def test_delivery_keeps_failure_and_raw_result_when_explanation_cannot_finish(tmp_path, status):
    o, c, s, contract = setup(tmp_path)
    c = load(o, start(o, c, s, contract), s)
    c, _ = step(o, c, s, compute_reply())
    c = _execute(o, c, s)
    task = tasks.state(c)
    rid = task.result_ids[0]
    results = deepcopy(protocol(c)["results"])
    results[rid].update(status=status, failure={"kind": "runtime_failed", "message": "Offline fixture"})
    c = put(c, results=results)
    ended = tasks.exhausted(c, task, s, "model_call_limit")
    assert protocol(ended)["results"] == results
    report = protocol(ended)["outcome"]
    assert report["status"] == ("partial" if status == "partial" else "blocked")
    assert report["execution_context"][0]["status"] == status
    assert not report["scientific_approval"]


def test_legacy_explanations_remain_readable():
    c = put(execution_case(), explanations={OTHER: {"status": "available", "text": "Old prose", "refs": [OTHER]}})
    view = review_worker_interpretations(c, [{"result_id": OTHER}])[0]
    assert view["scope"] == "result_interpretation"
    assert view["source_result_ids"] == [OTHER]
    assert view["text_preview"] == "Old prose" and not view["scientific_approval"]
    assert view["source_observations"] is None
