"""Review sources retain their exact execution context, including failed attempts."""

from copy import deepcopy

import pytest

from mechcal.runtime.command_loop import _comment, planner_payload
from mechcal.runtime.command_operations import OperationRejected, check_refs, protocol, put, revise
from mechcal.schemas import CaseRun
from test_command_protocol import ScriptClient, case, judgment, orchestrator, store
from test_command_record_identity import RESULT, UNIT, source_case

DISPATCH = "command:failed-attempt"
OTHER = "execution:successful-attempt"
OTHER_UNIT = OTHER + ":E001"
OTHER_DISPATCH = "command:successful-attempt"


def execution_case():
    c = source_case()
    p = deepcopy(protocol(c))
    failed = p["results"][RESULT]
    failed.update(dispatch_id=DISPATCH, capability_id="microscopic.fixture", round_id="R002",
                  agent_name="microscopic", failure={"kind": "parse_failed"})
    successful = deepcopy(failed)
    successful.update(tool_result_id=OTHER, dispatch_id=OTHER_DISPATCH, status="success",
                      round_id="R001", failure={}, structured_results={
                          "parsed_observables": {"effective_parameters": {"state_root": 3}}})
    successful["evidence_units"][0].update(evidence_id=OTHER_UNIT, source_report_id=OTHER,
                                           status="present")
    actions = {DISPATCH: dict(dispatch_id=DISPATCH, capability_id="microscopic.fixture",
        agent_name="microscopic", round_id="R002", input_refs=["input:smiles"], task="Test a state.",
        tool_args={"state_root": 1, "spatial_grid": "medium"})}
    actions[OTHER_DISPATCH] = {**actions[DISPATCH], "dispatch_id": OTHER_DISPATCH,
                              "round_id": "R001", "tool_args": {"state_root": 3}}
    return put(c, results={RESULT: failed, OTHER: successful}, actions=actions)


def capture_review(tmp_path, c, *, role="reviewer", refs=None, limitations=None):
    update = judgment().model_copy(deep=True)
    update.changes[0].refs = refs or ["input:smiles"]
    update.changes[0].limitation_refs = limitations or []
    c = revise(c, update)
    c = put(c, phase="review", review_role=role,
            review_changes=[change.model_dump(mode="json") for change in update.changes])
    before = c.model_dump(mode="json")
    o, s = orchestrator(tmp_path, reviews=True), store(tmp_path, c)
    client = ScriptClient(["The requested S1 calculation failed; this is not positive evidence."])
    o.deps.photophysics_arbiter.llm_client = client
    o.deps.mechanism_agenda.llm_client = client
    reviewed = _comment(o, c, s)
    for key in ("results", "actions", "arguments", "ranking", "revision", "rounds"):
        assert protocol(reviewed).get(key) == before["runtime"]["command_protocol"].get(key)
    for key in ("portfolio", "evidence_ledger", "artifact_manifest"):
        assert reviewed.model_dump(mode="json")[key] == before[key]
    assert planner_payload(o, reviewed)["review_feedback"] == protocol(reviewed)["review"]
    payload = client.calls[0]["payload"]
    assert protocol(reviewed)["review"]["reviewed_record_ids"] == payload["reference_ids"]
    return payload


@pytest.mark.parametrize("role", ["reviewer", "final_reviewer"])
def test_failed_attempt_retains_requested_state_not_inferred_effective_state(tmp_path, role):
    c = execution_case()
    payload = capture_review(tmp_path, c, role=role, limitations=[UNIT])
    contexts = payload["execution_context"]
    assert len(contexts) == 1
    context = contexts[0]
    assert context["source_refs"] == [UNIT]
    assert context["result_id"] == RESULT
    assert context["status"] == "failed"
    assert context["failure"] == {"kind": "parse_failed"}
    assert context["request"]["dispatch_id"] == DISPATCH
    assert context["request"]["input_refs"] == ["input:smiles"]
    assert context["request"]["parameters"]["values"]["state_root"] == 1
    assert context["effective_parameters"] is None
    assert DISPATCH in payload["reference_ids"]
    assert OTHER_DISPATCH not in payload["reference_ids"]
    with pytest.raises(OperationRejected, match="Failed computation"):
        check_refs(c, [UNIT], ["refs"], scientific=True)


def test_success_and_failure_remain_distinct_and_repeated_sources_deduplicate(tmp_path):
    payload = capture_review(tmp_path, execution_case(), refs=[OTHER, OTHER_UNIT], limitations=[UNIT, RESULT])
    contexts = {row["result_id"]: row for row in payload["execution_context"]}
    assert set(contexts) == {RESULT, OTHER}
    assert contexts[OTHER]["source_refs"] == [OTHER, OTHER_UNIT]
    assert contexts[OTHER]["request"]["parameters"]["values"] == {"state_root": 3}
    assert contexts[OTHER]["effective_parameters"]["values"] == {"state_root": 3}
    assert contexts[OTHER]["effective_parameters"]["source_ref"] == OTHER
    assert contexts[RESULT]["effective_parameters"] is None


def test_missing_legacy_dispatch_is_not_reconstructed_from_other_attempt(tmp_path):
    c = execution_case()
    c = put(c, actions={OTHER_DISPATCH: protocol(c)["actions"][OTHER_DISPATCH]})
    payload = capture_review(tmp_path, c, limitations=[UNIT])
    assert payload["execution_context"][0]["request"] is None
    assert payload["execution_context"][0]["effective_parameters"] is None
    assert DISPATCH not in payload["reference_ids"]


@pytest.mark.parametrize("field,value", [
    ("dispatch_id", "wrong"), ("capability_id", "macro.other"),
    ("round_id", "R999"), ("agent_name", "macro"), ("case_id", "other-case"),
])
def test_conflicting_request_identity_cannot_supply_provenance(tmp_path, field, value):
    c = execution_case()
    actions = deepcopy(protocol(c)["actions"])
    actions[DISPATCH][field] = value
    with pytest.raises(RuntimeError, match="identity|ownership"):
        capture_review(tmp_path, put(c, actions=actions), limitations=[UNIT])


def test_no_cross_case_context_or_whole_history_in_smiles_review(tmp_path):
    payload = capture_review(tmp_path, execution_case())
    assert payload["execution_context"] == []
    assert payload["reference_ids"] == ["A1", "input:smiles"]
    payload = capture_review(tmp_path / "other", case())
    assert payload["execution_context"] == []


def test_large_parameters_are_explicitly_deferred_not_silently_truncated(tmp_path):
    c = execution_case()
    actions = deepcopy(protocol(c)["actions"])
    large = list(range(10000))
    actions[DISPATCH]["tool_args"]["fragments"] = {"fragment": large}
    payload = capture_review(tmp_path, put(c, actions=actions), limitations=[UNIT])
    params = payload["execution_context"][0]["request"]["parameters"]
    assert params["values"] == {"state_root": 1, "spatial_grid": "medium"}
    assert params["source_ref"] == DISPATCH
    assert params["deferred"][0]["path"] == "/tool_args/fragments"
    assert params["deferred"][0]["bytes"] > 2000
    assert len(str(payload["execution_context"])) < 2500
    assert actions[DISPATCH]["tool_args"]["fragments"]["fragment"] == large


def test_checkpoint_roundtrip_preserves_execution_links(tmp_path):
    c = execution_case()
    restored = CaseRun.model_validate_json(c.model_dump_json())
    payload = capture_review(tmp_path, restored, limitations=[UNIT])
    assert payload["execution_context"][0]["request"]["parameters"]["values"]["state_root"] == 1


def test_previous_argument_sources_are_available_for_change_review_only(tmp_path):
    c = execution_case()
    previous = judgment().changes[0].model_dump(mode="json")
    previous["limitation_refs"] = [UNIT]
    c = put(c, review_previous_arguments=[previous])
    payload = capture_review(tmp_path, c)
    assert payload["execution_context"][0]["source_refs"] == [UNIT]
    payload = capture_review(tmp_path / "final", c, role="final_reviewer")
    assert payload["execution_context"] == []


def test_cited_output_artifact_links_to_producing_execution_not_upstream_history(tmp_path):
    c = execution_case()
    results = deepcopy(protocol(c)["results"])
    artifact = {"artifact_id": "artifact:output", "kind": "fixture", "created_by_round": "R001",
                "created_by_agent": "microscopic", "status": "available", "file_paths": {}, "metadata": {}}
    results[OTHER]["artifact_updates"] = [artifact]
    from mechcal.schemas.artifacts import ArtifactRecord

    c = put(c, results=results)
    c = c.touch(artifact_manifest=c.artifact_manifest.model_copy(
        update={"artifacts": [ArtifactRecord.model_validate(artifact)]}))
    payload = capture_review(tmp_path, c, refs=[artifact["artifact_id"]])
    assert len(payload["execution_context"]) == 1
    assert payload["execution_context"][0]["result_id"] == OTHER
    assert RESULT not in payload["reference_ids"]


def test_effective_parameters_are_reported_values_not_request_defaults(tmp_path):
    c = execution_case()
    actions = deepcopy(protocol(c)["actions"])
    actions[OTHER_DISPATCH]["tool_args"] = {"state_root": 2}
    payload = capture_review(tmp_path, put(c, actions=actions), refs=[OTHER_UNIT])
    context = payload["execution_context"][0]
    assert context["request"]["parameters"]["values"]["state_root"] == 2
    assert context["effective_parameters"]["values"]["state_root"] == 3


def test_precommand_legacy_source_without_execution_remains_reviewable(tmp_path):
    c = execution_case()
    unit = deepcopy(protocol(c)["results"][OTHER]["evidence_units"][0])
    unit.update(evidence_id="legacy:unit", source_report_id="legacy:report")
    from mechcal.schemas.evidence import EvidenceUnit

    c = c.touch(evidence_ledger=c.evidence_ledger.with_items([EvidenceUnit.model_validate(unit)]))
    payload = capture_review(tmp_path, c, refs=["legacy:unit"])
    assert "legacy:unit" in payload["observations"]
    assert payload["execution_context"] == []
