"""Offline acceptance. Scripted model decisions are not scientific performance."""

import json
from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from mechcal.capabilities import default_capability_registry
from mechcal.runtime import MechCALOrchestrator, OrchestratorConfig
from mechcal.runtime.command_loop import _apply, _comment, _execute, _local_error
from mechcal.runtime.command_operations import (
    OperationRejected, check_owner, finish, internal_arguments, prepare_computation,
    protocol, put, read_records, revise,
)
from mechcal.runtime.command_schema import native_format
from mechcal.runtime.persistence import RunStore
from mechcal.runtime.llm import JsonCompletionResult, TextCompletionResult
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import (
    Commentary, DescribeTools, Finalize, ReadEvidence, ReviseJudgment, envelope_model,
)
from test_question_loop import initial_case

REGISTRY = default_capability_registry()
TOOL = "macro.screen_rotor_torsion_topology"
LABELS = ["RIM_RIR_RIV", "ICT_TICT_CT", "ESIPT_PT"]


def case():
    c = initial_case()
    c = c.touch(runtime={"tool_environment": {"amesp_enabled": False},
                         "run_limits": {"scientific_tools_allowed": True}})
    return put(c, version=2, case_id=c.case_id, phase="choose", turn=1, revision=0,
               rounds=0, arguments={}, ranking={}, results={}, computations={})


def rank(label, args=None):
    return dict(kind="ranking", label=label, confidence=.4, priority=.5,
                rationale="Offline fixture, not science.", argument_ids=args or [])


def judgment():
    return ReviseJudgment.model_validate(dict(kind="revise_judgment", changes=[
        dict(kind="argument", id="A1", label=LABELS[0], text="A structural hypothesis.", refs=["input:smiles"]),
        rank(LABELS[0], ["A1"]), rank(LABELS[1]), rank(LABELS[2]),
    ]))


def computed(c=None):
    c = c or case()
    return put(c, loaded_tools={TOOL: REGISTRY.definition_version(TOOL)}, visible_tools=[TOOL])


def command(args=None, tool=TOOL):
    return envelope_model(REGISTRY, [tool]).model_validate({"command": dict(
        kind="compute", capability_id=tool, arguments=args or {}, purpose="Inspect structure."
    )}).command


class ScriptClient:
    settings = None
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def complete_json_result(self, **kw):
        self.calls.append(kw)
        value = next(self.replies)
        if isinstance(value, Exception):
            raise value
        raw = value if isinstance(value, str) else json.dumps(value)
        return JsonCompletionResult(data={}, raw_text=raw, finish_reason="stop",
                                    usage={"input_tokens": 10, "output_tokens": 5})

    def complete_text_result(self, **kw):
        self.calls.append(kw)
        value = next(self.replies)
        if isinstance(value, Exception):
            raise value
        if isinstance(value, TextCompletionResult):
            return value
        assert isinstance(value, str)
        return TextCompletionResult(raw_text=value, finish_reason="stop",
                                    usage={"input_tokens": 10, "output_tokens": 5})


def wire(model, data):
    return model.model_validate(data).model_dump(mode="json")


def store(tmp_path, c):
    s = RunStore(tmp_path, c.case_id)
    s.initialize()
    return s


def orchestrator(tmp_path, *, reviews=False):
    cfg = OrchestratorConfig(run_base_dir=tmp_path, decision_mode="question_loop_v1",
        max_exploration_rounds=1, planner_command_protocol=True,
        enable_support_auditor=reviews, enable_agenda_coverage_reviewer=reviews)
    return MechCALOrchestrator(cfg)


def test_atomic_update_and_runtime_metadata():
    c = case()
    before = c.model_dump(mode="json")
    updated = revise(c, judgment())
    assert c.model_dump(mode="json") == before
    assert updated.portfolio.current == LABELS[0]
    assert updated.portfolio.runner_up == LABELS[1]
    assert protocol(updated)["revision"] == 1
    assert len(updated.portfolio.hypotheses) == 3
    bad = judgment().model_copy(deep=True)
    bad.changes[0].refs = ["foreign-case:observation"]
    with pytest.raises(OperationRejected) as err:
        revise(c, bad)
    assert err.value.issues[0]["path"] == ["changes", 0, "refs", 0]
    assert c.model_dump(mode="json") == before


def test_linked_withdraw_requires_atomic_ranking_edit():
    c = revise(case(), judgment())
    bad = ReviseJudgment.model_validate({"kind": "revise_judgment", "changes": [
        {"kind": "withdraw_argument", "id": "A1"}]})
    with pytest.raises(OperationRejected, match="missing"):
        revise(c, bad)
    good = ReviseJudgment.model_validate({"kind": "revise_judgment", "changes": [
        {"kind": "withdraw_argument", "id": "A1"}, rank(LABELS[0])]})
    new = revise(c, good)
    assert not protocol(new)["arguments"]
    assert "A1" in protocol(c)["arguments"]


def test_explanations_are_prose_not_required_reason_fields():
    c = revise(case(), judgment())
    assert protocol(c)["arguments"]["A1"]["text"] == "A structural hypothesis."
    assert not {"warrant", "boundary", "case_id", "revision"} & judgment().model_dump().keys()
    result = finish(c, Finalize(kind="finalize", answer="Bounded answer."))
    assert len(result.final_answer.mechanism_predictions) == 3
    assert result.final_answer.mechanism_predictions[0].evidence == ["A structural hypothesis."]
    assert result.final_answer.mechanism_predictions[1].evidence == []


def test_native_schemas_cover_entire_registry_without_json_in_json():
    for spec in REGISTRY.snapshot().capabilities:
        schema = native_format(envelope_model(REGISTRY, [spec.capability_id]))
        assert "JSON-encoded value" not in json.dumps(schema)
        Draft202012Validator.check_schema(schema["json_schema"]["schema"])
    for model in (Commentary, ReviseJudgment, ReadEvidence):
        assert "JSON-encoded" not in json.dumps(native_format(model))


def test_fragment_list_maps_once_and_checks_duplicate_names():
    cid = "microscopic.analyze_hole_electron_from_bundle"
    c = command({"fragments": [{"name": "donor", "atoms": [1, 2]}]}, tool=cid)
    assert internal_arguments(c, REGISTRY.require(cid))["fragments"] == {"donor": [1, 2]}
    bad = command({"fragments": [{"name": "x", "atoms": [1]}, {"name": "x", "atoms": [2]}]}, tool=cid)
    with pytest.raises(ValueError, match="unique"):
        internal_arguments(bad, REGISTRY.require(cid))


def test_cross_case_storage_is_system_error():
    c = case()
    c.artifact_manifest.case_id = "foreign"
    with pytest.raises(RuntimeError, match="ownership"):
        check_owner(c)


@pytest.mark.parametrize("ref", ["missing", "foreign-case:result"])
def test_unknown_and_cross_case_read_do_not_fall_back(ref):
    with pytest.raises(OperationRejected, match="not in this case"):
        read_records(case(), ReadEvidence(kind="read_evidence", records=[{"ref": ref}]))


def test_invalid_read_path_is_local():
    with pytest.raises(OperationRejected, match="path"):
        read_records(case(), ReadEvidence(kind="read_evidence", records=[
            {"ref": "input:smiles", "paths": ["/not-there"]}]))


def test_loaded_tools_and_ablations_cannot_be_bypassed():
    with pytest.raises(OperationRejected, match="not loaded"):
        prepare_computation(case(), command(), REGISTRY, amesp_enabled=False)
    c = computed()
    c.runtime["ablation"] = {"enabled_capability_owners": ["microscopic"]}
    with pytest.raises(OperationRejected, match="ablation"):
        prepare_computation(c, command(), REGISTRY, amesp_enabled=True)


def test_duplicate_computation_reuses_record_without_execution():
    c = computed()
    req, digest = prepare_computation(c, command(), REGISTRY, amesp_enabled=False)
    assert req is not None
    c = put(c, computations={digest: "saved-result"})
    req2, digest2 = prepare_computation(c, command(), REGISTRY, amesp_enabled=False)
    assert req2 is None and digest2 == digest


def test_missing_source_and_disabled_tools_stay_local():
    cid = "microscopic.run_excited_state_relaxation"
    c = put(case(), loaded_tools={cid: REGISTRY.definition_version(cid)})
    with pytest.raises(OperationRejected):
        prepare_computation(c, command(tool=cid), REGISTRY, amesp_enabled=True)


@pytest.mark.parametrize("raw", ['{"command":', '{"command":{"kind":"finalize"}}', '{}'])
def test_truncated_missing_fields_return_exact_local_errors(tmp_path, raw):
    from mechcal.agents.command_client import ask
    c = case()
    s = store(tmp_path, c)
    with pytest.raises(OperationRejected) as err:
        ask(ScriptClient([raw]), envelope_model(REGISTRY, []), {}, s)
    assert err.value.issues
    log = [json.loads(x) for x in (s.root / "history/command_requests.jsonl").read_text().splitlines()]
    assert log[-1]["raw_text"] == raw


def test_description_queries_do_not_update_science_and_retry_keeps_allowance(tmp_path):
    o, c = orchestrator(tmp_path), case()
    s = store(tmp_path, c)
    request = DescribeTools(kind="describe_tools", capability_ids=[TOOL])
    first = _apply(o, c, request, s)
    second = _apply(o, first, request, s)
    assert protocol(second)["revision"] == 0
    assert protocol(second)["rounds"] == 0 and second.portfolio == c.portfolio
    with pytest.raises(OperationRejected, match="Two description"):
        _apply(o, second, request, s)
    repaired = _local_error(second, [{"path": ["command"], "message": "bad", "code": "schema",
                                    "remedy": "correct"}], s)
    assert protocol(repaired)["description_batches"] == 2
    assert protocol(case()).get("loaded_tools") is None


def test_failed_result_cannot_support_argument():
    c = put(case(), results={"failed-result": {"status": "failed"}})
    data = judgment().model_copy(deep=True)
    data.changes[0].refs = ["failed-result"]
    with pytest.raises(OperationRejected, match="not scientific"):
        revise(c, data)


def test_local_rejection_can_be_corrected_without_mutating_prior_state(tmp_path):
    c, o = case(), orchestrator(tmp_path)
    s = store(tmp_path, c)
    c = _local_error(c, [{"path": ["command", "changes", 0, "refs"], "code": "unknown_reference",
                         "message": "unknown", "remedy": "use local ID"}], s)
    changed = _apply(o, c, judgment(), s)
    assert protocol(changed)["local_errors"] == 0
    assert protocol(changed)["revision"] == 1
    assert protocol(c)["revision"] == 0


def test_completed_tool_survives_explanation_error(tmp_path):
    o, c = orchestrator(tmp_path), computed()
    s = store(tmp_path, c)
    c = _apply(o, c, command(), s)
    completed = _execute(o, c, s)
    assert protocol(completed)["phase"] == "explain"
    assert len(protocol(completed)["results"]) == 1
    raw = deepcopy(protocol(completed)["results"])
    worker = o.deps.workers["macro"]
    worker.report_writer.enable_llm = True
    worker.report_writer.llm_client = ScriptClient([TextCompletionResult(
        raw_text="An interrupted explanation", finish_reason="length", usage={})])
    explained = _comment(o, completed, s)
    assert protocol(explained)["results"] == raw
    assert protocol(explained)["explanations"][protocol(completed)["pending_result"]]["status"] == "unavailable"
    saved = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    assert protocol(saved)["results"] == raw


def test_full_offline_loop_and_local_repair(tmp_path):
    o = orchestrator(tmp_path)
    model = envelope_model(REGISTRY, [])
    client = ScriptClient([
        '{"command":',
        wire(model, {"command": judgment().model_dump(mode="json")}),
        wire(model, {"command": {"kind": "finalize", "answer": "Offline bounded answer."}}),
    ])
    o.deps.planner.llm_client = client
    result = o.run(smiles="CCO", user_query="Rank three mechanisms.", case_id="protocol-test")
    assert result.status == "finalized"
    assert len(result.final_answer.mechanism_predictions) == 3
    assert protocol(result)["revision"] == 1
    assert protocol(result)["rounds"] == 0
    assert len(client.calls) == 3
    assert client.calls[1]["payload"]["last_operation"]["status"] == "rejected"
    assert "case_id" not in client.calls[1]["payload"]


def test_resume_does_not_replay_saved_computation(tmp_path):
    o = orchestrator(tmp_path)
    model = envelope_model(REGISTRY, [])
    tool_model = envelope_model(REGISTRY, [TOOL])
    o.deps.planner.llm_client = ScriptClient([
        wire(model, {"command": {"kind": "describe_tools", "capability_ids": [TOOL]}}),
        wire(tool_model, {"command": command().model_dump(mode="json")}),
        ConnectionError("offline injected outage"),
    ])
    with pytest.raises(ConnectionError):
        o.run(smiles="CCO", user_query="Rank mechanisms.", case_id="resume-test")
    checkpoint = tmp_path / "resume-test/command_checkpoint.json"
    saved = CaseRun.model_validate_json(checkpoint.read_text())
    facts = deepcopy(protocol(saved)["results"])
    assert len(facts) == 1
    o.deps.planner.llm_client = ScriptClient([
        wire(tool_model, {"command": judgment().model_dump(mode="json")}),
        wire(tool_model, {"command": {"kind": "finalize", "answer": "Offline answer."}}),
    ])
    resumed = o.run(smiles="CCO", user_query="Rank mechanisms.", case_id="resume-test", resume=True)
    assert resumed.status == "finalized"
    assert protocol(resumed)["results"] == facts and protocol(resumed)["rounds"] == 1
