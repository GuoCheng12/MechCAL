"""Native message, execution and persistence contracts; no external model calls."""

from dataclasses import replace
import json

import pytest
from openai.types.chat import ChatCompletion
from openai.types.responses import Response
from pydantic import ValidationError

from mechcal.agents.command_tools import (
    NativeCommandClient, ask_tools, extract_exchange, replay_exchange, tool_contracts,
)
from mechcal.runtime import MechCALOrchestrator
from mechcal.runtime.command_loop import _apply, _comment, _execute, planner_payload, save
from mechcal.runtime.command_operations import OperationRejected, check_refs, protocol, put, read_records, revise
from mechcal.runtime.llm import OpenAICompatibleSettings
from mechcal.schemas.commands import DescribeTools, ReadEvidence, envelope_model
from test_command_protocol import REGISTRY, TOOL, ScriptClient, case, command, computed, judgment, orchestrator, store


def reply(name="read_evidence", arguments=None, *, call_id="call_test", complete=True):
    if name == "read_evidence":
        arguments = arguments if arguments is not None else {"records": [{"ref": "input:smiles", "paths": []}]}
        arguments = {**arguments, "records": [{"paths": [], "view": "data", "page": None, **r} for r in arguments["records"]]}
    call = {"id": call_id, "name": name,
            "arguments": json.dumps(arguments if arguments is not None else {"records": [{"ref": "input:smiles", "paths": []}]})}
    return {"request_id": "planner-" + call_id, "protocol": "responses", "complete": complete,
            "calls": [call], "items": [{"type": "function_call", "call_id": call_id,
                                         "name": name, "arguments": call["arguments"]}]}


class NativeFake:
    settings = None
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def complete_tool_turn(self, **kw):
        self.requests.append(kw)
        return next(self.replies)


def test_function_contracts_do_not_publish_runtime_identity():
    tools, models = tool_contracts(REGISTRY, [TOOL])
    for tool in tools:
        assert not {"case_id", "revision", "kind", "capability_id"} & tool["parameters"]["properties"].keys()
        assert "JSON-encoded value" not in json.dumps(tool)
        assert tool["name"] in models


def test_all_registry_capabilities_have_unique_native_names():
    ids = [s.capability_id for s in REGISTRY.snapshot().capabilities]
    tools, models = tool_contracts(REGISTRY, ids)
    assert len(models) == len(ids) + 9
    assert {"reduce_evidence", "compare_evidence"} <= models.keys()
    assert "review_final" in models
    assert all(len(t["name"]) <= 64 for t in tools)


@pytest.mark.parametrize("cid", [s.capability_id for s in REGISTRY.snapshot().capabilities])
def test_native_compute_crosses_checkpoint_boundary_as_data(cid):
    _, models = tool_contracts(REGISTRY, [cid])
    model, identity = next(v for v in models.values() if v[1]["kind"] == "compute")
    # No scientific parameters are invented. This test exercises serialization
    # and class identity, independent of each calculator's required inputs.
    args_model = model.model_fields["arguments"].annotation
    native = model.model_construct(**identity, arguments=args_model.model_construct(), purpose="Interface test")
    persisted = {"command": native.model_dump(mode="json")}
    restore = envelope_model(REGISTRY, [cid])
    try:
        restored = restore.model_validate(persisted)
        assert restored.command.capability_id == cid
    except ValidationError as exc:
        own = [e for e in exc.errors() if e["loc"][1] == model.__name__]
        assert own and all(e["type"] != "model_type" for e in own)


def test_native_compute_survives_full_loop_model_boundary(tmp_path):
    o = orchestrator(tmp_path)
    o.config = replace(o.config, command_native_tools=True)
    name = next(t["name"] for t in tool_contracts(REGISTRY, [TOOL])[0] if t["name"].startswith("compute_"))
    fake = NativeFake([
        reply("describe_tools", {"capability_ids": [TOOL]}),
        reply(name, {"arguments": command().arguments.model_dump(mode="json"), "purpose": "Inspect structure."}, call_id="compute"),
    ])
    o.deps.planner.llm_client = fake
    original = fake.complete_tool_turn
    def stop_after_compute(**kw):
        result = original(**kw)
        if len(fake.requests) == 2:
            (tmp_path / "native-compute/STOP_REQUESTED").touch()
        return result
    fake.complete_tool_turn = stop_after_compute
    c = o.run(smiles="CCO", user_query="Offline protocol fixture.", case_id="native-compute")
    assert c.status == "paused" and protocol(c)["phase"] == "execute"
    assert protocol(c)["ticket"]["request"]["capability_id"] == TOOL


def test_native_end_to_end_preserves_result_and_final_links(tmp_path):
    o = orchestrator(tmp_path)
    o.config = replace(o.config, command_native_tools=True)
    name = next(t["name"] for t in tool_contracts(REGISTRY, [TOOL])[0] if t["name"].startswith("compute_"))
    update = judgment().model_dump(mode="json", exclude={"kind"})
    fake = NativeFake([
        reply("describe_tools", {"capability_ids": [TOOL]}, call_id="describe"),
        reply(name, {"arguments": command().arguments.model_dump(mode="json"), "purpose": "Inspect structure."}, call_id="compute"),
        reply("revise_judgment", update, call_id="revise"),
        reply("finalize", {"answer": "Offline control-flow fixture, not scientific performance.", "limitations": []}, call_id="finalize"),
    ])
    o.deps.planner.llm_client = fake
    c = o.run(smiles="CCO", user_query="Offline protocol fixture.", case_id="native-final")
    assert c.status == "finalized"
    assert len(protocol(c)["results"]) == 1
    assert protocol(c)["rounds"] == 1 and protocol(c)["revision"] == 1
    assert len(c.final_answer.mechanism_predictions) == 3
    assert c.final_answer.mechanism_predictions[0].evidence_refs == ["input:smiles"]
    messages = list((tmp_path / "native-final/operation_results").glob("*.json"))
    assert len(messages) == 4
    output = json.loads((tmp_path / "native-final/operation_results/planner-compute.json").read_text())
    assert output["outcome"]["result_id"] in protocol(c)["results"]


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_replay_retains_exact_reasoning_and_call_result_pair(protocol):
    if protocol == "chat_completions":
        raw = {"choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": "Inspect a stored result.", "reasoning_content": "private test reasoning",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_evidence", "arguments": "{}"}}]}}]}
    else:
        raw = {"status": "completed", "output": [{"type": "reasoning", "encrypted_content": "opaque-test"},
            {"type": "function_call", "call_id": "c1", "name": "read_evidence", "arguments": "{}"}]}
    exchange = extract_exchange(raw, protocol, "r1")
    messages = replay_exchange(exchange, {"status": "rejected", "executed": False}, protocol)
    assert messages[:-1] == exchange["items"]
    assert messages[-1].get("call_id", messages[-1].get("tool_call_id")) == "c1"
    assert json.loads(messages[-1].get("output", messages[-1].get("content")))["executed"] is False


def test_schema_error_has_original_call_and_exact_argument_path(tmp_path):
    c = case()
    incoming = reply(arguments={"records": []})
    with pytest.raises(OperationRejected) as error:
        ask_tools(NativeFake([incoming]), REGISTRY, [], {}, store(tmp_path, c))
    assert {k: v for k, v in error.value.exchange.items() if k != "current_payload_delivered"} == incoming
    assert error.value.exchange["current_payload_delivered"] is True
    assert error.value.issues[0]["path"] == ["arguments", "records"]
    assert protocol(c)["revision"] == 0


@pytest.mark.parametrize("fault", ["truncated", "unknown", "multiple", "json", "no_call"])
def test_invalid_call_never_becomes_an_operation(tmp_path, fault):
    incoming = reply()
    if fault == "truncated":
        incoming["complete"] = False
    if fault == "unknown":
        incoming["calls"][0]["name"] = "not_visible"
    if fault == "multiple":
        incoming["calls"].append({**incoming["calls"][0], "id": "c2", "name": "finalize"})
    if fault == "json":
        incoming["calls"][0]["arguments"] = '{"records":'
    if fault == "no_call":
        incoming["calls"] = []
    with pytest.raises(OperationRejected):
        ask_tools(NativeFake([incoming]), REGISTRY, [], {}, store(tmp_path, case()))


def test_provider_repeated_call_ids_are_system_error():
    calls = [reply()["items"][0]] * 2
    with pytest.raises(RuntimeError, match="duplicate"):
        extract_exchange({"status": "completed", "output": calls}, "responses", "r1")


def test_correction_loop_preserves_rejected_call_and_case_state(tmp_path):
    o = orchestrator(tmp_path)
    o.config = replace(o.config, command_native_tools=True)
    fake = NativeFake([reply(arguments={"records": [{"ref": "foreign-case", "paths": []}]}), reply(call_id="c2")])
    o.deps.planner.llm_client = fake
    original = fake.complete_tool_turn
    def stop_after_second(**kw):
        result = original(**kw)
        if len(fake.requests) == 2:
            (tmp_path / "native-case/STOP_REQUESTED").touch()
        return result
    fake.complete_tool_turn = stop_after_second
    c = o.run(smiles="CCO", user_query="Offline protocol fixture.", case_id="native-case")
    assert c.status == "paused" and protocol(c)["rounds"] == 0
    assert protocol(c)["revision"] == 0
    assert fake.requests[1]["exchange"]["calls"][0]["id"] == "call_test"
    assert fake.requests[1]["outcome"]["issues"][0]["code"] == "unknown_reference"
    assert fake.requests[1]["outcome"]["issues"][0]["path"] == ["arguments", "records", 0, "ref"]
    assert "last_operation" not in fake.requests[1]["payload"]


def test_new_call_cannot_inherit_previous_completion(tmp_path):
    o, c = orchestrator(tmp_path), put(case(), latest_exchange=reply())
    s = store(tmp_path, c)
    c = save(_apply(o, c, ReadEvidence(kind="read_evidence", records=[{"ref": "input:smiles"}]), s), s)
    first = s.root / "operation_results/planner-call_test.json"
    before = first.read_text()
    c = save(put(c, latest_exchange=reply(call_id="new"), pending_command={"not_executed": True}), s)
    assert not (s.root / "operation_results/planner-new.json").exists()
    assert first.read_text() == before
    c = save(_apply(o, c, ReadEvidence(kind="read_evidence", records=[{"ref": "input:smiles"}]), s), s)
    assert (s.root / "operation_results/planner-new.json").exists()
    checkpoint = (s.root / "command_checkpoint.json").read_text()
    with pytest.raises(RuntimeError, match="Completed tool message conflicts"):
        save(put(c, outcome={"status": "fabricated"}), s)
    assert (s.root / "command_checkpoint.json").read_text() == checkpoint


def test_argument_cannot_shadow_calculation_record():
    c = put(case(), results={"A1": {"status": "success", "value": 1}})
    with pytest.raises(OperationRejected, match="already belongs"):
        revise(c, judgment())
    assert not protocol(c)["arguments"]


def test_worker_explanations_are_queryable_but_not_observations(tmp_path):
    c = put(case(), explanations={"r1": {"status": "available", "text": "Interpretation", "refs": ["input:smiles"]}})
    query = ReadEvidence(kind="read_evidence", records=[{"ref": "explanation:r1"}])
    assert read_records(c, query)[0]["data"]["source_result_id"] == "r1"
    assert planner_payload(orchestrator(tmp_path), c)["explanation_index"][0]["ref"] == "explanation:r1"
    with pytest.raises(OperationRejected, match="not a raw observation"):
        check_refs(c, ["explanation:r1"], [], scientific=True)


def test_advertised_memory_paths_match_actual_record_shape(tmp_path):
    raw = {"capability_id": TOOL, "status": "success", "summary": "test", "failure": {},
           "structured_results": {"parsed_observables": {"energy": 1}, "a/b~c": 2}}
    c = put(case(), results={"result": raw})
    index = planner_payload(orchestrator(tmp_path), c)["result_index"][0]
    assert "fields" not in index
    for path in index["paths"]:
        query = ReadEvidence(kind="read_evidence", records=[{"ref": "result", "paths": [path]}])
        assert path in read_records(c, query)[0]["data"]
    assert "/structured_results/a~1b~0c" in index["paths"]


def test_large_read_returns_navigable_metadata_not_truncated_evidence():
    raw = {"structured_results": {"observations": {"energy": 1, "verbose_log": "x" * 25000}}}
    c = put(case(), results={"result": raw})
    query = ReadEvidence(kind="read_evidence", records=[{"ref": "result", "paths": ["/structured_results/observations"]}])
    with pytest.raises(OperationRejected) as error:
        read_records(c, query)
    layout = error.value.issues[0]["details"]["record_layouts"][0]
    assert layout["children"][0]["path"] == "/structured_results/observations/energy"
    assert "x" * 100 not in json.dumps(layout)
    selected = ReadEvidence(kind="read_evidence", records=[{"ref": "result", "paths": [layout["children"][0]["path"]]}])
    assert read_records(c, selected)[0]["data"] == {"/structured_results/observations/energy": 1}
    assert protocol(c)["results"]["result"] == raw


def test_unknown_path_returns_exact_existing_parent_children():
    c = put(case(), results={"result": {"structured_results": {"known": 1}}})
    query = ReadEvidence(kind="read_evidence", records=[{"ref": "result", "paths": ["/structured_results/guessed"]}])
    with pytest.raises(OperationRejected) as error:
        read_records(c, query)
    assert error.value.issues[0]["details"]["record_layout"]["children"][0]["path"] == "/structured_results/known"


def test_read_schema_disallows_bare_field_names():
    with pytest.raises(ValidationError):
        ReadEvidence(kind="read_evidence", records=[{"ref": "result", "paths": ["parsed_observables"]}])


@pytest.mark.parametrize("path", ["/-1", "/01", "/+1"])
def test_array_pointer_does_not_silently_change_meaning(path):
    c = put(case(), results={"result": [1, 2]})
    query = ReadEvidence(kind="read_evidence", records=[{"ref": "result", "paths": [path]}])
    with pytest.raises(OperationRejected):
        read_records(c, query)


def test_worker_source_identity_is_runtime_owned(tmp_path):
    o, c = orchestrator(tmp_path), computed()
    s = store(tmp_path, c)
    completed = _execute(o, _apply(o, c, command(), s), s)
    writer = o.deps.workers["macro"].report_writer
    writer.enable_llm = True
    writer.llm_client = ScriptClient(["Interpretation of the supplied result."])
    explained = _comment(o, completed, s)
    rid = protocol(completed)["pending_result"]
    feedback = protocol(explained)["explanations"][rid]
    assert feedback["status"] == "available" and feedback["refs"] == [rid]
    assert protocol(explained)["results"] == protocol(completed)["results"]


def test_native_error_does_not_reference_hidden_discriminator(tmp_path):
    o = orchestrator(tmp_path)
    o.config = replace(o.config, command_native_tools=True)
    fake = NativeFake([reply("finalize", {"answer": "Too early", "limitations": []})])
    o.deps.planner.llm_client = fake
    original = fake.complete_tool_turn
    def stop(**kw):
        result = original(**kw)
        (tmp_path / "native-early-final/STOP_REQUESTED").touch()
        return result
    fake.complete_tool_turn = stop
    c = o.run(smiles="CCO", user_query="Offline protocol fixture.", case_id="native-early-final")
    assert protocol(c)["outcome"]["issues"][0]["path"] == ["name"]


def test_raw_result_survives_report_projection_error_and_resume(tmp_path, monkeypatch):
    o, c = orchestrator(tmp_path), computed()
    s = store(tmp_path, c)
    c = save(_apply(o, c, command(), s), s)
    writer = o.deps.workers["macro"].report_writer
    original = writer.deterministic_report
    def fail(**kw):
        raise RuntimeError("Injected report projection error")
    monkeypatch.setattr(writer, "deterministic_report", fail)
    with pytest.raises(RuntimeError, match="projection"):
        _execute(o, c, s)
    saved = type(c).model_validate_json((s.root / "command_checkpoint.json").read_text())
    assert len(protocol(saved)["results"]) == 1
    assert not saved.evidence_ledger.items
    monkeypatch.setattr(writer, "deterministic_report", original)
    monkeypatch.setattr(o.deps.workers["macro"], "execute_tool", lambda *a, **kw: pytest.fail("Duplicate calculation"))
    restored = _execute(o, saved, s)
    assert len(protocol(restored)["results"]) == 1 and protocol(restored)["rounds"] == 1


def test_visible_schema_window_does_not_erase_loaded_tools(tmp_path):
    o, c = orchestrator(tmp_path), case()
    ids = [s.capability_id for s in REGISTRY.snapshot().capabilities if s.owner_agent in {"macro", "microscopic"}][:18]
    s = store(tmp_path, c)
    for start in range(0, 18, 6):
        c = put(c, description_batches=0)
        c = _apply(o, c, DescribeTools(kind="describe_tools", capability_ids=ids[start:start + 6]), s)
    assert len(protocol(c)["loaded_tools"]) == 18
    assert len(protocol(c)["visible_tools"]) == 12
    c = _apply(o, c, DescribeTools(kind="describe_tools", capability_ids=ids[:1]), s)
    assert ids[0] in protocol(c)["visible_tools"]


@pytest.mark.parametrize("api_protocol", ["responses", "chat_completions"])
def test_native_request_is_not_json_response_envelope(tmp_path, monkeypatch, api_protocol):
    import mechcal.agents.command_tools as transport
    seen = []
    def create(settings, args):
        seen.append(args)
        if api_protocol == "responses":
            return Response.model_construct(status="completed", output=[], usage=None)
        return ChatCompletion.model_validate({"id": "r", "created": 1, "model": "test", "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "test"}}]})
    monkeypatch.setattr(transport, "create_completion", create)
    client = NativeCommandClient(OpenAICompatibleSettings(base_url="https://api.deepseek.com", model="test", api_key="test",
        api_protocol=api_protocol, max_retries=0))
    client.complete_tool_turn(system_prompt="test", payload={}, tools=tool_contracts(REGISTRY, [])[0],
        exchange=None, outcome=None, store=store(tmp_path, case()))
    assert "response_format" not in seen[0]
    assert "include" not in seen[0]
    assert seen[0]["tools"] and seen[0]["tool_choice"] == "auto"
    assert seen[0]["messages"][-1]["role"] == "user"


def test_current_request_follows_previous_tool_result(tmp_path, monkeypatch):
    import mechcal.agents.command_tools as transport
    seen = []
    def create(settings, args):
        seen.append(args)
        return Response.model_construct(status="completed", output=[], usage=None)
    monkeypatch.setattr(transport, "create_completion", create)
    client = NativeCommandClient(OpenAICompatibleSettings(base_url="https://api.deepseek.com", model="test", api_key="test",
        api_protocol="responses", max_retries=0))
    client.complete_tool_turn(system_prompt="test", payload={"new_request": True}, tools=[],
        exchange=reply(), outcome={"status": "read"}, store=store(tmp_path, case()))
    messages = seen[0]["messages"]
    assert messages[-2]["type"] == "function_call_output"
    assert json.loads(messages[-1]["content"]) == {"new_request": True}
