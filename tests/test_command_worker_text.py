"""Plain Worker transport and source ownership, without external requests."""

import json

import httpx
import pytest

from mechcal.agents.command_client import ask_worker_text
from mechcal.runtime.command_loop import _apply, _comment, _execute
from mechcal.runtime.command_operations import OperationRejected, protocol
from mechcal.runtime.llm import OpenAIJsonClient, TextCompletionResult
from test_command_protocol import ScriptClient, case, command, computed, orchestrator, store
from test_responses_api import response_body, settings, wire  # noqa: F401


def chat_body(text, finish_reason="stop", **message):
    return {"id": "chat_test", "object": "chat.completion", "created": 0,
            "model": "test-model", "choices": [{"index": 0, "finish_reason": finish_reason,
            "message": {"role": "assistant", "content": text, **message}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}}


@pytest.mark.parametrize("protocol_name", ["responses", "chat_completions"])
def test_text_wire_has_no_json_constraint_and_preserves_generation_settings(wire, monkeypatch, tmp_path, protocol_name):
    requests, replies = wire
    text = 'First observation: "weak" transition.\n\nA second paragraph, including braces {x}, is ordinary text.'
    replies.append((200, response_body(text) if protocol_name == "responses" else chat_body(text)))
    log = tmp_path / "usage.jsonl"
    monkeypatch.setenv("MECHCAL_LLM_TELEMETRY_PATH", str(log))
    client = OpenAIJsonClient(settings(api_protocol=protocol_name, reasoning_effort="high",
                                       max_tokens=1024, temperature=None, top_p=1.0))
    result = client.complete_text_result(system_prompt="Explain in prose.", payload={"result": {"energy_eV": 2.893}})
    assert result.raw_text == text and result.response_error is None
    assert result.usage["total_tokens"] == 18
    body = json.loads(requests[0].content)
    assert not {"response_format", "tools", "tool_choice", "text"} & body.keys()
    messages = body["input"] if protocol_name == "responses" else body["messages"]
    assert json.loads(messages[-1]["content"]) == {"result": {"energy_eV": 2.893}}
    assert body.get("reasoning", {}).get("effort", body.get("reasoning_effort")) == "high"
    assert body.get("max_output_tokens", body.get("max_tokens")) == 1024
    event = json.loads(log.read_text())
    assert event["success"] and event["response_format_bytes"] == 0
    assert event["usage"] == result.usage
    assert "offline-test" not in log.read_text()


@pytest.mark.parametrize("protocol_name,body", [
    ("responses", response_body("Partial text", status="incomplete", incomplete_details={"reason": "max_output_tokens"})),
    ("responses", response_body("")),
    ("responses", response_body(output=[{"id": "msg", "type": "message", "role": "assistant",
        "status": "completed", "content": [{"type": "refusal", "refusal": "Unavailable."}]}])),
    ("chat_completions", chat_body("Partial text", finish_reason="length")),
    ("chat_completions", chat_body("")),
    ("chat_completions", chat_body("Refusal.", refusal="Unavailable.")),
    ("chat_completions", chat_body(None, reasoning_content="Reasoning only")),
])
def test_incomplete_empty_refused_text_never_becomes_valid_explanation(wire, tmp_path, protocol_name, body):
    requests, replies = wire
    replies.append((200, body))
    client = OpenAIJsonClient(settings(api_protocol=protocol_name))
    with pytest.raises(OperationRejected) as err:
        ask_worker_text(client, {}, store(tmp_path, case()))
    assert err.value.issues[0]["code"] == "unavailable_explanation"
    assert len(requests) == 1
    logs = [json.loads(l) for l in (store(tmp_path, case()).root / "history/command_requests.jsonl").read_text().splitlines()]
    assert logs[-1]["usage"]["total_tokens"] == 18


def test_text_does_not_execute_unrequested_function(wire, tmp_path):
    requests, replies = wire
    body = response_body("Prose.")
    body["output"].append({"type": "function_call", "name": "compute_anything", "arguments": "{}",
                           "call_id": "unexpected", "id": "fc_1"})
    replies.append((200, body))
    with pytest.raises(OperationRejected, match="Unexpected tool call"):
        ask_worker_text(OpenAIJsonClient(settings()), {}, store(tmp_path, case()))
    assert len(requests) == 1


def test_transient_transport_retry_is_bounded_and_counted(wire, monkeypatch, tmp_path):
    requests, replies = wire
    replies.extend([httpx.ReadTimeout("Offline timeout"), (200, response_body("Explanation."))])
    log = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("MECHCAL_LLM_TELEMETRY_PATH", str(log))
    result = OpenAIJsonClient(settings(max_retries=1)).complete_text_result(system_prompt="Explain.", payload={})
    assert result.raw_text == "Explanation." and len(requests) == 2
    events = [json.loads(l) for l in log.read_text().splitlines()]
    assert [e["success"] for e in events] == [False, True]
    assert events[0]["call_id"] == events[1]["call_id"]


def test_timeout_exhaustion_keeps_saved_calculation_and_makes_no_extra_attempt(wire, tmp_path):
    requests, replies = wire
    replies.extend([httpx.ReadTimeout("Offline timeout"), httpx.ReadTimeout("Offline timeout")])
    o, c = orchestrator(tmp_path), computed()
    s = store(tmp_path, c)
    completed = _execute(o, _apply(o, c, command(), s), s)
    writer = o.deps.workers["macro"].report_writer
    writer.enable_llm = True
    writer.llm_client = OpenAIJsonClient(settings(max_retries=1))
    with pytest.raises(Exception):
        _comment(o, completed, s)
    assert len(requests) == 2
    saved = json.loads((s.root / "command_checkpoint.json").read_text())
    assert saved["runtime"]["command_protocol"]["results"] == protocol(completed)["results"]


def test_plain_worker_preserves_prose_and_runtime_binds_source(tmp_path):
    o, c = orchestrator(tmp_path), computed()
    s = store(tmp_path, c)
    completed = _execute(o, _apply(o, c, command(), s), s)
    text = 'Weak transition, f = 0.017.\n\nThis does not establish "CT". Braces {a,b} are prose.'
    writer = o.deps.workers["macro"].report_writer
    writer.enable_llm = True
    writer.llm_client = ScriptClient([text])
    updated = _comment(o, completed, s)
    rid = protocol(completed)["pending_result"]
    explanation = protocol(updated)["explanations"][rid]
    assert explanation == {"status": "available", "text": text, "refs": [rid]}
    assert protocol(updated)["results"] == protocol(completed)["results"]
    assert not protocol(updated)["ranking"]
    assert not {"response_format", "output_contract"} & writer.llm_client.calls[0].keys()
    events = [json.loads(l) for l in (s.root / "history/command_requests.jsonl").read_text().splitlines()]
    assert all(e["transport"] == "plain_text" for e in events)
    assert "response_format" not in events[0]


def test_brace_text_is_not_reparsed_or_given_execution_semantics(tmp_path):
    # Even JSON-looking text is just commentary; no extraction, repair or action.
    text = '{"command": "compute", "source_result_id": "invented"}\nMore prose.'
    result = ask_worker_text(ScriptClient([text]), {}, store(tmp_path, case()))
    assert result.text == text


def test_partial_worker_text_is_logged_but_not_marked_available(tmp_path):
    o, c = orchestrator(tmp_path), computed()
    s = store(tmp_path, c)
    completed = _execute(o, _apply(o, c, command(), s), s)
    writer = o.deps.workers["macro"].report_writer
    writer.enable_llm = True
    writer.llm_client = ScriptClient([TextCompletionResult(
        raw_text="Partial", finish_reason="length", usage={"total_tokens": 7})])
    updated = _comment(o, completed, s)
    rid = protocol(completed)["pending_result"]
    assert protocol(updated)["explanations"][rid]["status"] == "unavailable"
    assert protocol(updated)["results"] == protocol(completed)["results"]
    assert updated.portfolio == completed.portfolio
