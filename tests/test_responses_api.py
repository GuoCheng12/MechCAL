import json
from dataclasses import replace

import httpx
import pytest

from mechcal.agents.planner import PlannerAgent
from mechcal.eval.llm import OpenAITextClient
from mechcal.eval.mechanism_ranking import _support_judge_settings
from mechcal.runtime.llm import OpenAICompatibleSettings, OpenAIJsonClient, run_llm_smoke


def settings(**overrides):
    return OpenAICompatibleSettings(
        base_url="https://provider.test/v1", api_key="offline-test",
        model="test-model", **overrides,
    )


def response_body(text='{"ok":true}', **overrides):
    return {
        "id": "resp_test", "object": "response", "created_at": 0,
        "model": "test-model", "status": "completed", "error": None,
        "incomplete_details": None, "metadata": {},
        "output": [{"id": "msg_test", "type": "message", "role": "assistant",
                    "status": "completed", "content": [
                        {"type": "output_text", "text": text, "annotations": []},
                    ]}],
        "usage": {"input_tokens": 10, "output_tokens": 8, "total_tokens": 18,
                  "input_tokens_details": {"cached_tokens": 2},
                  "output_tokens_details": {"reasoning_tokens": 3}},
        **overrides,
    }


@pytest.fixture
def wire(monkeypatch):
    requests = []
    replies = []
    clients = []
    original_client = httpx.Client

    def handle(request):
        requests.append(request)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, httpx.Response):
            return reply
        status, body = reply
        return httpx.Response(status, json=body)

    class client(original_client):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handle), **kwargs)
            clients.append(self)

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setattr("mechcal.runtime.llm.time.sleep", lambda _: None)
    yield requests, replies
    assert all(client.is_closed for client in clients)


def test_responses_json_schema_usage_and_telemetry(wire, monkeypatch, tmp_path):
    requests, replies = wire
    replies.append((200, response_body()))
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("MECHCAL_LLM_TELEMETRY_PATH", str(log))
    schema = {"name": "test", "strict": True, "schema": {
        "type": "object", "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"], "additionalProperties": False,
    }}
    result = OpenAIJsonClient(settings(
        reasoning_effort="high", max_tokens=1024, temperature=None, top_p=None,
    )).complete_json_result(
        system_prompt="Return JSON.", payload={"case_id": "example"},
        schema_feedback="Include ok.",
        response_format={"type": "json_schema", "json_schema": schema},
    )
    assert result.data == {"ok": True}
    assert result.finish_reason == "stop"
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 8,
                            "total_tokens": 18, "cached_tokens": 2, "reasoning_tokens": 3}
    assert result.response_metadata["response_id"] == "resp_test"
    assert requests[0].url.path == "/v1/responses"
    body = json.loads(requests[0].content)
    assert body["input"][0] == {"role": "system", "content": "Return JSON."}
    assert json.loads(body["input"][1]["content"]) == {
        "case_id": "example", "schema_feedback": "Include ok.",
    }
    assert body["text"]["format"] == {"type": "json_schema", **schema}
    assert body["reasoning"] == {"effort": "high"}
    assert body["max_output_tokens"] == 1024
    assert body["store"] is False and body["stream"] is False
    assert not {"messages", "max_tokens", "temperature", "top_p",
                "previous_response_id", "conversation", "tools"}.intersection(body)
    event = json.loads(log.read_text())
    assert event["api_protocol"] == "responses" and event["success"]
    assert event["usage"] == result.usage
    assert "offline-test" not in log.read_text()


def test_chat_compatibility_is_explicit(wire):
    requests, replies = wire
    replies.append((200, {
        "id": "chat_test", "object": "chat.completion", "created": 0,
        "model": "test-model", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": '{"ok":true}'}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    }))
    result = OpenAIJsonClient(settings(api_protocol="chat_completions", seed=42)).complete_json_result(
        system_prompt="Return JSON.", payload={},
    )
    assert result.data == {"ok": True}
    assert requests[0].url.path == "/v1/chat/completions"
    body = json.loads(requests[0].content)
    assert body["seed"] == 42 and body["temperature"] == 0
    assert body["response_format"] == {"type": "json_object"}
    assert "input" not in body


def test_deepseek_wire_expands_definitions_and_keeps_strict_schema(wire):
    from test_deepseek_schema import sample_format

    requests, replies = wire
    replies.append((200, response_body('{"answer":{"score":0.5}}')))
    fmt = sample_format()
    client = OpenAIJsonClient(replace(settings(), base_url="https://api.deepseek.com"))
    result = client.complete_json_result(system_prompt="Return JSON.", payload={}, response_format=fmt)
    body = json.loads(requests[0].content)
    assert result.data == {"answer": {"score": .5}}
    assert body["text"]["format"]["strict"]
    assert "$ref" not in json.dumps(body["text"]["format"])
    assert "$defs" in fmt["json_schema"]["schema"]


@pytest.mark.parametrize("text", [
    '<tool_call name="context_read">{"action":"prepare_context"}</tool_call>',
    '<\uff5c\uff5cDSML\uff5c\uff5c calls><\uff5c\uff5cDSML\uff5c\uff5c invoke name="context_read">'
    '<\uff5c\uff5cDSML\uff5c\uff5c parameter name="data" string="false">{"action":"prepare_context"}'
    '</\uff5c\uff5cDSML\uff5c\uff5c parameter></\uff5c\uff5cDSML\uff5c\uff5c invoke></\uff5c\uff5cDSML\uff5c\uff5c calls>',
    '```json\n{"ok":true}\n```',
    '{"ok":true}\n{"second":true}',
])
def test_strict_json_never_extracts_a_nested_object_from_protocol_markup(wire, text):
    from test_deepseek_schema import sample_format

    requests, replies = wire
    replies.append((200, response_body(text)))
    result = OpenAIJsonClient(settings()).complete_json_result(
        system_prompt="Return JSON.", payload={}, response_format=sample_format(),
        preserve_invalid_response=True)
    assert result.data == {} and "complete JSON object" in result.parse_error
    assert result.raw_text == text and result.usage["total_tokens"] == 18
    assert len(requests) == 1


@pytest.mark.parametrize("text,message", [
    ('{"response":{"kind":"decision","data":{}}}}', "Extra data"),
    ('{"response": {\n"kind":"decision" "data":{}}}', "Expecting ',' delimiter"),
    ('{"response":{"kind":"decision","data":', "Expecting value"),
])
def test_strict_json_reports_real_syntax_location_without_repairing_text(wire, text, message):
    from test_deepseek_schema import sample_format

    requests, replies = wire
    replies.append((200, response_body(text)))
    result = OpenAIJsonClient(settings()).complete_json_result(
        system_prompt="Return JSON.", payload={}, response_format=sample_format(),
        preserve_invalid_response=True)
    with pytest.raises(json.JSONDecodeError) as exc:
        json.loads(text)
    assert message in result.parse_error
    assert f"line {exc.value.lineno}, column {exc.value.colno}" in result.parse_error
    assert f"offset {exc.value.pos}" in result.parse_error
    assert result.raw_text == text and result.data == {}
    assert result.usage["total_tokens"] == 18 and len(requests) == 1


def test_opt_in_suffix_normalization_preserves_raw_reply_schema_usage_and_audit(wire, tmp_path, monkeypatch):
    from test_deepseek_schema import sample_format
    from mechcal.runtime.retry import complete_recorded_json

    requests, replies = wire
    original = '{"answer":{"score":0.5}}}}'
    replies.append((200, response_body(original)))
    telemetry = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("MECHCAL_LLM_TELEMETRY_PATH", str(telemetry))
    client = OpenAIJsonClient(replace(settings(), normalize_json_suffix=True))
    events = []
    result = complete_recorded_json(client, system_prompt="Return JSON.", payload={},
                                   response_format=sample_format(), event_sink=events.append)
    assert result == {"answer": {"score": .5}}
    response = events[-1]
    assert response["raw_text"] == original and response["parse_error"] is None
    assert response["usage"]["total_tokens"] == 18
    assert response["transport_normalization"]["removed_suffix"] == "}}"
    assert response["transport_normalization"]["values_modified"] is False
    event = json.loads(telemetry.read_text())
    assert event["transport_normalization"] == response["transport_normalization"]
    assert event["success"] and len(requests) == 1
    assert json.loads(requests[0].content)["text"]["format"]["strict"] is True


@pytest.mark.parametrize("text", ['{"answer":{"score":"bad"}}}',
                                  '{"answer":{"score":0.5}} another answer',
                                  '{"answer":{"score":0.5}'])
def test_opt_in_suffix_normalization_still_rejects_semantic_and_truncation_errors(wire, text):
    from test_deepseek_schema import sample_format

    requests, replies = wire
    replies.append((200, response_body(text)))
    result = OpenAIJsonClient(replace(settings(), normalize_json_suffix=True)).complete_json_result(
        system_prompt="Return JSON.", payload={}, response_format=sample_format(), preserve_invalid_response=True)
    assert result.data == {} and result.parse_error
    assert result.raw_text == text and len(requests) == 1
    assert not (result.response_metadata or {}).get("transport_normalization")




def event_stream(*events):
    content = ": keepalive\r\n\r\n" + "".join(
        f"event: {item['type']}\r\ndata: {json.dumps(item)}\r\n\r\n" for item in events
    )
    return httpx.Response(200, content=content.encode(), headers={"content-type": "text/event-stream; charset=utf-8"})


def test_unrequested_responses_stream_retains_terminal_text_usage_and_identity(wire):
    requests, replies = wire
    replies.append(event_stream(
        {"type": "response.created", "response": response_body(status="in_progress", output=[])},
        {"type": "response.output_text.delta", "delta": "not the final answer"},
        {"type": "response.completed", "response": response_body()},
    ))
    result = OpenAIJsonClient(settings()).complete_json_result(system_prompt="JSON", payload={})
    assert result.data == {"ok": True}
    assert result.usage["total_tokens"] == 18
    assert result.response_metadata["response_id"] == "resp_test"
    assert json.loads(requests[0].content)["stream"] is False
    assert len(requests) == 1


@pytest.mark.parametrize("events, message", [
    ([{"type": "response.output_text.delta", "delta": '{"ok":true}'}], "without a terminal"),
    ([{"type": "response.created", "response": response_body()},
      {"type": "response.completed", "response": response_body(id="other")}], "identity"),
    ([{"type": "response.completed", "response": response_body(status="in_progress")}], "terminal state"),
    ([{"type": "error", "code": "server_error", "message": "Test error"}], "error event"),
])
def test_invalid_streams_retry_identical_request_without_accepting_deltas(wire, events, message):
    requests, replies = wire
    replies.extend([event_stream(*events), event_stream(*events)])
    with pytest.raises(httpx.RemoteProtocolError, match=message):
        OpenAIJsonClient(settings(max_retries=1)).complete_json(system_prompt="JSON", payload={})
    assert len(requests) == 2 and requests[0].content == requests[1].content


@pytest.mark.parametrize("status", ["failed", "incomplete"])
def test_terminal_unsuccessful_stream_is_not_promoted_to_completion(wire, status):
    requests, replies = wire
    replies.append(event_stream({"type": "response." + status,
                                 "response": response_body(status=status)}))
    with pytest.raises(ValueError, match="unfinished response"):
        OpenAIJsonClient(settings()).complete_json(system_prompt="JSON", payload={})
    assert len(requests) == 1


def test_malformed_stream_json_is_a_transport_failure(wire):
    _, replies = wire
    replies.append(httpx.Response(200, content=b'event: response.completed\ndata: {broken\n\n',
                                  headers={"content-type": "text/event-stream"}))
    with pytest.raises(httpx.RemoteProtocolError, match="malformed event JSON"):
        OpenAIJsonClient(settings(max_retries=0)).complete_json(system_prompt="JSON", payload={})


@pytest.mark.parametrize("body, match", [
    (response_body(status="incomplete", incomplete_details={"reason": "max_output_tokens"}),
     "unfinished response"),
    (response_body(status="failed", error={"code": "server_error", "message": "Failed"}),
     "unfinished response"),
    (response_body(output=[]), "no final text"),
    (response_body(output=[{"type": "reasoning", "id": "r", "summary": [
        {"type": "summary_text", "text": '{"ok":true}'},
    ]}]), "no final text"),
    (response_body(output=[{"type": "message", "id": "m", "role": "assistant",
        "status": "completed", "content": [{"type": "refusal", "refusal": "No"}]}]),
     "refused"),
    (response_body(text="not JSON"), "JSON object"),
])
def test_invalid_outputs_never_become_success(wire, body, match):
    requests, replies = wire
    replies.append((200, body))
    with pytest.raises(ValueError, match=match):
        OpenAIJsonClient(settings()).complete_json(system_prompt="JSON", payload={})
    assert len(requests) == 1


def test_truncated_but_parseable_json_is_preserved_as_failure(wire):
    _, replies = wire
    replies.append((200, response_body(
        status="incomplete", incomplete_details={"reason": "max_output_tokens"},
    )))
    result = OpenAIJsonClient(settings()).complete_json_result(
        system_prompt="JSON", payload={}, preserve_invalid_response=True,
    )
    assert result.data == {} and result.parse_error
    assert result.raw_text == '{"ok":true}' and result.finish_reason == "length"
    assert result.usage["total_tokens"] == 18


@pytest.mark.parametrize("body", [response_body(text='{"action":'), response_body(
    status="incomplete", incomplete_details={"reason": "max_output_tokens"},
)])
def test_recorded_completion_retains_invalid_wire_output_before_retry(wire, body):
    from mechcal.runtime.retry import complete_recorded_json, RejectedResponseError

    requests, replies = wire
    replies.append((200, body))
    events = []
    with pytest.raises(RejectedResponseError) as error:
        complete_recorded_json(OpenAIJsonClient(settings()), event_sink=events.append,
                               system_prompt="JSON", payload={})
    assert len(requests) == 1 and len(events) == 2
    assert events[0]["kind"] == "model_request"
    assert events[1]["parse_error"] and events[1]["usage"]["total_tokens"] == 18
    assert events[1]["raw_text"] == error.value.response
    assert "offline-test" not in json.dumps(events)


def test_transient_failure_retries_same_endpoint(wire):
    requests, replies = wire
    replies.extend([(503, {"error": {"message": "Unavailable"}}), (200, response_body())])
    result = OpenAIJsonClient(settings(max_retries=1)).complete_json(system_prompt="JSON", payload={})
    assert result["ok"]
    assert len(requests) == 2 and requests[0].content == requests[1].content
    assert all(request.url.path == "/v1/responses" for request in requests)


@pytest.mark.parametrize("status", [401, 404, 400])
def test_no_silent_protocol_fallback(wire, status):
    from openai import APIStatusError

    requests, replies = wire
    replies.append((status, {"error": {"message": "Request rejected"}}))
    with pytest.raises(APIStatusError):
        OpenAIJsonClient(settings()).complete_json(system_prompt="JSON", payload={})
    assert len(requests) == 1


def test_retry_exhaustion(wire):
    requests, replies = wire
    replies.extend([(503, {"error": {"message": "Unavailable"}})] * 3)
    from openai import APIStatusError

    with pytest.raises(APIStatusError):
        OpenAIJsonClient(settings(max_retries=2)).complete_json(system_prompt="JSON", payload={})
    assert len(requests) == 3


def test_text_and_smoke_use_responses(wire):
    requests, replies = wire
    replies.extend([(200, response_body(text="OK")), (200, response_body(text="Answer"))])
    config = settings(max_tokens=200)
    assert run_llm_smoke(config) == "OK"
    assert OpenAITextClient(config).complete_text(system_prompt="Explain.", payload={}) == "Answer"
    for request in requests:
        body = json.loads(request.content)
        assert request.url.path == "/v1/responses"
        assert "text" not in body and body["max_output_tokens"] == 200


def test_thinking_tags_and_reasoning_summary(wire):
    _, replies = wire
    body = response_body(text='<think>Internal</think><answer>{"ok":true}</answer>')
    body["output"].insert(0, {"type": "reasoning", "id": "r",
        "summary": [{"type": "summary_text", "text": "A summary"}]})
    replies.append((200, body))
    result = OpenAIJsonClient(settings()).complete_json_result(
        system_prompt="JSON", payload={}, answer_tag="answer", omit_response_format=True,
    )
    assert result.data == {"ok": True} and result.reasoning_text == "A summary"


def test_protocol_settings_propagate(monkeypatch):
    monkeypatch.delenv("MECHCAL_OPENAI_API_PROTOCOL", raising=False)
    assert OpenAICompatibleSettings.from_env().api_protocol == "responses"
    monkeypatch.setenv("MECHCAL_OPENAI_API_PROTOCOL", "chat_completions")
    assert OpenAICompatibleSettings.from_env().api_protocol == "chat_completions"
    config = settings(api_protocol="chat_completions")
    planner = PlannerAgent(llm_client=OpenAIJsonClient(config), enable_llm_decisions=True)
    assert planner.llm_client.settings == config
    monkeypatch.delenv("MECHCAL_SUPPORT_JUDGE_API_PROTOCOL", raising=False)
    assert _support_judge_settings(config).api_protocol == "chat_completions"
    monkeypatch.setenv("MECHCAL_SUPPORT_JUDGE_API_PROTOCOL", "responses")
    assert _support_judge_settings(config).api_protocol == "responses"
    assert _support_judge_settings(None).api_protocol == "responses"
    with pytest.raises(ValueError, match="api_protocol"):
        replace(config, api_protocol="typo")


def test_seed_not_silently_dropped(wire):
    requests, _ = wire
    with pytest.raises(ValueError, match="does not support seed"):
        OpenAIJsonClient(settings(seed=7)).complete_json(system_prompt="JSON", payload={})
    assert not requests


def test_timeout_retries_and_default_json_mode(wire):
    requests, replies = wire
    replies.extend([httpx.ReadTimeout("timed out"), (200, response_body())])
    result = OpenAIJsonClient(settings()).complete_json(system_prompt="JSON", payload={})
    assert result == {"ok": True} and len(requests) == 2
    body = json.loads(requests[-1].content)
    assert body["text"] == {"format": {"type": "json_object"}}


def test_base_url_not_rewritten(wire):
    requests, replies = wire
    replies.append((200, response_body()))
    config = replace(settings(), base_url="https://provider.test")
    OpenAIJsonClient(config).complete_json(system_prompt="JSON", payload={})
    assert requests[0].url.path == "/responses"


def test_existing_codex_responses_client_rejects_empty_output(wire):
    from mechcal.eval.mechanism_baselines import CodexResponsesJsonClient

    _, replies = wire
    replies.append((200, response_body(output=[])))
    client = CodexResponsesJsonClient(base_url="https://provider.test", api_key="offline-test")
    with pytest.raises(ValueError, match="no final text"):
        client.complete_json(system_prompt="JSON", payload={})


@pytest.mark.parametrize("extra", [{"store": True}, {"previous_response_id": "resp_other"}])
def test_extra_body_cannot_change_context_policy(wire, extra):
    requests, _ = wire
    with pytest.raises(ValueError, match="cannot override"):
        OpenAIJsonClient(settings()).complete_json(system_prompt="JSON", payload={}, extra_body=extra)
    assert not requests


@pytest.mark.parametrize("bad", [
    httpx.Response(200, text="temporary gateway failure"),
    httpx.Response(200, json="unexpected plain string"),
    httpx.Response(200, json={"error": {"message": "upstream failure"}}),
    httpx.Response(200, json=response_body(output=None)),
    httpx.Response(200, json=response_body(output=[None])),
    httpx.Response(200, json=response_body(output=[{"type": "message", "content": None}])),
    httpx.Response(200, json=response_body(output=[{"type": "message", "content": [
        {"type": "output_text", "text": None}]}])),
])
def test_malformed_success_envelope_retries_identical_strict_request(wire, bad):
    requests, replies = wire
    replies.extend([bad, (200, response_body())])
    schema = {"type": "json_schema", "json_schema": {"name": "test", "strict": True,
        "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}},
                   "required": ["ok"], "additionalProperties": False}}}
    result = OpenAIJsonClient(settings(max_retries=1)).complete_json_result(
        system_prompt="JSON", payload={}, response_format=schema, preserve_invalid_response=True,
    )
    assert result.data == {"ok": True} and len(requests) == 2
    assert requests[0].content == requests[1].content
    assert json.loads(requests[1].content)["text"]["format"]["strict"] is True


def test_malformed_envelope_exhaustion_is_transport_pause_not_schema_or_zero(wire, monkeypatch, tmp_path):
    from mechcal.runtime.api_failures import api_interruption_reason
    requests, replies = wire
    replies.extend([(200, "private gateway text")] * 3)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("MECHCAL_LLM_TELEMETRY_PATH", str(log))
    with pytest.raises(httpx.RemoteProtocolError) as caught:
        OpenAIJsonClient(settings(max_retries=2)).complete_json_result(
            system_prompt="JSON", payload={}, preserve_invalid_response=True,
        )
    assert api_interruption_reason(caught.value) == "api_transport_failure"
    assert len(requests) == 3
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert all(not e["success"] and e["usage"] is None for e in events)
    assert all(e["error_type"] == "RemoteProtocolError" for e in events)
    assert "private gateway text" not in log.read_text()


@pytest.mark.parametrize("value", [None, "{\"ok\":true}", b"html", [], {}])
def test_invalid_envelope_accessors_never_raise_attribute_error(value):
    from mechcal.runtime.llm import response_text, validate_response, _response_reasoning
    for function in (response_text, validate_response, _response_reasoning):
        with pytest.raises(httpx.RemoteProtocolError):
            function(value)
