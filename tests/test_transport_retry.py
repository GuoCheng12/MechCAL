"""Real SDK exception paths over an offline HTTP transport; no paid API requests."""

import json

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from mechcal.runtime.api_failures import api_interruption_reason, api_pause_reason
from mechcal.runtime.llm import OpenAIJsonClient, _is_transient_llm_exception
from test_responses_api import response_body, settings, wire


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ReadError, httpx.WriteError,
                                   httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectTimeout])
def test_sdk_wrapped_connection_errors_retry_identical_request(wire, monkeypatch, tmp_path, error):
    requests, replies = wire
    replies.extend([error("fixture"), (200, response_body())])
    path = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("MECHCAL_LLM_TELEMETRY_PATH", str(path))
    assert OpenAIJsonClient(settings(max_retries=2)).complete_json(system_prompt="JSON", payload={}) == {"ok": True}
    assert len(requests) == 2 and requests[0].content == requests[1].content
    assert requests[0].url == requests[1].url
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["attempt_index"] for row in rows] == [1, 2]
    assert rows[0]["usage"] is None
    assert error.__name__ in rows[0]["error_cause_types"]
    assert rows[1]["error_cause_types"] == []
    assert "offline-test" not in path.read_text()


@pytest.mark.parametrize("retries", [0, 1, 2])
def test_retry_exhaustion_pauses_without_fabricating_output(wire, retries):
    requests, replies = wire
    replies.extend([httpx.ReadError("fixture") for _ in range(retries + 1)])
    with pytest.raises(APIConnectionError) as raised:
        OpenAIJsonClient(settings(max_retries=retries)).complete_json(system_prompt="JSON", payload={})
    assert len(requests) == retries + 1
    assert api_interruption_reason(raised.value) == "api_transport_failure"
    assert api_pause_reason(raised.value) is None


@pytest.mark.parametrize("succeed_on_third", [False, True])
def test_180_second_timeout_retries_stop_after_three_requests(wire, monkeypatch, tmp_path, succeed_on_third):
    requests, replies = wire
    replies.extend([httpx.ReadTimeout("fixture") for _ in range(2)])
    replies.append((200, response_body()) if succeed_on_third else httpx.ReadTimeout("fixture"))
    path = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("MECHCAL_LLM_TELEMETRY_PATH", str(path))
    client = OpenAIJsonClient(settings(timeout_seconds=180, max_retries=2))
    if succeed_on_third:
        assert client.complete_json(system_prompt="JSON", payload={}) == {"ok": True}
    else:
        with pytest.raises(APITimeoutError):
            client.complete_json(system_prompt="JSON", payload={})
    assert len(requests) == 3
    assert all(request.content == requests[0].content for request in requests)
    assert all(request.extensions["timeout"]["read"] == 180 for request in requests)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["attempt_index"] for row in rows] == [1, 2, 3]
    assert all(row["attempt_count"] == 3 for row in rows)
    assert rows[-1]["success"] == succeed_on_third


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504])
def test_retryable_http_statuses(wire, status):
    requests, replies = wire
    replies.extend([(status, {"error": {"message": "fixture"}}), (200, response_body())])
    assert OpenAIJsonClient(settings(max_retries=1)).complete_json(system_prompt="JSON", payload={})["ok"]
    assert len(requests) == 2


@pytest.mark.parametrize("status,code", [(400, "invalid_request"), (401, "invalid_api_key"),
                                         (403, "forbidden"), (404, "not_found"),
                                         (422, "unprocessable"), (429, "insufficient_quota")])
def test_permanent_errors_do_not_retry_or_masquerade_as_connection_failure(wire, status, code):
    requests, replies = wire
    replies.append((status, {"error": {"message": "500 timeout connection reset", "code": code}}))
    with pytest.raises(APIStatusError) as raised:
        OpenAIJsonClient(settings(max_retries=2)).complete_json(system_prompt="JSON", payload={})
    assert len(requests) == 1
    assert api_interruption_reason(raised.value) != "api_transport_failure"


def test_local_program_errors_are_not_retried_based_on_their_message():
    for exc in (ValueError("500"), RuntimeError("timeout"), KeyError("429")):
        assert not _is_transient_llm_exception(exc)
        assert api_interruption_reason(exc) is None
