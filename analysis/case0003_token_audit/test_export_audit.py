import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("export_audit", Path(__file__).with_name("export_audit.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def req(attempt=1):
    return {"event": "request", "id": "id-1", "attempt": attempt, "role": "planner",
            "timestamp": 1000, "messages": [{"role": "user", "content": "hello"}], "tools": []}


def resp(attempt=1, usage=None):
    return {"event": "response", "id": "id-1", "attempt": attempt, "role": "planner",
            "usage": usage, "elapsed_seconds": 2, "raw_response": {"model": "test"}}


def test_attempt_pairing_and_subset_counters():
    usage = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13,
             "cached_tokens": 8, "reasoning_tokens": 2}
    calls = audit.telemetry([req(), resp(usage=usage), req(2), resp(2, usage)])
    assert audit.totals(calls)["total_tokens"] == 26
    assert calls[1]["provenance"]["request_line"] == 3
    assert calls[1]["original_message_fingerprints"][0]["previous_identical_occurrences"] == 1


def test_missing_usage_is_not_zero():
    call = audit.telemetry([req(), resp()])[0]
    assert call["input_tokens"] is None
    assert audit.totals([call])["total_tokens"] is None
    assert not call["usage_recorded"]


@pytest.mark.parametrize("rows", [[req(), req()], [resp()], [req()]])
def test_incomplete_or_ambiguous_history_is_not_silently_published(rows):
    with pytest.raises(ValueError):
        audit.telemetry(rows)


def test_inconsistent_usage_fails():
    with pytest.raises(ValueError):
        audit.telemetry([req(), resp(usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 99})])


def test_path_and_credentials_redaction_preserves_task_ids():
    task = "task-" + "a" * 24
    secret = "sk-" + "a" * 32
    data = (task + " " + secret + " /home/example/workspace/test").encode()
    result = audit.sanitize(data, Path("/home/example/workspace"), Path("/home/example")).decode()
    assert task in result
    assert secret not in result
    assert "/SOURCE_WORKSPACE/test" in result


def test_nested_usage():
    usage = {"input_tokens": 20, "output_tokens": 5,
             "input_tokens_details": {"cached_tokens": 4},
             "output_tokens_details": {"reasoning_tokens": 3}}
    call = audit.telemetry([req(), resp(usage=usage)])[0]
    assert (call["total_tokens"], call["cached_input_tokens"], call["reasoning_output_tokens"]) == (25, 4, 3)
