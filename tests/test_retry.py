import pytest
from pydantic import BaseModel

from mechcal.runtime.retry import RetryPolicy, SchemaRetryError, SchemaRetryRunner


class Response(BaseModel):
    value: int


def test_diagnostics_preserve_response_and_reset_it_after_producer_error():
    events = []
    feedbacks = []

    def producer(feedback):
        feedbacks.append(feedback)
        if len(feedbacks) == 1:
            return {"value": "invalid"}
        if len(feedbacks) == 2:
            raise ValueError("No parsed response in this attempt")
        return {"value": 3}

    result = SchemaRetryRunner(Response, on_validation_error=events.append).run(producer)
    assert result.value == 3
    assert [e["response"] for e in events] == [{"value": "invalid"}, None]
    assert [e["validation_stage"] for e in events] == ["schema", "producer"]
    assert feedbacks == [None, events[0]["error"], events[1]["error"]]


def test_diagnostics_do_not_extend_retry_limit():
    events = []
    runner = SchemaRetryRunner(Response, policy=RetryPolicy(max_attempts=2),
                               on_validation_error=events.append)
    with pytest.raises(SchemaRetryError) as error:
        runner.run(lambda _: {})
    assert len(error.value.errors) == len(events) == 2
    assert [e["attempt"] for e in events] == [1, 2]


def test_diagnostic_storage_failure_is_not_a_model_retry():
    calls = []

    def sink(event):
        calls.append(event)
        raise OSError("diagnostic storage unavailable")

    with pytest.raises(OSError, match="storage unavailable"):
        SchemaRetryRunner(Response, on_validation_error=sink).run(lambda _: {})
    assert len(calls) == 1


def test_default_runner_remains_compatible_without_diagnostics():
    assert SchemaRetryRunner(Response).run(lambda _: {"value": 1}).value == 1
    with pytest.raises(SchemaRetryError) as error:
        SchemaRetryRunner(Response).run(lambda _: {})
    assert len(error.value.errors) == 3


def test_format_failures_do_not_spend_semantic_correction_opportunity():
    from mechcal.runtime.retry import ModelContractError

    replies = iter([{}, {"value": "bad"}, {"value": -1}, {"value": -2}, {"value": 1}])
    events = []
    def validate(value):
        if value.value < 0:
            raise ModelContractError("value must be nonnegative")
    result = SchemaRetryRunner(Response, validators=[validate], separate_validation_budget=True,
                               event_sink=events.append).run(lambda _: next(replies))
    assert result.value == 1
    failures = [e for e in events if e["kind"] == "validation_error"]
    assert failures[-1]["failure_counts"] == {"format": 2, "validator": 2}
    assert events[-1] == {"kind": "accepted", "attempt": 5}


@pytest.mark.parametrize("sequence,counts", [
    ([{}, {}, {}], {"format": 3, "validator": 0}),
    ([{"value": -1}] * 3, {"format": 0, "validator": 3}),
    ([{}, {"value": -1}, {}, {"value": -1}, {}], {"format": 3, "validator": 2}),
    ([{"value": -1}, {}, {"value": -1}, {}, {"value": -1}], {"format": 2, "validator": 3}),
])
def test_split_budgets_are_cumulative_and_stop_at_either_limit(sequence, counts):
    from mechcal.runtime.retry import ModelContractError

    events = []
    def validate(value):
        raise ModelContractError("Rejected executable contract")
    replies = iter(sequence)
    with pytest.raises(SchemaRetryError) as exc:
        SchemaRetryRunner(Response, validators=[validate], separate_validation_budget=True,
                          event_sink=events.append).run(lambda _: next(replies))
    assert len(exc.value.errors) == len(sequence)
    assert events[-2]["failure_counts"] == counts
    assert events[-1]["kind"] == "exhausted"
    assert events[-1]["attempt"] <= 5


def test_split_budget_does_not_retry_program_failure():
    def producer(_):
        raise ValueError("local configuration broken")
    with pytest.raises(ValueError, match="local configuration"):
        SchemaRetryRunner(Response, repair_context=True,
                          separate_validation_budget=True).run(producer)
