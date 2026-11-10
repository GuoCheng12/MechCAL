from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class SchemaRetryError(RuntimeError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("\n".join(errors))
        self.errors = errors


class ComponentResponseRejected(SchemaRetryError):
    """Bounded model correction exhausted without invalidating accepted state."""

    failure_kind = "component_response_rejected"


class AuditorReviewRejected(ComponentResponseRejected):
    failure_kind = "support_audit_rejected"

    def __str__(self) -> str:
        return "Support Auditor failed to produce a valid review:\n" + super().__str__()


class PlannerCommandRejected(ComponentResponseRejected):
    """Bounded command correction exhausted; accepted state remains resumable."""

    failure_kind = "planner_command_rejected"

    def __init__(self, errors: list[str], *, action: str) -> None:
        super().__init__(errors)
        self.action = action


class ModelContractError(ValueError):
    """An explicitly identified model-output violation, not a local program error."""


class RequestValidationError(ModelContractError):
    """Correctable request violations, not transport or program failures."""

    def __init__(self, issues: list[dict[str, Any]]) -> None:
        self.issues = issues
        super().__init__("\n".join(item["message"] for item in issues))


class RejectedResponseError(ModelContractError):
    def __init__(self, message: str, response: Any, *, issues=None) -> None:
        super().__init__(message)
        self.response = response
        self.issues = issues or []


def complete_recorded_json(
    client: Any, *, event_sink: Callable[[dict[str, Any]], None] | None = None,
    compact_context: bool = False, **kwargs: Any,
) -> Any:
    """Persist scientific requests and responses, not client settings or credentials."""
    from mechcal.runtime.llm import OpenAIJsonClient

    if compact_context:
        from mechcal.runtime.context_compaction import compact_payload, serialized
        from mechcal.runtime.structured_output import request_payload
        original = request_payload(kwargs.get("payload", {}), kwargs.get("response_format"))
        kwargs["payload"] = compact_payload(original)
        if event_sink is not None:
            event_sink({"kind": "context_projection", "version": "lossless_data_v1",
                        "before_bytes": len(serialized(original).encode()),
                        "after_bytes": len(serialized(kwargs["payload"]).encode())})
    if event_sink is not None:
        from copy import deepcopy
        from mechcal.runtime.structured_output import request_payload
        request = {key: deepcopy(kwargs[key]) for key in
                   ("system_prompt", "payload", "response_format", "schema_feedback") if key in kwargs}
        request["payload"] = request_payload(request.get("payload", {}), request.get("response_format"))
        event_sink({"kind": "model_request", "request": request})
    if isinstance(client, OpenAIJsonClient):
        completion = client.complete_json_result(**kwargs, preserve_invalid_response=True)
        event = {"kind": "model_response", "response": completion.data,
                 "raw_text": completion.raw_text, "usage": completion.usage,
                 "finish_reason": completion.finish_reason, "parse_error": completion.parse_error}
        if (completion.response_metadata or {}).get("transport_normalization"):
            event["transport_normalization"] = completion.response_metadata["transport_normalization"]
    else:
        event = {"kind": "model_response", "response": client.complete_json(**kwargs)}
    if event_sink is not None:
        event_sink(event)
    if event.get("parse_error"):
        raise RejectedResponseError(event["parse_error"], event["raw_text"])
    return event["response"]


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3


class SchemaRetryRunner(Generic[T]):
    def __init__(
        self,
        response_model: type[T],
        *,
        policy: RetryPolicy | None = None,
        validators: list[Callable[[T], None]] | None = None,
        on_validation_error: Callable[[dict[str, Any]], None] | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        repair_context: bool = False,
        strict_errors: bool = False,
        normalized_feedback: bool = False,
        separate_validation_budget: bool = False,
    ) -> None:
        self.response_model = response_model
        self.policy = policy or RetryPolicy()
        self.validators = validators or []
        self.on_validation_error = on_validation_error
        self.event_sink = event_sink
        self.repair_context = repair_context
        self.strict_errors = strict_errors or repair_context
        self.normalized_feedback = normalized_feedback
        self.separate_validation_budget = separate_validation_budget

    def run(self, producer: Callable[[str | None], Any]) -> T:
        feedback: str | None = None
        errors: list[str] = []
        failures = {"format": 0, "validator": 0}
        max_attempts = self.policy.max_attempts * (2 if self.separate_validation_budget else 1)
        if self.separate_validation_budget:
            max_attempts -= 1
        attempt = 0
        for attempt in range(1, max_attempts + 1):
            if self.event_sink is not None:
                self.event_sink({"kind": "attempt_started", "attempt": attempt,
                                 "max_attempts": max_attempts})
            payload = None
            validation_stage = "producer"
            try:
                payload = producer(feedback)
                validation_stage = "schema"
                parsed = self.response_model.model_validate(payload)
                validation_stage = "validator"
                for validator in self.validators:
                    validator(parsed)
            except (ValidationError, ValueError) as exc:
                # Recorded producers identify response errors explicitly. Configuration
                # or logging errors before validation must not trigger paid correction.
                if (self.strict_errors
                        and not isinstance(exc, (ModelContractError, ValidationError))):
                    raise
                error = str(exc)
                errors.append(error)
                rejected = getattr(exc, "response", payload)
                if self.normalized_feedback and validation_stage == "validator":
                    # Explicit record reuse is expanded before validation. Repair
                    # paths must address that expanded candidate, not the sparse wire response.
                    rejected = parsed.model_dump(mode="json")
                rejected = rejected.model_dump(mode="json") if isinstance(rejected, BaseModel) else rejected
                issues = (exc.errors(include_url=False, include_context=False, include_input=False)
                          if isinstance(exc, ValidationError) else getattr(exc, "issues", []))
                bucket = "validator" if validation_stage == "validator" else "format"
                failures[bucket] += 1
                event = {
                    "kind": "validation_error", "attempt": attempt, "max_attempts": max_attempts,
                    "validation_stage": validation_stage, "error_type": type(exc).__name__,
                    "error": error, "issues": issues, "response": rejected,
                }
                context = {"error": error, "issues": issues, "rejected_response": rejected}
                if self.separate_validation_budget:
                    event["failure_counts"] = dict(failures)
                    event["max_failures_per_stage"] = self.policy.max_attempts
                repair_failure = getattr(exc, "repair_failure", None)
                if repair_failure is not None:
                    event["repair_failure"] = context["repair_failure"] = repair_failure
                feedback = json.dumps(context) if self.repair_context else error
                if self.on_validation_error is not None:
                    self.on_validation_error(event)
                if self.event_sink is not None:
                    self.event_sink(event)
                if self.separate_validation_budget and failures[bucket] >= self.policy.max_attempts:
                    break
                continue
            if self.event_sink is not None:
                self.event_sink({"kind": "accepted", "attempt": attempt})
            return parsed
        if self.event_sink is not None:
            self.event_sink({"kind": "exhausted", "attempt": attempt, "errors": errors})
        raise SchemaRetryError(errors)
