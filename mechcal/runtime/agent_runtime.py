from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from mechcal.runtime.retry import RetryPolicy, SchemaRetryError, SchemaRetryRunner
from mechcal.schemas.failures import FailureKind, FailureReport

T = TypeVar("T", bound=BaseModel)


class BaseAgentRuntime(Generic[T]):
    """Shared schema-first runtime for MechCAL agents."""

    def __init__(
        self,
        *,
        agent_name: str,
        response_model: type[T],
        retry_policy: RetryPolicy | None = None,
        validators: list[Callable[[T], None]] | None = None,
    ) -> None:
        self.agent_name = agent_name
        self.response_model = response_model
        self.retry_runner = SchemaRetryRunner(
            response_model,
            policy=retry_policy,
            validators=validators,
        )

    def run_schema_retry(self, producer: Callable[[str | None], Any]) -> T:
        return self.retry_runner.run(producer)

    def schema_failure(
        self,
        exc: SchemaRetryError,
        *,
        kind: FailureKind,
    ) -> FailureReport:
        return FailureReport(
            kind=kind,
            message=(
                f"{self.agent_name} failed to produce a valid "
                f"{self.response_model.__name__} after schema retries."
            ),
            recoverable=True,
            details={
                "agent_name": self.agent_name,
                "response_model": self.response_model.__name__,
                "attempt_errors": list(exc.errors),
            },
        )
