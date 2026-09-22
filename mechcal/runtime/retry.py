from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class SchemaRetryError(RuntimeError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("\n".join(errors))
        self.errors = errors


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
    ) -> None:
        self.response_model = response_model
        self.policy = policy or RetryPolicy()
        self.validators = validators or []

    def run(self, producer: Callable[[str | None], Any]) -> T:
        feedback: str | None = None
        errors: list[str] = []
        for _ in range(self.policy.max_attempts):
            try:
                payload = producer(feedback)
                parsed = self.response_model.model_validate(payload)
                for validator in self.validators:
                    validator(parsed)
                return parsed
            except (ValidationError, ValueError) as exc:
                feedback = str(exc)
                errors.append(feedback)
        raise SchemaRetryError(errors)
