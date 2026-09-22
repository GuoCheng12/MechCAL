from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Any


class BudgetExceeded(RuntimeError):
    def __init__(self, budget_name: str, message: str) -> None:
        super().__init__(message)
        self.budget_name = budget_name


@dataclass(frozen=True)
class BudgetLimits:
    max_model_calls: int
    max_tool_calls: int
    max_total_tokens: int
    max_output_tokens_per_call: int = 2400

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive.")


@dataclass
class BudgetUsage:
    model_calls: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model_calls_by_role: Counter[str] = field(default_factory=Counter)
    tool_calls_by_owner: Counter[str] = field(default_factory=Counter)
    tool_calls_by_capability: Counter[str] = field(default_factory=Counter)
    token_overrun: int = 0


class SharedBudget:
    """Thread-safe per-case budget shared by all ChemGraph roles and tools."""

    def __init__(self, limits: BudgetLimits) -> None:
        self.limits = limits
        self._usage = BudgetUsage()
        self._lock = Lock()

    def reserve_model_call(self, *, role: str, estimated_prompt_tokens: int) -> int:
        with self._lock:
            if self._usage.model_calls >= self.limits.max_model_calls:
                raise BudgetExceeded(
                    "model_calls",
                    f"Model-call budget exhausted at {self._usage.model_calls} calls.",
                )
            remaining_tokens = (
                self.limits.max_total_tokens
                - _reservation_margin_tokens(self.limits.max_total_tokens)
                - self._usage.total_tokens
            )
            if remaining_tokens <= max(1, estimated_prompt_tokens):
                raise BudgetExceeded(
                    "total_tokens",
                    "Token budget cannot accommodate the next model input.",
                )
            self._usage.model_calls += 1
            self._usage.model_calls_by_role[role] += 1
            return min(
                self.limits.max_output_tokens_per_call,
                max(1, remaining_tokens - estimated_prompt_tokens),
            )

    def record_model_usage(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ) -> None:
        with self._lock:
            normalized_total = max(total_tokens, prompt_tokens + completion_tokens)
            self._usage.prompt_tokens += max(0, prompt_tokens)
            self._usage.completion_tokens += max(0, completion_tokens)
            self._usage.total_tokens += max(0, normalized_total)
            self._usage.token_overrun = max(
                0,
                self._usage.total_tokens - self.limits.max_total_tokens,
            )
            if self._usage.token_overrun:
                raise BudgetExceeded(
                    "total_tokens",
                    "Actual model usage exceeded the per-case token budget.",
                )

    def reserve_tool_call(self, *, owner: str, capability_id: str) -> None:
        with self._lock:
            if self._usage.tool_calls >= self.limits.max_tool_calls:
                raise BudgetExceeded(
                    "tool_calls",
                    f"Tool-call budget exhausted at {self._usage.tool_calls} calls.",
                )
            self._usage.tool_calls += 1
            self._usage.tool_calls_by_owner[owner] += 1
            self._usage.tool_calls_by_capability[capability_id] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "limits": asdict(self.limits),
                "usage": {
                    "model_calls": self._usage.model_calls,
                    "tool_calls": self._usage.tool_calls,
                    "prompt_tokens": self._usage.prompt_tokens,
                    "completion_tokens": self._usage.completion_tokens,
                    "total_tokens": self._usage.total_tokens,
                    "model_calls_by_role": dict(self._usage.model_calls_by_role),
                    "tool_calls_by_owner": dict(self._usage.tool_calls_by_owner),
                    "tool_calls_by_capability": dict(
                        self._usage.tool_calls_by_capability
                    ),
                    "token_overrun": self._usage.token_overrun,
                },
            }


def _reservation_margin_tokens(max_total_tokens: int) -> int:
    return min(10_000, max(1, max_total_tokens // 20))
