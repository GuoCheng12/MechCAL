from __future__ import annotations

import json
import logging
from typing import Any

import tiktoken
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr

from mechcal.baselines.chemgraph_mas.budget import SharedBudget

LOGGER = logging.getLogger(__name__)


class BudgetedChatOpenAI(ChatOpenAI):
    """ChatOpenAI with one shared accounting boundary for every MAS role."""

    _shared_budget: SharedBudget = PrivateAttr()

    def __init__(self, *, shared_budget: SharedBudget, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._shared_budget = shared_budget

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        allowance = self._reserve(messages)
        if self.use_responses_api:
            kwargs = _responses_initial_request_kwargs(messages, kwargs)
        kwargs["max_tokens"] = min(int(kwargs.get("max_tokens") or allowance), allowance)
        result = super()._generate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
        _normalize_responses_content(result)
        self._record(result)
        if self.use_responses_api and _chat_result_is_empty(result):
            if _recover_post_tool_empty_response(result, messages):
                return result
            retry_allowance = self._reserve(messages)
            retry_kwargs = _responses_empty_retry_kwargs(kwargs)
            retry_kwargs["max_tokens"] = min(
                int(retry_kwargs.get("max_tokens") or retry_allowance),
                retry_allowance,
            )
            result = super()._generate(
                messages,
                stop=stop,
                run_manager=run_manager,
                **retry_kwargs,
            )
            _normalize_responses_content(result)
            self._record(result)
        return result

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        allowance = self._reserve(messages)
        if self.use_responses_api:
            kwargs = _responses_initial_request_kwargs(messages, kwargs)
        kwargs["max_tokens"] = min(int(kwargs.get("max_tokens") or allowance), allowance)
        result = await super()._agenerate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
        _normalize_responses_content(result)
        self._record(result)
        if self.use_responses_api and _chat_result_is_empty(result):
            if _recover_post_tool_empty_response(result, messages):
                return result
            retry_allowance = self._reserve(messages)
            retry_kwargs = _responses_empty_retry_kwargs(kwargs)
            retry_kwargs["max_tokens"] = min(
                int(retry_kwargs.get("max_tokens") or retry_allowance),
                retry_allowance,
            )
            result = await super()._agenerate(
                messages,
                stop=stop,
                run_manager=run_manager,
                **retry_kwargs,
            )
            _normalize_responses_content(result)
            self._record(result)
        return result

    def _reserve(self, messages: list[BaseMessage]) -> int:
        return self._shared_budget.reserve_model_call(
            role=_role_from_messages(messages),
            estimated_prompt_tokens=_estimate_message_tokens(messages),
        )

    def _record(self, result: ChatResult) -> None:
        prompt_tokens, completion_tokens, total_tokens = _chat_result_usage(result)
        self._shared_budget.record_model_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )


def _role_from_messages(messages: list[BaseMessage]) -> str:
    if not messages:
        return "unknown"
    content = str(messages[0].content)
    for role in ("PLANNER", "EXECUTOR", "AGGREGATOR"):
        if f"[ROLE: {role}]" in content:
            return role.lower()
    return "unknown"


def _estimate_message_tokens(messages: list[BaseMessage]) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    serializable = []
    for message in messages:
        serializable.append(
            {
                "type": message.type,
                "content": message.content,
                "additional_kwargs": message.additional_kwargs,
            }
        )
    text = json.dumps(serializable, ensure_ascii=False, default=str)
    return len(encoding.encode(text)) + 8 * len(messages)


def _chat_result_usage(result: ChatResult) -> tuple[int, int, int]:
    token_usage = (result.llm_output or {}).get("token_usage") or {}
    prompt_tokens = _usage_int(token_usage, "prompt_tokens", "input_tokens")
    completion_tokens = _usage_int(
        token_usage,
        "completion_tokens",
        "output_tokens",
    )
    total_tokens = _usage_int(token_usage, "total_tokens")
    if prompt_tokens or completion_tokens or total_tokens:
        return prompt_tokens, completion_tokens, total_tokens

    if result.generations:
        usage = getattr(result.generations[0].message, "usage_metadata", None) or {}
        prompt_tokens = _usage_int(usage, "input_tokens", "prompt_tokens")
        completion_tokens = _usage_int(
            usage,
            "output_tokens",
            "completion_tokens",
        )
        total_tokens = _usage_int(usage, "total_tokens")
    return prompt_tokens, completion_tokens, total_tokens


def _usage_int(payload: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int | float):
            return max(0, int(value))
    return 0


def _normalize_responses_content(result: ChatResult) -> None:
    for generation in result.generations:
        content = generation.message.content
        if not isinstance(content, list):
            continue
        fragments: list[str] = []
        for block in content:
            if isinstance(block, str):
                fragments.append(block)
                continue
            if not isinstance(block, dict):
                continue
            if block.get("type") in {"text", "output_text"}:
                text = block.get("text")
                if isinstance(text, str):
                    fragments.append(text)
            elif block.get("type") == "refusal":
                refusal = block.get("refusal")
                if isinstance(refusal, str):
                    fragments.append(refusal)
        generation.message.content = "\n".join(fragments)
        if fragments or generation.message.tool_calls:
            continue
        LOGGER.warning(
            "Responses API returned no output text or tool call: "
            "block_types=%s metadata=%s usage=%s",
            [
                block.get("type")
                for block in content
                if isinstance(block, dict)
            ],
            generation.message.response_metadata,
            generation.message.usage_metadata,
        )


def _chat_result_is_empty(result: ChatResult) -> bool:
    if not result.generations:
        return True
    message = result.generations[0].message
    content = message.content
    has_content = bool(content) if isinstance(content, list) else bool(str(content).strip())
    return not has_content and not message.tool_calls


def _responses_initial_request_kwargs(
    messages: list[BaseMessage],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    request_kwargs = dict(kwargs)
    role = _role_from_messages(messages)
    has_tool_result = any(isinstance(message, ToolMessage) for message in messages)
    if role == "executor" and request_kwargs.get("tools") and not has_tool_result:
        request_kwargs["tool_choice"] = "required"
    elif role in {"planner", "aggregator"} and not request_kwargs.get("tools"):
        request_kwargs["response_format"] = {"type": "json_object"}
    return request_kwargs


def _responses_empty_retry_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    retry_kwargs = dict(kwargs)
    if retry_kwargs.get("tools"):
        retry_kwargs["tool_choice"] = "required"
    else:
        retry_kwargs["response_format"] = {"type": "json_object"}
    return retry_kwargs


def _recover_post_tool_empty_response(
    result: ChatResult,
    messages: list[BaseMessage],
) -> bool:
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        content = message.content
        if isinstance(content, str):
            text = _compact_tool_message_for_planner(content)
        else:
            text = json.dumps(content, ensure_ascii=False, default=str)
        if not text.strip() or not result.generations:
            return False
        result.generations[0].message.content = text
        return True
    return False


def _compact_tool_message_for_planner(content: str) -> str:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return content
    if not isinstance(payload, dict):
        return content

    structured = payload.get("structured_results")
    compact_structured: dict[str, Any] = {}
    if isinstance(structured, dict):
        for key in ("selected_route", "result_name", "metrics"):
            if key in structured:
                compact_structured[key] = structured[key]

    compact_evidence: list[dict[str, Any]] = []
    evidence = payload.get("evidence")
    if isinstance(evidence, list):
        for item in evidence[:3]:
            if not isinstance(item, dict):
                continue
            compact_evidence.append(
                {
                    key: item[key]
                    for key in (
                        "claim",
                        "basis",
                        "support",
                        "status",
                        "summary",
                        "limits",
                    )
                    if key in item
                }
            )

    compact = {
        key: payload[key]
        for key in ("status", "capability_id", "selected_route", "summary")
        if key in payload
    }
    if compact_structured:
        compact["structured_results"] = compact_structured
    if compact_evidence:
        compact["evidence"] = compact_evidence
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
