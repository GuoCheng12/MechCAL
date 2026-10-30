from __future__ import annotations

import datetime as dt
import inspect
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mechcal.runtime.api_failures import exception_type_chain

OMIT_REASONING_EFFORT_VALUES = {"", "auto", "default", "none", "omit"}
TELEMETRY_ENV = "MECHCAL_LLM_TELEMETRY_PATH"
TELEMETRY_KEY_CANDIDATES = {
    "case_id",
    "public_case_id",
    "round_id",
    "stage",
    "task",
    "agent_name",
    "capability_id",
    "route",
    "tool_loading_mode",
}


@dataclass(frozen=True)
class OpenAICompatibleSettings:
    base_url: str
    model: str
    api_key: str
    reasoning_effort: str | None = None
    timeout_seconds: float = 60.0
    max_retries: int = 2
    max_tokens: int | None = None
    trust_env: bool = False
    temperature: float | None = 0.0
    top_p: float | None = 1.0
    seed: int | None = None
    api_protocol: str = "responses"
    normalize_json_suffix: bool = False

    def __post_init__(self) -> None:
        if self.api_protocol not in {"responses", "chat_completions"}:
            raise ValueError("api_protocol must be responses or chat_completions")

    @classmethod
    def from_env(cls) -> OpenAICompatibleSettings:
        return cls(
            base_url=os.environ.get("MECHCAL_OPENAI_BASE_URL", "").strip(),
            model=os.environ.get("MECHCAL_OPENAI_MODEL", "deepseek-v4-flash").strip(),
            api_key=os.environ.get("MECHCAL_OPENAI_API_KEY", "").strip(),
            reasoning_effort=_reasoning_effort_from_env(),
            timeout_seconds=float(os.environ.get("MECHCAL_OPENAI_TIMEOUT", "60")),
            max_retries=int(os.environ.get("MECHCAL_OPENAI_MAX_RETRIES", "2")),
            max_tokens=_optional_int_from_env("MECHCAL_OPENAI_MAX_TOKENS"),
            trust_env=_env_bool("MECHCAL_OPENAI_TRUST_ENV", default=False),
            temperature=_optional_float_from_env(
                "MECHCAL_OPENAI_TEMPERATURE", default=0.0
            ),
            top_p=_optional_float_from_env("MECHCAL_OPENAI_TOP_P", default=1.0),
            seed=_optional_int_from_env("MECHCAL_OPENAI_SEED"),
            api_protocol=os.environ.get("MECHCAL_OPENAI_API_PROTOCOL", "responses").strip(),
            normalize_json_suffix=_env_bool("MECHCAL_OPENAI_NORMALIZE_JSON_SUFFIX", default=False),
        )

    def missing_fields(self) -> list[str]:
        required = {
            "MECHCAL_OPENAI_BASE_URL": self.base_url,
            "MECHCAL_OPENAI_MODEL": self.model,
            "MECHCAL_OPENAI_API_KEY": self.api_key,
        }
        return [name for name, value in required.items() if not value]


@dataclass(frozen=True)
class JsonCompletionResult:
    data: dict[str, Any]
    raw_text: str
    finish_reason: str | None
    usage: dict[str, object]
    parse_error: str | None = None
    reasoning_text: str = ""
    response_metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class TextCompletionResult:
    raw_text: str
    finish_reason: str | None
    usage: dict[str, object]
    response_error: str | None = None
    reasoning_text: str = ""
    response_metadata: dict[str, object] | None = None


def run_llm_smoke(settings: OpenAICompatibleSettings) -> str:
    request_args: dict[str, Any] = {
        "model": settings.model,
        "messages": [
            {
                "role": "user",
                "content": "Reply with the single word OK.",
            }
        ],
    }
    if settings.reasoning_effort is not None:
        request_args["reasoning_effort"] = settings.reasoning_effort
    _add_generation_args(request_args, settings)
    response = create_completion(settings, request_args)
    validate_response(response)
    return response_text(response).strip()


class OpenAIJsonClient:
    def __init__(self, settings: OpenAICompatibleSettings | None = None) -> None:
        self.settings = settings or OpenAICompatibleSettings.from_env()

    def is_configured(self) -> bool:
        return not self.settings.missing_fields()

    def complete_text_result(self, *, system_prompt: str, payload: dict[str, Any]) -> TextCompletionResult:
        """Natural-language delivery, without schema injection or JSON parsing."""
        if not self.is_configured():
            raise RuntimeError("OpenAI-compatible settings are incomplete: " + ", ".join(self.settings.missing_fields()))
        user_content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        args = {"model": self.settings.model, "messages": [
            {"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]}
        if self.settings.reasoning_effort is not None:
            args["reasoning_effort"] = self.settings.reasoning_effort
        _add_generation_args(args, self.settings)
        telemetry_path = _telemetry_path_from_env()
        call_id = uuid.uuid4().hex
        attempts = max(1, self.settings.max_retries + 1)
        for attempt in range(1, attempts + 1):
            started = time.perf_counter()
            response, error, response_error = None, None, None
            content = ""
            try:
                response = create_completion(self.settings, args)
                content = response_text(response)
                try:
                    validate_response(response)
                    if _response_finish_reason(response) != "stop":
                        raise ValueError("Text response did not complete normally.")
                    if not content.strip():
                        raise ValueError("No final explanation text was returned.")
                    if hasattr(response, "output"):
                        unexpected = any(item.type not in {"message", "reasoning"} for item in response.output)
                    else:
                        message = response.choices[0].message
                        unexpected = bool(getattr(message, "tool_calls", None) or getattr(message, "refusal", None))
                    if unexpected:
                        raise ValueError("Unexpected tool call or refusal in a text-only response.")
                except ValueError as exc:
                    response_error = str(exc)
                return TextCompletionResult(
                    raw_text=content, finish_reason=_response_finish_reason(response),
                    usage=_response_usage(response) or {}, response_error=response_error,
                    reasoning_text=_response_reasoning(response), response_metadata=_response_metadata(response))
            except Exception as exc:
                error = exc
                if attempt == attempts or not _is_transient_llm_exception(exc):
                    raise
                time.sleep(min(2.0 * attempt, 8.0))
            finally:
                if telemetry_path is not None:
                    _append_telemetry_event(telemetry_path, {
                        "schema_version": "1.0.0", "record_type": "llm_complete_text_call",
                        "call_id": call_id, "attempt_index": attempt, "attempt_count": attempts,
                        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
                        "model": self.settings.model, "api_protocol": self.settings.api_protocol,
                        "base_url_host": _url_host(self.settings.base_url),
                        "reasoning_effort": self.settings.reasoning_effort,
                        "timeout_seconds": self.settings.timeout_seconds,
                        "configured_max_retries": self.settings.max_retries, "max_tokens": self.settings.max_tokens,
                        "temperature": self.settings.temperature, "top_p": self.settings.top_p, "seed": self.settings.seed,
                        "latency_seconds": round(time.perf_counter() - started, 4),
                        "success": error is None and response_error is None,
                        "error_type": type(error).__name__ if error else "IncompleteTextResponse" if response_error else None,
                        "error_cause_types": exception_type_chain(error),
                        "system_prompt_chars": len(system_prompt), "payload_json_bytes": len(user_content.encode()),
                        "response_format_bytes": 0, "response_text_chars": len(content),
                        "finish_reason": _response_finish_reason(response), "usage": _response_usage(response),
                        "payload_metadata": _payload_metadata(payload), "response_metadata": _response_metadata(response),
                    })
        raise RuntimeError("Text completion exhausted without a response.")

    def complete_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        schema_feedback: str | None = None,
        response_format: dict[str, Any] | None = None,
        omit_response_format: bool = False,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.complete_json_result(
            system_prompt=system_prompt,
            payload=payload,
            schema_feedback=schema_feedback,
            response_format=response_format,
            omit_response_format=omit_response_format,
            extra_body=extra_body,
        ).data

    def complete_json_result(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        schema_feedback: str | None = None,
        response_format: dict[str, Any] | None = None,
        omit_response_format: bool = False,
        extra_body: dict[str, Any] | None = None,
        answer_tag: str | None = None,
        parse_after_tag: str | None = None,
        preserve_invalid_response: bool = False,
    ) -> JsonCompletionResult:
        if not self.is_configured():
            missing = ", ".join(self.settings.missing_fields())
            raise RuntimeError(f"OpenAI-compatible settings are incomplete: {missing}")

        from mechcal.runtime.structured_output import provider_response_format, request_payload
        response_format = provider_response_format(response_format, self.settings.base_url)
        user_payload = request_payload(payload, None if omit_response_format else response_format)
        if schema_feedback:
            user_payload["schema_feedback"] = schema_feedback
        user_content = json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))
        request_args: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
        }
        if not omit_response_format:
            request_args["response_format"] = response_format or {"type": "json_object"}
        if extra_body:
            request_args["extra_body"] = extra_body
        if self.settings.reasoning_effort is not None:
            request_args["reasoning_effort"] = self.settings.reasoning_effort
        _add_generation_args(request_args, self.settings)
        telemetry_path = _telemetry_path_from_env()
        call_id = uuid.uuid4().hex
        caller = _infer_telemetry_caller() if telemetry_path is not None else {}
        last_error: Exception | None = None
        attempts = max(1, int(self.settings.max_retries) + 1)
        for attempt_index in range(1, attempts + 1):
            data: dict[str, Any] = {}
            response: Any = None
            response_content = ""
            started = time.perf_counter()
            error: Exception | None = None
            parse_error: str | None = None
            normalization = None
            try:
                response = create_completion(self.settings, request_args)
                response_content = response_text(response)
                reasoning_content = _response_reasoning(response)
                json_text = response_content
                try:
                    validate_response(response)
                    if answer_tag is not None:
                        json_text = _extract_tagged_text(response_content, answer_tag)
                    elif parse_after_tag is not None:
                        json_text, inline_reasoning = _split_after_closing_tag(
                            response_content,
                            parse_after_tag,
                        )
                        if not reasoning_content:
                            reasoning_content = inline_reasoning
                    strict_json = bool(not omit_response_format and response_format
                                       and response_format.get("type") == "json_schema")
                    if strict_json and self.settings.normalize_json_suffix:
                        from mechcal.runtime.json_transport import normalize_redundant_suffix
                        json_text, normalization = normalize_redundant_suffix(
                            json_text, response_format["json_schema"]["schema"])
                    data = _parse_json_object(json_text, strict=strict_json)
                except ValueError as exc:
                    if not preserve_invalid_response:
                        raise
                    parse_error = f"{type(exc).__name__}: {exc}"
                    data = {}
                return JsonCompletionResult(
                    data=data,
                    raw_text=response_content,
                    finish_reason=_response_finish_reason(response),
                    usage=_response_usage(response) or {},
                    parse_error=parse_error,
                    reasoning_text=reasoning_content,
                    response_metadata=({**(_response_metadata(response) or {}),
                                        "transport_normalization": normalization}
                                       if normalization else _response_metadata(response)),
                )
            except Exception as exc:
                error = exc
                last_error = exc
                if attempt_index >= attempts or not _is_transient_llm_exception(exc):
                    raise
                time.sleep(min(2.0 * attempt_index, 8.0))
            finally:
                if telemetry_path is not None:
                    elapsed = round(time.perf_counter() - started, 4)
                    _append_telemetry_event(
                        telemetry_path,
                        {
                            "schema_version": "1.0.0",
                            "record_type": "llm_complete_json_call",
                            "call_id": call_id,
                            "attempt_index": attempt_index,
                            "attempt_count": attempts,
                            "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
                            "model": self.settings.model,
                            "api_protocol": self.settings.api_protocol,
                            "base_url_host": _url_host(self.settings.base_url),
                            "timeout_seconds": self.settings.timeout_seconds,
                            "configured_max_retries": self.settings.max_retries,
                            "max_tokens": self.settings.max_tokens,
                            "temperature": self.settings.temperature,
                            "top_p": self.settings.top_p,
                            "seed": self.settings.seed,
                            "reasoning_effort": self.settings.reasoning_effort,
                            "latency_seconds": elapsed,
                            "success": error is None and parse_error is None,
                            "error_type": (
                                type(error).__name__
                                if error
                                else "ResponseParseError"
                                if parse_error
                                else None
                            ),
                            "error_message": (
                                _truncate(str(error), 300)
                                if error
                                else _truncate(parse_error, 300)
                                if parse_error
                                else None
                            ),
                            "error_cause_types": exception_type_chain(error),
                            "schema_feedback_present": schema_feedback is not None,
                            "system_prompt_chars": len(system_prompt),
                            "payload_json_chars": len(user_content),
                            "payload_json_bytes": len(user_content.encode()),
                            "payload_field_bytes": {
                                key: len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())
                                for key, value in user_payload.items()
                            },
                            "response_format_bytes": len(json.dumps(
                                request_args.get("response_format", {}), ensure_ascii=False,
                                separators=(",", ":"),
                            ).encode()),
                            "response_json_chars": len(response_content),
                            "finish_reason": _response_finish_reason(response),
                            "payload_keys": sorted(str(key) for key in payload.keys()),
                            "payload_metadata": _payload_metadata(payload),
                            "response_action": _response_action(data),
                            "caller": caller,
                            "usage": _response_usage(response),
                            "response_metadata": _response_metadata(response),
                            "transport_normalization": normalization,
                        },
                    )
        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM JSON completion failed without an exception.")


def _response_action(data: dict[str, Any]) -> str | None:
    for key in ("planner_decision", "decision", "response"):
        if isinstance(data.get(key), dict):
            data = data[key]
            break
    value = data.get("action")
    return value if isinstance(value, str) else None


def create_completion(
    settings: OpenAICompatibleSettings,
    request_args: dict[str, Any],
    *,
    sdk_retries: int = 0,
) -> Any:
    import httpx
    from openai import OpenAI

    args = dict(request_args)
    if settings.api_protocol == "responses":
        args = _responses_request(args)
    with OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        max_retries=sdk_retries,
        http_client=httpx.Client(
            timeout=settings.timeout_seconds,
            trust_env=settings.trust_env,
        ),
    ) as client:
        if settings.api_protocol == "responses":
            raw = client.responses.with_raw_response.create(**args, timeout=settings.timeout_seconds)
            media_type = raw.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            response = (_streamed_response(raw.http_response, client)
                        if media_type == "text/event-stream" else raw.parse())
        else:
            response = client.chat.completions.create(**args, timeout=settings.timeout_seconds)
        _validate_response_shape(response, protocol=settings.api_protocol)
        return response


def _streamed_response(http_response: Any, client: Any) -> Any:
    """Read the terminal object when a relay streams despite stream=false."""
    import httpx
    from openai import Stream
    from openai.types.responses import ResponseStreamEvent

    terminal = None
    response_id = None
    try:
        with Stream(cast_to=ResponseStreamEvent, response=http_response, client=client) as events:
            for event in events:
                kind = getattr(event, "type", None)
                if kind == "error":
                    raise httpx.RemoteProtocolError("Responses stream reported an error event")
                if kind in {"response.created", "response.completed", "response.incomplete", "response.failed"}:
                    value = getattr(event, "response", None)
                    identity = getattr(value, "id", None)
                    if not identity or (response_id is not None and identity != response_id):
                        raise httpx.RemoteProtocolError("Responses stream has inconsistent response identity")
                    response_id = identity
                    if kind != "response.created":
                        if terminal is not None or getattr(value, "status", None) != kind.split(".")[1]:
                            raise httpx.RemoteProtocolError("Responses stream has inconsistent terminal state")
                        terminal = value
    except json.JSONDecodeError as exc:
        raise httpx.RemoteProtocolError("Responses stream contains malformed event JSON") from exc
    if terminal is None:
        raise httpx.RemoteProtocolError("Responses stream ended without a terminal response")
    # Existing validation handles failed/incomplete generations; never synthesize
    # completion from text deltas or accept a truncated stream as a final answer.
    return terminal


def _responses_request(args: dict[str, Any]) -> dict[str, Any]:
    if "seed" in args:
        raise ValueError("Responses API does not support seed; set MECHCAL_OPENAI_SEED=omit")
    args = dict(args)
    # Preserve existing prompts and local context ownership; never chain agent calls.
    args["input"] = args.pop("messages")
    args["store"] = False
    args["stream"] = False
    if "max_tokens" in args:
        args["max_output_tokens"] = args.pop("max_tokens")
    if "reasoning_effort" in args:
        args["reasoning"] = {"effort": args.pop("reasoning_effort")}
    if "response_format" in args:
        format_spec = dict(args.pop("response_format"))
        if format_spec.get("type") == "json_schema":
            format_spec = {"type": "json_schema", **format_spec["json_schema"]}
        args["text"] = {"format": format_spec}
    extra = args.get("extra_body") or {}
    reserved = set(args) | {
        "previous_response_id", "conversation", "tools", "tool_choice",
        "messages", "response_format", "reasoning_effort", "max_tokens", "seed",
    }
    if reserved.intersection(extra):
        raise ValueError("extra_body cannot override Responses protocol or context fields")
    return args


def response_text(response: Any) -> str:
    _validate_response_shape(response)
    if hasattr(response, "output"):
        return response.output_text or ""
    return response.choices[0].message.content or ""


def validate_response(response: Any) -> None:
    _validate_response_shape(response)
    if not hasattr(response, "output"):
        return
    if response.status != "completed":
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None) or response.status
        raise ValueError(f"Responses API returned an unfinished response: {reason}")
    for item in response.output:
        for content in getattr(item, "content", None) or []:
            if content.type == "refusal":
                raise ValueError("Responses API refused the request")
    if not response_text(response).strip():
        raise ValueError("Responses API returned no final text")


def _response_reasoning(response: Any) -> str:
    _validate_response_shape(response)
    if not hasattr(response, "output"):
        return getattr(response.choices[0].message, "reasoning_content", None) or ""
    return "\n".join(
        part.text
        for item in response.output if item.type == "reasoning"
        for part in (getattr(item, "summary", None) or [])
        if getattr(part, "text", None)
    )


def _validate_response_shape(response: Any, *, protocol: str | None = None) -> None:
    """Reject broken service envelopes before treating text as a model answer."""
    import httpx

    responses = hasattr(response, "output")
    if responses:
        valid = (isinstance(response.output, list)
                 and isinstance(getattr(response, "status", None), str))
        if valid:
            for item in response.output:
                if not isinstance(getattr(item, "type", None), str):
                    valid = False
                    break
                if item.type == "message":
                    content = getattr(item, "content", None)
                    if not isinstance(content, list) or any(
                        not isinstance(getattr(part, "type", None), str)
                        or (part.type == "output_text" and not isinstance(getattr(part, "text", None), str))
                        for part in content
                    ):
                        valid = False
                        break
    else:
        choices = getattr(response, "choices", None)
        valid = (isinstance(choices, list) and bool(choices)
                 and getattr(choices[0], "message", None) is not None
                 and isinstance(getattr(choices[0].message, "content", None), (str, type(None))))
    if protocol is not None:
        valid = valid and (responses == (protocol == "responses"))
    if not valid:
        # No response body or credentials in the error. Existing transport retries
        # and checkpoint pause handling apply; never parse a gateway page as JSON.
        length = f", length={len(response)}" if isinstance(response, (str, bytes)) else ""
        raise httpx.RemoteProtocolError(
            f"Malformed {protocol or 'LLM'} API response envelope: "
            f"type={type(response).__name__}{length}; expected a structured completion object"
        )


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _reasoning_effort_from_env() -> str | None:
    raw = os.environ.get("MECHCAL_OPENAI_REASONING_EFFORT")
    if raw is None:
        return None
    normalized = raw.strip()
    if normalized.lower() in OMIT_REASONING_EFFORT_VALUES:
        return None
    return normalized


def _optional_float_from_env(name: str, *, default: float | None) -> float | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip()
    if normalized.lower() in OMIT_REASONING_EFFORT_VALUES:
        return None
    return float(normalized)


def _optional_int_from_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    normalized = raw.strip()
    if normalized.lower() in OMIT_REASONING_EFFORT_VALUES:
        return None
    return int(normalized)


def _is_transient_llm_exception(exc: Exception) -> bool:
    import httpx
    from openai import APIStatusError
    from mechcal.runtime.api_failures import api_pause_reason, is_api_transport_error

    if api_pause_reason(exc):
        return False
    if is_api_transport_error(exc):
        return True
    if isinstance(exc, (APIStatusError, httpx.HTTPStatusError)):
        status = getattr(exc, "status_code", exc.response.status_code)
        return status in {408, 409, 429} or 500 <= status < 600
    return False


def _parse_json_object(content: str, *, strict: bool = False) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        if strict:
            nearby = content[max(0, exc.pos - 80):exc.pos + 80]
            raise ValueError(
                "JSON Schema mode requires one complete JSON object as final text. "
                f"JSON syntax error: {exc.msg} at line {exc.lineno}, column {exc.colno} "
                f"(zero-based character offset {exc.pos}). Nearby text: {json.dumps(nearby)}. "
                "Correct the syntax and return the entire response_format envelope, preserving "
                "the intended scientific content. Do not add markup, prose, or a second object. "
                "Query branch names are JSON discriminator values, not callable functions."
            ) from exc
        parsed = _extract_json_object(content)
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON response must be a JSON object.")
    return parsed


def _extract_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    fenced = _extract_fenced_json(stripped)
    if fenced is not None:
        return _parse_json_object(fenced)
    start = stripped.find("{")
    if start < 0:
        raise ValueError("LLM response did not contain a JSON object.")
    in_string = False
    escaped = False
    depth = 0
    for index, char in enumerate(stripped[start:], start=start):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : index + 1]
                parsed = json.loads(candidate)
                if not isinstance(parsed, dict):
                    raise ValueError("Extracted JSON response must be an object.")
                return parsed
    raise ValueError("LLM response contained an unterminated JSON object.")


def _extract_fenced_json(content: str) -> str | None:
    fence_start = content.find("```")
    if fence_start < 0:
        return None
    body_start = content.find("\n", fence_start + 3)
    if body_start < 0:
        return None
    fence_end = content.find("```", body_start + 1)
    if fence_end < 0:
        return None
    return content[body_start + 1 : fence_end].strip()


def _extract_tagged_text(content: str, tag: str) -> str:
    start_token = f"<{tag}>"
    end_token = f"</{tag}>"
    start = content.find(start_token)
    if start < 0:
        raise ValueError(f"LLM response did not contain {start_token}.")
    start += len(start_token)
    end = content.find(end_token, start)
    if end < 0:
        raise ValueError(f"LLM response did not contain {end_token}.")
    return content[start:end].strip()


def _split_after_closing_tag(content: str, tag: str) -> tuple[str, str]:
    end_token = f"</{tag}>"
    end = content.rfind(end_token)
    if end < 0:
        return content, ""
    reasoning = content[:end].strip()
    start_token = f"<{tag}>"
    start = reasoning.rfind(start_token)
    if start >= 0:
        reasoning = reasoning[start + len(start_token) :].strip()
    return content[end + len(end_token) :].strip(), reasoning


def _add_generation_args(
    request_args: dict[str, Any], settings: OpenAICompatibleSettings
) -> None:
    if settings.max_tokens is not None:
        request_args["max_tokens"] = settings.max_tokens
    if settings.temperature is not None:
        request_args["temperature"] = settings.temperature
    if settings.top_p is not None:
        request_args["top_p"] = settings.top_p
    if settings.seed is not None:
        request_args["seed"] = settings.seed


def _telemetry_path_from_env() -> Path | None:
    raw = os.environ.get(TELEMETRY_ENV, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _append_telemetry_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def _infer_telemetry_caller() -> dict[str, str]:
    this_file = Path(__file__).resolve()
    for frame in inspect.stack()[2:]:
        filename = Path(frame.filename).resolve()
        if filename == this_file:
            continue
        module = frame.frame.f_globals.get("__name__", "")
        if str(module).startswith("openai"):
            continue
        return {
            "module": str(module),
            "function": frame.function,
            "file": str(filename),
            "line": str(frame.lineno),
        }
    return {}


def _payload_metadata(payload: dict[str, Any]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for key in TELEMETRY_KEY_CANDIDATES:
        value = _find_payload_value(payload, key)
        if value is not None:
            metadata[key] = _truncate(str(value), 120)
    return metadata


def _find_payload_value(payload: Any, key: str, *, depth: int = 0) -> Any:
    if depth > 4:
        return None
    if isinstance(payload, dict):
        if key in payload and _is_scalar(payload[key]):
            return payload[key]
        for value in payload.values():
            found = _find_payload_value(value, key, depth=depth + 1)
            if found is not None:
                return found
    if isinstance(payload, list):
        for item in payload[:20]:
            found = _find_payload_value(item, key, depth=depth + 1)
            if found is not None:
                return found
    return None


def _is_scalar(value: Any) -> bool:
    return isinstance(value, str | int | float | bool) or value is None


def _response_usage(response: Any) -> dict[str, int] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    values = {
        "prompt_tokens": getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", None)),
        "completion_tokens": getattr(
            usage, "completion_tokens", getattr(usage, "output_tokens", None)
        ),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
    input_details = getattr(usage, "input_tokens_details", None) or getattr(
        usage, "prompt_tokens_details", None
    )
    output_details = getattr(usage, "output_tokens_details", None) or getattr(
        usage, "completion_tokens_details", None
    )
    values["cached_tokens"] = getattr(input_details, "cached_tokens", None)
    values["reasoning_tokens"] = getattr(output_details, "reasoning_tokens", None)
    return {key: int(value) for key, value in values.items() if value is not None}


def _response_finish_reason(response: Any) -> str | None:
    if hasattr(response, "output"):
        if response.status == "completed":
            return "stop"
        reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
        return "length" if reason == "max_output_tokens" else str(reason or response.status)
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    finish_reason = getattr(choices[0], "finish_reason", None)
    return str(finish_reason) if finish_reason is not None else None


def _response_metadata(response: Any) -> dict[str, object] | None:
    if response is None:
        return None
    metadata = getattr(response, "metadata", None)
    if metadata is None and hasattr(response, "model_dump"):
        payload = response.model_dump()
        if isinstance(payload, dict):
            metadata = payload.get("metadata")
    if hasattr(metadata, "model_dump"):
        metadata = metadata.model_dump()
    if not isinstance(metadata, dict):
        metadata = {}
    if hasattr(response, "output"):
        metadata = {
            **metadata,
            "response_id": response.id,
            "model": response.model,
            "status": response.status,
            "api_protocol": "responses",
        }
    return {
        str(key): value
        for key, value in metadata.items()
        if _is_scalar(value)
    }


def _url_host(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc or parsed.path


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
