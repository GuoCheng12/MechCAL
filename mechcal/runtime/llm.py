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

OMIT_REASONING_EFFORT_VALUES = {"", "auto", "default", "none", "omit"}
TELEMETRY_ENV = "MECHCAL_LLM_TELEMETRY_PATH"
TELEMETRY_KEY_CANDIDATES = {
    "case_id",
    "public_case_id",
    "round_id",
    "task",
    "agent_name",
    "capability_id",
    "route",
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


def run_llm_smoke(settings: OpenAICompatibleSettings) -> str:
    import httpx
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        max_retries=0,
        http_client=httpx.Client(
            timeout=settings.timeout_seconds,
            trust_env=settings.trust_env,
        ),
    )
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
    response = client.chat.completions.create(
        **request_args,
        timeout=settings.timeout_seconds,
    )
    message = response.choices[0].message
    return (message.content or "").strip()


class OpenAIJsonClient:
    def __init__(self, settings: OpenAICompatibleSettings | None = None) -> None:
        self.settings = settings or OpenAICompatibleSettings.from_env()

    def is_configured(self) -> bool:
        return not self.settings.missing_fields()

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

        import httpx
        from openai import OpenAI

        client = OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            max_retries=0,
            http_client=httpx.Client(
                timeout=self.settings.timeout_seconds,
                trust_env=self.settings.trust_env,
            ),
        )
        user_payload = dict(payload)
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
            response: Any = None
            response_content = ""
            started = time.perf_counter()
            error: Exception | None = None
            parse_error: str | None = None
            try:
                response = client.chat.completions.create(
                    **request_args,
                    timeout=self.settings.timeout_seconds,
                )
                choice = response.choices[0]
                response_content = choice.message.content or ""
                reasoning_content = (
                    getattr(choice.message, "reasoning_content", None) or ""
                )
                json_text = response_content
                try:
                    if answer_tag is not None:
                        json_text = _extract_tagged_text(response_content, answer_tag)
                    elif parse_after_tag is not None:
                        json_text, inline_reasoning = _split_after_closing_tag(
                            response_content,
                            parse_after_tag,
                        )
                        if not reasoning_content:
                            reasoning_content = inline_reasoning
                    data = _parse_json_object(json_text)
                except ValueError as exc:
                    if not preserve_invalid_response:
                        raise
                    parse_error = f"{type(exc).__name__}: {exc}"
                    data = {}
                return JsonCompletionResult(
                    data=data,
                    raw_text=response_content,
                    finish_reason=(
                        str(choice.finish_reason)
                        if choice.finish_reason is not None
                        else None
                    ),
                    usage=_response_usage(response) or {},
                    parse_error=parse_error,
                    reasoning_text=reasoning_content,
                    response_metadata=_response_metadata(response),
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
                            "schema_feedback_present": schema_feedback is not None,
                            "system_prompt_chars": len(system_prompt),
                            "payload_json_chars": len(user_content),
                            "response_json_chars": len(response_content),
                            "finish_reason": _response_finish_reason(response),
                            "payload_keys": sorted(str(key) for key in payload.keys()),
                            "payload_metadata": _payload_metadata(payload),
                            "caller": caller,
                            "usage": _response_usage(response),
                            "response_metadata": _response_metadata(response),
                        },
                    )
        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM JSON completion failed without an exception.")


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
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in (
            "timeout",
            "timed out",
            "readtimeout",
            "connecttimeout",
            "temporarily unavailable",
            "connection reset",
            "connection aborted",
            "server disconnected",
            "rate limit",
            "429",
            "500",
            "502",
            "503",
            "504",
        )
    )


def _parse_json_object(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
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
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
    return {key: int(value) for key, value in values.items() if value is not None}


def _response_finish_reason(response: Any) -> str | None:
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
        return None
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
