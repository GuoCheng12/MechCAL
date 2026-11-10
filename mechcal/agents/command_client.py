"""One recorded model operation, with no snapshot reconstruction or hidden retries."""

import json
import time

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from mechcal.runtime.command_operations import OperationRejected
from mechcal.runtime.command_schema import native_format
from mechcal.runtime.prompts import load_prompt
from mechcal.schemas.commands import CommentaryText


def schema_issues(schema, data):
    """Keep all errors, selecting typed command branches rather than other commands."""
    issues = []

    def resolve(node):
        if "$ref" in node:
            return schema["$defs"][node["$ref"].removeprefix("#/$defs/")]
        return node

    def collect(error):
        path = list(error.absolute_path)
        if error.validator == "anyOf" and isinstance(error.instance, dict):
            matches = []
            for i, node in enumerate(error.validator_value):
                props = resolve(node).get("properties", {})
                discriminants = {k: p["enum"] for k, p in props.items()
                                 if k in {"kind", "capability_id"} and "enum" in p}
                if discriminants and all(error.instance.get(k) in choices
                                         for k, choices in discriminants.items()):
                    matches.append(i)
            if len(matches) == 1:
                for child in error.context:
                    if list(child.schema_path)[0] == matches[0]:
                        collect(child)
                return
        if error.validator == "additionalProperties" and isinstance(error.instance, dict):
            for key in error.instance.keys() - error.schema.get("properties", {}).keys():
                issues.append(dict(path=[*path, key], code="unexpected_field",
                    message=f"Field {key!r} is not in this command's contract.",
                    remedy="Remove this field. Put scientific commentary in a declared text field.", record_id=None))
            return
        if error.validator == "required" and isinstance(error.instance, dict):
            for key in set(error.validator_value) - error.instance.keys():
                issues.append(dict(path=[*path, key], code="missing_field",
                    message=f"Required command field {key!r} is missing.",
                    remedy="Supply the declared field, without changing unrelated accepted records.", record_id=None))
            return
        issues.append(dict(path=path, code=error.validator, message=error.message,
                           remedy="Use the declared command field and allowed values.", record_id=None))

    for error in Draft202012Validator(schema).iter_errors(data):
        collect(error)
    # Some required-field errors share the same missing-key list.
    return list({json.dumps(i, sort_keys=True): i for i in issues}.values())


def ask(client, model, payload, store, *, role="planner"):
    prompt = load_prompt("planner_commands.md" if role == "planner" else "command_commentary.md")
    fmt = native_format(model, "mechcal_command_v2" if role == "planner" else "mechcal_commentary_v2")
    started = time.time()
    request_id = f"{role}-{time.time_ns()}"
    store.append_jsonl("history/command_requests.jsonl", {
        "event": "request", "id": request_id, "role": role, "timestamp": started,
        "system_prompt": prompt, "payload": payload, "response_format": fmt,
    })
    # API/transport errors propagate to the runtime. Invalid output is a local
    # operation outcome and never a zero scientific score.
    try:
        response = client.complete_json_result(
            system_prompt=prompt, payload=payload, response_format=fmt,
            preserve_invalid_response=True,
        )
    except Exception as exc:
        store.append_jsonl("history/command_requests.jsonl", {
            "event": "transport_error", "id": request_id, "role": role,
            "error_type": type(exc).__name__, "elapsed_seconds": time.time() - started,
        })
        raise
    store.append_jsonl("history/command_requests.jsonl", {
        "event": "response", "id": request_id, "role": role,
        "raw_text": response.raw_text, "reasoning_text": response.reasoning_text,
        "usage": response.usage, "parse_error": response.parse_error,
        "elapsed_seconds": time.time() - started, "finish_reason": response.finish_reason,
    })
    if response.parse_error or response.finish_reason in {"length", "max_output_tokens"}:
        raise OperationRejected([dict(
            path=[], code="incomplete_response",
            message=response.parse_error or "The provider marked this response as truncated.",
            remedy="Resubmit a complete command. No operation ran.", record_id=None,
        )])
    # Parse the entire raw answer, not a convenient nested fragment or a guessed
    # completion. The transport may have preserved a malformed partial object.
    try:
        data = json.loads(response.raw_text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise OperationRejected([dict(
            path=[], code="invalid_json", message="The command is not one complete JSON object.",
            remedy="Resubmit the command. Nothing was executed or committed.", record_id=None,
        )]) from exc
    issues = schema_issues(native_format(model, strict=False)["json_schema"]["schema"], data)
    if issues:
        raise OperationRejected(issues)
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise OperationRejected([dict(path=list(e["loc"]), code=e["type"], message=e["msg"],
            remedy="Correct the indicated fields; no operation ran.", record_id=None)
            for e in exc.errors(include_input=False, include_url=False)]) from exc


def ask_worker_text(client, payload, store):
    """Worker writes prose; the runtime owns the typed record and its source."""
    return _ask_text(client, payload, store, role="worker", prompt_name="command_worker_text.md")


def ask_reviewer_text(client, payload, store, *, role):
    if role not in {"coverage_reviewer", "reviewer", "final_reviewer"}:
        raise ValueError("Unknown scientific review role.")
    return _ask_text(client, payload, store, role=role, prompt_name="command_reviewer_text.md")


def _ask_text(client, payload, store, *, role, prompt_name):
    prompt = load_prompt(prompt_name)
    started = time.time()
    request_id = f"{role}-{time.time_ns()}"
    store.append_jsonl("history/command_requests.jsonl", {
        "event": "request", "id": request_id, "role": role, "transport": "plain_text",
        "timestamp": started, "system_prompt": prompt, "payload": payload,
    })
    try:
        response = client.complete_text_result(system_prompt=prompt, payload=payload)
    except Exception as exc:
        store.append_jsonl("history/command_requests.jsonl", {
            "event": "transport_error", "id": request_id, "role": role, "transport": "plain_text",
            "error_type": type(exc).__name__, "elapsed_seconds": time.time() - started,
        })
        raise
    store.append_jsonl("history/command_requests.jsonl", {
        "event": "response", "id": request_id, "role": role, "transport": "plain_text",
        "raw_text": response.raw_text, "reasoning_text": response.reasoning_text,
        "usage": response.usage, "response_error": response.response_error,
        "finish_reason": response.finish_reason, "elapsed_seconds": time.time() - started,
    })
    if response.response_error or response.finish_reason != "stop" or not response.raw_text.strip():
        raise OperationRejected([dict(path=[], code="unavailable_explanation" if role == "worker" else "unavailable_review",
            message=response.response_error or "No complete commentary text was returned.",
            remedy="Saved calculations and accepted judgment are retained. Complete commentary may be requested later.",
            record_id=None)])
    # This is an internal record, not a schema or identifier the model must fill.
    return CommentaryText(text=response.raw_text)
