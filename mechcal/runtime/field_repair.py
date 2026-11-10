"""Bounded field corrections: the model supplies values; unrelated data stays fixed."""

from copy import deepcopy
import json
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, ValidationError

from mechcal.runtime.retry import RejectedResponseError


class FieldCorrection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: list[StrictStr | StrictInt] = Field(min_length=1)
    value: Any


class ResponseCorrections(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repairs: list[FieldCorrection] = Field(min_length=1, max_length=64)


def correction_contract(response_model: type[BaseModel], paths: list[list], *, schema=None) -> dict:
    """Derive each replacement type from the same model that validates acceptance."""
    source = schema if schema is not None else response_model.model_json_schema()

    def resolve(node):
        while "$ref" in node or "anyOf" in node:
            if "$ref" in node:
                ref = node["$ref"]
                if not ref.startswith("#/$defs/"):
                    raise RuntimeError(f"Unsupported correction schema reference: {ref}")
                node = source["$defs"][ref.removeprefix("#/$defs/")]
            else:
                choices = [choice for choice in node["anyOf"] if choice.get("type") != "null"]
                if len(choices) != 1:
                    break
                node = choices[0]
        return node

    def descend(node, key):
        node = resolve(node)
        choices = node.get("oneOf", node.get("anyOf"))
        if choices:
            matches = []
            for choice in choices:
                try:
                    value = descend(choice, key)
                except RuntimeError:
                    continue
                if value not in matches:
                    matches.append(value)
            if not matches:
                raise RuntimeError("Correction field is absent from every schema branch.")
            return matches[0] if len(matches) == 1 else {"anyOf": matches}
        if isinstance(key, str) and key in node.get("properties", {}):
            return node["properties"][key]
        if isinstance(key, str) and isinstance(node.get("additionalProperties"), dict):
            return node["additionalProperties"]
        if type(key) is int and key >= 0 and node.get("type") == "array":
            return node["items"]
        raise RuntimeError(f"Correction field is absent from the response schema: {key}")

    variants = []
    for path in paths:
        node = source
        for key in path:
            node = descend(node, key)
        variants.append({
            "type": "object", "additionalProperties": False, "required": ["path", "value"],
            "properties": {
                "path": {"type": "array", "prefixItems": [{"const": key} for key in path],
                         "minItems": len(path), "maxItems": len(path)},
                "value": deepcopy(node),
            },
        })
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": source.get("$defs", {}),
        "type": "object", "additionalProperties": False, "required": ["repairs"],
        "properties": {"repairs": {"type": "array", "minItems": 1, "maxItems": 64,
                                   "items": {"anyOf": variants}}},
    }


def apply_corrections(original: dict, response: Any, allowed: list[list]) -> dict:
    corrections = ResponseCorrections.model_validate(response)
    paths = [tuple(item.path) for item in corrections.repairs]
    if len(paths) != len(set(paths)) or not set(paths) <= {tuple(p) for p in allowed}:
        raise ValueError("Use distinct authorized field paths only; do not rewrite unrelated fields.")
    if any(a != b and b[:len(a)] == a for a in paths for b in paths):
        raise ValueError("Overlapping correction paths are not allowed.")
    result = deepcopy(original)
    for item in corrections.repairs:
        parent = result
        for key in item.path[:-1]:
            parent = _lookup(parent, key)
        key = item.path[-1]
        if not (isinstance(parent, dict) and isinstance(key, str)):
            _lookup(parent, key)
        parent[key] = deepcopy(item.value)
    return result


def _lookup(parent: Any, key: str | int) -> Any:
    if isinstance(parent, dict) and isinstance(key, str) and key in parent:
        return parent[key]
    if isinstance(parent, list) and type(key) is int and 0 <= key < len(parent):
        return parent[key]
    raise ValueError("Correction path must address an existing field or list entry.")


class FieldRepairProducer:
    """Used inside the existing retry budget, not a separate retry loop."""

    def __init__(self, full: Callable, repair: Callable, *, response_model: type[BaseModel], event_sink=None):
        self.full = full
        self.repair = repair
        self.event_sink = event_sink
        self.response_model = response_model

    def __call__(self, feedback: str | None) -> Any:
        context = json.loads(feedback) if feedback else {}
        issues = context.get("issues", [])
        original = context.get("rejected_response")
        if isinstance(original, dict):
            issues = schema_repair_issues(self.response_model, original, issues)
        if not isinstance(original, dict) or not issues or not all(i.get("repair_paths") for i in issues):
            return self.full(feedback)
        allowed = list(dict.fromkeys(tuple(p) for i in issues for p in i["repair_paths"]))
        payload = {
            "stage": "planner_field_repair", "rejected_response": original,
            "issues": issues, "allowed_paths": [list(p) for p in allowed],
            "output_contract": correction_contract(self.response_model, [list(p) for p in allowed]),
            "last_failed_correction": context.get("repair_failure"),
        }
        try:
            response = self.repair(payload, feedback)
        except RejectedResponseError as exc:
            raise _correction_error(exc, original, issues, exc.response) from exc
        try:
            repaired = apply_corrections(original, response, payload["allowed_paths"])
            self.response_model.model_validate(repaired)
        except (ValueError, ValidationError) as exc:
            # Preserve the full rejected candidate and its authorized paths, even
            # when the correction itself is malformed. Never turn it into state.
            raise _correction_error(exc, original, issues, response) from exc
        if self.event_sink:
            self.event_sink({"kind": "field_repair_applied", "repairs": response["repairs"],
                             "response": repaired})
        return repaired


def schema_repair_issues(response_model, original, issues):
    """Authorize only schema-addressable leaves; never guess paths from error text."""
    scoped = []
    for issue in issues:
        path = list(issue.get("loc", ()))
        if issue.get("repair_paths") or not path or issue.get("type") == "extra_forbidden":
            scoped.append(issue)
            continue
        try:
            correction_contract(response_model, [path])
            parent = original
            for key in path[:-1]:
                parent = _lookup(parent, key)
            if not (isinstance(parent, dict) and isinstance(path[-1], str)):
                _lookup(parent, path[-1])
        except (ValueError, RuntimeError):
            scoped.append(issue)
        else:
            scoped.append({**issue, "repair_paths": [path]})
    return scoped


def _correction_error(error, original, issues, submitted):
    rejected = RejectedResponseError(f"Field correction rejected: {error}", original)
    rejected.issues = issues
    rejected.repair_failure = {
        "submitted_correction": submitted,
        "errors": (error.errors(include_url=False, include_context=False, include_input=False)
                   if isinstance(error, ValidationError) else [{"message": str(error)}]),
    }
    return rejected
