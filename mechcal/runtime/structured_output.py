"""Strict Planner wire schemas derived from the existing runtime contracts.

Open maps and tuple paths are JSON strings on the wire, not guessed fixed keys.
They are decoded and validated against the original schema before runtime use.
"""

from copy import deepcopy
import json
from urllib.parse import urlparse

from jsonschema import Draft202012Validator


def provider_response_format(response_format, base_url):
    """Inline local definitions for DeepSeek; keep the original local validator."""
    if (urlparse(base_url).hostname != "api.deepseek.com" or not response_format
            or response_format.get("type") != "json_schema"):
        return response_format
    result = deepcopy(response_format)
    schema = result["json_schema"]["schema"]
    definitions = schema.get("$defs", {})

    def expand(value, active=()):
        if isinstance(value, list):
            return [expand(item, active) for item in value]
        if not isinstance(value, dict):
            return value
        if "$ref" in value:
            ref = value["$ref"]
            if not ref.startswith("#/$defs/") or ref in active:
                raise ValueError("DeepSeek schema requires non-recursive local $defs references")
            name = ref.removeprefix("#/$defs/").replace("~1", "/").replace("~0", "~")
            if name not in definitions:
                raise ValueError("DeepSeek schema contains an unresolved local reference")
            target = expand(definitions[name], (*active, ref))
            siblings = expand({k: v for k, v in value.items() if k != "$ref"}, active)
            # A reference and its siblings are conjunctive. Never silently replace
            # a target constraint with a different sibling constraint.
            for key in target.keys() & siblings.keys() - {"description", "title"}:
                if target[key] != siblings[key]:
                    raise ValueError(f"Cannot inline conflicting reference constraint: {key}")
            return {**target, **siblings}
        return {key: expand(item, active) for key, item in value.items() if key != "$defs"}

    result["json_schema"]["schema"] = expand(schema)
    Draft202012Validator.check_schema(result["json_schema"]["schema"])
    return result


def request_payload(payload, response_format):
    """Do not send the same output contract in both user text and wire schema."""
    result = deepcopy(payload)
    if not response_format or response_format.get("type") != "json_schema":
        return result
    name = response_format.get("json_schema", {}).get("name")
    if name not in {"planner_response_v1", "mechcal_result_review_v1", "support_audit_v1"}:
        return result
    result["output_contract"] = f"Use the {name} JSON Schema supplied in response_format."
    if name == "planner_response_v1" and "memory_access" in result:
        result["memory_access"].pop("query_schema", None)
        if result.get("context_version") == "planner_delta_v1":
            kinds = [kind for branch in response_format["json_schema"]["schema"]["properties"]["response"]["anyOf"]
                     for kind in branch["properties"]["kind"]["enum"]]
            result["memory_access"]["query_contract"] = (
                "Use only the currently permitted branches: " + ", ".join(kinds)
                + ". Query arrays obey the remaining allowances in the schema."
            )
        else:
            result["memory_access"]["query_contract"] = (
            "The memory_query branch of response_format."
            if result["memory_access"].get("queries_remaining", 0) > 0
            else "No query branch; allowance exhausted."
            )
    return result


def _json_string(node):
    return (not any(k in node for k in ("type", "$ref", "anyOf", "oneOf", "const", "enum"))
            or "prefixItems" in node
            or (node.get("type") == "object" and "properties" not in node
                and node.get("additionalProperties") is not False))


def _strict(node):
    if ("properties" in node and node.get("type") != "object"
            or node.get("type") == "object" and ("anyOf" in node or "oneOf" in node)):
        raise ValueError("Strict schemas require complete object alternatives, not property-only intersections")
    if _json_string(node):
        result = {
            "type": "string",
            "minLength": 1,
            "description": "A JSON-encoded value, never an empty string. Encode an empty object as {} "
            "and an empty array as []. Logical schema: "
            + json.dumps(node, ensure_ascii=False, separators=(",", ":")),
        }
        if node.get("type") == "object":
            result["pattern"] = r"^\s*\{[\s\S]*\}\s*$"
        elif node.get("type") == "array" or "prefixItems" in node:
            result["pattern"] = r"^\s*\[[\s\S]*\]\s*$"
        return result
    result = {k: deepcopy(v) for k, v in node.items()
              if k not in {"default", "discriminator", "$schema", "title"}}
    if "const" in result:
        value = result.pop("const")
        result["enum"] = [value]
        result.setdefault("type", "string" if isinstance(value, str) else
                          "boolean" if isinstance(value, bool) else
                          "null" if value is None else "number")
    if "oneOf" in result:
        result["anyOf"] = result.pop("oneOf")
    if "anyOf" in result:
        result["anyOf"] = [_strict(item) for item in result["anyOf"]]
    if result.get("type") == "object":
        if result.get("properties") and result.get("additionalProperties") not in (None, False):
            raise ValueError("Mixed fixed/open objects need an explicit wire contract")
        result["properties"] = {k: _strict(v) for k, v in result.get("properties", {}).items()}
        result["required"] = list(result["properties"])
        result["additionalProperties"] = False
    if "items" in result:
        result["items"] = _strict(result["items"])
    if "$defs" in result:
        result["$defs"] = {k: _strict(v) for k, v in result["$defs"].items()}
    return result


def _namespace(value, prefix):
    if isinstance(value, list):
        return [_namespace(item, prefix) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key == "$ref" and item.startswith("#/$defs/"):
            result[key] = "#/$defs/" + prefix + item.removeprefix("#/$defs/")
        elif key == "$defs":
            result[key] = {prefix + k: _namespace(v, prefix) for k, v in item.items()}
        else:
            result[key] = _namespace(item, prefix)
    return result


def _walk(value, node, source, wire, *, encode, path=()):
    if "$ref" in node:
        return _walk(value, source["$defs"][node["$ref"].removeprefix("#/$defs/")],
                     source, wire, encode=encode, path=path)
    if _json_string(node):
        if encode:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                f"Invalid JSON encoding at {list(path)}: expected {node.get('type', 'JSON value')}; "
                "empty objects must be encoded as '{}', not an empty string."
            ) from exc
    choices = node.get("anyOf", node.get("oneOf"))
    if choices:
        for choice in choices:
            schema = {**(choice if encode else _strict(choice)),
                      "$defs": (source if encode else wire).get("$defs", {})}
            if Draft202012Validator(schema).is_valid(value):
                return _walk(value, choice, source, wire, encode=encode, path=path)
        raise ValueError("Value does not match any declared output branch")
    if isinstance(value, dict) and node.get("type") == "object":
        if encode:
            # Required wire fields use their declared defaults, never inferred science.
            value = dict(value)
            for key, prop in node.get("properties", {}).items():
                if key not in value and "default" in prop:
                    value[key] = deepcopy(prop["default"])
        return {key: _walk(item, node["properties"][key], source, wire, encode=encode, path=(*path, key))
                for key, item in value.items()}
    if isinstance(value, list) and node.get("type") == "array":
        return [_walk(item, node["items"], source, wire, encode=encode, path=(*path, i))
                for i, item in enumerate(value)]
    return value


class PlannerOutputContract:
    """One versioned envelope permits the decision and currently allowed queries."""

    def __init__(self, payload):
        from mechcal.capabilities.discovery import DescribeToolsRequest
        from mechcal.runtime.planner_contract_policy import can_reopen_context, context_read_schema, decision_schema

        self.logical = {"decision": decision_schema(payload["output_contract"], payload)}
        if payload.get("stage") == "planner_context_lookup":
            self.logical["decision"] = context_read_schema(self.logical["decision"], payload)
        memory = payload.get("memory_access", {})
        delta = payload.get("context_version") == "planner_delta_v1"
        if delta:
            from mechcal.schemas.memory import ContextNeeded
            if can_reopen_context(payload):
                self.logical["context_needed"] = ContextNeeded.model_json_schema()
            if payload.get("stage") in {"planner_assimilate", "planner_question_act"}:
                from mechcal.schemas.memory import ContextReadPlan
                self.logical["context_read"] = context_read_schema(ContextReadPlan.model_json_schema(), payload)
        elif memory.get("queries_remaining", 0) > 0:
            self.logical["memory_query"] = deepcopy(memory["query_schema"])
        if not delta and payload.get("tool_discovery", {}).get("remaining_batches", 0) > 0:
            self.logical["describe_tools"] = DescribeToolsRequest.model_json_schema()
        self.wire = {key: _strict(schema) for key, schema in self.logical.items()}
        variants, definitions = [], {}
        for key, schema in self.wire.items():
            schema = _namespace(schema, key + "_")
            definitions.update(schema.pop("$defs", {}))
            variants.append({
                "type": "object", "additionalProperties": False,
                "required": ["kind", "data"],
                "properties": {"kind": {"type": "string", "enum": [key]}, "data": schema},
            })
        if payload.get("stage") == "planner_context_lookup":
            # Batched and dependent reads share the exact MemoryQuery contract.
            # Reuse its schema definition instead of sending it twice.
            for variant in variants:
                data = variant["properties"]["data"]
                for name, definition in definitions.items():
                    if data == definition:
                        variant["properties"]["data"] = {"$ref": "#/$defs/" + name}
                        break
        self.schema = {
            "type": "object", "additionalProperties": False, "$defs": definitions,
            "required": ["response"], "properties": {"response": {"anyOf": variants}},
        }
        Draft202012Validator.check_schema(self.schema)

    def response_format(self):
        return {"type": "json_schema", "json_schema": {
            "name": "planner_response_v1", "strict": True, "schema": self.schema,
        }}

    def decode(self, response):
        Draft202012Validator(self.schema).validate(response)
        envelope = response["response"]
        kind = envelope["kind"]
        result = _walk(envelope["data"], self.logical[kind], self.logical[kind], self.wire[kind], encode=False)
        Draft202012Validator(self.logical[kind]).validate(result)
        return result

    def encode(self, response, kind="decision"):
        """For test fixtures/replays; live model responses are never filled in."""
        Draft202012Validator(self.logical[kind]).validate(response)
        result = {"response": {"kind": kind, "data": _walk(
            response, self.logical[kind], self.logical[kind], self.wire[kind], encode=True,
        )}}
        Draft202012Validator(self.schema).validate(result)
        return result
