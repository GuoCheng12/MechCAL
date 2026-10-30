"""Native JSON contracts: reject open maps instead of encoding JSON in strings."""

from copy import deepcopy

from jsonschema import Draft202012Validator

from mechcal.runtime.structured_output import _strict


def native_format(model, name="mechcal_command_v2"):
    source = model.model_json_schema()

    def check(node):
        if isinstance(node, list):
            for child in node:
                check(child)
        elif isinstance(node, dict):
            if node.get("type") == "object" and "properties" not in node:
                if node.get("additionalProperties") not in (None, False):
                    raise ValueError("Model-writable maps need an explicit object/list contract.")
            if "prefixItems" in node:
                raise ValueError("Model-writable tuples need a typed list contract.")
            for child in node.values():
                check(child)

    check(source)
    schema = _strict(_complete_variants(source))
    if "JSON-encoded value" in str(schema):
        raise ValueError("Nested JSON strings are forbidden in command protocol.")
    Draft202012Validator.check_schema(schema)
    return {"type": "json_schema", "json_schema": {
        "name": name, "strict": True, "schema": schema,
    }}


def _complete_variants(value):
    """Expand registry field intersections without dropping array item types."""
    if isinstance(value, list):
        return [_complete_variants(v) for v in value]
    if not isinstance(value, dict):
        return value
    node = {k: _complete_variants(v) for k, v in value.items()}
    if node.get("type") != "object" or "anyOf" not in node:
        return node
    base = {k: v for k, v in node.items() if k != "anyOf"}
    variants = []
    for constraints in node["anyOf"]:
        if set(constraints) - {"properties", "required"}:
            raise ValueError("Unsupported registry intersection; define a complete typed alternative.")
        branch = deepcopy(base)
        for key, condition in constraints.get("properties", {}).items():
            prop = branch["properties"][key]
            if "type" in condition and "anyOf" in prop:
                alternatives = [v for v in prop["anyOf"] if v.get("type") == condition["type"]]
                if len(alternatives) != 1:
                    raise ValueError("Registry condition does not select one declared parameter type.")
                prop = {**{k: v for k, v in prop.items() if k != "anyOf"}, **alternatives[0]}
            branch["properties"][key] = {**prop, **condition}
        branch["required"] = list(dict.fromkeys([*branch.get("required", []),
                                                 *constraints.get("required", [])]))
        variants.append(branch)
    definitions = base.get("$defs")
    for branch in variants:
        branch.pop("$defs", None)
    return {"anyOf": variants, **({"$defs": definitions} if definitions else {})}
