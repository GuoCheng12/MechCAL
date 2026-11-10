"""Small schema builders for the same field relationships enforced at runtime."""

from copy import deepcopy


NONBLANK = {"type": "string", "minLength": 1, "pattern": r"\S"}


def object_variants(schema, variants):
    """Publish complete object alternatives, not unsupported partial intersections."""
    base = deepcopy(schema)
    definitions = base.pop("$defs", None)
    branches = []
    for fields in variants:
        branch = deepcopy(base)
        for name, constraint in fields.items():
            prop = branch["properties"][name]
            if "type" in constraint or "anyOf" in constraint or "$ref" in constraint:
                prop = {key: value for key, value in prop.items()
                        if key in {"title", "description"}}
            branch["properties"][name] = {**prop, **deepcopy(constraint)}
        branch["required"] = list(dict.fromkeys([*branch.get("required", []), *fields]))
        branches.append(branch)
    result = {"anyOf": branches}
    if definitions is not None:
        result["$defs"] = definitions
    return result


def field_constraints(schema, variants):
    """Full JSON Schema for MCP cards, retaining their flat parameter index."""
    result = deepcopy(schema)
    result["anyOf"] = [{"required": list(fields), "properties": deepcopy(fields)} for fields in variants]
    return result
