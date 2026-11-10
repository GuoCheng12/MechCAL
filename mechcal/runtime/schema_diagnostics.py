"""Field-local schema feedback without dumping unrelated union alternatives."""


def schema_issues(error):
    if (error.validator == "additionalProperties" and error.validator_value is False
            and isinstance(error.instance, dict) and not error.schema.get("patternProperties")):
        extra = sorted(set(error.instance) - set(error.schema.get("properties", {})))
        if extra:
            return [{"path": [*error.absolute_path, name], "type": "extra_forbidden",
                     "code": "schema_additionalProperties", "requires_scientific_context": False,
                     "message": f"Remove undeclared field {name!r}."}
                    for name in extra]
    if error.validator in {"anyOf", "oneOf"} and isinstance(error.instance, dict):
        branches = error.schema[error.validator]
        # A discriminator must distinguish branches. Ordinary value enums (for
        # example judgment_change) are constraints, not branch selectors.
        choices_by_key = {}
        for branch in branches:
            for key, prop in branch.get("properties", {}).items():
                choices = prop.get("enum", [prop["const"]] if "const" in prop else [])
                if choices:
                    choices_by_key.setdefault(key, []).append(choices)
        selectors = {key for key, choices in choices_by_key.items()
                     if len(choices) == len(branches)
                     and all(not any(value in right for value in left)
                             for i, left in enumerate(choices) for right in choices[i + 1:])}
        matching = []
        for index, branch in enumerate(branches):
            discriminators = {
                key: prop.get("enum", [prop["const"]] if "const" in prop else [])
                for key, prop in branch.get("properties", {}).items()
                if key in selectors
            }
            if discriminators and all(error.instance.get(key) in choices for key, choices in discriminators.items()):
                matching.append(index)
        if len(matching) == 1:
            children = [child for child in error.context if child.schema_path[0] == matching[0]]
            return [issue for child in children for issue in schema_issues(child)]
    if error.validator in {"anyOf", "oneOf"}:
        # Nullable/ref alternatives often have no discriminator. A direct type
        # mismatch rules out a branch without guessing the intended field value.
        branches = error.schema[error.validator]
        matching = [index for index in range(len(branches))
                    if not any(child.schema_path[0] == index and child.validator == "type"
                               and not child.path for child in error.context)]
        if len(matching) == 1:
            children = [child for child in error.context if child.schema_path[0] == matching[0]]
            return [issue for child in children for issue in schema_issues(child)]
    message = error.message
    if error.validator == "maxItems":
        message = f"At most {error.validator_value} items; received {len(error.instance)}."
    elif error.validator in {"anyOf", "oneOf"}:
        message = "Value does not match the declared alternatives at this field."
    return [{"path": list(error.absolute_path), "code": "schema_" + str(error.validator), "message": message}]


def format_issues(issues):
    def pointer(path):
        return "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in path)
    return "\n".join(f"{pointer(issue['path'])}: {issue['message']}" for issue in issues)
