"""Auditable transport cleanup without completing or changing JSON values."""

import json

from jsonschema import Draft202012Validator


def normalize_redundant_suffix(text, schema):
    """Only discard closing delimiters after a complete, schema-valid root object."""
    start = len(text) - len(text.lstrip(" \t\r\n"))
    try:
        value, end = json.JSONDecoder().raw_decode(text, idx=start)
    except json.JSONDecodeError:
        return text, None
    suffix = text[end:]
    if (not isinstance(value, dict) or not suffix.strip()
            or any(char not in "}] \t\r\n" for char in suffix)):
        return text, None
    # Reject malformed fields even when the redundant suffix is harmless.
    if next(Draft202012Validator(schema).iter_errors(value), None) is not None:
        return text, None
    return text[:end], {
        "kind": "redundant_json_closing_suffix",
        "start_offset": end,
        "removed_suffix": suffix,
        "schema_validated": True,
        "values_modified": False,
    }
