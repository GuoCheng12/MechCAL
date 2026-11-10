"""Lossless request-local views. Never alter stored evidence or output contracts."""

from collections import Counter
from copy import deepcopy
import json


DATA_FIELDS = frozenset({
    "observations", "evidence_ledger", "pending_results", "tool_execution_result",
    "computations", "retrieved_memory", "execution_history", "calculation_history",
    "source_checks",
})
REF = "$mechcal_data"
TABLE = "$mechcal_table"
ENCODING = "context_encoding"
SHARED = "shared_context_data"
MIN_SHARED_BYTES = 384


def serialized(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _containers(value, *, strings=False):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _containers(child, strings=strings)
    elif isinstance(value, list):
        yield value
        for child in value:
            yield from _containers(child, strings=strings)
    elif strings and isinstance(value, str):
        yield value


def compact_payload(payload):
    """Share identical subtrees and tabulate rows, without dropping values.

    Legacy mode encodes only evidence data. Delta mode also shares identical
    judgment text and nested table cells; tool/output schemas and retrieval
    contracts remain literal. References stay inside this request, not a cache.
    """
    if ENCODING in payload or SHARED in payload:
        return deepcopy(payload)
    delta = payload.get("context_version") == "planner_delta_v1"
    excluded = {"output_contract", "memory_access", "tool_discovery", "capability_cards"}
    fields = [key for key in payload if key not in excluded] if delta else [key for key in payload if key in DATA_FIELDS]
    counts = Counter()
    for key in fields:
        for value in _containers(payload[key], strings=delta):
            # Do not interpret a tool's own field as one of our view markers.
            if isinstance(value, dict) and (REF in value or TABLE in value):
                return deepcopy(payload)
            wire = serialized(value)
            if len(wire.encode()) >= MIN_SHARED_BYTES:
                counts[wire] += 1
    shared, ids = {}, {}
    tables = 0

    def encode(value, *, share=True):
        nonlocal tables
        if not isinstance(value, (dict, list)) and not (delta and isinstance(value, str)):
            return value
        wire = serialized(value)
        if share and counts[wire] > 1:
            if wire not in ids:
                rid = f"D{len(ids) + 1}"
                ids[wire] = rid
                shared[rid] = encode(value, share=False)
            return {REF: ids[wire]}
        if isinstance(value, dict):
            return {key: encode(child) for key, child in value.items()}
        if isinstance(value, str):
            return value
        # Only rectangular scalar tables. Missing fields are never filled with null,
        # and row order, units, atom/state indices and numerical precision survive.
        if len(value) >= (4 if delta else 8) and all(isinstance(row, dict) and row for row in value):
            columns = list(value[0])
            if all(row.keys() == value[0].keys()
                   and (delta or all(not isinstance(cell, (dict, list)) for cell in row.values())) for row in value):
                table = {TABLE: {"columns": columns,
                                "rows": [[encode(row[key]) if delta else row[key] for key in columns] for row in value]}}
                if len(serialized(table)) + 64 < len(wire):
                    tables += 1
                    return table
        return [encode(child) for child in value]

    result = deepcopy(payload)
    for key in fields:
        result[key] = encode(payload[key])
    if not shared and not tables:
        return result
    result[SHARED] = shared
    result[ENCODING] = {
        "version": "lossless_data_v1",
        "fields": fields,
        "read_as": (
            "$mechcal_data refers to shared_context_data in THIS request. "
            "$mechcal_table is a list of objects: zip columns with each row, preserving row order. "
            "All values, conditions, failures and limitations are retained. "
            "For citations/source_paths use the ORIGINAL evidence/result ID and field path, "
            "not D identifiers or table columns/rows. These are display conventions, not evidence."
        ),
    }
    # Small inputs must not grow merely to carry encoding instructions.
    return result if len(serialized(result)) < len(serialized(payload)) else deepcopy(payload)


def expand_payload(payload):
    """Offline integrity check: reconstruct the exact pre-view request data."""
    encoding = payload.get(ENCODING)
    if not encoding or encoding.get("version") != "lossless_data_v1":
        return deepcopy(payload)
    shared = payload[SHARED]

    def expand(value, active=()):
        if isinstance(value, list):
            return [expand(child, active) for child in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {REF}:
            rid = value[REF]
            if rid in active or rid not in shared:
                raise ValueError(f"Invalid request-local data reference: {rid}")
            return expand(shared[rid], (*active, rid))
        if set(value) == {TABLE}:
            table = value[TABLE]
            return [dict(zip(table["columns"], [expand(cell, active) for cell in row], strict=True))
                    for row in table["rows"]]
        return {key: expand(child, active) for key, child in value.items()}

    result = deepcopy(payload)
    for key in encoding["fields"]:
        result[key] = expand(result[key])
    result.pop(ENCODING)
    result.pop(SHARED)
    return result
