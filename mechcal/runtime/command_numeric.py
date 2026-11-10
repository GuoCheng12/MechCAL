"""Source-bound arithmetic on explicit selections, never scientific inference."""

import math

from mechcal.runtime.command_memory import READ_BYTES, digest, json_bytes, resolve_pointer
from mechcal.runtime.command_operations import check_refs, protocol, put, record_index, reject
from mechcal.runtime.command_review import review_execution_context

MAX_ROWS = 100000
MAX_MATCH_INDICES = 32


def operands(command):
    return [command.source] if command.kind == "reduce_evidence" else [command.left, command.right]


def outcome_ids(outcome):
    if outcome.get("status") == "read_batch":
        return set().union(*(outcome_ids(row) for row in outcome["call_results"].values()))
    return {outcome["numeric_check"]["id"]} if outcome.get("status") == "numeric_analysis" else set()


def outcome_refs(outcome):
    if outcome.get("status") == "read_batch":
        return set().union(*(outcome_refs(row) for row in outcome["call_results"].values()))
    return {op["ref"] for op in outcome["numeric_check"]["operands"]} if outcome.get("status") == "numeric_analysis" else set()


def parent_context(value, path):
    if not path:
        return None
    parent_path = path.rsplit("/", 1)[0]
    parent = resolve_pointer(value, parent_path)
    if not isinstance(parent, dict):
        return None
    fields, deferred = {}, []
    for name, item in parent.items():
        escaped = str(name).replace("~", "~0").replace("/", "~1")
        if not isinstance(item, (list, dict)) and json_bytes(item) <= 256 and json_bytes(fields) < 2048:
            fields[name] = item
        else:
            deferred.append(parent_path + "/" + escaped)
    return {"path": parent_path, "scalar_fields": fields, "deferred_paths": deferred}


def selected_value(record, path, field, ref):
    try:
        return resolve_pointer(record, path)
    except (KeyError, IndexError, TypeError, ValueError):
        reject(field, "unknown_path", "Numeric source path does not exist: " + path,
               "Use read_evidence(view=directory) for an exact path; no missing value was inferred.", record_id=ref)


def finite_number(value, field, ref):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        reject(field, "numeric_operand", "Every selected value must be a finite numeric scalar.",
               "Choose the actual numeric field. Strings, booleans, missing values and non-finite values are not coerced or skipped.",
               record_id=ref)
    return value


def evaluate_operand(case, operand, field, *, scope, index):
    if operand.ref not in scope:
        reject([*field, "ref"], "out_of_scope", "Numeric source is outside this agent's read scope.",
               "Choose a permitted source ID; no other records were accessed.", record_id=operand.ref)
    check_refs(case, [operand.ref], [*field, "ref"], scalar=True, index=index)
    record = index.available[operand.ref]
    selected = selected_value(record, operand.path, [*field, "path"], operand.ref)
    indices = None
    if operand.reduction == "scalar":
        paths = [operand.path + operand.value_path]
    else:
        if not isinstance(selected, list) or not 1 <= len(selected) <= MAX_ROWS:
            reject([*field, "path"], "numeric_array", f"Reductions require a nonempty array with at most {MAX_ROWS} rows.",
                   "Select the original condition-specific array, not a displayed page.", record_id=operand.ref)
        indices = operand.row_indices if operand.row_indices is not None else list(range(len(selected)))
        if any(i >= len(selected) for i in indices):
            reject([*field, "row_indices"], "unknown_row", "A selected row is outside the original array.",
                   "Read the original array and use its zero-based row indices.", record_id=operand.ref)
        paths = [operand.path + "/" + str(i) + operand.value_path for i in indices]
    values = [finite_number(selected_value(record, path, field, operand.ref), field, operand.ref) for path in paths]
    try:
        value = (max(abs(number) for number in values) if operand.reduction == "max_abs" else
                 values[0] if operand.reduction == "scalar" else
                 {"min": min, "max": max, "sum": math.fsum}[operand.reduction](values))
    except OverflowError:
        reject(field, "numeric_overflow", "Reduction result is not finite.", "Report the original values instead.", record_id=operand.ref)
    matches = [] if operand.reduction == "sum" else [i for i, number in enumerate(values)
        if (abs(number) if operand.reduction == "max_abs" else number) == value]
    return {**operand.model_dump(), "source_version": digest(record), "value": value,
            "considered_count": len(values), "missing_or_skipped_count": 0,
            "array_size": len(selected) if indices is not None else None,
            "selection_scope": "explicit_rows" if operand.row_indices is not None else "complete_array" if indices is not None else "scalar",
            "values_digest": digest(values),
            "matching_indices": [indices[i] for i in matches[:MAX_MATCH_INDICES]] if indices is not None else [],
            "omitted_matching_indices": max(0, len(matches) - MAX_MATCH_INDICES) if indices is not None else 0,
            "matching_count": len(matches),
            "matches": [{"path": paths[i], "value": values[i], "context": parent_context(record, paths[i])}
                        for i in matches[:4]],
            "omitted_match_details": max(0, len(matches) - 4),
            "selection_context": parent_context(record, operand.path)}


def calculate(case, command, *, scope=None, index=None, allowance=READ_BYTES):
    index = record_index(case, index)
    scope = index.available.keys() if scope is None else scope
    fields = [["source"]] if command.kind == "reduce_evidence" else [["left"], ["right"]]
    source = [evaluate_operand(case, op, field, scope=scope, index=index) for op, field in zip(operands(command), fields)]
    value = source[0]["value"]
    if command.kind == "compare_evidence":
        right = source[1]["value"]
        if command.operation == "ratio" and right == 0:
            reject(["right"], "zero_denominator", "Ratio denominator is zero.", "Report the values or choose the intended nonzero comparison.")
        try:
            value = value / right if command.operation == "ratio" else value - right
            valid = math.isfinite(value)
        except OverflowError:
            valid = False
        if not valid:
            reject(["operation"], "numeric_overflow", "Arithmetic result is not finite.", "Report the original values instead.")
    result = {"case_id": case.case_id, "command": command.model_dump(), "operands": source, "value": value,
              "execution_context": review_execution_context(case, {op.ref for op in operands(command)}, index.available),
              "policy": "Deterministic arithmetic on the explicitly selected data, not new evidence. "
                        "An explicit subset describes only those rows; atom/fragment membership is not inferred. "
                        "Conditions remain source-specific. No unit conversion, state-character matching, "
                        "causal attribution or scientific comparability is inferred. Cite the original source IDs and paths."}
    result["id"] = "numeric:" + digest(result)
    outcome = {"status": "numeric_analysis", "numeric_check": result}
    if json_bytes(outcome) > allowance:
        reject([], "response_too_large", "Numeric result and provenance exceed the read allowance.",
               "Request fewer operations in this turn; no result was truncated.")
    return outcome, []


def remember(case, result):
    if result["case_id"] != case.case_id:
        raise RuntimeError("Numeric receipt belongs to another case")
    expected = "numeric:" + digest({k: v for k, v in result.items() if k != "id"})
    if result["id"] != expected:
        raise RuntimeError("Numeric receipt content does not match its identity")
    saved = dict(protocol(case).get("numeric_checks", {}))
    saved.pop(result["id"], None)
    saved[result["id"]] = result
    return put(case, numeric_checks=saved)


def view(case, *, scope=None, index=None, exclude_ids=()):
    index = record_index(case, index)
    entries, omitted, versions = [], 0, {}
    for key, row in reversed(list(protocol(case).get("numeric_checks", {}).items())):
        if row["case_id"] != case.case_id or row["id"] != key:
            raise RuntimeError("Numeric receipt identity or case ownership mismatch")
        if key in exclude_ids or (scope is not None and any(op["ref"] not in scope for op in row["operands"])):
            continue
        for op in row["operands"]:
            if op["ref"] not in versions:
                versions[op["ref"]] = digest(index.available[op["ref"]]) if op["ref"] in index.available else None
        if any(versions[op["ref"]] != op["source_version"] for op in row["operands"]):
            continue
        if len(entries) < 4 and json_bytes(entries + [row]) <= 8000:
            entries.append(row)
        else:
            omitted += 1
    return {"checks": entries, "omitted_checks": omitted,
            "policy": "Source-version-bound arithmetic, not independent evidence. Recompute older checks from their original sources if needed."}
