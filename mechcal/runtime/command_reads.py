"""Bounded read receipts shared by Planner and scoped Worker retrieval."""

from mechcal.runtime.command_memory import READ_BYTES, json_bytes, remember_reads, record_layout, resolve_pointer
from mechcal.runtime.command_operations import OperationRejected, read_records, record_index
from mechcal.schemas.commands import ReadEvidence


READ_KINDS = {"read_evidence", "reduce_evidence", "compare_evidence"}


def query_refs(command):
    if command.kind == "read_batch":
        return [ref for call in command.calls if call.command for ref in query_refs(call.command)]
    if command.kind == "read_evidence":
        return [row.ref for row in command.records]
    if command.kind == "read_context":
        return ["context:planner"] * len(command.fields)
    if command.kind == "reduce_evidence":
        return [command.source.ref]
    return [command.left.ref, command.right.ref]


def query_results(case, command, *, allowed_refs=None, allowance=READ_BYTES, index=None):
    if command.kind == "read_evidence":
        return read_selection_results(case, command, allowed_refs=allowed_refs, allowance=allowance, index=index)
    from mechcal.runtime.command_numeric import calculate
    return calculate(case, command, scope=allowed_refs, allowance=allowance, index=index)


def bounded_query(case, selection, *, allowance, index):
    """Byte-bound query pages. Formal scientific citations still use strict read_records."""
    if allowance < 256:
        raise OperationRejected([dict(path=["records", 0], code="response_too_large",
            message="This read turn has no remaining data allowance.", record_id=selection.ref,
            remedy="Keep the accepted fields and request the remaining selections in a later read.")])

    def attempt(query):
        try:
            row = read_records(case, ReadEvidence(kind="read_evidence", records=[query]), index=index)[0]
            if json_bytes(row) <= allowance:
                return row
        except OperationRejected as exc:
            if any(issue["code"] != "response_too_large" for issue in exc.issues):
                raise
        return None

    row = attempt(selection)
    if row is not None:
        return row
    if selection.view == "data" and selection.page and len(selection.paths or [""]) == 1:
        # Find the longest whole-row prefix that fits. Never cut values or treat
        # an oversized row as missing evidence; expose its child directory instead.
        best, low, high = None, 1, selection.page.limit
        while low <= high:
            size = (low + high) // 2
            query = selection.model_copy(update={"page": selection.page.model_copy(update={"limit": size})})
            candidate = attempt(query)
            if candidate is None or json_bytes(candidate) + 160 > allowance:
                high = size - 1
            else:
                best, low = candidate, size + 1
        if best is not None:
            best["requested_page"] = selection.page.model_dump()
            best["page_policy"] = "Byte-limited whole rows. Continue at pages.next_offset; this is not the full list."
            return best
        path = (selection.paths or [""])[0]
        values = resolve_pointer(index.available[selection.ref], path)
        offset = selection.page.offset
        layout_path = path + "/" + str(offset) if offset < len(values) else path
        value = values[offset] if offset < len(values) else values
        raise OperationRejected([dict(path=["records", 0, "page"], code="nested_page_required",
            message="One whole row exceeds this read allowance; no values were returned.", record_id=selection.ref,
            remedy="Use an exact child read below, then page a nested list. Other rows remain available.",
            details={"record_layout": record_layout(value, layout_path, limit=8)})])
    # Retain the strict reader's actionable field layout when the source itself
    # exceeds the global limit, rather than replacing it with a generic error.
    read_records(case, ReadEvidence(kind="read_evidence", records=[selection]), index=index)
    raise OperationRejected([dict(path=["records", 0, "paths", 0], code="response_too_large",
        message="This field exceeds the remaining read-turn allowance.", record_id=selection.ref,
        remedy="Use a directory, narrower fields or a smaller page; accepted fields are retained.")])


def read_selection_results(case, command, *, allowed_refs=None, allowance=READ_BYTES, index=None):
    index = record_index(case, index)
    rows, issues, queries, values = [], [], [], []
    for i, query in enumerate(command.records):
        # Read-only requests may retain valid fields. Atomic scientific commits
        # still use strict read_records and reject any invalid citation.
        split = bool(query.paths) and not (query.view == "data" and query.page and len(query.paths) != 1)
        selections = [query.model_copy(update={"paths": [path]}) for path in query.paths] if split else [query]
        accepted, local, pages, valid_paths, page_metadata, coverage = {}, [], {}, [], {}, {}
        actual_page = query.page
        for j, selection in enumerate(selections):
            try:
                if allowed_refs is not None and query.ref not in allowed_refs:
                    raise OperationRejected([dict(path=["records", 0, "ref"], code="out_of_scope",
                        message="Record is outside this agent's read scope.", record_id=query.ref,
                        remedy="Use a ref in read_scope; no other records can be read.")])
                room = max(0, allowance - json_bytes(rows) - json_bytes(accepted) - 1000)
                row = bounded_query(case, selection, allowance=room, index=index)
                data = row["data"] if query.paths or query.view == "directory" else {"": row["data"]}
                candidate = {"ref": query.ref, "data": {**accepted, **data},
                             "coverage": {**coverage, **row["coverage"]},
                             "pages": {**pages, **row.get("pages", {})},
                             "selection": query.model_dump(exclude={"ref"})}
                if json_bytes(rows + [candidate]) > allowance:
                    raise OperationRejected([dict(path=["records", 0, "paths", 0], code="response_too_large",
                        message="This field exceeds the remaining read-turn allowance.", record_id=query.ref,
                        remedy="Use a directory, narrower fields or a smaller page; accepted fields are retained.")])
                accepted.update(data)
                pages.update(row.get("pages", {}))
                coverage.update(row["coverage"])
                if row.get("requested_page"):
                    actual_page = query.page.model_copy(update=row["selection"]["page"])
                    page_metadata = {key: row[key] for key in ("requested_page", "page_policy")}
                valid_paths.extend(selection.paths)
                queries.append(selection)
                values.append(row)
            except OperationRejected as exc:
                for issue in exc.issues:
                    path = issue["path"][2:]
                    if split and path[:1] == ["paths"]:
                        path = ["paths", j, *path[2:]]
                    local.append({**issue, "path": ["arguments", "records", i, *path]})
        if accepted:
            row = {"ref": query.ref, "data": accepted if query.paths or query.view == "directory" else accepted[""],
                   "coverage": coverage,
                   "selection": query.model_copy(update={"paths": valid_paths, "page": actual_page}).model_dump(exclude={"ref"}),
                   **page_metadata}
            if query.view == "directory":
                row["view"] = "directory"
            if pages:
                row["pages"] = pages
            if local:
                row.update(status="partial_read", issues=local,
                           requested_selection=query.model_dump(exclude={"ref"}))
            elif page_metadata:
                row["requested_selection"] = query.model_dump(exclude={"ref"})
        else:
            row = {"ref": query.ref, "status": "rejected", "issues": local,
                   "selection": query.model_dump(exclude={"ref"})}
        rows.append(row)
        issues.extend(local)
    outcome = {"status": "read" if not issues else "partial_read" if values else "rejected",
               "records": rows}
    if issues:
        outcome.update(issues=issues, executed=False, committed=False)
    handles = remember_reads([], queries, values)
    return outcome, handles


def batch_read_results(case, batch, *, allowed_refs=None, index=None):
    index = record_index(case, index)
    outcomes, handles = {}, []
    used = 0
    for call in batch.calls:
        if call.issues:
            outcome = {"status": "rejected", "executed": False, "committed": False, "issues": call.issues}
            read_handles = []
        else:
            try:
                outcome, read_handles = query_results(
                    case, call.command, allowed_refs=allowed_refs, allowance=max(0, READ_BYTES - used), index=index)
            except OperationRejected as exc:
                outcome = {"status": "rejected", "executed": False, "committed": False, "issues": exc.issues}
                read_handles = []
        outcomes[call.call_id] = outcome
        used += json_bytes(outcome)
        handles.extend(read_handles)
    return {"status": "read_batch", "call_results": outcomes}, handles


def merge_handles(previous, added):
    entries = { (row["ref"], row["path"], str(row["page"])): row for row in previous }
    for row in added:
        key = (row["ref"], row["path"], str(row["page"]))
        entries.pop(key, None)
        entries[key] = row
    return list(entries.values())
