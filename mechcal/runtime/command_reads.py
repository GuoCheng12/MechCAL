"""Bounded read receipts shared by Planner and scoped Worker retrieval."""

from mechcal.runtime.command_memory import READ_BYTES, json_bytes, remember_reads
from mechcal.runtime.command_operations import OperationRejected, read_records
from mechcal.schemas.commands import ReadEvidence


def read_selection_results(case, command, *, allowed_refs=None, allowance=READ_BYTES):
    rows, issues, queries, values = [], [], [], []
    for i, query in enumerate(command.records):
        # Read-only requests may retain valid fields. Atomic scientific commits
        # still use strict read_records and reject any invalid citation.
        split = bool(query.paths) and not (query.view == "data" and query.page and len(query.paths) != 1)
        selections = [query.model_copy(update={"paths": [path]}) for path in query.paths] if split else [query]
        accepted, local, pages, valid_paths = {}, [], {}, []
        for j, selection in enumerate(selections):
            try:
                if allowed_refs is not None and query.ref not in allowed_refs:
                    raise OperationRejected([dict(path=["records", 0, "ref"], code="out_of_scope",
                        message="Record is outside this agent's read scope.", record_id=query.ref,
                        remedy="Use a ref in read_scope; no other records can be read.")])
                row = read_records(case, ReadEvidence(kind="read_evidence", records=[selection]))[0]
                data = row["data"] if query.paths or query.view == "directory" else {"": row["data"]}
                candidate = {"ref": query.ref, "data": {**accepted, **data}, "selection": query.model_dump(exclude={"ref"})}
                if json_bytes(rows + [candidate]) > allowance:
                    raise OperationRejected([dict(path=["records", 0, "paths", 0], code="response_too_large",
                        message="This field exceeds the remaining read-turn allowance.", record_id=query.ref,
                        remedy="Use a directory, narrower fields or a smaller page; accepted fields are retained.")])
                accepted.update(data)
                pages.update(row.get("pages", {}))
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
                   "selection": query.model_copy(update={"paths": valid_paths}).model_dump(exclude={"ref"})}
            if query.view == "directory":
                row["view"] = "directory"
            if pages:
                row["pages"] = pages
            if local:
                row.update(status="partial_read", issues=local)
        else:
            row = {"ref": query.ref, "status": "rejected", "issues": local}
        rows.append(row)
        issues.extend(local)
    outcome = {"status": "read" if not issues else "partial_read" if values else "rejected",
               "records": rows}
    if issues:
        outcome.update(issues=issues, executed=False, committed=False)
    handles = remember_reads([], queries, values)
    return outcome, handles


def batch_read_results(case, batch, *, allowed_refs=None):
    outcomes, handles = {}, []
    used = 0
    for call in batch.calls:
        if call.issues:
            outcome = {"status": "rejected", "executed": False, "committed": False, "issues": call.issues}
            read_handles = []
        else:
            outcome, read_handles = read_selection_results(
                case, call.command, allowed_refs=allowed_refs, allowance=max(0, READ_BYTES - used))
        outcomes[call.call_id] = outcome
        used += json_bytes(outcome)
        handles.extend(read_handles)
    return {"status": "read_batch", "call_results": outcomes}, handles


def merge_handles(previous, added):
    entries = { (row["ref"], row["path"], str(row["page"])): row for row in previous }
    for row in added:
        entries[(row["ref"], row["path"], str(row["page"]))] = row
    return list(entries.values())
