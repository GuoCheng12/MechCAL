"""Carry successful exact selections across roles, without copying source data."""

from mechcal.runtime.command_memory import digest, json_bytes, resolve_pointer
from mechcal.runtime.command_operations import protocol, record_index
from mechcal.runtime.command_reads import merge_handles


def selection_handles(outcome):
    """Keep the requested page together; directory/error metadata is not evidence."""
    if outcome.get("status") == "read_batch":
        handles = []
        for child in outcome["call_results"].values():
            handles = merge_handles(handles, selection_handles(child))
        return handles
    handles = []
    for row in outcome.get("records", []):
        selection = row.get("selection", {})
        if "data" not in row or selection.get("view") != "data":
            continue
        fields = row["data"] if selection.get("paths") else {"": row["data"]}
        for path, value in fields.items():
            handles.append({"ref": row["ref"], "path": path, "page": selection.get("page"),
                            "bytes": json_bytes(value), "sha256": digest(value), "requested": True})
    return merge_handles([], handles)


def capture(case, result_id, handles, *, index=None):
    index = record_index(case, index)
    # Never use audit logs as state. Only accepted selections in the current
    # session are eligible; the producing result binds even input-artifact reads.
    refs = {result_id, *(h["ref"] for h in handles)}
    return {"case_id": case.case_id, "result_id": result_id, "reads": handles,
            "sources": {ref: digest(index.available[ref]) for ref in sorted(refs)}}


def retained(case, *, allowed_refs=None, result_ids=None, index=None):
    index = record_index(case, index)
    valid, issues, versions = [], [], {}
    for result_id, bundle in protocol(case).get("read_handoffs", {}).items():
        if result_ids is not None and result_id not in result_ids:
            continue
        if bundle["case_id"] != case.case_id or bundle["result_id"] != result_id:
            raise RuntimeError("Source-read handoff belongs to another case or result")
        if allowed_refs is not None and result_id not in allowed_refs:
            continue
        for ref, version in bundle["sources"].items():
            if ref not in versions:
                versions[ref] = digest(index.available[ref]) if ref in index.available else None
        for handle in bundle["reads"]:
            ref = handle["ref"]
            if allowed_refs is not None and ref not in allowed_refs:
                continue
            if any(versions.get(r) != bundle["sources"].get(r) for r in (result_id, ref)):
                issues.append({"ref": ref, "path": handle["path"], "status": "source_changed_or_missing"})
                continue
            value = resolve_pointer(index.available[ref], handle["path"])
            if handle["page"]:
                page = handle["page"]
                value = value[page["offset"]:page["offset"] + page["limit"]]
            if digest(value) != handle["sha256"]:
                raise RuntimeError("Source-read handle conflicts with its recorded source version")
            valid.append(handle)
    return merge_handles([], valid), issues
