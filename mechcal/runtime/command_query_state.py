"""Case-local query receipts. Navigation metadata is never scientific evidence."""

from typing import Literal
from copy import deepcopy
import json

from pydantic import Field

from mechcal.runtime.command_memory import digest, json_bytes
from mechcal.runtime.command_operations import protocol, put, record_index
from mechcal.schemas.commands import CommandModel, ReadEvidence, ReadRecord
from mechcal.runtime.command_views import edit_draft_id


class QueryReceipt(CommandModel):
    case_id: str
    source_version: str
    selection: ReadRecord
    accepted_selection: ReadRecord | None = None
    status: Literal["read", "partial_read", "rejected"]
    role: str
    draft_id: str | None = None
    issues: list[dict] = Field(default_factory=list)


def remember(case, outcome, *, role, index=None):
    """Persist accepted ranges and local navigation failures, not model prose."""
    index = record_index(case, index)
    if outcome.get("status") == "read_batch":
        for child in outcome["call_results"].values():
            case = remember(case, child, role=role)
        return case
    if outcome.get("status") == "numeric_analysis":
        from mechcal.runtime.command_numeric import remember as remember_numeric
        return remember_numeric(case, outcome["numeric_check"])
    p = protocol(case)
    saved = dict(p.get("query_receipts", {}))
    # Navigation attempts can fail or be overwritten by another agent. Exact
    # Planner edit reads have their own bounded-by-draft set of metadata handles.
    edits = dict(p.get("edit_reads", {k: v for k, v in saved.items()
        if v.get("role") == "planner" and v.get("draft_id")
        and (v.get("accepted_selection") or {}).get("view") == "data"}))
    for row in outcome.get("records", []):
        ref = row["ref"]
        if ref not in index.available or "selection" not in row:
            continue  # Unknown/out-of-scope requests must not acquire read access.
        issues = row.get("issues", [])
        if any(i["code"] == "out_of_scope" for i in issues):
            continue
        selection = ReadRecord(ref=ref, **row.get("requested_selection", row["selection"]))
        receipt = QueryReceipt(case_id=case.case_id, source_version=digest(index.available[ref]),
            selection=selection, accepted_selection=ReadRecord(ref=ref, **row["selection"]) if "data" in row else None,
            status=row.get("status", "read"), role=role, issues=issues,
            draft_id=edit_draft_id(case, ref))
        key = digest(selection.model_dump(mode="json"))
        saved.pop(key, None)
        saved[key] = receipt.model_dump(mode="json")
        if (role == "planner" and receipt.draft_id and receipt.accepted_selection is not None
                and receipt.accepted_selection.view == "data"):
            edits = {k: v for k, v in edits.items() if v.get("draft_id") == receipt.draft_id}
            edits[digest(receipt.accepted_selection.model_dump(mode="json"))] = receipt.model_dump(mode="json")
    changes = {}
    if saved != p.get("query_receipts", {}):
        changes["query_receipts"] = saved
    if edits != p.get("edit_reads", {}):
        changes["edit_reads"] = edits
    return put(case, **changes) if changes else case


def _selection_keys(outcome):
    if outcome.get("status") == "read_batch":
        return {key for child in outcome["call_results"].values() for key in _selection_keys(child)}
    return {digest(ReadRecord(ref=row["ref"], **row.get("requested_selection", row["selection"])).model_dump(mode="json"))
            for row in outcome.get("records", []) if "selection" in row}


def view(case, *, allowed_refs=None, latest_outcome=None, index=None):
    """Expose bounded, version-valid corrections; successful values use working memory."""
    index = record_index(case, index)
    failures, omitted, successful, versions = [], 0, 0, {}
    latest = _selection_keys(latest_outcome or {})
    for raw in reversed(list(protocol(case).get("query_receipts", {}).values())):
        receipt = QueryReceipt.model_validate(raw)
        if receipt.case_id != case.case_id:
            raise RuntimeError("Query receipt belongs to another case")
        ref = receipt.selection.ref
        if ref not in index.available or (allowed_refs is not None and ref not in allowed_refs):
            continue
        if ref not in versions:
            versions[ref] = digest(index.available[ref])
        if versions[ref] != receipt.source_version:
            continue
        if not receipt.issues:
            successful += 1
            continue
        if digest(receipt.selection.model_dump(mode="json")) in latest:
            continue  # Already delivered in this request's native tool response.
        item = {"selection": receipt.selection.model_dump(mode="json"), "status": receipt.status,
                "accepted_selection": receipt.accepted_selection.model_dump(mode="json") if receipt.accepted_selection else None,
                "issues": receipt.issues}
        if len(failures) < 6 and json_bytes(failures + [item]) <= 6000:
            failures.append(item)
        else:
            omitted += 1
    return {"failed_reads": failures, "omitted_failures": omitted, "successful_selections": successful,
            "policy": "Query receipts are navigation only, not evidence. For unchanged sources use the returned "
                      "child paths or a different page instead of repeating an oversized selection. "
                      "A failed read does not mean the data are absent; exact reads remain available."}


def remember_session(case, session, *, role, registry=None, index=None):
    """Save query state at the same boundary as its native tool response."""
    outcome = session_outcome(session)
    case = remember(case, outcome, role=role, index=index)
    if outcome.get("status") == "described" and registry is not None:
        allowed = case.runtime.get("ablation", {}).get("enabled_capability_owners", ["macro", "microscopic"])
        p = protocol(case)
        loaded = dict(p.get("loaded_tools", {}))
        active = list(p.get("visible_tools", []))
        for detail in outcome["tools"]:
            cid = detail["capability_id"]
            spec = registry.get(cid)
            if spec is None or spec.owner_agent not in allowed or registry.definition_version(cid) != detail["definition_version"]:
                continue
            loaded[cid] = detail["definition_version"]
            active = [item for item in active if item != cid] + [cid]
        # This advertises schemas, not executions. prepare_computation still
        # checks current permissions, parameters, artifacts and run limits.
        shared = dict(p.get("shared_tool_definitions", {}))
        shared.update({d["capability_id"]: d["definition_version"] for d in outcome["tools"]
                       if loaded.get(d["capability_id"]) == d["definition_version"]})
        case = put(case, loaded_tools=loaded, visible_tools=active[-12:], shared_tool_definitions=shared)
    return case


def session_outcome(session):
    """Recover old receipt selections from the pending native request, not audit logs."""
    outcome = deepcopy(session.get("outcome") or {})
    for call in (session.get("exchange") or {}).get("calls", []):
        if call.get("name") != "read_evidence":
            continue
        result = outcome.get("call_results", {}).get(call["id"], {}) if outcome.get("status") == "read_batch" else outcome
        rows = result.get("records", [])
        if not rows:
            continue
        command = ReadEvidence.model_validate({"kind": "read_evidence", **json.loads(call["arguments"])})
        if len(rows) != len(command.records) or any(r["ref"] != q.ref for r, q in zip(rows, command.records)):
            raise RuntimeError("Saved read result does not match its pending query")
        for row, query in zip(rows, command.records):
            row.setdefault("selection", query.model_dump(exclude={"ref"}))
            if row.get("issues"):
                row.setdefault("requested_selection", query.model_dump(exclude={"ref"}))
    return outcome


def refresh_tools(case, registry):
    """Only current, permitted definitions can become native Planner functions."""
    p = protocol(case)
    allowed = case.runtime.get("ablation", {}).get("enabled_capability_owners", ["macro", "microscopic"])
    loaded = {cid: version for cid, version in p.get("loaded_tools", {}).items()
              if (spec := registry.get(cid)) is not None and spec.owner_agent in allowed
              and registry.definition_version(cid) == version}
    return put(case, loaded_tools=loaded,
               visible_tools=[cid for cid in p.get("visible_tools", []) if cid in loaded][-12:])


def tool_guidance(case, registry):
    """Shared schemas still need the registry's scientific usage boundaries."""
    p = protocol(case)
    guidance = []
    for cid in p.get("visible_tools", []):
        version = p.get("shared_tool_definitions", {}).get(cid)
        spec = registry.get(cid)
        if spec is None or version != registry.definition_version(cid):
            continue
        item = {"capability_id": cid,
                "limitations": spec.metadata.get("limitations", []),
                "usage_hints": spec.metadata.get("tool_arg_hints", {}),
                "allowed_argument_values": spec.allowed_argument_values,
                "execution_dependencies": spec.metadata.get("execution_dependencies", [])}
        guidance.append({k: v for k, v in item.items() if v})
    return guidance
