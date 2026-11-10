"""Read-only scientific facts and private retrieval state for Reviewer exploration."""

from mechcal.runtime.command_memory import digest
from mechcal.runtime.command_operations import check_owner, protocol, record_index
from mechcal.schemas import CaseInput, CaseRun, EvidenceLedger


MEMORY_REF = "private:reviewer_exploration"
RESULT_FIELDS = ("schema_version", "record_type", "tool_result_id", "round_id", "agent_name",
                 "capability_id", "selected_route", "status", "structured_results", "raw_results", "failure")
UNIT_FIELDS = ("evidence_id", "round_id", "source_report_id", "agent_name", "capability_id",
               "basis", "metrics", "observable", "artifact_refs", "status", "observable_tags", "limits")
ACTION_FIELDS = ("dispatch_id", "round_id", "agent_name", "capability_id", "route", "input_refs",
                 "tool_args", "result_id", "status")


def project(case, memory=None):
    """Build a fresh CaseRun, not a blacklist over the Planner's shared memory.

    Calculator payloads/artifacts stay available by their original IDs and field
    paths. Agent-written explanations, goals, evidence interpretations, ranking,
    global query caches and user metadata are never copied. This object is only
    passed to read operations; it must never replace the authoritative case.
    """
    check_owner(case)
    # Validate source ownership before projecting away fields or duplicate IDs.
    record_index(case)
    source = protocol(case)
    memory = memory or {}
    results = {}
    for rid, row in source.get("results", {}).items():
        results[rid] = {key: row[key] for key in RESULT_FIELDS if key in row}
        results[rid]["evidence_units"] = [
            {key: unit[key] for key in UNIT_FIELDS if key in unit}
            for unit in row.get("evidence_units", [])]
        results[rid]["artifact_updates"] = row.get("artifact_updates", [])
    actions = {ref: {key: row[key] for key in ACTION_FIELDS if key in row}
               for ref, row in source.get("actions", {}).items()}
    private = {"note": memory.get("note", ""), "proposals": memory.get("proposals", {})}
    return CaseRun(case_id=case.case_id,
        input=CaseInput(case_id=case.case_id, smiles=case.input.smiles, user_query="Read-only scientific exploration."),
        current_round_id=case.current_round_id,
        evidence_ledger=EvidenceLedger(case_id=case.case_id), artifact_manifest=case.artifact_manifest,
        runtime={"command_protocol": {"case_id": case.case_id, "results": results, "actions": actions,
            "arguments": {MEMORY_REF: private},
            "query_receipts": memory.get("query_receipts", {}),
            "numeric_checks": memory.get("numeric_checks", {})}})


def versions(projected):
    index = record_index(projected)
    return {ref: digest(row) for ref, row in index.available.items() if ref != MEMORY_REF}


def trial_receipts(case, proposal_ids):
    """Execution links only, never Planner edits, dispositions or explanations."""
    questions = (protocol(case).get("evidence_followup") or {}).get("questions", {})
    results = protocol(case).get("results", {})
    receipts = {}
    for qid in proposal_ids:
        trials = questions.get(qid, {}).get("trials", [])
        rows = []
        for trial in trials:
            rid = trial.get("result_id")
            if rid in results and rid not in {r["result_id"] for r in rows}:
                rows.append({"result_id": rid, "capability_id": results[rid].get("capability_id"),
                             "status": results[rid].get("status")})
        receipts[qid] = rows
    return receipts
