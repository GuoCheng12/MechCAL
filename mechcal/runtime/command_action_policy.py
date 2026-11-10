"""One effective read/delivery policy for advertising and executing operations."""

from copy import deepcopy
import json

from pydantic import BaseModel, ConfigDict, Field

from mechcal.runtime.command_operations import protocol, record_identity, record_index, reject
from mechcal.runtime.command_reads import READ_KINDS

QUERY_TURNS = 2
READ_CALLS = 4
READ_SELECTIONS = 6
COMPUTE_CALLS = 6
BATCH_READS = READ_KINDS | {"read_context"}
ASSESSMENT_CONTROLS = {"revise_judgment", "amend_judgment", "finalize", "amend_final",
                       "submit_final", "review_final", "delegate_computation", "manage_experiments"}


def turn_contract(available=None):
    contract = {"read_calls": READ_CALLS, "read_selections": READ_SELECTIONS,
            "batch_read_operations": sorted(BATCH_READS), "compute_calls": COMPUTE_CALLS,
            "hypothesis_followup_controls": sorted(ASSESSMENT_CONTROLS),
            "rule": "One control, up to four read-only calls, or up to six independent computations. "
                    "One update_hypotheses may precede computations or ONE listed control. "
                    "Never mix reads with mutations; multiple delegations require separate turns."}
    if available is None:
        return contract
    names = set(available)
    contract["available_operations"] = sorted(names)
    contract["batch_read_operations"] = sorted(BATCH_READS & names)
    contract["hypothesis_followup_controls"] = (sorted(ASSESSMENT_CONTROLS & names)
                                                if "update_hypotheses" in names else [])
    if not contract["batch_read_operations"]:
        contract["read_calls"] = contract["read_selections"] = 0
    if not any(name.startswith("compute_") for name in names):
        contract["compute_calls"] = 0
    contract["rule"] = ("Only available_operations may be called in this response. "
        "Use one listed control, or the listed read/computation groups within their nonzero limits. "
        "update_hypotheses may precede available computations or ONE listed hypothesis_followup_control. "
        "Never mix reads with mutations. Zero means unavailable, not permission to infer an operation.")
    return contract


class ActionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scientific_reads_remaining: int | None = None
    workspace_refs: list[str] = Field(default_factory=list)
    has_final_draft: bool = False

    def restricts(self, refs):
        return self.scientific_reads_remaining == 0 and not set(refs).issubset(self.workspace_refs)

    def guard_read(self, refs):
        if self.restricts(refs):
            reject(["records"], "assessment_read_limit",
                   "Scientific queries for this result group are exhausted; working records remain readable.",
                   "Use read_evidence for operation_policy.workspace_refs to inspect or correct the current work. "
                   "For further science, assess available results and choose a computation or delivery; "
                   "rewriting an assessment does not refill queries.",
                   details={"workspace_refs": self.workspace_refs})

    def filter_tools(self, tools, models):
        disabled = set()
        if not self.has_final_draft:
            disabled.update(("review_final", "amend_final", "submit_final"))
        if self.scientific_reads_remaining == 0 and not self.workspace_refs:
            disabled.add("read_evidence")
        if self.scientific_reads_remaining == 0:
            disabled.update({"reduce_evidence", "compare_evidence"})
        visible = deepcopy([tool for tool in tools if tool["name"] not in disabled])
        for tool in visible:
            if tool["name"] == "read_evidence" and self.scientific_reads_remaining == 0:
                tool["description"] += " Scientific query allowance exhausted; read only the listed working-record refs."
                tool["parameters"]["$defs"]["ReadRecord"]["properties"]["ref"]["enum"] = self.workspace_refs
        return visible, {name: model for name, model in models.items() if name not in disabled}


def read_allowance(case):
    p = protocol(case)
    exploration = p.get("exploration")
    if exploration is None:
        return None
    used = exploration.get("query_request_ids", []) if exploration.get("observed_results", []) == list(p.get("results", {})) else []
    return max(0, QUERY_TURNS - len(used))


def action_policy(case, *, index=None, scientific_read_limit_enabled=True):
    p = protocol(case)
    index = record_index(case, index)
    refs = {ref for ref in index.available if record_identity(case, ref, index=index)["workspace_read"]}
    return ActionPolicy(scientific_reads_remaining=(read_allowance(case) if scientific_read_limit_enabled else None),
                        workspace_refs=sorted(refs & index.available.keys()),
                        has_final_draft=bool(p.get("final_draft")))


def science_read_attempt(case, exchange):
    """Only source queries spend exploration allowance, including rejected refs."""
    if not exchange:
        return False
    workspace = set(action_policy(case).workspace_refs)
    for call in exchange.get("calls", []):
        if call["name"] not in READ_KINDS:
            continue
        try:
            arguments = json.loads(call["arguments"])
            rows = (arguments["records"] if call["name"] == "read_evidence" else
                    [arguments["source"]] if call["name"] == "reduce_evidence" else
                    [arguments["left"], arguments["right"]])
            refs = [row["ref"] for row in rows]
        except (ValueError, TypeError, KeyError):
            return True
        if not refs or any(not isinstance(ref, str) for ref in refs) or not set(refs).issubset(workspace):
            return True
    return False
