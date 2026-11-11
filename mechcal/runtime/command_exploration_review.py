"""Event-driven exploration with a dedicated agent and private case memory."""

from pydantic import Field

from mechcal.mechanism_pool import MECHANISM_POOL, mechanism_definitions
from mechcal.runtime import command_exploration_memory as facts, command_followup as followup
from mechcal.runtime import command_experiment_options as options, command_query_state as query_state
from mechcal.runtime import command_investigation as investigation
from mechcal.runtime.command_context import catalog_table
from mechcal.runtime.command_memory import digest, result_report
from mechcal.runtime.command_operations import check_owner, protocol, put, record_index
from mechcal.schemas.commands import CommandModel, ExperimentSuggestion, ReportExploration


class ExplorationMemory(CommandModel):
    case_id: str
    version: int = 1
    note: str = ""
    proposals: dict[str, ExperimentSuggestion] = Field(default_factory=dict)
    seen_versions: dict[str, str] = Field(default_factory=dict)
    attempts: list[dict] = Field(default_factory=list)
    query_receipts: dict[str, dict] = Field(default_factory=dict)
    numeric_checks: dict[str, dict] = Field(default_factory=dict)
    read_handles: list[dict] = Field(default_factory=list)
    session: dict | None = None


def enabled(case):
    return protocol(case).get("reviewer_exploration") is not None


def state(case):
    check_owner(case)
    raw = protocol(case).get("reviewer_exploration")
    return ExplorationMemory.model_validate(raw) if raw is not None else None


def save_state(case, memory, **updates):
    return put(case, reviewer_exploration=memory.model_dump(mode="json"), **updates)


def enable(case):
    return case if state(case) is not None else save_state(case, ExplorationMemory(case_id=case.case_id))


def due(case):
    """Operational boundaries trigger consideration, never scientific conclusions."""
    memory = state(case)
    if memory is None or memory.session is not None:
        return None
    projected = facts.project(case)
    current = facts.versions(projected)
    if current == memory.seen_versions:
        return None
    p = protocol(case)
    results = p.get("results", {})
    if not results:
        return None
    choices = (p.get("investigation_policy") or {}).get("choices", [])
    if choices and choices[-1]["choice"]["decision"] == "conclude":
        return "before_conclusion"
    if not memory.attempts and any(r.get("status") in {"success", "partial"} for r in results.values()):
        return "first_observations"
    for row in (p.get("evidence_followup") or {}).get("questions", {}).values():
        question = row["value"]
        boundary = question["status"] in {"resolved", "deferred"} or question.get("assessment") in {
            "distinguished", "inconclusive"}
        if boundary and any(t.get("result_id") in current and
                current[t["result_id"]] != memory.seen_versions.get(t["result_id"]) for t in row.get("trials", [])):
            return "question_boundary"
    return None


def start(case, reason):
    memory = state(case)
    if memory is None or memory.session is not None:
        raise RuntimeError("Exploration episode cannot be started in this state")
    current = facts.versions(facts.project(case))
    memory.session = {"case_id": case.case_id, "mode": "exploration", "reason": reason,
        "source_basis": digest(current), "source_versions": current, "queries": 0,
        "read_handles": memory.read_handles}
    return save_state(case, memory, phase="exploration_review")


def _payload(case, projected, memory, catalog):
    session = memory.session
    current = session["source_versions"]
    changed = [ref for ref in current if current[ref] != memory.seen_versions.get(ref)]
    raw = protocol(projected)["results"]
    latest = [rid for rid in raw if rid in changed][-3:]
    own = list(memory.proposals.items())
    receipts = facts.trial_receipts(case, memory.proposals)
    return {"stage": "independent_exploration", "role": "exploration_reviewer",
        "smiles": projected.input.smiles, "mechanism_pool": list(MECHANISM_POOL),
        "mechanism_definitions": mechanism_definitions(), "tool_catalog": catalog_table(catalog),
        "results": [result_report(rid, raw[rid]) for rid in latest],
        "new_source_ids": changed, "private_note": memory.note,
        "own_proposals": [{"id": qid, **proposal.model_dump(mode="json"), "executions": receipts[qid]}
                          for qid, proposal in own[-8:]],
        "older_proposals": {"count": max(0, len(own) - 8), "read_ref": facts.MEMORY_REF},
        "fact_view_policy": "Only calculator facts and exact execution parameters are shared. "
            "Agent goals, interpretations, rankings and other roles' memories are excluded. "
            "The latest three reports are excerpts; all fact records remain readable. "
            "Private notes/proposals are context, not scientific evidence. No execution receipt means "
            "not tested through this proposal, not refuted. Failed computations are limitations only."}


def run_turn(o, case, store, *, catalog, describe_tools):
    memory = state(case)
    session = memory.session if memory else None
    if session is None or session.get("case_id") != case.case_id or session.get("mode") != "exploration":
        raise RuntimeError("Exploration retrieval session identity mismatch")
    projected = o.deps.exploration_agent.fact_view(case, memory.model_dump(mode="json"))
    index = record_index(projected)
    if digest(facts.versions(projected)) != session["source_basis"]:
        raise RuntimeError("Calculation sources changed during the exploration episode")
    payload = _payload(case, projected, memory, catalog)
    scope = index.available
    source_ids = set(scope) - {facts.MEMORY_REF}

    def accept(command):
        experiments = options.validate_rows(projected, command.experiments, catalog, source_ids)
        return {"status": "available", "text": command.text, "experiments": experiments}

    store.activity("Exploration Agent (isolated memory)", case.current_round_id)
    feedback, pending = o.deps.exploration_agent.review(
        projected, payload, store, scope=scope, state=session, index=index, compact_context=True,
        completion=("report_exploration", ReportExploration, accept), describe_tools=describe_tools)
    if pending is not None:
        retained = query_state.remember_session(projected, pending, role="exploration_reviewer", index=index)
        private = protocol(retained)
        memory.query_receipts = private.get("query_receipts", {})
        memory.numeric_checks = private.get("numeric_checks", {})
        memory.session = pending
        return save_state(case, memory)

    accepted = []
    if feedback["status"] == "available":
        current = followup.state(case)
        if current is None:
            raise RuntimeError("Exploration requires the existing question store")
        previous_ids = set(current.questions)
        current = options.store_options(case, current, feedback["experiments"], origin="exploration_reviewer")
        case = followup.save_state(case, current)
        case = investigation.offer_proposals(case, [qid for qid in current.questions if qid not in previous_ids])
        for row in feedback["experiments"]:
            proposal = ExperimentSuggestion.model_validate(row)
            qid = options.option_id(case, proposal, "exploration_reviewer")
            memory.proposals.setdefault(qid, proposal)
            accepted.append(qid)
        memory.note = feedback["text"]
    memory.read_handles = session.get("read_handles", [])
    memory.seen_versions = session["source_versions"]
    memory.attempts.append({"source_basis": session["source_basis"], "reason": session["reason"],
                            "status": feedback["status"], "proposal_ids": accepted})
    memory.session = None
    store.append_jsonl("history/exploration_reviews.jsonl", {"case_id": case.case_id,
        "reason": session["reason"], "source_basis": session["source_basis"],
        "read_scope_ids": sorted(scope), "proposal_ids": accepted, **feedback})
    # The private note never goes to Planner or Scientific Reviewer. Public proposals use
    # the normal question store. Advisory failure is recorded, not a run failure.
    return save_state(case, memory, phase="choose")
