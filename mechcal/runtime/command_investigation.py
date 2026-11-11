"""Separate experiment choices from beliefs, without choosing experiments in code."""

from copy import deepcopy

from mechcal.runtime.command_memory import digest
from mechcal.runtime.command_operations import check_owner, protocol, put, reject, record_index

STATE_REF = "investigation:state"


def enabled(case):
    return protocol(case).get("investigation_policy") is not None


def enable(case, *, offer_experiments=False):
    check_owner(case)
    if enabled(case):
        if offer_experiments and not protocol(case)["investigation_policy"].get("offer_experiments"):
            return put(case, investigation_policy={**protocol(case)["investigation_policy"], "offer_experiments": True})
        return case
    return put(case, investigation_policy={"case_id": case.case_id, "version": 1, "choices": [],
                                          **({"offer_experiments": True} if offer_experiments else {})})


def basis(case, *, index=None):
    p = protocol(case)
    followup = p.get("evidence_followup") or {}
    # Reads, wording edits and review prose do not invalidate an experiment choice.
    # New observations, changed beliefs/questions or impact responses do.
    build = lambda: digest({"case_id": case.case_id, "results": p.get("results", {}),
        "artifacts": case.artifact_manifest.model_dump(mode="json"),
        "assessments": (p.get("exploration") or {}).get("entries", {}),
        "questions": {qid: row["value"] for qid, row in followup.get("questions", {}).items()},
        "impacts": followup.get("impacts", {})})
    return record_index(case, index).memo("investigation_basis", build) if index is not None else build()


def current(case):
    check_owner(case)
    choices = (protocol(case).get("investigation_policy") or {}).get("choices", [])
    return choices[-1] if choices else None


def pending_proposals(case):
    check_owner(case)
    return (protocol(case).get("investigation_policy") or {}).get("pending_proposals")


def offer_proposals(case, question_ids):
    """A public planning handoff, not a new observation or a required experiment."""
    if not question_ids:
        return case
    if not enabled(case):
        raise RuntimeError("Proposal decisions require investigation selection")
    questions = (protocol(case).get("evidence_followup") or {}).get("questions", {})
    pending = pending_proposals(case) or {}
    ids = list(dict.fromkeys([*pending.get("question_ids", []), *question_ids]))
    if any(qid not in questions for qid in ids):
        raise RuntimeError("Proposal handoff refers to a missing case-local question")
    versions = {qid: digest(questions[qid]["value"]) for qid in ids}
    handoff = {"id": "proposal_decision:" + digest({"case_id": case.case_id, "questions": versions})[:24],
               "question_ids": ids, "question_versions": versions}
    return put(case, investigation_policy={**protocol(case)["investigation_policy"], "pending_proposals": handoff})


def record_choice(case, command):
    choice = getattr(command, "investigation", None)
    if choice is None:
        return case
    if not enabled(case):
        reject(["investigation"], "operation_disabled", "Investigation selection is not enabled.",
               "Use the advertised update_hypotheses fields.")
    questions = (protocol(case).get("evidence_followup") or {}).get("questions", {})
    for field, ids in (("question_id", [choice.question_id] if choice.question_id else []),
                       ("considered_question_ids", choice.considered_question_ids)):
        for qid in ids:
            if qid not in questions:
                reject(["investigation", field], "unknown_question", "Question is not in this case.",
                       "Use a saved or same-update question ID; use null for open exploration.", record_id=qid)
    p = protocol(case)
    value = {"basis": basis(case), "choice": choice.model_dump(mode="json")}
    prior = current(case)
    pending = pending_proposals(case)
    if not pending and prior and all(prior[k] == v for k, v in value.items()):
        return case
    state = deepcopy(p["investigation_policy"])
    row = {**value, "id": f"choice:{len(state['choices']) + 1}", "round": p.get("rounds", 0),
           "result_ids": list(p.get("results", {}))}
    if pending:
        row["proposal_handoff"] = state.pop("pending_proposals")
    state["choices"].append(row)
    return put(case, investigation_policy=state)


def guard(case, command):
    if not enabled(case):
        return
    computing = command.kind in {"compute", "compute_batch", "delegate_computation"}
    computing |= command.kind == "manage_experiments" and command.action == "retry"
    concluding = command.kind in {"finalize", "submit_final", "review_final", "amend_final"}
    if not (computing or concluding):
        return
    choice = current(case)
    expected = "investigate" if computing else "conclude"
    if pending_proposals(case) or not choice or choice["basis"] != basis(case) or choice["choice"]["decision"] != expected:
        reject(["investigation"], "investigation_choice_required",
               f"No current {expected} choice is recorded. Nothing was executed or delivered.",
               "Use update_hypotheses.investigation; it can accompany this operation in the same response. "
               "Compare useful experiments separately from confidence, or state why to conclude. "
               "No particular experiment, score change or conclusion is required.")


def view(case, options, *, index=None):
    """Bounded navigation; existing question storage is the only experiment table."""
    row = current(case)
    followup = protocol(case).get("evidence_followup") or {}
    pending = [{"id": iid, "label": item["label"], "disposition": item["disposition"]}
               for iid, item in followup.get("impacts", {}).items()
               if item["disposition"] in {"pending", "deferred"}]
    return {"state_ref": STATE_REF, "current_choice": row["choice"] if row else None,
        "choice_id": row["id"] if row else None, "choice_current": bool(row and row["basis"] == basis(case, index=index)),
        "experiment_options_field": "comparison_options", "experiment_option_count": len(options),
        "pending_impacts": pending[:12], "pending_impact_count": len(pending),
        "policy": "Beliefs are in exploration_state; experiment options are saved questions, not confidence ranks. "
                  "Choose investigation or conclusion with update_hypotheses.investigation; combine with the next operation. "
                  "Update every affected judgment, or explicitly retain/defer/dismiss its impact. Neither failure nor "
                  "an untested mechanism is negative evidence. Choices are planning records, never scientific evidence. "
                  "The code checks identity/freshness, not which experiment is scientifically best."}


def review_context(case, *, index=None):
    """Reused by the existing Reviewer, not another review round or endorsement."""
    row = current(case)
    choices = protocol(case)["investigation_policy"]["choices"]
    return {"current_choice": {k: row[k] for k in ("id", "round", "choice")} if row else None,
        "choice_current": bool(row and row["basis"] == basis(case, index=index)), "recent_choices": [
        {k: item[k] for k in ("id", "round", "choice")} for item in choices[-3:]],
        "read_ref": STATE_REF,
        "policy": "Check the decision to stop against remaining feasible comparisons, including candidates outside Top-3. "
                  "Compare expected information with redundant confirmation and preparation cost. "
                  "Question IDs considered are a Planner report, not proof of adequate coverage. "
                  "Suggest a concrete useful route when one is overlooked; disagreement is advisory. "
                  "No required label, exhaustive search, confidence change or wet-lab proof. "
                  "Keep this within the existing computational review; unchanged advice is reusable."}
