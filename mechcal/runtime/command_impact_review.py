"""One bounded association review per completed computation group."""

from mechcal.agents.command_worker import scoped_commentary_turn
from mechcal.mechanism_pool import MECHANISM_POOL, mechanism_definitions
from mechcal.runtime import command_followup as followup
from mechcal.runtime import command_query_state as query_state
from mechcal.runtime import command_numeric as numeric
from mechcal.runtime.command_memory import digest, result_report
from mechcal.runtime.command_operations import protocol, put, record_index, reject
from mechcal.runtime.command_review import candidate_overview, review_execution_context, reviewer_scope, review_worker_interpretations
from mechcal.runtime.command_comparison import question_evidence_view, question_source_refs
from mechcal.schemas.commands import ReportEvidenceImpacts


def review_payload(case, result_ids):
    p = protocol(case)
    return {"stage": "evidence_impact_review", "role": "impact_reviewer", "smiles": case.input.smiles,
            "mechanism_pool": list(MECHANISM_POOL), "candidates": candidate_overview(case),
            "mechanism_definitions": mechanism_definitions(),
            "results": [p.get("result_reports", {}).get(rid) or result_report(rid, p["results"][rid]) for rid in result_ids],
            "worker_explanations": review_worker_interpretations(case, [{"result_id": rid} for rid in result_ids]),
            "open_questions": (followup.view(case) or {}).get("questions", []),
            "association_source_ids": sorted(association_sources(case, result_ids)),
            "association_contract": "read_scope.impact_source=true marks IDs allowed in suggestions.refs. "
                                    "Artifacts and prior sources can be read for context, not cited as a new result."}


def association_sources(case, result_ids):
    results = protocol(case)["results"]
    return set(result_ids) | {unit["evidence_id"] for rid in result_ids
                            for unit in results[rid].get("evidence_units", []) if unit.get("evidence_id")}


def review_material(case, result_ids, *, index=None):
    index = record_index(case, index)
    payload = review_payload(case, result_ids)
    payload["execution_context"] = review_execution_context(case, set(result_ids), index.available)
    comparison = question_evidence_view(case, result_ids=result_ids, index=index)
    if comparison["questions"]:
        payload["question_evidence"] = comparison
    return payload


def validate_suggestions(case, result_ids, command):
    allowed = association_sources(case, result_ids)
    labels = set()
    for i, suggestion in enumerate(command.suggestions):
        if suggestion.label in labels:
            reject(["suggestions", i, "label"], "duplicate_edit", "One association per candidate per result group.",
                   "Combine the relevant new references for this candidate.")
        labels.add(suggestion.label)
        for j, ref in enumerate(suggestion.refs):
            if ref not in allowed:
                reject(["suggestions", i, "refs", j], "impact_source", "Association must cite a result or unit in this new group.",
                       "Use its exact source ID. A tool call, explanation or old observation is not a new result.", record_id=ref)
    return {"status": "available", "text": command.text,
            "suggestions": [s.model_dump(mode="json") for s in command.suggestions]}


def run_turn(o, case, store):
    p = protocol(case)
    ids = followup.unreviewed_results(case)
    if not ids:
        if p.get("impact_session") is not None:
            raise RuntimeError("Impact session has no pending result group")
        return put(case, phase="choose")
    session = p.get("impact_session") or {"result_ids": ids, "queries": 0}
    if session["result_ids"] != ids:
        raise RuntimeError("Impact review session belongs to a different result group")
    index = record_index(case)
    refs = {"input:smiles", *ids}
    for candidate in candidate_overview(case):
        refs.update(candidate.get("refs", []))
        refs.update(candidate.get("limitation_refs", []))
    for question in followup.state(case).questions.values():
        if question.value.status != "resolved":
            refs.update(trial["result_id"] for trial in question.trials[-3:] if trial.get("result_id"))
    payload = review_material(case, ids, index=index)
    if "question_evidence" in payload:
        refs.update(question_source_refs(payload["question_evidence"]))
    scope = reviewer_scope(case, refs, index.available)
    for rid in ids:
        for ref in ("report:" + rid, "explanation:" + rid):
            if ref in index.available:
                scope[ref] = index.available[ref]
    context_key = digest(payload)
    if session.get("context_key", context_key) != context_key:
        raise RuntimeError("Impact review sources changed within a read session")
    session["context_key"] = context_key
    store.activity("Evidence impact review", case.current_round_id)
    feedback, pending = scoped_commentary_turn(o.deps.photophysics_arbiter.llm_client, case, payload, store,
        scope=scope, state=session, role="impact_reviewer", prompt_name="command_impact_reviewer.md", index=index,
        completion=("report_evidence_impacts", ReportEvidenceImpacts,
                    lambda command: validate_suggestions(case, ids, command)),
        compact_context=o.config.command_compact_context)
    if pending is not None:
        if o.config.command_compact_context or numeric.outcome_ids(pending.get("outcome", {})):
            case = query_state.remember_session(case, pending, role="impact_reviewer", index=index)
        return put(case, impact_session=pending)
    case = followup.accept_review(case, ids, feedback)
    store.append_jsonl("history/command_commentary.jsonl", {
        "role": "impact_reviewer", "result_ids": ids, "revision": p["revision"], **feedback})
    return put(case, phase="choose", impact_session=None)
