"""Versioned mechanical updates; no scientific scores or conclusions are inferred."""

from mechcal.runtime.boundary_decisions import require
from mechcal.schemas.question_loop import RetainedJudgment


def expand_delta(case, update):
    delta = update.delta
    state = case.runtime.get("question_loop", {})
    require(case.runtime.get("incremental_judgment"), ["delta"], "Incremental updates are not enabled.")
    require(delta.base_revision == state.get("revision") and update.revision == delta.base_revision + 1,
            ["delta", "base_revision"], "Use the current accepted revision, not an earlier snapshot.")
    require(update.retained_judgment is None, ["retained_judgment"],
            "Delta already retains all prior records. Do not also select retained_judgment.")
    old_support = {a.support_id: a for a in case.mechanism_support_arguments}
    removed = delta.withdrawn_support_ids
    require(len(removed) == len(set(removed)) and set(removed) <= old_support.keys(),
            ["delta", "withdrawn_support_ids"], "Withdraw distinct existing support IDs only.")
    require(not set(removed) & {a.support_id for a in update.mechanism_support_arguments},
            ["delta", "withdrawn_support_ids"], "Do not withdraw and submit the same support record.")
    old = {h.name: h for h in case.portfolio.hypotheses}
    supplied = {h.name for h in update.portfolio.hypotheses}
    scores = delta.hypothesis_scores
    labels = [h.name for h in scores]
    require(len(labels) == len(set(labels)) and set(labels) <= old.keys() and not set(labels) & supplied,
            ["delta", "hypothesis_scores"], "Score updates need distinct existing labels, without full-record overlap.")
    if delta.scope == "local_audit":
        review = state.get("argument_review") or {}
        allowed = set(review.get("support_ids", []))
        require(state.get("argument_review_pending") and not state.get("pending_results"),
                ["delta", "scope"], "Local audit updates require pending audit feedback, not new results.")
        require(not update.portfolio.hypotheses and not scores and not update.questions
                and not update.candidate_responses and not update.ranking_update,
                ["delta", "scope"], "Ranking or investigation changes require scope=global with an explanation.")
        require(({a.support_id for a in update.mechanism_support_arguments} | set(removed)) <= allowed,
                ["delta", "scope"], "Changes outside the reviewed arguments require scope=global.")
        require(update.portfolio.current == case.portfolio.current
                and update.portfolio.runner_up == case.portfolio.runner_up,
                ["portfolio"], "Local audit repair must preserve the current ranking.")
    saved = state.get("assimilation", {}).get("ranking_update")
    # None reuses the entire prior comparison. Existing validators reject a stale
    # cutoff comparison if the submitted scores change its required pair.
    ranking = update.ranking_update
    if ranking is None and saved is not None:
        from mechcal.schemas.evidence_comparison import RankingUpdate
        ranking = RankingUpdate.model_validate({
            **saved, "basis": "unchanged", "explanation": delta.reason,
        })
    score_records = [
        old[s.name].model_copy(update=s.model_dump(include={"confidence", "differential_priority"}), deep=True) for s in scores
    ]
    retained = RetainedJudgment(
        hypothesis_labels=list(old),
        support_ids=[sid for sid in old_support if sid not in removed],
        question_ids=list(state.get("questions", {})),
        candidate_basis_labels=[b["label"] for b in (saved or {}).get("candidate_bases", [])],
    )
    candidate = update.model_copy(deep=True)
    candidate.portfolio.hypotheses.extend(score_records)
    candidate.ranking_update = ranking
    candidate.retained_judgment = retained
    candidate.delta = None
    from mechcal.runtime.decision_economy import expand_retained
    expand_retained(case, candidate)
    # Commit only after every base/upsert contract succeeds.
    for field in ("portfolio", "questions", "mechanism_support_arguments", "ranking_update",
                  "retained_judgment", "delta"):
        setattr(update, field, getattr(candidate, field))


def add_incremental_context(case, payload):
    if not case.runtime.get("incremental_judgment"):
        return payload
    state = case.runtime.get("question_loop", {})
    contract = payload.get("output_contract", {})
    target = contract.get("$defs", {}).get("QuestionAssimilation", contract)
    properties = target.get("properties", {})
    if "retained_judgment" in properties:
        properties["retained_judgment"] = {
            "type": "null", "description": "Not used in incremental mode; delta retains base records.",
        }
    if state.get("revision") and "delta" in properties:
        properties["delta"] = {"$ref": "#/$defs/JudgmentDelta"}
        target["required"] = list(dict.fromkeys([*target.get("required", []), "delta"]))
    payload["incremental_update"] = {
        "enabled": True, "base_revision": state.get("revision", 0),
        "initial_snapshot_required": not state.get("revision"),
        "contract": (
            "After the initial snapshot, use delta with this base_revision. "
            "Omitted hypotheses, arguments, questions and candidate bases are retained. "
            "Submit only changed/new full records. Use hypothesis_scores for score-only changes. "
            "Withdraw support explicitly; never delete prior questions or mechanism-pool entries. "
            "Null ranking_update reuses the prior comparison; changed cutoff requires a new comparison."
        ),
    }
    return payload


def planner_prompt(case, name):
    from mechcal.runtime.prompts import load_prompt
    prompt = load_prompt(name)
    if not case.runtime.get("incremental_judgment"):
        return prompt
    if name == "planner_assimilate.md":
        # The legacy reuse section is the final section of this maintained prompt.
        prompt = prompt.split("# Decision economy (only when decision_economy=true)", 1)[0]
    return prompt + "\n" + load_prompt("planner_incremental.md")
