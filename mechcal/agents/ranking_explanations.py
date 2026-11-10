"""Same Planner, read-only explanation step; never a second ranking authority."""

from mechcal.runtime.prompts import load_prompt
from mechcal.runtime.ranking_explanations import RankingExplanationDraft, comparison_enabled, retain_comparisons
from mechcal.runtime.retry import PlannerCommandRejected, SchemaRetryError, SchemaRetryRunner
from mechcal.schemas.ranking_explanations import RankingExplanations


def explain_ranking(planner, case, update, *, references, memory, cache, event_sink=None):
    if not comparison_enabled(case) or retain_comparisons(case, update):
        return update
    draft = RankingExplanationDraft(case, update)
    if draft.draft_id in cache:
        return draft.apply(cache[draft.draft_id])

    def record(event):
        if event_sink:
            event_sink({**event, "stage": "planner_explain_ranking", "draft_id": draft.draft_id,
                        "scientific_evidence": False})

    payload = draft.payload()
    record({"kind": "ranking_explanation_draft", "payload": payload})

    def produce(feedback):
        kwargs = dict(payload=payload, system_prompt=load_prompt("planner_explain_ranking.md"),
                      schema_feedback=feedback, event_sink=record)
        if memory is not None:
            # Reuse this decision's retrieval allowance, not a fresh quota per stage/retry.
            return memory.complete(planner.llm_client, references, **kwargs)
        return references.complete(planner.llm_client, **kwargs)

    try:
        response = SchemaRetryRunner(
            RankingExplanations, policy=planner.runtime.retry_runner.policy,
            validators=[draft.apply], event_sink=record, repair_context=True,
        ).run(produce)
    except SchemaRetryError as exc:
        raise PlannerCommandRejected(exc.errors, action="explain_ranking") from exc
    cache[draft.draft_id] = response
    return draft.apply(response)
