"""The same Planner completes final evidence, without repeating runtime linkage."""

from mechcal.agents.planner_references import PlannerReferences
from mechcal.runtime.final_delivery import apply_patch, candidate_case, preflight
from mechcal.runtime.prompts import load_prompt
from mechcal.runtime.questions import public_payload
from mechcal.runtime.scoped_contracts import final_delivery_schema
from mechcal.runtime.retry import PlannerCommandRejected, SchemaRetryError, SchemaRetryRunner
from mechcal.schemas.final_delivery import FinalDeliveryPatch


def complete_delivery(planner, case, draft, *, store, reserve_attempt, event_sink):
    current = candidate_case(case, draft)
    payload = public_payload(current, planner.capability_registry, stage="planner_complete_delivery")
    payload.update(
        output_contract=final_delivery_schema(FinalDeliveryPatch.model_json_schema(), current, draft),
        final_internal_audit=draft.audit.model_dump(mode="json") if draft.audit else None,
        final_delivery={
            "generation": draft.generation, "stage": draft.stage,
            "top3": [h.name for h in case.portfolio.sorted_hypotheses()[:3]],
            "no_evidence": [e.model_dump(mode="json") for e in draft.no_evidence],
            "audit_responses": draft.audit_responses,
            "reviewed_support_ids": draft.reviewed_support_ids,
            "final_answer_draft": draft.final_answer_draft,
            "issues": preflight(case, draft, planner.capability_registry),
        },
    )
    memory = planner._memory_session(current, store, payload)
    if memory is not None:
        payload = memory.payload
    references = PlannerReferences(current)
    accepted = None

    def produce(feedback):
        reserve_attempt()
        kwargs = dict(payload=payload, system_prompt=load_prompt("planner_complete_delivery.md"),
                      schema_feedback=feedback, event_sink=event_sink)
        if memory is not None:
            return memory.complete(planner.llm_client, references, **kwargs)
        return references.complete(planner.llm_client, **kwargs)

    def validate(patch):
        nonlocal accepted
        accepted = apply_patch(case, draft, patch)

    try:
        SchemaRetryRunner(FinalDeliveryPatch, policy=planner.runtime.retry_runner.policy,
                          validators=[validate], event_sink=event_sink, repair_context=True).run(produce)
    except SchemaRetryError as exc:
        raise PlannerCommandRejected(exc.errors, action="complete_delivery") from exc
    return accepted
