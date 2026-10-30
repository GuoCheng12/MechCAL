"""Planner command boundary: one request, typed feedback, bounded correction."""

import json
import time

from mechcal.agents.planner_references import PlannerReferences
from mechcal.agents.ranking_explanations import explain_ranking
from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime.judgment_commands import materialize_judgment, snapshot_feedback
from mechcal.agents.command_repair import CommandRepairProducer
from mechcal.runtime.question_changes import CREATE_FIELDS
from mechcal.runtime.prompts import load_prompt
from mechcal.runtime.questions import (
    apply_final_revision,
    public_payload,
    replace_state,
    state,
    validate_assimilation,
)
from mechcal.runtime.result_contract import constrain_disposition_schema
from mechcal.runtime.retry import (
    PlannerCommandRejected,
    RejectedResponseError,
    RequestValidationError,
    SchemaRetryError,
    SchemaRetryRunner,
)
from mechcal.schemas.judgment_commands import (
    CommandReceipt, FinalJudgmentCommand, FinalJudgmentProposal, JudgmentCommand, JudgmentProposal,
)
from mechcal.schemas.question_loop import FinalArgumentRevision


def run_judgment_command(
    planner,
    case,
    *,
    final=False,
    validation_event_sink=None,
    request_event_sink=None,
    memory_store=None,
):
    if not planner._can_use_llm():
        raise RuntimeError("Judgment commands require a configured Planner.")
    stage = "planner_final_revision" if final else "planner_assimilate"
    comparison_first = case.runtime.get("predecision_comparison", False)
    model = ((FinalJudgmentCommand if final else JudgmentCommand) if comparison_first
             else (FinalJudgmentProposal if final else JudgmentProposal))
    action = "revise_final" if final else "update_judgment"
    payload = public_payload(case, planner.capability_registry, stage=stage)
    payload["output_contract"] = model.model_json_schema()
    target = (
        payload["output_contract"]["$defs"]["JudgmentCommand" if comparison_first else "JudgmentProposal"]
        if final
        else payload["output_contract"]
    )
    initial = state(case).get("revision", 0) == 0
    full_pool = case.runtime.get("evidence_decision_contracts", False)
    definitions = payload["output_contract"]["$defs"]
    definitions["HypothesisChange"]["properties"]["name"]["enum"] = list(MECHANISM_POOL)
    if initial:
        hypothesis = definitions["HypothesisChange"]
        for field in ("confidence", "differential_priority", "rationale"):
            hypothesis["required"].append(field)
            prop = hypothesis["properties"][field]
            hypothesis["properties"][field] = next(
                choice for choice in prop["anyOf"] if choice.get("type") != "null"
            )
        target["properties"]["hypotheses"]["minItems"] = len(MECHANISM_POOL) if full_pool else 1
        target["required"].append("hypotheses")
        if full_pool:
            target["properties"]["hypotheses"]["maxItems"] = len(MECHANISM_POOL)
            ranking_type = "RankingUpdate" if comparison_first else "RankingRationale"
            target["properties"]["ranking_update"] = {"$ref": "#/$defs/" + ranking_type}
            target["required"].append("ranking_update")
            ranking = definitions[ranking_type]
            for field, count in (("candidate_bases", len(MECHANISM_POOL)),):
                ranking["properties"][field]["minItems"] = count
                if field not in ranking.get("required", []):
                    ranking.setdefault("required", []).append(field)
    if not state(case).get("questions"):
        target["properties"]["questions"]["minItems"] = 1
        target["properties"]["questions"].pop("default", None)
        target["required"] = [*target.get("required", []), "questions"]
    # Refinements reference the same definitions whether nested or top-level.
    constrain_disposition_schema(
        {"properties": target["properties"], "$defs": payload["output_contract"]["$defs"]},
        payload["disposition_requirements"],
    )
    payload["incremental_update"] = {
        "enabled": True,
        "interface": "judgment_proposal_v3",
        "base_revision": payload["revision"],
        "contract": "One entry per changed hypothesis; omitted/null fields retain prior values. "
        "Omitted records remain unchanged. Runtime owns case/revision, current/runner_up "
        "and historical change receipts. No delta, retained_judgment or portfolio output. "
        "Rank-bound explanations are requested separately against the frozen resulting order.",
    }
    payload["hypothesis_write_contract"] = {
        "initial_judgment": initial,
        "required_initial_labels": list(MECHANISM_POOL) if initial and full_pool else [],
        "new_label_required_fields": ["confidence", "differential_priority", "rationale"],
        "notice": "The final answer contains three predictions. The internal judgment retains "
        "the full pool when boundary_decision_context is enabled. New labels need explicit "
        "values, even at low priority; a name alone cannot inherit a judgment that does not exist. "
        "Provide candidate bases for all initial labels. Do not submit rank-bound comparisons in this proposal.",
    }
    payload["question_write_contract"] = {
        "create_required_fields": list(CREATE_FIELDS),
        "existing_question_ids": list(state(case).get("questions", {})),
        "notice": "Use create for a new ID, update for an existing ID. Omitted/null update fields "
        "retain accepted values; [] explicitly clears a list. Runtime statuses are read-only.",
    }
    if comparison_first:
        payload["incremental_update"].update(
            interface="compared_judgment_v1",
            contract="Read predecision_review before writing judgment. Submit ranking and its comparisons together. "
            "Nothing is frozen until validation and commit. Omitted records retain accepted values.",
        )
        payload["hypothesis_write_contract"]["notice"] = (
            "Initial judgment covers the full pool. Submit rank comparisons in ranking_update with your chosen order. "
            "Respond to predecision_review in comparison_response; disagreement is permitted. "
            "Next-test usefulness does not itself establish final mechanism support."
        )
    pending_ids = [r["result_id"] for r in payload["pending_results"]]
    payload["assimilation_context"] = {
        "purpose": "respond_to_argument_review"
        if payload["argument_review_pending"]
        else "assimilate_results"
        if pending_ids
        else "update_investigation",
        "required_disposition_result_ids": pending_ids,
        "result_review_is_current": bool(pending_ids) and payload["result_review"] is not None,
    }
    if final:
        payload.update(
            final_internal_audit=state(case)["final_audit"],
            previous_final_decision=state(case)["decision"],
            final_evidence_delivery={
                "current_arguments": [{"support_id": a.support_id, "label": a.target_label}
                                      for a in case.mechanism_support_arguments],
                "requirement": "For each final Top-3 label submit current argument IDs or an explicit "
                "no_evidence reason. Write any missing justified arguments in update.mechanism_support_arguments. "
                "Hypothesis explanations and references alone are not exported. Do not invent positive support.",
            },
        )
    memory = planner._memory_session(case, memory_store, payload)
    if memory is not None:
        payload = memory.payload
    refs = PlannerReferences(case)
    prompt = load_prompt("planner_judgment_command.md")
    if final:
        prompt += "\n" + load_prompt("planner_final_command.md")
    validation_history = []
    comparison_cache = {}
    accepted = None

    def complete(**kwargs):
        if memory is not None:
            return memory.complete(planner.llm_client, refs, **kwargs)
        return refs.complete(planner.llm_client, **kwargs)

    produce = CommandRepairProducer(complete=complete, refs=refs, model=model,
        payload=payload, prompt=prompt, event_sink=request_event_sink,
        validation_history=validation_history)

    def expand(item):
        command = item.update if final else item
        try:
            update = materialize_judgment(case, command)
        except RequestValidationError as exc:
            if not final:
                raise
            raise RequestValidationError([
                {**issue, "command_field": "update"} for issue in exc.issues
            ]) from exc
        if final:
            from mechcal.runtime.final_evidence import validate_final_evidence
            validate_final_evidence(update.portfolio, update.mechanism_support_arguments, item.evidence_submission)
            return FinalArgumentRevision(
                update=update,
                evidence_submission=item.evidence_submission,
                final_action=item.final_action.model_copy(deep=True),
                audit_responses=dict(item.audit_responses),
            )
        return update

    def validate(item):
        nonlocal accepted
        expanded = expand(item)
        try:
            validate_assimilation(
                expanded.update if final else expanded,
                replace_state(case, phase="assimilate") if final else case,
                check_comparisons=False,
            )
        except RequestValidationError as exc:
            raise snapshot_feedback(exc, expanded) from exc
        if not comparison_first:
            explained = explain_ranking(
                planner, case, expanded.update if final else expanded,
                references=refs, memory=memory, cache=comparison_cache,
                event_sink=request_event_sink,
            )
            expanded = expanded.model_copy(update={"update": explained}) if final else explained
        try:
            if final:
                apply_final_revision(case, expanded, planner.capability_registry)
            else:
                validate_assimilation(expanded, case)
        except RequestValidationError as exc:
            # Validation inspects a full snapshot; the model edits records by key.
            raise snapshot_feedback(exc, expanded) from exc
        accepted = expanded

    def record(event):
        if request_event_sink is not None:
            request_event_sink(event)
        if event["kind"] not in {"validation_error", "accepted"}:
            return
        rejected = event["kind"] == "validation_error"
        if rejected:
            validation_history.append({
                "attempt": event["attempt"], "error": event["error"],
                "issues": event.get("issues", []),
            })
        receipt = CommandReceipt(
            action=action,
            status="rejected" if rejected else "validated",
            case_id=case.case_id,
            base_revision=payload["revision"],
            attempt=event["attempt"],
            message=event.get(
                "error", "Command validated; the runtime will commit the resulting snapshot."
            ),
            issues=event.get("issues", []),
        )
        if request_event_sink is not None:
            request_event_sink(
                {
                    "kind": "command_receipt",
                    **receipt.model_dump(mode="json"),
                    "scientific_evidence": False,
                }
            )
        if rejected and validation_event_sink is not None:
            validation_event_sink(
                {
                    **event,
                    "timestamp": time.time(),
                    "stage": stage,
                    "case_id": case.case_id,
                    "round_id": f"R{state(case).get('round_index', 1):03d}",
                    "revision": payload["revision"] + 1,
                }
            )

    try:
        SchemaRetryRunner(
            model,
            policy=planner.runtime.retry_runner.policy,
            validators=[validate],
            event_sink=record,
            repair_context=True,
            separate_validation_budget=True,
        ).run(produce)
    except PlannerCommandRejected:
        raise
    except SchemaRetryError as exc:
        raise PlannerCommandRejected(exc.errors, action=action) from exc
    # Do not rematerialize the command and discard its validated explanations.
    assert accepted is not None
    return accepted
