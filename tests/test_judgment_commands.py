"""Offline interface tests. Scripted judgments are not scientific results."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest
from mechcal.agents.planner import PlannerAgent
from mechcal.runtime.judgment_commands import materialize_judgment
from mechcal.runtime.question_changes import question_view
from mechcal.runtime.questions import (
    apply_assimilation,
    replace_state,
    state,
    validate_assimilation,
)
from mechcal.runtime.retry import PlannerCommandRejected, RequestValidationError
from mechcal.schemas import CaseRun
from mechcal.schemas.judgment_commands import HypothesisChange, JudgmentCommand
from pydantic import ValidationError
from test_decision_economy import economy_start
from test_incremental_judgment import prepared
from test_question_loop import REGISTRY


def command(case, **changes):
    return JudgmentCommand(
        reason="Offline interface fixture; retain unmodified scientific content.",
        decision_assessment=state(case)["assimilation"]["decision_assessment"],
        coverage_response="No new coverage feedback.",
        **changes,
    )


def initial_command(update, existing_question_ids=()):
    return JudgmentCommand(
        reason="Offline initial judgment.",
        hypotheses=[
            HypothesisChange.model_validate(h.model_dump(exclude={"schema_version", "record_type"}))
            for h in update.portfolio.hypotheses
        ],
        questions=[{
            **question_view(q.model_dump(mode="json")),
            "decision_role": q.decision_role or "material",  # Explicit offline fixture role.
            "operation": "update" if q.question_id in existing_question_ids else "create",
        } for q in update.questions],
        mechanism_support_arguments=update.mechanism_support_arguments,
        ranking_update=update.ranking_update,
        decision_assessment=update.decision_assessment,
        coverage_response=update.coverage_response,
    )


def test_initial_command_and_subsequent_empty_update_preserve_full_snapshots():
    case, original = economy_start()
    case.runtime["incremental_judgment"] = True
    request = initial_command(original)
    before = case.model_dump()
    update = materialize_judgment(case, request)
    validate_assimilation(update, case)
    assert update.portfolio == original.portfolio
    assert case.model_dump() == before
    accepted = replace_state(apply_assimilation(case, update), phase="assimilate")
    request = command(accepted)
    assert len(request.model_dump_json()) < len(original.model_dump_json()) / 2
    updated = apply_assimilation(accepted, materialize_judgment(accepted, request))
    assert updated.portfolio == accepted.portfolio
    assert updated.mechanism_support_arguments == accepted.mechanism_support_arguments
    assert updated.evidence_ledger == accepted.evidence_ledger
    assert CaseRun.model_validate_json(updated.model_dump_json()) == updated


def test_single_hypothesis_write_preserves_unsupplied_fields_and_command():
    case = prepared()
    original = case.portfolio.hypotheses[0]
    request = command(
        case,
        hypotheses=[
            HypothesisChange(
                name=original.name,
                confidence=0.38,
                differential_priority=0.88,
                rationale="Updated interpretation.",
            )
        ],
    )
    before = request.model_dump()
    update = materialize_judgment(case, request)
    changed = next(h for h in update.portfolio.hypotheses if h.name == original.name)
    assert (changed.confidence, changed.differential_priority) == (0.38, 0.88)
    assert changed.evidence_refs == original.evidence_refs
    assert changed.validation_needed == original.validation_needed
    assert update.portfolio.hypotheses[1:] == case.portfolio.hypotheses[1:]
    assert request.model_dump() == before


@pytest.mark.parametrize(
    "legacy",
    [
        "delta",
        "portfolio",
        "retained_judgment",
        "hypothesis_scores",
        "case_id",
        "revision",
        "argument_changes",
    ],
)
def test_only_one_model_facing_interface(legacy):
    case = prepared()
    data = command(case).model_dump()
    data[legacy] = {}
    with pytest.raises(ValidationError):
        JudgmentCommand.model_validate(data)
    assert legacy not in JudgmentCommand.model_json_schema()["properties"]


@pytest.mark.parametrize(
    "fault", ["duplicate", "unknown", "empty", "withdraw_unknown", "withdraw_overlap"]
)
def test_bad_commands_are_atomic(fault):
    case = prepared()
    request = command(case)
    if fault == "duplicate":
        change = HypothesisChange(name=case.portfolio.current, confidence=0.38)
        request.hypotheses = [change, change]
    elif fault == "unknown":
        request.hypotheses = [HypothesisChange(name="invented", confidence=0.3)]
    elif fault == "empty":
        request.hypotheses = [HypothesisChange(name=case.portfolio.current)]
    elif fault == "withdraw_unknown":
        request.withdrawn_support_ids = ["unknown"]
    else:
        request.mechanism_support_arguments = [case.mechanism_support_arguments[0]]
        request.withdrawn_support_ids = [case.mechanism_support_arguments[0].support_id]
    baseline, wire = deepcopy(case.model_dump()), deepcopy(request.model_dump())
    with pytest.raises(RequestValidationError):
        materialize_judgment(case, request)
    assert case.model_dump() == baseline
    assert request.model_dump() == wire


def test_explicit_empty_list_clears_but_null_retains():
    case = prepared()
    label = case.portfolio.current
    original = next(h for h in case.portfolio.hypotheses if h.name == label)
    original.evidence_refs = ["input:smiles"]
    request = command(
        case, hypotheses=[HypothesisChange(name=label, confidence=0.2, evidence_refs=None)]
    )
    assert materialize_judgment(case, request).portfolio.hypotheses[0].evidence_refs == [
        "input:smiles"
    ]
    request.hypotheses[0].evidence_refs = []
    assert materialize_judgment(case, request).portfolio.hypotheses[0].evidence_refs == []


def test_unknown_scientific_refs_and_stale_commit_still_rejected():
    case = prepared()
    request = command(
        case, hypotheses=[HypothesisChange(name=case.portfolio.current, evidence_refs=["foreign"])]
    )
    with pytest.raises(ValueError, match="Unavailable or inadmissible"):
        validate_assimilation(materialize_judgment(case, request), case)
    update = materialize_judgment(case, command(case))
    next_case = replace_state(apply_assimilation(case, update), phase="assimilate")
    with pytest.raises(ValueError, match="revision"):
        apply_assimilation(next_case, update)
    other = case.model_copy(update={"case_id": "different-case"})
    with pytest.raises(ValueError, match="case"):
        apply_assimilation(other, update)


def test_local_audit_cannot_silently_change_ranking():
    case = prepared()
    support = case.mechanism_support_arguments[0]
    case = replace_state(
        case, argument_review_pending=True, argument_review={"support_ids": [support.support_id]}
    )
    request = command(
        case,
        scope="local_audit",
        audit_responses={support.support_id: "Scope clarified."},
        mechanism_support_arguments=[support.model_copy(update={"boundary": "Clarified scope."})],
    )
    update = materialize_judgment(case, request)
    validate_assimilation(update, case)
    assert update.portfolio == case.portfolio
    request.hypotheses = [HypothesisChange(name=case.portfolio.current, confidence=0.38)]
    with pytest.raises(RequestValidationError, match="scope=global"):
        materialize_judgment(case, request)
    request.scope = "global"
    assert materialize_judgment(case, request).portfolio.hypotheses[0].confidence == 0.38


class ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def is_configured(self):
        return True

    def complete_json(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        if kwargs["payload"]["stage"] == "planner_explain_ranking":
            return explanation_response(kwargs["payload"])
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(kwargs["payload"])
        return proposal_wire(response)


def proposal_wire(response):
    """Translate legacy offline fixtures, never production model responses."""
    result = deepcopy(response)
    body = result.get("update", result)
    ranking = body.get("ranking_update")
    if isinstance(ranking, dict):
        for field in ("record_type", "schema_version", "leading_comparison", "boundary_comparisons"):
            ranking.pop(field, None)
    return result


def explanation_response(payload):
    """Synthetic unresolved explanations for contract tests, not scientific evidence."""
    from test_question_loop import NOTE
    leader = payload["leading_target"]
    return dict(action="explain_ranking", draft_id=payload["draft_id"],
                leading=dict(challenger=payload["proposed_ranking"][1]["name"],
                             leading_hypothesis=NOTE, alternative_hypothesis=NOTE,
                             basis_for_preference=NOTE, preference_status="unresolved",
                             evidence_refs=[], previous_basis_update=NOTE) if leader else None,
                boundaries=[dict(target_id=t["target_id"], outcome="unresolved", basis="prior_only",
                                 evidence_refs=[], common_standard=NOTE, reason_for_order=NOTE,
                                 remaining_difference=NOTE) for t in payload["boundary_targets"]])


def planner(case, responses):
    case.runtime["planner_case_memory"] = False
    client = ScriptedClient(responses)
    return PlannerAgent(
        llm_client=client, enable_llm_decisions=True, capability_registry=REGISTRY
    ), client


def repair_fields(**values):
    def respond(payload):
        edits = []
        for target in payload["targets"]:
            edit = {"target_id": target["target_id"], "operation": target["operation"]}
            if target["operation"] == "replace":
                edit["value"] = values[target["field"]]
            edits.append(edit)
        return {"action": "repair_judgment", "draft_id": payload["draft_id"], "edits": edits}
    return respond


def test_planner_corrects_only_authorized_draft_field_without_evidence_mutation():
    case = prepared()
    good = command(case).model_dump(mode="json")
    bad = {**good, "portfolio": {}}
    agent, client = planner(case, [bad, repair_fields()])
    before = deepcopy(case.model_dump())
    events = []
    update = agent.assimilate(case, request_event_sink=events.append)
    assert len(client.calls) == 2
    assert all("output_contract" in call["payload"] for call in client.calls)
    properties = client.calls[0]["payload"]["output_contract"]["properties"]
    assert (
        "hypotheses" in properties and "portfolio" not in properties and "delta" not in properties
    )
    feedback = json.loads(client.calls[1]["schema_feedback"])
    assert "rejected_response" not in feedback
    assert client.calls[1]["payload"]["targets"][0]["field"] == "portfolio"
    assert "allowed_paths" not in feedback
    receipts = [event for event in events if event["kind"] == "command_receipt"]
    assert [r["status"] for r in receipts] == ["rejected", "validated"]
    assert all(not r["scientific_evidence"] for r in receipts)
    assert update.portfolio == case.portfolio
    assert case.model_dump() == before


def test_repair_exhaustion_is_bounded_and_program_errors_are_not_retried():
    case = prepared()
    bad = {**command(case).model_dump(mode="json"), "delta": {}}
    agent, client = planner(case, [bad] * 3)
    with pytest.raises(PlannerCommandRejected):
        agent.assimilate(case)
    assert len(client.calls) == 3
    agent, client = planner(case, [ValueError("local program failure")])
    with pytest.raises(ValueError, match="local program failure"):
        agent.assimilate(case)
    assert len(client.calls) == 1


def test_command_state_is_isolated_between_cases():
    first = prepared()
    second = first.model_copy(deep=True)
    second.case_id = "case-two"
    second.portfolio.hypotheses[0].rationale = "Second case's own interpretation."
    request = command(first)
    one, two = materialize_judgment(first, request), materialize_judgment(second, request)
    assert one.case_id != two.case_id
    assert one.portfolio.hypotheses[0].rationale != two.portfolio.hypotheses[0].rationale
    assert one.revision == two.revision


@pytest.mark.parametrize("has_questions", [False, True])
def test_command_contract_requires_questions_only_when_none_can_be_retained(has_questions):
    if has_questions:
        case = prepared()
        response = command(case)
    else:
        case, update = economy_start()
        case.runtime["incremental_judgment"] = True
        response = initial_command(update)
    agent, client = planner(case, [response.model_dump(mode="json")])
    agent.assimilate(case)
    contract = client.calls[0]["payload"]["output_contract"]
    assert ("questions" in contract["required"]) == (not has_questions)
    assert contract["properties"]["questions"].get("minItems", 0) == int(not has_questions)


@pytest.mark.parametrize("reverse_array", [False, True])
def test_cutoff_explanations_use_frozen_order_after_proposal_corrections(reverse_array):
    from test_boundary_decisions import start

    case, update = start()
    case.runtime["incremental_judgment"] = True
    good = initial_command(update).model_dump(mode="json")
    if reverse_array:
        good["hypotheses"].reverse()
    no_question = deepcopy(good)
    no_question["questions"] = []
    agent, client = planner(case, [no_question, repair_fields(questions=good["questions"])])
    before = deepcopy(case.model_dump())
    accepted = agent.assimilate(case)
    feedback = json.loads(client.calls[1]["schema_feedback"])
    assert len(feedback["validation_history"]) == 1
    assert "investigation question" in feedback["validation_history"][0]["error"]
    assert "rejected_response" not in feedback
    ranking = accepted.portfolio.sorted_hypotheses()
    explanation = client.calls[2]["payload"]
    assert [h["name"] for h in explanation["proposed_ranking"]] == [h.name for h in ranking]
    assert explanation["boundary_targets"][0] == {
        "target_id": "cutoff:1",
        "included_label": ranking[2].name, "excluded_label": ranking[3].name,
    }
    assert {h.name: h.differential_priority for h in ranking} == {
        h["name"]: h["differential_priority"] for h in good["hypotheses"]
    }
    assert case.model_dump() == before


def test_snapshot_feedback_uses_record_key_not_sparse_command_position():
    case = prepared()
    label = case.portfolio.hypotheses[-1].name
    bad = command(case, hypotheses=[HypothesisChange(name=label, evidence_refs=["foreign"])])
    agent, client = planner(
        case, [bad.model_dump(mode="json"), repair_fields(evidence_refs=[])]
    )
    agent.assimilate(case)
    issues = json.loads(client.calls[1]["schema_feedback"])["issues"]
    assert issues[0]["record_selector"] == {"name": label}
    assert issues[0]["feedback_scope"] == "materialized_snapshot"
    assert "repair_paths" not in issues[0] and "path" not in issues[0]


def test_failed_result_cannot_become_scientific_support_in_command():
    from mechcal.schemas.question_loop import QuestionAssimilation
    from test_question_loop import assimilation, pending_case

    case = pending_case()
    case.runtime["incremental_judgment"] = True
    update = QuestionAssimilation.model_validate(
        assimilation(
            dict(case_id=case.case_id, revision=1, pending_results=state(case)["pending_results"])
        )
    )
    request = initial_command(update, state(case)["questions"])
    request.dispositions = update.dispositions
    good = apply_assimilation(case, materialize_judgment(case, request))
    assert good.evidence_ledger == case.evidence_ledger
    request.dispositions[0].outcome = "informative"
    with pytest.raises(RequestValidationError, match="Failed execution"):
        validate_assimilation(materialize_judgment(case, request), case)


def command_loop(tmp_path, monkeypatch):
    from mechcal.runtime import MechCALOrchestrator
    from mechcal.schemas.question_loop import QuestionAssimilation, ResultReview
    from test_boundary_decisions import complete_update
    from test_decision_value import outcomes, value
    from test_evidence_comparison import ComparisonClient
    from test_question_loop import NOTE, TOOL, loop

    class Client(ComparisonClient):
        reject = False

        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            if payload["stage"] == "planner_explain_ranking":
                self.calls.append(deepcopy(payload))
                return explanation_response(payload)
            if payload["stage"] == "planner_judgment_repair":
                assert self.reject
                return {"action": "update_judgment", "portfolio": {}}
            if (
                payload["stage"] == "planner_assimilate"
                and payload["pending_results"]
                and self.reject
            ):
                self.calls.append(deepcopy(payload))
                return {"action": "update_judgment", "portfolio": {}}
            if payload["revision"] and not payload["retrieved_memory"]:
                return {"action": "read_case_records", "record_ids": ["Q1"]}
            data = super().complete_json(
                payload=payload, system_prompt=system_prompt, schema_feedback=schema_feedback
            )
            if data.get("action") == "describe_tools":
                return data
            stage = payload["stage"]
            if stage in {"planner_assimilate", "planner_final_revision"}:
                u = data["update"] if stage == "planner_final_revision" else data
                complete_update(u, payload["comparison_context"]["current_order"])
                for q in u["questions"]:
                    q["decision_role"] = "material"
                command_data = initial_command(QuestionAssimilation.model_validate(u),
                    [q["question_id"] for q in payload["questions"]]).model_dump(
                    mode="json"
                )
                command_data["dispositions"] = u["dispositions"]
                command_data["audit_responses"] = u["audit_responses"]
                if stage == "planner_final_revision":
                    from test_final_evidence_submission import submission_for
                    final_update = QuestionAssimilation.model_validate(u)
                    data["evidence_submission"] = submission_for(
                        final_update.portfolio, final_update.mechanism_support_arguments)
                    data["action"], data["update"] = "revise_final", command_data
                else:
                    data = command_data
            actions = [data["final_action"]] if stage == "planner_final_revision" else [data]
            for a in actions:
                for p in a.get("investigation_plans", []):
                    p["decision_value"] = value().model_dump(mode="json")
                    os = outcomes("support", "no_change")
                    os[0].affected_labels = [p["candidate_labels"][0]]
                    os[0].judgment_change = "weaken"
                    p["possible_outcomes"] = [o.model_dump(mode="json") for o in os]
                if a.get("action") in {"finalize", "stop"}:
                    a["continuation_assessment"] = dict(
                        scope="bounded_ranking",
                        remaining_uncertainty=NOTE,
                        decision_reason=NOTE,
                        routes=[dict(capability_id=TOOL, disposition="not_useful", reason=NOTE)],
                    )
            return proposal_wire(data)

    o, _ = loop(tmp_path, unlimited=True, discover=True)
    client = Client(1, True)
    o.deps.planner.llm_client = client
    original_review = o.deps.photophysics_arbiter.llm_client.complete_json

    def review_with_routing(**kwargs):
        response = original_review(**kwargs)
        if kwargs["payload"].get("require_explicit_audit_routing"):
            response["routing"] = {"mode": "global", "reason": NOTE}
        return response

    monkeypatch.setattr(
        o.deps.photophysics_arbiter.llm_client, "complete_json", review_with_routing
    )
    monkeypatch.setattr(
        o.deps.photophysics_arbiter,
        "review_results",
        lambda case, **kw: ResultReview(
            interpretations=[
                dict(
                    result_id=r["result_id"],
                    question_ids=r["question_ids"],
                    delivered=NOTE,
                    missing=NOTE,
                    boundary=NOTE,
                    evidence_refs=r["evidence_refs"],
                )
                for r in state(case).get("pending_results", [])
            ]
        ),
    )
    o = MechCALOrchestrator(
        replace(
            o.config,
            comparative_evidence=True,
            decision_value_checks=True,
            evidence_decision_contracts=True,
            decision_economy=True,
            enable_support_auditor=True,
            planner_case_memory=True,
            incremental_judgment=True,
        ),
        o.deps,
    )
    return o, client


@pytest.mark.parametrize("pause", [False, True])
def test_command_loop_final_audit_memory_pause_and_resume(tmp_path, monkeypatch, pause):
    from test_question_loop import NOTE

    o, client = command_loop(tmp_path, monkeypatch)
    client.reject = pause
    root = tmp_path / "command-loop"
    if pause:
        with pytest.raises(PlannerCommandRejected):
            o.run(smiles="CCO", user_query=NOTE, case_id="command-loop")
        saved = CaseRun.model_validate_json((root / "question_checkpoint.json").read_text())
        assert saved.status == "paused"
        assert state(saved)["phase"] == "assimilate" and state(saved)["revision"] > 0
        assert len(state(saved)["pending_results"]) == 1
        assert saved.runtime["termination"]["scientific_completion"] is False
        assert saved.runtime["termination"]["requires_explicit_resume"] is True
        with pytest.raises(FileExistsError):
            o.run(smiles="CCO", user_query=NOTE, case_id="command-loop")
        with pytest.raises(ValueError, match="differs"):
            o.run(smiles="CCC", user_query=NOTE, case_id="command-loop", resume=True)
        prior_tools = (root / "history/tool_events.jsonl").read_text()
    client.reject = False
    result = o.run(smiles="CCO", user_query=NOTE, case_id="command-loop", resume=pause)
    assert result.status == "finalized"
    assert result.runtime["worker_call_counts"]["macro"] == 1
    assert len(state(result)["results"]) == 1
    assert state(result)["final_revision_complete"]
    if pause:
        assert (root / "history/tool_events.jsonl").read_text() == prior_tools
    events = [
        json.loads(line)
        for line in (root / "history/planner_requests.jsonl").read_text().splitlines()
    ]
    assert any(e["kind"] == "command_receipt" and e["action"] == "revise_final" for e in events)
    assert all("rubric" not in p and "hidden_reference" not in p for p in client.calls)
    queries = [
        json.loads(line)
        for line in (root / "history/memory_queries.jsonl").read_text().splitlines()
    ]
    assert any(q["kind"] == "memory_query" for q in queries)
    before = len(client.calls)
    assert o.run(smiles="CCO", user_query=NOTE, case_id="command-loop", resume=True) == result
    assert len(client.calls) == before
