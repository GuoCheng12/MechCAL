"""Offline state-machine and adversarial tests. No scientific/model/tool claims."""

from copy import deepcopy
import random

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from mechcal.agents.planner import PlannerAgent
from mechcal.runtime.judgment_commands import materialize_judgment
from mechcal.runtime.questions import apply_assimilation, replace_state, state, validate_assimilation
from mechcal.runtime.ranking_explanations import RankingExplanationDraft
from mechcal.runtime.retry import PlannerCommandRejected, RequestValidationError
from mechcal.runtime.structured_output import PlannerOutputContract
from mechcal.schemas import CaseRun
from mechcal.schemas.hypothesis_tests import LeadingComparison
from mechcal.schemas.judgment_commands import JudgmentProposal, RankingRationale
from mechcal.schemas.ranking_explanations import RankingExplanations
from test_boundary_decisions import start
from test_judgment_commands import command, explanation_response, planner, proposal_wire
from test_question_loop import NOTE, REGISTRY


def prepared_case():
    case, update = start()
    case.runtime.update(incremental_judgment=True, hypothesis_tests=True, decision_economy=True)
    ranked = update.portfolio.sorted_hypotheses()
    update.ranking_update.leading_comparison = LeadingComparison(
        leader=ranked[0].name, challenger=ranked[1].name, leading_hypothesis=NOTE,
        alternative_hypothesis=NOTE, basis_for_preference=NOTE,
        preference_status="unresolved", evidence_refs=[], previous_basis_update=NOTE)
    for q in update.questions:
        q.decision_role = "material"
    return replace_state(apply_assimilation(case, update), phase="assimilate", round_index=1)


def proposal(case, changes=()):
    data = proposal_wire(command(case).model_dump(mode="json"))
    data["hypotheses"] = list(changes)
    if changes:
        data["ranking_update"] = RankingRationale(
            basis="reinterpretation", explanation=NOTE,
            alternatives_considered=[h.name for h in case.portfolio.hypotheses]).model_dump(mode="json")
    return data


def draft_for(case, data):
    update = materialize_judgment(case, JudgmentProposal.model_validate(data))
    validate_assimilation(update, case, check_comparisons=False)
    return RankingExplanationDraft(case, update)


@pytest.mark.parametrize("change", ["demote_leader", "promote_tail", "two_displaced", "tie", "zero"])
@pytest.mark.parametrize("strict", [False, True])
def test_actual_score_changes_bind_explanations_after_merge(change, strict):
    case = prepared_case()
    ranked = case.portfolio.sorted_hypotheses()
    if change == "promote_tail":
        changes = [dict(name=ranked[-1].name, differential_priority=1.0)]
    elif change == "two_displaced":
        changes = [dict(name=h.name, differential_priority=0) for h in ranked[:2]]
    else:
        value = {"demote_leader": (ranked[1].differential_priority or 0) - .001,
                 "tie": ranked[1].differential_priority, "zero": 0}[change]
        changes = [dict(name=ranked[0].name, differential_priority=max(0, value))]
    data = proposal(case, changes)
    expected = materialize_judgment(case, JudgmentProposal.model_validate(data))

    class Client:
        def __init__(self):
            self.calls = []

        def is_configured(self):
            return True

        def complete_json(self, **kwargs):
            payload = kwargs["payload"]
            self.calls.append(deepcopy(payload))
            result = explanation_response(payload) if payload["stage"] == "planner_explain_ranking" else data
            Draft202012Validator(payload["output_contract"]).validate(result)
            return PlannerOutputContract(payload).encode(result) if strict else deepcopy(result)

    case.runtime["planner_strict_outputs"] = strict
    client = Client()
    agent = PlannerAgent(llm_client=client, enable_llm_decisions=True, capability_registry=REGISTRY)
    before = case.model_dump_json()
    result = agent.assimilate(case)
    assert len(client.calls) == 2
    assert case.model_dump_json() == before
    assert result.portfolio == expected.portfolio
    assert result.ranking_update.leading_comparison.leader == result.portfolio.current
    validate_assimilation(result, case)
    assert "hypotheses" not in client.calls[-1]["output_contract"]["properties"]
    assert "leader" not in client.calls[-1]["output_contract"]["$defs"]["LeaderExplanation"]["properties"]
    assert "included_label" not in client.calls[-1]["output_contract"]["$defs"]["BoundaryExplanation"]["properties"]


def test_unchanged_and_noop_scores_reuse_but_interpretation_change_refreshes():
    case = prepared_case()
    h = case.portfolio.hypotheses[0]
    same = proposal(case)
    same["hypotheses"] = [dict(name=h.name, differential_priority=h.differential_priority)]
    agent, client = planner(case, [same])
    result = agent.assimilate(case)
    assert len(client.calls) == 1
    assert result.ranking_update.leading_comparison.model_dump(mode="json") == state(case)["assimilation"]["ranking_update"]["leading_comparison"]
    changed = proposal(case, [dict(name=h.name, rationale="A changed interpretation, not a changed score.")])
    agent, client = planner(case, [changed])
    result = agent.assimilate(case)
    assert len(client.calls) == 2
    assert result.portfolio.current == case.portfolio.current


@pytest.mark.parametrize("fault", ["stale", "foreign_case", "foreign_evidence", "operational_evidence",
                                    "missing_leading", "same_challenger", "missing_pair", "duplicate_pair",
                                    "foreign_target", "unreferenced_support", "edit_scores", "edit_leader"])
def test_explanation_cannot_mutate_scores_or_commit_invalid_content(fault):
    case = prepared_case()
    if fault == "operational_evidence":
        from test_runtime import _evidence_unit
        case.evidence_ledger.items.append(_evidence_unit(
            evidence_id="execution:failed", round_id="R001", source_report_id="failed-report",
            agent_name="microscopic_worker", family="geometry_precondition", summary=NOTE,
        ).model_copy(update={"status": "failed"}))
    draft = draft_for(case, proposal(case))
    response = explanation_response(draft.payload())
    if fault in {"stale", "foreign_case"}:
        other = case.model_copy(deep=True)
        if fault == "foreign_case":
            other.case_id = "another-case"
        else:
            state(other)["revision"] += 1
        response["draft_id"] = draft_for(other, proposal(other)).draft_id
    elif fault in {"foreign_evidence", "operational_evidence"}:
        response["leading"]["evidence_refs"] = ["foreign:E1" if fault == "foreign_evidence" else "execution:failed"]
    elif fault == "missing_leading":
        response["leading"] = None
    elif fault == "same_challenger":
        response["leading"]["challenger"] = draft.ranked[0].name
    elif fault == "missing_pair":
        response["boundaries"] = []
    elif fault == "duplicate_pair":
        response["boundaries"] *= 2
    elif fault == "foreign_target":
        response["boundaries"][0]["target_id"] = "foreign"
    elif fault == "unreferenced_support":
        response["leading"]["preference_status"] = "evidence_favors_leader"
    elif fault == "edit_scores":
        response["hypotheses"] = [{"name": case.portfolio.current, "differential_priority": 1}]
    else:
        response["leading"]["leader"] = draft.ranked[-1].name
    before, proposal_before = case.model_dump_json(), draft.update.model_dump_json()
    with pytest.raises((RequestValidationError, ValidationError)):
        draft.apply(RankingExplanations.model_validate(response))
    assert case.model_dump_json() == before
    assert draft.update.model_dump_json() == proposal_before


def test_comparison_retry_does_not_rewrite_proposal_or_restart_allowance():
    case = prepared_case()
    data = proposal(case, [dict(name=case.portfolio.current, differential_priority=0)])

    class Client:
        def __init__(self, repair):
            self.calls, self.repair = [], repair

        def is_configured(self):
            return True

        def complete_json(self, **kwargs):
            p = kwargs["payload"]
            self.calls.append(deepcopy(p))
            if p["stage"] == "planner_assimilate":
                return deepcopy(data)
            assert p["stage"] == "planner_explain_ranking"
            r = explanation_response(p)
            if not self.repair or len(self.calls) == 2:
                r["leading"] = None
            return r

    for repair in (True, False):
        client = Client(repair)
        agent = PlannerAgent(llm_client=client, enable_llm_decisions=True, capability_registry=REGISTRY)
        before = case.model_dump_json()
        if repair:
            result = agent.assimilate(case)
            validate_assimilation(result, case)
            assert len(client.calls) == 3
        else:
            with pytest.raises(PlannerCommandRejected) as error:
                agent.assimilate(case)
            assert error.value.action == "explain_ranking"
            assert len(client.calls) == 4  # One proposal, bounded three explanation attempts.
        assert case.model_dump_json() == before
        assert client.calls[1:] == [client.calls[1]] * (len(client.calls) - 1)


@pytest.mark.parametrize("seed", range(10))
def test_twenty_round_sequences_promote_demote_tie_and_persist(seed):
    rng = random.Random(seed)
    case = prepared_case()
    initial_evidence = case.evidence_ledger.model_dump_json()
    for turn in range(20):
        labels = rng.sample([h.name for h in case.portfolio.hypotheses], rng.randrange(1, 5))
        changes = [dict(name=label, differential_priority=rng.choice([0, .5, 1, rng.random()]))
                   for label in labels]
        request = proposal(case, changes)
        agent, client = planner(case, [request])
        result = agent.assimilate(case)
        scores = {h.name: h.differential_priority for h in case.portfolio.hypotheses}
        scores.update({h["name"]: h["differential_priority"] for h in changes})
        assert {h.name: h.differential_priority for h in result.portfolio.hypotheses} == scores
        assert result.ranking_update.leading_comparison.leader == result.portfolio.sorted_hypotheses()[0].name
        case = replace_state(apply_assimilation(case, result), phase="assimilate")
        assert state(case)["revision"] == turn + 2
        assert case.evidence_ledger.model_dump_json() == initial_evidence
        case = CaseRun.model_validate_json(case.model_dump_json())
        assert 1 <= len(client.calls) <= 2


def test_legacy_missing_priority_and_ties_use_one_ordering_function():
    case = prepared_case()
    h = case.portfolio.hypotheses[-1]
    h.differential_priority, h.confidence = None, 1.0
    update = materialize_judgment(case, JudgmentProposal.model_validate(proposal(case)))
    assert update.portfolio.current == h.name
    assert update.portfolio.sorted_hypotheses()[0].name == h.name


def test_incomplete_proposal_cannot_bypass_commit_validation():
    case = prepared_case()
    update = materialize_judgment(case, JudgmentProposal.model_validate(
        proposal(case, [dict(name=case.portfolio.current, differential_priority=0)])))
    validate_assimilation(update, case, check_comparisons=False)
    with pytest.raises(RequestValidationError):
        apply_assimilation(case, update)


@pytest.mark.parametrize("leading,boundary", [(False, False), (False, True), (True, False), (True, True)])
def test_explanation_targets_respect_feature_flags(leading, boundary):
    case = prepared_case()
    case.runtime.update(hypothesis_tests=leading, evidence_decision_contracts=boundary)
    data = proposal(case, [dict(name=case.portfolio.current, differential_priority=0)])
    agent, client = planner(case, [data])
    result = agent.assimilate(case)
    validate_assimilation(result, case)
    if leading or boundary:
        p = client.calls[-1]["payload"]
        assert bool(p["leading_target"]) == leading
        assert bool(p["boundary_targets"]) == boundary
        assert len(client.calls) == 2
    else:
        assert len(client.calls) == 1


def test_memory_allowance_shared_across_proposal_and_explanation(tmp_path):
    from mechcal.runtime.persistence import RunStore
    case = prepared_case()
    case.runtime["planner_case_memory"] = True
    case.runtime["planner_strict_outputs"] = True
    data = proposal(case, [dict(name=case.portfolio.current, differential_priority=0)])

    class Client:
        calls = []
        queried_proposal = False
        queried_explanation = False

        def is_configured(self):
            return True

        def complete_json(self, **kwargs):
            p = kwargs["payload"]
            self.calls.append(deepcopy(p))
            if p["stage"] == "planner_assimilate" and not self.queried_proposal:
                self.queried_proposal = True
                raw = {"action": "read_case_records", "record_ids": ["hypothesis:" + case.portfolio.current]}
            elif p["stage"] == "planner_explain_ranking" and not self.queried_explanation:
                self.queried_explanation = True
                assert p["memory_access"]["queries_remaining"] == 3
                raw = {"action": "read_case_records", "record_ids": ["judgment:last_comparison"]}
            else:
                raw = explanation_response(p) if p["stage"] == "planner_explain_ranking" else data
            if raw["action"] == "read_case_records":
                from mechcal.schemas.memory import MemoryQuery
                raw = MemoryQuery.model_validate(raw).model_dump(mode="json")
            return PlannerOutputContract(p).encode(
                raw, kind="memory_query" if raw["action"] == "read_case_records" else "decision")

    client = Client()
    agent = PlannerAgent(llm_client=client, enable_llm_decisions=True, capability_registry=REGISTRY)
    before = case.model_dump_json()
    result = agent.assimilate(case, memory_store=RunStore(tmp_path, case.case_id))
    assert len(client.calls) == 4
    assert client.calls[-1]["memory_access"]["queries_remaining"] == 2
    assert case.model_dump_json() == before
    validate_assimilation(result, case)


def test_transport_failure_during_explanation_never_commits_or_retries_proposal():
    case = prepared_case()
    data = proposal(case, [dict(name=case.portfolio.current, differential_priority=0)])

    class Client:
        calls = []

        def is_configured(self):
            return True

        def complete_json(self, **kwargs):
            p = kwargs["payload"]
            self.calls.append(p)
            if p["stage"] == "planner_assimilate":
                return deepcopy(data)
            raise TimeoutError("Offline simulated relay outage")

    client = Client()
    agent = PlannerAgent(llm_client=client, enable_llm_decisions=True, capability_registry=REGISTRY)
    before = case.model_dump_json()
    with pytest.raises(TimeoutError):
        agent.assimilate(case)
    assert len(client.calls) == 2
    assert case.model_dump_json() == before


def test_explanation_pause_and_resume_keep_completed_tools(tmp_path, monkeypatch):
    from test_judgment_commands import command_loop
    o, client = command_loop(tmp_path, monkeypatch)
    original = client.complete_json
    reject = True

    def completion(**kwargs):
        p = kwargs["payload"]
        if reject and p["stage"] == "planner_explain_ranking" and p["revision"] > 0:
            return dict(action="explain_ranking", draft_id=p["draft_id"], leading=None, boundaries=[])
        return original(**kwargs)

    monkeypatch.setattr(client, "complete_json", completion)
    with pytest.raises(PlannerCommandRejected) as error:
        o.run(smiles="CCO", user_query=NOTE, case_id="command-loop")
    assert error.value.action == "explain_ranking"
    root = tmp_path / "command-loop"
    saved = CaseRun.model_validate_json((root / "question_checkpoint.json").read_text())
    assert saved.status == "paused"
    assert len(state(saved)["pending_results"]) == 1
    tools = (root / "history/tool_events.jsonl").read_bytes()
    evidence = saved.evidence_ledger.model_dump_json()
    reject = False
    result = o.run(smiles="CCO", user_query=NOTE, case_id="command-loop", resume=True)
    assert result.status == "finalized"
    assert (root / "history/tool_events.jsonl").read_bytes() == tools
    assert result.evidence_ledger.model_dump_json() == evidence


def test_final_action_repair_reuses_explanation_of_identical_frozen_proposal():
    from test_boundary_decisions import ending
    from test_final_evidence_submission import submission_for
    from test_judgment_commands import repair_fields
    from mechcal.runtime.questions import decision_from_action, apply_final_revision

    case = prepared_case()
    original = decision_from_action(ending(case), case, "R001")
    case = replace_state(case, phase="final_revision", decision=original.model_dump(mode="json"),
                         final_audit={"status": "disabled"})
    update = proposal(case, [dict(name=case.portfolio.current, differential_priority=0)])
    expanded = materialize_judgment(case, JudgmentProposal.model_validate(update))
    final_action = ending(case)
    final_action.revision = expanded.revision
    data = dict(action="revise_final", update=update, final_action=final_action.model_dump(mode="json"),
                audit_responses={}, evidence_submission=submission_for(expanded.portfolio))
    data["evidence_submission"].reverse()  # Label binding, not a second ordinal sequence.
    data["final_action"]["continuation_assessment"] = None
    agent, client = planner(case, [data, repair_fields(
        continuation_assessment=final_action.continuation_assessment.model_dump(mode="json"))])
    before = case.model_dump_json()
    revision = agent.revise_final_arguments(case)
    assert [c["payload"]["stage"] for c in client.calls] == [
        "planner_final_revision", "planner_explain_ranking", "planner_judgment_repair"]
    assert case.model_dump_json() == before
    updated, _ = apply_final_revision(case, revision, REGISTRY)
    assert [s["label"] for s in updated.runtime["final_evidence_submission"]] == [
        h.name for h in updated.portfolio.sorted_hypotheses()[:3]]


def test_challenger_can_be_an_unranked_pool_member_without_automatic_promotion():
    case = prepared_case()
    case.runtime["evidence_decision_contracts"] = False
    update = materialize_judgment(case, JudgmentProposal.model_validate(proposal(case)))
    removed = update.portfolio.hypotheses.pop()
    draft = RankingExplanationDraft(case, update)
    response = explanation_response(draft.payload())
    response["leading"]["challenger"] = removed.name
    Draft202012Validator(draft.payload()["output_contract"]).validate(response)
    result = draft.apply(RankingExplanations.model_validate(response))
    assert result.ranking_update.leading_comparison.challenger == removed.name
    assert removed.name not in {h.name for h in result.portfolio.hypotheses}
