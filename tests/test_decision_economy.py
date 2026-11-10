"""Offline acceptance: no network, chemistry, GT or score-based decisions."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from mechcal.runtime import OrchestratorConfig
from mechcal.runtime.boundary_decisions import required_stop_routes
from mechcal.runtime.decision_economy import expand_retained, validate_plan_value
from mechcal.runtime.decision_value import validate_action_value
from mechcal.runtime.questions import apply_assimilation, public_payload, replace_state, state, validate_assimilation
from mechcal.runtime.retry import RequestValidationError
from mechcal.schemas import CaseRun
from mechcal.schemas.question_loop import QuestionAssimilation, RetainedJudgment
from test_boundary_decisions import start, complete_update
from test_decision_value import prepared, outcomes, value
from test_evidence_comparison import argument, update_for
from test_question_loop import REGISTRY


def economy_start():
    c, u = start()
    c = c.touch(runtime={**c.runtime, "decision_economy": True})
    for q in u.questions:
        q.decision_role = "material"
    u.mechanism_support_arguments = [argument()]
    return c, u


def next_update(c):
    d = complete_update(update_for(c), [h.name for h in c.portfolio.sorted_hypotheses()])
    u = QuestionAssimilation.model_validate(d)
    for q in u.questions:
        q.decision_role = "material"
    return u


def retained_update(c, u):
    u.retained_judgment = RetainedJudgment(
        hypothesis_labels=[h.name for h in c.portfolio.hypotheses],
        support_ids=[a.support_id for a in c.mechanism_support_arguments],
        question_ids=list(state(c)["questions"]),
        candidate_basis_labels=[b["label"] for b in state(c)["assimilation"]["ranking_update"]["candidate_bases"]],
    )
    u.portfolio.hypotheses = []
    u.mechanism_support_arguments = []
    u.questions = []
    u.ranking_update.candidate_bases = []
    return u


def test_reuse_is_smaller_explicit_idempotent_and_checkpoint_safe():
    c, u = economy_start()
    c = replace_state(apply_assimilation(c, u), phase="assimilate")
    full = next_update(c)
    compact = retained_update(c, full.model_copy(deep=True))
    assert len(compact.model_dump_json()) < len(full.model_dump_json())
    before = deepcopy(c.model_dump())
    validate_assimilation(compact, c)
    validate_assimilation(compact, c)
    assert c.model_dump() == before
    result = apply_assimilation(c, compact)
    assert result.portfolio == c.portfolio
    assert result.mechanism_support_arguments == c.mechanism_support_arguments
    assert result.evidence_ledger == c.evidence_ledger
    assert CaseRun.model_validate_json(result.model_dump_json()) == result
    assert state(result)["last_comparison"]["argument_changes"] == []
    public_payload(result, REGISTRY, stage="planner_question_act")


@pytest.mark.parametrize("fault", ["unknown", "duplicate", "disabled"])
def test_reuse_rejects_bad_ids_without_partial_mutation(fault):
    c, u = economy_start()
    c = replace_state(apply_assimilation(c, u), phase="assimilate")
    u = retained_update(c, next_update(c))
    if fault == "unknown":
        u.retained_judgment.support_ids = ["foreign"]
    elif fault == "duplicate":
        u.retained_judgment.support_ids *= 2
    else:
        c.runtime["decision_economy"] = False
    before = deepcopy(u.model_dump())
    with pytest.raises(RequestValidationError):
        expand_retained(c, u)
    assert u.model_dump() == before


@pytest.mark.parametrize("change", ["revised", "withdrawn", "added"])
def test_program_diff_requires_no_model_receipts_but_keeps_exact_audit(change):
    c, u = economy_start()
    c = replace_state(apply_assimilation(c, u), phase="assimilate")
    u = next_update(c)
    if change == "revised":
        u.mechanism_support_arguments[0].boundary = "Updated bounded interpretation."
    elif change == "withdrawn":
        u.mechanism_support_arguments = []
    else:
        u.mechanism_support_arguments.append(argument("new"))
    u.argument_changes = []
    result = apply_assimilation(c, u)
    edits = state(result)["last_comparison"]["argument_changes"]
    assert edits[0]["provenance"] == "program_diff"
    assert edits[0]["disposition"] == change
    assert result.evidence_ledger == c.evidence_ledger
    public_payload(result, REGISTRY, stage="planner_question_act")


def test_program_diff_does_not_hide_foreign_refs_or_retargeting():
    c, u = economy_start()
    c = replace_state(apply_assimilation(c, u), phase="assimilate")
    u = next_update(c)
    u.mechanism_support_arguments[0].target_label = "ICT_TICT_CT"
    with pytest.raises(RequestValidationError, match="bound to their mechanism"):
        validate_assimilation(u, c)
    u = next_update(c)
    u.mechanism_support_arguments[0].observation_refs = ["foreign:ref"]
    with pytest.raises(RequestValidationError):
        validate_assimilation(u, c)


def meaningful_plan():
    c, a = prepared()
    c.runtime["decision_economy"] = True
    p = a.investigation_plans[0]
    p.possible_outcomes[0].affected_labels = [p.candidate_labels[0]]
    p.possible_outcomes[0].judgment_change = "weaken"
    return c, a, p


def test_support_weakening_is_useful_without_forcing_rank_change():
    c, a, p = meaningful_plan()
    before = deepcopy(c.model_dump())
    validate_action_value(c, a, REGISTRY)
    assert c.model_dump() == before


@pytest.mark.parametrize("fault", ["missing_label", "foreign_label", "wrong_direction", "no_change"])
def test_vague_or_irrelevant_benefit_rejected(fault):
    c, a, p = meaningful_plan()
    o = p.possible_outcomes[0]
    if fault == "missing_label":
        o.affected_labels = []
    elif fault == "foreign_label":
        o.affected_labels = ["ESIPT_PT"]
    elif fault == "wrong_direction":
        o.judgment_change = "promote"
    else:
        p.possible_outcomes = outcomes("no_change")
    with pytest.raises(ValueError):
        validate_action_value(c, a, REGISTRY)


def test_preparation_needs_terminal_decision_not_more_preparation():
    c, a, p = meaningful_plan()
    good = deepcopy(p.possible_outcomes)
    p.purpose = "preparation"
    p.possible_outcomes = outcomes("preparation", "no_change")
    p.decision_value = value(downstream_capability_id="macro.screen_rotor_torsion_topology",
                             downstream_question="Would flexibility change the bounded support?")
    with pytest.raises(ValueError):
        validate_action_value(c, a, REGISTRY)
    p.decision_value.downstream_outcomes = outcomes("preparation")
    with pytest.raises(ValueError):
        validate_plan_value(p)
    p.decision_value.downstream_outcomes = good
    validate_action_value(c, a, REGISTRY)


def test_limitations_remain_unanswered_without_mandatory_route_enumeration():
    c, u = economy_start()
    u.questions[0].decision_role = "limitation"
    c = apply_assimilation(c, u)
    q = state(c)["questions"][u.questions[0].question_id]
    q["acceptable_capability_ids"] = ["macro.screen_hbond_network"]
    assert q["answer_state"] != "answered"
    assert "macro.screen_hbond_network" not in required_stop_routes(c)
    q["decision_role"] = "material"
    assert "macro.screen_hbond_network" in required_stop_routes(c)


def test_mode_is_opt_in_and_requires_boundary_contracts(tmp_path):
    config = OrchestratorConfig(run_base_dir=tmp_path)
    assert not config.decision_economy
    with pytest.raises(ValueError):
        replace(config, decision_economy=True)
    assert "decision_economy" in json.dumps(config.__dict__, default=str)


def test_repair_uses_expanded_candidate_not_sparse_indices():
    from mechcal.runtime.retry import SchemaRetryRunner
    from mechcal.runtime.field_repair import FieldRepairProducer
    c, u = economy_start()
    c = replace_state(apply_assimilation(c, u), phase="assimilate")
    sparse = retained_update(c, next_update(c)).model_dump(mode="json")
    sparse["ranking_update"]["candidate_bases"] = []
    def validate(item):
        expand_retained(c, item)
        if item.portfolio.hypotheses[0].rationale != "Reviewed statement.":
            raise RequestValidationError([dict(path="portfolio.hypotheses.0.rationale",
                repair_paths=[["portfolio", "hypotheses", 0, "rationale"]], message="Fixture review.")])
    def repair(payload, feedback):
        assert payload["rejected_response"]["retained_judgment"] is None
        assert len(payload["rejected_response"]["portfolio"]["hypotheses"]) == len(c.portfolio.hypotheses)
        return {"repairs": [{"path": payload["allowed_paths"][0], "value": "Reviewed statement."}]}
    producer = FieldRepairProducer(lambda _: sparse, repair, response_model=QuestionAssimilation)
    result = SchemaRetryRunner(QuestionAssimilation, validators=[validate],
                               repair_context=True, normalized_feedback=True).run(producer)
    assert result.portfolio.hypotheses[0].rationale == "Reviewed statement."
    assert result.portfolio.hypotheses[1:] == c.portfolio.hypotheses[1:]


def test_reuse_preserves_tied_candidate_slots_and_rejects_duplicate_updates():
    c, u = economy_start()
    c = replace_state(apply_assimilation(c, u), phase="assimilate")
    for h in c.portfolio.hypotheses:
        h.differential_priority = .5
    u = next_update(c)
    u.portfolio = c.portfolio.model_copy(deep=True)
    original_order = [h.name for h in c.portfolio.sorted_hypotheses()]
    u.retained_judgment = RetainedJudgment(hypothesis_labels=original_order[:-1])
    u.portfolio.hypotheses = u.portfolio.hypotheses[-1:]
    duplicate = u.model_copy(deep=True)
    duplicate.portfolio.hypotheses *= 2
    with pytest.raises(RequestValidationError, match="distinct IDs"):
        expand_retained(c, duplicate)
    expand_retained(c, u)
    assert [h.name for h in u.portfolio.sorted_hypotheses()] == original_order


@pytest.mark.parametrize("audit", [False, True])
def test_full_loop_sparse_updates_final_audit_and_resume(tmp_path, audit, monkeypatch):
    from mechcal.runtime import MechCALOrchestrator
    from mechcal.schemas.question_loop import ResultReview
    from test_evidence_comparison import ComparisonClient
    from test_question_loop import loop, NOTE, TOOL

    class Client(ComparisonClient):
        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            data = super().complete_json(payload=payload, system_prompt=system_prompt, schema_feedback=schema_feedback)
            if data.get("action") == "describe_tools":
                return data
            stage = payload["stage"]
            if stage in {"planner_assimilate", "planner_final_revision"}:
                u = data["update"] if stage == "planner_final_revision" else data
                complete_update(u, payload["comparison_context"]["current_order"])
                for q in u["questions"]:
                    q["decision_role"] = "material"
                if payload["comparison_context"]["initial_judgment"] is not None:
                    labels = [h["name"] for h in payload["portfolio"]["hypotheses"]]
                    u["retained_judgment"] = {"hypothesis_labels": labels}
                    u["portfolio"]["hypotheses"] = []
            actions = [data["final_action"]] if stage == "planner_final_revision" else [data]
            for a in actions:
                for p in a.get("investigation_plans", []):
                    p["decision_value"] = value().model_dump(mode="json")
                    os = outcomes("support", "no_change")
                    os[0].affected_labels = [p["candidate_labels"][0]]
                    os[0].judgment_change = "weaken"
                    p["possible_outcomes"] = [o.model_dump(mode="json") for o in os]
                if a.get("action") in {"finalize", "stop"}:
                    a["continuation_assessment"] = dict(scope="bounded_ranking", remaining_uncertainty=NOTE,
                        decision_reason=NOTE, routes=[dict(capability_id=TOOL, disposition="not_useful", reason=NOTE)])
            return data

    o, _ = loop(tmp_path, unlimited=True, discover=True)
    client = Client(1, True)
    o.deps.planner.llm_client = client
    monkeypatch.setattr(o.deps.photophysics_arbiter, "review_results", lambda case, **kw: ResultReview(
        interpretations=[dict(result_id=r["result_id"], question_ids=r["question_ids"],
                              delivered=NOTE, missing=NOTE, boundary=NOTE, evidence_refs=r["evidence_refs"])
                         for r in state(case).get("pending_results", [])]))
    o = MechCALOrchestrator(replace(o.config, comparative_evidence=True, decision_value_checks=True,
        evidence_decision_contracts=True, decision_economy=True, enable_support_auditor=audit), o.deps)
    result = o.run(smiles="CCO", user_query=NOTE, case_id="economy-loop")
    assert result.status == "finalized", result.failure
    assert len(state(result)["results"]) == 1
    assert result.runtime["decision_economy"]
    assert all("hidden_reference" not in p and "rubric" not in p for p in client.calls)
    before = len(client.calls)
    assert o.run(smiles="CCO", user_query=NOTE, case_id="economy-loop", resume=True) == result
    assert len(client.calls) == before
