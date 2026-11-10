"""Offline contracts and scripted workflow tests, not performance evidence."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from mechcal.runtime import MechCALOrchestrator, OrchestratorConfig
from mechcal.runtime.evidence_comparison import comparison_context, validate_plans
from mechcal.runtime.questions import apply_assimilation, public_payload, replace_state, state, validate_assimilation
from mechcal.schemas import CaseRun, MechanismSupportArgument
from mechcal.schemas.evidence_comparison import InvestigationPlan
from mechcal.schemas.question_loop import QuestionAssimilation
from test_question_loop import (
    NOTE, REGISTRY, QuestionClient, action, assimilation, initial_case, loop,
)


def argument(sid="structural", label="RIM_RIR_RIV"):
    return MechanismSupportArgument(
        support_id=sid, target_label=label, support_level="weak",
        finding="Synthetic structural observation for contract testing only.",
        warrant="Synthetic conditional physical connection, not a scientific result.",
        boundary=NOTE, observation_refs=["input:smiles"],
    )


def initial():
    c = initial_case()
    return c.touch(runtime={**c.runtime, "comparative_evidence": True})


def update_for(c):
    payload = public_payload(c, REGISTRY, stage="planner_assimilate")
    data = assimilation(payload)
    data["mechanism_support_arguments"] = [a.model_dump(mode="json") for a in c.mechanism_support_arguments]
    data["ranking_update"] = dict(
        basis="initial" if "initial_judgment" not in state(c) else "unchanged",
        explanation=NOTE, alternatives_considered=["RIM_RIR_RIV", "ESIPT_PT"],
        evidence_refs=["input:smiles"],
    )
    for d in data["dispositions"]:
        result = next(r for r in payload["pending_results"] if r["result_id"] == d["result_id"])
        labels = (result.get("investigation_plan") or {}).get("candidate_labels", ["RIM_RIR_RIV"])
        d["mechanism_effects"] = [dict(label=label, effect="unavailable" if d["outcome"] == "no_result"
                                      else "non_discriminating", reason=NOTE, evidence_refs=d["evidence_refs"])
                                 for label in labels]
    return data


def ready():
    c = initial()
    d = update_for(c)
    d["mechanism_support_arguments"] = [argument().model_dump(mode="json")]
    return apply_assimilation(c, QuestionAssimilation.model_validate(d))


def test_snapshot_is_immutable_and_current_argument_can_be_corrected():
    c = replace_state(ready(), phase="assimilate")
    before = deepcopy(state(c)["initial_judgment"])
    data = update_for(c)
    data["mechanism_support_arguments"][0]["warrant"] = "Corrected conditional explanation."
    data["argument_changes"] = [dict(support_id="structural", disposition="revised", reason="Correct the link.")]
    after = apply_assimilation(c, QuestionAssimilation.model_validate(data))
    assert state(after)["initial_judgment"] == before
    assert state(after)["argument_history"][0]["previous_argument"]["warrant"] == argument().warrant
    assert after.mechanism_support_arguments[0].warrant == "Corrected conditional explanation."
    assert after.mechanism_support_arguments[0].audit_status == "not_reviewed"
    assert c.mechanism_support_arguments[0].warrant == argument().warrant
    assert CaseRun.model_validate_json(after.model_dump_json()) == after


@pytest.mark.parametrize("mutation", ["delete", "warrant", "refs", "label"])
def test_no_silent_loss_or_mutation_of_arguments(mutation):
    c = replace_state(ready(), phase="assimilate")
    d = update_for(c)
    if mutation == "delete":
        d["mechanism_support_arguments"] = []
    else:
        key = {"warrant": "warrant", "refs": "observation_refs", "label": "target_label"}[mutation]
        d["mechanism_support_arguments"][0][key] = {
            "warrant": "Numerical description replaced the explanation.",
            "refs": ["input:smiles", "input:smiles"], "label": "ICT_TICT_CT",
        }[mutation]
    with pytest.raises(ValueError, match="Account once"):
        validate_assimilation(QuestionAssimilation.model_validate(d), c)


@pytest.mark.parametrize("disposition", ["withdrawn", "superseded"])
def test_legitimate_withdrawal_or_replacement_does_not_export_old_argument(disposition):
    c = replace_state(ready(), phase="assimilate")
    d = update_for(c)
    d["mechanism_support_arguments"] = [] if disposition == "withdrawn" else [argument("new").model_dump()]
    d["argument_changes"] = [dict(support_id="structural", disposition=disposition, reason="Corrected reasoning.",
                                  replacement_ids=[] if disposition == "withdrawn" else ["new"])]
    updated = apply_assimilation(c, QuestionAssimilation.model_validate(d))
    assert "structural" not in [a.support_id for a in updated.mechanism_support_arguments]
    assert state(updated)["argument_history"][0]["previous_argument"]["support_id"] == "structural"
    assert len(updated.evidence_ledger.items) == len(c.evidence_ledger.items)


def test_revised_id_cannot_change_target_and_superseded_must_exist():
    c = replace_state(ready(), phase="assimilate")
    d = update_for(c)
    d["mechanism_support_arguments"][0]["target_label"] = "ICT_TICT_CT"
    d["argument_changes"] = [dict(support_id="structural", disposition="revised", reason=NOTE)]
    with pytest.raises(ValueError, match="target mechanism"):
        validate_assimilation(QuestionAssimilation.model_validate(d), c)
    d["mechanism_support_arguments"] = []
    d["argument_changes"] = [dict(support_id="structural", disposition="superseded", reason=NOTE,
                                  replacement_ids=["foreign"])]
    with pytest.raises(ValueError, match="Replacement IDs"):
        validate_assimilation(QuestionAssimilation.model_validate(d), c)


def test_ranking_change_requires_comparison_but_not_direct_counterevidence():
    c = replace_state(ready(), phase="assimilate")
    d = update_for(c)
    d["portfolio"]["hypotheses"][2]["differential_priority"] = .8
    d["portfolio"].update(current="ESIPT_PT", runner_up="RIM_RIR_RIV")
    with pytest.raises(ValueError, match="Order or candidate membership changed"):
        validate_assimilation(QuestionAssimilation.model_validate(d), c)
    d["ranking_update"].update(basis="reinterpretation", explanation="Reconsider the relative structural hypotheses.")
    result = apply_assimilation(c, QuestionAssimilation.model_validate(d))
    assert result.portfolio.current == "ESIPT_PT"
    assert state(result)["initial_judgment"]["portfolio"]["current"] == "RIM_RIR_RIV"


@pytest.mark.parametrize("field,value", [("basis", "initial"), ("evidence_refs", ["foreign:ref"]),
                                        ("alternatives_considered", ["invented"]), ("explanation", " ")])
def test_invalid_comparison_rejected(field, value):
    c = replace_state(ready(), phase="assimilate")
    d = update_for(c)
    d["ranking_update"][field] = value
    with pytest.raises(ValueError):
        validate_assimilation(QuestionAssimilation.model_validate(d), c)


def test_failed_output_cannot_be_negative_evidence():
    c = ready()
    r = dict(result_id="X", question_ids=["Q1"], status="failed", evidence_refs=[], structured_results={})
    c = replace_state(c, phase="assimilate", results=[r], pending_results=[r])
    d = update_for(c)
    d["dispositions"][0]["mechanism_effects"][0]["effect"] = "weakens"
    with pytest.raises(ValueError, match="delivered scientific observation"):
        validate_assimilation(QuestionAssimilation.model_validate(d), c)
    d["dispositions"][0]["mechanism_effects"][0]["effect"] = "unavailable"
    result = apply_assimilation(c, QuestionAssimilation.model_validate(d))
    assert result.portfolio == c.portfolio


@pytest.mark.parametrize("replacement_ids", [[], ["structural"]])
def test_redundant_revision_receipt_is_recorded_without_inventing_a_change(replacement_ids):
    c = replace_state(ready(), phase="assimilate")
    d = update_for(c)
    d["argument_changes"] = [dict(support_id="structural", disposition="revised", reason="Reviewed and retained.",
                                  replacement_ids=replacement_ids)]
    result = apply_assimilation(c, QuestionAssimilation.model_validate(d))
    assert result.mechanism_support_arguments == c.mechanism_support_arguments
    assert state(result)["argument_history"][-1]["observed_change"] is False
    assert result.portfolio == c.portfolio
    assert result.evidence_ledger == c.evidence_ledger


def test_self_replacement_for_actual_revision_is_valid():
    c = replace_state(ready(), phase="assimilate")
    d = update_for(c)
    d["mechanism_support_arguments"][0]["boundary"] = "Corrected limitation."
    d["argument_changes"] = [dict(support_id="structural", disposition="revised", reason="Correct limitation.",
                                  replacement_ids=["structural"])]
    result = apply_assimilation(c, QuestionAssimilation.model_validate(d))
    assert state(result)["argument_history"][-1]["observed_change"] is True


def test_tail_candidate_addition_is_not_a_change_in_existing_order():
    c = replace_state(ready(), phase="assimilate")
    d = update_for(c)
    d["portfolio"]["hypotheses"].append(dict(name="HOST_GUEST_INTERACTION", differential_priority=.01, rationale=NOTE))
    d["ranking_update"]["alternatives_considered"].append("HOST_GUEST_INTERACTION")
    result = apply_assimilation(c, QuestionAssimilation.model_validate(d))
    assert state(result)["last_comparison"]["added_candidates"] == ["HOST_GUEST_INTERACTION"]
    assert [h.name for h in result.portfolio.sorted_hypotheses()[:3]] == [h.name for h in c.portfolio.sorted_hypotheses()[:3]]
    d["ranking_update"]["alternatives_considered"].remove("HOST_GUEST_INTERACTION")
    with pytest.raises(ValueError, match="Acknowledge"):
        validate_assimilation(QuestionAssimilation.model_validate(d), c)


@pytest.mark.parametrize("fault", ["duplicate", "unknown", "wrong_replacement", "fake_withdrawal"])
def test_compatible_receipts_still_reject_invalid_links(fault):
    c = replace_state(ready(), phase="assimilate")
    d = update_for(c)
    row = dict(support_id="structural", disposition="revised", reason=NOTE)
    d["argument_changes"] = [row]
    if fault == "duplicate":
        d["argument_changes"].append(dict(row))
    elif fault == "unknown":
        row["support_id"] = "foreign"
    elif fault == "wrong_replacement":
        row["replacement_ids"] = ["foreign"]
    else:
        row["disposition"] = "withdrawn"
    with pytest.raises(ValueError):
        validate_assimilation(QuestionAssimilation.model_validate(d), c)




def plan(dispatch_id="R001:test", purpose="discrimination"):
    return InvestigationPlan(
        dispatch_id=dispatch_id, purpose=purpose, candidate_labels=["RIM_RIR_RIV", "ICT_TICT_CT"],
        decision_question=NOTE, usefulness=NOTE,
        possible_outcomes=[dict(observation="Synthetic observation A", judgment_effect=NOTE),
                           dict(observation="Synthetic observation B", judgment_effect=NOTE)],
    )


@pytest.mark.parametrize("purpose", ["discrimination", "preparation", "exploration", "recovery"])
def test_plans_accept_comparison_and_prerequisites_without_forcing_rankings(purpose):
    c = ready()
    a = action(c.case_id, state(c)["revision"], dispatch=True)
    a.investigation_plans = [plan(purpose=purpose)]
    validate_plans(c, a)
    assert a.dispatch_requests[0].capability_id == "macro.screen_rotor_torsion_topology"


@pytest.mark.parametrize("fault", ["missing", "duplicate", "foreign", "single_outcome", "single_label"])
def test_plan_link_and_discrimination_requirements(fault):
    c = ready()
    a = action(c.case_id, state(c)["revision"], dispatch=True)
    p = plan()
    a.investigation_plans = [p]
    if fault == "missing":
        a.investigation_plans = []
    elif fault == "duplicate":
        a.investigation_plans.append(p)
    elif fault == "foreign":
        p.dispatch_id = "another:request"
    elif fault == "single_outcome":
        p.possible_outcomes = p.possible_outcomes[:1]
    else:
        p.candidate_labels = p.candidate_labels[:1]
    with pytest.raises(ValueError):
        validate_plans(c, a)


def test_feature_off_keeps_old_contract_and_cases_are_isolated(tmp_path):
    c = initial_case()
    apply_assimilation(c, QuestionAssimilation.model_validate(assimilation(dict(case_id=c.case_id, revision=0))))
    assert comparison_context(c) == {"enabled": False}
    c = ready()
    fresh = initial().touch(case_id="another")
    assert comparison_context(fresh)["initial_judgment"] is None
    assert comparison_context(c)["initial_judgment"]["case_id"] != fresh.case_id
    with pytest.raises(ValueError, match="requires question_loop"):
        OrchestratorConfig(run_base_dir=tmp_path, comparative_evidence=True)


class ComparisonClient(QuestionClient):
    def complete_json(self, *, payload, system_prompt, schema_feedback=None):
        data = super().complete_json(payload=payload, system_prompt=system_prompt, schema_feedback=schema_feedback)
        if data.get("action") == "describe_tools":
            return data
        stage = payload["stage"]
        if stage in {"planner_assimilate", "planner_final_revision"}:
            update = data["update"] if stage == "planner_final_revision" else data
            update["ranking_update"] = dict(
                basis="initial" if payload["comparison_context"]["initial_judgment"] is None else "unchanged",
                explanation=NOTE, alternatives_considered=["RIM_RIR_RIV", "ICT_TICT_CT"], evidence_refs=["input:smiles"],
            )
            for d in update["dispositions"]:
                d["mechanism_effects"] = [dict(label=label, effect="non_discriminating", reason=NOTE,
                                              evidence_refs=d["evidence_refs"])
                                         for label in ["RIM_RIR_RIV", "ICT_TICT_CT"]]
        elif stage == "planner_question_act":
            data["investigation_plans"] = [plan(r["dispatch_id"]).model_dump(mode="json")
                                           for r in data["dispatch_requests"]]
        return data


@pytest.mark.parametrize("audit", [False, True])
def test_end_to_end_persistence_final_revision_and_resume(tmp_path, audit, monkeypatch):
    from mechcal.schemas.question_loop import ResultReview

    o, _ = loop(tmp_path, discover=True)
    config = replace(o.config, comparative_evidence=True, enable_support_auditor=audit)
    client = ComparisonClient(1, True)
    o.deps.planner.llm_client = client
    o = MechCALOrchestrator(config, o.deps)
    monkeypatch.setattr(o.deps.photophysics_arbiter, "review_results", lambda case, **kw: ResultReview(
        interpretations=[dict(result_id=r["result_id"], question_ids=r["question_ids"],
                              delivered=NOTE, missing=NOTE, boundary=NOTE, evidence_refs=r["evidence_refs"])
                         for r in state(case).get("pending_results", [])]))
    result = o.run(smiles="CCO", user_query=NOTE, case_id="comparison")
    assert result.status == "finalized"
    assert result.round_ids == ["R001", "R002"]
    assert result.runtime["worker_call_counts"]["macro"] == 1
    s = state(result)
    assert s["initial_judgment"]["revision"] == 1
    assert s["results"][0]["investigation_plan"]["purpose"] == "discrimination"
    if audit:
        assert s["final_revision_complete"]
        assert (tmp_path / "comparison/rounds/R002/planner_final_revision.json").exists()
    initial_bytes = (tmp_path / "comparison/initial_judgment.json").read_bytes()
    before = len(client.calls)
    resumed = o.run(smiles="CCO", user_query=NOTE, case_id="comparison", resume=True)
    assert resumed == result and len(client.calls) == before
    assert (tmp_path / "comparison/initial_judgment.json").read_bytes() == initial_bytes
    assert json.loads(initial_bytes)["portfolio"]["current"] == "RIM_RIR_RIV"
    comparison_updates = sorted((tmp_path / "comparison/questions").glob("comparison_*.json"))
    assert len(comparison_updates) == s["revision"]
    assert all("rubric" not in p and "hidden_reference" not in p for p in client.calls)
    assert not any("comparison" in e.evidence_id for e in result.evidence_ledger.items)
