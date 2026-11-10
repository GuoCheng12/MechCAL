"""Offline contracts and context checks; scripted replies are not model results."""

import json

import pytest

from mechcal.agents.evidence_context import calculation_history, current_support_arguments
from mechcal.agents.evidence_references import EvidenceReferences
from mechcal.agents.mechanism_agenda import MechanismAgendaAgent
from mechcal.agents.photophysics_arbiter import (
    DisabledPhotophysicsWebSearch,
    PhotophysicsArbiterAgent,
)
from mechcal.agents.planner import PlannerAgent, _compact_planner_payload
from mechcal.capabilities import default_capability_registry
from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime import MechCALOrchestrator, OrchestratorConfig
from mechcal.runtime.orchestrator import _stateless_planner_case_view
from mechcal.schemas import (
    EvidenceLedger,
    HypothesisEntry,
    HypothesisPortfolio,
    MechanismSupportArgument,
    PlannerDecision,
)
from test_planner_owned_loop import ScriptedPlannerClient, normalize, reply, run_scripted
from test_runtime import _case_run_for_gate, _evidence_unit


class CorrectingClient:
    def __init__(self, malformed, *, recover=True):
        self.malformed = malformed
        self.recover = recover
        self.feedback = []

    def is_configured(self):
        return True

    def complete_json(self, *, payload, system_prompt, schema_feedback=None):
        self.feedback.append(schema_feedback)
        if schema_feedback and self.recover:
            return reply()
        return self.malformed


@pytest.mark.parametrize("problem", ["missing", "blank", "unknown_reference"])
def test_three_questions_are_required_and_reference_checked(problem):
    malformed = reply()
    if problem == "missing":
        del malformed["decision_assessment"]
    elif problem == "blank":
        malformed["decision_assessment"]["next_useful_step"] = "   "
    else:
        malformed["decision_assessment"]["evidence_refs"] = ["nonexistent_observation"]
    client = CorrectingClient(malformed)
    agent = PlannerAgent(llm_client=client, enable_llm_decisions=True)
    decision = agent.plan_initial(_case_run_for_gate(), round_id="R001")
    assert decision.action == "finalize"
    assert decision.decision_assessment is not None
    assert len(client.feedback) == 2
    assert client.feedback[0] is None
    assert client.feedback[1]


def test_missing_assessment_is_not_filled_by_runtime_after_retries():
    malformed = reply()
    del malformed["decision_assessment"]
    client = CorrectingClient(malformed, recover=False)
    decision = PlannerAgent(llm_client=client, enable_llm_decisions=True).plan_initial(
        _case_run_for_gate(), round_id="R001",
    )
    assert decision.action == "stop"
    assert decision.failure.kind == "llm_schema_invalid"
    assert decision.decision_assessment is None
    assert len(client.feedback) == 3


def test_old_decisions_remain_readable_without_assessment():
    legacy = reply()
    del legacy["decision_assessment"]
    assert PlannerDecision.model_validate(normalize(legacy)).decision_assessment is None


def history_case():
    case = _case_run_for_gate()
    evidence = [
        _evidence_unit(
            evidence_id=f"E{i}", round_id=f"R{i + 1:03d}",
            source_report_id=f"report{i}", agent_name="microscopic",
            family="charge_localization", summary=f"Fixture observation {i}.",
            metrics={"computed_observables": {
                "requested_parameters": {"angles": [-90, 0, 90, 180], "s1_nstates": 8},
                "effective_parameters": {"angles": [-90, 0, 90, 180], "s1_nstates": 8},
                "available_descriptors": ["natural_orbitals_no"],
                "missing_descriptors": ["hole_electron_separation"],
                "route_records": [{
                    "dihedral_atoms": [7, 8, 9, 10], "target_angle_deg": -90,
                }],
            }},
        )
        for i in range(15)
    ]
    hypotheses = [
        HypothesisEntry(name=label, differential_priority=0.9 - i * 0.05,
                        evidence_refs=[f"E{i}"])
        for i, label in enumerate(MECHANISM_POOL)
    ]
    arguments = [
        MechanismSupportArgument(
            support_id=f"support{i}", target_label=label, support_level="partial",
            finding="Fixture observation.", warrant="Fixture comparison.",
            boundary="Not an experimental result.", observation_refs=[f"E{i}"],
        )
        for i, label in enumerate(MECHANISM_POOL)
    ]
    return case.touch(
        portfolio=HypothesisPortfolio(hypotheses=hypotheses, current=MECHANISM_POOL[0]),
        evidence_ledger=EvidenceLedger(case_id=case.case_id, items=evidence),
        mechanism_support_arguments=arguments,
    )


class CapturingReviewer:
    def __init__(self):
        self.payload = None

    def is_configured(self):
        return True

    def complete_json(self, *, payload, system_prompt, schema_feedback=None):
        self.payload = payload
        if payload.get("task") == "agenda_coverage_review":
            return {
                "coverage_summary": "Several interpretations remain possible.",
                "decision_challenges": [
                    "Would the proposed repeat add anything beyond E0?",
                ],
                "agenda_items": [], "recommended_next_routes": [],
            }
        return {
            "support_argument_audits": [
                {"support_id": sid, "audit_status": "boundary_added",
                 "audit_notes": ["Offline fixture review."]}
                for sid in payload["allowed_support_ids"]
            ],
        }


def test_auditor_keeps_leading_claims_and_their_early_evidence():
    case = history_case()
    client = CapturingReviewer()
    review = PhotophysicsArbiterAgent(
        llm_client=client, web_search=DisabledPhotophysicsWebSearch(),
    ).review(case)
    payload = client.payload
    assert payload["allowed_support_ids"][:3] == ["support0", "support1", "support2"]
    shown = {row["evidence_id"]: row for row in payload["evidence_ledger"]}
    assert {"E0", "E1", "E2", "E14"} <= shown.keys()
    assert all(ref in shown for arg in payload["mechanism_support_arguments"]
               for ref in arg["observation_refs"])
    assert len(payload["evidence_index"]) == 15
    assert shown["E0"]["metrics"]["computed_observables"]["missing_descriptors"] == [
        "hole_electron_separation",
    ]
    assert {audit.support_id for audit in review.support_argument_audits} >= {
        "support0", "support1", "support2",
    }


def test_latest_argument_per_label_replaces_stale_one():
    case = history_case()
    new = case.mechanism_support_arguments[0].model_copy(update={"support_id": "latest"})
    case = case.touch(mechanism_support_arguments=[*case.mechanism_support_arguments, new])
    selected = current_support_arguments(case)
    assert selected[0].support_id == "latest"
    assert "support0" not in {item.support_id for item in selected}


def test_auditor_cannot_ground_a_claim_in_unseen_index_only_evidence():
    class IndexOnlyReviewer(CapturingReviewer):
        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            assert "E7" in {row["evidence_id"] for row in payload["evidence_index"]}
            assert "E7" not in payload["compact_evidence_ids"]
            return {"hypothesis_cards": [{
                "hypothesis_id": "unseen", "mechanism": "RIM_RIR_RIV",
                "status": "proxy_supported", "support_evidence_refs": ["E7"],
                "reasoning_summary": "Scripted unsupported use of an unseen detail.",
            }]}

    with pytest.raises(RuntimeError, match="supplied detailed observations"):
        PhotophysicsArbiterAgent(
            llm_client=IndexOnlyReviewer(), web_search=DisabledPhotophysicsWebSearch(),
            max_attempts=1,
        ).review(history_case())


def test_stateless_planner_keeps_all_current_arguments_and_early_observations():
    case = _stateless_planner_case_view(history_case())
    payload = _compact_planner_payload(
        case_run=case, round_id="R016", stage="planner_update", context={},
        capability_registry=default_capability_registry(),
    )
    assert {row["target_label"] for row in payload["current_support_arguments"]} == set(MECHANISM_POOL)
    shown = {row["evidence_id"] for row in payload["relevant_evidence"] + payload["new_round_evidence"]}
    assert {"E0", "E1", "E2", "E14"} <= shown
    assert payload["mechanism_pool"] == list(MECHANISM_POOL)


def test_reviewer_questions_and_old_conditions_reach_planner_without_auto_routes():
    case = history_case()
    client = CapturingReviewer()
    memo = MechanismAgendaAgent(llm_client=client).review_coverage(case, round_id="R016")
    assert memo.recommended_next_routes == []
    assert len(client.payload["current_evidence_ledger_summary"]) == 15
    history = client.payload["calculation_history"]
    assert history[0]["requested_parameters"]["angles"] == [-90, 0, 90, 180]
    assert history[0]["sampled_conditions"][0]["dihedral_atoms"] == [7, 8, 9, 10]
    case = case.touch(mechanism_agenda_coverage=memo)
    payload = _compact_planner_payload(
        case_run=case, round_id="R016", stage="planner_update", context={},
        capability_registry=default_capability_registry(),
    )
    assert payload["agenda_coverage_memo"]["decision_challenges"] == memo.decision_challenges


def test_coverage_reviewer_runs_before_each_decision_when_enabled(tmp_path):
    for enabled in (True, False):
        orchestrator = MechCALOrchestrator(OrchestratorConfig(
            run_base_dir=tmp_path, enable_agenda_coverage_reviewer=enabled,
        ))
        for round_index in (1, 2, 3, 4, 8):
            assert orchestrator._should_review_agenda_coverage(
                _case_run_for_gate(), round_index=round_index,
            ) is enabled


def test_history_and_assessment_persist_through_real_local_dispatches(tmp_path):
    client = ScriptedPlannerClient(finish_round=3)
    result = run_scripted(tmp_path, client, max_rounds=4)
    assert result.status == "finalized"
    assert result.runtime["decision_policy"] == "planner_assessment_loop_v6"
    history = calculation_history(result)
    assert len(history) == 2
    assert history[0]["round_id"] == "R001"
    assert history[0]["capability_id"] == "macro.screen_rotor_torsion_topology"
    assert history[0]["status"] == "success"
    assert history[0]["evidence_refs"]
    assert history[0]["artifact_refs"] is not None
    references = EvidenceReferences.from_ids([e.evidence_id for e in result.evidence_ledger.items])
    assert client.payloads[2]["calculation_history"] == references.encode(history)
    assert client.payloads[0]["previous_decision_assessment"] is None
    assert client.payloads[2]["previous_decision_assessment"]["round_id"] == "R002"
    saved = json.loads((tmp_path / "plan-a/rounds/R002/planner_decision.json").read_text())
    assert saved["decision_assessment"]["next_useful_step"]
