"""Plan A control-flow checks with scripted model replies, not model evaluations."""

import json

import pytest

from mechcal.agents.planner import (
    PlannerAgent,
    _compact_planner_payload,
    _normalize_planner_llm_response,
)
from mechcal.capabilities import default_capability_registry
from mechcal.runtime import MechCALOrchestrator, OrchestratorConfig
from mechcal.runtime.gates import DeterministicGate, GatePolicy
from mechcal.schemas import ArtifactRecord, PlannerDecision
from test_runtime import _case_run_for_gate, _fake_llm_deps, _fixture_planner_assessment


def reply(action="finalize", **extra):
    return {
        "decision_assessment": _fixture_planner_assessment(),
        "action": action,
        "portfolio": {
            "current": "RIM_RIR_RIV",
            "hypotheses": [
                {
                    "name": label,
                    "differential_priority": priority,
                    "rationale": "Offline control-flow fixture, not a scientific inference.",
                }
                for label, priority in [
                    ("RIM_RIR_RIV", 0.6),
                    ("ICT_TICT_CT", 0.3),
                    ("ESIPT_PT", 0.1),
                ]
            ],
        },
        "final_answer_draft": "Bounded fixture answer with unresolved mechanisms."
        if action == "finalize"
        else None,
        **extra,
    }


def normalize(response, **extra):
    return _normalize_planner_llm_response(
        response,
        round_id="R002",
        existing_portfolio={},
        previous_portfolio_hypotheses=[],
        allowed_evidence_ids=[],
        capability_registry=default_capability_registry(),
        **extra,
    )


@pytest.mark.parametrize("action", ["dispatch", "finalize", "stop"])
def test_coverage_cannot_change_model_action_or_insert_routes(action):
    selected = ["macro.screen_rotor_torsion_topology"] if action == "dispatch" else []
    context = {
        "artifact_available_capability_ids": ["microscopic.run_baseline_bundle"],
        "agenda_coverage_memo": {
            "agenda_items": [
                {
                    "urgency": "high",
                    "coverage_status": "under_screened",
                    "suggested_route": "microscopic.run_baseline_bundle",
                }
            ]
        },
    }
    result = normalize(reply(action, selected_capability_ids=selected), prompt_payload=context)
    assert result["action"] == action
    assert [r["capability_id"] for r in result["dispatch_requests"]] == selected


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"action": "typo"}, "action"),
        ({"final_answer_draft": None}, "final_answer_draft"),
        ({"round_id": "R001"}, "round_id"),
        ({"action": "dispatch", "selected_capability_ids": ["unknown.tool"]}, "Unknown"),
        ({"selected_capability_ids": ["macro.screen_rotor_torsion_topology"]}, "Only dispatch"),
    ],
)
def test_invalid_model_actions_are_not_silently_repaired(changes, message):
    with pytest.raises(ValueError, match=message):
        normalize(reply(**changes))


def test_repeated_route_with_different_parameters_is_preserved():
    spec = default_capability_registry().require("microscopic.run_torsion_snapshots")
    dispatches = [
        {
            "dispatch_id": f"R002:scan:{i}",
            "round_id": "R002",
            "agent_name": spec.owner_agent,
            "capability_id": spec.capability_id,
            "route": spec.route,
            "task": "Compare requested angles.",
            "objective": "Inspect torsional sensitivity.",
            "evidence_goal_family": spec.evidence_family,
            "tool_args": {"angles": angles},
        }
        for i, angles in enumerate(([0, 30], [45, 90]))
    ]
    result = PlannerDecision.model_validate(
        normalize(reply("dispatch", dispatch_requests=dispatches))
    )
    assert [r.tool_args["angles"] for r in result.dispatch_requests] == [[0, 30], [45, 90]]


def test_finalization_checks_shape_not_required_calculation_routes():
    case = _case_run_for_gate(with_prepared_structure=False)
    decision = PlannerDecision.model_validate(normalize(reply()))
    gate = DeterministicGate(GatePolicy(max_rounds=3))
    assert gate.apply(decision, case).status == "finalize"
    invalid = decision.model_copy(
        update={
            "portfolio": decision.portfolio.model_copy(
                update={"hypotheses": decision.portfolio.hypotheses[:2]}
            )
        }
    )
    assert gate.apply(invalid, case).status == "retry_policy"


def test_update_exposes_controlled_tools_and_reusable_artifacts():
    case = _case_run_for_gate()
    source = ArtifactRecord(
        artifact_id="R001:scan",
        kind="amesp_baseline_bundle",
        created_by_round="R001",
        created_by_agent="microscopic",
    )
    case = case.touch(artifact_manifest=case.artifact_manifest.with_records([source]))
    payload = _compact_planner_payload(
        case_run=case,
        round_id="R002",
        stage="planner_update",
        context={},
        capability_registry=default_capability_registry(),
    )
    ids = {card["capability_id"] for card in payload["capability_cards"]}
    assert {
        "microscopic.list_rotatable_dihedrals",
        "microscopic.run_relaxed_coordinate_scan",
        "microscopic.run_charge_population_panel",
    } <= ids
    assert "microscopic.select_computed_geometry" not in ids
    # The eager menu needs a scan bundle, not just an electronic baseline bundle.
    case = case.touch(artifact_manifest=case.artifact_manifest.with_records([
        source.model_copy(update={"artifact_id": "R001:relaxed_scan", "kind": "relaxed_coordinate_scan_bundle"})
    ]))
    with_scan = _compact_planner_payload(
        case_run=case, round_id="R002", stage="planner_update", context={},
        capability_registry=default_capability_registry(),
    )
    assert "microscopic.select_computed_geometry" in {card["capability_id"] for card in with_scan["capability_cards"]}
    assert "R001:scan" in {a["artifact_id"] for a in payload["artifact_manifest"]}


class ScriptedPlannerClient:
    def __init__(self, finish_round=8, initial_micro=False):
        self.payloads = []
        self.finish_round = finish_round
        self.initial_micro = initial_micro

    def is_configured(self):
        return True

    def complete_json(self, *, payload, system_prompt, schema_feedback=None):
        assert schema_feedback is None, schema_feedback
        self.payloads.append(payload)
        round_index = int(payload["round_id"][1:])
        if round_index == self.finish_round:
            return reply()
        capability = (
            "microscopic.run_baseline_bundle"
            if self.initial_micro and round_index == 1
            else "macro.screen_rotor_torsion_topology"
        )
        return reply("dispatch", selected_capability_ids=[capability])


def run_scripted(tmp_path, client, *, max_rounds):
    config = OrchestratorConfig(
        run_base_dir=tmp_path,
        max_rounds=max_rounds,
        enable_agenda_coverage_reviewer=False,
        enable_support_auditor=False,
    )
    deps = _fake_llm_deps(config)
    deps.planner = PlannerAgent(
        llm_client=client, enable_llm_decisions=True, capability_registry=deps.capability_registry
    )
    return MechCALOrchestrator(config, deps).run(
        smiles="CCO", user_query="Assess.", case_id="plan-a"
    )


def test_stable_ranking_does_not_stop_planner_or_trigger_forced_calibration(tmp_path):
    client = ScriptedPlannerClient(finish_round=8)
    result = run_scripted(tmp_path, client, max_rounds=10)
    assert result.status == "finalized"
    assert result.runtime["termination"]["reason"] == "planner_finalize_allowed"
    assert len(result.round_ids) == 8
    assert len(client.payloads) == 8
    assert len(result.mechanism_predictions) == 3
    assert not (tmp_path / "plan-a/history/planner_final_calibrations.jsonl").exists()
    assert all(p["run_limits"]["max_rounds"] == 10 for p in client.payloads)
    records = [
        json.loads(line)
        for line in (tmp_path / "plan-a/history/tool_events.jsonl").read_text().splitlines()
    ]
    requests = [row["dispatch"] for row in records if row["event"] == "agent_report"]
    assert len(requests) == 7
    assert {r["capability_id"] for r in requests} == {"macro.screen_rotor_torsion_topology"}


def test_unsupported_tool_feedback_reaches_next_planner_decision(tmp_path):
    client = ScriptedPlannerClient(finish_round=3, initial_micro=True)
    result = run_scripted(tmp_path, client, max_rounds=4)
    assert result.status == "finalized"
    feedback = client.payloads[1]["last_tool_feedback"]
    assert feedback[0]["status"] == "unsupported"
    assert feedback[0]["failure"]["kind"] == "capability_unsupported"
    assert "disabled" in feedback[0]["failure"]["message"].lower()
    assert client.payloads[2]["last_tool_feedback"][0]["agent_name"] == "macro"


def test_hard_limit_stays_a_limit_not_scientific_convergence(tmp_path):
    client = ScriptedPlannerClient(finish_round=99)
    result = run_scripted(tmp_path, client, max_rounds=3)
    assert result.status == "stopped"
    assert result.runtime["termination"]["reason"] == "max_rounds_reached"
    assert result.runtime["worker_call_counts"]["macro"] == 2
    assert len(client.payloads) == 3
