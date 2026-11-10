"""Regression coverage for unavailable tools and invalid Planner decisions."""

from types import SimpleNamespace
import json

import pytest

from mechcal.agents import planner
from mechcal.capabilities import default_capability_registry
from mechcal.runtime.gates import DeterministicGate, GatePolicy
from mechcal.runtime.orchestrator import MechCALOrchestrator, OrchestratorConfig
from mechcal.runtime.persistence import RunStore
from mechcal.runtime.output_assembler import OutputAssembler
from mechcal.schemas import (
    ArtifactManifest,
    CaseInput,
    CaseRun,
    EvidenceLedger,
    HypothesisEntry,
    HypothesisPortfolio,
    PlannerDecision,
    RoundRecord,
)
from mechcal.schemas.failures import FailureReport


def case_with_ranking():
    return CaseRun(
        case_id="failure-test",
        input=CaseInput(case_id="failure-test", smiles="CC", user_query="Assess."),
        status="running",
        current_round_id="R002",
        round_ids=["R001"],
        evidence_ledger=EvidenceLedger(case_id="failure-test"),
        artifact_manifest=ArtifactManifest(case_id="failure-test"),
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            hypotheses=[
                HypothesisEntry(name="RIM_RIR_RIV", confidence=0.4),
            ],
        ),
    )


def failed_decision(case):
    return PlannerDecision(
        decision_id="invalid",
        round_id="R002",
        portfolio=HypothesisPortfolio(),
        current_hypothesis="unknown",
        diagnosis="Invalid response.",
        action="stop",
        failure=FailureReport(
            kind="llm_schema_invalid",
            message="Retries exhausted.",
            details={"attempt_errors": ["missing artifact"]},
        ),
    )


@pytest.mark.parametrize("available", [[], ["macro.run_crystal_restriction_checklist"]])
def test_coverage_recommendations_never_create_dispatches(available):
    registry = default_capability_registry()
    context = {
        "artifact_available_capability_ids": available,
        "capability_cards": registry.cards(),
        "agenda_coverage_memo": {
            "agenda_items": [
                {
                    "urgency": "high",
                    "coverage_status": "needs_follow_up",
                    "suggested_route": "microscopic.run_frontier_orbital_partition",
                }
            ],
            "recommended_next_routes": [
                {
                    "capability_ids": [
                        "macro.run_crystal_restriction_checklist",
                    ]
                }
            ],
        },
    }
    normalized = planner._normalize_planner_llm_response(
        {"action": "stop", "portfolio": case_with_ranking().portfolio.model_dump(mode="json")},
        round_id="R002",
        capability_registry=registry,
        existing_portfolio={},
        previous_portfolio_hypotheses=[],
        allowed_evidence_ids=[],
        prompt_payload=context,
    )
    assert normalized["dispatch_requests"] == []
    assert normalized["action"] == "stop"


def test_compact_payload_records_actual_availability_without_geometry():
    payload = planner._compact_planner_payload(
        case_run=case_with_ranking(),
        round_id="R002",
        stage="planner_update",
        context={},
        capability_registry=default_capability_registry(),
    )
    available = payload["artifact_available_capability_ids"]
    assert "microscopic.run_frontier_orbital_partition" not in available
    assert "macro.screen_metal_triplet_prior" in available


def test_empty_availability_does_not_fall_back_to_full_catalog():
    context = {
        "artifact_available_capability_ids": [],
        "capability_cards": default_capability_registry().cards(),
    }
    assert planner._available_capability_ids_from_payload(context) == set()


def test_schema_failure_is_not_reported_as_normal_stop_or_budget_exhaustion():
    case = case_with_ranking()
    gate = DeterministicGate(GatePolicy(max_rounds=2))
    result = gate.apply(failed_decision(case), case)
    assert result.terminal
    assert result.status == "retry_schema"
    assert result.terminal_reason == "planner_decision_invalid"
    assert result.violations


def test_failed_calibration_does_not_overwrite_valid_portfolio(tmp_path):
    case = case_with_ranking()
    store = RunStore(tmp_path, case.case_id)
    store.initialize()
    store.write_case_run(case)
    deps = SimpleNamespace(
        planner=SimpleNamespace(
            calibrate_final_portfolio=lambda *args, **kwargs: failed_decision(case),
        )
    )
    orchestrator = MechCALOrchestrator(OrchestratorConfig(run_base_dir=tmp_path), deps=deps)
    with pytest.raises(RuntimeError, match="Retries exhausted"):
        orchestrator._calibrate_final_planner_portfolio(case, store, force=True)
    persisted = CaseRun.model_validate_json((store.root / "case_run.json").read_text())
    assert persisted.portfolio == case.portfolio
    assert (store.root / "rounds/R002/planner_final_calibration.json").exists()


def test_invalid_planner_decision_marks_case_failed(tmp_path, monkeypatch):
    case = case_with_ranking()
    decision = failed_decision(case)
    gate = DeterministicGate().apply(decision, case)
    record = RoundRecord(
        case_id=case.case_id,
        round_id="R002",
        round_index=2,
        planner_decision=decision,
        gate_result=gate,
    )
    store = RunStore(tmp_path, case.case_id)
    store.initialize()
    orchestrator = MechCALOrchestrator(
        OrchestratorConfig(run_base_dir=tmp_path), deps=SimpleNamespace()
    )
    monkeypatch.setattr(orchestrator, "_assemble_current_planner_predictions", lambda c, s: c)
    result = orchestrator._finish_case(case, decision, record, store, gate)
    assert result.status == "failed"
    assert result.runtime["failure"]["type"] == "llm_schema_invalid"
    assert result.runtime["failure"]["details"]["attempt_errors"] == ["missing artifact"]


def test_evaluator_does_not_score_empty_stopped_run(tmp_path, monkeypatch):
    from mechcal.cli import evaluate
    from mechcal.runtime.llm import OpenAICompatibleSettings

    config = OpenAICompatibleSettings(
        base_url="https://offline.test", model="test", api_key="not-used"
    )
    monkeypatch.setattr(evaluate.OpenAICompatibleSettings, "from_env", lambda: config)
    monkeypatch.setattr(
        evaluate,
        "_load_cases",
        lambda _: [
            {"case_id": "empty-run", "reference_mechanisms": []},
        ],
    )
    monkeypatch.setattr(
        evaluate,
        "LLMMechanismEvidenceSupportJudge",
        lambda **_: SimpleNamespace(
            model="no-api-test",
            rubric={"mechanism_evidence_rubric_version": "v2"},
        ),
    )
    monkeypatch.setattr(
        evaluate,
        "_run_or_load_mas",
        lambda *args, **kwargs: SimpleNamespace(
            status="stopped",
            runtime={},
            round_ids=["R001", "R002"],
        ),
    )
    monkeypatch.setattr(evaluate, "_prediction_payload", lambda _: [])
    calls = []
    monkeypatch.setattr(
        evaluate,
        "_judge_support_with_retries",
        lambda *args, **kwargs: calls.append(True) or {"support_scores": {}, "judgements": []},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "mechcal-evaluate",
            "--case-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--progress",
            "off",
        ],
    )
    evaluate.main()
    assert calls == []
    row = json.loads((tmp_path / "out/mas_eval/empty-run.json").read_text())
    assert row["evaluation_status"] == "invalid_output"
    assert all(value is None for value in row["scores"].values())


@pytest.mark.parametrize(
    "metrics",
    [
        {"state_count": 1, "missing_deliverable_count": 2},
        {"ct_descriptor_proxy": True, "electron_hole_separation_proxy": True},
        {"state_count": 0, "missing_deliverable_count": 1},
        {"state_count": 2, "oscillator_strength_range": 0.8},
        {"esipt_motif_proxy": True},
    ],
)
def test_normalizer_does_not_apply_rule_priority_adjustments(metrics):
    assert not hasattr(planner, "_apply_evidence_tier_priority_floor")
    assert not hasattr(planner, "_apply_low_margin_carry_forward")
    response = {
        "portfolio": {
            "current": "ICT_TICT_CT",
            "hypotheses": [
                {"name": "ICT_TICT_CT", "confidence": 0.7, "differential_priority": 0.7},
                {"name": "RIM_RIR_RIV", "confidence": 0.6, "differential_priority": 0.0},
            ],
        },
        "action": "finalize",
        "final_answer_draft": "Bounded candidate ranking.",
    }
    result = planner._normalize_planner_llm_response(
        response,
        round_id="R002",
        existing_portfolio={},
        previous_portfolio_hypotheses=[],
        allowed_evidence_ids=[],
        capability_registry=default_capability_registry(),
        prompt_payload={"new_round_evidence": [{"metrics": metrics}]},
    )
    assert {h["name"]: h["differential_priority"] for h in result["portfolio"]["hypotheses"]} == {
        "ICT_TICT_CT": 0.7,
        "RIM_RIR_RIV": 0.0,
    }


def test_output_assembler_never_replaces_planner_priority():
    import mechcal.runtime.output_assembler as assembly

    assert not hasattr(assembly, "_adjusted_candidate_priority")
    case = case_with_ranking().touch(
        portfolio=HypothesisPortfolio(
            current="ICT_TICT_CT",
            hypotheses=[
                HypothesisEntry(name="RIM_RIR_RIV", confidence=0.3, differential_priority=0.3),
                HypothesisEntry(name="ICT_TICT_CT", confidence=0.8, differential_priority=0.8),
            ],
        )
    )
    predictions = OutputAssembler().build_mechanism_predictions(case)
    assert [(p.label, p.differential_priority) for p in predictions] == [
        ("ICT_TICT_CT", 0.8),
        ("RIM_RIR_RIV", 0.3),
    ]


@pytest.mark.parametrize("value", [-0.1, 1.2, float("nan"), float("inf"), "not-a-score"])
def test_invalid_planner_priority_is_rejected_not_clipped(value):
    with pytest.raises(ValueError, match="priority"):
        planner._normalize_portfolio_payload(
            {
                "hypotheses": [{"name": "RIM_RIR_RIV", "differential_priority": value}],
            }
        )


def test_unknown_hypothesis_reference_is_rejected():
    case = case_with_ranking()
    decision = PlannerDecision(
        decision_id="invalid-citation",
        round_id="R002",
        action="stop",
        current_hypothesis="RIM_RIR_RIV",
        diagnosis="Citation validation test.",
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            hypotheses=[
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    differential_priority=0.9,
                    evidence_refs=["E_does_not_exist"],
                ),
            ],
        ),
    )
    with pytest.raises(ValueError, match="E_does_not_exist"):
        planner.PlannerAgent()._validate_planner_policy(decision, case)
