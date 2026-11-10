"""Scripted offline control-flow checks, not evidence of scientific performance."""

import json

import pytest

from mechcal.agents.mechanism_agenda import MechanismAgendaAgent
from mechcal.agents.planner import PlannerAgent
from mechcal.capabilities import default_capability_registry
from mechcal.capabilities.discovery import ToolDiscoverySession
from mechcal.runtime import MechCALOrchestrator, OrchestratorConfig
from mechcal.runtime.gates import DeterministicGate
from mechcal.runtime.retry import SchemaRetryError
from mechcal.schemas import PlannerDecision
from mechcal.schemas.decisions import StopReview, StoppingAssessment
from test_planner_owned_loop import normalize, reply
from test_runtime import _case_run_for_gate, _fake_llm_deps


TOOL = "macro.screen_rotor_torsion_topology"


def stopping_reply(action="finalize", *, basis="no_useful_available_step", **kwargs):
    return reply(action, stopping_assessment={
        "judgment_state": "unresolved", "finish_basis": basis,
        "reason": "Offline fixture: an unresolved bounded ranking, not a scientific result.",
    }, **kwargs)


def stopping_case(*, allowed=True):
    case = _case_run_for_gate()
    return case.touch(runtime={**case.runtime, "run_limits": {
        "stopping_protocol": "planner_stop_v1", "max_exploration_rounds": 2,
        "scientific_tools_allowed": allowed,
    }})


class ReviewerClient:
    def __init__(self, *, fail=False, challenge=True):
        self.calls = []
        self.fail = fail
        self.challenge = challenge

    def is_configured(self):
        return True

    def complete_json(self, *, payload, system_prompt, schema_feedback=None):
        self.calls.append(payload)
        if payload["task"] == "planner_stop_review":
            if self.fail:
                raise TimeoutError("Offline simulated reviewer timeout")
            return {
                "status": "reviewed",
                "challenge": "Could an untested route clarify the comparison?" if self.challenge else "",
                "rationale": "Offline advisory check, not evidence.",
                "capability_ids": [TOOL] if self.challenge else [],
                "evidence_refs": [],
            }
        return {"coverage_summary": "Offline coverage advice.", "decision_challenges": [],
                "agenda_items": [], "recommended_next_routes": []}


class PlannerClient:
    def __init__(self, *, mode="explore", discover=False, malformed=False):
        self.calls = []
        self.mode = mode
        self.discover = discover
        self.malformed = malformed

    def is_configured(self):
        return True

    def complete_json(self, *, payload, system_prompt, schema_feedback=None):
        self.calls.append(payload)
        if self.malformed:
            return reply()
        assert schema_feedback is None, schema_feedback
        if self.discover and payload["stage"] == "planner_initial":
            remaining = payload["tool_discovery"]["remaining_batches"]
            if remaining:
                cid = TOOL if remaining == 2 else "macro.macro_structure_scan"
                return {"action": "describe_tools", "capability_ids": [cid]}
        at_limit = not payload["run_limits"]["scientific_tools_allowed"]
        responding = payload["stage"] == "planner_stop_review_response"
        if responding and self.mode == "continue" and not at_limit:
            return stopping_reply(
                "dispatch", basis=None, selected_capability_ids=[TOOL],
                reviewer_response="Accept the question and request a permitted check.",
            )
        if at_limit or responding or self.mode != "explore":
            return stopping_reply(
                basis=None if at_limit else "no_useful_available_step",
                reviewer_response="Maintain the bounded answer; no further work is authorized."
                if responding else None,
                unresolved_gaps=["Fixture scientific comparison remains unresolved."],
            )
        return stopping_reply("dispatch", basis=None, selected_capability_ids=[TOOL])


def run_stopping(tmp_path, planner, reviewer=None, *, enabled=True, rounds=2, discover=False, unlimited=False):
    config = OrchestratorConfig(
        run_base_dir=tmp_path, max_exploration_rounds=rounds,
        enable_support_auditor=False, enable_agenda_coverage_reviewer=enabled,
        planner_tool_loading="on_demand" if discover else "eager",
        unlimited_exploration=unlimited,
    )
    deps = _fake_llm_deps(config)
    deps.planner = PlannerAgent(
        llm_client=planner, enable_llm_decisions=True,
        capability_registry=deps.capability_registry, tool_loading_mode=config.planner_tool_loading,
    )
    deps.mechanism_agenda = MechanismAgendaAgent(
        llm_client=reviewer or ReviewerClient(), capability_registry=deps.capability_registry,
    )
    return MechCALOrchestrator(config, deps).run(smiles="CCO", user_query="Assess.", case_id="stop-fixture")


def test_last_exploration_executes_and_its_results_reach_final_synthesis(tmp_path):
    client = PlannerClient()
    result = run_stopping(tmp_path, client)
    assert result.status == "finalized"
    assert result.runtime["worker_call_counts"]["macro"] == 2
    assert result.round_ids == ["R001", "R002", "R003"]
    final_payload = next(p for p in client.calls if p["stage"] == "planner_final_synthesis")
    assert final_payload["new_round_evidence"]
    assert final_payload["calculation_history"][-1]["round_id"] == "R002"
    assert final_payload["run_limits"]["exploration_rounds_used"] == 2
    terminal = result.runtime["termination"]
    assert terminal["reason"] == "exploration_limit_reached"
    assert terminal["limit_reached"] is True
    assert terminal["exploration_round_count"] == 2
    assert terminal["stopping_assessment"]["judgment_state"] == "unresolved"
    assert terminal["stopping_assessment"]["finish_basis"] is None
    assert terminal["unresolved_gaps"]
    assert len([p for p in client.calls if p["stage"] == "planner_stop_review_response"]) == 1


def test_planner_may_reject_review_without_forced_tools_or_ranking_changes(tmp_path):
    client, reviewer = PlannerClient(mode="finish"), ReviewerClient()
    result = run_stopping(tmp_path, client, reviewer)
    assert result.runtime["worker_call_counts"]["macro"] == 0
    assert result.runtime["termination"]["reason"] == "no_useful_available_step"
    assert result.runtime["termination"]["limit_reached"] is False
    assert len(result.evidence_ledger.items) == 0
    assert [h.name for h in result.portfolio.hypotheses] == ["RIM_RIR_RIV", "ICT_TICT_CT", "ESIPT_PT"]
    assert len([p for p in reviewer.calls if p["task"] == "planner_stop_review"]) == 1
    assert result.runtime["termination"]["reviewer_response"]
    assert (tmp_path / "stop-fixture/rounds/R001/planner_stop_proposal.json").exists()


def test_planner_may_accept_review_and_continue_without_resetting_query_allowance(tmp_path):
    client = PlannerClient(mode="continue", discover=True)
    result = run_stopping(tmp_path, client, rounds=1, discover=True)
    response_payload = next(p for p in client.calls if p["stage"] == "planner_stop_review_response")
    assert response_payload["tool_discovery"]["remaining_batches"] == 0
    assert response_payload["stop_review_feedback"]["challenge"]
    assert result.runtime["worker_call_counts"]["macro"] == 1
    assert result.runtime["termination"]["reason"] == "exploration_limit_reached"
    events = [json.loads(line) for line in
              (tmp_path / "stop-fixture/history/tool_discovery.jsonl").read_text().splitlines()]
    assert len([e for e in events if e["kind"] == "tool_description_query"]) == 2


def test_failed_review_is_not_approval_and_does_not_loop(tmp_path):
    client, reviewer = PlannerClient(mode="finish"), ReviewerClient(fail=True)
    result = run_stopping(tmp_path, client, reviewer)
    assert result.runtime["termination"]["stop_review_status"] == "unavailable"
    feedback = client.calls[-1]["stop_review_feedback"]
    assert feedback["status"] == "unavailable"
    assert "not approval" in feedback["rationale"]
    assert len([p for p in reviewer.calls if p["task"] == "planner_stop_review"]) == 1
    assert result.runtime["worker_call_counts"]["macro"] == 0


def test_disabled_reviewer_remains_disabled(tmp_path):
    client, reviewer = PlannerClient(mode="finish"), ReviewerClient()
    result = run_stopping(tmp_path, client, reviewer, enabled=False)
    assert reviewer.calls == []
    assert len(client.calls) == 1
    assert result.runtime["termination"]["stop_review_status"] == "disabled"


def test_missing_stop_assessment_is_not_invented(tmp_path):
    client = PlannerClient(malformed=True)
    result = run_stopping(tmp_path, client)
    assert result.status == "failed"
    assert len(client.calls) == 3
    assert result.runtime["termination"]["stopping_assessment"] is None
    assert result.runtime["worker_call_counts"]["macro"] == 0


def test_legacy_saved_decision_and_old_round_semantics_are_retained(tmp_path):
    old = PlannerDecision.model_validate(normalize(reply()))
    assert old.stopping_assessment is None
    assert old.reviewer_response is None
    config = OrchestratorConfig(run_base_dir=tmp_path, max_rounds=8)
    assert config.decision_round_limit == 8
    assert OrchestratorConfig(run_base_dir=tmp_path, max_exploration_rounds=8).decision_round_limit == 9


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_exploration_limits_rejected(tmp_path, limit):
    with pytest.raises(ValueError, match="positive"):
        OrchestratorConfig(run_base_dir=tmp_path, max_exploration_rounds=limit)


def test_runtime_gate_blocks_dispatch_at_limit_independently_of_planner():
    case = stopping_case(allowed=False)
    decision = PlannerDecision.model_validate(normalize(stopping_reply(
        "dispatch", basis=None, selected_capability_ids=[TOOL],
    )))
    gate = DeterministicGate().apply(decision, case)
    assert gate.terminal_reason == "exploration_limit_reached"
    assert gate.dispatch_requests == []


def test_unknown_or_disabled_reviewer_capabilities_are_rejected():
    class BadReviewer(ReviewerClient):
        def complete_json(self, **kwargs):
            return {"status": "reviewed", "rationale": "Fixture.", "capability_ids": ["unknown.tool"]}

    proposal = PlannerDecision.model_validate(normalize(stopping_reply()))
    with pytest.raises(SchemaRetryError, match="unknown or disabled"):
        MechanismAgendaAgent(llm_client=BadReviewer()).review_stop(stopping_case(), proposal)


def test_stop_review_has_no_rank_or_dispatch_contract():
    fields = StopReview.model_fields
    assert not {"rank", "priority", "dispatch_requests", "action"}.intersection(fields)


def test_query_budget_is_isolated_between_rounds_and_cases():
    registry = default_capability_registry()
    case = stopping_case()
    budget = {"case_id": case.case_id, "round_id": "R001", "batches_used": 2}
    case = case.touch(runtime={**case.runtime, "tool_discovery_budget": budget})
    session = ToolDiscoverySession(registry, case, "R001")
    with pytest.raises(ValueError, match="exhausted"):
        session.query({"action": "describe_tools", "capability_ids": [TOOL]})
    assert ToolDiscoverySession(registry, case, "R002").batches_used == 0
    other = case.model_copy(update={"case_id": "different-case"})
    assert ToolDiscoverySession(registry, other, "R001").batches_used == 0


@pytest.mark.parametrize("problem", ["missing_basis", "blank_reason", "missing_response", "tools_at_limit"])
def test_new_contract_errors_use_bounded_schema_correction(problem):
    case = stopping_case(allowed=problem != "tools_at_limit")
    if problem == "missing_response":
        case = case.touch(runtime={**case.runtime, "stop_review_feedback": {
            "status": "reviewed", "challenge": "Fixture challenge.",
        }})

    class CorrectingPlanner:
        def __init__(self):
            self.feedback = []

        def is_configured(self):
            return True

        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            self.feedback.append(schema_feedback)
            if schema_feedback:
                return stopping_reply(reviewer_response="Bounded fixture response.")
            if problem == "tools_at_limit":
                return stopping_reply("dispatch", selected_capability_ids=[TOOL])
            response = stopping_reply(basis=None if problem == "missing_basis" else "no_useful_available_step")
            if problem == "blank_reason":
                response["stopping_assessment"]["reason"] = "  "
            return response

    client = CorrectingPlanner()
    decision = PlannerAgent(llm_client=client, enable_llm_decisions=True).plan_initial(case, round_id="R002")
    assert decision.action == "finalize"
    assert len(client.feedback) == 2
    assert client.feedback[1]


def test_exhausted_discovery_budget_cannot_reset_via_review_response_or_format_retry():
    case = stopping_case()
    case = case.touch(runtime={
        **case.runtime,
        "tool_discovery_budget": {"case_id": case.case_id, "round_id": "R002", "batches_used": 2},
        "stop_review_feedback": {"status": "reviewed", "challenge": "Fixture."},
    })

    class QueryOnly:
        def __init__(self):
            self.remaining = []

        def is_configured(self):
            return True

        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            self.remaining.append(payload["tool_discovery"]["remaining_batches"])
            return {"action": "describe_tools", "capability_ids": [TOOL]}

    client = QueryOnly()
    decision = PlannerAgent(
        llm_client=client, enable_llm_decisions=True, tool_loading_mode="on_demand",
    ).respond_to_stop_review(case, round_id="R002")
    assert decision.failure.is_failure
    assert client.remaining == [0, 0, 0]
    assert decision.dispatch_requests == []
    assert decision.raw_response["tool_discovery_budget"]["batches_used"] == 2


def test_reviewer_response_cannot_bypass_loaded_tool_check():
    case = stopping_case()
    case = case.touch(runtime={**case.runtime, "stop_review_feedback": {
        "status": "reviewed", "challenge": "Try a check.",
    }})

    class UnloadedDispatch:
        def is_configured(self):
            return True

        def complete_json(self, **kwargs):
            return stopping_reply("dispatch", selected_capability_ids=[TOOL],
                                  reviewer_response="Accept the suggestion.")

    decision = PlannerAgent(llm_client=UnloadedDispatch(), enable_llm_decisions=True,
                            tool_loading_mode="on_demand").respond_to_stop_review(case, round_id="R002")
    assert decision.failure.is_failure
    assert decision.dispatch_requests == []


def test_failed_tool_is_visible_at_limit_and_not_replaced_with_scientific_evidence(tmp_path):
    class UnsupportedMicro(PlannerClient):
        def complete_json(self, *, payload, **kwargs):
            if payload["run_limits"]["scientific_tools_allowed"]:
                self.calls.append(payload)
                return stopping_reply("dispatch", basis=None,
                                      selected_capability_ids=["microscopic.run_baseline_bundle"])
            return super().complete_json(payload=payload, **kwargs)

    client = UnsupportedMicro()
    result = run_stopping(tmp_path, client, rounds=1, enabled=False)
    assert result.runtime["worker_call_counts"]["microscopic"] == 1
    final = client.calls[-1]
    assert final["calculation_history"][-1]["status"] == "unsupported"
    assert final["last_tool_feedback"][0]["failure"]["kind"] == "capability_unsupported"
    assert result.runtime["termination"]["reason"] == "exploration_limit_reached"


def test_new_protocol_cannot_silently_reuse_legacy_case_cache(tmp_path):
    from mechcal.cli.evaluate import _run_or_load_mas

    case = _case_run_for_gate().touch(status="finalized", runtime={"ablation": {"mode": "full_mechcal"}})
    root = tmp_path / case.case_id
    root.mkdir()
    (root / "case_run.json").write_text(case.model_dump_json())
    with pytest.raises(RuntimeError, match="different stopping protocol"):
        _run_or_load_mas(
            {"case_id": case.case_id}, output_dir=tmp_path, run_base_dir=tmp_path,
            max_rounds=8, max_exploration_rounds=8, ablation_mode="full_mechcal",
            enable_amesp=False, enable_support_auditor=False,
            enable_agenda_coverage_reviewer=False, enable_incremental_mechanism_portfolio=False,
            enable_support_auditor_web_search=False, continue_on_arbiter_failure=False,
            continue_on_planner_update_failure=False, verifier_failure_stop_threshold=0, refresh=False,
        )


def test_planner_can_finish_with_a_bounded_judgment_without_forced_calculations(tmp_path):
    class AdequatePlanner(PlannerClient):
        def complete_json(self, **kwargs):
            response = super().complete_json(**kwargs)
            response["stopping_assessment"].update(
                judgment_state="adequate_for_bounded_ranking", finish_basis="judgment_sufficient",
            )
            return response

    result = run_stopping(tmp_path, AdequatePlanner(mode="finish"))
    assert result.runtime["termination"]["reason"] == "judgment_sufficient"
    assert result.runtime["termination"]["limit_reached"] is False
    assert sum(result.runtime["worker_call_counts"].values()) == 0
    assert result.runtime["termination"]["unresolved_gaps"]


def test_assessment_checks_status_consistency_not_scientific_truth():
    with pytest.raises(ValueError, match="requires adequate"):
        StoppingAssessment(judgment_state="unresolved", finish_basis="judgment_sufficient", reason="Fixture.")


def test_stop_review_cannot_return_ranking_or_tool_actions():
    with pytest.raises(ValueError):
        StopReview.model_validate({"rationale": "Fixture.", "dispatch_requests": []})


def test_usage_metadata_distinguishes_review_response_and_synthesis():
    from mechcal.runtime.llm import _payload_metadata

    for stage in ("planner_stop_review_response", "planner_final_synthesis"):
        assert _payload_metadata({"stage": stage})["stage"] == stage
    assert _payload_metadata({"task": "planner_stop_review"})["task"] == "planner_stop_review"


def test_single_case_cli_exposes_new_policy_and_termination_without_an_api_call(monkeypatch):
    import typer
    from typer.testing import CliRunner

    from mechcal.cli import run_case

    configs = []

    class OfflineOrchestrator:
        def __init__(self, config, deps=None):
            configs.append(config)

        def run(self, **kwargs):
            return _case_run_for_gate().touch(status="finalized", runtime={
                "termination": {"reason": "exploration_limit_reached", "exploration_round_count": 8},
            })

    monkeypatch.setattr(run_case, "MechCALOrchestrator", OfflineOrchestrator)
    app = typer.Typer()
    app.command()(run_case.run)
    result = CliRunner().invoke(app, ["--smiles", "CCO", "--max-exploration-rounds", "8"])
    assert result.exit_code == 0, result.output
    assert configs[0].max_exploration_rounds == 8
    assert configs[0].decision_round_limit == 9
    assert json.loads(result.output)["termination"]["reason"] == "exploration_limit_reached"


def test_unlimited_exploration_continues_past_eight_then_planner_finishes(tmp_path):
    class FinishAtTen(PlannerClient):
        def complete_json(self, *, payload, **kwargs):
            if int(payload["round_id"][1:]) >= 10:
                self.mode = "finish"
            return super().complete_json(payload=payload, **kwargs)

    client = FinishAtTen()
    result = run_stopping(tmp_path, client, rounds=None, unlimited=True)
    assert result.status == "finalized"
    assert len(result.round_ids) == 10
    assert result.runtime["worker_call_counts"]["macro"] == 9
    assert result.runtime["termination"]["limit_reached"] is False
    assert result.runtime["termination"]["exploration_round_count"] == 9
    assert result.runtime["termination"]["reason"] == "no_useful_available_step"
    assert all(p["run_limits"]["max_rounds"] is None for p in client.calls)
    assert all(p["run_limits"]["exploration_rounds_remaining"] is None for p in client.calls)
    assert all(p["run_limits"]["scientific_tools_allowed"] for p in client.calls)
    assert not any(p["stage"] == "planner_final_synthesis" for p in client.calls)


def test_unlimited_config_is_explicit_and_conflicting_limits_are_rejected(tmp_path):
    assert OrchestratorConfig(run_base_dir=tmp_path, unlimited_exploration=True).decision_round_limit is None
    with pytest.raises(ValueError, match="not both"):
        OrchestratorConfig(run_base_dir=tmp_path, unlimited_exploration=True, max_exploration_rounds=8)


def test_unlimited_gate_does_not_apply_legacy_limit():
    case = stopping_case()
    case = case.touch(runtime={**case.runtime, "run_limits": {
        "stopping_protocol": "planner_stop_v1", "max_exploration_rounds": None,
        "unlimited_exploration": True, "scientific_tools_allowed": True,
    }})
    decision = PlannerDecision.model_validate(normalize(stopping_reply(
        "dispatch", basis=None, selected_capability_ids=[TOOL],
    )))
    decision = decision.model_copy(update={"round_id": "R099"})
    result = DeterministicGate().apply(decision, case)
    assert not result.terminal
    assert result.dispatch_requests
