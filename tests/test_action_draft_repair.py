"""Offline action repairs retain the tool plan instead of regenerating new work."""

from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from mechcal.agents.planner import PlannerAgent
from mechcal.runtime.hypothesis_tests import validate_tests
from mechcal.runtime.decision_value import validate_action_value
from mechcal.runtime.persistence import RunStore
from mechcal.runtime.questions import apply_assimilation, public_payload, state, validate_action
from mechcal.runtime.retry import RequestValidationError
from mechcal.runtime.schema_diagnostics import schema_issues
from mechcal.runtime.structured_output import PlannerOutputContract
from mechcal.schemas.question_loop import QuestionAction
from test_delta_context import WireClient
from test_hypothesis_tests import start, planned_test
from test_question_loop import action, REGISTRY


def fixture():
    case, update = start()
    case = apply_assimilation(case, update)
    case.runtime.update(planner_strict_outputs=True, planner_case_memory=True,
                        planner_delta_context=True, scoped_evidence_updates=True)
    request = action(case.case_id, state(case)["revision"], dispatch=True)
    plan = planned_test()
    plan.dispatch_id = request.dispatch_requests[0].dispatch_id
    request.investigation_plans = [plan]
    return case, request


def test_wrong_round_and_extra_field_are_reported_together():
    case, request = fixture()
    payload = public_payload(case, REGISTRY, stage="planner_question_act")
    payload.update(round_id="R001", output_contract=QuestionAction.model_json_schema())
    contract = PlannerOutputContract(payload)
    wire = contract.encode(request.model_dump(mode="json"))
    body = wire["response"]["data"]
    body["dispatch_requests"][0]["round_id"] = "R999"
    body["investigation_plans"][0]["usefulness_note"] = "Undeclared duplicate."
    issues = [i for e in Draft202012Validator(contract.schema).iter_errors(wire) for i in schema_issues(e)]
    assert {tuple(i["path"]) for i in issues} == {
        ("response", "data", "dispatch_requests", 0, "round_id"),
        ("response", "data", "investigation_plans", 0, "usefulness_note"),
    }
    assert next(i for i in issues if i["path"][-1] == "usefulness_note")["type"] == "extra_forbidden"


def test_strict_memory_action_repairs_only_bad_fields(tmp_path):
    case, request = fixture()
    store = RunStore(tmp_path, case.case_id)
    store.initialize()

    def repair(payload):
        edits = []
        assert {t["field"] for t in payload["targets"]} == {"round_id", "usefulness_note"}
        for target in payload["targets"]:
            edit = dict(target_id=target["target_id"], operation=target["operation"])
            if target["operation"] == "replace":
                edit["value"] = "R001"
            edits.append(edit)
        return dict(action="repair_judgment", draft_id=payload["draft_id"], edits=edits)

    client = WireClient([(request.model_dump(mode="json"), "decision"), (repair, "decision")])
    client.is_configured = lambda: True
    complete = client.complete_json
    def respond(**kwargs):
        first = not client.calls
        result = complete(**kwargs)
        if first:
            body = result["response"]["data"]
            body["dispatch_requests"][0]["round_id"] = "R999"
            body["investigation_plans"][0]["usefulness_note"] = "Undeclared duplicate."
        return result
    client.complete_json = respond
    before = case.model_dump_json()
    events = []
    agent = PlannerAgent(capability_registry=REGISTRY, llm_client=client, enable_llm_decisions=True)
    result = agent.plan_initial(case, round_id="R001", memory_store=store, request_event_sink=events.append)
    assert len(client.calls) == 2
    assert client.calls[-1]["payload"]["stage"] == "planner_judgment_repair"
    assert result.dispatch_requests == request.dispatch_requests
    assert result.raw_response["question_action"]["investigation_plans"] == [p.model_dump(mode="json") for p in request.investigation_plans]
    assert result.portfolio == case.portfolio and case.model_dump_json() == before
    assert any(e["kind"] == "draft_repair_applied" for e in events)


def test_dispatch_policy_repair_retains_tool_definition_and_plan(tmp_path):
    from mechcal.runtime.context_compaction import expand_payload

    case, request = fixture()
    cid = request.dispatch_requests[0].capability_id
    case.runtime.update(planner_tool_loading="on_demand", tool_discovery={
        "case_id": case.case_id, "loaded_tools": {cid: REGISTRY.definition_version(cid)}})
    bad = request.model_dump(mode="json")
    bad["dispatch_requests"][0]["agent_name"] = "microscopic"
    store = RunStore(tmp_path, case.case_id)
    store.initialize()

    def repair(payload):
        payload = expand_payload(payload)
        assert len(payload["targets"]) == 1
        assert payload["targets"][0]["field"] == 0  # One request, not whole_command.
        cards = payload["capability_cards"]
        assert [c["capability_id"] for c in cards] == [cid]
        assert cards[0]["tool_arg_schema"]
        issue = next(i for i in payload["issues"] if "execution_contract" in i)
        assert issue["execution_contract"]["capability_id"] == cid
        return dict(action="repair_judgment", draft_id=payload["draft_id"], edits=[
            dict(target_id=payload["targets"][0]["target_id"], operation="replace",
                 value=request.dispatch_requests[0].model_dump(mode="json"))])

    client = WireClient([(bad, "decision"), (repair, "decision")])
    client.is_configured = lambda: True
    before = case.model_dump_json()
    agent = PlannerAgent(capability_registry=REGISTRY, llm_client=client, enable_llm_decisions=True,
                         tool_loading_mode="on_demand")
    result = agent.plan_initial(case, round_id="R001", memory_store=store)
    assert len(client.calls) == 2
    assert result.dispatch_requests == request.dispatch_requests
    assert result.raw_response["question_action"]["investigation_plans"] == bad["investigation_plans"]
    assert case.model_dump_json() == before


def test_unloaded_dispatch_issues_have_typed_request_paths():
    from mechcal.runtime.questions import decision_from_action, validate_decision
    from mechcal.runtime.judgment_draft import JudgmentDraft

    case, request = fixture()
    case.runtime["planner_tool_loading"] = "on_demand"
    with pytest.raises(RequestValidationError) as exc:
        validate_decision(decision_from_action(request, case, "R001"), case, REGISTRY)
    assert all(i["path"] == ["dispatch_requests", 0] for i in exc.value.issues)
    draft = JudgmentDraft(case.case_id, state(case)["revision"], "planner_question_act",
                          QuestionAction, {"rejected_response": request.model_dump(mode="json"),
                          "issues": exc.value.issues, "error": str(exc.value)})
    assert [t["path"] for t in draft.targets.values()] == [["dispatch_requests", 0]]


def test_all_inconsistent_test_pairs_reported_without_changing_plans():
    case, request = fixture()
    request.investigation_plans *= 2
    for p in request.investigation_plans:
        p.candidate_labels = [p.hypothesis_test.target_label]
    before = deepcopy(request.model_dump())
    with pytest.raises(RequestValidationError) as exc:
        validate_tests(case, request)
    assert len(exc.value.issues) == 2
    assert request.model_dump() == before
    assert all(len(i["repair_paths"]) == 2 for i in exc.value.issues)


def invalid_outcomes_fixture():
    from test_decision_economy import meaningful_plan
    from test_decision_value import outcomes

    case, request, plan = meaningful_plan()
    benefits = outcomes("support", "support")
    for outcome in benefits:
        outcome.affected_labels = ["ESIPT_PT"]
        outcome.judgment_change = "weaken"
    second = plan.model_copy(deep=True)
    second.dispatch_id = "R001:second"
    second.possible_outcomes = deepcopy(benefits)
    request.dispatch_requests.append(request.dispatch_requests[0].model_copy(
        deep=True, update={"dispatch_id": second.dispatch_id}))
    request.investigation_plans.append(second)
    plan.purpose = "preparation"
    plan.possible_outcomes = outcomes("preparation", "no_change")
    plan.decision_value.downstream_capability_id = request.dispatch_requests[0].capability_id
    plan.decision_value.downstream_question = "Would the next observation change support?"
    plan.decision_value.downstream_outcomes = benefits
    return case, request


def test_all_outcome_conflicts_across_preparation_and_scientific_plans_reported():
    case, request = invalid_outcomes_fixture()
    before = request.model_dump_json(), case.model_dump_json()
    with pytest.raises(RequestValidationError) as exc:
        validate_action_value(case, request, REGISTRY)
    assert len(exc.value.issues) == 4
    assert {tuple(i["path"]) for i in exc.value.issues} == {
        ("investigation_plans", 0, "decision_value", "downstream_outcomes", 0, "affected_labels"),
        ("investigation_plans", 0, "decision_value", "downstream_outcomes", 1, "affected_labels"),
        ("investigation_plans", 1, "possible_outcomes", 0, "affected_labels"),
        ("investigation_plans", 1, "possible_outcomes", 1, "affected_labels"),
    }
    for issue in exc.value.issues:
        assert issue["candidate_labels"] == ["RIM_RIR_RIV", "ICT_TICT_CT"]
        assert issue["submitted_affected_labels"] == ["ESIPT_PT"]
        assert issue["repair_paths"][1] == [*issue["path"][:2], "candidate_labels"]
    assert before == (request.model_dump_json(), case.model_dump_json())


def test_action_preflight_combines_plan_value_and_dispatch_errors():
    case, request = invalid_outcomes_fixture()
    request.investigation_plans[0].usefulness = " "
    request.investigation_plans[0].decision_value.observation_scope = " "
    request.dispatch_requests[0].question_ids = ["missing-question"]
    with pytest.raises(RequestValidationError) as exc:
        validate_action(request, case, REGISTRY)
    paths = [i["path"] for i in exc.value.issues]
    assert ["investigation_plans", 0, "usefulness"] in paths
    assert ["investigation_plans", 0, "decision_value", "observation_scope"] in paths
    assert ["dispatch_requests", 0, "question_ids"] in paths
    assert sum(p[-1] == "affected_labels" for p in paths) == 4
    assert all(isinstance(p, list) for p in paths)


def test_strict_memory_action_repairs_all_nested_outcomes_in_one_call(tmp_path):
    case, request = invalid_outcomes_fixture()
    case.runtime.update(planner_strict_outputs=True, planner_case_memory=True,
                        planner_delta_context=True, scoped_evidence_updates=True)
    store = RunStore(tmp_path, case.case_id)
    store.initialize()

    def repair(payload):
        from mechcal.runtime.context_compaction import expand_payload
        payload = expand_payload(payload)
        targets = [t for t in payload["targets"] if t["field"] == "affected_labels"]
        assert len(targets) == 4
        return dict(action="repair_judgment", draft_id=payload["draft_id"], edits=[
            dict(target_id=t["target_id"], operation="replace", value=["ICT_TICT_CT"])
            for t in targets
        ])

    client = WireClient([(request.model_dump(mode="json"), "decision"), (repair, "decision")])
    client.is_configured = lambda: True
    before = case.model_dump_json()
    agent = PlannerAgent(capability_registry=REGISTRY, llm_client=client, enable_llm_decisions=True)
    result = agent.plan_initial(case, round_id="R001", memory_store=store)
    assert len(client.calls) == 2
    assert result.dispatch_requests == request.dispatch_requests
    assert result.portfolio == case.portfolio and case.model_dump_json() == before
    repaired = QuestionAction.model_validate(result.raw_response["question_action"])
    for plan in repaired.investigation_plans:
        assert plan.candidate_labels == ["RIM_RIR_RIV", "ICT_TICT_CT"]
        for outcome in plan.possible_outcomes + plan.decision_value.downstream_outcomes:
            if outcome.decision_effect == "support":
                assert outcome.affected_labels == ["ICT_TICT_CT"]


def invalid_stop_fixture():
    from test_boundary_decisions import start, ending

    case, update = start()
    case = apply_assimilation(case, update)
    request = ending(case)
    route = request.continuation_assessment.routes[0]
    route.disposition = "blocked"
    for cid in ("macro.screen_donor_acceptor_layout", "macro.screen_metal_triplet_prior"):
        request.continuation_assessment.routes.append(route.model_copy(update={"capability_id": cid}))
    return case, request


def test_stop_reports_all_route_and_finalization_errors_without_mutation():
    case, request = invalid_stop_fixture()
    request.ranking_uncertainty_material = None
    request.question_dispositions = {}
    request.ranking_justification = " "
    request.unresolved_gaps = []
    before = request.model_dump_json(), case.model_dump_json()
    with pytest.raises(RequestValidationError) as exc:
        validate_action(request, case, REGISTRY)
    paths = [i["path"] for i in exc.value.issues]
    assert paths == [["continuation_assessment", "routes", index] for index in range(3)] + [
        ["ranking_uncertainty_material"], ["question_dispositions"],
        ["ranking_justification"], ["unresolved_gaps"],
    ]
    assert all("callable, not scientifically sufficient" in i["message"] for i in exc.value.issues[:3])
    assert before == (request.model_dump_json(), case.model_dump_json())


def test_stop_invalid_inquiry_does_not_hide_other_routes():
    case, request = invalid_stop_fixture()
    request.continuation_assessment.routes[0].tool_args = {"not_an_argument": True}
    with pytest.raises(RequestValidationError) as exc:
        validate_action(request, case, REGISTRY)
    assert ["continuation_assessment", "routes", 0, "tool_args"] in [i["path"] for i in exc.value.issues]
    assert ["continuation_assessment", "routes", 1] in [i["path"] for i in exc.value.issues]
    assert ["continuation_assessment", "routes", 2] in [i["path"] for i in exc.value.issues]


def test_strict_memory_stop_repairs_all_routes_without_rewriting_science(tmp_path):
    case, request = invalid_stop_fixture()
    case.runtime.update(planner_strict_outputs=True, planner_case_memory=True,
                        planner_delta_context=True, scoped_evidence_updates=True)
    store = RunStore(tmp_path, case.case_id)
    store.initialize()

    def repair(payload):
        from mechcal.runtime.context_compaction import expand_payload
        payload = expand_payload(payload)
        targets = payload["targets"]
        assert len(targets) == 3
        edits = []
        for t in targets:
            route = dict(t["current_value"])
            route.update(disposition="not_useful", reason="Synthetic test: the available route repeats a known observation.")
            edits.append(dict(target_id=t["target_id"], operation="replace", value=route))
        return dict(action="repair_judgment", draft_id=payload["draft_id"], edits=edits)

    client = WireClient([(request.model_dump(mode="json"), "decision"), (repair, "decision")])
    client.is_configured = lambda: True
    before = case.model_dump_json()
    agent = PlannerAgent(capability_registry=REGISTRY, llm_client=client, enable_llm_decisions=True)
    result = agent.plan_initial(case, round_id="R001", memory_store=store)
    assert len(client.calls) == 2
    repaired = QuestionAction.model_validate(result.raw_response["question_action"])
    assert all(r.disposition == "not_useful" for r in repaired.continuation_assessment.routes)
    assert repaired.stopping_assessment == request.stopping_assessment
    assert result.portfolio == case.portfolio and case.model_dump_json() == before
