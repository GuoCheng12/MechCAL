"""Local audit repairs must preserve accepted science, including on strict wire paths."""

from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from mechcal.agents.planner import PlannerAgent
from mechcal.runtime.judgment_commands import materialize_judgment
from mechcal.runtime.judgment_draft import JudgmentDraft
from mechcal.runtime.persistence import RunStore
from mechcal.runtime.questions import public_payload, replace_state, state, validate_assimilation
from mechcal.runtime.retry import RequestValidationError
from mechcal.runtime.schema_diagnostics import schema_issues
from mechcal.runtime.structured_output import PlannerOutputContract, provider_response_format
from mechcal.schemas.judgment_commands import JudgmentCommand
from test_delta_context import WireClient
from test_incremental_judgment import prepared
from test_judgment_commands import command
from test_question_loop import REGISTRY


def fixture():
    case = prepared()
    case.runtime.update(predecision_comparison=True, planner_strict_outputs=True,
                        planner_delta_context=True, scoped_evidence_updates=True)
    sid = case.mechanism_support_arguments[0].support_id
    case = replace_state(case, argument_review_pending=True,
                         argument_review={"support_ids": [sid]}, pending_results=[])
    request = command(case, scope="local_audit", audit_responses={sid: "Scope clarified."},
        mechanism_support_arguments=[case.mechanism_support_arguments[0].model_copy(
            update={"boundary": "Clarified scope for the offline contract test."})])
    payload = public_payload(case, REGISTRY, stage="planner_assimilate")
    payload["output_contract"] = JudgmentCommand.model_json_schema()
    return case, request, PlannerOutputContract(payload)


def damaged(request, case):
    bad = request.model_copy(deep=True)
    bad.ranking_update = request_ranking(case)
    # A valid saved object is still a replacement, not the null retention command.
    bad.ranking_update.boundary_comparisons = []
    bad.audit_responses["report-id-not-support-id"] = "Acknowledged report."
    return bad


def test_live_wire_and_runtime_reject_independent_scope_errors_together():
    case, request, contract = fixture()
    bad = damaged(request, case)
    wire = contract.encode(request.model_dump(mode="json"))
    body = wire["response"]["data"]
    body["ranking_update"] = bad.ranking_update.model_dump(mode="json")
    body["audit_responses"] = bad.audit_responses
    provider = provider_response_format(contract.response_format(), "https://api.deepseek.com")
    for schema in (contract.schema, provider["json_schema"]["schema"]):
        issues = [issue for error in Draft202012Validator(schema).iter_errors(wire)
                  for issue in schema_issues(error)]
        paths = [issue["path"] for issue in issues]
        assert ["response", "data", "ranking_update"] in paths
        assert ["response", "data", "audit_responses", "report-id-not-support-id"] in paths
    before = case.model_dump_json()
    with pytest.raises(RequestValidationError) as exc:
        materialize_judgment(case, bad)
    assert {tuple(i["path"]) for i in exc.value.issues} == {
        ("audit_responses",), ("ranking_update",)}
    draft = JudgmentDraft(case.case_id, state(case)["revision"], "planner_assimilate",
        JudgmentCommand, {"rejected_response": bad.model_dump(mode="json"),
                          "issues": exc.value.issues, "error": str(exc.value)})
    assert {tuple(t["path"]) for t in draft.targets.values()} == {
        ("ranking_update",), ("scope",), ("audit_responses",)}
    edits = [{"target_id": tid, "operation": "replace", "value":
              None if target["field"] == "ranking_update" else request.audit_responses}
             for tid, target in draft.targets.items() if target["field"] != "scope"]
    fixed = JudgmentCommand.model_validate(draft.apply(
        {"action": "repair_judgment", "draft_id": draft.draft_id, "edits": edits}))
    assert fixed.scope == "local_audit"
    update = materialize_judgment(case, fixed)
    validate_assimilation(update, case)
    assert update.portfolio == case.portfolio
    assert update.ranking_update.boundary_comparisons == request_ranking(case).boundary_comparisons
    assert update.mechanism_support_arguments[1:] == case.mechanism_support_arguments[1:]
    assert case.model_dump_json() == before


def request_ranking(case):
    from mechcal.schemas.evidence_comparison import RankingUpdate
    return RankingUpdate.model_validate(state(case)["assimilation"]["ranking_update"])


@pytest.mark.parametrize("change", ["no_review", "new_results", "foreign", "blank", "missing"])
def test_pending_review_scope_and_ids_are_published(change):
    case, request, _ = fixture()
    if change == "no_review":
        case = replace_state(case, argument_review_pending=False)
    elif change == "new_results":
        case = replace_state(case, pending_results=[{"result_id": "new-result", "evidence_refs": []}])
    elif change == "foreign":
        request.audit_responses["foreign"] = "Not a support ID."
    elif change == "blank":
        request.audit_responses = {key: "  " for key in request.audit_responses}
    else:
        request.audit_responses = {}
    payload = public_payload(case, REGISTRY, stage="planner_assimilate")
    payload["output_contract"] = JudgmentCommand.model_json_schema()
    contract = PlannerOutputContract(payload)
    assert not Draft202012Validator(contract.logical["decision"]).is_valid(request.model_dump(mode="json"))
    with pytest.raises(RequestValidationError):
        materialize_judgment(case, request)


def test_strict_memory_path_corrects_local_edit_without_separate_ranking_call(tmp_path):
    case, request, _ = fixture()
    store = RunStore(tmp_path, case.case_id)
    store.initialize()
    bad = request.model_dump(mode="json")
    bad["ranking_update"] = request_ranking(case).model_dump(mode="json")
    bad["audit_responses"]["report-id-not-support-id"] = "Not a support ID."

    def repair(payload):
        repair = payload
        edits = []
        for target in repair["targets"]:
            if target["field"] == "ranking_update":
                value = None
            elif target["operation"] == "remove":
                edits.append(dict(target_id=target["target_id"], operation="remove"))
                continue
            elif target["field"] == "audit_responses":
                value = request.audit_responses
            else:
                continue
            edits.append(dict(target_id=target["target_id"], operation="replace", value=value))
        return dict(action="repair_judgment", draft_id=repair["draft_id"], edits=edits)

    client = WireClient([(bad, "decision"), (repair, "decision")])
    client.is_configured = lambda: True
    # Its invalid fixtures are not wire-encoded automatically; emit all wire fields
    # from a valid template, then introduce only the two faults under test.
    complete = client.complete_json
    def respond(**kwargs):
        if not client.calls:
            original = deepcopy(client.responses)
            client.responses = iter([(request.model_dump(mode="json"), "decision")])
            wire = complete(**kwargs)
            client.responses = original
            next(client.responses)
            wire["response"]["data"]["ranking_update"] = bad["ranking_update"]
            wire["response"]["data"]["audit_responses"] = bad["audit_responses"]
            return wire
        return complete(**kwargs)
    client.complete_json = respond
    before = case.model_dump_json()
    agent = PlannerAgent(capability_registry=REGISTRY, llm_client=client, enable_llm_decisions=True)
    update = agent.assimilate(case, memory_store=store)
    assert len(client.calls) == 2
    assert all(c["payload"]["stage"] != "planner_explain_ranking" for c in client.calls)
    assert update.portfolio == case.portfolio
    assert update.ranking_update.boundary_comparisons == request_ranking(case).boundary_comparisons
    assert case.model_dump_json() == before


def test_historical_command_remains_readable():
    case, request, _ = fixture()
    historical = damaged(request, case)
    assert JudgmentCommand.model_validate_json(historical.model_dump_json()) == historical
