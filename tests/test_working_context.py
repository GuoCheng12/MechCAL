"""Working-set boundaries: synthetic fixtures, never scientific evaluations."""

from copy import deepcopy
import json

import pytest

from mechcal.agents.mechanism_agenda import _coverage_payload
from mechcal.agents.photophysics_arbiter import PhotophysicsArbiterAgent
from mechcal.capabilities import default_capability_registry
from mechcal.capabilities.discovery import ToolDiscoverySession, case_catalog
from mechcal.runtime.case_memory import CaseMemoryStore, serialized
from mechcal.runtime.planner_context import build_context
from mechcal.runtime.questions import replace_state, state
from mechcal.runtime.structured_output import PlannerOutputContract, request_payload
from mechcal.schemas.memory import MemoryQuery
from mechcal.schemas.question_loop import QuestionAction
from test_case_memory import fixture
from test_tool_discovery import MACRO, BASELINE, RECOVER, case as tool_case, query


def context(stage="planner_assimilate"):
    return {
        "stage": stage, "case_id": "fixture", "revision": 2,
        "portfolio": {"hypotheses": [{"name": "A", "confidence": 0.5, "evidence_refs": ["old"]}]},
        "mechanism_pool": ["A", "B"],
        "observations": [
            {"evidence_id": "old", "metrics": {"value": 1, "units": "eV"}, "limits": ["old conditions"]},
            {"evidence_id": "new", "metrics": {"value": 2, "units": "eV"}, "limits": ["new conditions"]},
            {"evidence_id": "counter", "claim": "Conflicting observation", "status": "partial"},
        ],
        "source_checks": [{"evidence_id": name} for name in ("old", "new", "counter")],
        "evidence_index": [{"evidence_id": name, "claim": name} for name in ("old", "new", "counter")],
        "previous_support_arguments": [{"support_id": "S1", "target_label": "A", "finding": "Old finding",
                                        "boundary": "Only original conditions", "observation_refs": ["old"],
                                        "warrant": "An existing explanation"}],
        "pending_results": [{"result_id": "X2", "evidence_refs": ["new"],
                             "structured_results": {"extra": "Only in full execution"},
                             "investigation_plan": {"possible_outcomes": ["Keep this prospective expectation"]},
                             "worker_explanation": {"findings": ["Keep this new interpretation"]}}],
        "comparison_context": {"enabled": True, "initial_judgment": {"revision": 1},
                               "latest_update": {"evidence_refs": ["old", "counter"]}, "current_order": ["A", "B"]},
        "argument_review": {"support_ids": ["S1"], "routing": {"mode": "local_audit"},
                            "reason": "Old review"},
        "argument_review_pending": False,
        "execution_history": [{"result_id": "X1", "capability_id": MACRO, "status": "failed",
                               "requested_parameters": {"seed": 123}, "failure": {"kind": "no_result"},
                               "delivery": {"missing": ["energy"]}, "evidence_refs": ["old"]}],
    }


def test_old_citations_do_not_repin_the_entire_ledger():
    payload = context()
    original = deepcopy(payload)
    projected = build_context(payload)
    assert projected["observations"] == [payload["observations"][1]]
    assert projected["source_checks"] == [{"evidence_id": "new"}]
    assert projected["evidence_index"] == payload["evidence_index"]  # Includes counterevidence.
    assert projected["portfolio"] == payload["portfolio"]
    assert "latest_update" not in projected["comparison_context"]
    assert projected["argument_review"]["memory_record_id"] == "review:argument"
    assert projected["previous_support_arguments"][0]["boundary"] == "Only original conditions"
    assert payload == original


def test_new_result_preserves_plan_interpretation_and_exact_evidence_once():
    payload = context()
    projected = build_context(payload)
    pending = projected["pending_results"][0]
    assert pending["investigation_plan"] == payload["pending_results"][0]["investigation_plan"]
    assert pending["worker_explanation"] == payload["pending_results"][0]["worker_explanation"]
    assert pending["raw_result_record_id"] == "X2"
    assert "structured_results" not in pending
    assert projected["observations"][0] == payload["observations"][1]


@pytest.mark.parametrize("refs", [[], ["not_in_observations"]])
def test_unrepresented_new_raw_result_is_not_dropped(refs):
    payload = context()
    payload["pending_results"][0]["evidence_refs"] = refs
    assert build_context(payload)["pending_results"][0]["structured_results"] == {"extra": "Only in full execution"}


def test_action_uses_accepted_judgment_and_execution_conditions_not_raw_history():
    payload = context("planner_question_act")
    projected = build_context(payload)
    assert projected["observations"] == []
    assert "pending_results" not in projected
    assert projected["unassimilated_result_ids"] == ["X2"]
    assert projected["execution_history"][0]["requested_parameters"] == {"seed": 123}
    assert projected["execution_history"][0]["failure"] == {"kind": "no_result"}
    assert projected["execution_history"][0]["delivery"] == {"missing": ["energy"]}
    assert projected["portfolio"] == payload["portfolio"]


@pytest.mark.parametrize("stage", ["planner_assimilate", "planner_final_revision"])
def test_active_audit_and_final_revision_retain_required_observations(stage):
    payload = context(stage)
    payload["argument_review_pending"] = True
    payload["final_internal_audit"] = {"evidence_refs": ["counter"]}
    projected = build_context(payload)
    assert projected["argument_review"] == payload["argument_review"]
    assert projected["previous_support_arguments"] == payload["previous_support_arguments"]
    expected = {"new", "old", "counter"} if stage == "planner_final_revision" else {"old", "new"}
    assert {e["evidence_id"] for e in projected["observations"]} == expected


def test_omitted_history_is_exactly_retrievable_and_case_scoped(tmp_path):
    case, store = fixture(tmp_path)
    test = {"revision": 1, "round": 1, "result_id": "X1", "plan": {"question": "Original plan"},
            "assessment": {"comparison_effect": "non_discriminating"}}
    execution = {"result_id": "X1", "evidence_refs": [], "structured_results": {"raw": 2.5}}
    responses = [{"candidate_id": "C1", "decision": "defer", "reason": "Full old reasoning"}]
    case = replace_state(case, hypothesis_test_history=[test], results=[execution],
                         candidate_expansion={"responses": responses}, argument_review={"reason": "Review"})
    index = CaseMemoryStore(store, case)
    for record_id, expected in {
        "test:1:X1": test, "X1": execution, "history:candidate_responses": {"responses": responses},
        "review:argument": {"reason": "Review"},
    }.items():
        row = index.read(MemoryQuery(action="read_case_records", record_ids=[record_id]))["records"][0]
        assert row["record"] == expected
    other, other_store = fixture(tmp_path / "other")
    assert CaseMemoryStore(other_store, other).read(MemoryQuery(
        action="read_case_records", record_ids=["test:1:X1"]))["records"][0]["error"] == "unknown_record"


def test_tool_working_set_is_not_all_loaded_tools_and_survives_resume():
    registry = default_capability_registry()
    current = tool_case()
    current.runtime["planner_case_memory"] = True
    lookup = ToolDiscoverySession(registry, current, "R001")
    lookup.query(query(MACRO, BASELINE, RECOVER))
    current.runtime["tool_discovery"] = lookup.state.model_dump(mode="json")
    next_round = ToolDiscoverySession(registry, current, "R002")
    payload = next_round.augment_payload({"questions": [
        {"answer_state": "open", "acceptable_capability_ids": [MACRO]},
        {"answer_state": "answered", "acceptable_capability_ids": [BASELINE]},
    ]})
    assert [c["capability_id"] for c in payload["capability_cards"]] == [MACRO]
    assert set(payload["tool_discovery"]["loaded_ids"]) == {MACRO, BASELINE, RECOVER}
    next_round.query(query(BASELINE))
    next_round.query(query(RECOVER))
    current.runtime["tool_discovery_budget"] = next_round.budget_snapshot()
    restored = ToolDiscoverySession(registry, current, "R002")
    assert restored.batches_used == 2
    assert set(restored.augment_payload({})["tool_discovery"]["visible_ids"]) == {MACRO, BASELINE, RECOVER}
    assert ToolDiscoverySession(registry, current, "R003").augment_payload({})["capability_cards"] == []


def test_catalog_keeps_all_routes_and_readiness_but_no_empty_boilerplate():
    registry = default_capability_registry()
    current = tool_case(enabled=False)
    before = {row["capability_id"]: row for row in case_catalog(registry, current)}
    current.runtime["planner_case_memory"] = True
    after = {row["capability_id"]: row for row in case_catalog(registry, current)}
    assert before.keys() == after.keys()
    for key, row in after.items():
        assert row["status"] == before[key]["status"]
        assert row.get("missing_inputs", []) == before[key]["missing_inputs"]
        assert row.get("reason") == before[key].get("reason") or not before[key].get("reason")
        assert "molecular_input" not in row


def test_coverage_keeps_full_discoverability_without_every_detailed_tool_contract():
    registry = default_capability_registry()
    current = tool_case()
    current.runtime.update(planner_case_memory=True, decision_value_checks=True,
                           question_loop={"questions": {}, "pending_results": []})
    payload = _coverage_payload(current, registry, "R002", previous_planner_action=None)
    assert payload["capability_cards"] == case_catalog(registry, current)
    assert payload["tool_delivery_contracts"] == []


def test_wire_schema_remains_exact_while_duplicate_user_schema_is_omitted():
    payload = {"output_contract": QuestionAction.model_json_schema(), "revision": 2,
               "memory_access": {"queries_remaining": 4, "query_schema": MemoryQuery.model_json_schema()}}
    contract = PlannerOutputContract(payload)
    wire = contract.response_format()
    before = deepcopy((payload, wire))
    actual = request_payload(payload, wire)
    assert isinstance(actual["output_contract"], str)
    assert "query_schema" not in actual["memory_access"]
    assert actual["memory_access"]["queries_remaining"] == 4
    assert (payload, wire) == before
    query_body = MemoryQuery(action="read_case_records", record_ids=["X1"]).model_dump(mode="json")
    assert contract.decode(contract.encode(query_body, kind="memory_query"))["record_ids"] == ["X1"]
    assert request_payload(payload, {"type": "json_object"}) == payload
    assert len(serialized(actual)) < len(serialized(payload)) / 10


def test_transport_sends_schema_once_and_logs_actual_field_sizes(tmp_path, monkeypatch):
    from openai.types.responses import Response
    from mechcal.runtime.llm import OpenAIJsonClient, OpenAICompatibleSettings
    from test_responses_api import response_body

    request = {"output_contract": QuestionAction.model_json_schema(), "revision": 2}
    wire = PlannerOutputContract(request).response_format()
    captured = []

    def complete(settings, args):
        captured.append(deepcopy(args))
        return Response.model_validate(response_body("{}", parallel_tool_calls=False, tool_choice="none", tools=[]))

    monkeypatch.setattr("mechcal.runtime.llm.create_completion", complete)
    monkeypatch.setenv("MECHCAL_LLM_TELEMETRY_PATH", str(tmp_path / "calls.jsonl"))
    client = OpenAIJsonClient(OpenAICompatibleSettings(
        base_url="https://unused.test", model="fixture", api_key="fixture", max_retries=0,
    ))
    assert client.complete_json(system_prompt="fixture", payload=request, response_format=wire) == {}
    assert captured[0]["response_format"] == wire
    body = captured[0]["messages"][1]["content"]
    assert isinstance(json.loads(body)["output_contract"], str)
    event = json.loads((tmp_path / "calls.jsonl").read_text())
    assert event["payload_json_bytes"] == len(body.encode())
    assert event["payload_field_bytes"]["revision"] == 1


def test_schema_only_memory_query_reaches_store_then_returns_decision(tmp_path, monkeypatch):
    from openai.types.responses import Response
    from mechcal.agents.planner_references import PlannerReferences
    from mechcal.runtime.llm import OpenAIJsonClient, OpenAICompatibleSettings
    from mechcal.runtime.planner_memory import PlannerMemorySession
    from mechcal.runtime.questions import public_payload
    from test_case_memory import REGISTRY
    from test_responses_api import response_body

    current, store = fixture(tmp_path)
    current.runtime["planner_strict_outputs"] = True
    before = current.model_dump_json()
    payload = public_payload(current, REGISTRY, stage="planner_assimilate")
    payload["output_contract"] = {"type": "object", "properties": {"done": {"type": "boolean"}},
                                  "required": ["done"], "additionalProperties": False}
    memory = PlannerMemorySession(current, store, REGISTRY, "planner_assimilate", payload)
    contract = PlannerOutputContract(memory.augment(memory.payload))
    query_body = MemoryQuery(action="read_case_records", record_ids=["Q1"]).model_dump(mode="json")
    replies = iter([contract.encode(query_body, kind="memory_query"), contract.encode({"done": True})])
    calls = []

    def complete(settings, args):
        calls.append(deepcopy(args))
        return Response.model_validate(response_body(
            json.dumps(next(replies)), parallel_tool_calls=False, tool_choice="none", tools=[],
        ))

    monkeypatch.setattr("mechcal.runtime.llm.create_completion", complete)
    client = OpenAIJsonClient(OpenAICompatibleSettings(
        base_url="https://unused.test", model="fixture", api_key="fixture", max_retries=0,
    ))
    response = memory.complete(client, PlannerReferences(current), payload=memory.payload, system_prompt="fixture")
    assert response == {"done": True} and len(calls) == 2 and memory.used == 1
    first, second = [json.loads(call["messages"][1]["content"]) for call in calls]
    assert "query_schema" not in first["memory_access"]
    assert second["retrieved_memory"][0]["records"][0]["record"]["question_id"] == "Q1"
    assert current.model_dump_json() == before


def test_result_reviewer_retains_all_numeric_sources_but_only_target_questions():
    from test_worker_explanation import case_with_explanation, review_payload
    from test_result_review_schema import Client

    current = case_with_explanation()
    client = Client([review_payload()])
    PhotophysicsArbiterAgent(llm_client=client).review_results(current)
    before = client.calls[0]["payload"]
    current.runtime["planner_case_memory"] = True
    client = Client([review_payload()])
    PhotophysicsArbiterAgent(llm_client=client).review_results(current)
    after = client.calls[0]["payload"]
    assert after["pending_results"] == before["pending_results"]
    expected_ids = {q for r in state(current)["pending_results"] for q in r["question_ids"]}
    assert {q["question_id"] for q in after["questions"]} == expected_ids
