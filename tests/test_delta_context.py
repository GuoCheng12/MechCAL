"""Offline contracts for opt-in lookup/decision separation; no scientific performance claims."""

from copy import deepcopy
from dataclasses import replace
import json
import hashlib

import pytest
from jsonschema import Draft202012Validator

from mechcal.agents.planner_references import PlannerReferences
from mechcal.capabilities.discovery import ToolDiscoverySession
from mechcal.runtime import OrchestratorConfig
from mechcal.runtime.case_memory import CaseMemoryStore, pointer, serialized
from mechcal.runtime.delta_context import decision_context, lookup_context
from mechcal.runtime.planner_memory import PlannerMemorySession
from mechcal.runtime.questions import public_payload, replace_state
from mechcal.runtime.retry import RejectedResponseError
from mechcal.runtime.structured_output import PlannerOutputContract
from mechcal.schemas.memory import MemoryQuery
from test_case_memory import fixture
from test_question_loop import REGISTRY, TOOL


class WireClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete_json(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        response, kind = next(self.responses)
        if callable(response):
            response = response(kwargs["payload"])
        # Return a strict wire envelope, without using the subject's output parser.
        schema = kwargs["response_format"]["json_schema"]["schema"]
        branch = None
        for variant in schema["properties"]["response"]["anyOf"]:
            if variant["properties"]["kind"]["enum"] == [kind]:
                branch = variant["properties"]["data"]
                break
        if branch is None:
            return {"response": {"kind": kind, "data": response}}
        def fill(value, node):
            if "$ref" in node:
                node = schema["$defs"][node["$ref"].split("/")[-1]]
            if "anyOf" in node:
                for choice in node["anyOf"]:
                    candidate = fill(value, choice)
                    if Draft202012Validator({**choice, "$defs": schema.get("$defs", {})}).is_valid(candidate):
                        return candidate
                return value  # Deliberately invalid fixtures stay invalid.
            if node.get("type") == "object" and isinstance(value, dict):
                return {k: fill(value.get(k), v) for k, v in node["properties"].items()}
            if node.get("type") == "array" and isinstance(value, list):
                return [fill(v, node["items"]) for v in value]
            if isinstance(value, dict) and node.get("type") == "string":
                return json.dumps(value)
            return value
        return {"response": {"kind": kind, "data": fill(response, branch)}}


def setup(tmp_path, *, start_lookup=True, **extra):
    case, store = fixture(tmp_path, planner_delta_context=True, planner_strict_outputs=True,
                          scoped_evidence_updates=True, incremental_judgment=True, **extra)
    payload = public_payload(case, REGISTRY, stage="planner_question_act")
    payload["output_contract"] = {
        "type": "object", "additionalProperties": False, "required": ["action"],
        "properties": {"action": {"type": "string", "enum": ["fixture_done"]}},
    }
    memory = PlannerMemorySession(case, store, REGISTRY, payload["stage"], payload)
    if start_lookup:
        # Legacy lookup tests exercise an explicitly reopened read session.
        # Production iterative decisions now start without this pending flag.
        reopen(memory, memory.payload)
    return case, store, memory


def reopen(memory, payload):
    key = hashlib.sha256(serialized(payload).encode()).hexdigest()
    memory.delta_lookup_pending[key] = None


def query(**fields):
    return MemoryQuery(action="read_case_records", **fields).model_dump(mode="json"), "memory_query"


READY = ({"action": "prepare_context", "memory_queries": [], "describe_capability_ids": [], "focus_capability_ids": []}, "decision")
DONE = ({"action": "fixture_done"}, "decision")


def test_opt_in_dependencies_and_default(tmp_path):
    assert not OrchestratorConfig(run_base_dir=tmp_path).planner_delta_context
    with pytest.raises(ValueError, match="requires case memory"):
        replace(OrchestratorConfig(run_base_dir=tmp_path), planner_delta_context=True)


def test_lookup_read_decide_does_not_execute_or_change_science(tmp_path):
    case, store, memory = setup(tmp_path)
    before = serialized(case.model_dump(mode="json"))
    client = WireClient([query(record_ids=["Q1"], paths=["/question"]), READY, DONE])
    result = memory.complete(client, PlannerReferences(case), payload=memory.payload, system_prompt="unused")
    assert result == DONE[0]
    assert [c["payload"]["stage"] for c in client.calls] == [
        "planner_context_lookup", "planner_context_lookup", "planner_question_act"]
    assert memory.used == 1
    assert serialized(case.model_dump(mode="json")) == before
    assert "selected_fields" in serialized(client.calls[-1]["payload"]["retrieved_memory"])
    assert all(c["response_format"]["type"] == "json_schema" for c in client.calls)
    assert all(c["payload"].get("observations") is None for c in client.calls[:2])


def test_context_read_is_exact_and_isolated_by_snapshot(tmp_path):
    case, store, memory = setup(tmp_path)
    full = {**memory.payload, "large_values": {"unit": "eV", "values": list(range(5000))}}
    rid = memory.index.register_context(full)
    result = memory.index.read(MemoryQuery(action="read_case_records", record_ids=[rid], paths=["/large_values/unit"]))
    assert result["records"][0]["record"]["selected_fields"]["/large_values/unit"] == "eV"
    assert memory.index.search(MemoryQuery(action="search_case_records", kind="context"))["total"] == 1
    new = CaseMemoryStore(store, replace_state(case, revision=case.runtime["question_loop"]["revision"] + 1))
    assert new.read(MemoryQuery(action="read_case_records", record_ids=[rid]))["records"][0]["error"] == "unknown_record"
    with pytest.raises(ValueError, match="snapshot changed"):
        memory.index.read(MemoryQuery(action="read_case_records", record_ids=[rid]))


def test_resume_ready_and_queries_do_not_reset_allowances(tmp_path):
    case, store, memory = setup(tmp_path)
    client = WireClient([query(record_ids=["Q1"]), READY, DONE])
    memory.complete(client, PlannerReferences(case), payload=memory.payload, system_prompt="unused")
    restored = PlannerMemorySession(case, store, REGISTRY, memory.stage, public_payload(case, REGISTRY, stage=memory.stage))
    restored_client = WireClient([DONE])
    restored.complete(restored_client, PlannerReferences(case), payload=memory.payload, system_prompt="unused")
    assert restored.used == 1 and len(restored_client.calls) == 1
    assert restored_client.calls[0]["payload"]["stage"] == "planner_question_act"


def test_unknown_record_and_exhausted_queries_are_explicit(tmp_path):
    case, _, memory = setup(tmp_path)
    responses = [query(record_ids=["unknown"])] * 4 + [READY, DONE]
    client = WireClient(responses)
    memory.complete(client, PlannerReferences(case), payload=memory.payload, system_prompt="unused")
    assert memory.used == 4
    assert "unknown_record" in serialized(client.calls[-1]["payload"])
    assert client.calls[-1]["payload"]["memory_access"]["queries_remaining"] == 0
    with pytest.raises(RejectedResponseError, match="exhausted"):
        memory.query(query(record_ids=["Q1"])[0])


def test_tool_focus_requires_loaded_definition(tmp_path):
    case, store, memory = setup(tmp_path)
    discovery = ToolDiscoverySession(REGISTRY, case, "R001")
    bad = WireClient([({**READY[0], "focus_capability_ids": [TOOL]}, "decision")])
    with pytest.raises(RejectedResponseError, match="Read tool definitions"):
        memory.complete(bad, PlannerReferences(case), payload=memory.payload, system_prompt="unused", discovery=discovery)
    client = WireClient([
        ({"action": "describe_tools", "capability_ids": [TOOL]}, "describe_tools"),
        ({**READY[0], "focus_capability_ids": [TOOL]}, "decision"), DONE,
    ])
    memory.complete(client, PlannerReferences(case), payload=memory.payload, system_prompt="unused", discovery=discovery)
    assert discovery.batches_used == 1


def test_repair_focus_never_loads_unseen_tools(tmp_path):
    case, store, memory = setup(tmp_path, start_lookup=False)
    discovery = ToolDiscoverySession(REGISTRY, case, "R001")
    discovery.query({"action": "describe_tools", "capability_ids": [TOOL]})
    before = discovery.budget_snapshot()
    payload = {**memory.payload, "stage": "planner_judgment_repair",
               "repair_capability_ids": [TOOL, "microscopic.run_baseline_bundle", "unknown"]}
    client = WireClient([DONE])
    memory.complete(client, PlannerReferences(case), payload=payload, system_prompt="repair", discovery=discovery)
    assert [c["capability_id"] for c in client.calls[0]["payload"]["capability_cards"]] == [TOOL]
    assert discovery.budget_snapshot() == before
    assert set(discovery.state.loaded_tools) == {TOOL}
    assert [x["capability_id"] for x in client.calls[-1]["payload"]["capability_cards"]] == [TOOL]


def test_projection_handles_are_exact_and_bounds_preserved():
    data = {
        "stage": "planner_assimilate", "case_id": "x", "revision": 3,
        "portfolio": {"hypotheses": [{"name": "A", "confidence": .1, "rationale": "prior only",
                                      "validation_needed": ["solvent identity"], "evidence_refs": ["E1"]}]},
        "questions": [{"question_id": "Q", "question": "Check an alternative", "answer_state": "open",
                       "observable": "complex quantity", "missing_or_unresolved": "missing unit"}],
        "observations": [{"evidence_id": "E1", "limits": ["not an experiment"], "context": {"unit": "eV"},
                          "metrics": {"states": list(range(1000)), "state": "S1"}}],
        "argument_review": {"status": "unavailable", "correction": "must not infer approval"},
        "boundary_decision_context": {"current_candidate_bases": [
            {"label": "A", "positive_refs": ["E1"], "counter_refs": ["E2"], "inference_gap": "unresolved"}]},
        "execution_history": [{"result_id": "X", "failure": "unsupported", "requested_parameters": {"state": "S1"}}],
    }
    before = deepcopy(data)
    projected = decision_context(data, "context:fixture")
    assert data == before
    assert projected["argument_review"] == data["argument_review"]
    assert projected["observations"][0]["limits"] == ["not an experiment"]
    assert projected["observations"][0]["context"] == {"unit": "eV"}
    assert projected["boundary_decision_context"]["current_candidate_bases"][0]["counter_refs"] == ["E2"]
    def inspect(value):
        if isinstance(value, dict):
            if "read_path" in value:
                original = pointer(data, value["read_path"])
                assert all(original[k] == v for k, v in value.items() if k in original)
            for child in value.values(): inspect(child)
        elif isinstance(value, list):
            for child in value: inspect(child)
    inspect(projected)
    lookup = lookup_context(data, "context:fixture")
    assert lookup["candidates"][0]["name"] == "A"
    assert "observations" not in lookup


def test_lookup_contract_cannot_accept_a_scientific_decision(tmp_path):
    _, _, memory = setup(tmp_path)
    contract = PlannerOutputContract(lookup_context(memory.augment(memory.payload), "context:test"))
    with pytest.raises(Exception):
        contract.encode(DONE[0])


def test_batched_read_proceeds_without_ready_only_call(tmp_path):
    case, _, memory = setup(tmp_path)
    client = WireClient([({**READY[0], "memory_queries": [query(record_ids=["Q1"], paths=["/question"])[0]]}, "decision"), DONE])
    memory.complete(client, PlannerReferences(case), payload=memory.payload, system_prompt="unused")
    assert len(client.calls) == 2 and memory.used == 1
    assert client.calls[-1]["payload"]["retrieved_memory"]


def test_read_plan_exceeding_allowance_executes_no_queries(tmp_path):
    case, _, memory = setup(tmp_path)
    for _ in range(3):
        memory.query(query(record_ids=["Q1"])[0])
    client = WireClient([({**READY[0], "memory_queries": [query(record_ids=["Q1"])[0]] * 2}, "decision")])
    with pytest.raises(RejectedResponseError, match="/response/data/memory_queries: At most 1 items; received 2"):
        memory.complete(client, PlannerReferences(case), payload=memory.payload, system_prompt="unused")
    assert memory.used == 3


def test_lookup_failure_remains_lookup_on_format_retry(tmp_path):
    case, _, memory = setup(tmp_path)
    discovery = ToolDiscoverySession(REGISTRY, case, "R001")
    bad = WireClient([({**READY[0], "focus_capability_ids": [TOOL]}, "decision")])
    with pytest.raises(RejectedResponseError):
        memory.complete(bad, PlannerReferences(case), payload=memory.payload, system_prompt="unused", discovery=discovery)
    fixed = WireClient([READY, DONE])
    memory.complete(fixed, PlannerReferences(case), payload=memory.payload, system_prompt="unused",
                    schema_feedback='{"error":"unloaded tool"}', discovery=discovery)
    assert fixed.calls[0]["payload"]["stage"] == "planner_context_lookup"


def test_reopen_context_and_resume_does_not_reset_limit(tmp_path):
    case, store, memory = setup(tmp_path)
    need = ({"action": "need_context", "reason": "Read the original question"}, "context_needed")
    interrupted = WireClient([READY, need])
    with pytest.raises(StopIteration):
        memory.complete(interrupted, PlannerReferences(case), payload=memory.payload, system_prompt="unused")
    restored = PlannerMemorySession(case, store, REGISTRY, memory.stage, public_payload(case, REGISTRY, stage=memory.stage))
    client = WireClient([({**READY[0], "memory_queries": [query(record_ids=["Q1"])[0]]}, "decision"), DONE])
    restored.complete(client, PlannerReferences(case), payload=memory.payload, system_prompt="unused")
    assert client.calls[0]["payload"]["stage"] == "planner_context_lookup"
    assert client.calls[0]["payload"]["lookup_goal"] == "Read the original question"
    again = WireClient([need, READY, need])
    with pytest.raises(RejectedResponseError, match="declared alternatives"):
        restored.complete(again, PlannerReferences(case), payload=memory.payload, system_prompt="unused")
    assert max(restored.delta_reopens.values()) == 2
    branches = again.calls[-1]["response_format"]["json_schema"]["schema"]["properties"]["response"]["anyOf"]
    assert all(branch["properties"]["kind"]["enum"] != ["context_needed"] for branch in branches)


def test_missing_batched_record_is_not_silent_success(tmp_path):
    case, _, memory = setup(tmp_path)
    client = WireClient([({**READY[0], "memory_queries": [query(record_ids=["missing"])[0]]}, "decision"), READY, DONE])
    memory.complete(client, PlannerReferences(case), payload=memory.payload, system_prompt="unused")
    assert client.calls[1]["payload"]["stage"] == "planner_context_lookup"
    assert "unknown_record" in serialized(client.calls[1]["payload"])


@pytest.mark.parametrize("view", ["auto", "outline"])
def test_missing_batched_field_stays_in_lookup(tmp_path, view):
    case, _, memory = setup(tmp_path)
    read = query(record_ids=["Q1"], paths=["/nonexistent_field"], view=view)[0]
    client = WireClient([({**READY[0], "memory_queries": [read]}, "decision"), READY, DONE])
    memory.complete(client, PlannerReferences(case), payload=memory.payload, system_prompt="unused")
    assert client.calls[1]["payload"]["stage"] == "planner_context_lookup"
    assert "nonexistent_field" in serialized(client.calls[1]["payload"])
    assert memory.used == 1


def test_stale_focused_definition_reenters_lookup_without_reset(tmp_path):
    case, _, memory = setup(tmp_path)
    discovery = ToolDiscoverySession(REGISTRY, case, "R001")
    focused = ({**READY[0], "describe_capability_ids": [TOOL], "focus_capability_ids": [TOOL]}, "decision")
    memory.complete(WireClient([focused, DONE]), PlannerReferences(case), payload=memory.payload,
                    system_prompt="unused", discovery=discovery)
    discovery.state.loaded_tools.clear()
    client = WireClient([focused, DONE])
    memory.complete(client, PlannerReferences(case), payload=memory.payload, system_prompt="unused", discovery=discovery)
    assert client.calls[0]["payload"]["stage"] == "planner_context_lookup"
    assert discovery.batches_used == 2


def test_artifact_aliases_are_reversible_and_reserve_existing_ids():
    from mechcal.agents.evidence_references import _rewrite
    from mechcal.agents.planner_references import artifact_aliases
    ids = ["FILE001", "geometry:" + "a" * 40, "geometry:" + "a" * 40 + ":child"]
    aliases = artifact_aliases(ids)
    assert "FILE001" not in aliases and len(aliases) == 2
    source = {"source_artifact_id": ids[1], "reason": f"Compare {ids[1]} with {ids[2]}.", "literal": "FILE001"}
    encoded = _rewrite(source, {v: k for k, v in aliases.items()})
    assert _rewrite(encoded, aliases) == source


def test_nested_table_and_shared_text_expand_exactly():
    from mechcal.runtime.context_compaction import compact_payload, expand_payload
    repeated = "An exact source condition, not generated prose. " * 12
    data = {"context_version": "planner_delta_v1", "candidates": [
        {"label": f"M{i}", "conditions": {"unit": "eV", "method": repeated}, "counter_refs": ["E2"]}
        for i in range(12)], "review": {"condition": repeated}}
    assert expand_payload(compact_payload(data)) == data


def test_schema_metadata_removal_preserves_named_properties():
    from mechcal.runtime.delta_context import compact_tool_card
    schema = {"title": "Example", "type": "object", "properties": {
        "title": {"type": "string", "title": "Display name"},
        "$schema": {"type": "string"}}, "required": ["title"]}
    result = compact_tool_card({"tool_arg_schema": schema})["tool_arg_schema"]
    assert result["properties"] == {"title": {"type": "string"}, "$schema": {"type": "string"}}
    assert result["required"] == ["title"]


def test_read_failure_detection_does_not_reinterpret_scientific_error_fields():
    from mechcal.runtime.delta_session import read_needs_review
    assert not read_needs_review({"records": [{"record": {"selected_fields": {
        "/metrics": {"error": 0.2, "unit": "eV"}}}}]})


def test_context_cannot_cross_cases(tmp_path):
    case, store, memory = setup(tmp_path)
    with pytest.raises(ValueError, match="another case"):
        memory.index.register_context({**memory.payload, "case_id": "OTHER"})
    other = case.model_copy(update={"case_id": "OTHER"}, deep=True)
    with pytest.raises(ValueError, match="store/case mismatch"):
        CaseMemoryStore(store, other)


def test_batched_and_dependent_reads_share_exact_wire_definition(tmp_path):
    _, _, memory = setup(tmp_path)
    contract = PlannerOutputContract(lookup_context(memory.augment(memory.payload), "context:test"))
    variants = contract.schema["properties"]["response"]["anyOf"]
    branch = next(v for v in variants if v["properties"]["kind"]["enum"] == ["memory_query"])
    assert branch["properties"]["data"] == {"$ref": "#/$defs/decision_MemoryQuery"}
    request = query(record_ids=["Q1"])[0]
    assert contract.decode(contract.encode(request, kind="memory_query")) == request


def test_real_planner_assimilation_preserves_unchanged_judgment(tmp_path):
    from test_incremental_judgment import prepared
    from test_judgment_commands import command, explanation_response
    from test_case_memory import Client
    from mechcal.agents.planner import PlannerAgent
    from mechcal.runtime.context_compaction import expand_payload
    from mechcal.runtime.persistence import RunStore
    from mechcal.schemas.judgment_commands import JudgmentCommand
    from mechcal.schemas.ranking_explanations import RankingExplanations

    case = prepared()
    case.runtime.update(planner_case_memory=True, planner_delta_context=True,
                        planner_strict_outputs=True, scoped_evidence_updates=True)
    before = case.model_dump(mode="json")
    update = PlannerOutputContract({"output_contract": JudgmentCommand.model_json_schema()}).encode(
        command(case).model_dump(mode="json"))
    def explain(payload):
        return PlannerOutputContract({"output_contract": RankingExplanations.model_json_schema()}).encode(
            explanation_response(expand_payload(payload)))
    client = Client([update, explain])
    agent = PlannerAgent(llm_client=client, enable_llm_decisions=True, capability_registry=REGISTRY)
    store = RunStore(tmp_path, case.case_id)
    store.initialize()
    result = agent.assimilate(case, memory_store=store)
    assert result.portfolio == case.portfolio
    assert case.model_dump(mode="json") == before
    assert [c["payload"]["stage"] for c in client.calls] == [
        "planner_assimilate", "planner_explain_ranking"]


def test_real_planner_action_uses_delta_protocol(tmp_path):
    from mechcal.agents.planner import PlannerAgent
    from mechcal.schemas.question_loop import QuestionAction
    from test_case_memory import Client
    from test_question_loop import action

    case, store, _ = setup(tmp_path)
    answer = action(case.case_id, case.runtime["question_loop"]["revision"])
    result = PlannerOutputContract({"output_contract": QuestionAction.model_json_schema()}).encode(answer.model_dump(mode="json"))
    client = Client([result])
    agent = PlannerAgent(llm_client=client, enable_llm_decisions=True)
    decision = agent.plan_initial(case, round_id="R001", memory_store=store)
    assert decision.action == "finalize"
    assert [c["payload"]["stage"] for c in client.calls] == ["planner_question_act"]
