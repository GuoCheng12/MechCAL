"""Offline memory/flow tests. Synthetic observations are not scientific results."""

import json
import sqlite3
from copy import deepcopy

import pytest

from mechcal.agents.planner import PlannerAgent
from mechcal.agents.planner_references import PlannerReferences
from mechcal.capabilities.discovery import ToolDiscoverySession
from mechcal.runtime.case_memory import CaseMemoryStore, serialized
from mechcal.runtime.persistence import RunStore
from mechcal.runtime.planner_context import build_context
from mechcal.runtime.planner_memory import PlannerMemorySession
from mechcal.runtime.questions import public_payload, replace_state, state
from mechcal.runtime.retry import RejectedResponseError
from mechcal.schemas.memory import MemoryQuery
from test_question_loop import REGISTRY, TOOL, action, assimilation, ready_case


def fixture(tmp_path, **runtime):
    case = replace_state(ready_case(), round_index=1)
    case = case.touch(runtime={**case.runtime, "planner_case_memory": True, **runtime})
    store = RunStore(tmp_path, case.case_id)
    store.initialize()
    return case, store


def session(case, store, stage="planner_question_act"):
    return PlannerMemorySession(case, store, REGISTRY, stage, public_payload(case, REGISTRY, stage=stage))


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def is_configured(self):
        return True

    def complete_json(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        result = next(self.responses)
        return result(kwargs["payload"]) if callable(result) else result


def test_read_search_scope_and_index_rebuild(tmp_path):
    case, store = fixture(tmp_path)
    (store.root / "judge.json").write_text('{"rubric":"DO_NOT_RETRIEVE"}')
    (store.root / "hidden_reference.json").write_text('{"label":"DO_NOT_RETRIEVE"}')
    index = CaseMemoryStore(store, case)
    hits = index.search(MemoryQuery(action="search_case_records", query="rotor", kind="question"))
    assert hits["total"] == 1
    assert index.search(MemoryQuery(action="search_case_records", query="DO_NOT_RETRIEVE"))["total"] == 0
    assert index.read(MemoryQuery(action="read_case_records", record_ids=["../../judge.json"]))["records"][0]["error"]
    before = index.read(MemoryQuery(action="read_case_records", record_ids=["Q1"]))
    index.path.unlink()
    restored = CaseMemoryStore(store, case)
    assert restored.read(MemoryQuery(action="read_case_records", record_ids=["Q1"])) == before
    with pytest.raises(ValueError, match="mismatch"):
        CaseMemoryStore(RunStore(tmp_path, "other_case"), case)


def test_obsolete_or_rejected_judgments_not_current(tmp_path):
    case, store = fixture(tmp_path)
    store.write_json("questions/revision_0001.json", {
        "case_id": case.case_id, "revision": 1, "decision_assessment": {"current_best": "old interpretation"},
        "hidden_reference": "SECRET",
    })
    store.write_json("questions/revision_9999.json", {
        "case_id": case.case_id, "revision": 9999, "decision_assessment": {"current_best": "future"},
    })
    index = CaseMemoryStore(store, case)
    data = index.read(MemoryQuery(action="read_case_records", record_ids=["judgment:1"]))["records"][0]
    assert data["kind"] == "judgment" and "SECRET" not in serialized(data)
    assert index.search(MemoryQuery(action="search_case_records", query="future"))["total"] == 0


def test_snapshot_drift_and_corruption_are_explicit(tmp_path):
    case, store = fixture(tmp_path)
    index = CaseMemoryStore(store, case)
    changed = replace_state(case, revision=2)
    CaseMemoryStore(store, changed)
    with pytest.raises(ValueError, match="snapshot changed"):
        index.search(MemoryQuery(action="search_case_records"))
    index.path.write_bytes(b"not a database")
    with pytest.raises(sqlite3.DatabaseError):
        CaseMemoryStore(store, case)


def test_query_budget_duplicate_errors_and_resume(tmp_path):
    case, store = fixture(tmp_path)
    memory = session(case, store)
    original = case.model_dump(mode="json")
    query = {"action": "read_case_records", "record_ids": ["Q1"]}
    memory.query(query)
    memory.query(query)
    assert len(memory.loaded) == 1 and memory.used == 2
    memory.query({"action": "read_case_records", "record_ids": []})
    restored = session(case, store)
    assert restored.used == 3
    assert any(r.get("status") == "query_rejected" for r in restored.loaded.values())
    restored.query({"action": "read_case_records", "record_ids": ["unknown"]})
    with pytest.raises(RejectedResponseError, match="exhausted"):
        restored.query(query)
    assert case.model_dump(mode="json") == original
    assert session(case, store, stage="planner_assimilate").used == 0


def test_fragment_recovery_and_pagination(tmp_path):
    case, store = fixture(tmp_path)
    result = {"result_id": "X1", "status": "partial", "capability_id": TOOL,
              "structured_results": {"units": "eV", "trace": "1234567890" * 1800}}
    case = replace_state(case, results=[result], pending_results=[])
    index = CaseMemoryStore(store, case)
    first = index.read(MemoryQuery(action="read_case_records", record_ids=["X1"]))["records"][0]
    assert first["next_start"] and "json_fragment" in first
    second = index.read(MemoryQuery(action="read_case_records", record_ids=["X1"],
                                    start=first["next_start"]))["records"][0]
    assert json.loads(first["json_fragment"] + second["json_fragment"]) == result
    hits = index.search(MemoryQuery(action="search_case_records", limit=1))
    assert hits["next_offset"] == 1
    next_hits = index.search(MemoryQuery(action="search_case_records", limit=1, offset=1))
    assert hits["records"][0]["record_id"] != next_hits["records"][0]["record_id"]


def test_registry_preflight_not_execution(tmp_path):
    case, store = fixture(tmp_path, amesp_enabled=False)
    memory = session(case, store)
    response = memory.query({
        "action": "find_compatible_artifacts",
        "capability_id": "microscopic.analyze_hole_electron_from_bundle", "tool_args": {"state_root": 3},
    })
    assert response["capability_status"]["status"] in {"disabled", "missing_input"}
    assert not state(case)["results"]
    unknown = memory.query({"action": "find_compatible_artifacts", "capability_id": "nonexistent"})
    assert unknown["status"] == "query_rejected"


def test_projection_does_not_advertise_nonexistent_judgments():
    payload = {
        "stage": "planner_assimilate",
        "comparison_context": {
            "enabled": True, "initial_judgment": None, "latest_update": None,
            "read_only_audit": {"argument_snapshots": []},
        },
    }
    projected = build_context(payload)
    assert projected["comparison_context"] == payload["comparison_context"]
    payload["comparison_context"]["initial_judgment"] = {"revision": 1}
    payload["comparison_context"]["latest_update"] = {"revision": 2}
    projected = build_context(payload)
    assert projected["comparison_context"]["initial_judgment"]["memory_record_id"] == "judgment:initial"
    assert projected["comparison_context"]["read_only_audit"]["memory_record_id"] == "judgment:last_comparison"


def test_projection_preserves_current_state_without_mutation(tmp_path):
    case, store = fixture(tmp_path)
    payload = public_payload(case, REGISTRY, stage="planner_assimilate")
    payload["previous_worker_explanations"] = [{"text": "old report " * 500}]
    before = deepcopy(payload)
    projected = build_context(payload)
    assert len(serialized(projected)) < len(serialized(payload))
    assert projected["portfolio"] == payload["portfolio"]
    assert projected["pending_results"] == payload["pending_results"]
    assert projected["mechanism_pool"] == payload["mechanism_pool"]
    assert payload == before and "previous_worker_explanations" not in projected


def test_queries_precede_formal_assimilation_parsing(tmp_path):
    case, store = fixture(tmp_path)
    case = replace_state(case, phase="assimilate")
    client = Client([
        {"action": "read_case_records", "record_ids": ["Q1"]},
        lambda payload: assimilation(payload),
    ])
    planner = PlannerAgent(llm_client=client, enable_llm_decisions=True)
    update = planner.assimilate(case, memory_store=store)
    assert update.revision == state(case)["revision"] + 1
    assert len(client.calls) == 2
    assert client.calls[1]["payload"]["retrieved_memory"][0]["records"][0]["record"]["question_id"] == "Q1"
    assert state(case)["revision"] == 1 and not state(case)["results"]


def test_queries_precede_action_and_survive_formal_retry(tmp_path):
    case, store = fixture(tmp_path)
    good = action(case.case_id, state(case)["revision"]).model_dump(mode="json")
    client = Client([
        {"action": "read_case_records", "record_ids": ["Q1"]},
        {**good, "revision": 999},
        {"action": "read_case_records", "record_ids": ["Q1"]},
        good,
    ])
    planner = PlannerAgent(llm_client=client, enable_llm_decisions=True)
    decision = planner.plan_initial(case, round_id="R001", memory_store=store)
    assert decision.action == "finalize"
    assert client.calls[-1]["payload"]["memory_access"]["queries_remaining"] == 2


def test_final_stage_memory_loop_and_reference_aliases(tmp_path):
    case, store = fixture(tmp_path)
    memory = session(case, store, "planner_final_revision")
    refs = PlannerReferences(case)
    client = Client([{"action": "read_case_records", "record_ids": ["Q1"]}, {"final": "fixture"}])
    result = memory.complete(client, refs, payload=memory.payload, system_prompt="test")
    assert result == {"final": "fixture"}
    assert len(client.calls) == 2


def test_tool_discovery_and_history_queries_share_decision_not_allowance(tmp_path):
    case, store = fixture(tmp_path)
    discovery = ToolDiscoverySession(REGISTRY, case, "R001", store.write_tool_discovery_event)
    memory = session(case, store)
    client = Client([
        {"action": "describe_tools", "capability_ids": [TOOL]},
        {"action": "read_case_records", "record_ids": ["Q1"]},
        {"action": "finalize"},
    ])
    result = memory.complete(client, PlannerReferences(case), payload=memory.payload,
                             system_prompt="test", discovery=discovery)
    assert result["action"] == "finalize"
    assert discovery.batches_used == 1 and memory.used == 1
    assert len(client.calls) == 3


def test_exact_fields_include_conditions_and_missing_fields_are_explicit(tmp_path):
    case, store = fixture(tmp_path)
    result = {"result_id": "X1", "status": "partial", "structured_results": {
        "energy": 2.5, "unit": "eV", "state": 3, "geometry": list(range(9000)),
    }}
    case = replace_state(case, results=[result])
    index = CaseMemoryStore(store, case)
    data = index.read(MemoryQuery(
        action="read_case_records", record_ids=["X1"],
        paths=["/structured_results/energy", "/structured_results/unit",
               "/structured_results/state", "/absent"],
    ))["records"][0]["record"]
    assert data["selected_fields"]["/structured_results/energy"] == 2.5
    assert data["selected_fields"]["/structured_results/unit"] == "eV"
    assert data["selected_fields"]["/structured_results/state"] == 3
    assert "error" in data["selected_fields"]["/absent"]
    assert "geometry" not in serialized(data)


def test_actual_final_revision_can_query_before_formal_schema(tmp_path):
    from test_delivery_and_final_revision import final_fixture

    case, revision = final_fixture()
    case = replace_state(case, round_index=2, final_audit={"status": "unavailable"})
    case = case.touch(runtime={**case.runtime, "planner_case_memory": True})
    store = RunStore(tmp_path, case.case_id)
    store.initialize()
    client = Client([
        {"action": "read_case_records", "record_ids": ["E1"]},
        revision.model_dump(mode="json"),
    ])
    planner = PlannerAgent(llm_client=client, enable_llm_decisions=True)
    result = planner.revise_final_arguments(case, memory_store=store)
    assert result == revision and len(client.calls) == 2


def test_memory_is_opt_in_and_no_paths_are_implicitly_opened(tmp_path):
    case, store = fixture(tmp_path)
    planner = PlannerAgent()
    payload = public_payload(case, REGISTRY, stage="planner_question_act")
    with pytest.raises(ValueError, match="explicit"):
        planner._memory_session(case, None, payload)
    disabled = case.touch(runtime={**case.runtime, "planner_case_memory": False})
    assert planner._memory_session(disabled, None, payload) is None


def test_aliases_restore_original_evidence_ids_after_query(tmp_path):
    from test_runtime import _evidence_unit

    case, store = fixture(tmp_path)
    eid = "execution:0123456789abcdef0123:E001"
    observation = _evidence_unit(evidence_id=eid, round_id="R001", source_report_id="fixture",
                                 agent_name="macro", family="torsion_sensitivity", summary="test")
    case = case.touch(evidence_ledger=case.evidence_ledger.model_copy(update={"items": [observation]}))
    memory = session(case, store)
    client = Client([
        {"action": "read_case_records", "record_ids": ["E001"], "paths": ["/status"]},
        {"evidence_refs": ["E001"]},
    ])
    response = memory.complete(client, PlannerReferences(case), payload=memory.payload, system_prompt="test")
    assert response["evidence_refs"] == [eid]
    assert list(memory.loaded.values())[0]["records"][0]["record_id"] == eid
    assert client.calls[1]["payload"]["retrieved_memory"][0]["records"][0]["record_id"] == "E001"


def test_orchestrator_round_accounting_and_completed_resume_with_memory(tmp_path):
    from dataclasses import replace
    from test_question_loop import NOTE, QuestionClient, loop

    class QueryClient(QuestionClient):
        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            if payload["revision"] > 0 and not payload["retrieved_memory"]:
                return {"action": "read_case_records", "record_ids": ["Q1"]}
            return super().complete_json(payload=payload, system_prompt=system_prompt,
                                         schema_feedback=schema_feedback)

    orchestrator, _ = loop(tmp_path, discover=True)
    orchestrator.config = replace(orchestrator.config, planner_case_memory=True)
    client = QueryClient(discover=True)
    orchestrator.deps.planner.llm_client = client
    result = orchestrator.run(smiles="CCO", user_query=NOTE, case_id="fixture")
    assert result.status == "finalized"
    assert result.round_ids == ["R001", "R002"]
    assert result.runtime["worker_call_counts"]["macro"] == 1
    assert state(result)["phase"] == "done"
    queries = [json.loads(line) for line in
               (tmp_path / "fixture/history/memory_queries.jsonl").read_text().splitlines()]
    assert any(q["kind"] == "memory_query" for q in queries)
    count = len(client.calls)
    resumed = orchestrator.run(smiles="CCO", user_query=NOTE, case_id="fixture", resume=True)
    assert resumed == result and len(client.calls) == count
