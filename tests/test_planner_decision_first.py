"""Direct decisions and focused reads preserve source identity and science boundaries."""

from copy import deepcopy

import pytest

from mechcal.agents.planner_references import PlannerReferences
from mechcal.capabilities.discovery import ToolDiscoverySession
from mechcal.runtime.case_memory import pointer, serialized
from mechcal.runtime.context_compaction import expand_payload
from mechcal.runtime.delta_context import decision_context
from mechcal.runtime.delta_session import read_needs_review
from mechcal.runtime.memory_findings import findings_record
from mechcal.runtime.memory_views import retrieved_working_set
from mechcal.runtime.planner_memory import PlannerMemorySession
from mechcal.runtime.retry import RejectedResponseError
from mechcal.runtime.structured_output import PlannerOutputContract
from mechcal.schemas.memory import MemoryQuery
from test_delta_context import DONE, READY, WireClient, setup, query
from test_planner_working_state import scientific_payload
from test_question_loop import TOOL


def read_plan(*queries, **kwargs):
    return {**READY[0], "memory_queries": list(queries), **kwargs}, "context_read"


def test_first_call_can_decide_without_preparation_or_source_mutation(tmp_path):
    case, _, memory = setup(tmp_path, start_lookup=False)
    original = case.model_dump(mode="json")
    client = WireClient([DONE])
    assert memory.complete(client, PlannerReferences(case), payload=memory.payload, system_prompt="unused") == DONE[0]
    assert len(client.calls) == 1 and memory.used == 0
    assert client.calls[0]["payload"]["stage"] == "planner_question_act"
    assert case.model_dump(mode="json") == original


def test_direct_read_then_decide_has_no_ready_only_call(tmp_path):
    case, _, memory = setup(tmp_path, start_lookup=False)
    client = WireClient([read_plan(query(record_ids=["Q1"], paths=["/question"])[0]), DONE])
    memory.complete(client, PlannerReferences(case), payload=memory.payload, system_prompt="unused")
    assert [c["payload"]["stage"] for c in client.calls] == ["planner_question_act"] * 2
    assert memory.used == 1
    assert "/question" in serialized(expand_payload(client.calls[-1]["payload"])["retrieved_memory"])


def test_direct_failed_read_remains_visible_and_can_be_corrected(tmp_path):
    case, _, memory = setup(tmp_path, start_lookup=False)
    client = WireClient([read_plan(query(record_ids=["missing"])[0]),
                         read_plan(query(record_ids=["Q1"], paths=["/question"])[0]), DONE])
    memory.complete(client, PlannerReferences(case), payload=memory.payload, system_prompt="unused")
    assert "unknown_record" in serialized(client.calls[1]["payload"])
    assert memory.used == 2 and len(client.calls) == 3
    assert all(c["payload"]["stage"] == "planner_question_act" for c in client.calls)


def test_partial_read_can_be_used_with_explicit_unread_boundary(tmp_path):
    case, _, memory = setup(tmp_path, start_lookup=False)
    payload = {**memory.payload, "table": list(range(10000)), "temperature": None}
    def read(p):
        return read_plan(MemoryQuery(action="read_case_records", record_ids=[p["context_record_id"]],
                         paths=["/table", "/temperature"], view="full").model_dump(mode="json"))[0]
    client = WireClient([(read, "context_read"), DONE])
    memory.complete(client, PlannerReferences(case), payload=payload, system_prompt="unused")
    response = expand_payload(client.calls[-1]["payload"])["retrieved_memory"][0]
    assert response["records"][0]["record"]["deferred_paths"] == ["/table"]
    assert response["records"][0]["record"]["selected_fields"]["/temperature"] is None
    assert memory.used == 1


def test_empty_direct_read_cannot_loop_without_spending_allowance(tmp_path):
    case, _, memory = setup(tmp_path, start_lookup=False)
    with pytest.raises(RejectedResponseError, match="makes no progress"):
        memory.complete(WireClient([read_plan()]), PlannerReferences(case), payload=memory.payload, system_prompt="unused")


def test_direct_resume_retains_loaded_data_and_query_budget(tmp_path):
    case, store, memory = setup(tmp_path, start_lookup=False)
    with pytest.raises(StopIteration):
        memory.complete(WireClient([read_plan(query(record_ids=["Q1"])[0])]), PlannerReferences(case),
                        payload=memory.payload, system_prompt="unused")
    resumed = PlannerMemorySession(case, store, memory.registry, memory.stage, memory.payload)
    client = WireClient([DONE])
    resumed.complete(client, PlannerReferences(case), payload=memory.payload, system_prompt="unused")
    assert resumed.used == 1 and len(client.calls) == 1
    assert "Q1" in serialized(expand_payload(client.calls[0]["payload"])["retrieved_memory"])


def test_direct_read_plan_loads_tool_contract_without_dispatching(tmp_path):
    case, store, memory = setup(tmp_path, start_lookup=False)
    discovery = ToolDiscoverySession(memory.registry, case, "R001")
    client = WireClient([read_plan(describe_capability_ids=[TOOL], focus_capability_ids=[TOOL]), DONE])
    memory.complete(client, PlannerReferences(case), payload=memory.payload, system_prompt="unused", discovery=discovery)
    assert discovery.batches_used == 1 and memory.used == 0
    assert [c["capability_id"] for c in client.calls[-1]["payload"]["capability_cards"]] == [TOOL]


def test_argument_observations_and_limits_are_retained_for_all_labels():
    source = scientific_payload()
    base = source["previous_support_arguments"][0]
    source["previous_support_arguments"] = [
        {**base, "support_id": f"S{i}", "target_label": f"M{i}",
         "finding": f"Value {i} eV at fixed geometry", "boundary": "No solvent evidence"}
        for i in range(12)
    ]
    result = decision_context(source, "context:x")
    assert len(result["previous_support_arguments"]) == 12
    for before, after in zip(source["previous_support_arguments"], result["previous_support_arguments"], strict=True):
        assert after["finding"] == before["finding"]
        assert after["boundary"] == before["boundary"]
        assert after["observation_refs"] == before["observation_refs"]
    assert result["current_working_state"]["kind"] == "accepted_judgment_not_evidence"


def test_read_branch_does_not_change_scientific_output_schema(tmp_path):
    _, _, memory = setup(tmp_path, start_lookup=False)
    direct = PlannerOutputContract(decision_context(memory.augment(memory.payload), "context:x"))
    old = PlannerOutputContract({"output_contract": memory.payload["output_contract"]})
    assert direct.logical["decision"] == old.logical["decision"]
    request = read_plan(query(record_ids=["Q1"])[0])[0]
    assert direct.decode(direct.encode(request, kind="context_read")) == request


def finding_source():
    return {
        "result_id": "exec:1", "status": "partial", "question_ids": ["Q2"],
        "failure": {"message": "One part failed"},
        "worker_explanation": {"findings": [
            {"finding_id": f"F{i}", "observation": f"{i} eV", "conditions": "S1 fixed geometry",
             "interpretation": "Conditional interpretation", "limitations": "No emission result",
             "evidence_refs": ["E1"], "source_paths": ["/structured_results/value"]}
            for i in range(3)]},
        "explanation_review": {"status": "reviewed", "assessment": {"finding_audits": [
            {"finding_id": "F0", "status": "correction_required", "correction": "Environment unspecified"}]}},
    }


def test_findings_view_keeps_whole_units_with_review_and_source_links():
    source = finding_source()
    original = deepcopy(source)
    result = findings_record(source, offset=0, budget=12000)
    assert result["next_offset"] is None
    for path, value in result["selected_fields"].items():
        assert pointer(source, path) == value
    assert result["selected_fields"]["/explanation_review"] == source["explanation_review"]
    assert result["source_context"]["failure"] == source["failure"]
    assert source == original


def test_findings_pagination_preserves_corrections_on_each_page():
    source = finding_source()
    for f in source["worker_explanation"]["findings"]:
        f["observation"] += "x" * 4000
    pages = []
    offset = 0
    while True:
        page = findings_record(source, offset=offset, budget=6000)
        assert len(serialized(page)) < 6000
        assert page["selected_fields"]["/explanation_review"] == source["explanation_review"]
        assert len([p for p in page["selected_fields"] if p.startswith("/worker_explanation/")]) == 1
        pages.append({"records": [{"record_id": "X", "record": page}]})
        if page["next_offset"] is None:
            break
        assert read_needs_review(pages[-1])
        offset = page["next_offset"]
    merged = retrieved_working_set(pages)
    values = {k: v for response in merged for row in response["records"] for k, v in row["record"]["selected_fields"].items()}
    for i, f in enumerate(source["worker_explanation"]["findings"]):
        assert values[f"/worker_explanation/findings/{i}"] == f


def test_oversized_finding_cannot_lose_its_review_to_fit():
    source = finding_source()
    source["explanation_review"]["reason"] = "x" * 13000
    result = findings_record(source, offset=0, budget=12000)
    assert result["selected_fields"] == {}
    assert result["error"] == "finding_with_review_exceeds_budget"
    assert result["next_offset"] == 0


def test_absent_findings_are_not_synthetic_evidence():
    result = findings_record({"metrics": {"value": 0}}, offset=0, budget=12000)
    assert result["error"] == "no_stored_findings" and result["selected_fields"] == {}


@pytest.mark.parametrize("bad", [{"start": 2}, {"paths": ["/worker_explanation"]}])
def test_findings_view_uses_unambiguous_pagination(bad):
    with pytest.raises(ValueError):
        MemoryQuery(action="read_case_records", record_ids=["X"], view="findings", **bad)
