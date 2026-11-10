"""Evidence view tests use synthetic values, not scientific acceptance results."""

from copy import deepcopy
import json

import pytest

from mechcal.agents.planner_references import PlannerReferences
from mechcal.runtime.case_memory import CaseMemoryStore, pointer, serialized
from mechcal.runtime.context_compaction import expand_payload
from mechcal.runtime.memory_continuation import continuation_context
from mechcal.runtime.memory_views import source_context
from mechcal.runtime.observation_context import observation_card, project_observations
from mechcal.runtime.planner_context import build_context
from mechcal.runtime.planner_memory import PlannerMemorySession
from mechcal.runtime.questions import public_payload
from mechcal.runtime.structured_output import PlannerOutputContract
from mechcal.schemas.evidence import EvidenceUnit
from mechcal.schemas.memory import MemoryQuery, StoredMetricFields
from test_case_memory import Client, REGISTRY, fixture
from test_scoped_context import reviewed_impact_context


def evidence(rid="old"):
    return {
        "evidence_id": rid, "claim": "Synthetic observation", "summary": "Not a scientific result",
        "context": "Fixture geometry/state only", "basis": "computed", "status": "partial",
        "support": "weakens", "relation": "challenges", "limits": ["Counterevidence remains visible"],
        "artifact_refs": ["geometry:fixture"],
        "metrics": {
            "units": "eV", "energy": 1.2345678901234567, "method_provenance": [{"method": "fixture"}],
            "tool_args": {"charge": 1, "state_root": 3}, "limitations": ["Only a fixed geometry"],
            "molecular_context": {"atoms": 3}, "trace/a~b": [[1.1, 2.2, 3.3]] * 1000,
        },
    }


def recover(index, rid, path):
    start, chunks = 0, []
    while True:
        row = index.read(MemoryQuery(action="read_case_records", record_ids=[rid], paths=[path],
                                     view="full", start=start))["records"][0]
        if "record" in row:
            return row["record"]
        chunks.append(row["json_fragment"])
        if row["next_start"] is None:
            return json.loads("".join(chunks))
        start = row["next_start"]


def test_card_preserves_conditions_counterevidence_and_exact_small_values():
    source = evidence()
    before = deepcopy(source)
    card = observation_card(source)
    for key, value in source.items():
        if key != "metrics":
            assert card[key] == value
    for key in source["metrics"]:
        if key != "trace/a~b":
            assert card["metrics"][key] == source["metrics"][key]
    meta = StoredMetricFields.model_validate(card["metrics_memory"])
    assert meta.omitted_paths == ["/metrics/trace~1a~0b"] and not meta.complete
    assert "trace/a~b" not in card["metrics"]
    assert source == before and observation_card(card) == card
    assert len(serialized(card)) < len(serialized(source)) / 5


@pytest.mark.parametrize("value", [{}, {"energy": 1.23456789, "unit": "eV"}, {"trace": [1, 2]}])
def test_small_metrics_are_unchanged(value):
    source = {**evidence(), "metrics": value}
    assert observation_card(source) == source


def test_large_context_fields_are_not_truncated_to_make_a_saving():
    source = evidence()
    source["metrics"]["limitations"] = ["A condition " * 2000]
    card = observation_card(source)
    assert card["metrics"]["limitations"] == source["metrics"]["limitations"]


@pytest.mark.parametrize("metrics", [None, [], "not an EvidenceUnit"])
def test_other_memory_record_kinds_do_not_require_observation_metrics(metrics):
    assert source_context({"status": "failed", "metrics": metrics}) == {"status": "failed"}


def test_bounded_directory_has_full_lookup_path_and_no_missing_record_handle():
    source = evidence()
    source["metrics"] = {f"table{i}": [1] * 1000 for i in range(40)}
    card = observation_card(source)
    meta = card["metrics_memory"]
    assert len(meta["omitted_paths"]) == 12 and meta["omitted_field_count"] == 40
    assert meta["directory_path"] == "/metrics"
    source.pop("evidence_id")
    assert observation_card(source) == source  # Cannot hide data without a stable record ID.


def test_impact_review_pins_brief_old_source_but_new_source_is_full_first_time():
    payload = reviewed_impact_context()
    payload["result_review"]["impact_review"]["affected_labels"] = ["B"]
    payload["observations"] = [evidence("source:B"), evidence("new")]
    before = deepcopy(payload)
    projected = build_context(payload)
    assert projected["observations"] == [observation_card(evidence("source:B")), evidence("new")]
    for key in ("portfolio", "boundary_decision_context", "result_review", "evidence_index"):
        if key != "portfolio":
            assert projected[key] == payload[key]
    assert payload == before
    continuing = continuation_context(projected, [])
    assert continuing["observations"] == [observation_card(row) for row in payload["observations"]]
    assert continuing["pending_results"] == projected["pending_results"]
    assert continuing["previous_support_arguments"] == projected["previous_support_arguments"]


@pytest.mark.parametrize("stage", ["planner_assimilate", "planner_final_revision"])
def test_audit_and_final_views_keep_failures_and_limits_without_bulk_metrics(stage):
    payload = reviewed_impact_context()
    payload.update(stage=stage, argument_review_pending=True,
                   final_internal_audit={"evidence_refs": ["old"]})
    payload["observations"] = [evidence("old"), evidence("new")]
    first = build_context(payload)
    assert first["observations"][0] == observation_card(evidence("old"))
    assert first["observations"][1] == evidence("new")
    after = continuation_context(first, [])
    assert after["observations"] == project_observations(payload["observations"])
    assert after["argument_review"] == first["argument_review"]


def test_legacy_disabled_and_nested_repair_context():
    inner = {"scoped_evidence_updates": True, "stage": "planner_assimilate", "observations": [evidence()]}
    wrapper = {"stage": "planner_judgment_repair", "scientific_context": inner,
               "candidate_command": {"evidence_refs": ["old"]}, "targets": ["hypotheses"]}
    before = deepcopy(wrapper)
    after = continuation_context(wrapper, [])
    assert after["scientific_context"]["observations"] == [observation_card(evidence())]
    assert after["candidate_command"] == wrapper["candidate_command"] and wrapper == before
    disabled = {**inner, "scoped_evidence_updates": False}
    assert continuation_context(disabled, []) == disabled


@pytest.mark.parametrize("repair", [False, True])
def test_real_query_loop_resume_and_repair_keep_exact_requested_values(tmp_path, repair):
    case, store = fixture(tmp_path, scoped_evidence_updates=True, planner_strict_outputs=True)
    # Replace only fixture evidence content. The real memory index remains the source.
    old = EvidenceUnit(**evidence(), round_id="R001", source_report_id="fixture-report",
                       agent_name="fixture", capability_id="fixture.tool", family="geometry_precondition")
    case = case.touch(evidence_ledger=case.evidence_ledger.model_copy(update={"items": [old]}))
    before = case.model_dump_json()
    payload = public_payload(case, REGISTRY, stage="planner_assimilate")
    payload["observations"] = [old.model_dump(mode="json")]
    payload["output_contract"] = {"type": "object", "properties": {"done": {"type": "boolean"}},
                                  "required": ["done"], "additionalProperties": False}
    memory = PlannerMemorySession(case, store, REGISTRY, payload["stage"], payload)
    # Exercise first exposure and two explicit reads, without the fixture's unrelated impact scope.
    active = {**memory.payload, "observations": [old.model_dump(mode="json")]}
    contract = PlannerOutputContract(memory.augment(active))
    rid = old.evidence_id
    queries = [MemoryQuery(action="read_case_records", record_ids=[rid], paths=[p]).model_dump()
               for p in ("/metrics/energy", "/metrics/trace~1a~0b/0")]
    client = Client([*[contract.encode(q, kind="memory_query") for q in queries], contract.encode({"done": True})])
    assert memory.complete(client, PlannerReferences(case), payload=active, system_prompt="fixture",
                           schema_feedback="fixture correction" if repair else None) == {"done": True}
    wires = [expand_payload(call["payload"]) for call in client.calls]
    assert ("trace/a~b" in wires[0]["observations"][0]["metrics"]) is not repair
    for wire in wires[1:]:
        assert "trace/a~b" not in wire["observations"][0]["metrics"]
    records = [row["record"] for response in wires[-1]["retrieved_memory"] for row in response["records"]]
    values = {p: v for record in records for p, v in record["selected_fields"].items()}
    assert values == {q["paths"][0]: pointer(old.model_dump(mode="json"), q["paths"][0]) for q in queries}
    assert all(r["source_context"]["/metrics/tool_args"] == evidence()["metrics"]["tool_args"] for r in records)
    # Reopening a session cannot make old metrics reappear or forget selected values.
    resumed = PlannerMemorySession(case, store, REGISTRY, payload["stage"], payload)
    assert resumed.used == 2
    resumed_client = Client([contract.encode({"done": True})])
    resumed.complete(resumed_client, PlannerReferences(case), payload=active, system_prompt="fixture")
    wire = expand_payload(resumed_client.calls[0]["payload"])
    assert "trace/a~b" not in wire["observations"][0]["metrics"] and len(wire["retrieved_memory"]) == 2
    complete = recover(memory.index, rid, "/metrics/trace~1a~0b")
    assert complete["selected_fields"]["/metrics/trace~1a~0b"] == evidence()["metrics"]["trace/a~b"]
    assert case.model_dump_json() == before
    # A separate run cannot read this observation's replacement metrics.
    other, other_store = fixture(tmp_path / "other", scoped_evidence_updates=True)
    original = CaseMemoryStore(other_store, other).read(MemoryQuery(
        action="read_case_records", record_ids=[rid], paths=["/metrics/energy"]))["records"][0]
    assert original["error"] == "unknown_record"
