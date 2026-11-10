"""Lossless input views, separate from scientific selection and output schemas."""

from copy import deepcopy
import json

import pytest

from mechcal.runtime.context_compaction import (
    DATA_FIELDS, ENCODING, REF, SHARED, TABLE, compact_payload, expand_payload,
)
from mechcal.runtime.retry import complete_recorded_json
from mechcal.runtime.structured_output import request_payload


def observations():
    return {"states": [{"state_index": i, "energy_eV": 3.1 + i / 10,
                        "oscillator_strength": 0.001 * i, "unit": "eV"} for i in range(40)],
            "conditions": {"method": "synthetic test only", "distance_A": 10.0},
            "limitations": ["No experimental mechanism validation"], "failed_members": [2, 3]}


def payload():
    values = observations()
    return {"stage": "worker_result_explanation", "smiles": "CC",
            "tool_execution_result": {
                "structured_results": {"parsed_observables": deepcopy(values)},
                "evidence_units": [{"evidence_id": "E001", "status": "partial",
                                    "metrics": {"computed_observables": deepcopy(values)}}]},
            "output_contract": {"type": "object", "properties": {"finding": {"type": "string"}}}}


def test_repeated_results_are_stored_once_with_exact_round_trip():
    before = payload()
    original = deepcopy(before)
    after = compact_payload(before)
    assert after[ENCODING]["version"] == "lossless_data_v1"
    assert after["tool_execution_result"]["structured_results"]["parsed_observables"] == \
        after["tool_execution_result"]["evidence_units"][0]["metrics"]["computed_observables"]
    assert expand_payload(json.loads(json.dumps(after))) == original
    assert before == original and len(json.dumps(after)) < len(json.dumps(before))
    assert after["output_contract"] == before["output_contract"]


def test_nearly_equal_sources_are_not_merged_and_conditions_and_failure_survive():
    before = payload()
    second = before["tool_execution_result"]["evidence_units"][0]["metrics"]["computed_observables"]
    second["conditions"]["distance_A"] = 3.3
    second["limitations"].append("Missing matched analysis")
    second["states"][1]["energy_eV"] += 0.000001
    after = compact_payload(before)
    assert expand_payload(after) == before
    assert expand_payload(after)["tool_execution_result"]["evidence_units"][0]["status"] == "partial"


@pytest.mark.parametrize("field", sorted(DATA_FIELDS))
def test_each_enabled_data_field_is_reversible(field):
    before = {field: observations(), "output_contract": {"enum": ["schema must not change"]}}
    assert expand_payload(compact_payload(before)) == before


def test_scalar_table_preserves_indices_null_negative_zero_and_column_names():
    rows = [{"atom_index": i, "signed_charge": -0.0 if i == 0 else -i / 17,
             "missing": None, "valid": False, "element": "Au", "units": "electron_charge"}
            for i in range(50)]
    before = {"observations": rows}
    after = compact_payload(before)
    assert TABLE in after["observations"]
    restored = expand_payload(after)
    assert json.dumps(restored) == json.dumps(before)


def test_ragged_table_does_not_invent_missing_values():
    rows = [{"atom_index": i, "charge": i / 100} for i in range(40)]
    rows[2].pop("charge")
    before = {"observations": rows}
    after = compact_payload(before)
    assert after == before


@pytest.mark.parametrize("marker", [REF, TABLE])
def test_existing_scientific_markers_are_not_interpreted(marker):
    before = payload()
    before["tool_execution_result"][marker] = {"original": "not our encoding"}
    assert compact_payload(before) == before


def test_small_requests_are_not_inflated_and_projection_is_idempotent():
    before = {"observations": [{"x": 1}]}
    assert compact_payload(before) == before
    projected = compact_payload(payload())
    assert compact_payload(projected) == projected


def test_contracts_and_judgments_never_use_data_encoding():
    before = {"output_contract": observations(), "portfolio": observations(),
              "mechanism_support_arguments": observations(), "observations": observations()}
    after = compact_payload(before)
    for key in ("output_contract", "portfolio", "mechanism_support_arguments"):
        assert after[key] == before[key]
    assert expand_payload(after) == before


def test_no_shared_data_leaks_between_requests_or_cases():
    first = compact_payload(payload())
    other = payload()
    other["smiles"] = "NN"
    other["tool_execution_result"]["structured_results"]["parsed_observables"]["conditions"]["distance_A"] = 500
    second = compact_payload(other)
    assert expand_payload(first) == payload()
    assert expand_payload(second) == other


def test_invalid_or_cyclic_data_reference_fails_integrity_check():
    after = compact_payload(payload())
    key = next(iter(after[SHARED]))
    after[SHARED][key] = {REF: key}
    with pytest.raises(ValueError, match="Invalid request-local"):
        expand_payload(after)


class Client:
    def __init__(self):
        self.calls = []

    def complete_json(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return {"finding": "Synthetic output unchanged"}


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_request_and_log_match_and_response_and_schema_unchanged(enabled):
    client, events = Client(), []
    before = payload()
    fmt = {"type": "json_schema", "json_schema": {
        "name": "support_audit_v1", "strict": True, "schema": before["output_contract"]}}
    response = complete_recorded_json(client, system_prompt="test", payload=before,
                                     response_format=fmt, compact_context=enabled,
                                     schema_feedback="repair only this output", event_sink=events.append)
    expected = request_payload(before, fmt)
    emitted = client.calls[0]
    logged = next(e["request"] for e in events if e["kind"] == "model_request")
    assert expand_payload(logged["payload"]) == expected
    assert emitted["response_format"] == logged["response_format"] == fmt
    assert emitted["schema_feedback"] == "repair only this output"
    assert "compact_context" not in emitted
    assert response == {"finding": "Synthetic output unchanged"} and len(client.calls) == 1
    if enabled:
        assert logged["payload"] == emitted["payload"]
        assert any(e["kind"] == "context_projection" for e in events)
    else:
        assert emitted["payload"] == before
        assert not any(e["kind"] == "context_projection" for e in events)


def test_worker_source_paths_still_point_to_original_fields_after_expansion():
    from mechcal.agents.worker_explanation import resolve_source_path

    before = payload()
    restored = expand_payload(compact_payload(before))["tool_execution_result"]
    paths = ["/structured_results/parsed_observables/states/2/energy_eV",
             "/evidence_units/0/metrics/computed_observables/conditions",
             "/evidence_units/0/metrics/computed_observables/limitations"]
    for path in paths:
        assert resolve_source_path(restored, path) == resolve_source_path(before["tool_execution_result"], path)
