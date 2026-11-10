from copy import deepcopy

from mechcal.agents.observation_view import observation_view, repair_payload


def test_only_large_coefficient_tables_are_projected_and_sources_preserved():
    rows = [{"from_orbital": n, "to_orbital": n + 100, "coefficient": -n / 1000} for n in range(100)]
    original = {"artifact_refs": ["artifact:bundle"], "evidence_id": "E1", "units": "eV",
                "states": [{"state_index": 1, "components": rows, "energy": 3.14}],
                "atom_contributions": list(range(100)), "limits": ["vertical only"]}
    before = deepcopy(original)
    result = observation_view(original)
    assert original == before
    assert result["atom_contributions"] == list(range(100))
    assert result["states"][0]["components"][0] == rows[-1]
    assert result["states"][0]["energy"] == 3.14
    assert result["states"][0]["component_table_view"]["omitted_rows"] == 92
    assert result["artifact_refs"] == original["artifact_refs"]
    assert result["limits"] == original["limits"]


def test_small_or_unknown_component_data_remains_unchanged():
    for value in ({"state_index": 1, "components": [{"x": n} for n in range(100)]},
                  {"state_index": 1, "components": []}, {"components": list(range(100))}):
        assert observation_view(value) == value


def test_repair_does_not_resend_context_or_duplicate_rejected_response():
    correction = {"rejected_response": {"portfolio": [1, 2]}, "issues": [{"changes": ["before", "after"]}]}
    context = {"case_id": "C", "observations": ["large"], "previous_support_arguments": ["large"],
               "evidence_index": [{"evidence_id": "E", "claim": "original finding"}],
               "execution_history": [{"result_id": "R", "evidence_refs": ["E"], "investigation_plan": "large"}]}
    result = repair_payload(context, correction)
    assert result["rejected_response"] == correction["rejected_response"]
    assert "observations" not in result and "previous_support_arguments" not in result
    assert result["execution_history"] == [{"result_id": "R", "evidence_refs": ["E"]}]
