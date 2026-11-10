"""Reproduce mixed fresh/stale batches and exercise model-authorized correction."""

from copy import deepcopy

import pytest

from mechcal.agents.mechanism_agenda import MechanismAgendaAgent
from mechcal.runtime.candidate_expansion import new_observations, proposal_payload, validate_proposals
from mechcal.runtime.questions import replace_state, state
from mechcal.runtime.retry import RequestValidationError, SchemaRetryError
from mechcal.schemas.candidate_expansion import CandidateExpansion
from test_candidate_expansion import case_with_result, proposal


def mixed():
    case = case_with_result()
    old = case.evidence_ledger.items[-1].model_copy(update={"evidence_id": "E0"})
    case = case.touch(evidence_ledger=case.evidence_ledger.with_items([old]))
    previous = deepcopy(state(case)["results"][0])
    previous.update(result_id="X0", evidence_refs=["E0"])
    previous["structured_results"]["parsed_observables"]["method"] = "previous fixture"
    case = replace_state(case, results=[previous, *state(case)["results"]])
    batch = proposal()
    stale = proposal("E0").proposals[0]
    stale.hypothesis = "Old-only fixture explanation, not new evidence."
    batch.proposals.append(stale)
    return case, batch


class ScriptedClient:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def is_configured(self):
        return True

    def complete_json(self, *, payload, **kwargs):
        self.calls.append(deepcopy(payload))
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        return value(payload) if callable(value) else deepcopy(value)


def withdraw_invalid(payload):
    return {"repairs": [
        {"path": ["proposals"], "value": payload["retained_proposals"]},
        {"path": ["rationale"], "value": "The old-only item is explicitly withdrawn from this new-observation batch."},
    ]}


def run(case, outputs):
    client = ScriptedClient(outputs)
    agent = MechanismAgendaAgent(llm_client=client)
    events = []
    result = agent.propose_candidates(case, new_observations(case), event_sink=events.append)
    return result, client, events


def test_real_failure_shape_reports_exact_index_not_invalid_or_fabricated_source():
    c, batch = mixed()
    with pytest.raises(RequestValidationError) as err:
        validate_proposals(batch, proposal_payload(c, new_observations(c)))
    issue = err.value.issues[0]
    assert issue["proposal_index"] == 1
    assert issue["code"] == "candidate_no_new_observation"
    assert issue["cited_refs"] == ["E0"]
    assert issue["required_new_observation_refs"] == ["E1"]
    assert len(err.value.issues) == 1


def test_model_can_withdraw_only_invalid_item_without_losing_valid_one():
    c, batch = mixed()
    before = c.model_dump(mode="json")
    result, client, events = run(c, [batch.model_dump(mode="json"), withdraw_invalid])
    assert result.proposals == [batch.proposals[0]]
    assert c.model_dump(mode="json") == before
    assert len(client.calls) == 2
    repair = client.calls[1]
    assert repair["stage"] == "candidate_expansion_repair"
    assert "current_judgment" not in repair
    assert repair["retained_proposals"] == [batch.proposals[0].model_dump(mode="json")]
    assert repair["rejected_response"]["proposals"][1]["evidence_refs"] == ["E0"]
    assert any(e["kind"] == "field_repair_applied" for e in events)


def test_model_may_ground_second_proposal_in_new_data_not_be_forced_to_delete():
    c, batch = mixed()
    def correct(payload):
        items = deepcopy(payload["rejected_response"]["proposals"])
        items[1].update(hypothesis="Reinterpreted fixture using old and new data.",
                        observation_basis="Explicit fixture comparison, not a scientific finding.",
                        evidence_refs=["E0", "E1"])
        return {"repairs": [{"path": ["proposals"], "value": items}]}
    result, _, _ = run(c, [batch.model_dump(mode="json"), correct])
    assert len(result.proposals) == 2
    assert result.proposals[0] == batch.proposals[0]
    assert result.proposals[1].evidence_refs == ["E0", "E1"]


@pytest.mark.parametrize("mode", ["drop_valid", "change_valid", "wrong_format", "repeat_stale"])
def test_bad_correction_retains_original_and_uses_same_three_attempt_allowance(mode):
    c, batch = mixed()
    def bad(payload):
        if mode == "wrong_format":
            return payload["rejected_response"]
        if mode == "repeat_stale":
            items = payload["rejected_response"]["proposals"]
        elif mode == "drop_valid":
            items = []
        else:
            items = deepcopy(payload["retained_proposals"])
            items[0]["hypothesis"] = "Unauthorized rewrite of the valid explanation."
        return {"repairs": [{"path": ["proposals"], "value": items}]}
    result, client, _ = run(c, [batch.model_dump(mode="json"), bad, withdraw_invalid])
    assert len(client.calls) == 3
    assert result.proposals == [batch.proposals[0]]
    assert client.calls[2]["retained_proposals"] == [batch.proposals[0].model_dump(mode="json")]


def test_duplicate_explanation_can_be_removed_without_locking_both_items():
    c, batch = mixed()
    batch.proposals[1] = batch.proposals[0].model_copy(deep=True)
    batch.proposals[1].evidence_refs = ["E0", "E1"]
    result, client, _ = run(c, [batch.model_dump(mode="json"), withdraw_invalid])
    assert result.proposals == [batch.proposals[0]]
    assert len(client.calls[1]["retained_proposals"]) == 1


def test_empty_batch_requires_explicit_model_withdrawal_not_a_program_fallback():
    c, batch = mixed()
    batch.proposals = [batch.proposals[1]]
    result, client, _ = run(c, [batch.model_dump(mode="json"), withdraw_invalid])
    assert result.proposals == []
    assert "withdrawn" in result.rationale
    invalid = batch.model_dump(mode="json")
    failing = ScriptedClient([invalid, invalid, invalid])
    with pytest.raises(SchemaRetryError):
        MechanismAgendaAgent(llm_client=failing).propose_candidates(c, new_observations(c))
    assert len(failing.calls) == 3


def test_transport_or_program_failure_is_not_retried_as_bad_proposal():
    c, _ = mixed()
    for error in (ConnectionError("fixture transport failure"), RuntimeError("fixture program failure")):
        client = ScriptedClient([error])
        with pytest.raises(type(error)):
            MechanismAgendaAgent(llm_client=client).propose_candidates(c, new_observations(c))
        assert len(client.calls) == 1


def test_clean_batch_does_not_add_a_model_call():
    c, batch = mixed()
    batch.proposals = batch.proposals[:1]
    result, client, _ = run(c, [batch.model_dump(mode="json")])
    assert result == batch
    assert len(client.calls) == 1
