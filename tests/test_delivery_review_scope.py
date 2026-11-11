"""Proofreading preserves scientific dependencies and exact source-bound facts."""

from copy import deepcopy
from dataclasses import replace

import pytest
from jsonschema import Draft202012Validator

from mechcal.runtime import command_delivery_review as dr
from mechcal.runtime import command_numeric as numeric
from mechcal.runtime.command_compact import expand_text
from mechcal.runtime.command_loop import _apply, _comment
from mechcal.runtime.command_memory import digest
from mechcal.runtime.command_operations import OperationRejected, protocol, put, record_index
from mechcal.schemas.commands import AmendFinal, RecordReview, SubmitFinalAfterReview, CompareEvidence
from test_computational_review import fixture, report
from test_delivery_recovery import science
from test_native_commands import NativeFake, reply


def reviewed_case(tmp_path):
    c, o, s = fixture(tmp_path)
    o.config = replace(o.config, command_compact_context=True, command_task_context=True)
    fake = NativeFake([reply("record_review", report().model_dump(exclude={"kind"}))])
    o.deps.scientific_reviewer.llm_client = fake
    c = _comment(o, c, s)
    assert protocol(c)["final_review"]["status"] == "available"
    return c, o, s


def amend(c, o, s):
    draft = protocol(c)["final_draft"]
    return _apply(o, c, AmendFinal(kind="amend_final", draft_id=draft["draft_id"], replacements=[
        {"old_text": draft["command"]["answer"], "new_text": "A locally corrected protocol-fixture answer."}]), s)


def complete_review(c, o, s):
    fake = NativeFake([reply("record_review", {"text": "Fixture review, not a scientific result.", "findings": []})])
    o.deps.scientific_reviewer.llm_client = fake
    after = _comment(o, c, s)
    return after, expand_text(fake.requests[0]["payload"]), fake.requests[0]


def test_text_edit_switches_to_local_review_without_resetting_scientific_advice(tmp_path):
    c, o, s = reviewed_case(tmp_path)
    before = science(c)
    c = amend(c, o, s)
    after, payload, request = complete_review(c, o, s)
    sections = payload["sections"]
    assert sections["delivery_task"]["mode"] == "proofread"
    assert not {"tool_catalog", "computational_review", "candidate_overview", "pending_experiments"} & sections.keys()
    assert payload["read_scope"]["columns"] == ["ref"]
    assert payload["removed_sections"] == []
    assert "describe_tools" not in {t["name"] for t in request["tools"]}
    completion = next(t["parameters"] for t in request["tools"] if t["name"] == "record_review")
    assert "next_computations" not in completion["properties"]
    assert protocol(after)["computational_review"] == protocol(c)["computational_review"]
    assert science(after) == before
    final = _apply(o, after, SubmitFinalAfterReview(kind="submit_final",
        draft_id=protocol(after)["final_draft"]["draft_id"], route_responses=[]), s)
    assert final.status == "finalized"


@pytest.mark.parametrize("mutation", ["arguments", "ranking", "input", "results", "tools", "exploration"])
def test_scientific_changes_restore_full_review(tmp_path, mutation):
    c, o, s = reviewed_case(tmp_path)
    c = amend(c, o, s)
    p = protocol(c)
    if mutation == "input":
        c = c.touch(input=c.input.model_copy(update={"smiles": "CCN"}))
    elif mutation == "tools":
        o.config = replace(o.config, enable_amesp=not o.config.enable_amesp)
        # The fixture's tool is unaffected by the Amesp switch; change the stored
        # computational basis explicitly to represent changed registry availability.
        c = put(c, computational_review={**p["computational_review"], "basis": "changed-registry"})
    elif mutation == "results":
        c = put(c, results={**p.get("results", {}), "new": {"capability_id": "fixture", "status": "failed"}})
    else:
        changed = deepcopy(p[mutation])
        if mutation == "arguments":
            next(iter(changed.values()))["text"] += " Different interpretation."
        elif mutation == "ranking":
            next(iter(changed.values()))["confidence"] = .12
        else:
            next(iter(changed["entries"].values()))["confidence"] = .12
        c = put(c, **{mutation: changed})
    fake = NativeFake([reply("record_review", report().model_dump(exclude={"kind"}))])
    o.deps.scientific_reviewer.llm_client = fake
    _comment(o, c, s)
    payload = expand_text(fake.requests[0]["payload"])
    assert "delivery_task" not in payload["sections"]
    assert "tool_catalog" in payload["sections"] and "computational_review" in payload["sections"]


def test_repeated_local_review_retains_scientific_anchor_and_current_draft_gate(tmp_path):
    c, o, s = reviewed_case(tmp_path)
    original = protocol(c)["review_memory"]["final_reviewer"]["ref"]
    c = amend(c, o, s)
    with pytest.raises(OperationRejected):
        _apply(o, c, SubmitFinalAfterReview(kind="submit_final", draft_id=protocol(c)["final_draft"]["draft_id"]), s)
    c, _, _ = complete_review(c, o, s)
    c = amend(c, o, s)
    c, payload, _ = complete_review(c, o, s)
    assert payload["sections"]["delivery_task"]["scientific_review_ref"] == original
    assert protocol(c)["review_memory"]["final_reviewer"]["scientific_review_ref"] == original


def test_global_limitations_copied_into_predictions_are_not_new_accepted_science(tmp_path):
    c, o, s = reviewed_case(tmp_path)
    before = science(c)
    c = _apply(o, c, AmendFinal(kind="amend_final", draft_id=protocol(c)["final_draft"]["draft_id"],
                              limitations=["Different wording of a delivery limitation."]), s)
    c, payload, _ = complete_review(c, o, s)
    assert payload["sections"]["delivery_task"]["mode"] == "proofread"
    assert payload["sections"]["assembled_answer:limitations"] == ["Different wording of a delivery limitation."]
    assert science(c) == before


def test_legacy_full_receipt_can_supply_exact_scientific_basis(tmp_path):
    c, o, s = reviewed_case(tmp_path)
    old = deepcopy(protocol(c)["review_memory"]["final_reviewer"])
    old.pop("scientific_basis")
    old.pop("delivery_review_mode")
    c = put(c, review_memory={"final_reviewer": old},
            review_receipts={**protocol(c)["review_receipts"], old["ref"]: old})
    c, payload, _ = complete_review(amend(c, o, s), o, s)
    assert payload["sections"]["delivery_task"]["mode"] == "proofread"


def test_unavailable_review_does_not_enable_proofreading(tmp_path):
    c, o, s = reviewed_case(tmp_path)
    old = deepcopy(protocol(c)["review_memory"]["final_reviewer"])
    old["feedback"]["status"] = "unavailable"
    c = put(c, review_memory={"final_reviewer": old},
            review_receipts={**protocol(c)["review_receipts"], old["ref"]: old})
    fake = NativeFake([reply("record_review", report().model_dump(exclude={"kind"}))])
    o.deps.scientific_reviewer.llm_client = fake
    _comment(o, amend(c, o, s), s)
    assert "delivery_task" not in expand_text(fake.requests[0]["payload"])["sections"]


def numeric_case(tmp_path):
    c, o, s = reviewed_case(tmp_path)
    c = put(c, results={"R1": {"status": "success", "capability_id": "fixture",
        "values": {"left": .0689, "right": .0006, "state_index": 1, "units": "dimensionless"}}})
    receipt = deepcopy(protocol(c)["review_memory"]["final_reviewer"])
    receipt.update(source_reads=[{"ref": "R1", "path": "/values/left", "page": None,
                                 "sha256": digest(.0689), "bytes": 6}],
                   read_versions={"R1": digest(protocol(c)["results"]["R1"])})
    c = put(c, review_memory={"final_reviewer": receipt},
            review_receipts={**protocol(c)["review_receipts"], receipt["ref"]: receipt})
    return c, o, s


def test_fact_packet_keeps_exact_source_path_value_and_conditions(tmp_path):
    c, _, _ = numeric_case(tmp_path)
    packet = dr.fact_packets(c, {"R1"})["packets"][0]
    assert packet["value"] == .0689 and packet["source"]["paths"] == ["/values/left"]
    assert packet["parent_context"]["scalar_fields"]["state_index"] == 1
    assert packet["execution"]["result_id"] == "R1"
    assert packet["execution"]["request"] is None
    changed = deepcopy(protocol(c)["results"])
    changed["R1"]["values"]["left"] = .25
    assert not dr.fact_packets(put(c, results=changed), {"R1"})["packets"]
    assert not dr.fact_packets(c, {"foreign-record"})["packets"]


def test_oversized_facts_are_deferred_not_truncated_or_fabricated(tmp_path, monkeypatch):
    c, _, _ = numeric_case(tmp_path)
    monkeypatch.setattr(dr, "FACT_BYTES", 1)
    facts = dr.fact_packets(c, {"R1"})
    assert not facts["packets"] and facts["deferred"][0]["read"]["paths"] == ["/values/left"]


def comparison(**overrides):
    return CompareEvidence.model_validate({"kind": "compare_evidence", "operation": "ratio",
        "left": {"ref": "R1", "path": "/values/left"},
        "right": {"ref": "R1", "path": "/values/right"}, **overrides})


@pytest.mark.parametrize("operation,expected", [("ratio", 114.83333333333334), ("difference", .0683)])
def test_explicit_arithmetic_preserves_source_and_does_not_create_scientific_records(tmp_path, operation, expected):
    c, _, _ = numeric_case(tmp_path)
    before = c.model_dump(mode="json")
    outcome, _ = numeric.calculate(c, comparison(operation=operation), scope={"R1"}, index=record_index(c))
    assert outcome["numeric_check"]["value"] == pytest.approx(expected)
    assert outcome["numeric_check"]["operands"][0]["ref"] == "R1"
    assert c.model_dump(mode="json") == before


@pytest.mark.parametrize("value", [0, True, "0.0006", None, [0.0006], 10 ** 400])
def test_invalid_arithmetic_is_a_local_error_not_inferred(tmp_path, value):
    c, _, _ = numeric_case(tmp_path)
    rows = deepcopy(protocol(c)["results"])
    rows["R1"]["values"]["right"] = value
    c = put(c, results=rows)
    with pytest.raises(OperationRejected):
        numeric.calculate(c, comparison(), scope={"R1"}, index=record_index(c))


def test_arithmetic_cannot_read_outside_scope(tmp_path):
    c, _, _ = numeric_case(tmp_path)
    with pytest.raises(OperationRejected):
        numeric.calculate(c, comparison(), scope=set(), index=record_index(c))


def test_proofread_tools_schema_and_failure_query_allowance(tmp_path):
    c, o, s = reviewed_case(tmp_path)
    c = amend(c, o, s)
    bad = comparison(left={"ref": "foreign-case", "path": "/value"}).model_dump(exclude={"kind"})
    fake = NativeFake([reply("compare_evidence", bad, call_id=str(i)) for i in range(2)] + [
        reply("record_review", {"text": "No usable comparison; limitation retained.", "findings": []})])
    o.deps.scientific_reviewer.llm_client = fake
    for i in range(2):
        c = _comment(o, c, s)
        assert protocol(c)["reviewer_session"]["queries"] == i + 1
    for tool in fake.requests[0]["tools"]:
        Draft202012Validator.check_schema(tool["parameters"])
    c = _comment(o, c, s)
    assert {t["name"] for t in fake.requests[-1]["tools"]} == {"record_review"}
    assert protocol(c)["final_review"]["status"] == "available"


def test_open_factual_issue_and_sources_survive_proofreading(tmp_path):
    from mechcal.runtime import command_review_workflow as workflow

    c, o, s = reviewed_case(tmp_path)
    finding = RecordReview(kind="record_review", text="Fixture issue.", findings=[{
        "id": None, "target": "delivery:answer", "category": "fact", "status": "open",
        "text": "Check a precise operand.", "source_refs": ["input:smiles"], "focus": None}])
    c = workflow.accept_report(c, workflow.validate_report(c, finding, targets=workflow.review_targets(c),
                                                         source_refs=["input:smiles"]))
    c, payload, _ = complete_review(amend(c, o, s), o, s)
    assert payload["sections"]["review_workflow"]["issues"][0]["source_refs"] == ["input:smiles"]
    assert workflow.issue_view(c)[0]["status"] in {"open", "needs_recheck"}
    assert ["input:smiles"] in payload["read_scope"]["rows"]
