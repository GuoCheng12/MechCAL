"""Visible result units remain addressable without becoming positive evidence."""

from copy import deepcopy

import pytest

from mechcal.runtime.command_loop import _comment, planner_payload, save
from mechcal.runtime.command_operations import (
    OperationRejected, check_refs, protocol, put, read_records, records, revise,
)
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import ReadEvidence
from mechcal.schemas.evidence import EvidenceUnit
from test_command_protocol import ScriptClient, case, judgment, orchestrator, store

RESULT = "execution:fixture"
UNIT = RESULT + ":E001"


def source_case(*, result_status="failed", unit_status="failed"):
    unit = EvidenceUnit(
        evidence_id=UNIT, source_report_id=RESULT, round_id="R001",
        agent_name="microscopic", capability_id="microscopic.fixture",
        claim="Fixture diagnostic only.", context="Offline identity test.",
        basis="checklist", support="unresolved", summary="No validated observation.",
        family="charge_localization", status=unit_status,
    )
    return put(case(), results={RESULT: {
        "tool_result_id": RESULT, "status": result_status,
        "evidence_units": [unit.model_dump(mode="json")],
    }})


def test_nested_status_record_is_readable_without_promoting_it():
    c = source_case()
    before = c.model_dump(mode="json")
    query = ReadEvidence(kind="read_evidence", records=[{"ref": UNIT}])
    row = read_records(c, query)[0]
    assert row["data"] == protocol(c)["results"][RESULT]["evidence_units"][0]
    assert records(c)[UNIT] == row["data"]
    check_refs(c, [UNIT], ["refs"])
    assert UNIT not in {e.evidence_id for e in c.evidence_ledger.items}
    assert c.model_dump(mode="json") == before


@pytest.mark.parametrize("result_status,unit_status", [
    ("failed", "failed"), ("failed", "present"), ("unsupported", "present"),
    ("precondition_missing", "partial"), ("success", "missing"),
    ("success", "failed"), ("partial", "unsupported"),
])
def test_failed_unit_or_parent_is_limitation_only(result_status, unit_status):
    c = source_case(result_status=result_status, unit_status=unit_status)
    with pytest.raises(OperationRejected) as error:
        check_refs(c, [UNIT], ["refs"], scientific=True)
    assert error.value.issues[0]["code"] == "failed_evidence"
    assert error.value.issues[0]["record_id"] == UNIT
    check_refs(c, [UNIT], ["limitation_refs"], scientific=True, allow_failed=True)


def test_successful_unit_keeps_ledger_annotations_and_scientific_refs():
    c = source_case(result_status="success", unit_status="present")
    raw = protocol(c)["results"][RESULT]["evidence_units"][0]
    annotated = EvidenceUnit.model_validate({**raw, "claim_refs": ["derived-claim"]})
    c = c.touch(evidence_ledger=c.evidence_ledger.with_items([annotated]))
    assert records(c)[UNIT] == annotated.model_dump(mode="json")
    check_refs(c, [UNIT], ["refs"], scientific=True)
    assert not raw["claim_refs"]


def test_unknown_and_other_case_ids_remain_local_rejections():
    a = source_case()
    b = case()
    for c, ref in ((a, UNIT + ":invented"), (b, UNIT)):
        with pytest.raises(OperationRejected) as error:
            check_refs(c, [ref], ["refs"])
        assert error.value.issues[0]["code"] == "unknown_reference"


@pytest.mark.parametrize("damage", ["parent", "result_id", "case_id", "unit_case_id", "collision", "duplicate"])
def test_corrupt_record_identity_is_not_silently_resolved(damage):
    c = source_case()
    results = deepcopy(protocol(c)["results"])
    result = results[RESULT]
    if damage == "parent":
        result["evidence_units"][0]["source_report_id"] = "execution:other"
    elif damage == "result_id":
        result["tool_result_id"] = "execution:other"
    elif damage == "case_id":
        result["case_id"] = "other-case"
    elif damage == "unit_case_id":
        result["evidence_units"][0]["case_id"] = "other-case"
    elif damage == "collision":
        result["evidence_units"][0]["evidence_id"] = RESULT
    else:
        result["evidence_units"].append(deepcopy(result["evidence_units"][0]))
    with pytest.raises(RuntimeError):
        records(put(c, results=results))


def test_ledger_and_result_cannot_disagree_on_scientific_content():
    c = source_case(result_status="success", unit_status="present")
    raw = protocol(c)["results"][RESULT]["evidence_units"][0]
    changed = EvidenceUnit.model_validate({**raw, "summary": "Conflicting observation."})
    c = c.touch(evidence_ledger=c.evidence_ledger.with_items([changed]))
    with pytest.raises(RuntimeError):
        records(c)


def test_argument_cannot_shadow_nested_source_record():
    c = source_case()
    command = judgment().model_copy(deep=True)
    command.changes[0].id = UNIT
    command.changes[1].argument_ids = [UNIT]
    with pytest.raises(OperationRejected) as error:
        revise(c, command)
    assert error.value.issues[0]["code"] == "record_collision"
    with pytest.raises(RuntimeError):
        records(put(c, arguments={UNIT: command.changes[0].model_dump(mode="json")}))


def test_nested_limitation_commits_but_positive_use_does_not():
    c = source_case()
    update = judgment().model_copy(deep=True)
    update.changes[0].refs = [UNIT]
    with pytest.raises(OperationRejected) as error:
        revise(c, update)
    assert error.value.issues[0]["code"] == "failed_evidence"
    update.changes[0].refs = ["input:smiles"]
    update.changes[0].limitation_refs = [UNIT]
    updated = revise(c, update)
    assert protocol(updated)["arguments"]["A1"]["limitation_refs"] == [UNIT]
    assert UNIT not in updated.portfolio.hypotheses[0].evidence_refs


@pytest.mark.parametrize("role", ["reviewer", "final_reviewer"])
def test_reviewer_can_cite_nested_status_and_planner_receives_feedback(tmp_path, role):
    c = source_case()
    update = judgment().model_copy(deep=True)
    update.changes[0].limitation_refs = [RESULT]
    c = revise(c, update)
    c = put(c, phase="review", review_role=role,
            review_changes=[change.model_dump(mode="json") for change in update.changes])
    before = c.model_dump(mode="json")
    o, s = orchestrator(tmp_path, reviews=True), store(tmp_path, c)
    text = f"{UNIT} is missing evidence, not negative evidence."
    client = ScriptClient([text])
    o.deps.scientific_reviewer.llm_client = client
    reviewed = _comment(o, c, s)
    shown = client.calls[0]["payload"]["observations"][RESULT]["evidence_units"][0]
    assert shown["evidence_id"] == UNIT
    assert protocol(reviewed)["review"] == {"status": "available", "text": text,
        "reviewed_record_ids": client.calls[0]["payload"]["reference_ids"]}
    assert UNIT in protocol(reviewed)["review"]["reviewed_record_ids"]
    assert planner_payload(o, reviewed)["review_feedback"] == protocol(reviewed)["review"]
    for field in ("portfolio", "evidence_ledger", "claim_ledger", "artifact_manifest"):
        assert reviewed.model_dump(mode="json")[field] == before[field]
    for key in ("results", "arguments", "ranking", "revision", "rounds"):
        assert protocol(reviewed)[key] == before["runtime"]["command_protocol"][key]
    save(reviewed, s)
    restored = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    assert records(restored)[UNIT] == shown
    check_refs(restored, [UNIT], ["refs"])


def test_legacy_unaddressed_unit_remains_accessible_via_parent():
    c = put(case(), results={RESULT: {"status": "success", "evidence_units": [{"context": "Legacy fixture."}]}})
    assert RESULT in records(c)
    assert not any(ref.startswith(RESULT + ":") for ref in records(c))


def test_array_results_keep_exact_reads_and_pointer_validation():
    c = put(case(), results={RESULT: [1, {"value": 2}]})
    full = ReadEvidence(kind="read_evidence", records=[{"ref": RESULT}])
    assert read_records(c, full)[0]["data"] == [1, {"value": 2}]
    child = ReadEvidence(kind="read_evidence", records=[{"ref": RESULT, "paths": ["/1/value"]}])
    assert read_records(c, child)[0]["data"] == {"/1/value": 2}
    invalid = ReadEvidence(kind="read_evidence", records=[{"ref": RESULT, "paths": ["/-1"]}])
    with pytest.raises(OperationRejected) as error:
        read_records(c, invalid)
    assert error.value.issues[0]["code"] == "unknown_path"


def test_ledger_annotation_cannot_hide_failed_parent():
    c = source_case(unit_status="present")
    raw = protocol(c)["results"][RESULT]["evidence_units"][0]
    c = c.touch(evidence_ledger=c.evidence_ledger.with_items([EvidenceUnit.model_validate(raw)]))
    with pytest.raises(OperationRejected) as error:
        check_refs(c, [UNIT], ["refs"], scientific=True)
    assert error.value.issues[0]["code"] == "failed_evidence"


def test_ledger_cannot_mask_parent_and_child_id_collision():
    c = source_case()
    p = deepcopy(protocol(c))
    raw = p["results"][RESULT]["evidence_units"][0]
    raw["evidence_id"] = RESULT
    c = put(c, results=p["results"])
    c = c.touch(evidence_ledger=c.evidence_ledger.with_items([EvidenceUnit.model_validate(raw)]))
    with pytest.raises(RuntimeError):
        records(c)
