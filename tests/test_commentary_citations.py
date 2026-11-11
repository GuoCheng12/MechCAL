"""Review scope exposes supplied records without treating prose as citations."""

import pytest
from jsonschema import Draft202012Validator

from mechcal.runtime.command_loop import _comment, planner_payload, save
from mechcal.runtime.command_operations import OperationRejected, check_refs, protocol, put, records, revise
from mechcal.runtime.command_schema import native_format
from mechcal.schemas import CaseRun
from mechcal.schemas.artifacts import ArtifactRecord
from mechcal.schemas.commands import commentary_model
from test_command_protocol import ScriptClient, case, judgment, orchestrator, store
from test_command_record_identity import RESULT, UNIT, source_case


def review(tmp_path, c, reply, role="coverage_reviewer"):
    c = put(c, phase="review", review_role=role)
    o, s = orchestrator(tmp_path, reviews=True), store(tmp_path, c)
    client = ScriptClient([reply])
    o.deps.scientific_reviewer.llm_client = client
    result = _comment(o, c, s)
    return o, s, result, client.calls[0]


def citation_ids(request):
    return request["payload"]["reference_ids"]


@pytest.mark.parametrize("role", ["coverage_reviewer", "reviewer", "final_reviewer"])
@pytest.mark.parametrize("text", ["Compare structural hypotheses.", "Inspect input:smiles before selecting a calculation."])
def test_initial_review_has_an_addressable_input_without_requiring_evidence(tmp_path, role, text):
    c = case()
    before = c.model_dump(mode="json")
    o, s, reviewed, request = review(tmp_path, c, text, role)
    payload = request["payload"]
    assert payload["observations"] == {"input:smiles": {"smiles": c.input.smiles}}
    assert payload["reference_ids"] == citation_ids(request) == ["input:smiles"]
    assert "response_format" not in request
    assert protocol(reviewed)["review"] == {"status": "available", "text": text,
                                           "reviewed_record_ids": ["input:smiles"]}
    assert planner_payload(o, reviewed)["review_feedback"] == protocol(reviewed)["review"]
    for field in ("portfolio", "evidence_ledger", "claim_ledger", "artifact_manifest"):
        assert reviewed.model_dump(mode="json")[field] == before[field]
    for key in ("results", "arguments", "ranking", "revision", "rounds"):
        assert protocol(reviewed)[key] == before["runtime"]["command_protocol"][key]
    save(reviewed, s)
    restored = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    assert protocol(restored)["review"] == protocol(reviewed)["review"]


@pytest.mark.parametrize("ref", [
    "case.smiles", "mechanism_pool", "arguments", "observations", "ranking", "tool_catalog",
    "another-case:E001",
])
def test_field_names_and_unknown_ids_in_text_are_not_registered_as_sources(tmp_path, ref):
    text = f"Discuss {ref}, without creating a source record."
    _, _, reviewed, request = review(tmp_path, case(), text)
    assert ref not in citation_ids(request)
    assert protocol(reviewed)["review"]["text"] == text
    assert ref not in protocol(reviewed)["review"]["reviewed_record_ids"]
    with pytest.raises(OperationRejected):
        check_refs(reviewed, [ref], ["refs"], scientific=True)


def test_nested_failure_records_are_visible_but_unrelated_records_are_not(tmp_path):
    c = source_case()
    update = judgment().model_copy(deep=True)
    update.changes[0].limitation_refs = [RESULT]
    c = revise(c, update)
    c = put(c, review_changes=[v.model_dump(mode="json") for v in update.changes],
            results={**protocol(c)["results"], "execution:unshown": {"status": "failed"}})
    text = f"{UNIT} does not rule out a mechanism; reconsider A1."
    _, _, reviewed, request = review(tmp_path, c, text, "reviewer")
    ids = request["payload"]["reference_ids"]
    assert ids == citation_ids(request) == sorted(["input:smiles", "A1", RESULT, UNIT])
    assert "execution:unshown" in records(c) and "execution:unshown" not in ids
    assert protocol(reviewed)["review"] == {"status": "available", "text": text, "reviewed_record_ids": ids}
    assert not reviewed.evidence_ledger.items


def test_citation_scope_does_not_leak_between_cases_or_calls(tmp_path):
    first = source_case()
    update = judgment().model_copy(deep=True)
    update.changes[0].limitation_refs = [RESULT]
    first = revise(first, update)
    _, _, _, earlier = review(tmp_path / "first", first, "No claim.", "final_reviewer")
    assert UNIT in citation_ids(earlier)
    second = case().touch(case_id="different-case")
    second.input.case_id = second.case_id
    second.evidence_ledger.case_id = second.case_id
    second.artifact_manifest.case_id = second.case_id
    second = put(second, case_id=second.case_id)
    _, _, result, later = review(tmp_path / "second", second, f"Unverified mention of {UNIT}.")
    assert citation_ids(later) == ["input:smiles"]
    assert UNIT not in protocol(result)["review"]["reviewed_record_ids"]
    with pytest.raises(OperationRejected):
        check_refs(result, [UNIT], ["refs"], scientific=True)


def test_referenced_artifacts_are_available_without_loading_every_artifact(tmp_path):
    c = source_case()
    artifacts = [ArtifactRecord(artifact_id=ref, kind="fixture", created_by_round="R001",
                                created_by_agent="microscopic")
                 for ref in ("artifact:referenced", "artifact:unshown")]
    c = c.touch(artifact_manifest=c.artifact_manifest.with_records(artifacts))
    raw = {**protocol(c)["results"][RESULT], "artifact_refs": ["artifact:referenced"]}
    c = put(c, results={RESULT: raw})
    update = judgment().model_copy(deep=True)
    update.changes[0].limitation_refs = [RESULT]
    c = revise(c, update)
    text = "artifact:referenced documents the attempted calculation."
    _, _, reviewed, request = review(tmp_path, c, text, "final_reviewer")
    assert "artifact:referenced" in citation_ids(request)
    assert "artifact:unshown" not in citation_ids(request)
    assert protocol(reviewed)["review"] == {"status": "available", "text": text,
                                           "reviewed_record_ids": citation_ids(request)}


def test_empty_scope_allows_only_uncited_prose():
    model = commentary_model([])
    schema = native_format(model)["json_schema"]["schema"]
    validator = Draft202012Validator(schema)
    assert validator.is_valid({"text": "General advice.", "refs": []})
    assert not validator.is_valid({"text": "General advice.", "refs": ["input:smiles"]})
    assert model.model_validate({"text": "General advice.", "refs": []}).refs == []


def test_model_validation_and_wire_schema_enforce_the_same_ids():
    from pydantic import ValidationError

    model = commentary_model(["source:b", "source:a", "source:b"])
    schema = native_format(model)["json_schema"]["schema"]
    assert schema["properties"]["refs"]["items"]["enum"] == ["source:a", "source:b"]
    for refs in ([], ["source:a"], ["source:b", "source:a"]):
        reply = {"text": "Scientific prose stays unchanged.", "refs": refs}
        assert Draft202012Validator(schema).is_valid(reply)
        assert model.model_validate(reply).model_dump() == reply
    with pytest.raises(ValidationError):
        model.model_validate({"text": "Unknown reference.", "refs": ["foreign"]})
