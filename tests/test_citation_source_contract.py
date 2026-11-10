"""Exact citations follow stored provenance, not matching ID strings or guessed paths."""

from copy import deepcopy
import json

import pytest

from mechcal.agents.command_tools import ask_tools, parse_tool_turn, tool_contracts
from mechcal.runtime.command_loop import _apply, _model_issue_paths, planner_payload, save
from mechcal.runtime.command_operations import OperationRejected, protocol, put, record_identity, records, revise
from mechcal.runtime.command_review import review_citations, reviewer_scope
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import ReadRecord
from test_command_draft_repair import corrected, fixture_case, reject_and_store, rejected_proposal
from test_command_protocol import REGISTRY, judgment, orchestrator, store
from test_command_record_identity import RESULT, UNIT, source_case
from test_native_commands import NativeFake, reply


def source():
    c = source_case(result_status="success", unit_status="present")
    results = deepcopy(protocol(c)["results"])
    results[RESULT]["structured_results"] = {"metrics": {"value": 2.5}}
    return put(c, results=results)


def proposal(*, ref=UNIT, citation_ref=RESULT, path="/structured_results/metrics/value"):
    cmd = judgment().model_copy(deep=True)
    cmd.changes[0].refs = [ref]
    cmd.changes[0].citations = [ReadRecord(ref=citation_ref, paths=[path])]
    return cmd


def test_child_argument_can_select_exact_parent_values_without_rewriting_either_id(tmp_path):
    c, cmd = source(), proposal()
    original = c.model_dump_json()
    accepted = revise(c, cmd)
    assert c.model_dump_json() == original
    argument = protocol(accepted)["arguments"]["A1"]
    assert argument == cmd.changes[0].model_dump(mode="json")
    assert protocol(accepted)["results"] == protocol(c)["results"]
    citations = review_citations(accepted, [argument])
    assert citations[0]["ref"] == RESULT
    assert citations[0]["data"] == {"/structured_results/metrics/value": 2.5}
    assert RESULT in reviewer_scope(accepted, {UNIT}, records(accepted))
    s = store(tmp_path, accepted)
    save(accepted, s)
    restored = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    assert review_citations(restored, [argument]) == citations


def test_advertised_source_relation_matches_validation():
    c = source()
    assert record_identity(c, UNIT)["citation_source_refs"] == [UNIT, RESULT]
    assert record_identity(c, RESULT)["citation_source_refs"] == [RESULT]
    assert record_identity(c, "input:smiles")["citation_source_refs"] == ["input:smiles"]


@pytest.mark.parametrize("target", ["unrelated", RESULT + ":E002", "foreign-case", "report:" + RESULT])
def test_unrelated_sibling_foreign_and_interpretation_are_not_parent_aliases(target):
    c = source()
    results = deepcopy(protocol(c)["results"])
    results["unrelated"] = {"status": "success", "structured_results": {"metrics": {"value": 2.5}}}
    sibling = deepcopy(results[RESULT]["evidence_units"][0])
    sibling["evidence_id"] = RESULT + ":E002"
    results[RESULT]["evidence_units"].append(sibling)
    c = put(c, results=results)
    before = c.model_dump_json()
    with pytest.raises(OperationRejected) as exc:
        revise(c, proposal(citation_ref=target, path=""))
    assert any(i["code"] == "citation_scope" for i in exc.value.issues)
    assert c.model_dump_json() == before


@pytest.mark.parametrize("citation_ref,path", [
    (UNIT, "/structured_results/metrics/value"),
    (RESULT, "/metrics/value"),
    (RESULT, "/structured_results/metrics/invented"),
])
def test_provenance_never_guesses_or_moves_a_field_path(citation_ref, path):
    c = source()
    with pytest.raises(OperationRejected) as exc:
        revise(c, proposal(citation_ref=citation_ref, path=path))
    assert any(i["code"] == "unknown_path" for i in exc.value.issues)
    assert protocol(c)["revision"] == 0


@pytest.mark.parametrize("failed", ["parent", "unit"])
def test_parent_permission_does_not_turn_failed_observation_into_support(failed):
    c = source()
    results = deepcopy(protocol(c)["results"])
    row = results[RESULT] if failed == "parent" else results[RESULT]["evidence_units"][0]
    row["status"] = "failed"
    c = put(c, results=results)
    cmd = proposal()
    with pytest.raises(OperationRejected) as exc:
        revise(c, cmd)
    assert any(i["code"] == "failed_evidence" for i in exc.value.issues)
    cmd.changes[0].refs = []
    cmd.changes[0].limitation_refs = [UNIT]
    fixed = revise(c, cmd)
    assert protocol(fixed)["arguments"]["A1"]["refs"] == []
    assert fixed.portfolio.hypotheses[0].evidence_refs == []


def test_unknown_ref_cannot_gain_parent_permission_by_prefix():
    c = source()
    with pytest.raises(OperationRejected) as exc:
        revise(c, proposal(ref=RESULT + ":invented"))
    assert {i["code"] for i in exc.value.issues} >= {"unknown_reference", "citation_scope"}


def test_directory_is_not_a_quantitative_citation():
    cmd = proposal()
    cmd.changes[0].citations[0].view = "directory"
    with pytest.raises(OperationRejected) as exc:
        revise(source(), cmd)
    assert any(i["code"] == "citation_scope" for i in exc.value.issues)


def test_record_corruption_still_interrupts_instead_of_rebinding():
    c = source()
    results = deepcopy(protocol(c)["results"])
    results[RESULT]["evidence_units"][0]["source_report_id"] = "different-parent"
    with pytest.raises(RuntimeError, match="parent ownership"):
        revise(put(c, results=results), proposal())


def test_native_repair_target_is_bound_to_request_not_generated_by_model(tmp_path):
    o, c = orchestrator(tmp_path), fixture_case()
    s = store(tmp_path, c)
    c = reject_and_store(o, c, rejected_proposal(), s)
    did = protocol(c)["judgment_draft"]["draft_id"]
    cmd = corrected(did)
    fake = NativeFake([reply("repair_judgment", cmd.model_dump(exclude={"kind", "draft_id"}))])
    parsed, _ = ask_tools(fake, REGISTRY, [], planner_payload(o, c, native=True), s)
    assert parsed == cmd
    definition = next(t for t in fake.requests[0]["tools"] if t["name"] == "repair_judgment")
    assert "draft_id" not in definition["parameters"]["properties"]
    assert protocol(_apply(o, c, parsed, s))["revision"] == 1
    # A request prepared against a discarded/replaced draft must never rebase.
    replacement = rejected_proposal().model_copy(deep=True)
    replacement.note = "A distinct pending transaction."
    changed = reject_and_store(o, c, replacement, s)
    with pytest.raises(OperationRejected) as exc:
        _apply(o, changed, parsed, s)
    assert exc.value.issues[0]["code"] == "stale_draft"


def test_bound_target_cannot_be_overridden_and_legacy_contract_is_retained():
    tools, models = tool_contracts(REGISTRY, [], with_draft=True, draft_id="request-bound-draft")
    cmd = corrected("malicious-or-mistyped-draft")
    with pytest.raises(OperationRejected):
        parse_tool_turn(reply("repair_judgment", cmd.model_dump(exclude={"kind"})), tools, models)
    old_tools, old_models = tool_contracts(REGISTRY, [], with_draft=True)
    assert parse_tool_turn(reply("repair_judgment", cmd.model_dump(exclude={"kind"})), old_tools, old_models) == cmd
    assert "request-bound-draft" not in json.dumps(tools)


def test_discard_is_also_request_bound(tmp_path):
    o, c = orchestrator(tmp_path), fixture_case()
    s = store(tmp_path, c)
    c = reject_and_store(o, c, rejected_proposal(), s)
    fake = NativeFake([reply("discard_judgment", {})])
    cmd, _ = ask_tools(fake, REGISTRY, [], planner_payload(o, c, native=True), s)
    assert cmd.draft_id == protocol(c)["judgment_draft"]["draft_id"]
    discarded = _apply(o, c, cmd, s)
    assert protocol(discarded)["judgment_draft"] is None
    assert protocol(discarded)["arguments"] == {}


@pytest.mark.parametrize("code", ["stale_draft", "stale_revision"])
def test_stale_bound_target_feedback_identifies_state_not_an_unwritable_argument(code):
    issue = {"path": ["draft_id"], "code": code, "remedy": "Use the displayed ID."}
    native = _model_issue_paths([issue], native=True)[0]
    assert native["path"] == ["judgment_draft", "draft_id"]
    assert "do not generate a draft_id" in native["remedy"]
    legacy = _model_issue_paths([issue], native=False)[0]
    assert legacy["path"] == ["command", "draft_id"]
    assert legacy["remedy"] == issue["remedy"]
