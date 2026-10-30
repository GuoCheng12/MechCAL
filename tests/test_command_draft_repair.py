"""Local judgment correction without rewriting prose or accepting partial state."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from mechcal.agents.command_tools import ask_tools, tool_contracts
from mechcal.runtime.command_draft import draft_view, make_draft
from mechcal.runtime.command_loop import _apply, _local_error, planner_payload, save
from mechcal.runtime.command_operations import OperationRejected, check_refs, finish, protocol, put, read_records, revise
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import DiscardJudgment, Finalize, ReadEvidence, RepairJudgment, ReviseJudgment
from test_command_protocol import LABELS, REGISTRY, case, judgment, orchestrator, rank, store
from test_native_commands import NativeFake, reply


def rejected_proposal():
    return ReviseJudgment.model_validate({"kind": "revise_judgment", "changes": [
        {"kind": "argument", "id": "A1", "label": LABELS[0], "text": "Preserve this scientific paragraph exactly.",
         "refs": ["failed", "good:/value"]},
        {"kind": "argument", "id": "A2", "label": LABELS[1], "text": "Preserve this other paragraph.", "refs": ["good"]},
        rank(LABELS[0], ["A2"]), rank(LABELS[1], ["A2"]), rank(LABELS[2]),
    ]})


def fixture_case():
    return put(case(), results={"failed": {"status": "failed", "failure": {"message": "Fixture failed."}},
                                "good": {"status": "success", "value": 1}})


def reject_and_store(o, c, cmd, s):
    with pytest.raises(OperationRejected) as error:
        _apply(o, c, cmd, s)
    exc = error.value
    return _local_error(c, exc.issues, s, exchange=exc.exchange, judgment_draft=exc.judgment_draft)


def corrected(draft_id):
    return RepairJudgment(kind="repair_judgment", draft_id=draft_id, edits=[
        {"kind": "argument_patch", "change_index": 0, "refs": ["good"], "limitation_refs": ["failed"]},
        {"kind": "ranking_patch", "change_index": 2, "argument_ids": ["A1"]},
    ])


def test_all_known_issues_return_together_and_complete_draft_survives_reads_resume(tmp_path):
    o, c = orchestrator(tmp_path), fixture_case()
    s = store(tmp_path, c)
    proposal = rejected_proposal()
    before = c.model_dump_json()
    pending = reject_and_store(o, c, proposal, s)
    draft = protocol(pending)["judgment_draft"]
    assert {i["code"] for i in draft["issues"]} == {"failed_evidence", "unknown_reference", "argument_link"}
    assert draft["command"] == proposal.model_dump(mode="json")
    assert protocol(pending)["arguments"] == protocol(pending)["ranking"] == {}
    assert protocol(pending)["revision"] == protocol(pending)["rounds"] == 0
    assert c.model_dump_json() == before
    pending = save(_apply(o, pending, ReadEvidence(kind="read_evidence", records=[{"ref": "good"}]), s), s)
    restored = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    assert protocol(restored)["judgment_draft"] == draft
    assert (s.root / "judgment_drafts" / (draft["draft_id"] + ".json")).exists()
    view = planner_payload(o, restored)["judgment_draft"]
    assert "Preserve this scientific paragraph" not in json.dumps(view)
    rows = read_records(restored, ReadEvidence(kind="read_evidence", records=[{
        "ref": view["ref"], "paths": ["/command/changes/0/text"]}]))
    assert rows[0]["data"]["/command/changes/0/text"] == proposal.changes[0].text
    with pytest.raises(OperationRejected, match="not a raw observation"):
        check_refs(restored, [view["ref"]], [], scientific=True)
    assert not draft_view(fixture_case())


def test_single_patch_batch_commits_every_record_and_preserves_prose_and_failed_raw_result(tmp_path):
    o, c = orchestrator(tmp_path), fixture_case()
    s = store(tmp_path, c)
    c = reject_and_store(o, c, rejected_proposal(), s)
    draft = protocol(c)["judgment_draft"]
    fixed = save(_apply(o, c, corrected(draft["draft_id"]), s), s)
    p = protocol(fixed)
    assert p["revision"] == 1 and p["judgment_draft"] is None
    assert p["arguments"]["A1"]["text"] == rejected_proposal().changes[0].text
    assert p["arguments"]["A2"]["text"] == rejected_proposal().changes[1].text
    assert p["arguments"]["A1"]["refs"] == ["good"]
    assert p["arguments"]["A1"]["limitation_refs"] == ["failed"]
    assert p["results"] == protocol(c)["results"]
    assert p["ranking"][LABELS[0]]["argument_ids"] == ["A1"]
    assert p["receipt"]["closed_draft_id"] == draft["draft_id"]
    final = finish(fixed, Finalize(kind="finalize", answer="Bounded fixture."))
    assert final.final_answer.evidence_refs == ["good"]
    assert final.final_answer.mechanism_predictions[0].evidence == [rejected_proposal().changes[0].text]
    # Duplicate delivery cannot replay the accepted transaction.
    with pytest.raises(OperationRejected, match="No current draft"):
        _apply(o, fixed, corrected(draft["draft_id"]), s)
    assert protocol(fixed)["revision"] == 1


def test_partial_repair_stays_only_in_draft_and_uses_a_new_id(tmp_path):
    o, c = orchestrator(tmp_path), fixture_case()
    s = store(tmp_path, c)
    c = reject_and_store(o, c, rejected_proposal(), s)
    old = protocol(c)["judgment_draft"]
    partial = RepairJudgment(kind="repair_judgment", draft_id=old["draft_id"], edits=[
        {"kind": "argument_patch", "change_index": 0, "refs": ["good"], "limitation_refs": ["failed"]}])
    updated = reject_and_store(o, c, partial, s)
    draft = protocol(updated)["judgment_draft"]
    assert draft["draft_id"] != old["draft_id"]
    assert [e["code"] for e in draft["issues"]] == ["argument_link"]
    assert draft["command"]["changes"][0]["refs"] == ["good"]
    assert protocol(updated)["arguments"] == {} and protocol(updated)["revision"] == 0
    with pytest.raises(OperationRejected, match="No current draft"):
        _apply(o, updated, partial, s)
    assert protocol(updated)["judgment_draft"] == draft


@pytest.mark.parametrize("edits,code", [
    ([{"kind": "argument_patch", "change_index": 90, "refs": []}], "invalid_edit_target"),
    ([{"kind": "argument_patch", "change_index": 0, "refs": []}] * 2, "invalid_edit_target"),
    ([{"kind": "ranking_patch", "change_index": 0, "argument_ids": []}], "edit_kind"),
    ([{"kind": "argument_patch", "change_index": 0}], "empty_edit"),
])
def test_invalid_patch_is_local_and_cannot_erase_pending_or_accepted_records(tmp_path, edits, code):
    o, c = orchestrator(tmp_path), fixture_case()
    s = store(tmp_path, c)
    c = reject_and_store(o, c, rejected_proposal(), s)
    original = c.model_dump_json()
    cmd = RepairJudgment(kind="repair_judgment", draft_id=protocol(c)["judgment_draft"]["draft_id"], edits=edits)
    with pytest.raises(OperationRejected) as error:
        _apply(o, c, cmd, s)
    assert error.value.issues[0]["code"] == code
    assert error.value.judgment_draft is None and c.model_dump_json() == original


def test_owner_staleness_and_corruption_are_not_silently_rebased(tmp_path):
    o, c = orchestrator(tmp_path), fixture_case()
    s = store(tmp_path, c)
    c = reject_and_store(o, c, rejected_proposal(), s)
    did = protocol(c)["judgment_draft"]["draft_id"]
    stale = put(c, revision=1)
    with pytest.raises(OperationRejected, match="changed after"):
        _apply(o, stale, corrected(did), s)
    discarded = _apply(o, stale, DiscardJudgment(kind="discard_judgment", draft_id=did), s)
    assert not protocol(discarded)["judgment_draft"]
    corrupt = deepcopy(protocol(c)["judgment_draft"])
    corrupt["command"]["changes"][0]["text"] = "Overwritten"
    with pytest.raises(RuntimeError, match="identity"):
        _apply(o, put(c, judgment_draft=corrupt), corrected(did), s)
    corrupt["case_id"] = "foreign-case"
    with pytest.raises(RuntimeError, match="ownership"):
        _apply(o, put(c, judgment_draft=corrupt), corrected(did), s)


def test_discard_and_explicit_replacement_never_merge_draft_half_into_accepted_science(tmp_path):
    o, c = orchestrator(tmp_path), revise(fixture_case(), judgment())
    s = store(tmp_path, c)
    c = reject_and_store(o, c, rejected_proposal(), s)
    with pytest.raises(OperationRejected, match="uncommitted"):
        finish(c, Finalize(kind="finalize", answer="Cannot silently ignore pending edits."))
    discarded = _apply(o, c, DiscardJudgment(kind="discard_judgment", draft_id=protocol(c)["judgment_draft"]["draft_id"]), s)
    assert protocol(discarded)["arguments"] == protocol(c)["arguments"]
    assert protocol(discarded)["revision"] == protocol(c)["revision"]
    assert finish(discarded, Finalize(kind="finalize", answer="Use accepted records.")).final_answer
    replaced = _apply(o, c, judgment(), s)
    assert protocol(replaced)["judgment_draft"] is None
    assert "A2" not in protocol(replaced)["arguments"]


def test_additions_and_removals_use_original_indices_and_commit_atomically(tmp_path):
    o, c = orchestrator(tmp_path), fixture_case()
    s = store(tmp_path, c)
    c = reject_and_store(o, c, rejected_proposal(), s)
    cmd = RepairJudgment(kind="repair_judgment", draft_id=protocol(c)["judgment_draft"]["draft_id"], edits=[
        {"kind": "remove_change", "change_index": 0},
        {"kind": "ranking_patch", "change_index": 2, "argument_ids": ["A3"]},
    ], additional_changes=[{"kind": "argument", "id": "A3", "label": LABELS[0], "text": "New explicit text.", "refs": ["good"]}])
    done = _apply(o, c, cmd, s)
    assert set(protocol(done)["arguments"]) == {"A2", "A3"}
    assert protocol(done)["ranking"][LABELS[0]]["argument_ids"] == ["A3"]


def test_failed_only_argument_is_a_limitation_not_positive_final_evidence(tmp_path):
    cmd = judgment().model_copy(deep=True)
    cmd.changes[0].refs = []
    cmd.changes[0].limitation_refs = ["failed"]
    cmd.changes[0].text = "The requested calculation failed; no descriptor is available."
    c = revise(fixture_case(), cmd)
    answer = finish(c, Finalize(kind="finalize", answer="Bounded fixture.")).final_answer
    assert answer.evidence_refs == []
    assert answer.mechanism_predictions[0].evidence == []
    assert answer.mechanism_predictions[0].limitations == [cmd.changes[0].text]
    cmd.changes[0].limitation_refs = ["foreign-record"]
    with pytest.raises(OperationRejected, match="not in this case"):
        revise(fixture_case(), cmd)


def test_structured_calls_expose_repair_only_with_a_draft(tmp_path):
    before, _ = tool_contracts(REGISTRY, [])
    after, _ = tool_contracts(REGISTRY, [], with_draft=True)
    assert {t["name"] for t in after} - {t["name"] for t in before} == {"repair_judgment", "discard_judgment"}
    o, c = orchestrator(tmp_path), fixture_case()
    s = store(tmp_path, c)
    c = reject_and_store(o, c, rejected_proposal(), s)
    cmd = corrected(protocol(c)["judgment_draft"]["draft_id"])
    fake = NativeFake([reply("repair_judgment", cmd.model_dump(exclude={"kind", "draft_id"}))])
    parsed, _ = ask_tools(fake, REGISTRY, [], planner_payload(o, c, native=True), s)
    assert parsed == cmd


@pytest.mark.parametrize("fault", ["truncated", "missing", "unknown_ref"])
def test_bad_repair_output_preserves_saved_draft(tmp_path, fault):
    o, c = orchestrator(tmp_path), fixture_case()
    s = store(tmp_path, c)
    c = reject_and_store(o, c, rejected_proposal(), s)
    old = deepcopy(protocol(c)["judgment_draft"])
    raw = corrected(old["draft_id"]).model_dump(exclude={"kind", "draft_id"})
    incoming = reply("repair_judgment", raw)
    if fault == "truncated":
        incoming["complete"] = False
    elif fault == "missing":
        raw.pop("edits")
        incoming = reply("repair_judgment", raw)
    else:
        raw["edits"][0]["refs"] = ["foreign-record"]
        incoming = reply("repair_judgment", raw)
    fake = NativeFake([incoming])
    with pytest.raises(OperationRejected) as err:
        cmd, exchange = ask_tools(fake, REGISTRY, [], planner_payload(o, c), s)
        _apply(o, put(c, latest_exchange=exchange), cmd, s)
    updated = _local_error(c, err.value.issues, s, exchange=err.value.exchange, judgment_draft=err.value.judgment_draft)
    assert protocol(updated)["revision"] == 0 and not protocol(updated)["arguments"]
    if fault != "unknown_ref":
        assert protocol(updated)["judgment_draft"] == old


def test_full_native_loop_rejected_proposal_repair_finalize_and_review_separation(tmp_path):
    o = orchestrator(tmp_path)
    o.config = replace(o.config, command_native_tools=True)
    bad = judgment().model_copy(deep=True)
    bad.changes[0].refs = ["unknown"]
    fake = NativeFake([])

    def next_turn(**kw):
        fake.requests.append(kw)
        if len(fake.requests) == 1:
            return reply("revise_judgment", bad.model_dump(exclude={"kind"}), call_id="bad")
        if len(fake.requests) == 2:
            draft = kw["payload"]["judgment_draft"]
            assert not kw["payload"]["ranking"] and kw["outcome"]["issues"][0]["path"][0] == "draft"
            cmd = RepairJudgment(kind="repair_judgment", draft_id=draft["draft_id"], edits=[
                {"kind": "argument_patch", "change_index": 0, "refs": ["input:smiles"]}])
            return reply("repair_judgment", cmd.model_dump(exclude={"kind", "draft_id"}), call_id="repair")
        assert len(fake.requests) == 3 and kw["payload"]["judgment_draft"] is None
        return reply("finalize", {"answer": "Offline fixture.", "limitations": []}, call_id="done")

    fake.complete_tool_turn = next_turn
    o.deps.planner.llm_client = fake
    result = o.run(smiles="CCO", user_query="Offline fixture.", case_id="draft-native")
    assert result.status == "finalized" and protocol(result)["rounds"] == 0
    assert result.final_answer.mechanism_predictions[0].evidence == [bad.changes[0].text]
    assert len(list((tmp_path / "draft-native/judgment_drafts").glob("*.json"))) == 1


def test_old_case_without_drafts_or_limitation_refs_still_reads_and_finishes():
    c = revise(case(), judgment())
    args = deepcopy(protocol(c)["arguments"])
    args["A1"].pop("limitation_refs")
    c = CaseRun.model_validate_json(put(c, arguments=args).model_dump_json())
    assert draft_view(c) is None
    assert finish(c, Finalize(kind="finalize", answer="Legacy fixture.")).final_answer.evidence_refs == ["input:smiles"]
