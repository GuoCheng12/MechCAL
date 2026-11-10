"""Exact editing context without changing science or treating memory as proof."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest
from pydantic import ValidationError

from mechcal.runtime import command_memory
from mechcal.runtime.command_compact import expand_text
from mechcal.runtime.command_edit_context import EDIT_BYTES, edit_context
from mechcal.runtime.command_loop import _apply, _local_error, planner_payload
from mechcal.runtime.command_operations import OperationRejected, protocol, put, read_records
from mechcal.runtime.command_task_context import prepare_planner
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import AmendFinal, ReadEvidence, TextReplacement
from test_command_task_context import ready
from test_delivery_recovery import science
from test_native_commands import reply


def read(c, o, s, ref="delivery:answer", paths=None, view="data", page=None):
    args = {"records": [{"ref": ref, "paths": paths or [], "view": view, "page": page}]}
    exchange = reply(arguments=args)
    return _apply(o, put(c, latest_exchange=exchange), ReadEvidence(kind="read_evidence", **args), s)


def shown(c, o):
    return expand_text(planner_payload(o, c, native=True))


def data_in(context, path):
    return next((row.get("data") for row in context["retained"] if row["path"] == path), None)


def test_single_canonical_location_and_direct_read_paths(tmp_path):
    c, o, s = ready(tmp_path)
    before = science(c)
    payload = shown(c, o)
    assert payload["final_draft"]["read_ref"] == "delivery:answer"
    assert payload["edit_context"]["read"] == {
        "ref": "delivery:answer", "paths": ["/answer", "/limitations"], "view": "data", "page": None}
    query = payload["edit_context"]["read"]
    current = _apply(o, c, ReadEvidence(kind="read_evidence", records=[query]), s)
    assert protocol(current)["outcome"]["status"] == "read"
    assert science(current) == before


@pytest.mark.parametrize("alias", ["delivery:answer", "snapshot_command", "snapshot_content", "snapshot_root"])
def test_alias_reads_pin_the_same_editable_text(tmp_path, alias):
    c, o, s = ready(tmp_path)
    draft = protocol(c)["final_draft"]
    ref = "delivery:answer" if alias == "delivery:answer" else draft["draft_id"]
    paths = {"delivery:answer": [], "snapshot_command": ["/command"],
             "snapshot_content": ["/content/answer", "/content/limitations"], "snapshot_root": []}[alias]
    c = read(c, o, s, ref, paths)
    context = shown(c, o)["edit_context"]
    assert context["read"] is None
    assert all(row["location"] == "latest_tool_result" and "data" not in row for row in context["retained"])
    c = put(c, latest_exchange=None, outcome=None)
    context = shown(c, o)["edit_context"]
    assert data_in(context, "/answer") == draft["command"]["answer"]


def test_editing_text_survives_cache_pressure_and_failed_edit(tmp_path, monkeypatch):
    c, o, s = ready(tmp_path)
    before = science(c)
    c = read(c, o, s, paths=["/answer", "/limitations"])
    monkeypatch.setattr(command_memory, "WORKING_BYTES", 0)
    draft = protocol(c)["final_draft"]
    with pytest.raises(OperationRejected) as failed:
        _apply(o, c, AmendFinal(kind="amend_final", draft_id=draft["draft_id"], replacements=[
            {"old_text": "Missing exact passage.", "new_text": "Not committed."}]), s)
    c = _local_error(c, failed.value.issues, s, exchange=reply("amend_final", {}, call_id="bad"))
    payload = shown(c, o)
    assert data_in(payload["edit_context"], "/answer") == draft["command"]["answer"]
    assert payload["edit_context"]["read"] is None
    assert not payload["working_memory"]["findings"]
    assert json.dumps(payload).count(draft["command"]["answer"]) == 1
    c = _apply(o, c, AmendFinal(kind="amend_final", draft_id=draft["draft_id"], replacements=[
        {"old_text": draft["command"]["answer"], "new_text": "A corrected protocol-fixture answer."}]), s)
    assert protocol(c)["phase"] == "review"
    assert science(c) == before


def test_changed_draft_invalidates_old_read_receipts_and_new_read_refreshes(tmp_path):
    c, o, s = ready(tmp_path)
    c = read(c, o, s)
    draft = protocol(c)["final_draft"]
    c = _apply(o, c, AmendFinal(kind="amend_final", draft_id=draft["draft_id"], replacements=[
        {"old_text": draft["command"]["answer"], "new_text": "New protocol-fixture version."}]), s)
    context = shown(c, o)["edit_context"]
    assert context["draft_id"] != draft["draft_id"]
    assert not context["retained"] and context["read"] is not None
    c = read(c, o, s)
    c = put(c, latest_exchange=None, outcome=None)
    assert data_in(shown(c, o)["edit_context"], "/answer") == "New protocol-fixture version."


def test_answer_and_indexed_limitation_edits_share_one_atomic_draft(tmp_path):
    c, o, s = ready(tmp_path)
    c = _apply(o, c, AmendFinal(kind="amend_final", draft_id=protocol(c)["final_draft"]["draft_id"],
                              limitations=["A bounded comparison.", "Keep this limitation."]), s)
    original = deepcopy(protocol(c)["final_draft"])
    before = science(c)
    c = _apply(o, c, AmendFinal(kind="amend_final", draft_id=original["draft_id"], replacements=[
        {"path": "/answer", "old_text": "A bounded comparison.", "new_text": "A corrected answer."},
        {"path": "/limitations/0", "old_text": "A bounded comparison.", "new_text": "A corrected limitation."}]), s)
    command = protocol(c)["final_draft"]["command"]
    assert command["answer"] == "A corrected answer."
    assert command["limitations"] == ["A corrected limitation.", "Keep this limitation."]
    assert original["command"]["limitations"][0] == "A bounded comparison."
    assert science(c) == before and protocol(c)["phase"] == "review"


def test_wrong_field_edit_returns_explicit_remedy_without_partial_commit(tmp_path):
    c, o, s = ready(tmp_path)
    c = _apply(o, c, AmendFinal(kind="amend_final", draft_id=protocol(c)["final_draft"]["draft_id"],
                              limitations=["A limitation-only sentence."]), s)
    before = c.model_dump(mode="json")
    draft = protocol(c)["final_draft"]
    with pytest.raises(OperationRejected) as error:
        _apply(o, c, AmendFinal(kind="amend_final", draft_id=draft["draft_id"], replacements=[
            {"old_text": draft["command"]["answer"], "new_text": "Must not be partially committed."},
            {"old_text": "A limitation-only sentence.", "new_text": "Wrong field."}]), s)
    issue = error.value.issues[0]
    assert issue["path"] == ["replacements", 1, "old_text"]
    assert "/answer" in issue["message"] and "/limitations/<index>" in issue["remedy"]
    assert c.model_dump(mode="json") == before


@pytest.mark.parametrize("fault", ["missing_index", "conflicting_list", "stale_draft"])
def test_indexed_limitation_boundary_failure_is_atomic(tmp_path, fault):
    c, o, s = ready(tmp_path)
    c = _apply(o, c, AmendFinal(kind="amend_final", draft_id=protocol(c)["final_draft"]["draft_id"],
                              limitations=["Existing limitation."]), s)
    before = c.model_dump(mode="json")
    args = {"kind": "amend_final", "draft_id": protocol(c)["final_draft"]["draft_id"], "replacements": [
        {"path": "/limitations/0", "old_text": "Existing limitation.", "new_text": "Edited limitation."}]}
    if fault == "missing_index":
        args["replacements"][0]["path"] = "/limitations/1"
    elif fault == "conflicting_list":
        args["limitations"] = ["Another list."]
    else:
        args["draft_id"] = "final:old"
    with pytest.raises(OperationRejected):
        _apply(o, c, AmendFinal.model_validate(args), s)
    assert c.model_dump(mode="json") == before


@pytest.mark.parametrize("path", ["/ranking", "/limitations", "/limitations/-1", "/limitations/01", "/answer/nested"])
def test_edit_paths_cannot_escape_deliverable_fields(path):
    with pytest.raises(ValidationError):
        TextReplacement(path=path, old_text="source", new_text="replacement")


def test_directory_does_not_claim_text_is_present(tmp_path):
    c, o, s = ready(tmp_path)
    c = read(c, o, s, view="directory")
    assert not shown(c, o)["edit_context"]["retained"]


def test_both_aliases_deduplicate_in_next_edit_request(tmp_path):
    c, o, s = ready(tmp_path)
    c = read(c, o, s)
    c = read(c, o, s, protocol(c)["final_draft"]["draft_id"], ["/content/answer", "/content/limitations"])
    c = put(c, latest_exchange=None, outcome=None)
    payload = shown(c, o)
    assert len(payload["edit_context"]["retained"]) == 2
    assert json.dumps(payload).count(protocol(c)["final_draft"]["command"]["answer"]) == 1


def test_missing_replay_route_must_include_text_not_just_a_read_receipt(tmp_path):
    c, o, s = ready(tmp_path)
    c = read(c, o, s, paths=["/answer"])
    c = put(c, outcome_request_id="different-request")
    assert data_in(shown(c, o)["edit_context"], "/answer") is not None


def test_old_receipts_and_checkpoints_remain_readable(tmp_path):
    c, o, s = ready(tmp_path)
    c = read(c, o, s)
    receipts = deepcopy(protocol(c)["query_receipts"])
    for r in receipts.values():
        r.pop("draft_id")
    c = CaseRun.model_validate(put(c, query_receipts=receipts, edit_reads={}).model_dump(mode="json"))
    assert not shown(c, o)["edit_context"]["retained"]
    rows = read_records(c, ReadEvidence(kind="read_evidence", records=[{
        "ref": protocol(c)["final_draft"]["draft_id"], "paths": ["/content/answer"]}]))
    assert rows[0]["data"]["/content/answer"] == protocol(c)["final_draft"]["command"]["answer"]


def test_wrong_case_receipt_is_not_used(tmp_path):
    c, o, s = ready(tmp_path)
    c = read(c, o, s)
    receipts = deepcopy(protocol(c)["edit_reads"])
    next(iter(receipts.values()))["case_id"] = "other-case"
    with pytest.raises(RuntimeError, match="another case"):
        edit_context(put(c, edit_reads=receipts))


@pytest.mark.parametrize("legacy", [False, True])
def test_reviewer_read_does_not_evict_planners_editing_text(tmp_path, legacy):
    from mechcal.runtime import command_query_state

    c, o, s = ready(tmp_path)
    c = read(c, o, s)
    if legacy:
        saved = c.model_dump(mode="json")
        saved["runtime"]["command_protocol"].pop("edit_reads")
        c = CaseRun.model_validate(saved)
    c = command_query_state.remember(c, protocol(c)["outcome"], role="final_reviewer")
    c = put(c, latest_exchange=None, outcome=None)
    assert data_in(shown(c, o)["edit_context"], "/answer") == protocol(c)["final_draft"]["command"]["answer"]


def test_failed_read_does_not_erase_previously_obtained_text(tmp_path):
    from mechcal.runtime import command_query_state

    c, o, s = ready(tmp_path)
    c = read(c, o, s)
    outcome = {"status": "rejected", "records": [{"ref": "delivery:answer", "status": "rejected",
        "selection": {"paths": [], "view": "data", "page": None},
        "issues": [{"code": "response_too_large", "message": "Read allowance consumed by earlier batch members."}]}]}
    c = command_query_state.remember(c, outcome, role="planner")
    c = put(c, latest_exchange=None, outcome=outcome)
    assert data_in(shown(c, o)["edit_context"], "/answer") == protocol(c)["final_draft"]["command"]["answer"]


def test_old_draft_read_and_unknown_ref_do_not_load_current_text(tmp_path):
    c, o, s = ready(tmp_path)
    for ref in ("final:old", "another-case:answer"):
        current = read(c, o, s, ref=ref)
        assert protocol(current)["outcome"]["status"] == "rejected"
        assert not shown(current, o)["edit_context"]["retained"]


def test_oversized_retained_field_is_explicitly_deferred(tmp_path):
    from mechcal.schemas.commands import Finalize

    c, o, s = ready(tmp_path)
    c = _apply(o, c, Finalize(kind="finalize", answer="Protocol fixture. " * 800), s)
    c = read(c, o, s, paths=["/answer"])
    context = shown(c, o)["edit_context"]
    assert context["fields"][0]["present_in_request"]
    assert context["retained"][0]["location"] == "latest_tool_result"
    assert context["retained"][0]["retained_for_next_request"] is False
    c = put(c, latest_exchange=None, outcome=None)
    context = shown(c, o)["edit_context"]
    assert context["deferred"][0]["path"] == "/answer"
    assert not context["fields"][0]["present_in_request"]
    assert sum(command_memory.json_bytes(row) for row in context["retained"]) < EDIT_BYTES


@pytest.mark.parametrize("compact", [True, False])
def test_edit_view_survives_task_context_and_compaction_modes(tmp_path, compact):
    c, o, s = ready(tmp_path)
    o.config = replace(o.config, command_compact_context=compact, command_task_context=compact)
    c = read(c, o, s)
    c = put(c, latest_exchange=None, outcome=None)
    expected = protocol(c)["final_draft"]["command"]["answer"]
    payload = planner_payload(o, c, native=True)
    if compact:
        c, payload = prepare_planner(c, payload)
    assert data_in(expand_text(payload)["edit_context"], "/answer") == expected


def test_review_correction_preserves_edit_text_and_can_submit(tmp_path):
    from mechcal.runtime.command_loop import _comment
    from mechcal.schemas.commands import ReviewFinal, SubmitFinal
    from test_native_commands import NativeFake

    c, o, s = ready(tmp_path)
    before = science(c)
    c = read(c, o, s)
    draft = protocol(c)["final_draft"]
    c = _apply(o, c, ReviewFinal(kind="review_final", draft_id=draft["draft_id"]), s)
    broken = reply("record_review", {"text": "Incomplete protocol fixture.", "findings": []})
    broken["complete"] = False
    o.deps.mechanism_agenda.llm_client = NativeFake([
        broken, reply("record_review", {"text": "Protocol fixture; scientific limits unchanged.", "findings": []})])
    c = _comment(o, c, s)
    assert protocol(c)["phase"] == "review"
    assert data_in(shown(c, o)["edit_context"], "/answer") == draft["command"]["answer"]
    c = _comment(o, c, s)
    c = _apply(o, c, SubmitFinal(kind="submit_final", draft_id=draft["draft_id"]), s)
    assert c.status == "finalized" and science(c) == before
