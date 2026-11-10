"""New advice is a decision event, not another observation or mandatory work."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from mechcal.agents.command_tools import ask_tools
from mechcal.runtime import command_exploration_review as er, command_investigation as inv
from mechcal.runtime import command_hypotheses as h
from mechcal.runtime import command_review_workflow as workflow
from mechcal.runtime.command_compact import comparison_rows, expand_text
from mechcal.runtime.command_loop import _local_error, planner_payload
from mechcal.runtime.command_operations import OperationRejected, check_refs, protocol, put, read_records
from mechcal.runtime.command_task_context import prepare_planner, task_for
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import Finalize, ReadEvidence, UpdateHypothesesWithInvestigation
from test_command_protocol import TOOL, store
from test_experiment_options import suggestion
from test_investigation_policy import choice, select
from test_native_commands import NativeFake, reply
from test_reviewer_exploration import prepared, finish_reply, turn


def reviewed_case(tmp_path):
    c, o = prepared(tmp_path)
    o.config = replace(o.config, command_task_context=True, command_worker_tasks=True)
    c = workflow.enable(c)
    c = select(c, decision="conclude")
    c = put(c, arguments={"old:argument": {"id": "old:argument", "label": "ICT_TICT_CT", "refs": ["result:new"],
        "text": "Previously accepted interpretation."}},
        review={"status": "available", "text": "OLD_REVIEW: no more editing needed.", "findings": []})
    return c, o


def offered(c, o, s):
    o.deps.photophysics_arbiter.llm_client = NativeFake([finish_reply(experiments=[suggestion()])])
    return turn(er.start(c, "before_conclusion"), o, s)


def test_new_proposals_restore_full_interface_not_old_revision(tmp_path):
    c, o = reviewed_case(tmp_path)
    s = store(tmp_path, c)
    c = offered(c, o, s)
    before = deepcopy(c.model_dump(mode="json"))
    payload = planner_payload(o, c, native=True)
    c, view = prepare_planner(c, payload)
    view = expand_text(view)
    assert view["context_task"] == "exploration_decision"
    assert view["planner_interface"] == "full" and not view["deferred_context"]
    assert view["exploration_state"]["new_result_ids"] == []
    assert view["proposal_decision"]["question_ids"] == inv.pending_proposals(c)["question_ids"]
    option = comparison_rows(view)[0]
    assert option["next_test"] == suggestion()["next_test"]
    assert option["outcomes"] == suggestion()["outcomes"]
    assert view["review_feedback"]["status"] == "historical"
    assert "OLD_REVIEW" not in json.dumps(view) and "PRIVATE_NOTE" not in json.dumps(view)
    records = read_records(c, ReadEvidence(kind="read_evidence", records=[{"ref": "review:feedback"}]))
    assert "OLD_REVIEW" in json.dumps(records)
    with pytest.raises(OperationRejected):
        check_refs(c, ["review:feedback"], [], scientific=True)
    fake = NativeFake([reply("update_hypotheses", {"note": "Decide.", "investigation": choice()})])
    ask_tools(fake, o.deps.capability_registry, [TOOL], view, s)
    request = fake.requests[0]
    names = {t["name"] for t in request["tools"]}
    assert "delegate_computation" in names and "describe_tools" in names
    assert "compute_" + TOOL.replace(".", "_") in names
    assert "Experimental measurements are not available" in request["system_prompt"]
    for key in ("ranking", "arguments", "rounds", "results", "exploration", "review"):
        assert protocol(c)[key] == before["runtime"]["command_protocol"][key]


@pytest.mark.parametrize("decision", ["investigate", "conclude"])
def test_handoff_survives_resume_and_is_consumed_by_either_valid_choice(tmp_path, decision):
    c, o = reviewed_case(tmp_path)
    c = offered(c, o, store(tmp_path, c))
    c = CaseRun.model_validate_json(c.model_dump_json())
    pending = deepcopy(inv.pending_proposals(c))
    before = deepcopy(protocol(c)["exploration"]["entries"])
    with pytest.raises(OperationRejected, match="No current conclude"):
        inv.guard(c, Finalize(kind="finalize", answer="Not evaluated yet."))
    c = select(c, decision=decision, considered_question_ids=pending["question_ids"])
    assert inv.pending_proposals(c) is None
    assert inv.current(c)["proposal_handoff"] == pending
    assert protocol(c)["exploration"]["entries"] == before
    if decision == "investigate":
        c, payload = prepare_planner(c, planner_payload(o, c, native=True))
        assert payload["planner_interface"] == "full"  # A decision-only turn must not fall back to revision.
    else:
        inv.guard(c, Finalize(kind="finalize", answer="A bounded conclusion is allowed."))


def test_reads_and_local_errors_do_not_erase_pending_decision(tmp_path):
    c, o = reviewed_case(tmp_path)
    s = store(tmp_path, c)
    c = offered(c, o, s)
    pending = deepcopy(inv.pending_proposals(c))
    read_records(c, ReadEvidence(kind="read_evidence", records=[{"ref": "review:feedback"}]))
    c = _local_error(c, [{"path": ["arguments"], "code": "test", "message": "Invalid parameter.",
                          "remedy": "Correct it."}], s)
    assert inv.pending_proposals(c) == pending
    assert task_for(c, {}) == "exploration_decision"
    with pytest.raises(OperationRejected):
        select(c, question_id="unknown")
    assert inv.pending_proposals(c) == pending


def test_repair_has_priority_without_losing_exploration_handoff(tmp_path):
    c, o = reviewed_case(tmp_path)
    c = offered(c, o, store(tmp_path, c))
    assert task_for(c, {"judgment_draft": {"draft_id": "draft:test"}}) == "repair"
    assert task_for(c, {}) == "exploration_decision"


def test_duplicate_or_empty_advice_does_not_reopen_consumed_decision(tmp_path):
    c, o = reviewed_case(tmp_path)
    s = store(tmp_path, c)
    c = offered(c, o, s)
    c = select(c, decision="conclude")
    before = deepcopy(inv.current(c))
    c = offered(c, o, s)
    assert inv.pending_proposals(c) is None and inv.current(c) == before
    o.deps.photophysics_arbiter.llm_client = NativeFake([finish_reply()])
    c = turn(er.start(c, "before_conclusion"), o, s)
    assert inv.pending_proposals(c) is None


def test_handoff_case_ownership_and_missing_ids(tmp_path):
    c, o = reviewed_case(tmp_path)
    with pytest.raises(RuntimeError, match="missing case-local"):
        inv.offer_proposals(c, ["foreign:question"])
    c = offered(c, o, store(tmp_path, c))
    protocol(c)["investigation_policy"]["case_id"] = "foreign"
    with pytest.raises(RuntimeError, match="ownership"):
        inv.pending_proposals(c)


def test_old_checkpoint_without_handoff_still_routes_normally(tmp_path):
    c, o = reviewed_case(tmp_path)
    assert inv.pending_proposals(c) is None
    payload = expand_text(planner_payload(o, c, native=True))
    payload["interim_review_status"]["automatic_review_reason"] = None
    assert task_for(c, payload) == "revision"
    assert "OLD_REVIEW" in json.dumps(payload["review_feedback"])


def test_belief_only_update_does_not_consume_new_experiment_advice(tmp_path):
    c, o = reviewed_case(tmp_path)
    c = offered(c, o, store(tmp_path, c))
    before = deepcopy(inv.pending_proposals(c))
    c, _ = h.update(c, UpdateHypothesesWithInvestigation(kind="update_hypotheses", note="Retain judgments."))
    assert inv.pending_proposals(c) == before
