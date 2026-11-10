"""Review scheduling does not decide science or weaken the submission contract."""

from copy import deepcopy
from dataclasses import replace

import pytest

from mechcal.runtime import command_followup as followup
from mechcal.runtime import command_operations as ops
from mechcal.runtime.command_delivery import reviewed
from mechcal.runtime.command_loop import _apply, _comment, _execute, planner_payload, save
from mechcal.runtime.command_review import interim_review_trigger, reference_contract, review_basis
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import Finalize, ReviseJudgment, SubmitFinal
from test_command_context_efficiency import expand
from test_command_handoff import prose, unchanged_science
from test_command_protocol import case, command, computed, judgment, orchestrator, rank, ScriptClient, store
from test_native_commands import NativeFake


def reviewed_case(tmp_path, *, native=False, text="Correct the source attribution. A comparison may help."):
    c, o = case(), orchestrator(tmp_path, reviews=True)
    o.config = replace(o.config, command_native_tools=native)
    s = store(tmp_path, c)
    c = _apply(o, c, judgment(), s)
    assert ops.protocol(c)["phase"] == "review"
    client = NativeFake([prose(text)]) if native else ScriptClient([text])
    o.deps.photophysics_arbiter.llm_client = client
    after = _comment(o, c, s)
    unchanged_science(c, after)
    return after, o, s


@pytest.mark.parametrize("change", ["text", "confidence", "priority", "support", "citation", "note"])
def test_same_source_edits_return_to_action_selection(tmp_path, change):
    c, o, s = reviewed_case(tmp_path)
    edit = judgment().model_copy(deep=True)
    if change == "text":
        edit.changes[0].text = "A materially different interpretation, not automatically endorsed."
    elif change in {"confidence", "priority"}:
        setattr(edit.changes[1], change, .2)
    elif change == "support":
        edit.changes[1].support = "partial"
    elif change == "citation":
        from mechcal.schemas.commands import ReadRecord
        edit.changes[0].citations = [ReadRecord(ref="input:smiles", paths=["/smiles"])]
    else:
        edit.changes[1].rationale = "Reconsidered using the same observations."
    new = _apply(o, c, edit, s)
    p = ops.protocol(new)
    assert p["phase"] == "choose"
    assert p["revision"] > p["interim_review"]["revision"]
    assert p["review"] == ops.protocol(c)["review"]
    assert not p.get("final_review")
    assert interim_review_trigger(new) is None


@pytest.mark.parametrize("change", ["new_success", "new_failure", "changed_value", "changed_parameters", "new_scope"])
def test_new_facts_or_claim_scope_schedule_review(tmp_path, change):
    c, o, s = reviewed_case(tmp_path)
    edit = judgment().model_copy(deep=True)
    edit.changes[0].text = "Revised after a scientific update."
    if change.startswith("new_") and change != "new_scope":
        c = ops.put(c, results={"result:R1": {"status": "success" if change == "new_success" else "failed"}})
    elif change == "changed_value":
        c = c.touch(input=c.input.model_copy(update={"smiles": "CCO"}))
    elif change == "changed_parameters":
        c = ops.put(c, actions={"op:1": {"tool_args": {"temperature": 298}}})
    else:
        edit = ReviseJudgment.model_validate({"kind": "revise_judgment", "changes": [rank("PACKING_HOST_MATRIX_CONFINEMENT")]})
    new = _apply(o, c, edit, s)
    assert ops.protocol(new)["phase"] == "review"
    assert ops.protocol(new)["review_trigger"] == ("new_claim_scope" if change == "new_scope" else "new_observations")


@pytest.mark.parametrize("changed", [False, True])
def test_explicit_review_is_honored_even_for_identical_records(tmp_path, changed):
    c, o, s = reviewed_case(tmp_path)
    edit = judgment().model_copy(deep=True)
    edit.request_review = True
    if changed:
        edit.changes[0].text = "Interpretation needs another scientific check."
    new = _apply(o, c, edit, s)
    assert ops.protocol(new)["phase"] == "review"
    assert ops.protocol(new)["review_trigger"] == "explicit"
    assert ops.protocol(new)["revision"] == ops.protocol(c)["revision"] + int(changed)


def test_unavailable_review_does_not_create_scheduling_receipt(tmp_path):
    c, o, _ = reviewed_case(tmp_path, text="")
    assert ops.protocol(c)["review"]["status"] == "unavailable"
    assert not ops.protocol(c)["interim_review"]
    assert interim_review_trigger(c) == "initial_claims"


def test_correction_then_compute_and_checkpoint_preserve_sources(tmp_path):
    c, o, s = reviewed_case(tmp_path, native=True)
    c = computed(c)
    edit = judgment().model_copy(deep=True)
    edit.changes[0].text = "Attribution corrected; not independent evidence."
    c = save(_apply(o, c, edit, s), s)
    restored = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    assert ops.protocol(restored)["interim_review"] == ops.protocol(c)["interim_review"]
    assert interim_review_trigger(restored) is None
    scheduled = _apply(o, restored, command(), s)
    assert ops.protocol(scheduled)["phase"] == "execute"
    executed = _execute(o, scheduled, s)
    assert len(ops.protocol(executed)["results"]) == 1
    assert interim_review_trigger(executed) == "new_observations"
    assert ops.protocol(executed)["arguments"] == ops.protocol(c)["arguments"]


def test_deferred_edits_still_need_exact_final_review(tmp_path):
    c, o, s = reviewed_case(tmp_path)
    o.deps.mechanism_agenda.llm_client = ScriptClient(["Reporting error remains."])
    c = _comment(o, _apply(o, c, Finalize(kind="finalize", answer="First draft."), s), s)
    old_id = ops.protocol(c)["final_draft"]["draft_id"]
    edit = judgment().model_copy(deep=True)
    edit.changes[0].text = "A scientifically different assertion."
    c = _apply(o, c, edit, s)
    assert ops.protocol(c)["phase"] == "choose"
    with pytest.raises(ops.OperationRejected):
        _apply(o, c, SubmitFinal(kind="submit_final", draft_id=old_id), s)
    c = _apply(o, c, Finalize(kind="finalize", answer="Revised draft."), s)
    assert ops.protocol(c)["phase"] == "review"
    assert not reviewed(c, ops.protocol(c)["final_draft"])
    o.deps.mechanism_agenda.llm_client = ScriptClient(["The new assertion is unsupported."])
    new = _comment(o, c, s)
    sent = o.deps.mechanism_agenda.llm_client.calls[0]["payload"]
    assert sent["arguments"][0]["text"] == edit.changes[0].text
    assert "unsupported" in ops.protocol(new)["review"]["text"]


@pytest.mark.parametrize("native", [False, True])
def test_reviewer_and_validator_share_citation_eligibility(tmp_path, native):
    c, o, s = reviewed_case(tmp_path, native=native)
    c = followup.enable(ops.put(c, results={"result:ok": {"status": "success"}, "result:fail": {"status": "failed"}},
                              actions={"op:1": {"result_id": "result:fail"}}))
    contract = reference_contract(c, ops.records(c))
    for ref in ops.records(c):
        for field, allow_failed in (("support_refs", False), ("limitation_refs", True)):
            issues = ops.reference_issues(c, [ref], [field], scientific=True, allow_failed=allow_failed)
            assert (not issues) == (ref in contract[field])
    assert "followup:state" in contract["context_only_refs"]
    assert "op:1" not in contract["support_refs"]
    assert "result:fail" not in contract["support_refs"]
    issue = ops.reference_issues(c, ["followup:state"], ["limitation_refs"], scientific=True, allow_failed=True)[0]
    assert "omit it from both" in issue["remedy"]
    edit = judgment().model_copy(deep=True)
    edit.request_review = True
    c = _apply(o, c, edit, s)
    client = NativeFake([prose()]) if native else ScriptClient(["Use actual observations, not follow-up metadata."])
    o.deps.photophysics_arbiter.llm_client = client
    _comment(o, c, s)
    sent = expand(client.requests[0]["payload"])["sections"] if native else client.calls[0]["payload"]
    assert "followup:state" in sent["reference_contract"]["context_only_refs"]
    assert set(sent["reference_ids"]) == set().union(*map(set, sent["reference_contract"].values()))


def test_reads_and_explanations_do_not_create_new_observations(tmp_path):
    c, o, _ = reviewed_case(tmp_path)
    before = review_basis(c)
    c = ops.put(c, explanations={"result:1": {"status": "available", "text": "Interpretation only."}},
                review={"status": "available", "text": "More advice."})
    assert review_basis(c) == before
    before_state = deepcopy(c.model_dump(mode="json"))
    payload = planner_payload(o, c)
    assert payload["interim_review_status"]["automatic_review_reason"] is None
    assert c.model_dump(mode="json") == before_state
