"""Offline draft/patch tests; scripted judgments are not scientific results."""

import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from mechcal.runtime.case_memory import CaseMemoryStore
from mechcal.runtime.judgment_commands import materialize_judgment, snapshot_feedback
from mechcal.runtime.judgment_draft import JudgmentDraft
from mechcal.runtime.planner_context import build_context
from mechcal.runtime.persistence import RunStore
from mechcal.runtime.question_changes import apply_question_changes, question_view
from mechcal.runtime.questions import public_payload, state, validate_assimilation
from mechcal.runtime.retry import PlannerCommandRejected, RejectedResponseError, RequestValidationError
from mechcal.schemas import CaseRun
from mechcal.schemas.judgment_commands import FinalJudgmentCommand, JudgmentCommand, QuestionChange, QuestionUpdate
from mechcal.schemas.memory import MemoryQuery
from test_case_memory import fixture
from test_decision_economy import economy_start
from test_incremental_judgment import prepared
from test_judgment_commands import command, initial_command, planner, repair_fields
from test_question_loop import REGISTRY


def schema_draft(data, *, model=JudgmentCommand, case_id="case", revision=2):
    with pytest.raises(ValidationError) as caught:
        model.model_validate(data)
    return JudgmentDraft(case_id, revision, "assimilate", model, {
        "rejected_response": data, "error": str(caught.value),
        "issues": caught.value.errors(include_url=False, include_context=False, include_input=False),
    })


def test_question_create_update_retention_and_legacy_roundtrip():
    case, update = economy_start()
    q = initial_command(update).questions[0]
    created = apply_question_changes({}, [q])[0]
    previous = {q.question_id: created.model_dump(mode="json")}
    previous[q.question_id]["status"] = "computable"
    previous[q.question_id]["evidence_refs"] = ["input:smiles"]
    baseline = deepcopy(previous)
    patch = QuestionChange(operation="update", question_id=q.question_id,
                           answer_state="partial", assessment="A bounded partial finding.",
                           observable=None, evidence_refs=[])
    result = apply_question_changes(previous, [patch])[0]
    assert result.observable == created.observable
    assert result.question == created.question and result.expected_basis == created.expected_basis
    assert result.status == "computable" and result.answer_state == "partial"
    assert result.evidence_refs == [] and previous == baseline
    state(case)["questions"] = {q.question_id: result.model_dump(mode="json")}
    assert CaseRun.model_validate_json(case.model_dump_json()) == case


@pytest.mark.parametrize("fault", ["duplicate", "existing_create", "unknown_update", "empty_update", "missing_create"])
def test_question_changes_reject_identity_and_creation_errors_atomically(fault):
    case = prepared()
    prior = state(case)["questions"]
    qid = next(iter(prior))
    data = {**question_view(prior[qid]), "operation": "update"}
    if fault == "existing_create":
        data["operation"] = "create"
    elif fault == "unknown_update":
        data["question_id"] = "NEW"
    elif fault == "empty_update":
        data = dict(operation="update", question_id=qid)
    elif fault == "missing_create":
        data = dict(operation="create", question_id="NEW", answer_state="open")
    changes = [QuestionChange.model_validate(data)] * (2 if fault == "duplicate" else 1)
    before = deepcopy(prior)
    with pytest.raises(RequestValidationError):
        apply_question_changes(prior, changes)
    assert prior == before


@pytest.mark.parametrize("field", ["status", "execution_state", "record_type", "schema_version"])
def test_question_wire_does_not_accept_persisted_runtime_fields(field):
    with pytest.raises(ValidationError):
        QuestionChange.model_validate({"operation": "update", "question_id": "Q1", field: "partial"})


@pytest.mark.parametrize("answer", ["open", "partial", "answered", "deferred"])
def test_context_and_memory_use_scientific_progress_not_legacy_status(tmp_path, answer):
    case, store = fixture(tmp_path, incremental_judgment=True)
    q = state(case)["questions"]["Q1"]
    q.update(status="computable", answer_state=answer, assessment="The actual partial interpretation.")
    before = case.model_dump(mode="json")
    payload = public_payload(case, REGISTRY, stage="planner_assimilate")
    projected = build_context(payload)["questions"][0]
    assert "status" not in projected and "execution_state" not in projected
    assert ("assessment" in projected) == (answer in {"open", "partial"})
    record = CaseMemoryStore(store, case).read(
        MemoryQuery(action="read_case_records", record_ids=["Q1"]))["records"][0]["record"]
    assert record["assessment"] == q["assessment"]
    assert record == question_view(q) and case.model_dump(mode="json") == before


def test_r006_style_status_error_and_header_omissions_preserve_good_content():
    case = prepared()
    good = command(case, questions=[QuestionUpdate(operation="update", question_id="Q1",
                    answer_state="partial", assessment="Partial evidence, not a completed answer.")]).model_dump(mode="json")
    bad = deepcopy(good)
    bad["questions"][0]["answer_state"] = "computable"
    bad.pop("decision_assessment")
    bad.pop("coverage_response")
    draft = schema_draft(bad)
    fixed = draft.apply(repair_fields(answer_state="partial",
                        decision_assessment=good["decision_assessment"],
                        coverage_response=good["coverage_response"])(draft.payload()))
    assert fixed == good and draft.original == bad
    validate_assimilation(materialize_judgment(case, JudgmentCommand.model_validate(fixed)), case)


@pytest.mark.parametrize("fault", ["foreign_draft", "unknown_target", "duplicate", "wrong_operation", "no_value", "extra_field"])
def test_unauthorized_edits_never_replace_draft_or_accepted_state(fault):
    bad = command(prepared()).model_dump(mode="json")
    bad.pop("coverage_response")
    draft = schema_draft(bad)
    patch = repair_fields(coverage_response="Reviewed.")(draft.payload())
    if fault == "foreign_draft":
        patch["draft_id"] = "different"
    elif fault == "unknown_target":
        patch["edits"][0]["target_id"] = "different"
    elif fault == "duplicate":
        patch["edits"] *= 2
    elif fault == "wrong_operation":
        patch["edits"][0]["operation"] = "remove"
    elif fault == "no_value":
        patch["edits"][0].pop("value")
    else:
        patch["hypotheses"] = []
    with pytest.raises(RejectedResponseError) as caught:
        draft.apply(patch)
    assert caught.value.response == bad and draft.original == bad
    assert caught.value.issues == draft.feedback["issues"]


def test_draft_binding_rejects_cross_case_revision_and_stale_edits():
    bad = command(prepared()).model_dump(mode="json")
    bad.pop("coverage_response")
    draft = schema_draft(bad)
    patch = repair_fields(coverage_response="Reviewed.")(draft.payload())
    other_data = {**bad, "reason": "A different candidate."}
    for other in (schema_draft(bad, case_id="other"), schema_draft(bad, revision=3), schema_draft(other_data)):
        with pytest.raises(RejectedResponseError, match="draft_id"):
            other.apply(patch)


def test_invalid_json_during_repair_preserves_full_draft_and_attempt_budget():
    case = prepared()
    good = command(case).model_dump(mode="json")
    bad = {**good, "questions": [{"operation": "update", "question_id": "Q1", "status": "partial"}]}
    # Remove legacy status, retaining an explicit scientific update.
    bad["questions"][0]["answer_state"] = "partial"
    agent, client = planner(case, [bad, RejectedResponseError("Invalid JSON", "{broken"), repair_fields()])
    events = []
    before = case.model_dump()
    result = agent.assimilate(case, request_event_sink=events.append)
    drafts = [e for e in events if e["kind"] == "judgment_draft"]
    assert len(client.calls) == 3 and len(drafts) == 2
    assert drafts[0]["draft_id"] == drafts[1]["draft_id"]
    assert drafts[1]["response"] == bad
    assert result.questions[0].answer_state == "partial" and case.model_dump() == before
    assert result.decision_assessment.model_dump(mode="json") == good["decision_assessment"]
    agent, client = planner(case, [bad, RejectedResponseError("Invalid JSON", "{"), RejectedResponseError("Invalid JSON", "{")])
    with pytest.raises(PlannerCommandRejected):
        agent.assimilate(case)
    assert len(client.calls) == 3 and case.model_dump() == before


def test_scientific_validation_runs_after_patch_without_committing_invalid_refs():
    case = prepared()
    bad = command(case, questions=[QuestionUpdate(operation="update", question_id="Q1",
                            evidence_refs=["foreign"], answer_state="partial")]).model_dump(mode="json")
    bad.pop("coverage_response")
    agent, client = planner(case, [bad, repair_fields(coverage_response="Checked."),
                                   repair_fields(evidence_refs=["input:smiles"])])
    before = case.model_dump()
    result = agent.assimilate(case)
    assert client.calls[1]["payload"]["candidate_command"] == bad
    assert "scientific_context" in client.calls[1]["payload"]
    assert len(client.calls) == 3 and result.questions[0].evidence_refs == ["input:smiles"]
    assert result.questions[0].answer_state == "partial"
    assert result.coverage_response == "Checked." and case.model_dump() == before


@pytest.mark.parametrize("nested", [False, True])
def test_nested_sparse_record_selection_is_bound_to_stable_keys(nested):
    from mechcal.schemas.question_loop import QuestionAssimilation
    from test_boundary_decisions import start

    case, snapshot = start()
    case.runtime["incremental_judgment"] = True
    full = initial_command(snapshot).model_dump(mode="json")
    if nested:
        collection = ["ranking_update", "candidate_bases"]
        rows = full["ranking_update"]["candidate_bases"]
        field = "positive_refs"
    else:
        collection = ["portfolio", "hypotheses"]
        rows = full["hypotheses"]
        field = "evidence_refs"
    chosen = rows[-1]
    chosen[field] = ["foreign"]
    # Snapshot positions differ from the one-record command sent by the model.
    data = snapshot.model_dump(mode="json")
    parent = data
    for key in collection:
        parent = parent[key]
    parent[-1][field] = ["foreign"]
    enriched = snapshot_feedback(RequestValidationError([{
        "path": [*collection, len(rows) - 1, field], "message": "Invalid reference.",
    }]), QuestionAssimilation.model_validate(data))
    if nested:
        full["ranking_update"]["candidate_bases"] = [chosen]
    else:
        full["hypotheses"] = [chosen]
    draft = JudgmentDraft(case.case_id, 0, "assimilate", JudgmentCommand, {
        "rejected_response": full, "error": str(enriched), "issues": enriched.issues,
    })
    target = next(iter(draft.targets.values()))
    assert target["path"][-2:] == [0, field]
    fixed = draft.apply(repair_fields(**{field: []})(draft.payload()))
    expected = deepcopy(full)
    if nested:
        expected["ranking_update"]["candidate_bases"][0][field] = []
    else:
        expected["hypotheses"][0][field] = []
    assert fixed == expected


def test_nested_final_command_schema_target_leaves_other_parts_unchanged():
    from test_question_loop import action
    from test_final_evidence_submission import submission_for
    case = prepared()
    good = dict(action="revise_final", update=command(case).model_dump(mode="json"),
                final_action=action(case.case_id, 2).model_dump(mode="json"), audit_responses={},
                evidence_submission=submission_for(case.portfolio, case.mechanism_support_arguments))
    bad = deepcopy(good)
    del bad["update"]["coverage_response"]
    draft = schema_draft(bad, model=FinalJudgmentCommand)
    assert next(iter(draft.targets.values()))["path"] == ["update", "coverage_response"]
    fixed = draft.apply(repair_fields(coverage_response=good["update"]["coverage_response"])(draft.payload()))
    FinalJudgmentCommand.model_validate(fixed)
    assert fixed == good


def test_draft_repairs_share_memory_allowance_and_stay_out_of_evidence(tmp_path):
    case = prepared()
    state(case)["round_index"] = 1
    bad = {**command(case).model_dump(mode="json"), "obsolete": "DRAFT_ONLY_MARKER"}
    query = {"action": "read_case_records", "record_ids": ["Q1"]}
    agent, client = planner(case, [query, bad, query, repair_fields()])
    case.runtime["planner_case_memory"] = True
    before = case.model_dump()
    store = RunStore(tmp_path, case.case_id)
    store.initialize()
    events = []
    agent.assimilate(case, memory_store=store, request_event_sink=events.append)
    assert len(client.calls) == 4
    assert client.calls[-1]["payload"]["memory_access"]["queries_remaining"] == 2
    assert client.calls[-1]["payload"]["retrieved_memory"]
    assert case.model_dump() == before
    assert not CaseMemoryStore(store, case).search(
        MemoryQuery(action="search_case_records", query="DRAFT_ONLY_MARKER"))["total"]
    logs = [json.loads(line) for line in (store.root / "history/memory_queries.jsonl").read_text().splitlines()]
    operations = [e["operation"] for e in logs if e["kind"] == "memory_model_call"]
    assert operations == ["read_case_records", "decision", "read_case_records", "repair_judgment"]
    assert sum(e["kind"] == "attempt_started" for e in events) == 2


def test_nested_collection_selectors_map_each_sparse_level():
    raw = {"dispositions": [{"result_id": "X2", "mechanism_effects": [
        {"label": "ICT_TICT_CT", "evidence_refs": ["foreign"]}]}]}
    issue = {
        "snapshot_path": ["dispositions", 7, "mechanism_effects", 3, "evidence_refs"],
        "record_selectors": [
            {"collection_path": ["dispositions"], "selector": {"result_id": "X2"}},
            {"collection_path": ["dispositions", 7, "mechanism_effects"], "selector": {"label": "ICT_TICT_CT"}},
        ],
        "message": "Invalid reference.",
    }
    draft = JudgmentDraft("case", 2, "assimilate", JudgmentCommand, {
        "rejected_response": raw, "error": "Invalid reference.", "issues": [issue],
    })
    assert next(iter(draft.targets.values()))["path"] == ["dispositions", 0, "mechanism_effects", 0, "evidence_refs"]


def test_final_materialization_error_stays_under_update():
    raw = {"update": {"questions": [{"operation": "update", "question_id": "Q2", "assessment": "Partial."}]}}
    draft = JudgmentDraft("case", 2, "final", FinalJudgmentCommand, {
        "rejected_response": raw, "error": "Unknown ID.", "issues": [{
            "command_field": "update", "path": ["questions", 0, "operation"],
            "record_selector": {"question_id": "Q2"}, "message": "Unknown ID.",
        }],
    })
    assert next(iter(draft.targets.values()))["path"] == ["update", "questions", 0, "operation"]


def test_final_action_semantic_error_is_not_redirected_to_judgment_update():
    raw = {"update": {"reason": "Unaffected judgment."}, "final_action": {"continuation_assessment": None}}
    draft = JudgmentDraft("case", 2, "final", FinalJudgmentCommand, {
        "rejected_response": raw, "error": "Missing continuation assessment.", "issues": [{
            "command_field": "final_action", "snapshot_path": ["continuation_assessment"],
            "message": "Missing continuation assessment.",
        }],
    })
    assert next(iter(draft.targets.values()))["path"] == ["final_action", "continuation_assessment"]


def test_missing_required_scientific_content_has_context_but_enum_patch_is_focused():
    data = command(prepared()).model_dump(mode="json")
    missing = deepcopy(data)
    del missing["decision_assessment"]
    assert schema_draft(missing).requires_context
    bad = deepcopy(data)
    bad["scope"] = "invalid"
    assert not schema_draft(bad).requires_context


def test_repair_contract_is_valid_json_schema_and_matches_actual_patches():
    from jsonschema import Draft202012Validator
    data = command(prepared()).model_dump(mode="json")
    missing = deepcopy(data)
    del missing["decision_assessment"]
    missing["unexpected"] = True
    draft = schema_draft(missing)
    schema = draft.payload()["output_contract"]
    Draft202012Validator.check_schema(schema)
    patch = repair_fields(decision_assessment=data["decision_assessment"])(draft.payload())
    Draft202012Validator(schema).validate(patch)
    assert draft.apply(patch) == data


def test_new_question_missing_fields_are_reported_together():
    change = QuestionChange(operation="create", question_id="Q_NEW", answer_state="open")
    with pytest.raises(RequestValidationError) as caught:
        apply_question_changes({}, [change])
    fields = {issue["path"][-1] for issue in caught.value.issues}
    assert {"mechanism_id", "question", "observable", "assessment"} <= fields
    assert all(issue["type"] == "missing" for issue in caught.value.issues)


def test_malformed_whole_command_repair_keeps_recoverable_original():
    original = command(prepared()).model_dump(mode="json")
    draft = JudgmentDraft("case", 2, "assimilate", JudgmentCommand, {
        "rejected_response": original, "error": "Unscoped semantic inconsistency.", "issues": [],
    })
    with pytest.raises(RejectedResponseError) as caught:
        draft.apply(repair_fields(whole_command=[])(draft.payload()))
    assert caught.value.response == original


@pytest.mark.parametrize("incremental", [False, True])
def test_coverage_result_reviewer_and_worker_share_question_view(incremental):
    from mechcal.agents.mechanism_agenda import _coverage_payload
    from mechcal.agents.photophysics_arbiter import PhotophysicsArbiterAgent
    from mechcal.agents.worker_explanation import explain_result
    from test_worker_explanation import ExplanationClient, fixture_result

    case = prepared()
    case.runtime["incremental_judgment"] = incremental
    request, result = fixture_result()
    state(case)["pending_results"] = [{"result_id": "fixture:result", "question_ids": ["Q1"],
                                      "evidence_refs": [], "capability_id": request.capability_id}]
    q = state(case)["questions"]["Q1"]
    q.update(status="computable", answer_state="partial")
    before = case.model_dump()
    expected = question_view(q) if incremental else q
    coverage = _coverage_payload(case, REGISTRY, "R001", previous_planner_action=None)
    assert coverage["question_state"]["Q1"] == expected
    client = ExplanationClient()
    explain_result(client, request, case, result, REGISTRY.require(request.capability_id), None)
    assert client.calls[0]["questions"] == [expected]

    class Capture:
        def complete_json(self, **kwargs):
            self.payload = kwargs["payload"]
            raise RuntimeError("Captured input without model access")

    capture = Capture()
    with pytest.raises(RuntimeError, match="Captured input"):
        PhotophysicsArbiterAgent(llm_client=capture).review_results(case)
    assert capture.payload["questions"] == [expected]
    assert case.model_dump() == before
