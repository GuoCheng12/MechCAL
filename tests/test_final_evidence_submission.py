"""Delivery completeness, not chemical correctness or a guaranteed score."""

from copy import deepcopy

import pytest
from pydantic import TypeAdapter, ValidationError

from mechcal.runtime.final_evidence import validate_final_evidence
from mechcal.runtime.output_assembler import OutputAssembler
from mechcal.runtime.retry import RequestValidationError
from mechcal.runtime.structured_output import PlannerOutputContract
from mechcal.schemas.final_evidence import FinalEvidenceSubmission
from mechcal.schemas.judgment_commands import FinalJudgmentCommand
from test_evidence_delivery_contract import support
from test_incremental_judgment import prepared
from test_judgment_commands import command
from test_question_loop import action


def submission_for(portfolio, arguments=()):
    """Explicit synthetic decisions for offline model fixtures only."""
    result = []
    for h in sorted(portfolio.hypotheses, key=lambda h: (
        h.differential_priority if h.differential_priority is not None else h.confidence
    ), reverse=True)[:3]:
        ids = [a.support_id for a in arguments if a.target_label == h.name]
        result.append({"label": h.name, "status": "submitted", "support_ids": ids} if ids else {
            "label": h.name, "status": "no_evidence", "reason": "Offline fixture submits no argument."})
    return result


def parsed(rows):
    return TypeAdapter(list[FinalEvidenceSubmission]).validate_python(rows)


def final_command(case):
    return dict(action="revise_final", update=command(case).model_dump(mode="json"),
                final_action=action(case.case_id, 2).model_dump(mode="json"), audit_responses={},
                evidence_submission=submission_for(case.portfolio, case.mechanism_support_arguments))


def test_final_command_cannot_silently_omit_evidence_delivery():
    data = final_command(prepared())
    del data["evidence_submission"]
    with pytest.raises(ValidationError) as error:
        FinalJudgmentCommand.model_validate(data)
    assert ("evidence_submission",) in {e["loc"] for e in error.value.errors()}


@pytest.mark.parametrize("bad", [
    {"label": "RIM_RIR_RIV", "status": "submitted", "support_ids": []},
    {"label": "RIM_RIR_RIV", "status": "no_evidence", "reason": "  "},
    {"label": "RIM_RIR_RIV", "status": "no_evidence", "reason": "No link", "support_ids": ["S1"]},
])
def test_invalid_declarations_rejected(bad):
    with pytest.raises(ValidationError):
        parsed([bad])


def test_strict_wire_preserves_both_submission_variants():
    case = prepared()
    data = final_command(case)
    api = PlannerOutputContract({"output_contract": FinalJudgmentCommand.model_json_schema()})
    assert api.decode(api.encode(data)) == data


@pytest.mark.parametrize("fault", ["missing", "duplicate", "foreign_id", "wrong_label", "duplicate_id", "false_empty"])
def test_delivery_matches_updated_top3_and_arguments(fault):
    case = prepared()
    rows = submission_for(case.portfolio, case.mechanism_support_arguments)
    submitted = next(r for r in rows if r["status"] == "submitted")
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif fault == "foreign_id":
        submitted["support_ids"] = ["foreign"]
    elif fault == "wrong_label":
        other = next(r for r in rows if r["label"] != submitted["label"])
        other.update(status="submitted", support_ids=submitted["support_ids"])
        other.pop("reason", None)
    elif fault == "duplicate_id":
        submitted["support_ids"] *= 2
    else:
        submitted.update(status="no_evidence", reason="No support")
        submitted.pop("support_ids")
    before = case.model_dump_json()
    with pytest.raises(RequestValidationError):
        validate_final_evidence(case.portfolio, case.mechanism_support_arguments, parsed(rows))
    assert case.model_dump_json() == before


def test_delivery_is_keyed_by_label_not_duplicate_array_order():
    case = prepared()
    rows = submission_for(case.portfolio, case.mechanism_support_arguments)
    validate_final_evidence(case.portfolio, case.mechanism_support_arguments, parsed(list(reversed(rows))))


def test_explicit_no_evidence_is_preserved_as_limitation_not_positive_evidence():
    case = prepared().touch(mechanism_support_arguments=[])
    rows = submission_for(case.portfolio)
    validate_final_evidence(case.portfolio, [], parsed(rows))
    case.runtime["final_evidence_submission"] = rows
    predictions = OutputAssembler().build_mechanism_predictions(case)
    assert all(not p.evidence for p in predictions)
    assert all(any("No evidence submitted:" in x for x in p.limitations) for p in predictions)


def test_existing_arguments_are_retained_and_new_arguments_export_exactly():
    from mechcal.runtime.judgment_commands import materialize_judgment
    from mechcal.runtime.questions import apply_assimilation
    from mechcal.schemas.judgment_commands import JudgmentCommand
    case = prepared()
    added = support("new-offline-argument", "input:smiles")
    sparse = command(case, mechanism_support_arguments=[added])
    api = PlannerOutputContract({"output_contract": JudgmentCommand.model_json_schema()})
    sparse = JudgmentCommand.model_validate(api.decode(api.encode(sparse.model_dump(mode="json"))))
    updated = apply_assimilation(case, materialize_judgment(case, sparse))
    rows = submission_for(updated.portfolio, updated.mechanism_support_arguments)
    validate_final_evidence(updated.portfolio, updated.mechanism_support_arguments, parsed(rows))
    assert len(updated.mechanism_support_arguments) == len(case.mechanism_support_arguments) + 1
    prediction = next(p for p in OutputAssembler().build_mechanism_predictions(updated) if p.label == added.target_label)
    assert any(d.finding == added.finding and d.evidence_refs == added.observation_refs for d in prediction.evidence_details)


def test_export_detects_arguments_lost_after_final_submission():
    case = prepared()
    case.runtime["final_evidence_submission"] = submission_for(case.portfolio, case.mechanism_support_arguments)
    with pytest.raises(RequestValidationError, match="support_ids"):
        OutputAssembler().build_mechanism_predictions(case.touch(mechanism_support_arguments=[]))


def test_missing_argument_repair_can_edit_both_delivery_and_argument_records():
    from mechcal.runtime.judgment_draft import JudgmentDraft
    case = prepared().touch(mechanism_support_arguments=[])
    data = final_command(case)
    data["evidence_submission"][0] = {"label": data["evidence_submission"][0]["label"],
                                      "status": "submitted", "support_ids": ["new"]}
    with pytest.raises(RequestValidationError) as error:
        validate_final_evidence(case.portfolio, [], parsed(data["evidence_submission"]))
    draft = JudgmentDraft(case.case_id, 1, "planner_final_revision", FinalJudgmentCommand, {
        "rejected_response": data, "error": str(error.value), "issues": error.value.issues})
    assert {tuple(t["path"]) for t in draft.targets.values()} == {
        ("evidence_submission", 0), ("update", "mechanism_support_arguments")}


def test_evidence_delivery_still_runs_when_auditor_disabled(tmp_path, monkeypatch):
    from dataclasses import replace
    from test_judgment_commands import command_loop
    from test_question_loop import NOTE
    o, client = command_loop(tmp_path, monkeypatch)
    o.config = replace(o.config, enable_support_auditor=False)
    result = o.run(smiles="CCO", user_query=NOTE, case_id="command-loop")
    assert result.status == "finalized"
    assert len(result.runtime["final_evidence_submission"]) == 3
    assert result.runtime["question_loop"]["final_audit"]["status"] == "disabled"
    assert any(p["stage"] == "planner_final_revision" for p in client.calls)


def test_no_evidence_conflict_allows_explicit_withdrawal_instead_of_forcing_support():
    from mechcal.runtime.judgment_draft import JudgmentDraft
    case = prepared()
    data = final_command(case)
    index = next(i for i, row in enumerate(data["evidence_submission"]) if row["status"] == "submitted")
    label = data["evidence_submission"][index]["label"]
    data["evidence_submission"][index] = {"label": label, "status": "no_evidence", "reason": "The link is unjustified."}
    with pytest.raises(RequestValidationError) as error:
        validate_final_evidence(case.portfolio, case.mechanism_support_arguments, parsed(data["evidence_submission"]))
    draft = JudgmentDraft(case.case_id, 1, "planner_final_revision", FinalJudgmentCommand, {
        "rejected_response": data, "error": str(error.value), "issues": error.value.issues})
    assert {tuple(t["path"]) for t in draft.targets.values()} == {
        ("evidence_submission", index), ("update", "withdrawn_support_ids")}


def test_missing_submission_is_repaired_without_rerunning_tools_or_mutating_case():
    from test_delivery_and_final_revision import final_fixture
    from test_judgment_commands import initial_command, planner, repair_fields
    from mechcal.runtime.questions import state, apply_final_revision
    from test_question_loop import REGISTRY
    case, revision = final_fixture()
    case.runtime["incremental_judgment"] = True
    state(case)["final_audit"] = {"status": "disabled", "review": None}
    data = dict(action="revise_final",
                update=initial_command(revision.update, state(case)["questions"]).model_dump(mode="json"),
                final_action=revision.final_action.model_dump(mode="json"),
                audit_responses=revision.audit_responses)
    rows = submission_for(revision.update.portfolio, revision.update.mechanism_support_arguments)
    agent, client = planner(case, [data, repair_fields(evidence_submission=rows)])
    before = case.model_dump_json()
    final = agent.revise_final_arguments(case)
    assert len(client.calls) == 2
    assert case.model_dump_json() == before
    updated, _ = apply_final_revision(case, final, REGISTRY)
    assert updated.runtime["final_evidence_submission"] == rows
    assert updated.evidence_ledger == case.evidence_ledger
    assert any(p.evidence for p in OutputAssembler().build_mechanism_predictions(updated))
