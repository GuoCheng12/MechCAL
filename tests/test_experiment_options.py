"""Experiment advice is discoverable work, never automatic scientific evidence."""

from copy import deepcopy
from dataclasses import replace

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from mechcal.runtime import command_experiment_options as options, command_followup as f
from mechcal.runtime import command_hypotheses as h, command_investigation as inv
from mechcal.runtime.command_impact_review import run_turn, validate_suggestions
from mechcal.runtime.command_loop import save
from mechcal.runtime.command_operations import OperationRejected, check_refs, protocol, put
from mechcal.runtime.command_review_workflow import comparison_options
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import ReportEvidenceImpactsWithExperiments, UpdateHypothesesWithInvestigation
from test_command_followup import observed, report, enabled
from test_command_protocol import LABELS, REGISTRY, TOOL, store
from test_investigation_policy import choice, select
from test_native_commands import NativeFake, reply


def prepared(tmp_path):
    o = enabled(tmp_path)
    o.config = replace(o.config, command_review_workflow=True, command_computational_review=True,
                       command_investigation_policy=True, command_experiment_options=True,
                       command_compact_context=True, enable_agenda_coverage_reviewer=True,
                       enable_support_auditor=True)
    return inv.enable(observed(), offer_experiments=True), o


def suggestion(**changes):
    return {"labels": [LABELS[2]], "source_refs": ["result:new"],
            "question": "Can a controlled change distinguish the untested alternative?",
            "next_test": "Inspect a structural prerequisite before a comparison.",
            "outcomes": "Present and absent prerequisites have different implications; neither proves the mechanism.",
            "route": {"capability_id": TOOL, "input_refs": ["input:smiles"]}, **changes}


def proposal(**changes):
    return ReportEvidenceImpactsWithExperiments(**report().model_dump(), experiments=[suggestion(**changes)])


def catalog(c):
    return REGISTRY.compact_catalog(c.artifact_manifest, amesp_enabled=False)


def accept(c, cmd=None):
    feedback = validate_suggestions(c, ["result:new"], cmd or proposal(), catalog=catalog(c),
                                    source_ids={"input:smiles", "result:new"})
    return f.accept_review(c, ["result:new"], feedback)


def test_advice_is_stored_without_mutating_science_or_calling_tools(tmp_path):
    c, _ = prepared(tmp_path)
    before = deepcopy(c.model_dump(mode="json"))
    out = accept(c)
    qid, question = next(iter(f.state(out).questions.items()))
    assert question.origin == "impact_reviewer" and question.source_refs == ["result:new"]
    assert question.value.id == qid and question.trials == []
    for field in ("portfolio", "evidence_ledger", "artifact_manifest"):
        assert out.model_dump(mode="json")[field] == before[field]
    for field in ("results", "arguments", "ranking", "rounds", "revision", "exploration"):
        assert protocol(out)[field] == before["runtime"]["command_protocol"][field]
    assert c.model_dump(mode="json") == before
    with pytest.raises(OperationRejected, match="not a raw observation"):
        check_refs(out, ["followup:state"], [], scientific=True)


def test_low_rank_option_is_visible_and_selectable_without_reordering(tmp_path):
    c, _ = prepared(tmp_path)
    c = accept(c)
    qid = next(iter(f.state(c).questions))
    rows = comparison_options(c, catalog(c))
    assert rows[0]["id"] == qid and rows[0]["advisory_option"]
    entries = deepcopy(h.state(c).entries)
    out = select(c, question_id=qid, considered_question_ids=[qid])
    assert h.state(out).entries == entries
    assert inv.current(out)["choice"]["question_id"] == qid


@pytest.mark.parametrize("status", ["failed", "partial", "unsupported", "success"])
def test_attempt_can_motivate_a_question_without_becoming_positive_or_negative_evidence(tmp_path, status):
    c, _ = prepared(tmp_path)
    c = put(c, results={"result:new": {"status": status}})
    out = accept(c)
    assert len(f.state(out).questions) == 1
    assert h.state(out).entries == h.state(c).entries


@pytest.mark.parametrize("changes", [
    {"source_refs": ["foreign:result"]},
    {"route": {"capability_id": TOOL, "input_refs": ["foreign:artifact"]}},
    {"contrast": {"baseline_refs": ["foreign:result"]}},
    {"route": {"capability_id": "imagined.tool"}},
    {"labels": [LABELS[0], LABELS[0]]},
])
def test_bad_advice_is_local_atomic_and_identifies_its_field(tmp_path, changes):
    c, _ = prepared(tmp_path)
    before = c.model_dump(mode="json")
    with pytest.raises(OperationRejected) as err:
        accept(c, proposal(**changes))
    assert err.value.issues[0]["path"][:2] == ["experiments", 0]
    assert c.model_dump(mode="json") == before


def test_existing_but_unexposed_reference_rejected(tmp_path):
    c, _ = prepared(tmp_path)
    c = put(c, results={**protocol(c)["results"], "result:old": {"status": "success"}})
    with pytest.raises(OperationRejected, match="review scope"):
        accept(c, proposal(contrast={"baseline_refs": ["result:old"]}))


def test_identical_option_does_not_reopen_or_overwrite_planner_edits_after_resume(tmp_path):
    c, _ = prepared(tmp_path)
    c = accept(c)
    qid, old = next(iter(f.state(c).questions.items()))
    edited = old.value.model_copy(update={"status": "deferred", "question": "Planner refined this comparison."})
    c, _ = h.update(c, UpdateHypothesesWithInvestigation(kind="update_hypotheses", note="Defer for now.", questions=[edited]))
    s = store(tmp_path, c)
    save(c, s)
    c = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    current = options.accept_options(c, f.state(c), [suggestion()])
    assert len(current.questions) == 1 and current.questions[qid].value == edited
    assert current.questions[qid].origin == "impact_reviewer"
    assert current.questions[qid].source_refs == ["result:new"]


def test_new_option_invalidates_choice_but_does_not_prevent_declining_it(tmp_path):
    c, _ = prepared(tmp_path)
    c = select(c)
    out = accept(c)
    assert inv.current(out)["basis"] != inv.basis(out)
    out = select(out, decision="conclude", considered_question_ids=list(f.state(out).questions))
    assert inv.current(out)["choice"]["decision"] == "conclude"
    assert f.state(out).questions  # No forced resolution or experiment.


def test_native_review_exposes_catalog_and_saves_options_in_same_call(tmp_path):
    c, o = prepared(tmp_path)
    client = NativeFake([reply("report_evidence_impacts", proposal().model_dump(mode="json", exclude={"kind"}))])
    o.deps.photophysics_arbiter.llm_client = client
    c = run_turn(o, put(c, phase="impact_review"), store(tmp_path, c), catalog=catalog(c))
    assert len(client.requests) == 1 and len(f.state(c).questions) == 1
    material = client.requests[0]["payload"]
    assert "experiment_generation" in material
    assert material["experiment_generation"]["tool_catalog"]
    contract = next(t["parameters"] for t in client.requests[0]["tools"] if t["name"] == "report_evidence_impacts")
    Draft202012Validator.check_schema(contract)
    assert "experiments" in contract["properties"]
    c = run_turn(o, c, store(tmp_path, c), catalog=catalog(c))
    assert len(client.requests) == 1


def test_bad_id_gets_one_local_correction_with_no_lost_results(tmp_path):
    c, o = prepared(tmp_path)
    client = NativeFake([reply("report_evidence_impacts", proposal(source_refs=["bad"]).model_dump(exclude={"kind"})),
                        reply("report_evidence_impacts", proposal().model_dump(exclude={"kind"}))])
    o.deps.photophysics_arbiter.llm_client = client
    s = store(tmp_path, c)
    c = save(run_turn(o, c, s, catalog=catalog(c)), s)
    assert protocol(c)["impact_session"]["corrections"] == 1
    assert not f.state(c).questions and "result:new" in protocol(c)["results"]
    c = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    c = run_turn(o, c, s, catalog=catalog(c))
    assert len(f.state(c).questions) == 1 and protocol(c)["impact_session"] is None


def test_old_contract_empty_advice_unavailable_review_and_case_isolation(tmp_path):
    c, o = prepared(tmp_path)
    with pytest.raises(ValueError, match="requires command_investigation_policy"):
        replace(o.config, command_investigation_policy=False)
    with pytest.raises(ValueError, match="requires the result-impact Reviewer"):
        replace(o.config, enable_support_auditor=False)
    assert not replace(o.config, command_experiment_options=False,
                       enable_support_auditor=False).effective_support_auditor
    old = observed()
    assert "experiments" not in validate_suggestions(old, ["result:new"], report())
    c = f.accept_review(c, ["result:new"], {"status": "unavailable", "reason": "Fixture failure."})
    assert not f.state(c).questions and not f.unreviewed_results(c)
    assert not options.enabled(observed())
    protocol(c)["evidence_followup"]["case_id"] = "another-case"
    with pytest.raises(RuntimeError, match="ownership"):
        options.review_context(c, [])
    with pytest.raises(ValidationError):
        ReportEvidenceImpactsWithExperiments(**report().model_dump(), experiments=[suggestion()] * 4)
