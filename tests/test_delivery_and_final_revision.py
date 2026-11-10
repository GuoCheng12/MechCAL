"""Offline contract tests; synthetic values are not scientific validation."""

from copy import deepcopy
from dataclasses import replace

import pytest

from mechcal.runtime.questions import (
    apply_final_revision, decision_from_action, delivery_summary, dependency_inputs,
    public_payload, replace_state, state, validate_action,
    request_signature, selected_source_id,
)
from mechcal.schemas import MechanismSupportArgument, PhotophysicsReview
from mechcal.schemas.question_loop import FinalArgumentRevision, ResultReview
from mechcal.tools.amesp_microscopic import (
    MICROSCOPIC_ROUTE_PROFILES, AmespMicroscopicRunner, _available_ct_descriptors,
    _parsed_observables,
)
from mechcal.schemas import MicroscopicArtifactBundle
from test_question_loop import REGISTRY, NOTE, QuestionClient, action, assimilation, loop, ready_case, request
from test_runtime import _evidence_unit
from test_tool_parameters import case_with_structure


def test_energy_and_headings_are_not_transition_or_density_values(tmp_path):
    profile = MICROSCOPIC_ROUTE_PROFILES["run_targeted_state_characterization"]
    states = [{"state_index": 1, "excitation_energy_ev": 3.0, "oscillator_strength": .1}]
    parsed = _parsed_observables(profile=profile, s0_text="density matrix\nMayer\ngross orbital population",
                                s1_text="Ground to excited state transition electric dipole moments",
                                excited_states=states)
    assert parsed["available_descriptors"] == ["state_ordering"]
    assert set(parsed["missing_descriptors"]) == {"dominant_transitions", "state_family_overlap"}
    assert parsed["excited_states"] == states
    result = AmespMicroscopicRunner()._write_bundle(
        bundle_id="test", capability_name="run_targeted_state_characterization", profile=profile,
        status="available", source_artifact_ids=["fixture"], geometry_source="fixture",
        workdir=tmp_path, file_paths={}, steps=[], s0_text="", s1_text="",
        missing_deliverables=[], limitations=[NOTE], parsed_overrides=parsed,
    )
    assert result.bundle.status == "partial"
    assert "dominant_transitions" in result.bundle.missing_deliverables


def test_old_ct_flags_do_not_become_values():
    bundle = MicroscopicArtifactBundle(
        bundle_id="fixture", capability_name="fixture", geometry_source="fixture", status="available",
        parsed_observables={"available_descriptors": ["dominant_transitions", "mulliken_charges"]},
        excited_states=[{"state_index": 1, "excitation_energy_ev": 3.0}],
    )
    assert _available_ct_descriptors(bundle) == ["state_ordering"]
    card = next(c for c in REGISTRY.cards() if c["capability_id"] == "microscopic.run_targeted_state_characterization")
    assert "dominant_transitions" not in card["delivery_contract"]["unimplemented_structured_descriptors"]
    assert "state_family_overlap" in card["delivery_contract"]["unimplemented_structured_descriptors"]


def test_preflight_scan_scope_uses_selected_source_not_all_routes(tmp_path):
    case = case_with_structure(tmp_path, "CP")
    source = case.artifact_manifest.artifacts[0]
    scan = "microscopic.run_relaxed_coordinate_scan"
    assert REGISTRY.readiness(scan, case.artifact_manifest, amesp_enabled=True)["status"] == "disabled"
    assert REGISTRY.readiness("microscopic.run_baseline_bundle", case.artifact_manifest, amesp_enabled=True)["status"] == "ready"
    other = case_with_structure(tmp_path / "organic", "CCO").artifact_manifest.artifacts[0]
    other = other.model_copy(update={"artifact_id": "organic"})
    manifest = case.artifact_manifest.with_records([other])
    assert REGISTRY.readiness(scan, manifest, amesp_enabled=True)["status"] == "ready"
    assert REGISTRY.availability_violations(request(cid=scan, args={"source_artifact_id": source.artifact_id}),
                                           manifest, amesp_enabled=True)
    assert not REGISTRY.availability_violations(request(cid=scan, args={"source_artifact_id": "organic"}),
                                               manifest, amesp_enabled=True)


def test_conformer_preflight_does_not_embed_or_optimize(tmp_path, monkeypatch):
    from rdkit.Chem import AllChem
    case = case_with_structure(tmp_path, "CCO")
    monkeypatch.setattr(AllChem, "MMFFHasAllMoleculeParams", lambda _: False)
    monkeypatch.setattr(AllChem, "UFFHasAllMoleculeParams", lambda _: False)
    def forbidden(*args, **kwargs):
        raise AssertionError("Preflight must not embed or compute")
    monkeypatch.setattr(AllChem, "EmbedMultipleConfs", forbidden)
    for route in ("list_available_conformers", "run_conformer_bundle", "run_conformer_state_probe"):
        result = REGISTRY.readiness("microscopic." + route, case.artifact_manifest, amesp_enabled=True)
        assert result["status"] == "missing_input"
        assert "MMFF94" in str(result["missing_inputs"])
    assert REGISTRY.readiness("microscopic.run_torsion_snapshots", case.artifact_manifest,
                              amesp_enabled=True)["status"] == "ready"


def test_delivery_value_comparison_does_not_treat_inventory_as_evidence():
    def packet(rid, charges):
        return dict(result_id=rid, status="partial", structured_results={"parsed_observables": {
            "mulliken_charges": charges, "available_descriptors": ["mulliken_charges"],
            "missing_descriptors": ["dominant_transitions"], "section_presence": {"density_matrix": True}}})
    first = packet("old", [0.1, -0.1])
    current = packet("new", [0.1, -0.1])
    summary = delivery_summary(current, [first])
    assert summary["identical_value_matches"] == {"mulliken_charges": ["old"]}
    assert summary["returned_value_fields"] == ["mulliken_charges"]
    assert delivery_summary(packet("changed", [0.2, -0.2]), [first])["identical_value_matches"] == {}


def test_shared_substeps_allow_new_analysis_but_reuse_refs_must_be_real():
    old = request(cid="microscopic.run_targeted_charge_redistribution_analysis", args={"source_artifact_id": "B"})
    new = request(cid="microscopic.run_targeted_state_characterization", args={"source_artifact_id": "B"})
    case = replace_state(ready_case(), results=[dict(
        result_id="old", capability_id=old.capability_id, status="partial",
        dependency_inputs=dependency_inputs(old),
        dependencies=REGISTRY.require(old.capability_id).metadata["execution_dependencies"],
    )])
    act = action(case.case_id, 1, dispatch=True)
    act.dispatch_requests = [new]
    validate_action(act, case, REGISTRY)
    new.reuse_result_ids = ["old"]
    new.expected_new_information = "Offline fixture: a different postprocessing target."
    validate_action(act, case, REGISTRY)
    new.reuse_result_ids = ["foreign"]
    with pytest.raises(ValueError, match="same-case"):
        validate_action(act, case, REGISTRY)
    new.reuse_result_ids = []
    new.expected_new_information = ""
    new.tool_args["source_artifact_id"] = "different_geometry"
    validate_action(act, case, REGISTRY)


def test_stop_reason_consistency_does_not_require_all_questions_answered():
    case = ready_case()
    act = action(case.case_id, 1)
    act.stopping_assessment = act.stopping_assessment.model_copy(update={
        "judgment_state": "adequate_for_bounded_ranking", "finish_basis": "judgment_sufficient"})
    with pytest.raises(ValueError, match="Material unresolved"):
        validate_action(act, case, REGISTRY)
    act.ranking_uncertainty_material = False
    validate_action(act, case, REGISTRY)  # An open question need not invalidate a bounded ranking.


def test_default_source_change_is_not_a_duplicate_request(tmp_path):
    prepared = case_with_structure(tmp_path, "CCO").artifact_manifest
    case = ready_case().touch(artifact_manifest=prepared)
    req = request(cid="microscopic.run_torsion_snapshots", args={"dihedral_id": "dih:001", "angles": [0, 90]})
    case = replace_state(case, results=[dict(result_id="old", status="success",
        signature=request_signature(req, REGISTRY),
        resolved_source_artifact_id=selected_source_id(req, case, REGISTRY))])
    act = action(case.case_id, 1, dispatch=True)
    act.dispatch_requests = [req]
    with pytest.raises(ValueError, match="retry_reason"):
        validate_action(act, case, REGISTRY)
    new = prepared.artifacts[-1].model_copy(update={"artifact_id": "new-geometry"})
    case = case.touch(artifact_manifest=prepared.with_records([new]))
    validate_action(act, case, REGISTRY)


def final_fixture():
    case = ready_case()
    observation = _evidence_unit(evidence_id="E1", round_id="R001", source_report_id="fixture",
                                 agent_name="macro", family="torsion_sensitivity", summary=NOTE)
    argument = MechanismSupportArgument(support_id="S1", target_label="RIM_RIR_RIV", support_level="weak",
                                       finding=NOTE, warrant=NOTE, boundary=NOTE,
                                       observation_refs=[observation.evidence_id], audit_status="accepted")
    case = case.touch(evidence_ledger=case.evidence_ledger.model_copy(update={"items": [observation]}),
                      mechanism_support_arguments=[argument])
    previous = decision_from_action(action(case.case_id, 1), case, "R002")
    case = replace_state(case, phase="final_revision", decision=previous.model_dump(mode="json"))
    payload = public_payload(case, REGISTRY, stage="planner_final_revision")
    update = assimilation(payload)
    update["mechanism_support_arguments"] = [argument.model_dump(mode="json")]
    revision = FinalArgumentRevision(update=update, final_action=action(case.case_id, 2),
                                     audit_responses={"S1": NOTE})
    return case, revision


def test_final_revision_changes_warrant_not_observations_or_audit_approval():
    case, revision = final_fixture()
    before = case.model_dump_json()
    revision.update.mechanism_support_arguments[0].warrant = "Revised offline fixture warrant."
    updated, decision = apply_final_revision(case, revision, REGISTRY)
    assert updated.mechanism_support_arguments[0].warrant == "Revised offline fixture warrant."
    assert updated.mechanism_support_arguments[0].audit_status == "not_reviewed"
    assert updated.evidence_ledger == case.evidence_ledger
    assert decision.mechanism_support_arguments == updated.mechanism_support_arguments
    assert case.model_dump_json() == before


@pytest.mark.parametrize("invalid", ["tools", "foreign_ref", "missing_response", "stale", "unknown_label", "too_few"])
def test_final_revision_rejects_contract_violations(invalid):
    case, revision = final_fixture()
    if invalid == "tools":
        revision.final_action.dispatch_requests = [request()]
    elif invalid == "foreign_ref":
        revision.update.mechanism_support_arguments[0].observation_refs = ["other-case"]
    elif invalid == "missing_response":
        revision.audit_responses = {}
    elif invalid == "unknown_label":
        revision.update.mechanism_support_arguments[0].target_label = "invented"
    elif invalid == "too_few":
        revision.update.portfolio.hypotheses = revision.update.portfolio.hypotheses[:2]
    else:
        revision.update.revision = 1
    with pytest.raises(ValueError):
        apply_final_revision(case, revision, REGISTRY)


@pytest.mark.parametrize("audit_fails", [False, True])
def test_final_revision_once_with_resume_and_unavailable_review(tmp_path, monkeypatch, audit_fails):
    o, client = loop(tmp_path)
    o.config = replace(o.config, enable_support_auditor=True, continue_on_arbiter_failure=audit_fails)
    calls = []
    def audit(case, **kwargs):
        calls.append(case.current_round_id)
        if audit_fails:
            raise TimeoutError("Offline injected final audit failure")
        return PhotophysicsReview(review_id="audit", case_id=case.case_id, round_id=case.current_round_id)
    monkeypatch.setattr(o.deps.photophysics_arbiter, "review", audit)
    monkeypatch.setattr(o.deps.photophysics_arbiter, "review_results",
                        lambda _, **kw: ResultReview(status="unavailable", reason=NOTE))
    original = o._finish_case
    def crash(*args, **kwargs):
        raise KeyboardInterrupt("Before final assembly, after revision checkpoint")
    monkeypatch.setattr(o, "_finish_case", crash)
    with pytest.raises(KeyboardInterrupt):
        o.run(smiles="CCO", user_query=NOTE, case_id="final")
    revisions = [p for p in client.calls if p["stage"] == "planner_final_revision"]
    assert len(revisions) == len(calls) == 1
    assert revisions[0]["pending_results"] == []
    assert "capability_catalog" not in revisions[0]
    assert revisions[0]["final_internal_audit"]["status"] == ("unavailable" if audit_fails else "reviewed")
    monkeypatch.setattr(o, "_finish_case", original)
    result = o.run(smiles="CCO", user_query=NOTE, case_id="final", resume=True)
    assert result.status == "finalized"
    assert result.round_ids == ["R001", "R002"]
    assert len(state(result)["results"]) == 1
    assert len(calls) == 1
    assert sum(p["stage"] == "planner_final_revision" for p in client.calls) == 1
    before = deepcopy(client.calls)
    assert o.run(smiles="CCO", user_query=NOTE, case_id="final", resume=True) == result
    assert client.calls == before


def test_final_feedback_reaches_planner_and_revised_warrant_reaches_output(tmp_path, monkeypatch):
    o, _ = loop(tmp_path)
    o.config = replace(o.config, enable_support_auditor=True)
    class Client(QuestionClient):
        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            result = super().complete_json(payload=payload, system_prompt=system_prompt,
                                           schema_feedback=schema_feedback)
            if payload["stage"] == "planner_assimilate" and payload["evidence_index"]:
                result["mechanism_support_arguments"] = [dict(
                    support_id="S1", target_label="RIM_RIR_RIV", support_level="weak",
                    finding=NOTE, warrant=NOTE, boundary=NOTE,
                    observation_refs=[payload["evidence_index"][0]["evidence_id"]],
                )]
            elif payload["stage"] == "planner_final_revision":
                audit = payload["final_internal_audit"]["review"]["support_argument_audits"][0]
                assert audit["audit_notes"] == ["Offline request to clarify the warrant."]
                assert audit["argument_assessment"]["mechanism_link_status"] == "missing"
                args = deepcopy(payload["previous_support_arguments"])
                args[0]["warrant"] = "Revised offline fixture explanation, not a scientific result."
                result["update"]["mechanism_support_arguments"] = args
                result["final_action"]["final_answer_draft"] = "Updated offline final draft."
            return result
    client = Client()
    o.deps.planner.llm_client = client
    def audit(case, **kwargs):
        return PhotophysicsReview(review_id="final-audit", case_id=case.case_id,
            round_id=case.current_round_id, support_argument_audits=[dict(
                support_id="S1", audit_status="boundary_added",
                audit_notes=["Offline request to clarify the warrant."], boundary_patch=NOTE,
                argument_assessment=dict(
                    finding_status="verified", finding_reason=NOTE,
                    mechanism_link_status="missing", mechanism_link_reason=NOTE,
                    boundary_effect="scope_limit", boundary_reason=NOTE,
                ))])
    monkeypatch.setattr(o.deps.photophysics_arbiter, "review", audit)
    monkeypatch.setattr(o.deps.photophysics_arbiter, "review_results",
                        lambda _, **kw: ResultReview(status="unavailable", reason=NOTE))
    result = o.run(smiles="CCO", user_query=NOTE, case_id="revised")
    assert result.status == "finalized"
    assert result.mechanism_support_arguments[0].audit_status == "not_reviewed"
    assert result.final_answer.answer.startswith("Updated offline final draft.")
    assert "Revised offline fixture explanation" in result.final_answer.model_dump_json()
    assert sum(p["stage"] == "planner_final_revision" for p in client.calls) == 1
    assert len(state(result)["results"]) == 1
    assert result.runtime["termination"]["final_revision_complete"] is True


def test_final_revision_failure_is_bounded_and_does_not_dispatch(tmp_path, monkeypatch):
    from mechcal.runtime.retry import SchemaRetryError
    o, _ = loop(tmp_path)
    o.config = replace(o.config, enable_support_auditor=True)
    class InvalidClient(QuestionClient):
        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            if payload["stage"] == "planner_final_revision":
                self.calls.append(deepcopy(payload))
                if sum(p["stage"] == "planner_final_revision" for p in self.calls) > 1:
                    assert "cannot reopen" in schema_feedback
                return dict(update=assimilation(payload),
                            final_action=action(payload["case_id"], payload["revision"] + 1,
                                                dispatch=True).model_dump(mode="json"),
                            audit_responses={})
            return super().complete_json(payload=payload, system_prompt=system_prompt,
                                         schema_feedback=schema_feedback)
    client = InvalidClient()
    o.deps.planner.llm_client = client
    monkeypatch.setattr(o.deps.photophysics_arbiter, "review_results",
                        lambda _, **kw: ResultReview(status="unavailable", reason=NOTE))
    with pytest.raises(SchemaRetryError):
        o.run(smiles="CCO", user_query=NOTE, case_id="invalid-final")
    import json
    saved = json.loads((tmp_path / "invalid-final/question_checkpoint.json").read_text())
    assert saved["runtime"]["question_loop"]["phase"] == "final_revision"
    assert len(saved["runtime"]["question_loop"]["results"]) == 1
    assert saved["final_answer"] is None
    assert sum(p["stage"] == "planner_final_revision" for p in client.calls) == 3
    failures = (tmp_path / "invalid-final/history/planner_final_revision_failures.jsonl").read_text().splitlines()
    assert len(failures) == 3
