"""Geometry and final-audit regressions; fixtures are not scientific results."""

from pathlib import Path

import pytest
from rdkit import Chem

from mechcal.agents.photophysics_arbiter import (
    DisabledPhotophysicsWebSearch,
    PhotophysicsArbiterAgent,
)
from mechcal.agents.planner import PlannerAgent
from mechcal.capabilities import default_capability_registry
from mechcal.chem import structure_prep
from mechcal.runtime import MechCALOrchestrator, OrchestratorConfig
from mechcal.schemas import ArtifactManifest, MechanismSupportArgument
from mechcal.tools.amesp_baseline import AmespBaselineError, AmespStepResult
from mechcal.tools.ground_geometry import optimize_ground_geometry
from mechcal.tools.local_micro import LocalMicroscopicTool
from mechcal.tools.local_structure import LocalStructureTool
from test_evidence_references import citation_case
from test_planner_owned_loop import ScriptedPlannerClient, reply
from test_runtime import _fake_llm_deps
from test_tool_parameters import case_with_structure, request


def seed_case(tmp_path):
    case = case_with_structure(tmp_path, "CO")
    artifact = case.artifact_manifest.artifacts[0]
    seed = artifact.model_copy(update={"kind": "initial_structure", "artifact_id": "seed"})
    return case.touch(
        artifact_manifest=case.artifact_manifest.model_copy(update={"artifacts": [seed]})
    )


def test_missing_forcefield_preserves_only_an_initial_seed(tmp_path, monkeypatch):
    ci = case_with_structure(tmp_path / "normal", "CO").input

    def missing(*args):
        raise structure_prep.StructurePrepError("forcefield_failed", "Fixture missing parameters")

    monkeypatch.setattr(structure_prep, "_select_force_field", missing)
    artifact, failure = LocalStructureTool().prepare(ci, round_id="R000", workspace=tmp_path)
    assert failure.is_failure and failure.recoverable
    assert artifact.kind == "initial_structure"
    assert artifact.metadata["geometry_status"] == "embedded_unoptimized"
    assert artifact.metadata["force_field"] is None
    assert Path(artifact.file_paths["xyz"]).is_file()
    registry = default_capability_registry()
    manifest = ArtifactManifest(case_id=ci.case_id, artifacts=[artifact])
    assert registry.validate_dispatch(request("run_bright_dark_state_ordering", {}), manifest)
    assert not registry.validate_dispatch(request("optimize_ground_state_geometry", {}), manifest)
    with pytest.raises(structure_prep.StructurePrepError, match="Fixture missing"):
        structure_prep.prepare_structure_from_smiles(
            structure_prep.StructurePrepRequest(
                smiles="CO", label="strict", workdir=tmp_path / "strict"
            )
        )


class GeometryFixture:
    timeout_seconds = 180
    npara = 1
    maxcore_mb = 1000

    def __init__(self, mode="converged"):
        self.mode = mode

    def _run_step(self, **kwargs):
        self.call = kwargs
        directory = kwargs["workdir"]
        aip, aop = directory / "fixture.aip", directory / "fixture.aop"
        aip.write_text("Parser fixture only")
        force = 0.01 if self.mode == "unconverged" else 0.0
        lines = ["Cartesian Force (Hartrees/Bohr):"]
        lines += [f"{symbol} {force} 0.0 0.0" for symbol in kwargs["symbols"]]
        lines += [""]
        lines += ["Final Energy: -1.000000"]
        aop.write_text("\n".join(lines))
        return AmespStepResult(
            "fixture", aip, aop, directory / "stdout", directory / "stderr", 0, True, 0
        )


def test_converged_recovery_preserves_graph_and_delivers_prepared_structure(tmp_path):
    case = seed_case(tmp_path)
    runner = GeometryFixture()
    result = optimize_ground_geometry(
        request("optimize_ground_state_geometry", {}), case, {"max_opt_cycles": 17}, runner
    )
    assert result.status == "success"
    artifact = result.artifact_updates[0]
    assert artifact.kind == "prepared_structure"
    assert artifact.metadata["geometry_status"] == "ground_state_optimized"
    assert artifact.metadata["computed_observables"]["effective_parameters"]["max_opt_cycles"] == 17
    assert artifact.metadata["computed_observables"]["max_force_ev_per_angstrom"] == 0
    original = Chem.MolFromMolFile(
        case.artifact_manifest.artifacts[0].file_paths["sdf"], removeHs=False
    )
    recovered = Chem.MolFromMolFile(artifact.file_paths["sdf"], removeHs=False)
    assert Chem.MolToSmiles(original) == Chem.MolToSmiles(recovered)
    assert result.evidence_units[0].support == "not_applicable"
    assert not artifact.metadata["computed_observables"]["minimum_verified_by_frequencies"]


@pytest.mark.parametrize("mode", ["unconverged", "overlap"])
def test_bad_recovery_keeps_raw_files_but_publishes_no_prepared_geometry(
    tmp_path, mode, monkeypatch
):
    if mode == "overlap":
        from mechcal.tools import ground_geometry

        original = ground_geometry.BFGS.run

        def overlap(optimizer, **kwargs):
            result = original(optimizer, **kwargs)
            optimizer.atoms.positions[1] = optimizer.atoms.positions[0]
            return result

        monkeypatch.setattr(ground_geometry.BFGS, "run", overlap)
    case = seed_case(tmp_path)
    with pytest.raises(AmespBaselineError) as caught:
        optimize_ground_geometry(
            request("optimize_ground_state_geometry", {}),
            case,
            {"max_opt_cycles": 1},
            GeometryFixture(mode),
        )
    assert Path(caught.value.file_paths["aop"]).is_file()
    assert not list(tmp_path.rglob("ground_geometry.sdf"))
    assert all(a.kind != "prepared_structure" for a in case.artifact_manifest.artifacts)


def test_recovery_honors_disabled_backend(tmp_path):
    result = LocalMicroscopicTool(enable_amesp=False).run(
        request("optimize_ground_state_geometry", {}), seed_case(tmp_path)
    )
    assert result.status == "unsupported" and not result.artifact_updates


def test_reviewer_sees_initial_geometry_and_recovery_boundary(tmp_path):
    from mechcal.agents.mechanism_agenda import _coverage_payload

    payload = _coverage_payload(
        seed_case(tmp_path), default_capability_registry(), "R001", previous_planner_action=None
    )
    assert payload["artifact_manifest"][0]["kind"] == "initial_structure"
    cards = {c["capability_id"]: c for c in payload["capability_cards"]}
    assert "Cartesian" in cards["microscopic.optimize_ground_state_geometry"]["description"]
    assert "No independent" in cards["macro.run_aggregate_contact_proxy"]["description"]


def test_final_audit_rejects_missing_or_duplicate_support_ids():
    case = citation_case(["E1"])
    arguments = [
        MechanismSupportArgument(
            support_id=f"final:{index}",
            target_label=label,
            support_level="weak",
            finding="Fixture observation.",
            warrant="Fixture link.",
            boundary="Fixture limitation.",
            observation_refs=["E1"],
        )
        for index, label in enumerate(("ICT_TICT_CT", "RIM_RIR_RIV"))
    ]
    case = case.touch(
        mechanism_support_arguments=arguments,
        runtime={**case.runtime, "final_support_audit_round": "R004"},
    )

    class Client:
        calls = 0

        def is_configured(self):
            return True

        def complete_json(self, *, payload, schema_feedback=None, **kwargs):
            self.calls += 1
            assert payload["require_complete_support_audit"]
            ids = payload["allowed_support_ids"]
            selected = (
                ids[:1] if self.calls == 1 else ([ids[0], ids[0]] if self.calls == 2 else ids)
            )
            return {
                "support_argument_audits": [
                    {"support_id": s, "audit_status": "accepted"} for s in selected
                ]
            }

    client = Client()
    review = PhotophysicsArbiterAgent(
        llm_client=client, web_search=DisabledPhotophysicsWebSearch(), max_attempts=3
    ).review(case)
    assert client.calls == 3
    assert {a.support_id for a in review.support_argument_audits} == {
        a.support_id for a in arguments
    }


def test_new_final_arguments_are_audited_without_overriding_planner_rank(tmp_path):
    class Client(ScriptedPlannerClient):
        def complete_json(self, *, payload, **kwargs):
            if payload["round_id"] == "R001":
                return reply(
                    "dispatch", selected_capability_ids=["macro.screen_rotor_torsion_topology"]
                )
            return reply(
                mechanism_support_arguments=[
                    {
                        "support_id": "R002:new",
                        "target_label": "RIM_RIR_RIV",
                        "support_level": "weak",
                        "finding": "Offline fixture.",
                        "warrant": "Offline fixture.",
                        "boundary": "Offline fixture.",
                        "observation_refs": [payload["evidence_index"][0]["evidence_id"]],
                    }
                ]
            )

    config = OrchestratorConfig(
        run_base_dir=tmp_path,
        max_rounds=3,
        enable_agenda_coverage_reviewer=False,
        enable_support_auditor=True,
        enable_incremental_mechanism_portfolio=False,
    )
    deps = _fake_llm_deps(config)
    deps.planner = PlannerAgent(
        llm_client=Client(), enable_llm_decisions=True, capability_registry=deps.capability_registry
    )
    result = MechCALOrchestrator(config, deps).run(
        smiles="CCO", user_query="Fixture", case_id="final-audit"
    )
    assert result.status == "finalized" and result.round_ids == ["R001", "R002"]
    assert result.photophysics_review.round_id == "R002"
    assert result.mechanism_support_arguments[-1].audit_status != "not_reviewed"
    assert [h.name for h in result.portfolio.sorted_hypotheses()] == [
        "RIM_RIR_RIV",
        "ICT_TICT_CT",
        "ESIPT_PT",
    ]


def test_output_does_not_resurrect_older_stronger_arguments_after_final_audit():
    from mechcal.runtime.output_assembler import OutputAssembler
    from mechcal.schemas import HypothesisEntry, HypothesisPortfolio

    old = MechanismSupportArgument(
        support_id="R002:old",
        target_label="ICT_TICT_CT",
        support_level="strong",
        finding="Old fixture assertion.",
        warrant="Old fixture link.",
        boundary="Old limitation.",
        observation_refs=["E1"],
        audit_status="accepted",
    )
    new = old.model_copy(
        update={
            "support_id": "R003:new",
            "support_level": "unsupported",
            "audit_status": "unsupported",
            "finding": "Revised fixture assertion.",
        }
    )
    case = citation_case(["E1", "E2"]).touch(
        portfolio=HypothesisPortfolio(
            current="ICT_TICT_CT",
            hypotheses=[
                HypothesisEntry(name="ICT_TICT_CT", differential_priority=0.8, evidence_refs=["E1"])
            ],
        ),
        mechanism_support_arguments=[old, new],
    )
    prediction = OutputAssembler().build_mechanism_predictions(case)[0]
    assert prediction.evidence_support == "unsupported"
    assert prediction.evidence_refs == ["E1"]
    assert prediction.evidence_details[0].finding == "Revised fixture assertion."
    assert "Old fixture assertion" not in str(prediction.evidence)
    assert prediction.differential_priority == 0.8


class NativeFixture(GeometryFixture):
    """Native Amesp optimization outcomes per coordinate system; force calls fall through to the parent."""

    def __init__(self, outcomes):
        super().__init__()
        self.outcomes, self.coords = outcomes, []

    def _run_step(self, **kwargs):
        if "opt" not in kwargs["keywords"]:
            return super()._run_step(**kwargs)
        coord = next(l.split()[1] for _, lines in kwargs["block_lines"] for l in lines if l.startswith("optcoord"))
        self.coords.append(coord)
        directory = kwargs["workdir"]
        aip, aop = directory / f"native_{coord}.aip", directory / f"native_{coord}.aop"
        aip.write_text("fixture")
        outcome = self.outcomes[coord]
        if outcome != "converged":
            raise AmespBaselineError("runtime_failed", outcome, file_paths={"aop": str(aop)})
        table = [f"    {k}   0.0000100   {t}   YES" for k, t in
                 (("RMS Force", "0.0003000"), ("Max Force", "0.0004500"), ("RMS Step", "0.0012000"), ("Max Step", "0.0018000"))]
        atoms = [f"{s} {x} {y} {z}" for s, (x, y, z) in zip(kwargs["symbols"], kwargs["coordinates"])]
        aop.write_text("\n".join(table + ["Geometry Optimization Converged after 42 Steps.", "Final Geometry(angstroms):",
                                          str(len(atoms)), "", *atoms, "Final Energy:   -1.500000"]))
        return AmespStepResult("native", aip, aop, directory / "stdout", directory / "stderr", 0, True, 0)


@pytest.mark.parametrize("outcomes, optimizer, coords", [
    ({"red": "converged"}, "amesp_native_redundant_internal", ["red"]),
    ({"red": "Amesp backend stopped: Linear angle in Tors !", "car": "converged"}, "amesp_native_cartesian", ["red", "car"]),
    ({"red": "Amesp backend stopped: Linear angle in Tors !", "car": "Fail to convergence on Geom Opt!"}, "ASE_BFGS", ["red", "car"]),
    ({"red": "Fail to convergence on Geom Opt!"}, "ASE_BFGS", ["red"]),        # Cartesian only after a linear-angle stop
])
def test_native_optimizer_order_and_record(tmp_path, outcomes, optimizer, coords):
    runner = NativeFixture(outcomes)
    result = optimize_ground_geometry(request("optimize_ground_state_geometry", {}), seed_case(tmp_path), {}, runner)
    obs = result.artifact_updates[0].metadata["computed_observables"]
    assert runner.coords == coords and obs["optimizer"] == optimizer
    assert [a["optimizer"] for a in obs["optimizer_attempts"]][-1] == optimizer
    assert obs["optimizer_attempts"][-1]["outcome"] == "converged"
    assert all(a["outcome"] == "failed" and a["reason"] for a in obs["optimizer_attempts"][:-1])
    if optimizer.startswith("amesp_native"):
        assert obs["convergence_criterion"].startswith("Amesp geomtol normal") and obs["final_energy_hartree"] == -1.5
        assert obs["final_convergence_table"]["Max Force"]["converged"] and obs["optimization_steps"] == 42
    else:
        assert obs["convergence_criterion"] == "maximum_atomic_force_at_most_0.02_eV_per_angstrom"
