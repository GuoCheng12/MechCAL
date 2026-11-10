from pathlib import Path

import pytest

from mechcal.agents.execution_planner import WorkerExecutionPlanner
from mechcal.agents.planner import (
    _compact_evidence_metrics,
    _initial_ranking_metrics,
    _normalize_dispatch_payload,
)
from mechcal.capabilities import default_capability_registry
from mechcal.schemas import ArtifactManifest, CaseInput, CaseRun, DispatchRequest, EvidenceLedger
from mechcal.schemas.tool_arguments import validate_tool_args
from mechcal.tools.amesp_microscopic import (
    AmespMicroscopicError,
    MicroscopicRunOptions,
    _charge_scheme_record,
    _conformer_inventory,
    _conformer_members,
    _frontier_fragment_atoms,
    _microscopic_workdir,
    _parse_mulliken_charges,
    _torsion_snapshot_members,
)
from mechcal.tools.local_micro import LocalMicroscopicTool
from mechcal.tools.local_structure import LocalStructureTool


def case_with_structure(tmp_path, smiles="c1ccccc1-c1ccccc1"):
    ci = CaseInput(case_id="probe", smiles=smiles, user_query="Tool test")
    artifact, failure = LocalStructureTool().prepare(ci, round_id="R000", workspace=tmp_path)
    assert not failure.is_failure
    return CaseRun(
        case_id=ci.case_id,
        input=ci,
        evidence_ledger=EvidenceLedger(case_id=ci.case_id),
        artifact_manifest=ArtifactManifest(case_id=ci.case_id, artifacts=[artifact]),
    )


def request(route, args):
    spec = default_capability_registry().require("microscopic." + route)
    return DispatchRequest(
        dispatch_id="R001:probe",
        round_id="R001",
        agent_name="microscopic",
        capability_id=spec.capability_id,
        route=route,
        task="Probe",
        objective="Probe",
        evidence_goal_family=spec.evidence_family,
        tool_args=args,
    )


@pytest.mark.parametrize(
    "route,args",
    [
        ("run_torsion_snapshots", {"typo": 1}),
        ("run_torsion_snapshots", {"max_members": 99}),
        ("run_torsion_snapshots", {"max_members": True}),
        ("run_torsion_snapshots", {"angles": [0, 360]}),
        ("run_torsion_snapshots", {"angles": [-180, 180]}),
        ("run_torsion_snapshots", {"angles": [0, 90], "max_members": 3}),
        ("run_torsion_snapshots", {"dihedral_atoms": [0, 1, 1, 2]}),
        ("run_torsion_snapshots", {"dihedral_id": "dih:001", "dihedral_atoms": [0, 1, 2, 3]}),
        ("run_conformer_bundle", {"conformer_ids": ["rdkit:1", "rdkit:1"]}),
        ("run_solvation_polarity_proxy", {"solvents": ["water", "bad"]}),
        ("run_solvation_polarity_proxy", {"solvents": ["water", "water"]}),
        ("run_frontier_orbital_partition", {"donor_atoms": [0], "acceptor_atoms": [0]}),
        ("run_frontier_orbital_partition", {"donor_atoms": [0]}),
        ("run_bright_dark_state_ordering", {"s1_nstates": None}),
        ("run_targeted_charge_analysis", {"s1_nstates": 3}),
    ],
)
def test_bad_args_are_rejected_not_silently_changed(route, args):
    with pytest.raises(ValueError):
        validate_tool_args(route, args)


def test_dispatch_to_worker_keeps_parameters(tmp_path):
    case = case_with_structure(tmp_path)
    req = request(
        "run_torsion_snapshots",
        {"dihedral_atoms": [4, 5, 6, 7], "angles": [0, 45, 90], "s1_nstates": 3},
    )
    normalized = DispatchRequest.model_validate(
        _normalize_dispatch_payload(req.model_dump(), round_id="R001")
    )
    planner = WorkerExecutionPlanner(capability_registry=default_capability_registry())
    plan = planner.deterministic_plan(agent_name="microscopic", request=normalized, case_run=case)
    assert plan.tool_args == req.tool_args
    with pytest.raises(ValueError, match="must not override"):
        planner._normalize_llm_plan(
            {"tool_args": {"angles": [0, 60, 90]}},
            request=req,
            capability=default_capability_registry().require(req.capability_id),
            case_run=case,
        )


def test_invalid_parameters_and_source_are_typed_failures(tmp_path):
    case = case_with_structure(tmp_path)
    tool = LocalMicroscopicTool(enable_amesp=True)
    bad = tool.run(request("run_torsion_snapshots", {"max_members": 99}), case)
    assert bad.status == "failed" and bad.failure.kind == "tool_args_invalid"
    missing = tool.run(
        request("run_torsion_snapshots", {"source_artifact_id": "does-not-exist"}), case
    )
    assert missing.status == "precondition_missing"


def test_explicit_torsion_angles_match_coordinates(tmp_path):
    from rdkit import Chem
    from rdkit.Chem import rdMolTransforms

    artifact = case_with_structure(tmp_path).artifact_manifest.artifacts[0]
    members, _ = _torsion_snapshot_members(
        artifact, tmp_path / "scan", max_members=3, dihedral_atoms=[4, 5, 6, 7], angles=[0, 45, 90]
    )
    for member, angle in zip(members, [0, 45, 90], strict=True):
        mol = Chem.MolFromXYZFile(str(member.xyz_path))
        assert rdMolTransforms.GetDihedralDeg(mol.GetConformer(), 4, 5, 6, 7) == pytest.approx(
            angle, abs=1e-4
        )
    with pytest.raises(AmespMicroscopicError, match="Unknown dihedral_id"):
        _torsion_snapshot_members(artifact, tmp_path / "bad", max_members=3, dihedral_id="invented")


def test_charge_parser_only_reads_final_named_table():
    text = "C 55.55\n Mulliken charges:\n 1 C -0.1\n 2 H 0.1\n Sum of Mulliken charges = 0.0\n C 77.0\n"
    assert _parse_mulliken_charges(text) == [-0.1, 0.1]
    valid = _charge_scheme_record("mulliken", Path("aop"), text, symbols=["C", "H"], charge=0)
    assert valid["valid"] and len(valid["charges"]) == 2
    invalid = _charge_scheme_record("mulliken", Path("aop"), text, symbols=["C", "H"], charge=1)
    assert not invalid["valid"] and invalid["charges"] == []
    assert len(invalid["raw_charge_rows"]) == 2
    assert _parse_mulliken_charges(
        text + text.replace("-0.1", "-0.2").replace(" H 0.1", " H 0.2")
    ) == [-0.2, 0.2]


def test_fragments_are_explicit_and_zero_based(tmp_path):
    artifact = case_with_structure(tmp_path, "Nc1ccc(C=O)cc1").artifact_manifest.artifacts[0]
    assert _frontier_fragment_atoms(artifact)["donor_atoms"] == []
    fragments = _frontier_fragment_atoms(
        artifact, MicroscopicRunOptions(donor_atoms=[0], acceptor_atoms=[5, 6])
    )
    assert fragments["donor_atoms"] == [0] and fragments["acceptor_atoms"] == [5, 6]
    assert fragments["atom_index_base"] == 0


def test_conformer_discovery_matches_selected_members(tmp_path):
    artifact = case_with_structure(tmp_path, "CCCCCC").artifact_manifest.artifacts[0]
    inventory, _, _ = _conformer_inventory(artifact)
    ids = [r["conformer_id"] for r in inventory["conformers"]][:2]
    assert len(ids) == 2
    members, _ = _conformer_members(
        artifact, tmp_path / "series", max_members=2, member_prefix="conformer", conformer_ids=ids
    )
    assert [m.metadata["conformer_id"] for m in members] == ids


def test_observables_reach_both_planner_payloads():
    observations = {
        "route_records": [{"angle": n, "oscillator": 0} for n in [0, 45, 90]],
        "missing_descriptors": ["test"],
        "effective_parameters": {"angles": [0, 45, 90]},
    }
    metrics = {"computed_observables": observations}
    assert _compact_evidence_metrics(metrics)["computed_observables"] == observations
    assert _initial_ranking_metrics(metrics)["computed_observables"] == observations


def test_calculation_directories_are_unique():
    xyz = Path("/tmp/probe/artifacts/structure/prepared.xyz")
    assert _microscopic_workdir(xyz, "R001", "run_torsion_snapshots") != _microscopic_workdir(
        xyz, "R001", "run_torsion_snapshots"
    )


def test_followup_retains_molecular_charge():
    from mechcal.schemas import ArtifactRecord, MicroscopicArtifactBundle
    from mechcal.tools.amesp_microscopic import MICROSCOPIC_ROUTE_PROFILES, _charge_and_multiplicity
    from mechcal.tools.local_micro import _artifact_from_bundle
    bundle = MicroscopicArtifactBundle(
        bundle_id="R001:charged", capability_name="run_baseline_bundle", status="available",
        source_artifact_ids=["structure:prepared"], geometry_source="test",
        parsed_observables={"molecular_context": {"charge": 1, "multiplicity": 1}})
    artifact = _artifact_from_bundle(bundle, profile=MICROSCOPIC_ROUTE_PROFILES["run_baseline_bundle"], round_id="R001")
    assert _charge_and_multiplicity(artifact) == (1, 1)
    indirect = ArtifactRecord(artifact_id="test", kind="amesp_baseline_bundle", created_by_round="R001",
                              created_by_agent="microscopic", metadata={"bundle": bundle.model_dump()})
    assert _charge_and_multiplicity(indirect) == (1, 1)
