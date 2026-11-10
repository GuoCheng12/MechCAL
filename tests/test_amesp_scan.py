from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.units import Hartree, Bohr

from mechcal.schemas import ArtifactRecord
from mechcal.schemas.tool_arguments import RelaxedScanArgs
from mechcal.tools.amesp_baseline import AmespBaselineError, _read_xyz_symbols_and_coordinates
from mechcal.tools.amesp_scan import parse_forces, coordinate_error, AmespForceCalculator
from mechcal.tools.local_micro import LocalMicroscopicTool
from tests.test_tool_parameters import case_with_structure, request


@pytest.mark.parametrize(
    "updates",
    [
        {"atom_indices": [0, 0]},
        {"values": [1.0, 1.0]},
        {"values": [float("nan"), 1.0]},
        {"values": [0.2, 1.0]},
        {"state_root": True},
        {"state_root": 2},
        {"max_wall_seconds": 0},
        {"coordinate": "dihedral", "atom_indices": [0, 1, 2, 3], "values": [-180, 180]},
    ],
)
def test_scan_arguments(updates):
    with pytest.raises(ValueError):
        RelaxedScanArgs.model_validate(
            dict(coordinate="bond", atom_indices=[0, 1], values=[1.1, 1.2]) | updates
        )


def test_force_sign_units_and_atom_order():
    text = "Final Energy: -3.0\nCartesian Force (Hartrees/Bohr):\n\n C 0.1 0.2 -0.3\n H 0.0 0.0 0.0\n\n"
    energy, forces = parse_forces(text, ["C", "H"])
    assert energy == -3 * Hartree
    np.testing.assert_allclose(forces[0], np.array([0.1, 0.2, -0.3]) * Hartree / Bohr)
    for bad in [
        text.replace("Force", "Gradient"),
        text.replace("0.1", "nan"),
        text.replace(" H 0.0 0.0 0.0", ""),
    ]:
        with pytest.raises(AmespBaselineError):
            parse_forces(bad, ["C", "H"])
    with pytest.raises(AmespBaselineError):
        parse_forces(text, ["H", "C"])
    assert coordinate_error(-179, 179, "dihedral") == 2


def test_deadline_prevents_backend_call(tmp_path):
    calculator = AmespForceCalculator(None, tmp_path, 0, 1, None, 0)
    with pytest.raises(AmespBaselineError, match="wall-time"):
        calculator.execute("test", Atoms("H2"))


def add_geometry_bundle(case, tmp_path, *, converged=True):
    template = case.artifact_manifest.artifacts[0]
    src = Path(template.file_paths["xyz"])
    bundle = dict(
        source_artifact_ids=[template.artifact_id],
        file_paths={"analysis_geometry_xyz": str(src)},
        parsed_observables={
            "state_root": 0,
            "route_records": [
                {"member_id": "scan:01", "geometry_converged": converged, "xyz_path": str(src)}
            ],
            "molecular_context": {"charge": 0, "multiplicity": 1},
        },
    )
    artifact = ArtifactRecord(
        artifact_id="computed:test",
        kind="relaxed_coordinate_scan_bundle",
        created_by_round="R001",
        created_by_agent="microscopic",
        metadata={"bundle": bundle},
    )
    case.artifact_manifest = case.artifact_manifest.with_records([artifact])
    return artifact


def test_geometry_selection_preserves_coordinates_and_clears_stale_metadata(tmp_path):
    case = case_with_structure(tmp_path, "CCO")
    add_geometry_bundle(case, tmp_path)
    result = LocalMicroscopicTool().run(
        request(
            "select_computed_geometry",
            {"source_artifact_id": "computed:test", "member_id": "scan:01"},
        ),
        case,
    )
    assert result.status == "success"
    artifact = result.artifact_updates[0]
    assert artifact.kind == "prepared_structure"
    assert (
        "force_field" not in artifact.metadata and "selected_conformer_id" not in artifact.metadata
    )
    before = _read_xyz_symbols_and_coordinates(
        Path(case.artifact_manifest.artifacts[0].file_paths["xyz"])
    )
    after = _read_xyz_symbols_and_coordinates(Path(artifact.file_paths["xyz"]))
    assert before == after
    assert artifact.metadata["parent_artifact_id"] == "computed:test"


@pytest.mark.parametrize(
    "problem",
    ["unconverged", "wrong_charge", "bad_member", "wrong_kind", "missing_file", "missing_source"],
)
def test_geometry_selection_never_falls_back(tmp_path, problem):
    case = case_with_structure(tmp_path, "CCO")
    artifact = add_geometry_bundle(case, tmp_path, converged=problem != "unconverged")
    args = {"source_artifact_id": artifact.artifact_id, "member_id": "scan:01"}
    if problem == "wrong_charge":
        artifact.metadata["bundle"]["parsed_observables"]["molecular_context"]["charge"] = 1
    elif problem == "bad_member":
        args["member_id"] = "scan:absent"
    elif problem == "wrong_kind":
        artifact.kind = "arbitrary"
    elif problem == "missing_file":
        artifact.metadata["bundle"]["parsed_observables"]["route_records"][0]["xyz_path"] = str(
            tmp_path / "absent.xyz"
        )
    elif problem == "missing_source":
        args["source_artifact_id"] = "missing"
    result = LocalMicroscopicTool().run(request("select_computed_geometry", args), case)
    assert result.status == "precondition_missing"
    assert not result.artifact_updates


@pytest.mark.parametrize("root", [1, 2])
def test_excited_root_relaxation_suspended_before_backend(tmp_path, root):
    case = case_with_structure(tmp_path, "C=O")
    result = LocalMicroscopicTool(enable_amesp=True).run(
        request("run_excited_state_relaxation", {"state_root": root}), case
    )
    assert result.status == "unsupported"
    assert "finite-difference" in result.failure.message


def test_s1_scan_and_old_s1_geometry_cannot_bypass_boundary(tmp_path):
    case = case_with_structure(tmp_path, "CCO")
    artifact = add_geometry_bundle(case, tmp_path)
    artifact.metadata["bundle"]["parsed_observables"]["state_root"] = 1
    tool = LocalMicroscopicTool(enable_amesp=True)
    result = tool.run(
        request(
            "select_computed_geometry",
            {"source_artifact_id": artifact.artifact_id, "member_id": "scan:01"},
        ),
        case,
    )
    assert result.status == "unsupported"

    result = tool.run(
        request(
            "run_relaxed_coordinate_scan",
            {"coordinate": "bond", "atom_indices": [0, 1], "values": [1.4, 1.5], "state_root": 1},
        ),
        case,
    )
    assert result.status == "unsupported"


def test_missing_scan_topology_is_typed_without_reembedding(tmp_path):
    case = case_with_structure(tmp_path, "CCO")
    case.artifact_manifest.artifacts[0].file_paths["sdf"] = str(tmp_path / "missing.sdf")
    result = LocalMicroscopicTool(enable_amesp=True).run(
        request(
            "run_relaxed_coordinate_scan",
            {"coordinate": "bond", "atom_indices": [0, 1], "values": [1.4, 1.5]},
        ),
        case,
    )
    assert result.status == "precondition_missing"
    assert not result.artifact_updates
