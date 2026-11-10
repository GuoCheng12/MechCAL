"""Offline observation delivery checks. Synthetic coordinates are not benchmark evidence."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from mechcal.capabilities import default_capability_registry
from mechcal.runtime.observation_delivery import delivery_receipt, requirement_violations, validate_question_delivery
from mechcal.runtime.retry import RequestValidationError
from mechcal.schemas import ArtifactRecord, MicroscopicArtifactBundle, ToolExecutionResult
from mechcal.schemas.observation_delivery import ObservableRequirement
from mechcal.schemas.tool_arguments import validate_tool_args
from mechcal.tools.geometry_measurements import measure_geometry
from mechcal.tools.local_micro import LocalMicroscopicTool
from test_tool_parameters import case_with_structure, request


ROUTE = "extract_geometry_descriptors_from_bundle"
XYZ = [[0., 1., 0.], [0., 0., 0.], [1., 0., 0.], [1., 0., 1.]]
MEASUREMENTS = [
    {"measurement_id": "d", "kind": "distance", "atom_indices": [0, 1]},
    {"measurement_id": "a", "kind": "angle", "atom_indices": [0, 1, 2]},
    {"measurement_id": "t", "kind": "dihedral", "atom_indices": [0, 1, 2, 3]},
]


def geometry_case(tmp_path):
    case = case_with_structure(tmp_path, "CCCC")
    xyz = tmp_path / "selected.xyz"
    xyz.write_text("4\nSynthetic coordinate contract fixture\nC 0 1 0\nC 0 0 0\nC 1 0 0\nC 1 0 1\n")
    bundle = MicroscopicArtifactBundle(
        bundle_id="test:bundle", capability_name="run_baseline_bundle", status="available",
        source_artifact_ids=[], geometry_source="synthetic_fixture",
        file_paths={"analysis_geometry_xyz": str(xyz)}, parsed_observables={},
    )
    artifact = ArtifactRecord(
        artifact_id="test:bundle", kind="amesp_baseline_bundle", status="available",
        created_by_round="R001", created_by_agent="microscopic",
        file_paths={"manifest": str(tmp_path / "artifacts/baseline/manifest.json")},
        metadata={"bundle": bundle.model_dump(mode="json")},
    )
    case.artifact_manifest.artifacts.append(artifact)
    return case


def test_general_coordinate_measurements():
    rows = measure_geometry(["C"] * 4, XYZ, MEASUREMENTS)
    assert rows["d"]["value"] == pytest.approx(1)
    assert rows["a"]["value"] == pytest.approx(90)
    assert abs(rows["t"]["value"]) == pytest.approx(90)
    assert rows["d"]["unit"] == "angstrom"
    assert rows["t"]["unit"] == "degree"
    # An independent coordinate implementation verifies the sign, not only magnitude.
    from rdkit import Chem
    from rdkit.Chem import rdMolTransforms
    mol = Chem.MolFromXYZBlock("4\nfixture\nC 0 1 0\nC 0 0 0\nC 1 0 0\nC 1 0 1\n")
    assert rows["t"]["value"] == pytest.approx(rdMolTransforms.GetDihedralDeg(mol.GetConformer(), 0, 1, 2, 3))


@pytest.mark.parametrize("fault", ["duplicate", "negative", "boolean", "wrong_count", "unknown_kind", "duplicate_id"])
def test_measurement_argument_rejection(fault):
    items = deepcopy(MEASUREMENTS)
    if fault == "duplicate_id":
        items.append(items[0])
    elif fault == "unknown_kind":
        items[0]["kind"] = "packing"
    else:
        items[0]["atom_indices"] = {"duplicate": [0, 0], "negative": [-1, 0], "boolean": [True, 0], "wrong_count": [0, 1, 2]}[fault]
    with pytest.raises(ValueError):
        validate_tool_args(ROUTE, {"measurements": items})


def test_degenerate_or_foreign_geometry_is_not_a_number():
    rows = measure_geometry(["C"] * 4, [[i, 0, 0] for i in range(4)], [MEASUREMENTS[-1]])
    assert rows["t"]["value"] is None and rows["t"]["status"] == "undefined"
    with pytest.raises(ValueError, match="outside"):
        measure_geometry(["C"] * 3, XYZ[:3], [MEASUREMENTS[-1]])
    with pytest.raises(ValueError, match="finite"):
        measure_geometry(["C"], [[float("nan"), 0, 0]], [])


def test_successful_extents_do_not_fulfill_requested_dihedral():
    req = request(ROUTE, {"measurements": [MEASUREMENTS[-1]]})
    old = {"parsed_observables": {"geometry_descriptors": {"axis_extents_angstrom": [23.242808, 5.091165, 6.904171]}}}
    assert delivery_receipt(req, old)["status"] == "unmet"
    new = {"parsed_observables": {"geometry_measurements": measure_geometry(["C"] * 4, XYZ, req.tool_args["measurements"])}}
    assert delivery_receipt(req, new)["status"] == "delivered"
    new["parsed_observables"]["geometry_measurements"]["t"]["unit"] = "radian"
    assert delivery_receipt(req, new)["status"] == "unmet"


@pytest.mark.parametrize("value", [None, True, "90", {}, [], float("nan"), float("inf")])
def test_non_numeric_values_cannot_fulfill_numeric_contract(value):
    req = request(ROUTE, {})
    req.required_observables = [ObservableRequirement(path="/v", value_type="number")]
    assert delivery_receipt(req, {"v": value})["status"] == "unmet"


def test_inventory_cannot_claim_file_contents_before_execution():
    req = request("inspect_raw_artifact_bundle", {})
    req.required_observables = [ObservableRequirement(path="/parsed_observables/file_contents", value_type="string")]
    violations = requirement_violations(default_capability_registry().require(req.capability_id), req)
    assert violations and "not a documented output" in violations[0]
    assert delivery_receipt(request(ROUTE, {}), {})["status"] == "not_specified"


@pytest.mark.parametrize("bad", [None, 3, "angle", [None], [{"measurement_id": "t"}]])
def test_invalid_measurement_arguments_remain_explicit_failures(bad, tmp_path):
    req = request(ROUTE, {"measurements": bad})
    case = geometry_case(tmp_path)
    assert default_capability_registry().validate_dispatch(req, case.artifact_manifest)
    assert delivery_receipt(req, {})["status"] == "invalid_request"


def test_requested_measurement_requires_parameter_selection():
    req = request(ROUTE, {})
    req.required_observables = [ObservableRequirement(
        path="/parsed_observables/geometry_measurements/t/value", value_type="number",
    )]
    spec = default_capability_registry().require(req.capability_id)
    assert requirement_violations(spec, req)
    req.tool_args["measurements"] = [MEASUREMENTS[-1]]
    assert requirement_violations(spec, req) == []


def test_unmet_preparation_cannot_close_question_but_independent_data_can():
    row = {"result_id": "r", "evidence_refs": ["e"], "investigation_plan": {"purpose": "preparation"},
           "observation_receipt": {"status": "unmet"}}
    case = SimpleNamespace(runtime={"question_loop": {"results": [row]}})
    question = SimpleNamespace(question_id="q", result_ids=["r"], evidence_refs=["e"], answer_state="answered")
    update = SimpleNamespace(questions=[question])
    with pytest.raises(RequestValidationError):
        validate_question_delivery(case, update)
    question.answer_state = "partial"
    validate_question_delivery(case, update)
    question.answer_state = "answered"
    question.evidence_refs.append("independent:measurement")
    validate_question_delivery(case, update)
    question.evidence_refs = ["e"]
    row["investigation_plan"]["purpose"] = "discrimination"
    validate_question_delivery(case, update)  # Do not replace scientific interpretation of partial data.


@pytest.mark.parametrize("transport", ["local", "stdio"])
def test_real_geometry_tool_through_both_transports(tmp_path, transport):
    case = geometry_case(tmp_path)
    req = request(ROUTE, {"source_artifact_id": "test:bundle", "measurements": MEASUREMENTS})
    before = case.model_dump_json()
    if transport == "local":
        result = LocalMicroscopicTool(enable_amesp=False).run(req, case)
    else:
        from mechcal.tools import StdioMcpTransport
        payload = StdioMcpTransport().call_tool(
            server_name="aie-microscopic-mcp", tool_name="run_microscopic",
            arguments={"dispatch_request": req.model_dump(mode="json"), "case_run": case.model_dump(mode="json")},
        )
        result = ToolExecutionResult.model_validate(payload["tool_result"])
    assert result.status == "success", result.failure
    parsed = result.structured_results["parsed_observables"]
    assert parsed["geometry_measurements"]["a"]["value"] == pytest.approx(90)
    assert len(parsed["geometry_sha256"]) == 64
    assert parsed["source_artifact_id"] == "test:bundle"
    assert parsed["atom_map"][0]["xyz_angstrom"] == XYZ[0]
    assert delivery_receipt(req, result.structured_results)["status"] == "delivered"
    assert case.model_dump_json() == before
