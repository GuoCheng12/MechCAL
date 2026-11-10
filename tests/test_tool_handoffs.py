"""Cross-molecule contracts for honest tool scope and computed geometry reuse."""

from types import SimpleNamespace

import pytest

from mechcal.capabilities import default_capability_registry
from mechcal.runtime.command_operations import protocol, put
from mechcal.runtime.command_route_status import route_status
from mechcal.schemas import ArtifactRecord, CaseRun, MicroscopicArtifactBundle
from mechcal.schemas.microscopic import MEMBER_SERIES_KINDS
from mechcal.tools.amesp_microscopic import AmespMicroscopicError, AmespMicroscopicRunner, _ct_descriptor_payload
from mechcal.tools.local_micro import LocalMicroscopicTool
from tests.test_tool_parameters import case_with_structure, request


def series(case, kind="torsion_snapshot_bundle"):
    template = case.artifact_manifest.artifacts[0]
    artifact = ArtifactRecord(artifact_id="series:one", kind=kind,
        created_by_round="R001", created_by_agent="microscopic", metadata={"bundle": {
            "source_artifact_ids": [template.artifact_id],
            "parsed_observables": {"molecular_context": {"charge": 0, "multiplicity": 1},
                "route_records": [{"member_id": "angle:90", "first_oscillator_strength": 0.2}],
                "member_artifacts": [{"member_id": "angle:90", "member_status": "success",
                                      "prepared_xyz_path": template.file_paths["xyz"]}]}}})
    case.artifact_manifest = case.artifact_manifest.with_records([artifact])
    return artifact


@pytest.mark.parametrize("smiles", ["CCCO", "c1ccccc1-c1ccncc1", "NCC(=O)O"])
@pytest.mark.parametrize("kind", MEMBER_SERIES_KINDS)
def test_series_selection_cross_molecule_and_bundle(tmp_path, smiles, kind):
    case = case_with_structure(tmp_path, smiles)
    artifact = series(case, kind)
    registry = default_capability_registry()
    args = {"source_artifact_id": artifact.artifact_id, "member_id": "angle:90"}
    req = request("select_computed_geometry", args)
    assert registry.validate_dispatch(req, case.artifact_manifest) == []
    assert registry.availability_violations(req, case.artifact_manifest, amesp_enabled=False) == []
    before = case.model_dump_json()
    result = LocalMicroscopicTool().run(req, case)
    assert result.status == "success"
    selected = result.artifact_updates[0]
    assert selected.metadata["geometry_validation"] == "fixed_geometry_series_member"
    assert selected.metadata["geometry_source"] == "explicit_computed_selection"
    assert "geometry_converged" not in selected.metadata
    assert selected.metadata["parent_member_id"] == "angle:90"
    assert "converged computed geometry" not in result.summary
    assert case.model_dump_json() == before
    case.artifact_manifest = case.artifact_manifest.with_records([selected])
    assert registry.readiness("microscopic.run_baseline_bundle", case.artifact_manifest,
        amesp_enabled=True, source_artifact_id=selected.artifact_id)["status"] == "ready"
    assert registry.readiness("microscopic.run_baseline_bundle", case.artifact_manifest,
        amesp_enabled=False, source_artifact_id=selected.artifact_id)["status"] == "disabled"


@pytest.mark.parametrize("fault", ["failed", "no_result", "duplicate_member", "duplicate_result",
    "conflicting_failure", "missing_xyz", "missing_topology", "wrong_charge", "unknown_member", "foreign_case"])
def test_series_selection_rejects_invalid_provenance(tmp_path, fault):
    case = case_with_structure(tmp_path, "CCCO")
    artifact = series(case)
    obs = artifact.metadata["bundle"]["parsed_observables"]
    args = {"source_artifact_id": artifact.artifact_id, "member_id": "angle:90"}
    if fault == "failed":
        obs["member_artifacts"][0]["member_status"] = "failed"
    elif fault == "no_result":
        obs["route_records"] = []
    elif fault == "duplicate_member":
        obs["member_artifacts"] *= 2
    elif fault == "duplicate_result":
        obs["route_records"] *= 2
    elif fault == "conflicting_failure":
        obs["member_failures"] = [{"member_id": "angle:90"}]
    elif fault == "missing_xyz":
        obs["member_artifacts"][0]["prepared_xyz_path"] = str(tmp_path / "missing.xyz")
    elif fault == "missing_topology":
        artifact.metadata["bundle"]["source_artifact_ids"] = ["absent"]
    elif fault == "wrong_charge":
        obs["molecular_context"]["charge"] = 1
    elif fault == "unknown_member":
        args["member_id"] = "absent"
    elif fault == "foreign_case":
        case = case_with_structure(tmp_path / "other", "CCN")
    result = LocalMicroscopicTool().run(request("select_computed_geometry", args), case)
    assert result.status == "precondition_missing"
    assert not result.artifact_updates
    assert all(e.status == "missing" for e in result.evidence_units)


def test_legacy_placeholder_disabled_at_discovery_adapter_and_runner(tmp_path):
    case = case_with_structure(tmp_path)
    registry = default_capability_registry()
    cid = "microscopic.extract_torsion_candidates_from_bundle"
    assert registry.readiness(cid, case.artifact_manifest, amesp_enabled=True)["status"] == "disabled"
    result = LocalMicroscopicTool(enable_amesp=True).run(request(cid.split(".")[1], {}), case)
    assert result.status == "unsupported"
    assert all(e.status == "unsupported" for e in result.evidence_units)
    assert "list_rotatable_dihedrals" in result.failure.message
    with pytest.raises(AmespMicroscopicError, match="list_rotatable_dihedrals"):
        AmespMicroscopicRunner().run(capability_name=cid.split(".")[1], source_artifact=case.artifact_manifest.artifacts[0],
                                    case_id=case.case_id, round_id="R001")


def test_scalar_extraction_only_promises_delivered_fields():
    # Parser-only test: no backend, no invented spatial values.
    bundle = MicroscopicArtifactBundle(bundle_id="baseline", capability_name="run_baseline_bundle",
        geometry_source="prepared", status="available", excited_states=[{"state_index": 1, "oscillator_strength": 0.1}],
        parsed_observables={"dominant_transitions": [{"state_index": 1}], "mulliken_charges": [0.0]})
    payload, missing, limitations = _ct_descriptor_payload(None, bundle)
    assert missing == []
    assert "hole_electron_separation" in payload["outside_scope"]
    assert "hole_electron_separation" not in payload["available_descriptors"]
    assert any("not excited-state" in v for v in limitations)
    bundle.parsed_observables = {}
    _, missing, _ = _ct_descriptor_payload(None, bundle)
    assert set(missing) == {"mulliken_charges", "dominant_transitions"}


def test_bound_route_uses_result_outputs_not_unrelated_latest_structure(tmp_path):
    case = case_with_structure(tmp_path, "CCO")
    artifact = series(case)
    case = put(case, results={"result:series": {"tool_result_id": "result:series", "status": "success",
        "capability_id": "microscopic.run_torsion_snapshots", "artifact_updates": [artifact.model_dump(mode="json")]}})
    registry = default_capability_registry()
    cid = "microscopic.select_computed_geometry"
    tool = next(t for t in registry.compact_catalog(case.artifact_manifest, amesp_enabled=True) if t["capability_id"] == cid)
    route = SimpleNamespace(capability_id=cid, input_refs=["result:series"])
    before = case.model_dump_json()
    state = route_status(case, route, tool, registry=registry, amesp_enabled=True)
    assert state["status"] == "ready"
    assert state["source_checks"][0]["source_artifact_id"] == artifact.artifact_id
    assert case.model_dump_json() == before
    # A good structure elsewhere must not make a different bound source valid.
    route.input_refs = [case.artifact_manifest.artifacts[0].artifact_id]
    assert route_status(case, route, tool, registry=registry, amesp_enabled=True)["status"] == "missing_input"
    route.input_refs = ["other_case:artifact"]
    assert route_status(case, route, tool, registry=registry, amesp_enabled=True)["status"] == "missing_input"
    assert route_status(case, route, None, registry=registry)["status"] == "not_enabled"
    restored = CaseRun.model_validate_json(case.model_dump_json())
    assert protocol(restored) == protocol(case)


def test_missing_input_advertises_preparation_without_running_or_substituting(tmp_path):
    case = case_with_structure(tmp_path, "CCCO")
    artifact = series(case)
    registry = default_capability_registry()
    cid = "microscopic.run_baseline_bundle"
    tool = next(t for t in registry.compact_catalog(case.artifact_manifest, amesp_enabled=True) if t["capability_id"] == cid)
    route = SimpleNamespace(capability_id=cid, input_refs=[artifact.artifact_id])
    before = case.model_dump_json()
    state = route_status(case, route, tool, registry=registry, amesp_enabled=True)
    assert state["status"] == "missing_input"
    assert state["preparation_options"] == [{"capability_id": "microscopic.select_computed_geometry",
                                             "source_artifact_id": artifact.artifact_id}]
    assert case.model_dump_json() == before


def test_discovery_finds_valid_member_but_execution_never_substitutes(tmp_path):
    case = case_with_structure(tmp_path, "CCCO")
    a = series(case)
    obs = a.metadata["bundle"]["parsed_observables"]
    obs["route_records"].insert(0, {"member_id": "angle:bad"})
    obs["member_artifacts"].insert(0, {"member_id": "angle:bad", "member_status": "success",
                                      "prepared_xyz_path": str(tmp_path / "absent.xyz")})
    registry = default_capability_registry()
    cid = "microscopic.select_computed_geometry"
    assert registry.readiness(cid, case.artifact_manifest, amesp_enabled=False)["status"] == "ready"
    assert registry.readiness(cid, case.artifact_manifest, amesp_enabled=False,
        source_artifact_id=a.artifact_id, tool_args={"member_id": "angle:bad"})["status"] == "missing_input"


def test_bound_input_is_checked_even_when_default_input_is_unsupported(tmp_path, monkeypatch):
    case = case_with_structure(tmp_path)
    registry = default_capability_registry()
    cid = "microscopic.run_baseline_bundle"
    source = case.artifact_manifest.artifacts[0].artifact_id
    tool = {"capability_id": cid, "status": "disabled", "reason": "Default input outside scope."}
    calls = []
    def readiness(_, __, **kwargs):
        calls.append(kwargs)
        return {"status": "ready", "missing_inputs": [], "reason": None}
    monkeypatch.setattr(registry, "readiness", readiness)
    result = route_status(case, SimpleNamespace(capability_id=cid, input_refs=[source]), tool,
                          registry=registry, amesp_enabled=True)
    assert result["status"] == "ready"
    assert calls == [{"amesp_enabled": True, "source_artifact_id": source}]
