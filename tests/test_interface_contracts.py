"""All registered interfaces: contracts only, never fabricated chemistry results."""

from copy import deepcopy

import pytest

from mechcal.capabilities import CapabilityRegistry
from mechcal.capabilities.catalog import default_capabilities
from mechcal.capabilities.external_qc import external_capabilities
from mechcal.mcp.server import _worker_payload
from mechcal.schemas import AgentExecutionPlan, ArtifactManifest, ArtifactRecord, DispatchRequest
from mechcal.schemas.tool_arguments import argument_model
from mechcal.tools.amesp_microscopic import MICROSCOPIC_ROUTE_PROFILES, AmespMicroscopicRunner
from mechcal.tools.local_macro import LocalMacroTool, _ROUTE_HANDLERS
from mechcal.tools.local_micro import LocalMicroscopicTool
from mechcal.tools.qc_process import QCConfig
from test_amesp_scan import add_geometry_bundle
from test_tool_parameters import case_with_structure, request

SPECS = default_capabilities() + external_capabilities(QCConfig())
REGISTRY = CapabilityRegistry(SPECS)


def dispatch(spec, **args):
    return DispatchRequest(dispatch_id="audit", round_id="R001", agent_name=spec.owner_agent,
                           capability_id=spec.capability_id, route=spec.route,
                           evidence_goal_family=spec.evidence_family, task="Interface test", objective="Interface test",
                           tool_args=args)


class MustNotExecute:
    def run(self, *args, **kwargs):
        raise AssertionError("A rejected request must not execute a tool")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    return case_with_structure(tmp_path_factory.mktemp("interface"), "CCO")


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.capability_id)
@pytest.mark.parametrize("fault", ["parameter", "owner", "route"])
def test_every_interface_rejects_invalid_request_before_execution(spec, fault, prepared):
    req = dispatch(spec)
    if fault == "parameter":
        req.tool_args = {"not_a_parameter": 1}
    elif fault == "owner":
        req.agent_name = "microscopic" if spec.owner_agent == "macro" else "macro"
    else:
        req.route = "unregistered_route"
    result = _worker_payload(MustNotExecute(), registry=REGISTRY, owner=spec.owner_agent,
        dispatch_request=req.model_dump(mode="json"), case_run=prepared.model_dump(mode="json"),
        execution_plan=None, schema_feedback=None)["tool_result"]
    assert result["status"] == "failed" and result["structured_results"]["boundary_only"]
    assert not result["artifact_updates"] and not result["tool_calls"]
    assert all(e["basis"] == "checklist" and e["status"] == "failed" for e in result["evidence_units"])


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.capability_id)
def test_catalog_schema_handler_and_missing_input_contract(spec):
    schema = argument_model(spec.route, spec.owner_agent).model_json_schema()
    assert schema["additionalProperties"] is False
    empty = REGISTRY.readiness(spec.capability_id, ArtifactManifest(case_id="empty"), amesp_enabled=True)
    if spec.owner_agent == "macro":
        assert spec.route in _ROUTE_HANDLERS and empty["status"] == "ready"
    elif not spec.metadata.get("external_backend"):
        profile = MICROSCOPIC_ROUTE_PROFILES[spec.route]
        assert empty["status"] in {"missing_input", "disabled"}
        if profile.route_kind != "unsupported":
            assert spec.input_artifact_kinds, "A supported microscopic route must declare its source kind"
            assert profile.artifact_kind in spec.outputs
    else:
        assert empty["status"] == "disabled"


@pytest.mark.parametrize("spec", [s for s in SPECS if s.owner_agent == "macro"], ids=lambda s: s.route)
def test_every_macro_route_returns_bounded_structural_results(spec, prepared):
    req = dispatch(spec)
    result = _worker_payload(LocalMacroTool(), registry=REGISTRY, owner="macro",
        dispatch_request=req.model_dump(mode="json"), case_run=prepared.model_dump(mode="json"),
        execution_plan=None, schema_feedback=None)["tool_result"]
    assert result["status"] == "success"
    assert result["structured_results"]["selected_route"] == spec.route
    assert result["evidence_units"] and all(e["limits"] for e in result["evidence_units"])


@pytest.mark.parametrize("fault", ["round_id", "dispatch_id", "capability_id", "agent_name", "selected_route",
                                   "precondition_status", "override", "ablation"])
def test_worker_plan_and_ablation_cannot_bypass_mcp(fault, prepared):
    spec = SPECS[0]
    req = dispatch(spec, focus_tags=["rotor"])
    plan = AgentExecutionPlan(plan_id="plan", round_id=req.round_id, dispatch_id=req.dispatch_id,
        capability_id=req.capability_id, agent_name=req.agent_name, selected_route=req.route,
        tool_arg_rationale="Fixture", parameter_adjustment_hint="Fixture", tool_args=req.tool_args)
    case = prepared.model_copy(deep=True)
    if fault == "override":
        plan.tool_args = {"focus_tags": ["unrequested"]}
    elif fault == "ablation":
        case.runtime["ablation"] = {"enabled_capability_owners": ["microscopic"]}
    elif fault == "agent_name":
        plan.agent_name = "microscopic"
    elif fault == "precondition_status":
        plan.precondition_status = "missing"
    else:
        setattr(plan, fault, "different")
    result = _worker_payload(MustNotExecute(), registry=REGISTRY, owner="macro",
        dispatch_request=req.model_dump(mode="json"), case_run=case.model_dump(mode="json"),
        execution_plan=plan.model_dump(mode="json"), schema_feedback=None)["tool_result"]
    assert result["status"] == "failed"


@pytest.mark.parametrize("problem", [None, "member", "unconverged", "missing_file", "state", "charge", "source"])
def test_selection_readiness_agrees_with_execution(tmp_path, problem):
    case = case_with_structure(tmp_path, "CCO")
    source = add_geometry_bundle(case, tmp_path, converged=problem != "unconverged")
    args = {"source_artifact_id": source.artifact_id, "member_id": "scan:01"}
    obs = source.metadata["bundle"]["parsed_observables"]
    if problem == "member":
        args["member_id"] = "missing"
    elif problem == "missing_file":
        obs["route_records"][0]["xyz_path"] = str(tmp_path / "absent.xyz")
    elif problem == "state":
        obs["state_root"] = 1
    elif problem == "charge":
        obs["molecular_context"]["charge"] = 1
    elif problem == "source":
        args["source_artifact_id"] = "absent"
    req = request("select_computed_geometry", args)
    violations = REGISTRY.availability_violations(req, case.artifact_manifest, amesp_enabled=False)
    result = LocalMicroscopicTool().run(req, case)
    assert bool(violations) == (problem is not None)
    assert (result.status == "success") == (problem is None)


@pytest.mark.parametrize("route", ["run_baseline_bundle", "run_bright_dark_state_ordering"])
def test_selected_geometry_is_not_reoptimized(tmp_path, monkeypatch, route):
    case = case_with_structure(tmp_path, "CCO")
    source = case.artifact_manifest.artifacts[0].model_copy(deep=True)
    source.metadata["geometry_source"] = "explicit_computed_selection"
    runner = AmespMicroscopicRunner()
    observed = []

    def single(**kwargs):
        observed.append(deepcopy(kwargs))
        # Stop before any external calculation. Reaching this call proves fixed-point dispatch.
        raise RuntimeError("fixed_point_probe")

    def optimize(**kwargs):
        pytest.fail("Explicitly selected coordinates must not be reoptimized")

    monkeypatch.setattr(runner, "_run_singlepoint", single)
    monkeypatch.setattr(runner, "_run_step", optimize)
    with pytest.raises(RuntimeError, match="fixed_point_probe"):
        runner.run(capability_name=route, source_artifact=source, case_id=case.case_id, round_id="R001")
    assert len(observed) == 1


@pytest.mark.parametrize("route", ["parse_snapshot_outputs", "inspect_raw_artifact_bundle",
                                   "run_targeted_state_characterization"])
def test_missing_bundle_manifest_is_typed(tmp_path, route):
    case = case_with_structure(tmp_path, "CCO")
    source = ArtifactRecord(artifact_id="bundle", kind="amesp_baseline_bundle", created_by_agent="microscopic",
        created_by_round="R001", file_paths={"manifest": str(tmp_path / "missing.json")})
    case.artifact_manifest = case.artifact_manifest.with_records([source])
    result = LocalMicroscopicTool(enable_amesp=True).run(request(route, {"source_artifact_id": "bundle"}), case)
    assert result.status == "precondition_missing" and not result.artifact_updates


def test_selected_member_is_required():
    with pytest.raises(ValueError):
        argument_model("select_computed_geometry").model_validate({"source_artifact_id": "scan"})


def test_spatial_preflight_uses_requested_state(monkeypatch):
    from mechcal.tools import amesp_microscopic as micro
    seen = []
    monkeypatch.setattr(micro, "backend_unavailable", lambda: None)
    monkeypatch.setattr(micro, "_bundle_from_artifact", lambda artifact: object())
    def source_data(bundle, state_root):
        seen.append(state_root)
        return ({}, "", ["C"], [], [], .5, 1.0, [0, 0, 0])
    monkeypatch.setattr(micro, "source_data", source_data)
    source = ArtifactRecord(artifact_id="bundle", kind="amesp_baseline_bundle",
        created_by_agent="microscopic", created_by_round="R001")
    micro.microscopic_preflight("analyze_hole_electron_from_bundle", source, tool_args={"state_root": 3})
    assert seen == [3]


@pytest.mark.parametrize("route,args,status,kind", [
    ("run_excited_state_relaxation", {}, "unsupported", "capability_unsupported"),
    ("run_baseline_bundle", {"source_artifact_id": "absent"}, "precondition_missing", "precondition_missing"),
])
def test_mcp_failure_categories_remain_distinct(prepared, route, args, status, kind):
    req = request(route, args)
    case = prepared.model_copy(deep=True)
    case.runtime["tool_environment"] = {"amesp_enabled": True}
    result = _worker_payload(MustNotExecute(), registry=REGISTRY, owner="microscopic",
        dispatch_request=req.model_dump(mode="json"), case_run=case.model_dump(mode="json"),
        execution_plan=None, schema_feedback=None)["tool_result"]
    assert result["status"] == status and result["failure"]["kind"] == kind
    assert result["failure"]["recoverable"] == (status != "unsupported")
