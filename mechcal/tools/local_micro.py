from __future__ import annotations

from typing import Any

from mechcal.schemas import (
    AgentExecutionPlan,
    ArtifactRecord,
    CaseRun,
    DispatchRequest,
    EvidenceUnit,
    MicroscopicArtifactBundle,
    ToolExecutionResult,
)
from mechcal.schemas.failures import FailureReport
from mechcal.tools.amesp_baseline import AmespBaselineError
from mechcal.tools.amesp_microscopic import (
    MICROSCOPIC_ROUTE_PROFILES,
    AmespMicroscopicError,
    AmespMicroscopicRunner,
    MicroscopicRouteProfile,
)


class LocalMicroscopicTool:
    def __init__(
        self,
        *,
        enable_amesp: bool = False,
        amesp_runner: AmespMicroscopicRunner | None = None,
    ) -> None:
        self.enable_amesp = enable_amesp
        self.amesp_runner = amesp_runner or AmespMicroscopicRunner()

    def run(
        self,
        request: DispatchRequest,
        case_run: CaseRun,
        *,
        execution_plan: AgentExecutionPlan | None = None,
        schema_feedback: str | None = None,
    ) -> ToolExecutionResult:
        del schema_feedback
        profile = MICROSCOPIC_ROUTE_PROFILES.get(request.route)
        if profile is None:
            return self._failed_result(
                request,
                status="unsupported",
                failure=FailureReport(
                    kind="capability_unsupported",
                    message=f"Microscopic route is not implemented in MechCAL: {request.route}",
                    recoverable=True,
                ),
            )
        if profile.route_kind == "unsupported":
            return self._failed_result(
                request,
                status="unsupported",
                failure=FailureReport(
                    kind="capability_unsupported",
                    message="Low-cost excited-state relaxation has not been validated in MechCAL.",
                    recoverable=False,
                ),
            )
        amesp_required = profile.route_kind in {
            "baseline",
            "prepared_photophysics",
            "prepared_series",
            "targeted",
        }
        if amesp_required and not self.enable_amesp:
            return self._failed_result(
                request,
                status="unsupported",
                failure=FailureReport(
                    kind="capability_unsupported",
                    message=(
                        "Amesp execution is disabled for this microscopic adapter. "
                        "Enable MECHCAL_ENABLE_AMESP or pass --amesp."
                    ),
                    recoverable=True,
                ),
            )
        try:
            tool_args = _bounded_microscopic_tool_args(request.route, execution_plan)
            source_artifact = self._source_artifact(
                profile,
                case_run,
                preferred_artifact_id=tool_args.get("source_artifact_id"),
            )
            result = self.amesp_runner.run(
                capability_name=request.route,
                source_artifact=source_artifact,
                case_id=case_run.case_id,
                round_id=request.round_id,
                tool_args=tool_args,
            )
        except AmespMicroscopicError as exc:
            status = _status_for_amesp_error(exc.code)
            return self._failed_result(
                request,
                status=status,
                failure=FailureReport(
                    kind=_failure_kind_for_amesp_error(exc.code),
                    message=exc.message,
                    recoverable=True,
                    details={"code": exc.code, **exc.details},
                ),
            )
        except AmespBaselineError as exc:
            return self._failed_result(
                request,
                status="failed",
                failure=FailureReport(
                    kind="runtime_failed",
                    message=exc.message,
                    recoverable=True,
                    details={
                        "code": exc.code,
                        "file_paths": exc.file_paths,
                        **exc.details,
                    },
                ),
            )

        artifact = _artifact_from_bundle(
            result.bundle,
            profile=profile,
            round_id=request.round_id,
        )
        evidence = _evidence_from_bundle(
            result.bundle,
            profile=profile,
            request=request,
            artifact=artifact,
            case_run=case_run,
        )
        return ToolExecutionResult(
            tool_result_id=f"{request.round_id}:microscopic",
            round_id=request.round_id,
            agent_name="microscopic",
            dispatch_id=request.dispatch_id,
            capability_id=request.capability_id,
            selected_route=request.route,
            tool_calls=[f"amesp.{request.route}"],
            raw_results={
                "tool_args": tool_args,
                "bundle_manifest": str(result.manifest_path),
                "step_records": [
                    item.model_dump(mode="json") for item in result.bundle.step_records
                ],
            },
            structured_results={
                "selected_route": request.route,
                "tool_args": tool_args,
                "bundle_id": result.bundle.bundle_id,
                "bundle_status": result.bundle.status,
                "parsed_observables": result.bundle.parsed_observables,
                "missing_deliverables": result.bundle.missing_deliverables,
                "limitations": result.bundle.limitations,
            },
            status="success",
            evidence_units=[evidence],
            artifact_updates=[artifact],
            summary=(
                f"Microscopic completed route {request.route} and returned a typed "
                f"{profile.artifact_kind} artifact."
            ),
            failure=FailureReport(),
        )

    def _source_artifact(
        self,
        profile: MicroscopicRouteProfile,
        case_run: CaseRun,
        *,
        preferred_artifact_id: Any = None,
    ) -> ArtifactRecord:
        kind_by_route = {
            "baseline": "prepared_structure",
            "bundle_discovery": "amesp_baseline_bundle",
            "parse_only": "amesp_baseline_bundle",
            "prepared_photophysics": "prepared_structure",
            "prepared_discovery": "prepared_structure",
            "prepared_series": "prepared_structure",
            "targeted": "amesp_baseline_bundle",
        }
        required_kind = kind_by_route[profile.route_kind]
        candidates = [
            artifact
            for artifact in case_run.artifact_manifest.artifacts
            if artifact.kind == required_kind and artifact.status in {"available", "partial"}
        ]
        if not candidates:
            raise AmespMicroscopicError(
                "precondition_missing",
                f"Microscopic route requires artifact kind {required_kind}.",
                details={
                    "required_artifact_kind": required_kind,
                    "available_artifacts": case_run.artifact_manifest.artifact_ids(),
                },
            )
        if preferred_artifact_id is not None:
            preferred_id = str(preferred_artifact_id)
            return next(
                (
                    artifact
                    for artifact in candidates
                    if artifact.artifact_id == preferred_id
                ),
                candidates[-1],
            )
        return candidates[-1]

    def _failed_result(
        self,
        request: DispatchRequest,
        *,
        status: str,
        failure: FailureReport,
    ) -> ToolExecutionResult:
        report_id = f"{request.round_id}:microscopic"
        evidence = EvidenceUnit(
            evidence_id=f"{report_id}:{request.route}:status",
            round_id=request.round_id,
            source_report_id=report_id,
            agent_name="microscopic",
            capability_id=request.capability_id,
            claim=_microscopic_status_claim(request, failure=failure),
            context="microscopic route status",
            basis="checklist",
            support="unresolved",
            limits=[
                "No computed microscopic evidence was produced for this route outcome.",
                "Planner may use this only as a capability boundary or missing-evidence "
                "record; it must not be treated as spectra, quantum-yield, lifetime, "
                "3D, or quantum-chemical evidence.",
            ],
            observable=request.route,
            family=request.evidence_goal_family,
            relation="unknown",
            status=_evidence_status_for_tool_status(status),
            observable_tags=[request.route],
            summary=_microscopic_failure_summary(request, failure=failure),
        )
        return ToolExecutionResult(
            tool_result_id=report_id,
            round_id=request.round_id,
            agent_name="microscopic",
            dispatch_id=request.dispatch_id,
            capability_id=request.capability_id,
            selected_route=request.route,
            tool_calls=[f"amesp.{request.route}"],
            raw_results={"failure": failure.model_dump(mode="json")},
            structured_results={
                "selected_route": request.route,
                "runtime_status": status,
                "boundary_only": True,
                "failure_kind": failure.kind,
            },
            status=status,  # type: ignore[arg-type]
            evidence_units=[evidence],
            summary=f"Microscopic route {request.route} returned a typed failure.",
            failure=failure,
        )


def _evidence_status_for_tool_status(status: str) -> str:
    return {
        "partial": "partial",
        "precondition_missing": "missing",
        "unsupported": "unsupported",
    }.get(status, "failed")


def _artifact_from_bundle(
    bundle: MicroscopicArtifactBundle,
    *,
    profile: MicroscopicRouteProfile,
    round_id: str,
) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=bundle.bundle_id,
        kind=profile.artifact_kind,
        created_by_round=round_id,
        created_by_agent="microscopic",
        status=bundle.status,
        file_paths=bundle.file_paths,
        observable_tags=[*profile.evidence_tags, bundle.capability_name],
        reusable_for=["microscopic", "artifact_inspection"],
        metadata={
            "capability_name": bundle.capability_name,
            "bundle": bundle.model_dump(mode="json"),
        },
    )


def _evidence_from_bundle(
    bundle: MicroscopicArtifactBundle,
    *,
    profile: MicroscopicRouteProfile,
    request: DispatchRequest,
    artifact: ArtifactRecord,
    case_run: CaseRun,
) -> EvidenceUnit:
    report_id = f"{request.round_id}:microscopic"
    return EvidenceUnit(
        evidence_id=f"{report_id}:{request.route}",
        round_id=request.round_id,
        source_report_id=report_id,
        agent_name="microscopic",
        capability_id=request.capability_id,
        claim=_microscopic_bundle_claim(
            bundle,
            profile=profile,
            request=request,
            case_run=case_run,
        ),
        context=f"computed microscopic {profile.artifact_kind}",
        basis="computed",
        support="not_applicable" if bundle.status == "available" else "unresolved",
        limits=[
            "Microscopic evidence is bounded by the configured route and parsed "
            "artifact deliverables."
        ],
        observable=request.route,
        family=request.evidence_goal_family,
        relation="unknown",
        status="present" if bundle.status == "available" else "partial",
        artifact_refs=[*bundle.source_artifact_ids, artifact.artifact_id],
        observable_tags=[*profile.evidence_tags, request.route],
        summary=(
            f"Microscopic route {request.route} produced a typed {profile.artifact_kind} "
            f"with status {bundle.status}."
        ),
        metrics=_microscopic_evidence_metrics(bundle),
    )


def _microscopic_bundle_claim(
    bundle: MicroscopicArtifactBundle,
    *,
    profile: MicroscopicRouteProfile,
    request: DispatchRequest,
    case_run: CaseRun,
) -> str:
    parsed = bundle.parsed_observables if isinstance(bundle.parsed_observables, dict) else {}
    route_summary = parsed.get("route_summary")
    summary = route_summary if isinstance(route_summary, dict) else {}
    smiles = case_run.input.smiles
    route = request.route

    if route == "run_baseline_bundle":
        bright = parsed.get("bright_state")
        oscillator = None
        if isinstance(bright, dict):
            oscillator = bright.get("oscillator_strength")
        if not parsed.get("state_count"):
            available = parsed.get("available_descriptors")
            available_text = ", ".join(available) if isinstance(available, list) else "none"
            return (
                "Low-cost S0 baseline route retained successful ground-state "
                f"observables ({available_text}), but S1 vertical excitation, "
                "state ordering, and oscillator-strength evidence were unavailable. "
                "Planner may use this only as partial microscopic evidence and owns "
                "any PL, photochemistry, conical-intersection, or dynamics interpretation."
            )
        return (
            "Low-cost S0/S1 baseline route records state-ordering and oscillator "
            f"strength observables (f={oscillator if oscillator is not None else 'n/a'}). "
            "Planner owns any PL, photochemistry, conical-intersection, or dynamics "
            "interpretation."
        )
    if route == "run_conformer_bundle":
        osc_range = summary.get("oscillator_strength_range", "n/a")
        return (
            "Conformer bundle route records conformer-dependent brightness observables "
            f"(oscillator-strength range {osc_range}). It does not assign a mechanism."
        )
    if route == "run_torsion_snapshots":
        osc_range = summary.get("oscillator_strength_range", "n/a")
        if _is_neutral_aromatic_alkene(smiles):
            return (
                "Torsion snapshot route records brightness observables along an "
                "aromatic alkene rotation coordinate "
                f"(oscillator-strength range {osc_range}). Planner owns any torsional, "
                "isomerization-like, or CI-pathway interpretation."
            )
        return (
            "Torsion snapshot route records whether rotated geometries modulate "
            f"computed brightness (oscillator-strength range {osc_range}). It does "
            "not assign RIM, TICT, or nonradiative-pathway status."
        )
    if route in {
        "run_targeted_localized_orbital_analysis",
        "run_targeted_natural_orbital_analysis",
    }:
        descriptors = parsed.get("available_descriptors") or []
        return (
            "Targeted orbital route records available charge-localization or CT "
            f"descriptors ({', '.join(str(item) for item in descriptors) or 'none listed'}). "
            "Planner owns any ICT, TICT, or intermolecular charge-transfer interpretation."
        )
    return (
        f"Microscopic route {route} produced a typed {profile.artifact_kind} artifact "
        "with route-local observables only. Planner owns scientific synthesis."
    )


def _microscopic_evidence_metrics(bundle: MicroscopicArtifactBundle) -> dict[str, object]:
    parsed = bundle.parsed_observables if isinstance(bundle.parsed_observables, dict) else {}
    metrics: dict[str, object] = {
        "state_count": parsed.get("state_count", 0),
        "missing_deliverable_count": len(bundle.missing_deliverables),
    }
    bright = parsed.get("bright_state")
    if isinstance(bright, dict):
        oscillator = bright.get("oscillator_strength")
        if isinstance(oscillator, int | float):
            metrics["oscillator_strength"] = float(oscillator)
        state_index = bright.get("state_index")
        if isinstance(state_index, int | float):
            metrics["bright_state_index"] = int(state_index)
    return metrics


def _microscopic_status_claim(
    request: DispatchRequest,
    *,
    failure: FailureReport,
) -> str:
    if request.route == "unsupported_excited_state_relaxation":
        return (
            "This route did not compute conical-intersection optimization, trajectory "
            "surface hopping, nonadiabatic relaxation, CI barriers, or dynamics "
            "deliverables in the current tool scope."
        )
    return (
        f"Microscopic route {request.route} did not produce computed evidence "
        f"({failure.kind}); this remains an unresolved evidence need. The MAS may "
        "continue with SMILES-level macro/status evidence, but no photophysical "
        "calculation should be inferred from this failed route."
    )


def _microscopic_failure_summary(
    request: DispatchRequest,
    *,
    failure: FailureReport,
) -> str:
    return (
        f"{failure.message} Route {request.route} is recorded as boundary-only "
        "status evidence, not computed microscopic support."
    )


def _is_neutral_aromatic_alkene(smiles: str) -> bool:
    aromatic = smiles.count("c")
    hetero_tokens = sum(smiles.count(token) for token in ("N", "O", "S", "n", "o", "s"))
    return aromatic >= 12 and "=" in smiles and hetero_tokens == 0


def _bounded_microscopic_tool_args(
    route: str,
    execution_plan: AgentExecutionPlan | None,
) -> dict[str, Any]:
    raw = execution_plan.tool_args if execution_plan is not None else {}
    args: dict[str, Any] = {}
    source_artifact_id = raw.get("source_artifact_id")
    if source_artifact_id is not None:
        args["source_artifact_id"] = str(source_artifact_id)
    if route in {
        "run_conformer_bundle",
        "run_conformer_state_probe",
        "run_torsion_snapshots",
        "run_torsion_brightness_coupling_scan",
    }:
        args["max_members"] = _bounded_int(raw.get("max_members"), minimum=2, maximum=6)
    if route in {
        "run_baseline_bundle",
        "run_bright_dark_state_ordering",
        "run_conformer_bundle",
        "run_conformer_state_probe",
        "run_torsion_snapshots",
        "run_torsion_brightness_coupling_scan",
        "run_targeted_state_characterization",
        "run_targeted_transition_dipole_analysis",
        "run_targeted_charge_redistribution_analysis",
        "run_ris_state_characterization",
        "run_targeted_localized_orbital_analysis",
        "run_targeted_natural_orbital_analysis",
        "run_solvation_polarity_proxy",
    }:
        args["s1_nstates"] = _bounded_int(raw.get("s1_nstates"), minimum=1, maximum=10)
        args["td_tout"] = _bounded_int(raw.get("td_tout"), minimum=1, maximum=10)
    if route == "run_solvation_polarity_proxy":
        args["solvents"] = _bounded_choice_list(
            raw.get("solvents"),
            allowed={
                "water",
                "acetonitrile",
                "methanol",
                "ethanol",
                "toluene",
                "benzene",
                "acetone",
                "chloroform",
            },
            maximum=4,
        )
    if route == "run_charge_population_panel":
        args["charge_schemes"] = _bounded_choice_list(
            raw.get("charge_schemes"),
            allowed={"mulliken", "lowdin", "hirshfeld", "cm5"},
            maximum=4,
        )
    return {key: value for key, value in args.items() if value is not None}


def _status_for_amesp_error(code: str) -> str:
    if code == "precondition_missing":
        return "precondition_missing"
    if code == "capability_unsupported":
        return "unsupported"
    if code.startswith("partial_"):
        return "partial"
    return "failed"


def _failure_kind_for_amesp_error(code: str) -> str:
    if code in {"precondition_missing", "capability_unsupported"}:
        return code
    if code.startswith("partial_"):
        return "partial_evidence"
    return "runtime_failed"


def _bounded_int(value: Any, *, minimum: int, maximum: int) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return min(max(numeric, minimum), maximum)


def _bounded_choice_list(
    value: Any,
    *,
    allowed: set[str],
    maximum: int,
) -> list[str] | None:
    if value is None:
        return None
    raw_items = value if isinstance(value, list) else [value]
    selected = []
    for item in raw_items:
        normalized = str(item).strip().lower()
        if normalized in allowed and normalized not in selected:
            selected.append(normalized)
        if len(selected) >= maximum:
            break
    return selected or None
