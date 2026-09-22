from __future__ import annotations

import re
from dataclasses import dataclass
from math import dist
from pathlib import Path
from typing import Any, Literal

from mechcal.schemas import ArtifactRecord, MicroscopicArtifactBundle, MicroscopicStepRecord
from mechcal.tools.amesp_baseline import (
    AmespBaselineError,
    AmespBaselineRunner,
    AmespStepResult,
    _parse_excited_states,
    _parse_final_energy,
    _parse_final_geometry,
    _read_xyz_symbols_and_coordinates,
    _safe_label,
    _write_xyz,
)

RouteKind = Literal[
    "baseline",
    "bundle_discovery",
    "parse_only",
    "prepared_photophysics",
    "prepared_discovery",
    "prepared_series",
    "targeted",
    "unsupported",
]
_GROUND_TO_EXCITED_DIPOLE = "Ground to excited state transition electric dipole moments"
_EXCITED_TO_EXCITED_DIPOLE = "Excited to excited state transition electric dipole moments"


@dataclass(frozen=True)
class MicroscopicRouteProfile:
    route_kind: RouteKind
    artifact_kind: str
    evidence_tags: list[str]
    requested_descriptors: list[str]
    s1_mode: str | None = None


@dataclass(frozen=True)
class MicroscopicRouteResult:
    bundle: MicroscopicArtifactBundle
    manifest_path: Path


@dataclass(frozen=True)
class MicroscopicRunOptions:
    max_members: int | None = None
    s1_nstates: int | None = None
    td_tout: int | None = None
    solvents: list[str] | None = None
    charge_schemes: list[str] | None = None


@dataclass(frozen=True)
class PreparedGeometryMember:
    member_id: str
    label: str
    symbols: list[str]
    coordinates: list[list[float]]
    xyz_path: Path
    metadata: dict[str, object]


class AmespMicroscopicError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


MICROSCOPIC_ROUTE_PROFILES: dict[str, MicroscopicRouteProfile] = {
    "run_baseline_bundle": MicroscopicRouteProfile(
        route_kind="baseline",
        artifact_kind="amesp_baseline_bundle",
        evidence_tags=["amesp_s0_s1_baseline", "oscillator_strength"],
        requested_descriptors=["state_ordering", "oscillator_strength", "mulliken_charges"],
        s1_mode="tda_atb_excdip",
    ),
    "run_bright_dark_state_ordering": MicroscopicRouteProfile(
        route_kind="baseline",
        artifact_kind="bright_dark_state_ordering_bundle",
        evidence_tags=["bright_dark_state_ordering", "state_ordering", "oscillator_strength"],
        requested_descriptors=["state_ordering", "oscillator_strength", "bright_state"],
        s1_mode="tda_atb_excdip",
    ),
    "run_frontier_orbital_partition": MicroscopicRouteProfile(
        route_kind="prepared_photophysics",
        artifact_kind="frontier_orbital_partition_bundle",
        evidence_tags=["frontier_orbital_partition", "charge_localization"],
        requested_descriptors=["frontier_orbital_energies", "frontier_orbital_partition"],
    ),
    "run_charge_population_panel": MicroscopicRouteProfile(
        route_kind="prepared_photophysics",
        artifact_kind="charge_population_panel_bundle",
        evidence_tags=["charge_population_panel", "charge_localization"],
        requested_descriptors=[
            "mulliken_charges",
            "lowdin_charges",
            "hirshfeld_charges",
            "cm5_charges",
        ],
    ),
    "run_solvation_polarity_proxy": MicroscopicRouteProfile(
        route_kind="prepared_photophysics",
        artifact_kind="solvation_polarity_proxy_bundle",
        evidence_tags=["solvation_polarity_proxy", "state_ordering", "oscillator_strength"],
        requested_descriptors=[
            "solvent_state_ordering",
            "solvent_energy_shift",
            "solvent_oscillator_shift",
        ],
        s1_mode="tda_atb_gbsa",
    ),
    "list_rotatable_dihedrals": MicroscopicRouteProfile(
        route_kind="prepared_discovery",
        artifact_kind="rotatable_dihedral_inventory",
        evidence_tags=["rotatable_dihedrals", "geometry_precondition"],
        requested_descriptors=["stable dihedral descriptors"],
    ),
    "list_available_conformers": MicroscopicRouteProfile(
        route_kind="prepared_discovery",
        artifact_kind="conformer_inventory",
        evidence_tags=["conformer_inventory", "conformer_sensitivity"],
        requested_descriptors=["stable conformer descriptors"],
    ),
    "list_artifact_bundles": MicroscopicRouteProfile(
        route_kind="bundle_discovery",
        artifact_kind="artifact_bundle_inventory",
        evidence_tags=["artifact_bundle_inventory", "raw_artifact_inspection"],
        requested_descriptors=["artifact bundle descriptors"],
    ),
    "list_artifact_bundle_members": MicroscopicRouteProfile(
        route_kind="bundle_discovery",
        artifact_kind="artifact_bundle_member_inventory",
        evidence_tags=["artifact_bundle_members", "raw_artifact_inspection"],
        requested_descriptors=["artifact bundle member descriptors"],
    ),
    "run_conformer_bundle": MicroscopicRouteProfile(
        route_kind="prepared_series",
        artifact_kind="conformer_bundle",
        evidence_tags=["conformer_sensitivity", "oscillator_strength"],
        requested_descriptors=["state_ordering", "oscillator_strength", "conformer_sensitivity"],
        s1_mode="tda_atb_excdip",
    ),
    "run_conformer_state_probe": MicroscopicRouteProfile(
        route_kind="prepared_series",
        artifact_kind="conformer_state_probe_bundle",
        evidence_tags=["conformer_state_probe", "oscillator_strength"],
        requested_descriptors=["state_ordering", "oscillator_strength", "conformer_sensitivity"],
        s1_mode="tda_atb_excdip",
    ),
    "run_torsion_snapshots": MicroscopicRouteProfile(
        route_kind="prepared_series",
        artifact_kind="torsion_snapshot_bundle",
        evidence_tags=["torsion_sensitivity", "oscillator_strength"],
        requested_descriptors=["state_ordering", "oscillator_strength", "torsion_sensitivity"],
        s1_mode="tda_atb_excdip",
    ),
    "run_torsion_brightness_coupling_scan": MicroscopicRouteProfile(
        route_kind="prepared_series",
        artifact_kind="torsion_brightness_coupling_bundle",
        evidence_tags=[
            "torsion_brightness_coupling",
            "torsion_sensitivity",
            "oscillator_strength",
        ],
        requested_descriptors=[
            "state_ordering",
            "oscillator_strength",
            "torsion_sensitivity",
            "torsion_brightness_coupling",
        ],
        s1_mode="tda_atb_excdip",
    ),
    "extract_ct_descriptors_from_bundle": MicroscopicRouteProfile(
        route_kind="parse_only",
        artifact_kind="ct_descriptor_bundle",
        evidence_tags=["ct_descriptor_surrogates", "artifact_inspection"],
        requested_descriptors=[
            "ct_localization_proxy",
            "dominant_transitions",
            "charge_distribution",
        ],
    ),
    "run_targeted_state_characterization": MicroscopicRouteProfile(
        route_kind="targeted",
        artifact_kind="targeted_state_characterization_bundle",
        evidence_tags=["state_characterization", "oscillator_strength"],
        requested_descriptors=["state_ordering", "dominant_transitions", "state_family_overlap"],
        s1_mode="tda_atb_excdip",
    ),
    "run_targeted_charge_analysis": MicroscopicRouteProfile(
        route_kind="targeted",
        artifact_kind="targeted_charge_analysis_bundle",
        evidence_tags=["charge_analysis", "mulliken_charges"],
        requested_descriptors=["mulliken_charges", "ground_state_energy"],
    ),
    "run_targeted_density_population_analysis": MicroscopicRouteProfile(
        route_kind="targeted",
        artifact_kind="targeted_density_population_bundle",
        evidence_tags=["density_population_analysis"],
        requested_descriptors=["density_matrix", "gross_orbital_populations", "mayer_bond_order"],
    ),
    "run_targeted_transition_dipole_analysis": MicroscopicRouteProfile(
        route_kind="targeted",
        artifact_kind="targeted_transition_dipole_bundle",
        evidence_tags=["transition_dipole", "oscillator_strength"],
        requested_descriptors=[
            "ground_to_excited_transition_dipoles",
            "excited_to_excited_transition_dipoles",
        ],
        s1_mode="tda_atb_excdip",
    ),
    "run_targeted_charge_redistribution_analysis": MicroscopicRouteProfile(
        route_kind="targeted",
        artifact_kind="targeted_charge_redistribution_bundle",
        evidence_tags=["charge_redistribution", "charge_analysis"],
        requested_descriptors=[
            "mulliken_charges",
            "charge_redistribution_summary",
            "geometry_rmsd_angstrom",
        ],
        s1_mode="tda_atb_excdip",
    ),
    "run_ris_state_characterization": MicroscopicRouteProfile(
        route_kind="targeted",
        artifact_kind="ris_state_characterization_bundle",
        evidence_tags=["ris_state_characterization", "state_ordering"],
        requested_descriptors=["state_ordering", "dominant_transitions", "mulliken_charges"],
        s1_mode="tda_ris",
    ),
    "run_targeted_localized_orbital_analysis": MicroscopicRouteProfile(
        route_kind="targeted",
        artifact_kind="targeted_localized_orbital_bundle",
        evidence_tags=["localized_orbital_analysis"],
        requested_descriptors=["localized_orbitals_pm", "molecular_orbital_files"],
        s1_mode="tda_b3lyp",
    ),
    "run_targeted_natural_orbital_analysis": MicroscopicRouteProfile(
        route_kind="targeted",
        artifact_kind="targeted_natural_orbital_bundle",
        evidence_tags=["natural_orbital_analysis"],
        requested_descriptors=["natural_orbitals_no", "molecular_orbital_files"],
        s1_mode="tda_b3lyp",
    ),
    "parse_snapshot_outputs": MicroscopicRouteProfile(
        route_kind="parse_only",
        artifact_kind="parsed_snapshot_bundle",
        evidence_tags=["snapshot_output_parse", "state_ordering"],
        requested_descriptors=["state_ordering", "oscillator_strength"],
    ),
    "extract_torsion_candidates_from_bundle": MicroscopicRouteProfile(
        route_kind="parse_only",
        artifact_kind="torsion_candidate_bundle",
        evidence_tags=["torsion_candidates", "torsion_sensitivity"],
        requested_descriptors=["rotatable_dihedral_candidates"],
    ),
    "extract_geometry_descriptors_from_bundle": MicroscopicRouteProfile(
        route_kind="parse_only",
        artifact_kind="geometry_descriptor_bundle",
        evidence_tags=["geometry_descriptors", "geometry_precondition"],
        requested_descriptors=["coordinate_geometry_descriptors"],
    ),
    "inspect_raw_artifact_bundle": MicroscopicRouteProfile(
        route_kind="parse_only",
        artifact_kind="raw_artifact_inventory",
        evidence_tags=["raw_artifact_inspection"],
        requested_descriptors=["raw_file_inventory", "extractable_observable_inventory"],
    ),
    "unsupported_excited_state_relaxation": MicroscopicRouteProfile(
        route_kind="unsupported",
        artifact_kind="unsupported_excited_state_relaxation",
        evidence_tags=["unsupported_excited_state_relaxation"],
        requested_descriptors=[],
    ),
}


class AmespMicroscopicRunner:
    def __init__(self, baseline_runner: AmespBaselineRunner | None = None) -> None:
        self.baseline_runner = baseline_runner or AmespBaselineRunner()

    def run(
        self,
        *,
        capability_name: str,
        source_artifact: ArtifactRecord,
        case_id: str,
        round_id: str,
        tool_args: dict[str, Any] | None = None,
    ) -> MicroscopicRouteResult:
        profile = _require_profile(capability_name)
        run_options = _microscopic_run_options(tool_args or {})
        runners = {
            "baseline": self._run_baseline,
            "bundle_discovery": self._discover_bundle,
            "parse_only": self._extract_from_bundle,
            "prepared_photophysics": self._run_prepared_photophysics,
            "prepared_discovery": self._discover_prepared,
            "prepared_series": self._run_prepared_series,
            "targeted": self._run_targeted,
        }
        if profile.route_kind == "unsupported":
            raise AmespMicroscopicError(
                "capability_unsupported",
                "Low-cost excited-state relaxation has not been validated for the Amesp adapter.",
            )
        return runners[profile.route_kind](
            capability_name=capability_name,
            profile=profile,
            source_artifact=source_artifact,
            case_id=case_id,
            round_id=round_id,
            run_options=run_options,
        )

    def _run_baseline(
        self,
        *,
        capability_name: str,
        profile: MicroscopicRouteProfile,
        source_artifact: ArtifactRecord,
        case_id: str,
        round_id: str,
        run_options: MicroscopicRunOptions,
    ) -> MicroscopicRouteResult:
        xyz_path = _artifact_xyz_path(source_artifact)
        symbols, coordinates = _read_xyz_or_raise(xyz_path)
        workdir = _microscopic_workdir(xyz_path, round_id, capability_name)
        workdir.mkdir(parents=True, exist_ok=True)
        label = _safe_label(f"{case_id}_{round_id}_{capability_name}")
        charge, multiplicity = _charge_and_multiplicity(source_artifact)

        steps: list[MicroscopicStepRecord] = []
        limitations: list[str] = []
        missing_deliverables: list[str] = []
        geometry_source = "s0_optimized_geometry"
        run_symbols = symbols
        run_coordinates = coordinates

        try:
            s0 = self._run_step(
                step_id="s0_optimization",
                method_label="aTB1-opt-force",
                label=f"{label}_s0",
                workdir=workdir,
                keywords=["aTB1", "opt", "force"],
                block_lines=[
                    ("opt", ["maxcyc 2000", "gediis off", "maxstep 0.3"]),
                    ("scf", ["maxcyc 2000", "vshift 500"]),
                ],
                charge=charge,
                multiplicity=multiplicity,
                symbols=symbols,
                coordinates=coordinates,
            )
            steps.append(_step_record(s0, "aTB1-opt-force"))
            run_symbols, run_coordinates = _final_geometry_or_raise(s0.aop_path)
        except AmespBaselineError as exc:
            limitations.append(f"S0 optimization unavailable: {exc.message}")
            missing_deliverables.append("S0 optimized geometry")
            geometry_source = "prepared_geometry_fixed_point_fallback"
            s0 = self._run_singlepoint(
                label=f"{label}_s0sp",
                workdir=workdir,
                charge=charge,
                multiplicity=multiplicity,
                symbols=symbols,
                coordinates=coordinates,
            )
            steps.append(_step_record(s0, "aTB1-force"))

        analysis_xyz = workdir / "analysis_geometry.xyz"
        _write_xyz(
            analysis_xyz,
            label=f"{label}_{geometry_source}",
            symbols=run_symbols,
            coordinates=run_coordinates,
        )
        s1: AmespStepResult | None = None
        failed_s1_paths: dict[str, str] = {}
        try:
            s1 = self._run_vertical(
                label=f"{label}_s1",
                workdir=workdir,
                charge=charge,
                multiplicity=multiplicity,
                symbols=run_symbols,
                coordinates=run_coordinates,
                mode=profile.s1_mode or "tda_atb_excdip",
                run_options=run_options,
            )
            steps.append(_step_record(s1, profile.s1_mode or "tda_atb_excdip"))
        except AmespBaselineError as exc:
            limitations.append(
                "S1 vertical excitation unavailable; retained only successful S0 "
                f"fixed-geometry observables. Amesp reported: {exc.message}"
            )
            missing_deliverables.extend(
                [
                    "S1 vertical excitation",
                    *[
                        f"S1-derived descriptor: {descriptor}"
                        for descriptor in profile.requested_descriptors
                        if descriptor
                        in {
                            "state_ordering",
                            "oscillator_strength",
                            "bright_state",
                            "dominant_transitions",
                            "state_family_overlap",
                            "ground_to_excited_transition_dipoles",
                            "excited_to_excited_transition_dipoles",
                        }
                    ],
                ]
            )
            failed_s1_paths = {
                f"failed_s1_{key}": value for key, value in exc.file_paths.items()
            }
        return self._write_bundle(
            bundle_id=f"{round_id}:{profile.artifact_kind}",
            capability_name=capability_name,
            profile=profile,
            status="partial" if missing_deliverables else "available",
            source_artifact_ids=[source_artifact.artifact_id],
            geometry_source=geometry_source,
            workdir=workdir,
            file_paths={
                "analysis_geometry_xyz": str(analysis_xyz),
                "source_xyz": str(xyz_path),
                "s0_aip": str(s0.aip_path),
                "s0_aop": str(s0.aop_path),
                **(
                    {"s1_aip": str(s1.aip_path), "s1_aop": str(s1.aop_path)}
                    if s1 is not None
                    else failed_s1_paths
                ),
            },
            steps=steps,
            s0_text=_read_text(s0.aop_path),
            s1_text=_read_text(s1.aop_path) if s1 is not None else "",
            missing_deliverables=missing_deliverables,
            limitations=limitations,
        )

    def _discover_prepared(
        self,
        *,
        capability_name: str,
        profile: MicroscopicRouteProfile,
        source_artifact: ArtifactRecord,
        case_id: str,
        round_id: str,
        run_options: MicroscopicRunOptions,
    ) -> MicroscopicRouteResult:
        del case_id, run_options
        xyz_path = _artifact_xyz_path(source_artifact)
        workdir = _microscopic_workdir(xyz_path, round_id, capability_name)
        workdir.mkdir(parents=True, exist_ok=True)
        builders = {
            "list_rotatable_dihedrals": _rotatable_dihedral_inventory,
            "list_available_conformers": _conformer_inventory,
        }
        parsed, missing, limitations = builders[capability_name](source_artifact)
        return self._write_bundle(
            bundle_id=f"{round_id}:{profile.artifact_kind}",
            capability_name=capability_name,
            profile=profile,
            status="partial" if missing else "available",
            source_artifact_ids=[source_artifact.artifact_id],
            geometry_source="prepared_structure",
            workdir=workdir,
            file_paths={"source_xyz": str(xyz_path)},
            steps=[],
            s0_text="",
            s1_text="",
            missing_deliverables=missing,
            limitations=limitations,
            parsed_overrides=parsed,
        )

    def _discover_bundle(
        self,
        *,
        capability_name: str,
        profile: MicroscopicRouteProfile,
        source_artifact: ArtifactRecord,
        case_id: str,
        round_id: str,
        run_options: MicroscopicRunOptions,
    ) -> MicroscopicRouteResult:
        del case_id, run_options
        source_bundle = _bundle_from_artifact(source_artifact)
        workdir = _artifact_workdir(source_artifact) / round_id / capability_name
        workdir.mkdir(parents=True, exist_ok=True)
        builders = {
            "list_artifact_bundles": _artifact_bundle_inventory,
            "list_artifact_bundle_members": _artifact_bundle_members,
        }
        parsed, missing, limitations = builders[capability_name](source_artifact, source_bundle)
        return self._write_bundle(
            bundle_id=f"{round_id}:{profile.artifact_kind}",
            capability_name=capability_name,
            profile=profile,
            status="partial" if missing else "available",
            source_artifact_ids=[source_artifact.artifact_id],
            geometry_source=source_bundle.geometry_source,
            workdir=workdir,
            file_paths=dict(source_bundle.file_paths),
            steps=[],
            s0_text="",
            s1_text="",
            missing_deliverables=missing,
            limitations=limitations,
            parsed_overrides=parsed,
        )

    def _run_prepared_series(
        self,
        *,
        capability_name: str,
        profile: MicroscopicRouteProfile,
        source_artifact: ArtifactRecord,
        case_id: str,
        round_id: str,
        run_options: MicroscopicRunOptions,
    ) -> MicroscopicRouteResult:
        xyz_path = _artifact_xyz_path(source_artifact)
        workdir = _microscopic_workdir(xyz_path, round_id, capability_name)
        workdir.mkdir(parents=True, exist_ok=True)
        label = _safe_label(f"{case_id}_{round_id}_{capability_name}")
        charge, multiplicity = _charge_and_multiplicity(source_artifact)
        member_builders = {
            "run_conformer_bundle": lambda: _conformer_members(
                source_artifact,
                workdir,
                max_members=run_options.max_members or 3,
                member_prefix="conformer",
            ),
            "run_conformer_state_probe": lambda: _conformer_members(
                source_artifact,
                workdir,
                max_members=run_options.max_members or 2,
                member_prefix="conformer_probe",
            ),
            "run_torsion_snapshots": lambda: _torsion_snapshot_members(
                source_artifact,
                workdir,
                max_members=run_options.max_members or 3,
            ),
            "run_torsion_brightness_coupling_scan": lambda: _torsion_snapshot_members(
                source_artifact,
                workdir,
                max_members=run_options.max_members or 3,
            ),
        }
        members, generation_limitations = member_builders[capability_name]()
        if len(members) < 2:
            raise AmespMicroscopicError(
                "precondition_missing",
                f"{capability_name} requires at least two generated geometry members.",
                details={"generated_member_count": len(members)},
            )
        return self._run_geometry_member_series(
            capability_name=capability_name,
            profile=profile,
            source_artifact=source_artifact,
            round_id=round_id,
            label=label,
            workdir=workdir,
            charge=charge,
            multiplicity=multiplicity,
            members=members,
            source_xyz_path=xyz_path,
            generation_limitations=generation_limitations,
            run_options=run_options,
        )

    def _run_geometry_member_series(
        self,
        *,
        capability_name: str,
        profile: MicroscopicRouteProfile,
        source_artifact: ArtifactRecord,
        round_id: str,
        label: str,
        workdir: Path,
        charge: int,
        multiplicity: int,
        members: list[PreparedGeometryMember],
        source_xyz_path: Path,
        generation_limitations: list[str],
        run_options: MicroscopicRunOptions,
    ) -> MicroscopicRouteResult:
        steps: list[MicroscopicStepRecord] = []
        file_paths: dict[str, str] = {"source_xyz": str(source_xyz_path)}
        route_records: list[dict[str, object]] = []
        member_artifacts: list[dict[str, object]] = []
        member_failures: list[dict[str, object]] = []
        first_s0_text = ""
        first_s1_text = ""
        primary_energy: float | None = None
        primary_xyz_path = members[0].xyz_path

        for member in members:
            member_workdir = workdir / member.label
            member_workdir.mkdir(parents=True, exist_ok=True)
            file_paths[f"{member.label}_xyz"] = str(member.xyz_path)
            try:
                s0 = self._run_singlepoint(
                    label=f"{label}_{member.label}_s0sp",
                    workdir=member_workdir,
                    charge=charge,
                    multiplicity=multiplicity,
                    symbols=member.symbols,
                    coordinates=member.coordinates,
                )
                s1 = self._run_vertical(
                    label=f"{label}_{member.label}_s1",
                    workdir=member_workdir,
                    charge=charge,
                    multiplicity=multiplicity,
                    symbols=member.symbols,
                    coordinates=member.coordinates,
                    mode=profile.s1_mode or "tda_atb_excdip",
                    run_options=run_options,
                )
            except AmespBaselineError as exc:
                member_failures.append(
                    {
                        "member_id": member.member_id,
                        "member_label": member.label,
                        "failure_code": exc.code,
                        "failure_message": exc.message,
                        "file_paths": exc.file_paths,
                    }
                )
                member_artifacts.append(
                    {
                        "member_id": member.member_id,
                        "member_label": member.label,
                        "member_status": "failed",
                        "prepared_xyz_path": str(member.xyz_path),
                        "failure_code": exc.code,
                        "failure_message": exc.message,
                    }
                )
                continue

            s0_text = _read_text(s0.aop_path)
            s1_text = _read_text(s1.aop_path)
            final_energy = _parse_final_energy(s0_text)
            excited_states = _parse_excited_states(
                s1_text,
                reference_energy_hartree=final_energy,
            )
            if not first_s0_text:
                first_s0_text = s0_text
                first_s1_text = s1_text
            if final_energy is not None and (
                primary_energy is None or final_energy < primary_energy
            ):
                primary_energy = final_energy
                primary_xyz_path = member.xyz_path
            steps.extend(
                [
                    _step_record(s0, f"aTB1-force:{member.label}"),
                    _step_record(s1, f"{profile.s1_mode or 'tda_atb_excdip'}:{member.label}"),
                ]
            )
            file_paths.update(_member_step_file_paths(member.label, s0=s0, s1=s1))
            route_records.append(
                {
                    "member_id": member.member_id,
                    "member_label": member.label,
                    **member.metadata,
                    "final_energy_hartree": final_energy,
                    "first_excitation_energy_ev": (
                        excited_states[0].get("excitation_energy_ev")
                        if excited_states
                        else None
                    ),
                    "first_oscillator_strength": (
                        excited_states[0].get("oscillator_strength") if excited_states else None
                    ),
                    "state_count": len(excited_states),
                }
            )
            member_artifacts.append(
                {
                    "member_id": member.member_id,
                    "member_label": member.label,
                    "member_status": "success",
                    "prepared_xyz_path": str(member.xyz_path),
                    "s0_aip_path": str(s0.aip_path),
                    "s0_aop_path": str(s0.aop_path),
                    "s1_aip_path": str(s1.aip_path),
                    "s1_aop_path": str(s1.aop_path),
                }
            )

        if len(route_records) < 2:
            raise AmespMicroscopicError(
                "partial_member_series_insufficient_successes",
                f"{capability_name} produced fewer than two successful members.",
                details={
                    "successful_member_count": len(route_records),
                    "attempted_member_count": len(members),
                    "member_failures": member_failures,
                },
            )

        descriptor = (
            "torsion_sensitivity"
            if capability_name
            in {"run_torsion_snapshots", "run_torsion_brightness_coupling_scan"}
            else "conformer_sensitivity"
        )
        available_descriptors = sorted(
            {
                "state_ordering",
                "oscillator_strength",
                descriptor,
                *(
                    {"torsion_brightness_coupling"}
                    if capability_name == "run_torsion_brightness_coupling_scan"
                    else set()
                ),
            }
            & set(profile.requested_descriptors)
        )
        missing_descriptors = [
            descriptor_name
            for descriptor_name in profile.requested_descriptors
            if descriptor_name not in available_descriptors
        ]
        file_paths["analysis_geometry_xyz"] = str(primary_xyz_path)
        limitations = list(generation_limitations)
        if member_failures:
            limitations.append(
                "One or more generated geometry members failed, but at least two "
                "members completed and are retained in the typed bundle."
            )
        return self._write_bundle(
            bundle_id=f"{round_id}:{profile.artifact_kind}",
            capability_name=capability_name,
            profile=profile,
            status="partial" if member_failures else "available",
            source_artifact_ids=[source_artifact.artifact_id],
            geometry_source=f"generated_member_series:{capability_name}",
            workdir=workdir,
            file_paths=file_paths,
            steps=steps,
            s0_text=first_s0_text,
            s1_text=first_s1_text,
            missing_deliverables=[],
            limitations=limitations,
            parsed_overrides={
                "requested_descriptors": profile.requested_descriptors,
                "available_descriptors": available_descriptors,
                "missing_descriptors": missing_descriptors,
                "route_records": route_records,
                "member_artifacts": member_artifacts,
                "route_summary": _member_series_summary(route_records, member_failures),
                "member_failure_count": len(member_failures),
            },
        )

    def _extract_from_bundle(
        self,
        *,
        capability_name: str,
        profile: MicroscopicRouteProfile,
        source_artifact: ArtifactRecord,
        case_id: str,
        round_id: str,
        run_options: MicroscopicRunOptions,
    ) -> MicroscopicRouteResult:
        del case_id, run_options
        source_bundle = _bundle_from_artifact(source_artifact)
        workdir = _artifact_workdir(source_artifact) / round_id / capability_name
        workdir.mkdir(parents=True, exist_ok=True)
        builders = {
            "extract_ct_descriptors_from_bundle": _ct_descriptor_payload,
            "parse_snapshot_outputs": _snapshot_parse_payload,
            "extract_torsion_candidates_from_bundle": _torsion_candidate_payload,
            "extract_geometry_descriptors_from_bundle": _geometry_descriptor_payload,
            "inspect_raw_artifact_bundle": _raw_artifact_payload,
        }
        parsed, missing, limitations = builders[capability_name](source_artifact, source_bundle)
        return self._write_bundle(
            bundle_id=f"{round_id}:{profile.artifact_kind}",
            capability_name=capability_name,
            profile=profile,
            status="partial" if missing else "available",
            source_artifact_ids=[source_artifact.artifact_id],
            geometry_source=source_bundle.geometry_source,
            workdir=workdir,
            file_paths=dict(source_bundle.file_paths),
            steps=[],
            s0_text="",
            s1_text="",
            missing_deliverables=missing,
            limitations=limitations,
            parsed_overrides=parsed,
        )

    def _run_prepared_photophysics(
        self,
        *,
        capability_name: str,
        profile: MicroscopicRouteProfile,
        source_artifact: ArtifactRecord,
        case_id: str,
        round_id: str,
        run_options: MicroscopicRunOptions,
    ) -> MicroscopicRouteResult:
        xyz_path = _artifact_xyz_path(source_artifact)
        symbols, coordinates = _read_xyz_or_raise(xyz_path)
        workdir = _microscopic_workdir(xyz_path, round_id, capability_name)
        workdir.mkdir(parents=True, exist_ok=True)
        label = _safe_label(f"{case_id}_{round_id}_{capability_name}")
        charge, multiplicity = _charge_and_multiplicity(source_artifact)
        handlers = {
            "run_frontier_orbital_partition": self._run_frontier_orbital_partition,
            "run_charge_population_panel": self._run_charge_population_panel,
            "run_solvation_polarity_proxy": self._run_solvation_polarity_proxy,
        }
        return handlers[capability_name](
            capability_name=capability_name,
            profile=profile,
            source_artifact=source_artifact,
            round_id=round_id,
            label=label,
            workdir=workdir,
            charge=charge,
            multiplicity=multiplicity,
            symbols=symbols,
            coordinates=coordinates,
            source_xyz_path=xyz_path,
            run_options=run_options,
        )

    def _run_frontier_orbital_partition(
        self,
        *,
        capability_name: str,
        profile: MicroscopicRouteProfile,
        source_artifact: ArtifactRecord,
        round_id: str,
        label: str,
        workdir: Path,
        charge: int,
        multiplicity: int,
        symbols: list[str],
        coordinates: list[list[float]],
        source_xyz_path: Path,
        run_options: MicroscopicRunOptions,
    ) -> MicroscopicRouteResult:
        del run_options
        step = self._run_step(
            step_id="frontier_orbital_singlepoint",
            method_label="aTB1-force-out2",
            label=f"{label}_out2",
            workdir=workdir,
            keywords=["aTB1", "force"],
            block_lines=[("ope", ["out 2"]), ("scf", ["maxcyc 500", "vshift 500"])],
            charge=charge,
            multiplicity=multiplicity,
            symbols=symbols,
            coordinates=coordinates,
        )
        s0_text = _read_text(step.aop_path)
        parsed = _frontier_orbital_partition_payload(s0_text, source_artifact)
        missing = [
            descriptor
            for descriptor in profile.requested_descriptors
            if descriptor not in parsed["available_descriptors"]
        ]
        file_paths = {
            "source_xyz": str(source_xyz_path),
            "s0_aip": str(step.aip_path),
            "s0_aop": str(step.aop_path),
            **_existing_mo_file_paths(s0=step, s1=None),
        }
        return self._write_bundle(
            bundle_id=f"{round_id}:{profile.artifact_kind}",
            capability_name=capability_name,
            profile=profile,
            status="partial" if missing else "available",
            source_artifact_ids=[source_artifact.artifact_id],
            geometry_source="prepared_geometry_fixed_point",
            workdir=workdir,
            file_paths=file_paths,
            steps=[_step_record(step, "aTB1-force-out2")],
            s0_text=s0_text,
            s1_text="",
            missing_deliverables=missing,
            limitations=[
                "Frontier partition is a fixed-geometry low-cost orbital proxy."
            ],
            parsed_overrides=parsed,
        )

    def _run_charge_population_panel(
        self,
        *,
        capability_name: str,
        profile: MicroscopicRouteProfile,
        source_artifact: ArtifactRecord,
        round_id: str,
        label: str,
        workdir: Path,
        charge: int,
        multiplicity: int,
        symbols: list[str],
        coordinates: list[list[float]],
        source_xyz_path: Path,
        run_options: MicroscopicRunOptions,
    ) -> MicroscopicRouteResult:
        schemes = run_options.charge_schemes or ["mulliken", "lowdin", "hirshfeld", "cm5"]
        steps: list[MicroscopicStepRecord] = []
        scheme_records: list[dict[str, object]] = []
        failures: list[dict[str, object]] = []
        file_paths: dict[str, str] = {"source_xyz": str(source_xyz_path)}
        first_text = ""
        for scheme in schemes:
            try:
                step = self._run_step(
                    step_id=f"charge_population_{scheme}",
                    method_label=f"aTB1-force-charge-{scheme}",
                    label=f"{label}_{scheme}",
                    workdir=workdir,
                    keywords=["aTB1", "force"],
                    block_lines=[
                        ("ope", [f"charge {scheme}", "out 2"]),
                        ("scf", ["maxcyc 500", "vshift 500"]),
                    ],
                    charge=charge,
                    multiplicity=multiplicity,
                    symbols=symbols,
                    coordinates=coordinates,
                )
            except AmespBaselineError as exc:
                failures.append(
                    {
                        "scheme": scheme,
                        "failure_code": exc.code,
                        "failure_message": exc.message,
                        "file_paths": exc.file_paths,
                    }
                )
                continue
            text = _read_text(step.aop_path)
            first_text = first_text or text
            steps.append(_step_record(step, f"aTB1-force-charge-{scheme}"))
            file_paths[f"{scheme}_aip"] = str(step.aip_path)
            file_paths[f"{scheme}_aop"] = str(step.aop_path)
            file_paths.update(
                {
                    f"{scheme}_{key}": path
                    for key, path in _existing_mo_file_paths(s0=step, s1=None).items()
                }
            )
            scheme_records.append(_charge_scheme_record(scheme, step.aop_path, text))
        available = [f"{record['scheme']}_charges" for record in scheme_records]
        missing = [
            descriptor
            for descriptor in profile.requested_descriptors
            if descriptor not in available
        ]
        return self._write_bundle(
            bundle_id=f"{round_id}:{profile.artifact_kind}",
            capability_name=capability_name,
            profile=profile,
            status="partial" if missing or failures else "available",
            source_artifact_ids=[source_artifact.artifact_id],
            geometry_source="prepared_geometry_fixed_point",
            workdir=workdir,
            file_paths=file_paths,
            steps=steps,
            s0_text=first_text,
            s1_text="",
            missing_deliverables=missing,
            limitations=[
                *(
                    ["One or more charge schemes failed; successful schemes are retained."]
                    if failures
                    else []
                )
            ],
            parsed_overrides={
                "requested_descriptors": profile.requested_descriptors,
                "available_descriptors": sorted(available),
                "missing_descriptors": missing,
                "charge_schemes": scheme_records,
                "scheme_failures": failures,
            },
        )

    def _run_solvation_polarity_proxy(
        self,
        *,
        capability_name: str,
        profile: MicroscopicRouteProfile,
        source_artifact: ArtifactRecord,
        round_id: str,
        label: str,
        workdir: Path,
        charge: int,
        multiplicity: int,
        symbols: list[str],
        coordinates: list[list[float]],
        source_xyz_path: Path,
        run_options: MicroscopicRunOptions,
    ) -> MicroscopicRouteResult:
        solvents = run_options.solvents or ["water", "toluene"]
        steps: list[MicroscopicStepRecord] = []
        solvent_records: list[dict[str, object]] = []
        failures: list[dict[str, object]] = []
        file_paths: dict[str, str] = {"source_xyz": str(source_xyz_path)}
        first_text = ""
        for solvent in solvents:
            nstates = run_options.s1_nstates or self.baseline_runner.s1_nstates
            td_tout = run_options.td_tout or self.baseline_runner.td_tout
            try:
                step = self._run_step(
                    step_id=f"solvation_polarity_{solvent}",
                    method_label=f"aTB1-tda-gbsa-{solvent}",
                    label=f"{label}_{solvent}",
                    workdir=workdir,
                    keywords=["aTB1", "tda", "gbsa"],
                    block_lines=[
                        ("solvent", [f"solvent {solvent}"]),
                        ("ope", ["out 1"]),
                        ("atb", ["excdip on"]),
                        ("scf", ["maxcyc 2000", "vshift 500"]),
                        (
                            "posthf",
                            [
                                f"nstates {nstates}",
                                f"tout {td_tout}",
                            ],
                        ),
                    ],
                    charge=charge,
                    multiplicity=multiplicity,
                    symbols=symbols,
                    coordinates=coordinates,
                )
            except AmespBaselineError as exc:
                failures.append(
                    {
                        "solvent": solvent,
                        "failure_code": exc.code,
                        "failure_message": exc.message,
                        "file_paths": exc.file_paths,
                    }
                )
                continue
            text = _read_text(step.aop_path)
            first_text = first_text or text
            final_energy = _parse_final_energy(text)
            states = _parse_excited_states(text, reference_energy_hartree=final_energy)
            steps.append(_step_record(step, f"aTB1-tda-gbsa-{solvent}"))
            file_paths[f"{solvent}_aip"] = str(step.aip_path)
            file_paths[f"{solvent}_aop"] = str(step.aop_path)
            solvent_records.append(
                {
                    "solvent": solvent,
                    "final_energy_hartree": final_energy,
                    "state_count": len(states),
                    "first_state": states[0] if states else None,
                    "bright_state": _bright_state(states),
                }
            )
        parsed = _solvation_proxy_payload(profile, solvent_records, failures)
        missing = list(parsed["missing_descriptors"])
        return self._write_bundle(
            bundle_id=f"{round_id}:{profile.artifact_kind}",
            capability_name=capability_name,
            profile=profile,
            status="partial" if missing or failures else "available",
            source_artifact_ids=[source_artifact.artifact_id],
            geometry_source="prepared_geometry_fixed_point",
            workdir=workdir,
            file_paths=file_paths,
            steps=steps,
            s0_text=first_text,
            s1_text=first_text,
            missing_deliverables=missing,
            limitations=[
                "Solvation polarity is a fixed-geometry implicit-solvent proxy, "
                "not an experimental solvatochromism measurement.",
                *(
                    ["One or more solvent calculations failed; successful solvents are retained."]
                    if failures
                    else []
                ),
            ],
            parsed_overrides=parsed,
        )

    def _run_targeted(
        self,
        *,
        capability_name: str,
        profile: MicroscopicRouteProfile,
        source_artifact: ArtifactRecord,
        case_id: str,
        round_id: str,
        run_options: MicroscopicRunOptions,
    ) -> MicroscopicRouteResult:
        source_bundle = _bundle_from_artifact(source_artifact)
        xyz_path = _bundle_geometry_path(source_bundle)
        symbols, coordinates = _read_xyz_or_raise(xyz_path)
        workdir = _artifact_workdir(source_artifact) / round_id / capability_name
        workdir.mkdir(parents=True, exist_ok=True)
        label = _safe_label(f"{case_id}_{round_id}_{capability_name}")
        charge, multiplicity = _charge_and_multiplicity(source_artifact)
        orbital_spec = _orbital_analysis_spec(capability_name)
        if orbital_spec is None:
            s0 = self._run_singlepoint(
                label=f"{label}_s0sp",
                workdir=workdir,
                charge=charge,
                multiplicity=multiplicity,
                symbols=symbols,
                coordinates=coordinates,
            )
            s0_method_label = "aTB1-force"
        else:
            s0 = self._run_orbital_singlepoint(
                label=f"{label}_s0sp",
                workdir=workdir,
                charge=charge,
                multiplicity=multiplicity,
                symbols=symbols,
                coordinates=coordinates,
                method_lines=list(orbital_spec["method_lines"]),
                ope_lines=list(orbital_spec["ope_lines"]),
            )
            s0_method_label = str(orbital_spec["method_label"])
        steps = [_step_record(s0, s0_method_label)]
        s1 = self._optional_vertical(
            profile,
            label=f"{label}_s1",
            workdir=workdir,
            charge=charge,
            multiplicity=multiplicity,
            symbols=symbols,
            coordinates=coordinates,
            run_options=run_options,
        )
        if s1 is not None:
            steps.append(_step_record(s1, profile.s1_mode or "tda_atb_excdip"))
        s0_text = _read_text(s0.aop_path)
        s1_text = _read_text(s1.aop_path) if s1 else ""
        file_paths = {
            "analysis_geometry_xyz": str(xyz_path),
            "s0_aip": str(s0.aip_path),
            "s0_aop": str(s0.aop_path),
            **({"s1_aip": str(s1.aip_path), "s1_aop": str(s1.aop_path)} if s1 else {}),
            **_existing_mo_file_paths(s0=s0, s1=s1),
        }
        parsed_overrides = _orbital_parsed_observables(
            capability_name=capability_name,
            profile=profile,
            s0_text=s0_text,
            s1_text=s1_text,
            file_paths=file_paths,
        )
        missing_deliverables = list(parsed_overrides["missing_descriptors"])
        return self._write_bundle(
            bundle_id=f"{round_id}:{profile.artifact_kind}",
            capability_name=capability_name,
            profile=profile,
            status="partial" if missing_deliverables else "available",
            source_artifact_ids=[source_artifact.artifact_id],
            geometry_source=f"fixed_geometry_from:{source_bundle.bundle_id}",
            workdir=workdir,
            file_paths=file_paths,
            steps=steps,
            s0_text=s0_text,
            s1_text=s1_text,
            missing_deliverables=missing_deliverables,
            limitations=[
                "Targeted follow-up is fixed-geometry and bounded to one "
                "representative bundle geometry."
            ],
            parsed_overrides=parsed_overrides,
        )

    def _optional_vertical(
        self,
        profile: MicroscopicRouteProfile,
        *,
        label: str,
        workdir: Path,
        charge: int,
        multiplicity: int,
        symbols: list[str],
        coordinates: list[list[float]],
        run_options: MicroscopicRunOptions,
    ) -> AmespStepResult | None:
        if profile.s1_mode is None:
            return None
        try:
            return self._run_vertical(
                label=label,
                workdir=workdir,
                charge=charge,
                multiplicity=multiplicity,
                symbols=symbols,
                coordinates=coordinates,
                mode=profile.s1_mode,
                run_options=run_options,
            )
        except AmespBaselineError:
            if profile.s1_mode != "tda_ris":
                raise
            return self._run_vertical(
                label=f"{label}_atb_fallback",
                workdir=workdir,
                charge=charge,
                multiplicity=multiplicity,
                symbols=symbols,
                coordinates=coordinates,
                mode="tda_atb_excdip",
                run_options=run_options,
            )

    def _run_singlepoint(
        self,
        *,
        label: str,
        workdir: Path,
        charge: int,
        multiplicity: int,
        symbols: list[str],
        coordinates: list[list[float]],
    ) -> AmespStepResult:
        return self._run_step(
            step_id="s0_singlepoint",
            method_label="aTB1-force",
            label=label,
            workdir=workdir,
            keywords=["aTB1", "force"],
            block_lines=[("scf", ["maxcyc 2000", "vshift 500"])],
            charge=charge,
            multiplicity=multiplicity,
            symbols=symbols,
            coordinates=coordinates,
        )

    def _run_orbital_singlepoint(
        self,
        *,
        label: str,
        workdir: Path,
        charge: int,
        multiplicity: int,
        symbols: list[str],
        coordinates: list[list[float]],
        method_lines: list[str],
        ope_lines: list[str],
    ) -> AmespStepResult:
        blocks = [
            ("method", method_lines),
            ("ope", ope_lines),
            ("scf", ["maxcyc 2000", "vshift 500"]),
        ]
        return self._run_step(
            step_id="s0_orbital_singlepoint",
            method_label="sp-b3lyp-orbital",
            label=label,
            workdir=workdir,
            keywords=["b3lyp", "sto-3g"],
            block_lines=blocks,
            charge=charge,
            multiplicity=multiplicity,
            symbols=symbols,
            coordinates=coordinates,
        )

    def _run_vertical(
        self,
        *,
        label: str,
        workdir: Path,
        charge: int,
        multiplicity: int,
        symbols: list[str],
        coordinates: list[list[float]],
        mode: str,
        run_options: MicroscopicRunOptions | None = None,
    ) -> AmespStepResult:
        nstates = (
            run_options.s1_nstates
            if run_options is not None and run_options.s1_nstates is not None
            else self.baseline_runner.s1_nstates
        )
        td_tout = (
            run_options.td_tout
            if run_options is not None and run_options.td_tout is not None
            else self.baseline_runner.td_tout
        )
        keywords_by_mode = {
            "tda_atb_excdip": ["aTB1", "tda"],
            "tda_ris": ["b3lyp", "sto-3g", "tda-ris"],
            "tda_b3lyp": ["b3lyp", "sto-3g", "tda"],
        }
        blocks_by_mode = {
            "tda_atb_excdip": [
                ("ope", ["out 1"]),
                ("atb", ["excdip on"]),
                ("scf", ["maxcyc 2000", "vshift 500"]),
                (
                    "posthf",
                    [
                        f"nstates {nstates}",
                        f"tout {td_tout}",
                    ],
                ),
            ],
            "tda_ris": [
                ("ope", ["out 1"]),
                ("scf", ["maxcyc 2000", "vshift 500"]),
                (
                    "posthf",
                    [
                        f"nstates {nstates}",
                        f"tout {td_tout}",
                    ],
                ),
            ],
            "tda_b3lyp": [
                ("ope", ["out 1"]),
                ("scf", ["maxcyc 2000", "vshift 500"]),
                (
                    "posthf",
                    [
                        f"nstates {nstates}",
                        f"tout {td_tout}",
                    ],
                ),
            ],
        }
        return self._run_step(
            step_id="s1_vertical_excitation",
            method_label=mode,
            label=label,
            workdir=workdir,
            keywords=keywords_by_mode[mode],
            block_lines=blocks_by_mode[mode],
            charge=charge,
            multiplicity=multiplicity,
            symbols=symbols,
            coordinates=coordinates,
        )

    def _run_step(
        self,
        *,
        step_id: str,
        method_label: str,
        label: str,
        workdir: Path,
        keywords: list[str],
        block_lines: list[tuple[str, list[str]]],
        charge: int,
        multiplicity: int,
        symbols: list[str],
        coordinates: list[list[float]],
    ) -> AmespStepResult:
        del method_label
        return self.baseline_runner._run_step(
            step_id=step_id,
            label=label,
            workdir=workdir,
            keywords=keywords,
            block_lines=block_lines,
            charge=charge,
            multiplicity=multiplicity,
            symbols=symbols,
            coordinates=coordinates,
        )

    def _write_bundle(
        self,
        *,
        bundle_id: str,
        capability_name: str,
        profile: MicroscopicRouteProfile,
        status: Literal["available", "partial"],
        source_artifact_ids: list[str],
        geometry_source: str,
        workdir: Path,
        file_paths: dict[str, str],
        steps: list[MicroscopicStepRecord],
        s0_text: str,
        s1_text: str,
        missing_deliverables: list[str],
        limitations: list[str],
        parsed_overrides: dict[str, Any] | None = None,
    ) -> MicroscopicRouteResult:
        excited_states = _parse_excited_states(
            s1_text,
            reference_energy_hartree=_parse_final_energy(s0_text),
        )
        parsed = parsed_overrides or _parsed_observables(
            profile=profile,
            s0_text=s0_text,
            s1_text=s1_text,
            excited_states=excited_states,
        )
        manifest_path = workdir / "bundle_manifest.json"
        bundle = MicroscopicArtifactBundle(
            bundle_id=bundle_id,
            capability_name=capability_name,
            status=status,
            source_artifact_ids=source_artifact_ids,
            geometry_source=geometry_source,
            file_paths={**file_paths, "manifest": str(manifest_path), "workdir": str(workdir)},
            step_records=steps,
            excited_states=excited_states,
            parsed_observables=parsed,
            missing_deliverables=missing_deliverables,
            limitations=limitations,
        )
        manifest_path.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return MicroscopicRouteResult(bundle=bundle, manifest_path=manifest_path)


def _require_profile(capability_name: str) -> MicroscopicRouteProfile:
    profile = MICROSCOPIC_ROUTE_PROFILES.get(capability_name)
    if profile is None:
        raise AmespMicroscopicError(
            "capability_unsupported",
            f"Microscopic capability is not implemented in MechCAL: {capability_name}",
        )
    return profile


def _microscopic_run_options(tool_args: dict[str, Any]) -> MicroscopicRunOptions:
    return MicroscopicRunOptions(
        max_members=_bounded_int(tool_args.get("max_members"), minimum=2, maximum=6),
        s1_nstates=_bounded_int(tool_args.get("s1_nstates"), minimum=1, maximum=10),
        td_tout=_bounded_int(tool_args.get("td_tout"), minimum=1, maximum=10),
        solvents=_bounded_choice_list(
            tool_args.get("solvents"),
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
            default=["water", "toluene"],
            maximum=4,
        ),
        charge_schemes=_bounded_choice_list(
            tool_args.get("charge_schemes"),
            allowed={"mulliken", "lowdin", "hirshfeld", "cm5"},
            default=["mulliken", "lowdin", "hirshfeld", "cm5"],
            maximum=4,
        ),
    )


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
    default: list[str],
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
    return selected or default[:maximum]


def _artifact_xyz_path(artifact: ArtifactRecord) -> Path:
    raw_path = artifact.file_paths.get("xyz") or artifact.file_paths.get("analysis_geometry_xyz")
    if not raw_path:
        raise AmespMicroscopicError(
            "precondition_missing",
            "Source artifact does not expose an xyz geometry path.",
            details={"artifact_id": artifact.artifact_id},
        )
    return Path(raw_path)


def _bundle_geometry_path(bundle: MicroscopicArtifactBundle) -> Path:
    raw_path = bundle.file_paths.get("analysis_geometry_xyz") or bundle.file_paths.get("source_xyz")
    if not raw_path:
        raise AmespMicroscopicError(
            "precondition_missing",
            "Microscopic bundle does not expose a reusable analysis geometry.",
            details={"bundle_id": bundle.bundle_id},
        )
    return Path(raw_path)


def _read_xyz_or_raise(xyz_path: Path) -> tuple[list[str], list[list[float]]]:
    if not xyz_path.exists():
        raise AmespMicroscopicError(
            "precondition_missing",
            f"Required xyz file is missing: {xyz_path}",
        )
    symbols, coordinates = _read_xyz_symbols_and_coordinates(xyz_path)
    if not symbols:
        raise AmespMicroscopicError(
            "precondition_missing",
            f"Required xyz file is not parseable: {xyz_path}",
        )
    return symbols, coordinates


def _final_geometry_or_raise(aop_path: Path) -> tuple[list[str], list[list[float]]]:
    symbols, coordinates = _parse_final_geometry(_read_text(aop_path))
    if not symbols:
        raise AmespBaselineError(
            "parse_failed",
            "Amesp S0 output did not expose a parseable final geometry.",
            file_paths={"aop": str(aop_path)},
        )
    return symbols, coordinates


def _microscopic_workdir(xyz_path: Path, round_id: str, capability_name: str) -> Path:
    return xyz_path.parents[2] / "artifacts" / "microscopic" / round_id / capability_name


def _artifact_workdir(artifact: ArtifactRecord) -> Path:
    manifest_path = artifact.file_paths.get("manifest")
    if manifest_path:
        return Path(manifest_path).parents[2]
    workdir = artifact.file_paths.get("workdir")
    if workdir:
        return Path(workdir).parents[1]
    raise AmespMicroscopicError(
        "precondition_missing",
        "Microscopic artifact bundle does not expose a manifest/workdir path.",
        details={"artifact_id": artifact.artifact_id},
    )


def _charge_and_multiplicity(artifact: ArtifactRecord) -> tuple[int, int]:
    metadata = artifact.metadata or {}
    return int(metadata.get("charge", 0)), int(metadata.get("multiplicity", 1))


def _bundle_from_artifact(artifact: ArtifactRecord) -> MicroscopicArtifactBundle:
    bundle_payload = artifact.metadata.get("bundle") if artifact.metadata else None
    if bundle_payload is None:
        manifest_path = artifact.file_paths.get("manifest")
        if manifest_path:
            bundle_payload = Path(manifest_path).read_text(encoding="utf-8")
    try:
        if isinstance(bundle_payload, str):
            return MicroscopicArtifactBundle.model_validate_json(bundle_payload)
        return MicroscopicArtifactBundle.model_validate(bundle_payload)
    except Exception as exc:
        raise AmespMicroscopicError(
            "precondition_missing",
            "Microscopic artifact metadata does not contain a valid typed bundle.",
            details={"artifact_id": artifact.artifact_id, "error": str(exc)},
        ) from exc


def _step_record(step: AmespStepResult, method_label: str) -> MicroscopicStepRecord:
    return MicroscopicStepRecord(
        step_id=step.step_id,
        method_label=method_label,
        aip_path=str(step.aip_path),
        aop_path=str(step.aop_path),
        stdout_path=str(step.stdout_path),
        stderr_path=str(step.stderr_path),
        exit_code=step.exit_code,
        terminated_normally=step.terminated_normally,
        elapsed_seconds=step.elapsed_seconds,
    )


def _member_step_file_paths(
    member_label: str,
    *,
    s0: AmespStepResult,
    s1: AmespStepResult,
) -> dict[str, str]:
    paths = {
        f"{member_label}_s0_aip": str(s0.aip_path),
        f"{member_label}_s0_aop": str(s0.aop_path),
        f"{member_label}_s0_stdout": str(s0.stdout_path),
        f"{member_label}_s0_stderr": str(s0.stderr_path),
        f"{member_label}_s1_aip": str(s1.aip_path),
        f"{member_label}_s1_aop": str(s1.aop_path),
        f"{member_label}_s1_stdout": str(s1.stdout_path),
        f"{member_label}_s1_stderr": str(s1.stderr_path),
    }
    for label, path in _existing_mo_file_paths(s0=s0, s1=s1).items():
        paths[f"{member_label}_{label}"] = path
    return paths


def _existing_mo_file_paths(
    *,
    s0: AmespStepResult,
    s1: AmespStepResult | None,
) -> dict[str, str]:
    steps = {"s0": s0, **({"s1": s1} if s1 is not None else {})}
    return {
        f"{label}_mo": str(step.aip_path.with_suffix(".mo"))
        for label, step in steps.items()
        if step.aip_path.with_suffix(".mo").exists()
    }


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _parsed_observables(
    *,
    profile: MicroscopicRouteProfile,
    s0_text: str,
    s1_text: str,
    excited_states: list[dict[str, object]],
) -> dict[str, object]:
    available = _available_descriptors(profile, s0_text, s1_text, excited_states)
    return {
        "requested_descriptors": profile.requested_descriptors,
        "available_descriptors": available,
        "missing_descriptors": [
            descriptor
            for descriptor in profile.requested_descriptors
            if descriptor not in available
        ],
        "final_energy_hartree": _parse_final_energy(s0_text),
        "mulliken_charge_count": len(_parse_mulliken_charges(s0_text)),
        "state_count": len(excited_states),
        "bright_state": _bright_state(excited_states),
        "section_presence": {
            "ground_to_excited_transition_dipoles": _GROUND_TO_EXCITED_DIPOLE
            in s1_text,
            "excited_to_excited_transition_dipoles": _EXCITED_TO_EXCITED_DIPOLE
            in s1_text,
            "gross_orbital_populations": "gross orbital population" in s0_text.lower(),
            "mayer_bond_order": "mayer" in s0_text.lower(),
        },
    }


def _available_descriptors(
    profile: MicroscopicRouteProfile,
    s0_text: str,
    s1_text: str,
    excited_states: list[dict[str, object]],
) -> list[str]:
    candidates = {
        "ground_state_energy": _parse_final_energy(s0_text) is not None,
        "mulliken_charges": bool(_parse_mulliken_charges(s0_text)),
        "state_ordering": bool(excited_states),
        "oscillator_strength": any(
            state.get("oscillator_strength") is not None for state in excited_states
        ),
        "bright_state": bool(_bright_state(excited_states)),
        "dominant_transitions": bool(excited_states),
        "state_family_overlap": bool(excited_states),
        "ground_to_excited_transition_dipoles": _GROUND_TO_EXCITED_DIPOLE in s1_text,
        "excited_to_excited_transition_dipoles": _EXCITED_TO_EXCITED_DIPOLE
        in s1_text,
        "gross_orbital_populations": "gross orbital population" in s0_text.lower(),
        "density_matrix": "density matrix" in s0_text.lower(),
        "mayer_bond_order": "mayer" in s0_text.lower(),
        "molecular_orbital_files": False,
        "localized_orbitals_pm": False,
        "natural_orbitals_no": False,
    }
    return sorted(
        descriptor
        for descriptor in profile.requested_descriptors
        if candidates.get(descriptor, False)
    )


def _available_ct_descriptors(bundle: MicroscopicArtifactBundle) -> list[str]:
    available = set(bundle.parsed_observables.get("available_descriptors") or [])
    if bundle.excited_states:
        available.update({"state_ordering", "oscillator_strength", "dominant_transitions"})
    if "mulliken_charges" in available:
        available.add("charge_distribution")
    return sorted(available)


def _missing_descriptors(profile: MicroscopicRouteProfile, s0_text: str, s1_text: str) -> list[str]:
    parsed = _parsed_observables(
        profile=profile,
        s0_text=s0_text,
        s1_text=s1_text,
        excited_states=_parse_excited_states(
            s1_text,
            reference_energy_hartree=_parse_final_energy(s0_text),
        ),
    )
    return list(parsed["missing_descriptors"])


def _orbital_analysis_spec(capability_name: str) -> dict[str, object] | None:
    specs: dict[str, dict[str, object]] = {
        "run_targeted_localized_orbital_analysis": {
            "method_label": "sp-b3lyp-lmo-pm",
            "method_lines": ["lmo pm"],
            "ope_lines": ["mofile on"],
        },
        "run_targeted_natural_orbital_analysis": {
            "method_label": "sp-b3lyp-natorb-no",
            "method_lines": ["natorb no"],
            "ope_lines": ["mofile on"],
        },
    }
    return specs.get(capability_name)


def _orbital_parsed_observables(
    *,
    capability_name: str,
    profile: MicroscopicRouteProfile,
    s0_text: str,
    s1_text: str,
    file_paths: dict[str, str],
) -> dict[str, object]:
    excited_states = _parse_excited_states(
        s1_text,
        reference_energy_hartree=_parse_final_energy(s0_text),
    )
    parsed = _parsed_observables(
        profile=profile,
        s0_text=s0_text,
        s1_text=s1_text,
        excited_states=excited_states,
    )
    requested = set(profile.requested_descriptors)
    available = set(parsed["available_descriptors"])
    has_mo_file = any(
        key.endswith("_mo") and Path(path).exists()
        for key, path in file_paths.items()
    )
    orbital_descriptor = {
        "run_targeted_localized_orbital_analysis": "localized_orbitals_pm",
        "run_targeted_natural_orbital_analysis": "natural_orbitals_no",
    }.get(capability_name)
    if has_mo_file and orbital_descriptor in requested:
        available.update({"molecular_orbital_files", orbital_descriptor})
    parsed["available_descriptors"] = sorted(available)
    parsed["missing_descriptors"] = [
        descriptor for descriptor in profile.requested_descriptors if descriptor not in available
    ]
    if orbital_descriptor is not None:
        parsed["orbital_analysis"] = {
            "requested_orbital_descriptor": orbital_descriptor,
            "mo_file_paths": [
                path for key, path in sorted(file_paths.items()) if key.endswith("_mo")
            ],
            "mo_artifacts_available": has_mo_file,
        }
    return parsed


def _bright_state(excited_states: list[dict[str, object]]) -> dict[str, object]:
    if not excited_states:
        return {}
    return max(
        excited_states,
        key=lambda state: float(state.get("oscillator_strength") or 0.0),
    )


def _parse_mulliken_charges(text: str) -> list[float]:
    lower = text.lower()
    if "mulliken" not in lower:
        return []
    charge_lines = [
        line
        for line in text.splitlines()
        if re.search(r"\b[A-Z][a-z]?\b", line) and re.search(r"[-+]?\d+\.\d+", line)
    ]
    charges: list[float] = []
    for line in charge_lines:
        numbers = re.findall(r"[-+]?\d+\.\d+", line)
        if numbers:
            charges.append(float(numbers[-1]))
    return charges


def _frontier_orbital_partition_payload(
    text: str,
    artifact: ArtifactRecord,
) -> dict[str, object]:
    frontier = _parse_frontier_orbitals(text)
    coefficients = _parse_ao_coefficient_weights(text)
    fragments = _frontier_fragment_atoms(artifact)
    homo = _frontier_partition_record(frontier.get("homo"), coefficients, fragments)
    lumo = _frontier_partition_record(frontier.get("lumo"), coefficients, fragments)
    available = [
        descriptor
        for descriptor, present in {
            "frontier_orbital_energies": bool(frontier.get("homo") and frontier.get("lumo")),
            "frontier_orbital_partition": bool(homo and lumo),
        }.items()
        if present
    ]
    return {
        "requested_descriptors": [
            "frontier_orbital_energies",
            "frontier_orbital_partition",
        ],
        "available_descriptors": available,
        "missing_descriptors": [
            descriptor
            for descriptor in [
                "frontier_orbital_energies",
                "frontier_orbital_partition",
            ]
            if descriptor not in available
        ],
        "frontier_orbitals": {"homo": homo, "lumo": lumo},
        "fragment_definition": fragments,
    }


def _parse_frontier_orbitals(text: str) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    patterns = {
        "homo": r"HOMO\s*=\s*([-0-9.]+)\s+AU\s+index\s*:\s*(\d+)",
        "lumo": r"LUMO\s*=\s*([-0-9.]+)\s+AU\s+index\s*:\s*(\d+)",
    }
    for label, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            records[label] = {
                "index": int(match.group(2)),
                "energy_au": float(match.group(1)),
            }
    return records


def _parse_ao_coefficient_weights(text: str) -> dict[int, dict[int, float]]:
    weights_by_mo: dict[int, dict[int, float]] = {}
    in_coefficients = False
    current_mos: list[int] = []
    current_atom: int | None = None
    for line in text.splitlines():
        if "Molecular Orbital Coefficients:" in line:
            in_coefficients = True
            continue
        if in_coefficients and line.strip().startswith("Gross orbital populations"):
            break
        if not in_coefficients:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"(?:\d+\s*)+", stripped):
            current_mos = [int(value) for value in stripped.split()]
            continue
        if stripped.startswith("Eigenvalues") or re.fullmatch(r"(?:[OV]\s*)+", stripped):
            continue
        parts = stripped.split()
        if not parts or not parts[0].isdigit() or not current_mos:
            continue
        coeff_start = _coefficient_start(parts)
        if coeff_start is None:
            continue
        if len(parts) >= 4 and parts[1].isdigit() and re.fullmatch(r"[A-Za-z]+", parts[2]):
            current_atom = int(parts[1])
        if current_atom is None or len(parts) < coeff_start + len(current_mos):
            continue
        try:
            values = [float(value) for value in parts[coeff_start : coeff_start + len(current_mos)]]
        except ValueError:
            continue
        for mo_index, coefficient in zip(current_mos, values, strict=True):
            mo_weights = weights_by_mo.setdefault(mo_index, {})
            mo_weights[current_atom] = mo_weights.get(current_atom, 0.0) + coefficient**2
    return weights_by_mo


def _coefficient_start(parts: list[str]) -> int | None:
    if len(parts) >= 4 and parts[1].isdigit() and re.fullmatch(r"[A-Za-z]+", parts[2]):
        return 4
    return 2 if len(parts) >= 2 else None


def _frontier_fragment_atoms(artifact: ArtifactRecord) -> dict[str, object]:
    mol, limitations = _mol_from_artifact(artifact)
    if mol is None:
        return {"donor_atoms": [], "acceptor_atoms": [], "limitations": limitations}
    donor_atoms: list[int] = []
    acceptor_atoms: list[int] = []
    hetero_atoms: list[int] = []
    for atom in mol.GetAtoms():
        atom_number = atom.GetAtomicNum()
        atom_index = int(atom.GetIdx()) + 1
        if atom_number in {7, 8, 15, 16}:
            hetero_atoms.append(atom_index)
        if atom_number in {7, 8, 15, 16} and (
            atom.GetTotalNumHs() > 0 or atom.GetFormalCharge() > 0
        ):
            donor_atoms.append(atom_index)
        if atom_number in {7, 8, 16} and atom.GetFormalCharge() <= 0 and atom.GetTotalNumHs() == 0:
            acceptor_atoms.append(atom_index)
    return {
        "donor_atoms": donor_atoms or hetero_atoms,
        "acceptor_atoms": acceptor_atoms or hetero_atoms,
        "atom_index_base": "amesp_one_based_matches_prepared_structure_order",
        "limitations": limitations,
    }


def _frontier_partition_record(
    orbital: dict[str, object] | None,
    coefficients: dict[int, dict[int, float]],
    fragments: dict[str, object],
) -> dict[str, object] | None:
    if orbital is None:
        return None
    mo_index = int(orbital["index"])
    weights = coefficients.get(mo_index, {})
    total = sum(weights.values())
    if total <= 0:
        return {**orbital, "raw_squared_coeff_sum": 0.0}
    normalized = {atom: value / total for atom, value in weights.items()}
    donor_atoms = {int(atom) for atom in fragments.get("donor_atoms", [])}
    acceptor_atoms = {int(atom) for atom in fragments.get("acceptor_atoms", [])}
    return {
        **orbital,
        "raw_squared_coeff_sum": total,
        "donor_fragment_weight": sum(normalized.get(atom, 0.0) for atom in donor_atoms),
        "acceptor_fragment_weight": sum(normalized.get(atom, 0.0) for atom in acceptor_atoms),
        "top_atom_weights": [
            {"atom_index": atom, "weight": weight}
            for atom, weight in sorted(
                normalized.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:5]
        ],
    }


def _charge_scheme_record(scheme: str, aop_path: Path, text: str) -> dict[str, object]:
    lower = text.lower()
    return {
        "scheme": scheme,
        "aop_path": str(aop_path),
        "scheme_mentioned": scheme in lower,
        "charge_keyword_mentioned": "charge" in lower,
        "numeric_line_count": sum(
            1 for line in text.splitlines() if re.search(r"[-+]?\d+\.\d+", line)
        ),
    }


def _solvation_proxy_payload(
    profile: MicroscopicRouteProfile,
    solvent_records: list[dict[str, object]],
    failures: list[dict[str, object]],
) -> dict[str, object]:
    first_states = [
        record["first_state"] for record in solvent_records if record.get("first_state")
    ]
    bright_states = [
        record["bright_state"] for record in solvent_records if record.get("bright_state")
    ]
    available = [
        descriptor
        for descriptor, present in {
            "solvent_state_ordering": bool(first_states),
            "solvent_energy_shift": len(first_states) >= 2,
            "solvent_oscillator_shift": len(bright_states) >= 2,
        }.items()
        if present
    ]
    return {
        "requested_descriptors": profile.requested_descriptors,
        "available_descriptors": available,
        "missing_descriptors": [
            descriptor
            for descriptor in profile.requested_descriptors
            if descriptor not in available
        ],
        "solvent_records": solvent_records,
        "solvent_failures": failures,
        "first_excitation_energy_shift_ev": _state_delta(
            first_states,
            "excitation_energy_ev",
        ),
        "bright_oscillator_strength_delta": _state_delta(
            bright_states,
            "oscillator_strength",
        ),
    }


def _state_delta(states: list[object], key: str) -> float | None:
    if len(states) < 2:
        return None
    left = states[0] if isinstance(states[0], dict) else {}
    right = states[1] if isinstance(states[1], dict) else {}
    if left.get(key) is None or right.get(key) is None:
        return None
    return round(float(left[key]) - float(right[key]), 6)


def _rotatable_dihedral_inventory(
    artifact: ArtifactRecord,
) -> tuple[dict[str, object], list[str], list[str]]:
    mol, limitations = _mol_from_artifact(artifact)
    if mol is None:
        return (
            {"dihedral_candidates": [], "candidate_count": 0},
            ["stable dihedral descriptors"],
            limitations,
        )
    candidates = _rotatable_dihedrals_from_mol(mol)
    return (
        {
            "dihedral_candidates": candidates,
            "candidate_count": len(candidates),
            "selected_policy": "single_non_ring_heavy_atom_bonds",
        },
        [] if candidates else ["stable dihedral descriptors"],
        limitations,
    )


def _conformer_inventory(
    artifact: ArtifactRecord,
) -> tuple[dict[str, object], list[str], list[str]]:
    metadata = artifact.metadata or {}
    selected_id = metadata.get("selected_conformer_id", 0)
    count = int(metadata.get("conformer_count", 1) or 1)
    conformers = [
        {
            "conformer_id": f"prepared:{selected_id}",
            "source": "prepared_structure",
            "selected": True,
            "force_field": metadata.get("force_field"),
        }
    ]
    return (
        {
            "conformers": conformers,
            "available_conformer_count": len(conformers),
            "prepared_conformer_count": count,
        },
        [] if conformers else ["stable conformer descriptors"],
        [],
    )


def _conformer_members(
    artifact: ArtifactRecord,
    workdir: Path,
    *,
    max_members: int,
    member_prefix: str,
) -> tuple[list[PreparedGeometryMember], list[str]]:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ModuleNotFoundError as exc:
        raise AmespMicroscopicError(
            "precondition_missing",
            "RDKit is required to generate a multi-conformer series.",
        ) from exc

    smiles = artifact.metadata.get("canonical_smiles") or artifact.metadata.get("input_smiles")
    base_mol = Chem.MolFromSmiles(str(smiles)) if smiles else None
    if base_mol is None:
        raise AmespMicroscopicError(
            "precondition_missing",
            "Prepared artifact does not expose parseable SMILES for conformer generation.",
            details={"artifact_id": artifact.artifact_id},
        )
    mol = Chem.AddHs(base_mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xA1E
    conformer_ids = list(
        AllChem.EmbedMultipleConfs(mol, numConfs=max(max_members, 3), params=params)
    )
    if not conformer_ids:
        params.useRandomCoords = True
        conformer_ids = list(
            AllChem.EmbedMultipleConfs(mol, numConfs=max(max_members, 3), params=params)
        )
    ranked = _rank_conformers(AllChem, mol, conformer_ids)
    members_dir = workdir / "generated_conformers"
    members_dir.mkdir(parents=True, exist_ok=True)
    members = [
        _rdkit_member(
            mol=mol,
            conf_id=conf_id,
            member_id=f"{member_prefix}:{rank:02d}",
            label=f"{member_prefix}_{rank:02d}",
            xyz_path=members_dir / f"{member_prefix}_{rank:02d}.xyz",
            metadata={
                "conformer_rank": rank,
                "rdkit_conformer_id": int(conf_id),
                "force_field": force_field,
                "force_field_energy": energy,
            },
        )
        for rank, (conf_id, force_field, energy) in enumerate(ranked[:max_members], start=1)
    ]
    return members, []


def _torsion_snapshot_members(
    artifact: ArtifactRecord,
    workdir: Path,
    *,
    max_members: int,
) -> tuple[list[PreparedGeometryMember], list[str]]:
    try:
        from rdkit.Chem import rdMolTransforms
    except ModuleNotFoundError as exc:
        raise AmespMicroscopicError(
            "precondition_missing",
            "RDKit is required to generate a torsion snapshot series.",
        ) from exc

    mol, limitations = _mol_from_artifact(artifact)
    if mol is None:
        raise AmespMicroscopicError(
            "precondition_missing",
            "Prepared artifact does not expose RDKit-readable geometry for torsion snapshots.",
            details={"artifact_id": artifact.artifact_id, "limitations": limitations},
        )
    if mol.GetNumConformers() == 0:
        raise AmespMicroscopicError(
            "precondition_missing",
            "Prepared artifact molecule has no 3D conformer for torsion snapshots.",
            details={"artifact_id": artifact.artifact_id},
        )
    candidates = _rotatable_dihedrals_from_mol(mol)
    if not candidates:
        raise AmespMicroscopicError(
            "precondition_missing",
            "No non-ring heavy-atom rotatable dihedral was found for torsion snapshots.",
            details={"artifact_id": artifact.artifact_id},
        )

    selected = candidates[0]
    atom_indices = [int(index) for index in selected["atom_indices"]]
    current_angle = rdMolTransforms.GetDihedralDeg(mol.GetConformer(), *atom_indices)
    snapshot_dir = workdir / "generated_torsion_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    members: list[PreparedGeometryMember] = []
    for rank, target_angle in enumerate(
        _torsion_target_angles(current_angle, max_members),
        start=1,
    ):
        snapshot_mol = type(mol)(mol)
        rdMolTransforms.SetDihedralDeg(
            snapshot_mol.GetConformer(),
            *atom_indices,
            float(target_angle),
        )
        label = f"torsion_snapshot_{rank:02d}"
        members.append(
            _rdkit_member(
                mol=snapshot_mol,
                conf_id=snapshot_mol.GetConformer().GetId(),
                member_id=f"torsion_snapshot:{rank:02d}",
                label=label,
                xyz_path=snapshot_dir / f"{label}.xyz",
                metadata={
                    "snapshot_rank": rank,
                    "dihedral_id": selected["dihedral_id"],
                    "dihedral_atoms": atom_indices,
                    "initial_angle_deg": round(current_angle, 6),
                    "target_angle_deg": round(target_angle, 6),
                },
            )
        )
    return members, limitations


def _artifact_bundle_inventory(
    artifact: ArtifactRecord,
    bundle: MicroscopicArtifactBundle,
) -> tuple[dict[str, object], list[str], list[str]]:
    return (
        {
            "artifact_bundles": [
                {
                    "artifact_id": artifact.artifact_id,
                    "bundle_id": bundle.bundle_id,
                    "kind": artifact.kind,
                    "status": artifact.status,
                    "capability_name": bundle.capability_name,
                    "file_count": len(bundle.file_paths),
                    "state_count": len(bundle.excited_states),
                }
            ],
            "bundle_count": 1,
        },
        [],
        [],
    )


def _artifact_bundle_members(
    artifact: ArtifactRecord,
    bundle: MicroscopicArtifactBundle,
) -> tuple[dict[str, object], list[str], list[str]]:
    del artifact
    members = [
        {
            "member_id": f"step:{step.step_id}",
            "member_type": "amesp_step",
            "method_label": step.method_label,
            "aop_path": step.aop_path,
            "terminated_normally": step.terminated_normally,
        }
        for step in bundle.step_records
    ]
    members.extend(
        {
            "member_id": f"file:{name}",
            "member_type": "file",
            "path": path,
        }
        for name, path in sorted(bundle.file_paths.items())
        if name not in {"manifest", "workdir"}
    )
    return (
        {"bundle_id": bundle.bundle_id, "members": members, "member_count": len(members)},
        [] if members else ["artifact bundle member descriptors"],
        [],
    )


def _ct_descriptor_payload(
    artifact: ArtifactRecord,
    bundle: MicroscopicArtifactBundle,
) -> tuple[dict[str, object], list[str], list[str]]:
    del artifact
    missing = ["hole_electron_separation", "transition_density_cube"]
    return (
        {
            "source_bundle_id": bundle.bundle_id,
            "state_count": len(bundle.excited_states),
            "bright_state": _bright_state(bundle.excited_states),
            "available_descriptors": _available_ct_descriptors(bundle),
            "missing_descriptors": missing,
        },
        missing,
        [
            "CT extraction uses artifact-backed scalar surrogates only; "
            "density-grid CT descriptors are not inferred."
        ],
    )


def _snapshot_parse_payload(
    artifact: ArtifactRecord,
    bundle: MicroscopicArtifactBundle,
) -> tuple[dict[str, object], list[str], list[str]]:
    del artifact
    parsed_states = [
        {
            "state_index": state.get("state_index"),
            "excitation_energy_ev": state.get("excitation_energy_ev"),
            "oscillator_strength": state.get("oscillator_strength"),
        }
        for state in bundle.excited_states
    ]
    return (
        {
            "source_bundle_id": bundle.bundle_id,
            "parsed_states": parsed_states,
            "state_count": len(parsed_states),
            "bright_state": _bright_state(bundle.excited_states),
        },
        [] if parsed_states else ["per-snapshot excitation energies"],
        [],
    )


def _torsion_candidate_payload(
    artifact: ArtifactRecord,
    bundle: MicroscopicArtifactBundle,
) -> tuple[dict[str, object], list[str], list[str]]:
    del artifact
    geometry_path = bundle.file_paths.get("analysis_geometry_xyz") or bundle.file_paths.get(
        "source_xyz"
    )
    return (
        {
            "source_bundle_id": bundle.bundle_id,
            "geometry_path": geometry_path,
            "torsion_candidates": [],
            "candidate_count": 0,
        },
        ["rotatable_dihedral_candidates"],
        [
            "Bundle-only torsion extraction needs a connectivity graph; use "
            "list_rotatable_dihedrals on the prepared structure for exact candidates."
        ],
    )


def _geometry_descriptor_payload(
    artifact: ArtifactRecord,
    bundle: MicroscopicArtifactBundle,
) -> tuple[dict[str, object], list[str], list[str]]:
    del artifact
    xyz_path = _bundle_geometry_path(bundle)
    symbols, coordinates = _read_xyz_or_raise(xyz_path)
    descriptors = _geometry_descriptors(symbols, coordinates)
    return (
        {
            "source_bundle_id": bundle.bundle_id,
            "geometry_path": str(xyz_path),
            "geometry_descriptors": descriptors,
        },
        [],
        [],
    )


def _raw_artifact_payload(
    artifact: ArtifactRecord,
    bundle: MicroscopicArtifactBundle,
) -> tuple[dict[str, object], list[str], list[str]]:
    files = [
        {"label": label, "path": path, "exists": Path(path).exists()}
        for label, path in sorted(bundle.file_paths.items())
    ]
    observables = sorted(
        set(bundle.parsed_observables.get("available_descriptors") or [])
        | set(artifact.observable_tags)
    )
    return (
        {
            "source_bundle_id": bundle.bundle_id,
            "files": files,
            "file_count": len(files),
            "extractable_observables": observables,
        },
        [] if files else ["raw file inventory"],
        [],
    )


def _rank_conformers(AllChem, mol, conformer_ids: list[int]) -> list[tuple[int, str, float]]:
    force_field = _select_rdkit_force_field(AllChem, mol)
    ranked: list[tuple[int, str, float]] = []
    for conf_id in conformer_ids:
        forcefield = _build_rdkit_forcefield(AllChem, mol, conf_id, force_field)
        forcefield.Minimize(maxIts=200)
        ranked.append((int(conf_id), force_field, round(float(forcefield.CalcEnergy()), 8)))
    if not ranked:
        raise AmespMicroscopicError(
            "precondition_missing",
            "RDKit did not produce any optimizable conformers.",
        )
    return sorted(ranked, key=lambda item: item[2])


def _select_rdkit_force_field(AllChem, mol) -> str:
    if AllChem.MMFFHasAllMoleculeParams(mol):
        return "MMFF94"
    if AllChem.UFFHasAllMoleculeParams(mol):
        return "UFF"
    raise AmespMicroscopicError(
        "precondition_missing",
        "Neither MMFF94 nor UFF parameters are available for this molecule.",
    )


def _build_rdkit_forcefield(AllChem, mol, conf_id: int, force_field: str):
    builders = {
        "MMFF94": lambda: AllChem.MMFFGetMoleculeForceField(
            mol,
            AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94"),
            confId=conf_id,
        ),
        "UFF": lambda: AllChem.UFFGetMoleculeForceField(mol, confId=conf_id),
    }
    forcefield = builders[force_field]()
    if forcefield is None:
        raise AmespMicroscopicError(
            "precondition_missing",
            f"Failed to build {force_field} force field for conformer {conf_id}.",
        )
    return forcefield


def _rdkit_member(
    *,
    mol,
    conf_id: int,
    member_id: str,
    label: str,
    xyz_path: Path,
    metadata: dict[str, object],
) -> PreparedGeometryMember:
    conformer = mol.GetConformer(conf_id)
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coordinates = [
        [
            float(conformer.GetAtomPosition(atom_index).x),
            float(conformer.GetAtomPosition(atom_index).y),
            float(conformer.GetAtomPosition(atom_index).z),
        ]
        for atom_index in range(mol.GetNumAtoms())
    ]
    _write_xyz(xyz_path, label=label, symbols=symbols, coordinates=coordinates)
    return PreparedGeometryMember(
        member_id=member_id,
        label=label,
        symbols=symbols,
        coordinates=coordinates,
        xyz_path=xyz_path,
        metadata=metadata,
    )


def _torsion_target_angles(current_angle: float, max_members: int) -> list[float]:
    offsets = [0.0, 60.0, -60.0, 120.0, -120.0, 180.0]
    targets: list[float] = []
    for offset in offsets:
        normalized = ((current_angle + offset + 180.0) % 360.0) - 180.0
        if all(abs(normalized - existing) > 1e-6 for existing in targets):
            targets.append(normalized)
        if len(targets) >= max_members:
            break
    return targets


def _member_series_summary(
    route_records: list[dict[str, object]],
    member_failures: list[dict[str, object]],
) -> dict[str, object]:
    energies = [
        float(record["final_energy_hartree"])
        for record in route_records
        if record.get("final_energy_hartree") is not None
    ]
    oscillators = [
        float(record["first_oscillator_strength"])
        for record in route_records
        if record.get("first_oscillator_strength") is not None
    ]
    summary: dict[str, object] = {
        "attempted_member_count": len(route_records) + len(member_failures),
        "successful_member_count": len(route_records),
        "member_failure_count": len(member_failures),
        "bundle_completion_status": "partial" if member_failures else "complete",
    }
    if energies:
        summary["energy_range_hartree"] = round(max(energies) - min(energies), 8)
    if oscillators:
        summary["oscillator_strength_range"] = round(max(oscillators) - min(oscillators), 8)
    if member_failures:
        summary["member_failures"] = member_failures
    return summary


def _mol_from_artifact(artifact: ArtifactRecord) -> tuple[object | None, list[str]]:
    try:
        from rdkit import Chem
    except ModuleNotFoundError:
        return None, ["RDKit is unavailable; connectivity-based discovery cannot run."]

    sdf_path = artifact.file_paths.get("sdf")
    if sdf_path and Path(sdf_path).exists():
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        mol = supplier[0] if supplier and len(supplier) else None
        if mol is not None:
            return mol, []

    smiles = artifact.metadata.get("canonical_smiles") or artifact.metadata.get("input_smiles")
    mol = Chem.MolFromSmiles(str(smiles)) if smiles else None
    if mol is None:
        return None, ["Prepared artifact does not expose parseable SDF or SMILES."]
    return mol, []


def _rotatable_dihedrals_from_mol(mol: object) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for bond in mol.GetBonds():
        if bond.GetBondTypeAsDouble() != 1.0 or bond.IsInRing():
            continue
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if begin.GetAtomicNum() == 1 or end.GetAtomicNum() == 1:
            continue
        begin_neighbors = [
            atom for atom in begin.GetNeighbors() if atom.GetIdx() != end.GetIdx()
        ]
        end_neighbors = [
            atom for atom in end.GetNeighbors() if atom.GetIdx() != begin.GetIdx()
        ]
        if not begin_neighbors or not end_neighbors:
            continue
        left = max(begin_neighbors, key=lambda atom: atom.GetAtomicNum())
        right = max(end_neighbors, key=lambda atom: atom.GetAtomicNum())
        atom_indices = [left.GetIdx(), begin.GetIdx(), end.GetIdx(), right.GetIdx()]
        candidates.append(
            {
                "dihedral_id": f"dih:{len(candidates) + 1:03d}",
                "atom_indices": atom_indices,
                "atom_symbols": [mol.GetAtomWithIdx(index).GetSymbol() for index in atom_indices],
                "central_bond_indices": [begin.GetIdx(), end.GetIdx()],
                "bond_type": "single_non_ring",
            }
        )
    return candidates


def _geometry_descriptors(
    symbols: list[str],
    coordinates: list[list[float]],
) -> dict[str, object]:
    centroid = [
        round(sum(row[index] for row in coordinates) / len(coordinates), 6)
        for index in range(3)
    ]
    extents = [
        round(max(row[index] for row in coordinates) - min(row[index] for row in coordinates), 6)
        for index in range(3)
    ]
    pair_distances = (
        dist(left, right)
        for left_index, left in enumerate(coordinates)
        for right in coordinates[left_index + 1 :]
    )
    max_pair_distance = max(pair_distances, default=0.0)
    return {
        "atom_count": len(symbols),
        "heavy_atom_count": sum(1 for symbol in symbols if symbol != "H"),
        "centroid_angstrom": centroid,
        "axis_extents_angstrom": extents,
        "max_pair_distance_angstrom": round(max_pair_distance, 6),
    }
