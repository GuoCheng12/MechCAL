"""Bounded singlet-root relaxation with independently checked endpoint spectra."""

from dataclasses import dataclass, field
import hashlib
from math import isfinite
from pathlib import Path
import re

from mechcal.schemas.tool_arguments import RelaxationArgs
from mechcal.tools.amesp_baseline import (
    AmespBaselineError,
    AmespBaselineRunner,
    AmespStepResult,
    _parse_excited_states,
    _parse_final_energy,
    _parse_final_geometry,
    _write_xyz,
)


@dataclass
class RelaxationOutput:
    observables: dict = field(default_factory=dict)
    file_paths: dict[str, str] = field(default_factory=dict)
    steps: list[AmespStepResult] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    final_spectrum_text: str = ""


def optimization_history(text: str) -> list[dict]:
    """Keep each optimizer iteration separate from an endpoint state manifold."""
    pieces = re.split(r"Geom Opt Step:\s*(\d+)", text)
    return [
        {
            "step": int(pieces[i]),
            "optimizer_energy_hartree": _parse_final_energy(pieces[i + 1]),
            "excited_states": _parse_excited_states(pieces[i + 1], reference_energy_hartree=None),
        }
        for i in range(1, len(pieces), 2)
    ]


def converged_geometry(text: str, symbols: list[str]) -> tuple[list[list[float]], int]:
    match = re.search(r"Geometry Optimization Converged after (\d+) Steps", text)
    if not match:
        raise AmespBaselineError(
            "optimization_not_converged", "No explicit optimization convergence marker."
        )
    try:
        parsed_symbols, coordinates = _parse_final_geometry(text)
    except ValueError as exc:
        raise AmespBaselineError("parse_failed", "Malformed final geometry.") from exc
    if parsed_symbols != symbols or len(coordinates) != len(symbols):
        raise AmespBaselineError(
            "parse_failed", "Final geometry atom count/order differs from input."
        )
    if not all(isfinite(x) for row in coordinates for x in row):
        raise AmespBaselineError("parse_failed", "Non-finite final coordinates.")
    return coordinates, int(match.group(1))


def endpoint_states(text: str, count: int) -> list[dict]:
    states = _parse_excited_states(text, reference_energy_hartree=None)
    if [s["state_index"] for s in states] != list(range(1, count + 1)):
        raise AmespBaselineError("parse_failed", "Expected one complete endpoint state manifold.")
    for state in states:
        for key in (
            "excitation_energy_ev",
            "total_energy_hartree",
            "spin_square",
            "oscillator_strength",
        ):
            value = state.get(key)
            if not isinstance(value, (int, float)) or not isfinite(value):
                raise AmespBaselineError(
                    "parse_failed", f"Missing/invalid endpoint observable: {key}."
                )
        if abs(state["spin_square"]) > 0.05:
            raise AmespBaselineError(
                "parse_failed", "Endpoint states are not the requested singlet manifold."
            )
    return states


def run_relaxation(
    runner: AmespBaselineRunner,
    *,
    workdir: Path,
    symbols: list[str],
    coordinates: list[list[float]],
    charge: int,
    multiplicity: int,
    args: RelaxationArgs,
) -> RelaxationOutput:
    output = RelaxationOutput(
        observables={
            "method": "aTB1/TDA",
            "environment": "gas_phase",
            "state_root": args.state_root,
            "state_spin": "singlet",
            "atom_index_base": 0,
            "symbols": symbols,
            "initial_coordinates_angstrom": coordinates,
            "optimization_converged": False,
            "state_tracking": "fixed_ordinal_root_not_character_tracked",
            "minimum_verified_by_frequencies": False,
            "requested_descriptors": [
                "initial_state_manifold",
                "relaxed_geometry",
                "final_state_manifold",
            ],
            "available_descriptors": [],
            "comparison": None,
        }
    )
    if not runner.amesp_bin.is_file():
        raise AmespBaselineError(
            "amesp_binary_missing", f"Amesp binary not found: {runner.amesp_bin}"
        )
    with runner.amesp_bin.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    output.observables["backend"] = {"binary": str(runner.amesp_bin), "sha256": digest}

    common = [
        ("atb", ["excdip on"]),
        ("scf", ["maxcyc 500", "vshift 500", "conver 10"]),
        (
            "posthf",
            [
                f"nstates {args.s1_nstates}",
                f"root {args.state_root}",
                "tdspin s",
                f"tout {args.td_tout}",
                "conver 10",
            ],
        ),
    ]

    def execute(name, xyz, optimize=False):
        blocks = list(common)
        if optimize:
            blocks.append(("opt", [f"maxcyc {args.max_opt_cycles}", "gediis off", "maxstep 0.1"]))
        try:
            step = runner._run_step(
                step_id=name,
                label=name,
                workdir=workdir,
                keywords=["aTB1", "tda", *(["opt"] if optimize else [])],
                block_lines=blocks,
                charge=charge,
                multiplicity=multiplicity,
                symbols=symbols,
                coordinates=xyz,
            )
        except AmespBaselineError as exc:
            output.file_paths.update({f"{name}_{k}": v for k, v in exc.file_paths.items()})
            raise
        output.steps.append(step)
        output.file_paths.update(
            {
                f"{name}_aip": str(step.aip_path),
                f"{name}_aop": str(step.aop_path),
                f"{name}_stdout": str(step.stdout_path),
                f"{name}_stderr": str(step.stderr_path),
            }
        )
        return step.aop_path.read_text(encoding="utf-8", errors="replace")

    stage = "initial_state_manifold"
    try:
        initial_text = execute("initial_vertical", coordinates)
        initial = endpoint_states(initial_text, args.s1_nstates)
        release = re.search(r"Version\s+([^\n*]+)", initial_text)
        output.observables["backend"]["release"] = (
            release.group(1).strip() if release else "unparsed"
        )
        output.observables["initial_excited_states"] = initial
        output.observables["available_descriptors"].append(stage)

        stage = "relaxed_geometry"
        optimized_text = execute("excited_optimization", coordinates, optimize=True)
        output.observables["optimization_history"] = optimization_history(optimized_text)
        final_xyz, iterations = converged_geometry(optimized_text, symbols)
        geometry_path = workdir / "relaxed_geometry.xyz"
        _write_xyz(
            geometry_path,
            label=f"aTB1/TDA singlet root {args.state_root}",
            symbols=symbols,
            coordinates=final_xyz,
        )
        output.file_paths["analysis_geometry_xyz"] = str(geometry_path)
        output.observables.update(
            optimization_converged=True,
            optimization_steps=iterations,
            final_coordinates_angstrom=final_xyz,
            optimizer_final_energy_hartree=_parse_final_energy(optimized_text),
        )
        output.observables["available_descriptors"].append(stage)

        stage = "final_state_manifold"
        final_text = execute("final_vertical", final_xyz)
        final = endpoint_states(final_text, args.s1_nstates)
        output.final_spectrum_text = final_text
        output.observables["final_excited_states"] = final
        output.observables["available_descriptors"].append(stage)
        before, after = initial[args.state_root - 1], final[args.state_root - 1]
        output.observables["comparison"] = {
            "scope": "same_ordinal_singlet_root_at_input_and_relaxed_geometries",
            "excitation_energy_change_ev": after["excitation_energy_ev"]
            - before["excitation_energy_ev"],
            "total_excited_energy_change_hartree": after["total_energy_hartree"]
            - before["total_energy_hartree"],
            "oscillator_strength_change": after["oscillator_strength"]
            - before["oscillator_strength"],
        }
    except AmespBaselineError as exc:
        output.observables["stage_failure"] = {
            "stage": stage,
            "code": exc.code,
            "message": exc.message,
        }
        raw = output.file_paths.get("excited_optimization_aop")
        if raw and Path(raw).is_file():
            output.observables["optimization_history"] = optimization_history(
                Path(raw).read_text(errors="replace")
            )
        output.missing = [
            name
            for name in output.observables["requested_descriptors"]
            if name not in output.observables["available_descriptors"]
        ]
        if not output.observables["available_descriptors"]:
            raise AmespBaselineError(
                exc.code, exc.message, file_paths=output.file_paths, details=exc.details
            ) from exc
    output.observables["missing_descriptors"] = output.missing
    return output
