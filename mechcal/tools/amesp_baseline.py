from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AmespStepResult:
    step_id: str
    aip_path: Path
    aop_path: Path
    stdout_path: Path
    stderr_path: Path
    exit_code: int
    terminated_normally: bool
    elapsed_seconds: float

    def to_json(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "aip_path": str(self.aip_path),
            "aop_path": str(self.aop_path),
            "stdout_path": str(self.stdout_path),
            "stderr_path": str(self.stderr_path),
            "exit_code": self.exit_code,
            "terminated_normally": self.terminated_normally,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True)
class AmespBaselineResult:
    workdir: Path
    s0: AmespStepResult
    s1: AmespStepResult
    s0_optimized_xyz_path: Path
    final_energy_hartree: float | None
    excited_states: list[dict[str, object]] = field(default_factory=list)

    def file_paths(self) -> dict[str, str]:
        return {
            "workdir": str(self.workdir),
            "s0_aip": str(self.s0.aip_path),
            "s0_aop": str(self.s0.aop_path),
            "s0_stdout": str(self.s0.stdout_path),
            "s0_stderr": str(self.s0.stderr_path),
            "s1_aip": str(self.s1.aip_path),
            "s1_aop": str(self.s1.aop_path),
            "s1_stdout": str(self.s1.stdout_path),
            "s1_stderr": str(self.s1.stderr_path),
            "s0_optimized_xyz": str(self.s0_optimized_xyz_path),
        }


class AmespBaselineError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        file_paths: dict[str, str] | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.file_paths = file_paths or {}
        self.details = details or {}


class AmespBaselineRunner:
    def __init__(
        self,
        *,
        amesp_bin: Path | None = None,
        timeout_seconds: float | None = None,
        npara: int | None = None,
        maxcore_mb: int | None = None,
        s1_nstates: int = 1,
        td_tout: int = 1,
    ) -> None:
        self.amesp_bin = self._resolve_amesp_bin(amesp_bin)
        self.timeout_seconds = timeout_seconds or float(
            os.environ.get("MECHCAL_AMESP_TIMEOUT", "300")
        )
        self.npara = npara if npara is not None else _env_int("MECHCAL_AMESP_NPARA", 1)
        self.maxcore_mb = (
            maxcore_mb
            if maxcore_mb is not None
            else _env_int("MECHCAL_AMESP_MAXCORE_MB", 1000)
        )
        self.s1_nstates = max(1, int(s1_nstates))
        self.td_tout = max(1, int(td_tout))

    def run_baseline(
        self,
        *,
        prepared_artifact: Any,
        case_id: str,
        round_id: str,
    ) -> AmespBaselineResult:
        if not self.amesp_bin.exists():
            raise AmespBaselineError(
                "amesp_binary_missing",
                f"Amesp binary not found: {self.amesp_bin}",
            )
        xyz_path = Path(str(prepared_artifact.file_paths.get("xyz") or ""))
        if not xyz_path.exists():
            raise AmespBaselineError(
                "precondition_missing",
                "Prepared structure xyz file is missing.",
            )
        symbols, coordinates = _read_xyz_symbols_and_coordinates(xyz_path)
        if not symbols:
            raise AmespBaselineError(
                "precondition_missing",
                f"Prepared structure xyz file is not parseable: {xyz_path}",
            )

        metadata = dict(prepared_artifact.metadata or {})
        charge = int(metadata.get("charge", 0))
        multiplicity = int(metadata.get("multiplicity", 1))
        workdir = xyz_path.parents[2] / "artifacts" / "microscopic" / round_id
        workdir.mkdir(parents=True, exist_ok=True)
        label = _safe_label(f"{case_id}_{round_id}")

        s0 = self._run_step(
            step_id="s0_optimization",
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
        s0_text = s0.aop_path.read_text(encoding="utf-8", errors="replace")
        s0_symbols, s0_coordinates = _parse_final_geometry(s0_text)
        if not s0_symbols:
            raise AmespBaselineError(
                "parse_failed",
                "Amesp S0 output did not expose a parseable final geometry.",
                file_paths=_step_file_paths(s0),
            )
        s0_xyz_path = workdir / "s0_optimized.xyz"
        _write_xyz(
            s0_xyz_path,
            label=f"{label}_s0_optimized",
            symbols=s0_symbols,
            coordinates=s0_coordinates,
        )

        s1 = self._run_step(
            step_id="s1_vertical_excitation",
            label=f"{label}_s1",
            workdir=workdir,
            keywords=["aTB1", "tda"],
            block_lines=[
                ("ope", ["out 1"]),
                ("atb", ["excdip on"]),
                ("scf", ["maxcyc 2000", "vshift 500"]),
                ("posthf", [f"nstates {self.s1_nstates}", f"tout {self.td_tout}"]),
            ],
            charge=charge,
            multiplicity=multiplicity,
            symbols=s0_symbols,
            coordinates=s0_coordinates,
        )
        s1_text = s1.aop_path.read_text(encoding="utf-8", errors="replace")
        final_energy = _parse_final_energy(s0_text)
        return AmespBaselineResult(
            workdir=workdir,
            s0=s0,
            s1=s1,
            s0_optimized_xyz_path=s0_xyz_path,
            final_energy_hartree=final_energy,
            excited_states=_parse_excited_states(s1_text, reference_energy_hartree=final_energy),
        )

    def _run_step(
        self,
        *,
        step_id: str,
        label: str,
        workdir: Path,
        keywords: list[str],
        block_lines: list[tuple[str, list[str]]],
        charge: int,
        multiplicity: int,
        symbols: list[str],
        coordinates: list[list[float]],
    ) -> AmespStepResult:
        aip_path = workdir / f"{label}.aip"
        aop_path = workdir / f"{label}.aop"
        stdout_path = workdir / f"{label}.stdout.log"
        stderr_path = workdir / f"{label}.stderr.log"
        run_configs = [(self.npara, self.maxcore_mb)]
        safe_config = (1, min(self.maxcore_mb, 1000))
        if safe_config not in run_configs:
            run_configs.append(safe_config)
        last_error: AmespBaselineError | None = None
        for attempt_index, (npara, maxcore_mb) in enumerate(run_configs, start=1):
            try:
                return self._run_step_once(
                    step_id=step_id,
                    workdir=workdir,
                    aip_path=aip_path,
                    aop_path=aop_path,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    keywords=keywords,
                    block_lines=block_lines,
                    charge=charge,
                    multiplicity=multiplicity,
                    symbols=symbols,
                    coordinates=coordinates,
                    npara=npara,
                    maxcore_mb=maxcore_mb,
                    retry_attempt=attempt_index,
                )
            except AmespBaselineError as exc:
                last_error = exc
                exit_code = exc.details.get("exit_code")
                if attempt_index >= len(run_configs) or not _is_native_crash(exit_code):
                    raise
        if last_error is not None:
            raise last_error
        raise AmespBaselineError("runtime_failed", f"Amesp step {step_id} did not run.")

    def _run_step_once(
        self,
        *,
        step_id: str,
        workdir: Path,
        aip_path: Path,
        aop_path: Path,
        stdout_path: Path,
        stderr_path: Path,
        keywords: list[str],
        block_lines: list[tuple[str, list[str]]],
        charge: int,
        multiplicity: int,
        symbols: list[str],
        coordinates: list[list[float]],
        npara: int,
        maxcore_mb: int,
        retry_attempt: int,
    ) -> AmespStepResult:
        _write_amesp_input(
            aip_path=aip_path,
            keywords=keywords,
            charge=charge,
            multiplicity=multiplicity,
            symbols=symbols,
            coordinates=coordinates,
            block_lines=block_lines,
            npara=npara,
            maxcore_mb=maxcore_mb,
        )
        env = dict(os.environ)
        env.setdefault("KMP_STACKSIZE", "1g")
        bin_dir = str(self.amesp_bin.parent)
        env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                [str(self.amesp_bin), aip_path.name, aop_path.name],
                cwd=str(workdir),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout_path.write_text(exc.stdout or "", encoding="utf-8")
            stderr_path.write_text(exc.stderr or "", encoding="utf-8")
            raise AmespBaselineError(
                "runtime_failed",
                f"Amesp step {step_id} timed out after {self.timeout_seconds} seconds.",
                file_paths={
                    "aip": str(aip_path),
                    "aop": str(aop_path),
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                },
            ) from exc
        elapsed = round(time.perf_counter() - start, 4)
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        terminated = False
        if aop_path.exists():
            terminated = "Normal termination of Amesp!" in aop_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        outcome = AmespStepResult(
            step_id=step_id,
            aip_path=aip_path,
            aop_path=aop_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            exit_code=completed.returncode,
            terminated_normally=terminated,
            elapsed_seconds=elapsed,
        )
        if completed.returncode != 0 or not terminated:
            raise AmespBaselineError(
                "runtime_failed",
                f"Amesp step {step_id} did not terminate normally.",
                file_paths=_step_file_paths(outcome),
                details={
                    "exit_code": completed.returncode,
                    "npara": npara,
                    "maxcore_mb": maxcore_mb,
                    "retry_attempt": retry_attempt,
                },
            )
        return outcome

    def _resolve_amesp_bin(self, amesp_bin: Path | None) -> Path:
        if amesp_bin is not None:
            return amesp_bin.expanduser().resolve()
        env_bin = os.environ.get("MECHCAL_AMESP_BIN")
        if env_bin:
            return Path(env_bin).expanduser().resolve()
        return (
            Path(__file__).resolve().parents[2]
            / "third_party"
            / "Amesp"
            / "Bin"
            / "amesp"
        ).resolve()


def _step_file_paths(step: AmespStepResult) -> dict[str, str]:
    return {
        "aip": str(step.aip_path),
        "aop": str(step.aop_path),
        "stdout": str(step.stdout_path),
        "stderr": str(step.stderr_path),
    }


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _is_native_crash(exit_code: object) -> bool:
    return isinstance(exit_code, int) and exit_code < 0


def _write_amesp_input(
    *,
    aip_path: Path,
    keywords: list[str],
    charge: int,
    multiplicity: int,
    symbols: list[str],
    coordinates: list[list[float]],
    block_lines: list[tuple[str, list[str]]],
    npara: int,
    maxcore_mb: int,
) -> None:
    lines = [f"% npara {npara}", f"% maxcore {maxcore_mb}", f"! {' '.join(keywords)}"]
    for block_name, block_body in block_lines:
        lines.append(f">{block_name}")
        lines.extend(f"  {line}" for line in block_body)
        lines.append("end")
    lines.append(f">xyz {charge} {multiplicity}")
    for symbol, coordinate in zip(symbols, coordinates, strict=True):
        x, y, z = coordinate
        lines.append(f" {symbol:<2} {x: .8f} {y: .8f} {z: .8f}")
    lines.append("end")
    aip_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_xyz_symbols_and_coordinates(xyz_path: Path) -> tuple[list[str], list[list[float]]]:
    lines = xyz_path.read_text(encoding="utf-8", errors="replace").splitlines()
    symbols: list[str] = []
    coordinates: list[list[float]] = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) != 4:
            continue
        try:
            coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            continue
        symbols.append(parts[0])
    return symbols, coordinates


def _parse_final_geometry(text: str) -> tuple[list[str], list[list[float]]]:
    marker = "Final Geometry(angstroms):"
    index = text.rfind(marker)
    if index < 0:
        return [], []
    lines = text[index + len(marker):].splitlines()
    atom_count: int | None = None
    atom_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if atom_count is None:
            if stripped.isdigit():
                atom_count = int(stripped)
            continue
        atom_lines.append(stripped)
        if len(atom_lines) >= atom_count:
            break
    symbols: list[str] = []
    coordinates: list[list[float]] = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) != 4:
            return [], []
        symbols.append(parts[0])
        coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, coordinates


def _write_xyz(
    xyz_path: Path,
    *,
    label: str,
    symbols: list[str],
    coordinates: list[list[float]],
) -> None:
    lines = [str(len(symbols)), label]
    for symbol, coordinate in zip(symbols, coordinates, strict=True):
        x, y, z = coordinate
        lines.append(f"{symbol:<2} {x: .8f} {y: .8f} {z: .8f}")
    xyz_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_final_energy(text: str) -> float | None:
    matches = re.findall(r"Final Energy:\s*([-+]?\d+\.\d+)", text)
    if matches:
        return float(matches[-1])
    matches = re.findall(r"Current Energy\s*:\s*([-+]?\d+\.\d+)", text)
    return float(matches[-1]) if matches else None


def _parse_excited_states(
    text: str,
    *,
    reference_energy_hartree: float | None,
) -> list[dict[str, object]]:
    state_matches = re.findall(
        r"State\s+(\d+)\s*:\s*E\s*=\s*([-+]?\d+\.\d+)\s+eV",
        text,
    )
    td_matches = re.findall(
        r"E\(TD\)\s*=\s*([-+]?\d+\.\d+)\s+<S\*\*2>=\s*([-+]?\d+\.\d+)\s+f=\s*([-+]?\d+\.\d+)",
        text,
    )
    states: list[dict[str, object]] = []
    for offset, (state_index, excitation_energy_ev) in enumerate(state_matches):
        state: dict[str, object] = {
            "state_index": int(state_index),
            "excitation_energy_ev": float(excitation_energy_ev),
        }
        if offset < len(td_matches):
            total_energy, spin_square, oscillator = td_matches[offset]
            state.update(
                {
                    "total_energy_hartree": float(total_energy),
                    "spin_square": float(spin_square),
                    "oscillator_strength": float(oscillator),
                }
            )
        else:
            state["oscillator_strength"] = 0.0
        states.append(state)
    if states or reference_energy_hartree is None:
        return states
    for index, (energy, spin_square, oscillator) in enumerate(td_matches, start=1):
        total_energy = float(energy)
        states.append(
            {
                "state_index": index,
                "total_energy_hartree": total_energy,
                "spin_square": float(spin_square),
                "oscillator_strength": float(oscillator),
                "excitation_energy_ev": round(
                    (total_energy - reference_energy_hartree) * 27.211386245988,
                    6,
                ),
            }
        )
    return states


def _safe_label(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)[:80] or "mechcal"
