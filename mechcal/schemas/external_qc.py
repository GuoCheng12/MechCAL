"""Bounded arguments for opt-in scientific backend trials."""

from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class QCArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_artifact_id: str = Field(min_length=1)
    max_wall_seconds: int = Field(default=180, ge=5, le=600)


class DimerArgs(QCArgs):
    partner_artifact_id: str = Field(min_length=1, description="Same ID for a homodimer; another registered structure for a heterodimer.")
    max_candidates: int = Field(default=1, ge=1, le=4,
        description="Maximum optimized poses to return; search may yield fewer distinct poses.")
    max_opt_cycles: int = Field(default=150, ge=20, le=600,
        description="GFN2 optimization iterations per candidate; whole-route timeout still applies.")


class ContactScanArgs(QCArgs):
    atom_indices: list[Annotated[int, Field(strict=True, ge=0)]] = Field(min_length=2, max_length=2,
        description="One atom from each fragment of a registered modeled dimer, using its zero-based map.")
    distances_angstrom: list[Annotated[float, Field(ge=1.5, le=12, allow_inf_nan=False)]] = Field(min_length=2, max_length=6)

    @model_validator(mode="after")
    def distinct_points(self):
        if len(set(self.atom_indices)) != 2 or len(set(self.distances_angstrom)) != len(self.distances_angstrom):
            raise ValueError("Select distinct atoms and distinct separation values.")
        return self


class ExcitedArgs(QCArgs):
    state_root: int = Field(default=1, ge=1, le=3)
    nstates: int = Field(default=6, ge=2, le=10)

    @model_validator(mode="after")
    def state_window(self):
        if self.nstates < self.state_root + 2:
            raise ValueError("Keep at least two extra states above the selected root.")
        return self


class SpinComparisonArgs(QCArgs):
    nstates_per_spin: int = Field(default=3, ge=3, le=6,
        description="Number of vertical roots in each spin manifold at the unchanged source geometry. Request at least three; single-root ARPACK failed the numerical acceptance probe.")


class ExcitedScanArgs(ExcitedArgs):
    coordinate: Literal["bond", "dihedral"]
    atom_indices: list[Annotated[int, Field(strict=True, ge=0)]] = Field(min_length=2, max_length=4)
    values: list[Annotated[float, Field(allow_inf_nan=False)]] = Field(min_length=2, max_length=4)
    max_opt_cycles: int = Field(default=30, ge=1, le=60)
    fmax_ev_per_angstrom: float = Field(default=0.05, ge=0.02, le=0.1)

    @model_validator(mode="after")
    def valid_coordinate(self):
        n = 2 if self.coordinate == "bond" else 4
        if len(self.atom_indices) != n or len(set(self.atom_indices)) != n:
            raise ValueError("Coordinate requires distinct atom indices of the correct count.")
        low, high = (0.7, 2.5) if n == 2 else (-180, 180)
        values = self.values if n == 2 else [(v + 180) % 360 for v in self.values]
        if len(set(values)) != len(values) or any(not low <= v <= high for v in self.values):
            raise ValueError("Scan values must be distinct and within the coordinate range.")
        return self


EXTERNAL_ARGUMENTS = {
    "estimate_vertical_ip_ea": QCArgs,
    "analyze_ground_state_vibrations": QCArgs,
    "search_dimer_contacts": DimerArgs,
    "scan_intermolecular_distance": ContactScanArgs,
    "evaluate_excited_state_gradient": ExcitedArgs,
    "compare_singlet_triplet_states": SpinComparisonArgs,
    "evaluate_cis_s1_gradient": QCArgs,
    "evaluate_spin_orbit_couplings": QCArgs,
    "scan_excited_state_coordinate": ExcitedScanArgs,
}
