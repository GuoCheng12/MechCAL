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


class CISPathArgs(QCArgs):
    coordinate: Literal["bond", "dihedral"]
    atom_indices: list[Annotated[int, Field(strict=True, ge=0)]] = Field(min_length=2, max_length=4)
    values: list[Annotated[float, Field(allow_inf_nan=False)]] = Field(min_length=2, max_length=4)
    state_root: int = Field(default=1, ge=1, le=3,
        description="Initial energy-ordered root among three CIS singlets; subsequent points use state overlap.")

    @model_validator(mode="after")
    def valid_coordinate(self):
        n = 2 if self.coordinate == "bond" else 4
        low, high = (.7, 2.5) if n == 2 else (-180, 180)
        normalized = self.values if n == 2 else [(v+180)%360 for v in self.values]
        if len(self.atom_indices) != n or len(set(self.atom_indices)) != n:
            raise ValueError("Coordinate atom count/identity is invalid.")
        if len(set(normalized)) != len(normalized) or any(not low <= v <= high for v in self.values):
            raise ValueError("Coordinate values are repeated or outside the supported range.")
        return self


class AmespSOCArgs(QCArgs):
    max_wall_seconds: int = Field(default=300, ge=5, le=1800)
    nstates_per_spin: int = Field(default=3, ge=2, le=5)
    basis: Literal["def2-svp", "def2-tzvp"] = "def2-svp"
    integral_mode: Literal["conventional", "ricosx"] = Field(default="conventional",
        description="Explicit integral choice. RICOSX uses the Amesp RIJ/COSX approximation and automatic auxiliary basis; compare against conventional controls.")


class ConformerSamplingArgs(QCArgs):
    max_members: int = Field(default=4, ge=1, le=12,
        description="Maximum returned checked conformers, NOT a bound on CREST sampling work.")
    energy_window_kcal_mol: float = Field(default=6., ge=1., le=12., allow_inf_nan=False)
    search_level: Literal["quick", "squick"] = "squick"
    md_length_ps: float | None = Field(default=None, ge=.5, le=100., allow_inf_nan=False,
        description="Explicit CREST --mdlen in ps per metadynamics trajectory. Omit to retain CREST's automatic length, which can be tens of ps for flexible molecules. Short lengths reduce coverage; neither this nor max_members guarantees completion within max_wall_seconds.")


class ContactMotionArgs(QCArgs):
    coordinate_mode: Literal["bonded_dihedral", "fragment_azimuth"] = Field(default="bonded_dihedral",
        description="bonded_dihedral uses four consecutively bonded atoms around a single bond. fragment_azimuth uses off-axis reference anchors on opposite graph sides of a non-ring single or dative central bond, including near-linear coordination axes. It is a rigid relative orientation, not a bond torsional barrier.")
    atom_indices: list[Annotated[int, Field(strict=True, ge=0)]] = Field(min_length=4, max_length=4,
        description="[reference_A, axis_B, axis_C, reference_D] within fragment A. The C/D graph side rotates; fragment B stays fixed. In fragment_azimuth the reference anchors need not be adjacent to the axis, but must lie off-axis and on opposite sides.")
    angles_degrees: list[Annotated[float, Field(ge=-180, le=180, allow_inf_nan=False)]] = Field(min_length=2, max_length=6)

    @model_validator(mode="after")
    def distinct(self):
        if len(set(self.atom_indices)) != 4 or len({(v+180)%360 for v in self.angles_degrees}) != len(self.angles_degrees):
            raise ValueError("Use distinct torsion atoms and distinct periodic angles.")
        return self


class BroadSpinArgs(QCArgs):
    max_wall_seconds: int = Field(default=300, ge=5, le=1800)
    nstates_per_spin: int = Field(default=3, ge=2, le=6)
    functional: Literal["pbe", "pbe0"] = "pbe"
    basis: Literal["def2-svp", "def2-tzvp"] = "def2-svp"
    grid_level: int = Field(default=1, ge=1, le=3)
    scf_solver: Literal["diis", "newton"] = "diis"
    fragment_atom_indices: list[list[Annotated[int, Field(strict=True, ge=0)]]] = Field(default_factory=list,
        description="Optional disjoint partition of all atoms for Lowdin hole/electron populations. Empty returns per-atom populations.")

    @model_validator(mode="after")
    def valid_partition(self):
        indices = [i for group in self.fragment_atom_indices for i in group]
        if any(not group for group in self.fragment_atom_indices) or len(indices) != len(set(indices)):
            raise ValueError("Fragments must be nonempty and disjoint.")
        return self


EXTERNAL_ARGUMENTS = {
    "estimate_soc_amesp": AmespSOCArgs,
    "scan_cis_excitation_coordinate": CISPathArgs,
    "sample_conformers_crest": ConformerSamplingArgs,
    "compare_contact_torsion_response": ContactMotionArgs,
    "compare_spin_states_pyscf": BroadSpinArgs,
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
