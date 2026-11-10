"""Route-local arguments shared by Planner, workers and computation adapters."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator
from mechcal.schemas.contracts import field_constraints

Atom = Annotated[int, Field(strict=True, ge=0)]
Angle = Annotated[float, Field(ge=-180, le=180, allow_inf_nan=False)]


class SourceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_artifact_id: str | None = Field(default=None, min_length=1)


class StateArgs(SourceArgs):
    s1_nstates: int | None = Field(default=None, ge=1, le=10)
    td_tout: int | None = Field(default=None, ge=1, le=10)


class GeometryMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    measurement_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,47}$")
    kind: Literal["distance", "angle", "dihedral"]
    atom_indices: list[Atom] = Field(min_length=2, max_length=4,
        description="Zero-based indices in the selected source XYZ, not SMILES atom order.")

    @model_validator(mode="after")
    def index_count(self):
        count = {"distance": 2, "angle": 3, "dihedral": 4}[self.kind]
        if len(self.atom_indices) != count or len(set(self.atom_indices)) != count:
            raise ValueError(f"{self.kind} requires {count} distinct atom indices.")
        return self

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        return field_constraints(handler.resolve_ref_schema(handler(core_schema)), [
            {"kind": {"type": "string", "const": kind},
             "atom_indices": {"minItems": count, "maxItems": count, "uniqueItems": True}}
            for kind, count in {"distance": 2, "angle": 3, "dihedral": 4}.items()
        ])


class GeometryDescriptorArgs(SourceArgs):
    measurements: list[GeometryMeasurement] = Field(default_factory=list, max_length=12,
        description="Optional selected distance/angle/dihedral measurements. Omit for global extents and atom map only.")

    @model_validator(mode="after")
    def distinct_ids(self):
        ids = [m.measurement_id for m in self.measurements]
        if len(ids) != len(set(ids)):
            raise ValueError("measurement_id must be unique within the request.")
        return self


class HoleElectronArgs(SourceArgs):
    state_root: int = Field(default=1, ge=1, le=10)
    spatial_grid: Literal["medium", "high"] = "high"
    max_wall_seconds: int = Field(default=300, ge=10, le=300)
    fragments: dict[str, list[Atom]] | None = Field(default=None,
        description="Optional named disjoint groups in the selected calculation atom map. Reports sums of heavy-atom contributions, excluding hydrogen without renormalization.")

    @model_validator(mode="after")
    def disjoint_fragments(self):
        if self.fragments is not None:
            indices = [i for group in self.fragments.values() for i in group]
            if (not self.fragments or len(self.fragments) > 12
                    or any(not name.strip() or not group for name, group in self.fragments.items())
                    or len(indices) != len(set(indices))):
                raise ValueError("Use 1-12 named, nonempty, disjoint atom groups.")
        return self


class GroundGeometryArgs(SourceArgs):
    max_opt_cycles: int = Field(default=240, ge=1, le=300)
    max_wall_seconds: float = Field(default=240.0, ge=10.0, le=300.0)


class TorsionArgs(StateArgs):
    max_members: int | None = Field(default=None, ge=2, le=6)
    dihedral_id: str | None = Field(default=None, min_length=1)
    dihedral_atoms: list[Atom] | None = Field(default=None, min_length=4, max_length=4,
        description="Exactly four distinct indices. Mutually exclusive with dihedral_id.")
    angles: list[Angle] | None = Field(default=None, min_length=2, max_length=6,
        description="Distinct angles modulo 360; -180 and 180 are the same point. If max_members is set, it must equal the angle count.")

    @model_validator(mode="after")
    def coherent_selection(self):
        if self.dihedral_id is not None and self.dihedral_atoms is not None:
            raise ValueError("Select a dihedral_id OR zero-based dihedral_atoms, not both.")
        if self.dihedral_atoms is not None and len(set(self.dihedral_atoms)) != 4:
            raise ValueError("Four distinct atom indices are required.")
        if self.angles is not None:
            if len({(a + 180) % 360 for a in self.angles}) != len(self.angles):
                raise ValueError("Angles must represent distinct geometries.")
            if self.max_members is not None and self.max_members != len(self.angles):
                raise ValueError("max_members must equal the number of explicit angles.")
        return self


class RelaxationArgs(SourceArgs):
    """Initial rollout: gas-phase aTB1/TDA, closed-shell singlet roots only."""

    state_root: int = Field(default=1, ge=1, le=5)
    s1_nstates: int = Field(default=3, ge=1, le=10)
    td_tout: int = Field(default=3, ge=1, le=10)
    max_opt_cycles: int = Field(default=60, ge=1, le=200)

    @model_validator(mode="after")
    def root_in_window(self):
        if self.state_root > self.s1_nstates:
            raise ValueError("state_root must not exceed s1_nstates.")
        return self


class RelaxedScanArgs(SourceArgs):
    coordinate: Literal["bond", "dihedral"]
    atom_indices: list[Atom] = Field(min_length=2, max_length=4)
    values: list[Annotated[float, Field(allow_inf_nan=False)]] = Field(min_length=2, max_length=6)
    state_root: Literal[0, 1] = 0
    s1_nstates: int = Field(default=3, ge=1, le=10)
    td_tout: int = Field(default=3, ge=1, le=10)
    max_opt_cycles: int = Field(default=60, ge=1, le=150)
    fmax_ev_per_angstrom: float = Field(default=0.05, ge=0.01, le=0.1)
    max_wall_seconds: int = Field(default=300, ge=5, le=1800)

    @field_validator("state_root", mode="before")
    @classmethod
    def integer_root(cls, value):
        if type(value) is not int:
            raise ValueError("state_root must be integer 0 or 1.")
        return value

    @model_validator(mode="after")
    def coordinate_values(self):
        count = 2 if self.coordinate == "bond" else 4
        if len(self.atom_indices) != count or len(set(self.atom_indices)) != count:
            raise ValueError(f"{self.coordinate} requires {count} distinct atom indices.")
        values = self.values
        if self.coordinate == "bond":
            if any(not 0.7 <= v <= 3.0 for v in values):
                raise ValueError("Bond lengths must be in [0.7,3.0] angstrom.")
        else:
            if any(not -180 <= v <= 180 for v in values):
                raise ValueError("Dihedrals must be in [-180,180] degrees.")
            values = [(v + 180) % 360 for v in values]
        if len(set(values)) != len(values):
            raise ValueError("Scan points must be distinct.")
        if isinstance(self.state_root, bool):
            raise ValueError("state_root must be integer 0 or 1, not bool.")
        return self

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        return field_constraints(handler.resolve_ref_schema(handler(core_schema)), [
            {"coordinate": {"type": "string", "const": kind},
             "atom_indices": {"minItems": count, "maxItems": count, "uniqueItems": True},
             "values": {"items": {"type": "number", "minimum": low, "maximum": high}, "uniqueItems": True}}
            for kind, count, low, high in (("bond", 2, .7, 3.0), ("dihedral", 4, -180, 180))
        ])


class GeometrySelectionArgs(SourceArgs):
    source_artifact_id: str = Field(min_length=1)
    member_id: str = Field(min_length=1)


class ConformerArgs(StateArgs):
    max_members: int | None = Field(default=None, ge=2, le=6)
    conformer_ids: list[str] | None = Field(default=None, min_length=2, max_length=6,
        description="Distinct registered conformer IDs. If max_members is set, it must equal the selected count.")

    @model_validator(mode="after")
    def coherent_selection(self):
        if self.conformer_ids is not None:
            if len(set(self.conformer_ids)) != len(self.conformer_ids):
                raise ValueError("Conformer IDs must be distinct.")
            if self.max_members is not None and self.max_members != len(self.conformer_ids):
                raise ValueError("max_members must equal the number of selected conformers.")
        return self


class SolventArgs(StateArgs):
    solvents: (
        list[
            Literal[
                "water",
                "toluene",
                "acetonitrile",
                "methanol",
                "ethanol",
                "benzene",
                "acetone",
                "chloroform",
            ]
        ]
        | None
    ) = Field(default=None, min_length=2, max_length=4)

    @model_validator(mode="after")
    def distinct_solvents(self):
        if self.solvents is not None and len(set(self.solvents)) != len(self.solvents):
            raise ValueError("Solvents must be distinct.")
        return self


class ChargeArgs(SourceArgs):
    charge_schemes: list[Literal["mulliken", "lowdin", "hirshfeld", "cm5"]] | None = Field(
        default=None, min_length=1, max_length=4
    )

    @model_validator(mode="after")
    def distinct_schemes(self):
        if self.charge_schemes is not None and len(set(self.charge_schemes)) != len(
            self.charge_schemes
        ):
            raise ValueError("Charge schemes must be distinct.")
        return self


class FrontierArgs(SourceArgs):
    donor_atoms: list[Atom] | None = Field(default=None, min_length=1)
    acceptor_atoms: list[Atom] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def paired_fragments(self):
        if (self.donor_atoms is None) != (self.acceptor_atoms is None):
            raise ValueError("Provide both donor_atoms and acceptor_atoms.")
        if self.donor_atoms is not None:
            if set(self.donor_atoms) & set(self.acceptor_atoms):
                raise ValueError("Donor and acceptor atom sets must be disjoint.")
            for atoms in (self.donor_atoms, self.acceptor_atoms):
                if len(set(atoms)) != len(atoms):
                    raise ValueError("Fragment atom indices must be unique.")
        return self


class MacroArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    focus_tags: list[str] | None = None


class PlannedTorsionArgs(TorsionArgs):
    """The autonomous Planner selects the coordinate instead of inheriting a default."""

    model_config = ConfigDict(extra="forbid", strict=True, json_schema_extra={"anyOf": [
        {"required": ["dihedral_id"], "properties": {"dihedral_id": {"type": "string"}}},
        {"required": ["dihedral_atoms"], "properties": {"dihedral_atoms": {"type": "array"}}},
    ]})
    angles: list[Angle] = Field(min_length=2, max_length=6,
        description=TorsionArgs.model_fields["angles"].description)

    @model_validator(mode="after")
    def explicit_coordinate(self):
        if self.dihedral_id is None and self.dihedral_atoms is None:
            raise ValueError("Select dihedral_id or dihedral_atoms and angles. "
                             "list_rotatable_dihedrals returns coordinate IDs and a zero-based atom map. "
                             "Choose the coordinate relevant to the question; no coordinate will be inserted.")
        return self


class PlannedFrontierArgs(FrontierArgs):
    donor_atoms: list[Atom] = Field(min_length=1, description="Explicit donor fragment in the selected geometry's zero-based atom map.")
    acceptor_atoms: list[Atom] = Field(min_length=1, description="Explicit disjoint acceptor fragment in the same atom map.")


def planner_argument_model(route: str) -> type[BaseModel] | None:
    if route in {"run_torsion_snapshots", "run_torsion_brightness_coupling_scan"}:
        return PlannedTorsionArgs
    if route == "run_frontier_orbital_partition":
        return PlannedFrontierArgs
    return None


STATE_ROUTES = {
    "run_baseline_bundle",
    "run_bright_dark_state_ordering",
    "run_targeted_state_characterization",
    "run_targeted_transition_dipole_analysis",
    "run_targeted_charge_redistribution_analysis",
    "run_ris_state_characterization",
    "run_targeted_localized_orbital_analysis",
    "run_targeted_natural_orbital_analysis",
}


def argument_model(route: str, owner: str = "microscopic") -> type[BaseModel]:
    from mechcal.schemas.external_qc import EXTERNAL_ARGUMENTS
    if owner == "microscopic" and route in EXTERNAL_ARGUMENTS:
        return EXTERNAL_ARGUMENTS[route]
    if owner == "macro":
        return MacroArgs
    if route in {"run_torsion_snapshots", "run_torsion_brightness_coupling_scan"}:
        return TorsionArgs
    if route in {"run_conformer_bundle", "run_conformer_state_probe"}:
        return ConformerArgs
    return {
        "extract_geometry_descriptors_from_bundle": GeometryDescriptorArgs,
        "analyze_hole_electron_from_bundle": HoleElectronArgs,
        "optimize_ground_state_geometry": GroundGeometryArgs,
        "run_relaxed_coordinate_scan": RelaxedScanArgs,
        "select_computed_geometry": GeometrySelectionArgs,
        "run_excited_state_relaxation": RelaxationArgs,
        "run_solvation_polarity_proxy": SolventArgs,
        "run_charge_population_panel": ChargeArgs,
        "run_frontier_orbital_partition": FrontierArgs,
    }.get(route, StateArgs if route in STATE_ROUTES else SourceArgs)


def supplied_argument_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Omission selects defaults; explicit null is not a supplied argument."""
    schema = model.model_json_schema()
    for branch in [schema, *schema.get("anyOf", [])]:
        _omit_null_defaults(branch)
    return schema


def _omit_null_defaults(schema):
    for prop in schema.get("properties", {}).values():
        choices = prop.get("anyOf", [])
        non_null = [choice for choice in choices if choice.get("type") != "null"]
        if choices and len(non_null) != len(choices):
            prop.pop("anyOf")
            prop.update(non_null[0] if len(non_null) == 1 else {"anyOf": non_null})
        if prop.get("default", object()) is None:
            prop.pop("default")


def validate_tool_args(
    route: str, args: dict[str, Any], owner: str = "microscopic"
) -> dict[str, Any]:
    # Omitted means default; explicit null is not a request to silently use a default.
    if any(value is None for value in args.values()):
        raise ValueError("Omit optional arguments instead of passing null.")
    return argument_model(route, owner).model_validate(args).model_dump(exclude_none=True)


def merge_tool_args(request, execution_plan) -> dict[str, Any]:
    worker = execution_plan.tool_args if execution_plan is not None else {}
    for key, value in request.tool_args.items():
        if key in worker and worker[key] != value:
            raise ValueError(f"Worker must not override Planner parameter {key}.")
    return validate_tool_args(request.route, {**worker, **request.tool_args}, request.agent_name)
