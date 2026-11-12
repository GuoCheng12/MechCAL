"""Recipes: named tool chains with validated defaults, executed through the recorded gateway.

A recipe is a fast path for a standard observable. Every step is an ordinary recorded
gateway call; a failed step stops the recipe and is returned as-is. Scientific choices the
recipe cannot know (which state, which dihedral, which fragments) are optional arguments.
Geometry handling: when RDKit force-field preparation succeeded, its prepared structure is
used directly; otherwise (e.g. metal complexes) the aTB1 ground-state optimisation runs
first. Either way the geometry source is part of the recorded artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from mechcal.eval.tool_gateway import ToolGateway

DEFAULT_NSTATES = 10
TD_TOUT = 10  # >= 4 is required for later hole-electron analysis; 10 keeps transitions complete
# Fallback time limit for the PySCF singlet/triplet route when no cost estimate exists (0003: 923 s).
# With a cost model the limit is 1.5x the estimate, within 600-1800 s.
PYSCF_SPIN_WALL = 1200


def _spin_wall(ctx, args: dict) -> int:
    expected = ctx.estimate("microscopic.compare_spin_states_pyscf", args) if ctx.estimate else None
    return int(min(1800, max(600, 1.5 * expected))) if expected else PYSCF_SPIN_WALL


@dataclass
class RecipeContext:
    gateway: ToolGateway
    cache: dict = field(default_factory=dict)   # reusable artifact ids (geometry, bundle, dihedrals)
    source: str | None = None                   # explicit input geometry for the current recipe call
    estimate: Callable | None = None            # (tool, args) -> expected seconds, from the cost model


@dataclass
class RecipeResult:
    recipe: str
    status: str                                 # success | failed | rejected
    observations: list[dict]
    note: str = ""
    geometry: str | None = None                 # input geometry the recipe used

    def to_json(self) -> dict:
        return {"recipe": self.recipe, "status": self.status, "note": self.note, "geometry": self.geometry,
                "observations": self.observations}


def _step(ctx: RecipeContext, steps: list, tool: str, args: dict) -> dict | None:
    obs = ctx.gateway.call(tool, args)
    steps.append(obs)
    return obs if obs.get("status") in {"success", "partial"} else None


def _latest(ctx: RecipeContext, kind: str, prefix: str = "") -> str | None:
    """Latest artifact of a kind already produced in this run (by a recipe, tool or Worker)."""
    found = [a.artifact_id for a in ctx.gateway.case_run.artifact_manifest.artifacts
             if a.kind == kind and a.status in {"available", "partial"} and a.artifact_id.startswith(prefix)]
    return found[-1] if found else None


def geometry(ctx: RecipeContext, steps: list) -> str | None:
    geom = _geometry(ctx, steps)
    if geom:
        ctx.cache["last_geometry"] = geom
    return geom


def _geometry(ctx: RecipeContext, steps: list) -> str | None:
    gw = ctx.gateway
    if ctx.source:                                  # explicit geometry (e.g. fixed by the Planner)
        kind = next((a.kind for a in gw.case_run.artifact_manifest.artifacts if a.artifact_id == ctx.source), None)
        if kind == "prepared_structure":
            ctx.cache["geometry_source"] = "given"
            return ctx.source
        if kind != "initial_structure":
            return None
        derived = [a.artifact_id for a in gw.case_run.artifact_manifest.artifacts
                   if a.kind == "prepared_structure" and a.artifact_id.startswith("geometry:")
                   and gw.derived_from(a.artifact_id, ctx.source)]
        if derived:
            ctx.cache["geometry_source"] = "atb1_optimized"
            return derived[-1]
        obs = _step(ctx, steps, "microscopic.optimize_ground_state_geometry",
                    {"source_artifact_id": ctx.source, "max_opt_cycles": 2000, "max_wall_seconds": 300})
        if not obs or not obs["reusable_artifacts"]:
            return None
        ctx.cache["geometry_source"] = "atb1_optimized"
        return obs["reusable_artifacts"][0]
    if "geometry" in ctx.cache:
        return ctx.cache["geometry"]
    optimized = _latest(ctx, "prepared_structure", "geometry:")
    if optimized:
        ctx.cache.update(geometry=optimized, geometry_source="atb1_optimized")
        return optimized
    kind = next(a.kind for a in gw.case_run.artifact_manifest.artifacts if a.artifact_id == gw.prepared_artifact_id)
    if kind == "prepared_structure":
        ctx.cache["geometry"] = gw.prepared_artifact_id
        ctx.cache["geometry_source"] = "rdkit_forcefield"
        return gw.prepared_artifact_id
    obs = _step(ctx, steps, "microscopic.optimize_ground_state_geometry",
                {"source_artifact_id": gw.prepared_artifact_id, "max_opt_cycles": 2000, "max_wall_seconds": 300})
    if not obs or not obs["reusable_artifacts"]:
        return None
    ctx.cache["geometry"] = obs["reusable_artifacts"][0]
    ctx.cache["geometry_source"] = "atb1_optimized"
    return ctx.cache["geometry"]


def _bundle(ctx: RecipeContext, steps: list, nstates: int) -> str | None:
    geom = geometry(ctx, steps)
    if geom is None:
        return None
    gw = ctx.gateway
    existing = [a.artifact_id for a in gw.case_run.artifact_manifest.artifacts      # same geometry only
                if a.kind == "amesp_baseline_bundle" and a.status in {"available", "partial"}
                and gw.derived_from(a.artifact_id, geom)]
    if existing:
        return existing[-1]
    obs = _step(ctx, steps, "microscopic.run_baseline_bundle",
                {"source_artifact_id": geom, "s1_nstates": nstates, "td_tout": TD_TOUT})
    if not obs or not obs["reusable_artifacts"]:
        return None
    return obs["reusable_artifacts"][0]


def excited_states(ctx, steps, *, nstates: int = DEFAULT_NSTATES, **_):
    return _bundle(ctx, steps, int(nstates)) is not None


def charge_transfer(ctx, steps, *, state_root: int = 1, nstates: int = DEFAULT_NSTATES, fragments: dict | None = None, **_):
    bundle = _bundle(ctx, steps, int(nstates))
    if bundle is None:
        return False
    args = {"source_artifact_id": bundle, "state_root": int(state_root), "spatial_grid": "high", "max_wall_seconds": 900}
    if fragments:
        args["fragments"] = fragments
    return _step(ctx, steps, "microscopic.analyze_hole_electron_from_bundle", args) is not None


def torsion_response(ctx, steps, *, dihedral_id: str | None = None, dihedral_atoms: list | None = None,
                     angles: list | None = None, nstates: int = 6, **_):
    geom = geometry(ctx, steps)
    if geom is None:
        return False
    if dihedral_id is None and dihedral_atoms is None:
        listing = _step(ctx, steps, "microscopic.list_rotatable_dihedrals", {"source_artifact_id": geom})
        if listing is None:
            return False
        ids = [v for p, v in listing["labels"].items() if p.endswith("/dihedral_id")]
        if not ids:
            return False
        dihedral_id = ids[0]
    args = {"source_artifact_id": geom, "s1_nstates": int(nstates), "td_tout": TD_TOUT,
            **({"dihedral_id": dihedral_id} if dihedral_id is not None else {"dihedral_atoms": list(dihedral_atoms)})}
    if angles:
        args["angles"] = angles
    return _step(ctx, steps, "microscopic.run_torsion_brightness_coupling_scan", args) is not None


def spin_states(ctx, steps, *, nstates_per_spin: int = 3, functional: str | None = None, basis: str | None = None, **_):
    """TD-DFTB + SOC for neutral CHNO; PySCF DFT/TDA otherwise, or whenever a functional or basis is given."""
    geom = geometry(ctx, steps)
    if geom is None:
        return False
    elements = {a.GetSymbol() for a in _mol(ctx).GetAtoms()}
    if elements <= {"C", "H", "N", "O"} and _neutral(ctx) and functional is None and basis is None:
        ok = _step(ctx, steps, "microscopic.compare_singlet_triplet_states",
                   {"source_artifact_id": geom, "nstates_per_spin": int(nstates_per_spin)}) is not None
        _step(ctx, steps, "microscopic.evaluate_spin_orbit_couplings", {"source_artifact_id": geom})
        return ok
    args = {"source_artifact_id": geom, "nstates_per_spin": int(nstates_per_spin), "functional": functional or "pbe",
            "basis": basis or "def2-svp"}
    return _step(ctx, steps, "microscopic.compare_spin_states_pyscf", {**args, "max_wall_seconds": _spin_wall(ctx, args)}) is not None


def redox_levels(ctx, steps, **_):
    geom = geometry(ctx, steps)
    return geom is not None and _step(ctx, steps, "microscopic.estimate_vertical_ip_ea",
                                      {"source_artifact_id": geom}) is not None


def vibrations(ctx, steps, **_):
    geom = geometry(ctx, steps)
    return geom is not None and _step(ctx, steps, "microscopic.analyze_ground_state_vibrations",
                                      {"source_artifact_id": geom, "max_wall_seconds": 600}) is not None


def dimer_contacts(ctx, steps, *, max_candidates: int = 2, **_):
    geom = geometry(ctx, steps)
    return geom is not None and _step(ctx, steps, "microscopic.search_dimer_contacts",
                                      {"source_artifact_id": geom, "partner_artifact_id": geom,
                                       "max_candidates": int(max_candidates), "max_wall_seconds": 600}) is not None


def dimer_excited_states(ctx, steps, *, nstates: int = DEFAULT_NSTATES, state_root: int = 1, fragments: dict | None = None, **_):
    """Excited states of a modeled dimer and how the chosen state is shared between the two molecules."""
    gw = ctx.gateway
    records = {a.artifact_id: a for a in gw.case_run.artifact_manifest.artifacts}
    dimer = ctx.source if ctx.source and (records.get(ctx.source) and (records[ctx.source].metadata or {}).get("modeled_dimer")) else None
    if dimer is None:
        geom = geometry(ctx, steps)
        if geom is None:
            return False
        search = _step(ctx, steps, "microscopic.search_dimer_contacts",
                       {"source_artifact_id": geom, "partner_artifact_id": geom, "max_candidates": 1, "max_wall_seconds": 600})
        poses = [a for a in (search or {}).get("reusable_artifacts", []) if ":dimer_" in a]
        if not poses:
            return False
        dimer = poses[0]
        records = {a.artifact_id: a for a in gw.case_run.artifact_manifest.artifacts}
    if fragments is None:
        pair = (records[dimer].metadata or {}).get("fragment_atom_indices")
        if not (isinstance(pair, list) and len(pair) == 2):
            return False
        fragments = {"molecule_1": pair[0], "molecule_2": pair[1]}
    ctx.cache["last_geometry"] = dimer
    bundle = _step(ctx, steps, "microscopic.run_baseline_bundle",
                   {"source_artifact_id": dimer, "s1_nstates": int(nstates), "td_tout": TD_TOUT})
    if not bundle or not bundle["reusable_artifacts"]:
        return False
    return _step(ctx, steps, "microscopic.analyze_hole_electron_from_bundle",
                 {"source_artifact_id": bundle["reusable_artifacts"][0], "state_root": int(state_root), "spatial_grid": "high",
                  "max_wall_seconds": 900, "fragments": fragments}) is not None


def solvent_polarity(ctx, steps, *, nstates: int = 6, solvents: list | None = None, **_):
    geom = geometry(ctx, steps)
    if geom is None:
        return False
    args = {"source_artifact_id": geom, "s1_nstates": int(nstates), "td_tout": TD_TOUT}
    if solvents:
        args["solvents"] = solvents
    return _step(ctx, steps, "microscopic.run_solvation_polarity_proxy", args) is not None


def conformers(ctx, steps, *, max_members: int = 4, **_):
    geom = geometry(ctx, steps)
    return geom is not None and _step(ctx, steps, "microscopic.sample_conformers_crest",
                                      {"source_artifact_id": geom, "max_members": int(max_members),
                                       "max_wall_seconds": 600}) is not None


def _mol(ctx):
    from rdkit import Chem

    return Chem.MolFromSmiles(ctx.gateway.smiles)


def _neutral(ctx) -> bool:
    from rdkit import Chem

    return Chem.GetFormalCharge(_mol(ctx)) == 0


RECIPES: dict[str, tuple[Callable, str]] = {
    "excited_states": (excited_states, "Vertical singlet excited states (TDA-aTB1): energies, oscillator strengths, dominant transitions. Args: nstates."),
    "charge_transfer": (charge_transfer, "Excited states + hole-electron analysis of one state: centroid distance, overlap, t-index, optional fragment contributions. Args: state_root, nstates, fragments {name: [0-based atom indices]}."),
    "torsion_response": (torsion_response, "Rigid torsion scan of one rotatable dihedral with excited-state brightness at each angle. Args: dihedral_id (from list_rotatable_dihedrals; default first) or dihedral_atoms, angles, nstates."),
    "spin_states": (spin_states, "Singlet/triplet vertical states (TD-DFTB + PySOC SOC for neutral CHNO; PySCF DFT/TDA otherwise or when functional/basis is given; slow for metal complexes). Args: nstates_per_spin, functional, basis."),
    "redox_levels": (redox_levels, "Vertical ionization potential and electron affinity (IPEA-xTB). No args."),
    "vibrations": (vibrations, "Ground-state optimisation and vibrational modes with atom-resolved motions (xTB). No args."),
    "dimer_contacts": (dimer_contacts, "Modeled homodimer poses (xTB aISS): contacts and interaction energies. Modeled, not a crystal. Args: max_candidates."),
    "solvent_polarity": (solvent_polarity, "Implicit-solvent excited states in several solvents (polarity response). Args: nstates, solvents."),
    "conformers": (conformers, "CREST conformer sampling; returns atom-mapped conformers. Args: max_members."),
    "dimer_excited_states": (dimer_excited_states, "Excited states of a modeled dimer (given as source_artifact_id, or found by a dimer search) and a hole-electron analysis with the two molecules as fragments: how much of the hole and electron of one state lies on each molecule and the inter-molecule transfer share. Args: source_artifact_id (a modeled dimer), nstates, state_root, fragments (default: the two molecules)."),
}


M = "microscopic."
# Recipe argument -> (tool, tool argument) it is passed to; recipe cards show that argument's
# allowed type/range/enum from the tool schema, so models do not guess (e.g. unsupported solvents).
RECIPE_ARGS: dict[str, dict[str, tuple[str, str]]] = {
    "excited_states": {"nstates": (M + "run_baseline_bundle", "s1_nstates")},
    "charge_transfer": {"state_root": (M + "analyze_hole_electron_from_bundle", "state_root"),
                        "nstates": (M + "run_baseline_bundle", "s1_nstates"),
                        "fragments": (M + "analyze_hole_electron_from_bundle", "fragments")},
    "torsion_response": {"dihedral_id": (M + "run_torsion_brightness_coupling_scan", "dihedral_id"),
                         "dihedral_atoms": (M + "run_torsion_brightness_coupling_scan", "dihedral_atoms"),
                         "angles": (M + "run_torsion_brightness_coupling_scan", "angles"),
                         "nstates": (M + "run_torsion_brightness_coupling_scan", "s1_nstates")},
    "spin_states": {"nstates_per_spin": (M + "compare_spin_states_pyscf", "nstates_per_spin"),
                    "functional": (M + "compare_spin_states_pyscf", "functional"),
                    "basis": (M + "compare_spin_states_pyscf", "basis")},
    "dimer_contacts": {"max_candidates": (M + "search_dimer_contacts", "max_candidates")},
    "solvent_polarity": {"nstates": (M + "run_solvation_polarity_proxy", "s1_nstates"),
                         "solvents": (M + "run_solvation_polarity_proxy", "solvents")},
    "conformers": {"max_members": (M + "sample_conformers_crest", "max_members")},
    "dimer_excited_states": {"nstates": (M + "run_baseline_bundle", "s1_nstates"),
                             "state_root": (M + "analyze_hole_electron_from_bundle", "state_root"),
                             "fragments": (M + "analyze_hole_electron_from_bundle", "fragments")},
}


def run_recipe(ctx: RecipeContext, name: str, args: dict | None = None) -> RecipeResult:
    if name not in RECIPES:
        return RecipeResult(name, "rejected", [], f"Unknown recipe; choose one of {sorted(RECIPES)}")
    steps: list[dict] = []
    args = dict(args or {})
    ctx.source = args.pop("source_artifact_id", None)
    ctx.cache.pop("last_geometry", None)
    try:
        ok = RECIPES[name][0](ctx, steps, **args)
    except (TypeError, ValueError) as exc:
        return RecipeResult(name, "rejected", steps, f"Invalid recipe arguments: {exc}")
    finally:
        ctx.source = None
    note = f"geometry source: {ctx.cache.get('geometry_source', 'none')}"
    return RecipeResult(name, "success" if ok else "failed", steps, note, ctx.cache.get("last_geometry"))
