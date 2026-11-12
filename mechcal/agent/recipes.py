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
# PySCF DF-PBE/TDA singlets+triplets with the reference stability check: ~1200 s for a 56-atom Au
# complex at 8 threads (SCF 93 s, internal stability ~520 s, external ~270 s), so 900 s left it partial.
PYSCF_SPIN_WALL = 1500


@dataclass
class RecipeContext:
    gateway: ToolGateway
    cache: dict = field(default_factory=dict)   # reusable artifact ids (geometry, bundle, dihedrals)


@dataclass
class RecipeResult:
    recipe: str
    status: str                                 # success | failed | rejected
    observations: list[dict]
    note: str = ""

    def to_json(self) -> dict:
        return {"recipe": self.recipe, "status": self.status, "note": self.note, "observations": self.observations}


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
    if "geometry" in ctx.cache:
        return ctx.cache["geometry"]
    gw = ctx.gateway
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
                {"source_artifact_id": gw.prepared_artifact_id, "max_opt_cycles": 300, "max_wall_seconds": 300})
    if not obs or not obs["reusable_artifacts"]:
        return None
    ctx.cache["geometry"] = obs["reusable_artifacts"][0]
    ctx.cache["geometry_source"] = "atb1_optimized"
    return ctx.cache["geometry"]


def _bundle(ctx: RecipeContext, steps: list, nstates: int) -> str | None:
    key = f"bundle:{nstates}"
    if key in ctx.cache:
        return ctx.cache[key]
    existing = _latest(ctx, "amesp_baseline_bundle")
    if existing:
        ctx.cache[key] = ctx.cache["bundle"] = existing
        return existing
    geom = geometry(ctx, steps)
    if geom is None:
        return None
    obs = _step(ctx, steps, "microscopic.run_baseline_bundle",
                {"source_artifact_id": geom, "s1_nstates": nstates, "td_tout": TD_TOUT})
    if not obs or not obs["reusable_artifacts"]:
        return None
    ctx.cache[key] = ctx.cache["bundle"] = obs["reusable_artifacts"][0]
    return ctx.cache[key]


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


def bright_dark_states(ctx, steps, *, nstates: int = DEFAULT_NSTATES, **_):
    geom = geometry(ctx, steps)
    return geom is not None and _step(ctx, steps, "microscopic.run_bright_dark_state_ordering",
                                      {"source_artifact_id": geom, "s1_nstates": int(nstates), "td_tout": TD_TOUT}) is not None


def torsion_response(ctx, steps, *, dihedral_id: str | None = None, angles: list | None = None, nstates: int = 6, **_):
    geom = geometry(ctx, steps)
    if geom is None:
        return False
    if dihedral_id is None:
        listing = _step(ctx, steps, "microscopic.list_rotatable_dihedrals", {"source_artifact_id": geom})
        if listing is None:
            return False
        ids = [v for p, v in listing["labels"].items() if p.endswith("/dihedral_id")]
        if not ids:
            return False
        dihedral_id = ids[0]
    args = {"source_artifact_id": geom, "dihedral_id": dihedral_id, "s1_nstates": int(nstates), "td_tout": TD_TOUT}
    if angles:
        args["angles"] = angles
    return _step(ctx, steps, "microscopic.run_torsion_brightness_coupling_scan", args) is not None


def spin_states(ctx, steps, *, nstates_per_spin: int = 3, **_):
    geom = geometry(ctx, steps)
    if geom is None:
        return False
    elements = {a.GetSymbol() for a in _mol(ctx).GetAtoms()}
    if elements <= {"C", "H", "N", "O"} and _neutral(ctx):
        ok = _step(ctx, steps, "microscopic.compare_singlet_triplet_states",
                   {"source_artifact_id": geom, "nstates_per_spin": int(nstates_per_spin)}) is not None
        _step(ctx, steps, "microscopic.evaluate_spin_orbit_couplings", {"source_artifact_id": geom})
        return ok
    return _step(ctx, steps, "microscopic.compare_spin_states_pyscf",
                 {"source_artifact_id": geom, "nstates_per_spin": int(nstates_per_spin), "functional": "pbe",
                  "basis": "def2-svp", "max_wall_seconds": PYSCF_SPIN_WALL}) is not None


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
    "bright_dark_states": (bright_dark_states, "Bright/dark state ordering scan at the ground-state geometry. Args: nstates."),
    "torsion_response": (torsion_response, "Rigid torsion scan of one rotatable dihedral with excited-state brightness at each angle. Args: dihedral_id (from list_rotatable_dihedrals; default first), angles, nstates."),
    "spin_states": (spin_states, "Singlet/triplet vertical states (TD-DFTB + PySOC SOC for neutral CHNO; PySCF DF-PBE/TDA otherwise, slow: ~20 min for a metal complex). Args: nstates_per_spin."),
    "redox_levels": (redox_levels, "Vertical ionization potential and electron affinity (IPEA-xTB). No args."),
    "vibrations": (vibrations, "Ground-state optimisation and vibrational modes with atom-resolved motions (xTB). No args."),
    "dimer_contacts": (dimer_contacts, "Modeled homodimer poses (xTB aISS): contacts and interaction energies. Modeled, not a crystal. Args: max_candidates."),
    "solvent_polarity": (solvent_polarity, "Implicit-solvent excited states in several solvents (polarity response). Args: nstates, solvents."),
    "conformers": (conformers, "CREST conformer sampling; returns atom-mapped conformers. Args: max_members."),
}


M = "microscopic."
# Recipe argument -> (tool, tool argument) it is passed to; recipe cards show that argument's
# allowed type/range/enum from the tool schema, so models do not guess (e.g. unsupported solvents).
RECIPE_ARGS: dict[str, dict[str, tuple[str, str]]] = {
    "excited_states": {"nstates": (M + "run_baseline_bundle", "s1_nstates")},
    "charge_transfer": {"state_root": (M + "analyze_hole_electron_from_bundle", "state_root"),
                        "nstates": (M + "run_baseline_bundle", "s1_nstates"),
                        "fragments": (M + "analyze_hole_electron_from_bundle", "fragments")},
    "bright_dark_states": {"nstates": (M + "run_bright_dark_state_ordering", "s1_nstates")},
    "torsion_response": {"dihedral_id": (M + "run_torsion_brightness_coupling_scan", "dihedral_id"),
                         "angles": (M + "run_torsion_brightness_coupling_scan", "angles"),
                         "nstates": (M + "run_torsion_brightness_coupling_scan", "s1_nstates")},
    "spin_states": {"nstates_per_spin": (M + "compare_spin_states_pyscf", "nstates_per_spin")},
    "dimer_contacts": {"max_candidates": (M + "search_dimer_contacts", "max_candidates")},
    "solvent_polarity": {"nstates": (M + "run_solvation_polarity_proxy", "s1_nstates"),
                         "solvents": (M + "run_solvation_polarity_proxy", "solvents")},
    "conformers": {"max_members": (M + "sample_conformers_crest", "max_members")},
}


def run_recipe(ctx: RecipeContext, name: str, args: dict | None = None) -> RecipeResult:
    if name not in RECIPES:
        return RecipeResult(name, "rejected", [], f"Unknown recipe; choose one of {sorted(RECIPES)}")
    steps: list[dict] = []
    try:
        ok = RECIPES[name][0](ctx, steps, **(args or {}))
    except (TypeError, ValueError) as exc:
        return RecipeResult(name, "rejected", steps, f"Invalid recipe arguments: {exc}")
    note = f"geometry source: {ctx.cache.get('geometry_source', 'none')}"
    return RecipeResult(name, "success" if ok else "failed", steps, note)
