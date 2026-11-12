"""Tool cost estimates, so the Planner can weigh a method's scientific value against its time.

Estimates come from recorded wall times: a seed table measured on this machine (below), the calls
of the current run, and optionally a cross-run log (one JSON line per executed call). They are
scaled to the current molecule by atom count with a per-tool exponent, and always reported with
their basis ("measured on 56 atoms: 923 s"). A timed-out or partial call is a lower bound.
Estimates are approximate (the same tool can differ by 10x between molecules); they inform the
Planner, they never block a calculation by themselves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median

M = "microscopic."
# Measured 2026-11-12 on the development machine (Amesp npara 1, Multiwfn 16 threads,
# PySCF 32 threads / 16 GB). (tool, method key, atoms, seconds, finished?, note, basis functions)
SEED: list[tuple[str, str, int, float, bool, str, int | None]] = [
    # native aTB1 S0 optimization from an RDKit seed (11-molecule comparison)
    *[(M + "optimize_ground_state_geometry", "", n, s, True, "native aTB1, RDKit seed", None)
      for n, s in ((26, 1.9), (29, 4.2), (33, 5.7), (44, 17.0), (56, 25.4), (76, 40.1), (52, 205.9), (34, 165.4))],
    *[(M + "optimize_ground_state_geometry", "", n, s, False, "native aTB1, not converged in time", None)
      for n, s in ((43, 300.0), (86, 300.0), (25, 152.0))],
    (M + "optimize_ground_state_geometry", "", 56, 112.0, True, "0003: redundant stop + Cartesian 385 steps", None),
    (M + "run_baseline_bundle", "", 56, 20.4, True, "0003, aTB1 opt + TDA", None),
    (M + "analyze_hole_electron_from_bundle", "", 56, 6.3, True, "0003, Multiwfn 16 threads", None),
    (M + "compare_spin_states_pyscf", "pbe/grid1", 56, 923.0, True, "0003 Au complex, 3 states per spin, with stability check", 577),
    (M + "compare_spin_states_pyscf", "pbe/grid1", 34, 1500.0, False, "0141 Fe complex: SCF convergence trouble, states incomplete", 386),
    (M + "compare_spin_states_pyscf", "pbe0/grid1", 56, 1800.0, False, "0003: only singlets finished in 1800 s", None),
    (M + "compare_spin_states_pyscf", "pbe0/grid2", 56, 1200.0, False, "0003: no state finished in 1200 s", None),
    (M + "run_solvation_polarity_proxy", "", 56, 3.0, True, "0003, 3 solvents", None),
    (M + "run_torsion_brightness_coupling_scan", "", 56, 9.0, True, "0003, 4 angles", None),
    (M + "analyze_ground_state_vibrations", "", 56, 6.0, True, "0003, xTB", None),
    (M + "search_dimer_contacts", "", 56, 50.0, True, "0003, xTB aISS", None),
]
SIZE_BY_BASIS = {M + "compare_spin_states_pyscf"}      # scaled by basis-function count, not atom count
TRANSITION_METALS = set("Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd Hf Ta W Re Os Ir Pt Au Hg".split())


def basis_functions(smiles: str) -> int:
    """Approximate def2-SVP function count (H 5, Li-Ne 14, Na-Ar 18, K-Kr 31, heavier with ECP 36)."""
    from rdkit import Chem
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    def n(z):
        return 5 if z <= 2 else 14 if z <= 10 else 18 if z <= 18 else 31 if z <= 54 else 36
    return sum(n(a.GetAtomicNum()) for a in mol.GetAtoms())


EXPONENT = {M + "compare_spin_states_pyscf": 3.0, M + "optimize_ground_state_geometry": 2.0}
DEFAULT_EXPONENT = 1.5
RECIPE_STEPS = {           # main tool of each recipe (preparation steps are added when needed)
    "excited_states": [M + "run_baseline_bundle"],
    "charge_transfer": [M + "run_baseline_bundle", M + "analyze_hole_electron_from_bundle"],
    "torsion_response": [M + "run_torsion_brightness_coupling_scan"],
    "spin_states": [M + "compare_spin_states_pyscf"],
    "redox_levels": [M + "estimate_vertical_ip_ea"],
    "vibrations": [M + "analyze_ground_state_vibrations"],
    "dimer_contacts": [M + "search_dimer_contacts"],
    "solvent_polarity": [M + "run_solvation_polarity_proxy"],
    "conformers": [M + "sample_conformers_crest"],
    "dimer_excited_states": [M + "search_dimer_contacts", M + "run_baseline_bundle", M + "analyze_hole_electron_from_bundle"],
}
RECIPE_DEFAULTS = {"spin_states": {"functional": "pbe", "grid_level": 1}}


def method_key(tool: str, args: dict | None) -> str:
    args = args or {}
    if tool == M + "compare_spin_states_pyscf":
        return f"{args.get('functional', 'pbe')}/grid{args.get('grid_level', 1)}"
    return ""


@dataclass
class Record:
    tool: str
    key: str
    atoms: int
    seconds: float
    finished: bool
    note: str = ""
    size: int | None = None          # basis functions, for tools scaled by basis size
    metal: bool | None = None        # molecule contains a transition metal (None: not recorded)


THIS_RUN, LOGGED = "this run", "logged run"


class CostModel:
    def __init__(self, natoms: int, log_path: str | Path | None = None, smiles: str | None = None):
        self.natoms = natoms
        self.size = basis_functions(smiles) if smiles else None
        self.metals = sorted({a for a in TRANSITION_METALS if smiles and f"[{a}" in smiles})
        self.records = [Record(*r, metal=("Au" in r[5] or "Fe" in r[5])) for r in SEED]
        self.run_records: list[Record] = []          # recorded in this run only: the part appended to the log
        self.log_path = Path(log_path) if log_path else None
        if self.log_path and self.log_path.is_file():
            with self.log_path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                        note = d.get("note") or LOGGED
                        self.records.append(Record(d["tool"], d.get("key", ""), int(d["atoms"]), float(d["seconds"]),
                                                   bool(d["finished"]), LOGGED if note == THIS_RUN else note,
                                                   d.get("size"), d.get("metal")))
                    except (ValueError, KeyError, TypeError):
                        continue

    def add(self, tool: str, args: dict, seconds: float, status: str, note: str = THIS_RUN) -> Record:
        rec = Record(tool, method_key(tool, args), self.natoms, round(seconds, 1),
                     status in {"success"}, note, self.size if tool in SIZE_BY_BASIS else None, bool(self.metals))
        if status in {"success", "partial", "failed"} and seconds > 0:
            self.records.append(rec)
            self.run_records.append(rec)
        return rec

    def append_log(self, records: list[Record] | None = None) -> None:
        """Append records (default: this run's) to the cross-run log; loaded records are never written back."""
        records = self.run_records if records is None else records
        if not self.log_path or not records:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r.__dict__ | {"note": LOGGED if r.note == THIS_RUN else r.note}) + "\n")

    def estimate(self, tool: str, args: dict | None = None) -> dict:
        key = method_key(tool, args)
        rows = [r for r in self.records if r.tool == tool and r.key == key]
        same_kind = [r for r in rows if r.metal is None or r.metal == bool(self.metals)]
        mixed_kind = bool(rows) and not same_kind
        rows = same_kind or rows                      # metal molecules from metal records, organics from organic ones
        if not rows:
            return {"tool": tool, "method": key or None, "status": "unknown", "note": "no recorded run"}
        p = EXPONENT.get(tool, DEFAULT_EXPONENT)

        def scale(r):
            if tool in SIZE_BY_BASIS and self.size and r.size:
                return r.seconds * (self.size / r.size) ** p
            return r.seconds * (self.natoms / max(r.atoms, 1)) ** p
        done = [scale(r) for r in rows if r.finished]
        bounds = [scale(r) for r in rows if not r.finished]
        nearest = min(rows, key=lambda r: abs(r.atoms - self.natoms))
        basis = f"{len(rows)} recorded run(s); nearest: {nearest.atoms} atoms, {nearest.seconds:.0f} s" + \
                ("" if nearest.finished else " (did not finish)") + (f", {nearest.note}" if nearest.note else "")
        out = {"tool": tool, "method": key or None, "atoms": self.natoms, "basis": basis}
        if mixed_kind:
            out["caveat"] = ("only metal complexes recorded" if not self.metals else "no metal complex recorded")
        if done:
            out.update(status="estimate", seconds=round(median(done)), range=[round(min(done)), round(max(done))])
            if bounds and max(bounds) > max(done):
                out["note"] = f"some runs did not finish within {round(max(bounds))} s"
        else:
            out.update(status="lower_bound", at_least_seconds=round(max(bounds)))
        if tool in SIZE_BY_BASIS and self.metals:
            metal_runs = [f"{r.note}: {r.seconds:.0f} s{'' if r.finished else ' (did not finish)'}"
                          for r in rows if any(m in r.note for m in ("Au", "Fe", "metal"))]
            out["metal_note"] = ("Uncertain for metal complexes: SCF convergence varies more than size suggests. "
                                 + ("Recorded: " + "; ".join(metal_runs) if metal_runs else "No metal complex recorded."))
        return out

    def recipe_estimate(self, recipe: str, args: dict | None = None, *, need_geometry: bool, need_bundle: bool) -> dict:
        args = {**RECIPE_DEFAULTS.get(recipe, {}), **(args or {})}
        steps = list(RECIPE_STEPS.get(recipe, []))
        if not need_bundle and M + "run_baseline_bundle" in steps and recipe != "excited_states":
            steps.remove(M + "run_baseline_bundle")
        if recipe == "excited_states" and not need_bundle:
            steps = []
        if need_geometry:
            steps.insert(0, M + "optimize_ground_state_geometry")
        if recipe == "dimer_excited_states":         # the excited-state steps run on two molecules
            parts = [self.estimate(steps[0], args)] if steps and steps[0].endswith("geometry") else []
            doubled = CostModel.__new__(CostModel)
            doubled.__dict__.update(self.__dict__, natoms=2 * self.natoms, size=2 * self.size if self.size else None)
            parts += [(self if s.endswith("dimer_contacts") else doubled).estimate(s, args)
                      for s in steps if not s.endswith("geometry")]
            return self.combine(parts)
        return self.combine([self.estimate(s, args) for s in steps])

    @staticmethod
    def combine(parts: list[dict]) -> dict:
        if not parts:
            return {"status": "estimate", "seconds": 0, "steps": [], "note": "reuses existing results"}
        if any(p["status"] == "unknown" for p in parts):
            return {"status": "unknown", "steps": parts}
        total = sum(p.get("seconds", p.get("at_least_seconds", 0)) for p in parts)
        if any(p["status"] == "lower_bound" for p in parts):
            return {"status": "lower_bound", "at_least_seconds": total, "steps": parts}
        return {"status": "estimate", "seconds": total, "steps": parts}


def expected_seconds(est: dict) -> float | None:
    return est.get("seconds", est.get("at_least_seconds"))
