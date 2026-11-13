"""Reference panel: where a computed value falls among common chromophores (context without labels).

A single number says little about whether it sets a molecule apart: most molecules have a bright
state, some charge separation and soft torsions. The panel (``reference_panel.json``, built by
``examples/build_reference_panel.py``) holds descriptor values computed with the same tools and
default settings for a fixed list of widely studied organic chromophores that are not in the
benchmark. It carries no mechanism labels. At run time each tool result gets, per descriptor, its
percentile among the panel molecules, the panel median and interquartile range.

Descriptors are read from tool outputs by tool, not by mechanism family. Excited-state descriptors
of a modeled dimer are kept apart from those of a single molecule (a bundle counts as a dimer when
it has about twice the molecule's atoms).
"""

from __future__ import annotations

import json
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path
from statistics import median

PANEL_PATH = Path(__file__).with_name("reference_panel.json")
HARTREE_EV = 27.211386
M = "microscopic."

# descriptor -> (unit, meaning). Every descriptor is computed with the tool's default settings in the panel.
DESCRIPTORS: dict[str, tuple[str, str]] = {
    "s1_excitation_ev": ("eV", "lowest singlet vertical excitation energy (TDA-aTB1)"),
    "s1_oscillator_strength": ("", "oscillator strength of S1"),
    "s1_radiative_rate_per_s": ("s^-1", "Strickler-Berg-type radiative rate estimate of S1"),
    "s2_minus_s1_ev": ("eV", "gap between S2 and S1"),
    "max_oscillator_strength": ("", "largest oscillator strength among the computed singlets"),
    "he_centroid_distance_angstrom": ("angstrom", "S1 hole-electron centroid distance"),
    "he_overlap_sr": ("", "S1 hole-electron overlap Sr"),
    "he_t_index_angstrom": ("angstrom", "S1 hole-electron t index (positive: separated)"),
    "torsion_s0_energy_range_ev": ("eV", "S0 energy range over a rigid torsion scan of a rotatable bond"),
    "torsion_f_range": ("", "range of the first oscillator strength over that scan"),
    "lowest_frequency_cm1": ("cm^-1", "lowest real vibrational frequency (xTB)"),
    "modes_below_50_cm1": ("", "number of vibrational modes below 50 cm^-1"),
    "solvent_s1_range_ev": ("eV", "range of the first excitation energy across the default solvents"),
    "dimer_interaction_ev": ("eV", "most negative interaction energy of a modeled homodimer pose"),
    "dimer_interaction_ev_per_heavy_atom": ("eV", "that interaction energy per heavy atom of one molecule"),
    "dimer_s1_shift_ev": ("eV", "dimer S1 minus molecule S1 (same method; negative: red shift)"),
    "dimer_s1_oscillator_strength": ("", "oscillator strength of the dimer S1"),
    "dimer_s2_minus_s1_ev": ("eV", "gap between dimer S2 and S1"),
    "dimer_net_transfer_abs_percent": ("%", "net charge moved between the two molecules in the dimer S1 (absolute)"),
    "s1_minus_t1_ev": ("eV", "S1 minus T1 vertical gap (TD-DFTB)"),
    "max_s1_triplet_soc_cm1": ("cm^-1", "largest S1-triplet spin-orbit coupling (PySOC)"),
}


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _first(rows: list, key: str):
    return _num(rows[0].get(key)) if rows and isinstance(rows[0], dict) else None


def extract(tool: str, po: dict, *, natoms: int, heavy_atoms: int | None = None, source_atoms: int | None = None,
            monomer_s1: float | None = None) -> dict[str, float]:
    """Descriptor values in one tool result (``po``: parsed observables). Missing values are left out."""
    if not isinstance(po, dict):
        return {}
    out: dict[str, float | None] = {}
    if tool == M + "run_baseline_bundle":
        states = po.get("excited_states") or []
        sd = po.get("state_descriptors") or []
        atoms = _num(po.get("mulliken_charge_count")) or natoms
        dimer = atoms > 1.5 * natoms
        s1 = _first(states, "excitation_energy_ev")
        if dimer:
            out["dimer_s1_oscillator_strength"] = _first(states, "oscillator_strength")
            out["dimer_s2_minus_s1_ev"] = _first(sd, "gap_to_higher_state_ev")
            if s1 is not None and monomer_s1 is not None:
                out["dimer_s1_shift_ev"] = s1 - monomer_s1
        else:
            out["s1_excitation_ev"] = s1
            out["s1_oscillator_strength"] = _first(states, "oscillator_strength")
            if sd and isinstance(sd[0], dict):
                out["s1_radiative_rate_per_s"] = _num(sd[0].get("radiative_rate_estimate_per_s", 0.0))
                out["s2_minus_s1_ev"] = _num(sd[0].get("gap_to_higher_state_ev"))
            out["max_oscillator_strength"] = _num((po.get("bright_state") or {}).get("oscillator_strength"))
    elif tool == M + "analyze_hole_electron_from_bundle":
        he = po.get("hole_electron_spatial_analysis") or {}
        dimer = bool(source_atoms and source_atoms > 1.5 * natoms)
        if _num(he.get("state_index")) == 1:
            if dimer:
                net = _num((he.get("fragment_pair") or {}).get("net_transfer_first_to_second_percent"))
                out["dimer_net_transfer_abs_percent"] = abs(net) if net is not None else None
            else:
                out["he_centroid_distance_angstrom"] = _num(he.get("centroid_distance_angstrom"))
                out["he_overlap_sr"] = _num(he.get("spatial_overlap_sr"))
                out["he_t_index_angstrom"] = _num(he.get("t_index_angstrom"))
    elif tool == M + "run_torsion_brightness_coupling_scan":
        summary = po.get("route_summary") or {}
        e = _num(summary.get("energy_range_hartree"))
        out["torsion_s0_energy_range_ev"] = e * HARTREE_EV if e is not None else None
        out["torsion_f_range"] = _num(summary.get("oscillator_strength_range"))
    elif tool == M + "analyze_ground_state_vibrations":
        freqs = [f for f in (po.get("all_frequencies_cm_minus1") or []) if isinstance(f, (int, float)) and f > 0]
        if freqs:
            out["lowest_frequency_cm1"] = float(min(freqs))
            out["modes_below_50_cm1"] = float(sum(f < 50 for f in freqs))
    elif tool == M + "run_solvation_polarity_proxy":
        comp = (po.get("solvent_comparisons") or {}).get("first_excitation_energy_shift_ev") or {}
        out["solvent_s1_range_ev"] = _num(comp.get("available_value_range"))
    elif tool == M + "search_dimer_contacts":
        energies = [_num(c.get("interaction_energy_ev")) for c in po.get("candidates") or [] if isinstance(c, dict)]
        energies = [e for e in energies if e is not None]
        if energies:
            out["dimer_interaction_ev"] = min(energies)
            if heavy_atoms:
                out["dimer_interaction_ev_per_heavy_atom"] = min(energies) / heavy_atoms
    elif tool == M + "compare_singlet_triplet_states":
        out["s1_minus_t1_ev"] = _num(po.get("s1_minus_t1_ev"))
    elif tool == M + "evaluate_spin_orbit_couplings":
        socs = [_num(c.get("rss_cm_inverse")) for c in po.get("couplings") or []
                if isinstance(c, dict) and c.get("singlet") == 1]
        socs = [s for s in socs if s is not None]
        if socs:
            out["max_s1_triplet_soc_cm1"] = max(socs)
    return {k: v for k, v in out.items() if v is not None}


@lru_cache(maxsize=4)
def load_panel(path: str = str(PANEL_PATH)) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def distributions(panel: dict) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for row in panel.get("molecules", []):
        for k, v in (row.get("values") or {}).items():
            out.setdefault(k, []).append(float(v))
    return {k: sorted(v) for k, v in out.items()}


def _quartiles(xs: list[float]) -> list[float]:
    def q(f):
        i = f * (len(xs) - 1)
        lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)
    return [q(0.25), q(0.75)]


def context(values: dict[str, float], panel: dict | None = None, *, min_n: int = 8) -> list[dict]:
    """Per descriptor: value, percentile among the panel molecules, panel median and interquartile range."""
    panel = panel if panel is not None else load_panel()
    if not panel or not values:
        return []
    dist = distributions(panel)
    rows = []
    for key, value in values.items():
        xs = dist.get(key) or []
        if len(xs) < min_n:
            continue
        unit, meaning = DESCRIPTORS.get(key, ("", key))
        q25, q75 = _quartiles(xs)
        rows.append({"descriptor": key, "meaning": meaning, "value": round(value, 4), "unit": unit,
                     "panel_percentile": round(100 * bisect_right(xs, value) / len(xs)),
                     "panel_median": round(median(xs), 4), "panel_iqr": [round(q25, 4), round(q75, 4)],
                     "panel_n": len(xs)})
    return rows


def panel_note(panel: dict | None = None) -> str | None:
    """One line for prompts/cards describing what the panel is (or None when no panel is installed)."""
    panel = panel if panel is not None else load_panel()
    if not panel:
        return None
    return (f"reference_panel: percentile of a value among {len(panel.get('molecules', []))} common organic chromophores "
            "(isolated molecules, same tool and default settings, no mechanism labels, no metal complexes)")
