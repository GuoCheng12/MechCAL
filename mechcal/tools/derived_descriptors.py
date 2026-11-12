"""Quantities derived from already computed results (no new calculation), recorded with the result.

Each derivation is a fixed physical formula applied to recorded numbers, so it can be cited by
pointer like any other value. Assumptions are stated next to the numbers.
"""

from __future__ import annotations

EV_TO_WAVENUMBER = 8065.544          # cm-1 per eV
HARTREE_TO_EV = 27.211386245988
# Strickler-Berg / Einstein A from an oscillator strength: k_r = 2 pi e^2 nu^2 n^2 f / (eps0 m_e c^3)
# = 0.6670 s-1 cm2 * (wavenumber)^2 * f for refractive index n = 1.
RADIATIVE_CONSTANT = 0.6670

STATE_DESCRIPTOR_METHOD = ("Radiative rate k_r = 0.667 s^-1 cm^2 x (excitation wavenumber)^2 x oscillator strength "
                           "(Strickler-Berg form, refractive index 1, vertical absorption energy at the S0 geometry); "
                           "lifetime = 1/k_r. Gaps are differences of vertical excitation energies.")


def state_descriptors(excited_states: list[dict]) -> list[dict]:
    rows = [s for s in excited_states if isinstance(s.get("excitation_energy_ev"), (int, float))]
    out = []
    for i, s in enumerate(rows):
        e, f = float(s["excitation_energy_ev"]), s.get("oscillator_strength")
        row = {"state_index": s.get("state_index", i + 1)}
        if isinstance(f, (int, float)) and f > 0:
            k = RADIATIVE_CONSTANT * (e * EV_TO_WAVENUMBER) ** 2 * float(f)
            row.update(radiative_rate_estimate_per_s=k, radiative_lifetime_estimate_ns=1e9 / k)
        if i > 0:
            row["gap_to_lower_state_ev"] = e - float(rows[i - 1]["excitation_energy_ev"])
        if i + 1 < len(rows):
            row["gap_to_higher_state_ev"] = float(rows[i + 1]["excitation_energy_ev"]) - e
        out.append(row)
    return out


def torsion_profile(route_records: list[dict]) -> list[dict]:
    """Add S0 and S1 energies relative to the lowest S0 point of the scan (vertical S1 = S0 + excitation)."""
    energies = [r["final_energy_hartree"] for r in route_records if isinstance(r.get("final_energy_hartree"), (int, float))]
    if not energies:
        return route_records
    lowest = min(energies)
    out = []
    for r in route_records:
        r = dict(r)
        if isinstance(r.get("final_energy_hartree"), (int, float)):
            r["s0_relative_ev"] = (r["final_energy_hartree"] - lowest) * HARTREE_TO_EV
            if isinstance(r.get("first_excitation_energy_ev"), (int, float)):
                r["s1_relative_to_s0_minimum_ev"] = r["s0_relative_ev"] + r["first_excitation_energy_ev"]
        out.append(r)
    return out


def fragment_pair(fragments: list[dict]) -> dict | None:
    """For exactly two fragments: where the hole and the electron are, and how much charge moves.

    IFCT approximation (as in Multiwfn): hole and electron are treated as independent, so the share of
    pairs on different fragments is (h1 e2 + h2 e1) / 100. An excitation spread evenly over two identical
    molecules gives ~50 % with no charge moving; the net transfer (h1 e2 - h2 e1) / 100 measures charge
    actually moved from the first to the second fragment (0 for an even spread, ~100 for full transfer).
    """
    if len(fragments) != 2:
        return None
    (a, b) = fragments
    h1, e1, h2, e2 = a["hole_percent"], a["electron_percent"], b["hole_percent"], b["electron_percent"]
    return {"first": a["name"], "second": b["name"],
            "hole_percent": {a["name"]: h1, b["name"]: h2}, "electron_percent": {a["name"]: e1, b["name"]: e2},
            "transfer_first_to_second_percent": h1 * e2 / 100.0, "transfer_second_to_first_percent": h2 * e1 / 100.0,
            "inter_fragment_transfer_percent": (h1 * e2 + h2 * e1) / 100.0,
            "net_transfer_first_to_second_percent": (h1 * e2 - h2 * e1) / 100.0,
            "same_fragment_percent": (h1 * e1 + h2 * e2) / 100.0,
            "definition": ("IFCT approximation (hole and electron independent, heavy atoms): inter_fragment_transfer_percent "
                           "is ~50 for an excitation spread evenly over two identical fragments even though no charge "
                           "moves; net_transfer_first_to_second_percent is the charge actually moved (0 = even spread).")}
