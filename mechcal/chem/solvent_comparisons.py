"""Arithmetic summaries of recorded solvent calculations, not mechanism evidence."""

from math import isfinite


def solvent_comparisons(records: list[dict]) -> dict:
    comparisons = {}
    for name, state_key, value_key, unit in (
        ("first_excitation_energy_shift_ev", "first_state", "excitation_energy_ev", "eV"),
        ("bright_oscillator_strength_delta", "bright_state", "oscillator_strength", "dimensionless"),
    ):
        points = []
        for index, record in enumerate(records):
            state = record.get(state_key)
            if not isinstance(state, dict) or not state:
                continue
            value = state.get(value_key)
            valid = type(value) in (int, float) and isfinite(value)
            points.append({"record_index": index, "solvent": record.get("solvent"),
                           "state_index": state.get("state_index"),
                           "value": float(value) if valid else None})
        values = [p["value"] for p in points if p["value"] is not None]
        pair = points[:2]
        delta = (round(pair[0]["value"] - pair[1]["value"], 6)
                 if len(pair) == 2 and all(p["value"] is not None for p in pair) else None)
        comparisons[name] = {
            "unit": unit, "points": points,
            "first_pair_delta": delta,
            "delta_definition": "First listed state value minus second listed state value; not the full range.",
            "available_value_range": round(max(values) - min(values), 6) if len(values) >= 2 else None,
            "range_definition": "Maximum minus minimum across finite returned values only.",
            "complete_for_returned_records": len(values) == len(records),
            "scope": "Arithmetic only. Missing/failed calculations are not zero; electronic state identity is not tracked across conditions.",
        }
    return comparisons
