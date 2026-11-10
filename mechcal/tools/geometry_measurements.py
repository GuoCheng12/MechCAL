"""Coordinate measurements on an explicitly identified geometry, with no QC run."""

import numpy as np
from ase.geometry import get_angles, get_dihedrals

from mechcal.schemas.tool_arguments import GeometryMeasurement


def measure_geometry(symbols, coordinates, requests):
    xyz = np.asarray(coordinates, dtype=float)
    if xyz.shape != (len(symbols), 3) or not np.isfinite(xyz).all():
        raise ValueError("Expected one finite XYZ coordinate per atom.")
    output = {}
    for raw in requests:
        item = GeometryMeasurement.model_validate(raw)
        if max(item.atom_indices) >= len(symbols):
            raise ValueError(f"{item.measurement_id}: atom index is outside the selected geometry.")
        p = xyz[item.atom_indices]
        unit = "angstrom" if item.kind == "distance" else "degree"
        value, reason = None, ""
        try:
            if item.kind == "distance":
                value = float(np.linalg.norm(p[1] - p[0]))
            elif item.kind == "angle":
                value = float(get_angles([p[0] - p[1]], [p[2] - p[1]])[0])
            else:
                value = float(get_dihedrals([p[1] - p[0]], [p[2] - p[1]], [p[3] - p[2]])[0])
                value = (value + 180) % 360 - 180
            if not np.isfinite(value):
                value, reason = None, "Degenerate geometry: measurement is undefined."
        except ZeroDivisionError:
            reason = "Degenerate geometry: measurement is undefined."
        output[item.measurement_id] = {
            "kind": item.kind, "atom_indices": item.atom_indices,
            "elements": [symbols[i] for i in item.atom_indices], "value": value, "unit": unit,
            "status": "available" if value is not None else "undefined", "reason": reason,
        }
    return output
