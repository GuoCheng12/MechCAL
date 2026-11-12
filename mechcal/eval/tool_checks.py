"""Deterministic checks on recorded tool results: failure classes and result warnings.

Shared by every method that uses the recorded gateway, so all of them see the same signals.

- ``classify_failure``: a failed call is either a *system* error (bad arguments, timeout, no
  convergence, missing prerequisite; retryable with a technical change) or *not applicable*
  (the route cannot treat this molecule at all, e.g. no force-field parameters; never retried).
- ``result_warnings``: a call that finished but whose result looks suspicious. Each warning says
  whether a technical change can address it (``student_fixable``) or whether it needs a scientific
  judgement (e.g. imaginary modes, an inverted singlet-triplet order that may be genuine).
"""

from __future__ import annotations

import re
from typing import Any

# Messages (regex, case-insensitive) that mean the route cannot apply to this molecule/system.
NOT_APPLICABLE = (
    ("no_force_field", r"Neither MMFF94 nor UFF parameters"),
    ("unsupported_elements", r"Supported element scope|Unsupported element|neutral CHNO|element or atom count"),
    ("electron_count", r"even electron count|closed-shell"),
    ("route_disabled", r"disabled|validated with CREST|executable is unavailable|parameter directory is unavailable"),
)
NOT_APPLICABLE_KINDS = {"capability_unsupported", "unsupported"}
SCAN_JUMP_HARTREE = 0.05        # adjacent rigid-scan points differing by > ~31 kcal/mol
SINGLET_SPIN_SQUARE_MAX = 0.1   # closed-shell TDA singlets should have <S^2> = 0


def classify_failure(status: str, kind: str | None, message: str | None) -> dict | None:
    if status in {"success"}:
        return None
    text = message or ""
    for code, pattern in NOT_APPLICABLE:
        if re.search(pattern, text, re.IGNORECASE):
            return {"failure_class": "not_applicable", "failure_code": code, "retryable": False}
    if status == "unsupported" or (kind or "") in NOT_APPLICABLE_KINDS:
        return {"failure_class": "not_applicable", "failure_code": kind or status, "retryable": False}
    code = kind or status
    if re.search(r"timed out|wall time|timeout", text, re.IGNORECASE):
        code = "timeout"
    elif re.search(r"not conver|convergence", text, re.IGNORECASE):
        code = "not_converged"
    return {"failure_class": "system", "failure_code": code, "retryable": True}


def _warn(code: str, message: str, fixable: bool) -> dict:
    return {"code": code, "message": message, "student_fixable": fixable}


def _requested(po: dict, key: str) -> int | None:
    for block in ("requested_parameters", "effective_parameters", "request"):
        value = (po.get(block) or {}).get(key) if isinstance(po.get(block), dict) else None
        if isinstance(value, (int, float)):
            return int(value)
    return None


def result_warnings(tool: str, status: str, po: Any) -> list[dict]:
    """Warnings for a finished (success/partial) call from its parsed observables."""
    if status not in {"success", "partial"} or not isinstance(po, dict):
        return []
    out: list[dict] = []
    if status == "partial":
        out.append(_warn("partial_result", "Only part of the requested calculation completed.", True))

    wanted = _requested(po, "s1_nstates")
    rows = [("", po.get("state_count"))]
    rows += [(f"solvent record {i}", r.get("state_count")) for i, r in enumerate(po.get("solvent_records") or []) if isinstance(r, dict)]
    rows += [(f"scan point {i}", r.get("state_count")) for i, r in enumerate(po.get("route_records") or []) if isinstance(r, dict)]
    for where, count in rows:
        if wanted and isinstance(count, (int, float)) and count < wanted:
            out.append(_warn("fewer_states_than_requested",
                             f"{where or 'Result'}: {int(count)} excited states returned, {wanted} requested.", True))

    if po.get("optimization_converged") is False:
        out.append(_warn("not_converged", "Geometry optimization did not converge.", True))
    if po.get("scf_converged") is False:
        out.append(_warn("not_converged", "SCF did not converge.", True))
    for i, c in enumerate(po.get("candidates") or []):
        if isinstance(c, dict) and c.get("geometry_converged") is False:
            out.append(_warn("not_converged", f"Dimer candidate {i}: geometry not converged.", True))

    if po.get("imaginary_mode_indices_below_minus20"):
        out.append(_warn("imaginary_modes", f"Imaginary modes below -20 cm-1: {po['imaginary_mode_indices_below_minus20']}; "
                         "the geometry is not a minimum (re-optimisation gives a new geometry).", False))

    energies = [r.get("final_energy_hartree") for r in po.get("route_records") or [] if isinstance(r, dict)]
    energies = [e for e in energies if isinstance(e, (int, float))]
    for i in range(1, len(energies)):
        if abs(energies[i] - energies[i - 1]) > SCAN_JUMP_HARTREE:
            out.append(_warn("scan_energy_jump", f"Scan points {i - 1} and {i} differ by "
                             f"{abs(energies[i] - energies[i - 1]):.3f} Hartree; check that point.", True))
    failures = (po.get("route_summary") or {}).get("member_failure_count") if isinstance(po.get("route_summary"), dict) else None
    if isinstance(failures, (int, float)) and failures > 0:
        out.append(_warn("scan_points_failed", f"{int(failures)} scan point(s) failed.", True))

    for i, s in enumerate(po.get("excited_states") or []):
        sq = s.get("spin_square") if isinstance(s, dict) else None
        if isinstance(sq, (int, float)) and sq > SINGLET_SPIN_SQUARE_MAX:
            out.append(_warn("spin_contamination", f"Excited state {i + 1} has <S^2> = {sq:.3f} (singlet expected 0).", False))

    if "stability_check_status" in po:
        if po.get("scf_internal_stable") is False or po.get("scf_external_stable") is False:
            out.append(_warn("unstable_reference", "The SCF reference is unstable; states from it are unreliable "
                             "(fixing this changes the method).", False))
        elif po.get("stability_check_status") != "completed":
            out.append(_warn("stability_not_checked", "Reference stability check did not complete; states are provisional.", True))
    if po.get("spin_manifold_failures"):
        out.append(_warn("states_not_converged", f"{len(po['spin_manifold_failures'])} spin manifold(s)/state(s) failed.", True))
    states = [s for s in po.get("states") or [] if isinstance(s, dict) and isinstance(s.get("excitation_ev"), (int, float))]
    singlets = [s["excitation_ev"] for s in states if str(s.get("state", "")).startswith("S")]
    triplets = [s["excitation_ev"] for s in states if str(s.get("state", "")).startswith("T")]
    if singlets and triplets and min(triplets) > min(singlets):
        out.append(_warn("inverted_singlet_triplet", f"Lowest triplet ({min(triplets):.3f} eV) lies above the lowest singlet "
                         f"({min(singlets):.3f} eV). Unusual but possibly genuine (inverted gap); needs judgement.", False))
    wanted_spin = _requested(po, "nstates_per_spin")
    if wanted_spin and states:
        for prefix, rows_ in (("S", singlets), ("T", triplets)):
            if len(rows_) < wanted_spin:
                out.append(_warn("fewer_states_than_requested",
                                 f"{len(rows_)} {prefix} states returned, {wanted_spin} requested.", True))
    return out
