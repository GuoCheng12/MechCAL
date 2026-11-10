"""Validated spin/SOC parsing; not yet registered as an executable capability."""

import math
import re

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
STATE = re.compile(rf"^\s*State\s+(\d+)\s*:\s*E\s*=\s*({NUMBER})\s+eV", re.M)
DETAIL = re.compile(rf"E\(TD\)\s*=\s*({NUMBER})\s+<S\*\*2>\s*=\s*({NUMBER})\s+f\s*=\s*({NUMBER})")
COUPLING = re.compile(rf"^\s*(\d+)\s+(\d+)\s+\|<([ST])\|H_SO\|([ST])>\|\s*=\s*({NUMBER})\s*$")


def _number(value: str) -> float:
    result = float(value.replace("D", "E").replace("d", "e"))
    if not math.isfinite(result):
        raise ValueError("Nonfinite state/SOC value.")
    return result


def parse_spin_soc(text: str, *, nstates_per_spin: int) -> dict:
    """Map raw indices by explicit spin and energy, never by global output position."""
    if type(nstates_per_spin) is not int or not 1 <= nstates_per_spin <= 10:
        raise ValueError("An explicit positive spin-manifold size is required.")
    matches = list(STATE.finditer(text))
    if len(matches) != 2 * nstates_per_spin:
        raise ValueError("Incomplete singlet/triplet state table.")
    states = {}
    for index, match in enumerate(matches):
        raw_index = int(match[1])
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        details = DETAIL.findall(text[match.end():end])
        if len(details) != 1 or raw_index in states or raw_index == 0:
            raise ValueError("Missing, duplicate or ambiguous state detail.")
        total, spin, oscillator = map(_number, details[0])
        energy = _number(match[2])
        if spin not in (0.0, 2.0) or energy <= 0 or oscillator < 0:
            raise ValueError("Unsupported spin or invalid excitation/oscillator value.")
        states[raw_index] = dict(raw_state_index=raw_index, spin="S" if spin == 0 else "T",
            spin_square=spin, excitation_energy_ev=energy, total_energy_hartree=total,
            oscillator_strength=oscillator)
    for spin in ("S", "T"):
        manifold = sorted((r for r in states.values() if r["spin"] == spin), key=lambda r: (r["excitation_energy_ev"], r["raw_state_index"]))
        if len(manifold) != nstates_per_spin:
            raise ValueError("Requested spin manifold is incomplete.")
        for index, row in enumerate(manifold, 1):
            row["state_label"] = f"{spin}{index}"
    header = re.search(r"State--State\s+Sum of SOC\s+cm\^-1", text)
    if header is None:
        raise ValueError("SOC table or explicit cm^-1 units are missing; not a zero coupling.")
    if "Effective Charge:" not in text[:header.start()]:
        raise ValueError("Unrecognized SOC approximation; expected effective-charge output.")
    rows, seen = [], set()
    def label(raw_index):
        if raw_index == 0:
            return "S0"
        if raw_index not in states:
            raise ValueError("SOC references an unreported state.")
        return states[raw_index]["state_label"]
    for line in text[header.end():].splitlines():
        if "H_SO" not in line:
            continue
        match = COUPLING.fullmatch(line)
        if match is None:
            raise ValueError("Malformed SOC table row.")
        i, j = int(match[1]), int(match[2])
        left, right = label(i), label(j)
        pair = tuple(sorted((i, j)))
        if pair in seen or left[0] != match[3] or right[0] != match[4]:
            raise ValueError("Duplicate SOC pair or inconsistent spin labels.")
        seen.add(pair)
        value = _number(match[5])
        if value < 0:
            raise ValueError("SOC magnitude must be nonnegative.")
        rows.append(dict(raw_state_indices=[i, j], state_labels=[left, right], soc_cm_minus1=value))
    cross = [r for r in rows if r["state_labels"][0][0] != r["state_labels"][1][0] and "S0" not in r["state_labels"]]
    if len(cross) != nstates_per_spin ** 2:
        raise ValueError("Incomplete excited singlet-triplet coupling matrix.")
    return dict(states=list(states.values()), couplings=rows, singlet_triplet_couplings=cross,
        energy_unit="eV", soc_unit="cm^-1", state_label_definition="Energy order within each explicit spin manifold at this geometry; not character tracking.",
        interpretation_boundary="Vertical spin-state energies and effective-charge SOC magnitudes, not ISC rates, triplet populations or mechanism assignments.")
