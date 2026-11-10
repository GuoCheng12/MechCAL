"""Read printed Amesp transitions without assigning mechanism or state character."""

import re

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
GROUND_HEADER = "Ground to excited state transition electric dipole moments"
EXCITED_HEADER = "Excited to excited state transition electric dipole moments"


def _number(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def parse_transitions(text: str) -> dict:
    transitions = []
    blocks = re.finditer(r"^\s*State\s+(\d+)\s*:\s*E\s*=([^\n]*)(.*?)(?=^\s*State\s+\d+\s*:|\Z)",
                         text, re.M | re.S)
    for block in blocks:
        body = block.group(3).split("E(TD)", 1)
        if len(body) != 2:
            continue  # An unfinished state block is not a delivered transition list.
        components = [
            {"from_orbital": int(row[0]), "to_orbital": int(row[1]),
             "coefficient": _number(row[2])}
            for row in re.findall(rf"^\s*(\d+)\s*-->\s*(\d+)\s+({NUMBER})\s*$", body[0], re.M)
        ]
        if components:
            transitions.append({"state_index": int(block.group(1)), "components": components})
    return {
        "dominant_transitions": transitions,
        "ground_to_excited_transition_dipoles": _dipoles(text, GROUND_HEADER, ground=True),
        "excited_to_excited_transition_dipoles": _dipoles(text, EXCITED_HEADER, ground=False),
        "transition_conventions": {
            "state_index_base": 1, "ground_state_index": 0, "orbital_index_base": 1,
            "dipole_vector_unit": "atomic_unit",
            "boundary": "Printed MO coefficients, not probabilities or an exhaustive decomposition. "
                        "Dipole vectors and orbital indices alone do not assign charge transfer, "
                        "electron-hole separation, emitting-state identity or mechanism.",
        },
    }


def _dipoles(text: str, header: str, *, ground: bool) -> list[dict]:
    if header not in text:
        return []
    lines = text.split(header, 1)[1].splitlines()[1:]
    results = []
    columns_seen = False
    for line in lines:
        if "state-state" in line and all(name in line.split() for name in ("X", "Y", "Z", "Osc")):
            columns_seen = True
            continue
        if not columns_seen:
            if line.strip():
                break
            continue
        fields = line.split()
        expected = 7 if ground else 6
        if (len(fields) != expected or not all(v.isdigit() for v in fields[:2])
                or not all(re.fullmatch(NUMBER, v) for v in fields[2:])):
            break
        row = {"from_state": int(fields[0]), "to_state": int(fields[1]),
               "vector_au": [_number(v) for v in fields[2:5]],
               "oscillator_strength": _number(fields[-1])}
        if ground:
            row["reported_dip_column"] = _number(fields[5])
        results.append(row)
    return results
