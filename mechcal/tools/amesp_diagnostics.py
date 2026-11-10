"""Known backend applicability limits and explicit fatal output diagnostics."""

import re

# The installed Amesp backend reports this range explicitly for its STO-3G basis.
# This is not a claim about every implementation or about scientific accuracy.
STO3G_ELEMENTS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe".split()
)
STO3G_BOUNDARY = "Installed Amesp STO-3G supports H--Xe only."


def basis_precondition(keywords: list[str], symbols: list[str]) -> dict | None:
    if "sto-3g" not in {word.lower() for word in keywords}:
        return None
    unsupported = sorted(set(symbols) - STO3G_ELEMENTS)
    if not unsupported:
        return None
    return {
        "code": "capability_unsupported",
        "message": f"{STO3G_BOUNDARY} Unsupported input elements: {', '.join(unsupported)}. "
                   "Repeating the same method/basis cannot repair this input.",
        "details": {
            "reason": "basis_element_unsupported", "basis": "sto-3g",
            "supported_element_range": "H--Xe", "unsupported_elements": unsupported,
            "configured_keywords": list(keywords), "retry_same_configuration": False,
        },
    }


def backend_stop(texts: dict[str, str]) -> dict | None:
    for source, text in texts.items():
        match = re.search(r"^\s*Stop\s*:\s*([^\r\n]+)", text, flags=re.MULTILINE | re.IGNORECASE)
        if match is None:
            continue
        reason = match.group(1).strip()
        basis_limit = re.fullmatch(r"Only support H--Xe for sto-3g\.?", reason, flags=re.IGNORECASE)
        return {
            "code": "capability_unsupported" if basis_limit else "runtime_failed",
            "message": f"Amesp backend stopped: {reason[:1000]}",
            "details": {
                "reason": "basis_element_unsupported" if basis_limit else "backend_stop",
                "backend_stop_reason": reason[:1000], "diagnostic_source": source,
                **({"basis": "sto-3g", "supported_element_range": "H--Xe",
                    "retry_same_configuration": False} if basis_limit else {}),
            },
        }
    return None
