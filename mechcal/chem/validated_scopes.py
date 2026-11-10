"""Numerically checked interface scopes, not guarantees of mechanism validity."""

SPATIAL_ELEMENTS = frozenset({"C", "H", "N", "O", "F", "Au", "S", "B", "Cl", "Br", "Si"})
SPATIAL_MAX_ATOMS = 150
SPATIAL_SCOPE = (
    f"Closed-shell {','.join(sorted(SPATIAL_ELEMENTS))}, <={SPATIAL_MAX_ATOMS} atoms, "
    "gas-phase TDA-aTB1 only."
)

SCAN_ATOMIC_NUMBERS = {"H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
                       "Si": 14, "S": 16, "Cl": 17, "Br": 35}
SCAN_LEGACY_ELEMENTS = frozenset({"H", "C", "N", "O"})
SCAN_MAX_ATOMS = 80
SCAN_SCOPE = (
    "S0 aTB1 constrained scans, closed-shell, <=80 atoms. "
    "CHNO inputs retain their existing scope; B/F/S/Cl/Br/Si extensions are neutral only. "
    "No excited-state force calculations."
)
