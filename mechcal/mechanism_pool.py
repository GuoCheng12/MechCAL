from __future__ import annotations

MECHANISM_POOL = (
    "RIM_RIR_RIV",
    "PACKING_HOST_MATRIX_CONFINEMENT",
    "HOST_GUEST_INTERACTION",
    "ICT_TICT_CT",
    "ESIPT_PT",
    "PET_ET",
    "AGGREGATE_EXCITON_EXCIMER",
    "RADIATIVE_RATE_STATE_BALANCE",
    "TRIPLET_METAL_ENERGY_TRANSFER",
    "RACI_CI_ACCESS",
    "SOKR_ANTI_KASHA",
)

# Public taxonomy meanings, not case labels, scoring criteria or evidence thresholds.
MECHANISM_DEFINITIONS = {
    "RIM_RIR_RIV": "Restriction of intramolecular rotation, vibration or other motion that changes nonradiative decay.",
    "PACKING_HOST_MATRIX_CONFINEMENT": "Crystal, aggregate, host or matrix confinement that changes geometry, motion or decay pathways.",
    "HOST_GUEST_INTERACTION": "Guest binding, adsorption or pore entry that changes the host's structure or electronic properties.",
    "ICT_TICT_CT": "Intramolecular charge transfer, twisted charge transfer or charge-transfer excited states.",
    "ESIPT_PT": "Excited-state proton transfer or related proton-transfer photophysics.",
    "PET_ET": "Photoinduced electron transfer associated with quenching or emission turn-on.",
    "AGGREGATE_EXCITON_EXCIMER": "Aggregate excitons, excimers, exciplexes or H/J aggregation that changes electronic states.",
    "RADIATIVE_RATE_STATE_BALANCE": "Changes in radiative rate, oscillator strength, bright/dark state balance, emissive-state populations or radiative/nonradiative rate balance. This does not mean only a strongly allowed S1 state.",
    "TRIPLET_METAL_ENERGY_TRANSFER": "Triplet emission, room-temperature phosphorescence, delayed fluorescence, heavy-atom effects or metal-related energy transfer. Metal is not required for every member of this family.",
    "RACI_CI_ACCESS": "Restricted access to conical intersections or changes in conical-intersection-mediated nonradiative decay.",
    "SOKR_ANTI_KASHA": "Suppression of Kasha's rule, anti-Kasha emission or emission from a higher excited state.",
}


def mechanism_definitions():
    return {"version": "public-meanings-v1", "definitions": dict(MECHANISM_DEFINITIONS),
            "policy": "Families can overlap. These define the labels, not proof requirements or scoring rules. "
                      "A case-specific explanation is a revisable hypothesis, not the definition of its family."}
