from __future__ import annotations

# Seven families (author decision 2026-11-13): only families for which computation can give
# discriminating evidence. The earlier 11-family pool is kept below for reading old records;
# mechcal/eval/family_reduction.py maps its labels onto this pool.
MECHANISM_POOL = (
    "RIM_RIR_RIV",
    "ICT_TICT_CT",
    "ESIPT_PT",
    "PET_ET",
    "AGGREGATE_EXCITON_EXCIMER",
    "RADIATIVE_RATE_STATE_BALANCE",
    "TRIPLET_ISC_RTP_TADF",
)

# Public taxonomy meanings, not case labels, scoring criteria or evidence thresholds.
MECHANISM_DEFINITIONS = {
    "RIM_RIR_RIV": "Restriction of intramolecular rotation, vibration or other motion that changes nonradiative decay, "
                   "whether the restriction comes from the molecule itself or from its packing, host or matrix; "
                   "includes restricted access to a conical intersection along that motion.",
    "ICT_TICT_CT": "Intramolecular charge transfer, twisted charge transfer or charge-transfer excited states.",
    "ESIPT_PT": "Excited-state proton transfer or related proton-transfer photophysics.",
    "PET_ET": "Photoinduced electron transfer associated with quenching or emission turn-on.",
    "AGGREGATE_EXCITON_EXCIMER": "Intermolecular electronic interaction in aggregates, crystals or other condensed phases "
                                 "(aggregate excitons, H/J coupling, excimers, exciplexes), including packing-dependent "
                                 "changes of that interaction.",
    "RADIATIVE_RATE_STATE_BALANCE": "Changes in radiative rate, oscillator strength, bright/dark state balance, emissive-state populations or radiative/nonradiative rate balance. This does not mean only a strongly allowed S1 state.",
    "TRIPLET_ISC_RTP_TADF": "Triplet-state photophysics: intersystem crossing, room-temperature phosphorescence, thermally "
                            "activated delayed fluorescence or heavy-atom effects.",
}

ELEVEN_FAMILY_POOL = (
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


# The 0.2.0 workflow (mechcal/agents, mechcal/runtime, mechcal/schemas and their baselines) stays
# frozen on the 11-family pool.
ELEVEN_FAMILY_DEFINITIONS = {
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


def eleven_family_definitions():
    return {"version": "public-meanings-eleven-families", "definitions": dict(ELEVEN_FAMILY_DEFINITIONS),
            "policy": "Families can overlap. These define the labels, not proof requirements or scoring rules. "
                      "A case-specific explanation is a revisable hypothesis, not the definition of its family."}


def mechanism_definitions():
    return {"version": "seven-family pool (2026-11-13)", "definitions": dict(MECHANISM_DEFINITIONS),
            "policy": "Families can overlap. These define the labels, not proof requirements or scoring rules. "
                      "A case-specific explanation is a revisable hypothesis, not the definition of its family."}
