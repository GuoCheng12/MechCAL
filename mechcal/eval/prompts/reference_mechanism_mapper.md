You are helping construct a hidden-reference benchmark from paper-derived case notes.

Task:
Map the case's reference diagnosis units onto the fixed AIE/photophysics mechanism pool.

Use only the supplied diagnosis and evidence summaries. Do not use outside literature.
Do not infer mechanisms from SMILES alone when the reference diagnosis text does not support them.
Do not include rejected or weakened competing mechanisms as reference mechanisms, unless the diagnosis text clearly says the mechanism is a supported secondary contributor.

Mechanism labels:
- RIM_RIR_RIV: restriction of intramolecular motion, rotation, vibration, or related nonradiative molecular motion.
- PACKING_HOST_MATRIX_CONFINEMENT: crystal/aggregate/MOF/polymer/host/matrix/pressure confinement that restricts geometry or decay pathways.
- HOST_GUEST_INTERACTION: guest adsorption, pore entry, host-guest binding, guest-induced electronic/structural change.
- ICT_TICT_CT: intramolecular charge transfer, twisted charge transfer, CT excited state.
- ESIPT_PT: excited-state proton transfer or proton-transfer photophysics.
- PET_ET: photoinduced electron transfer or electron-transfer quenching/turn-on.
- AGGREGATE_EXCITON_EXCIMER: aggregate exciton, excimer, exciplex, H/J aggregate, new aggregate-state electronic species.
- RADIATIVE_RATE_STATE_BALANCE: radiative rate, oscillator strength, bright/dark state balance, Kr/knr or emissive-state population balance.
- TRIPLET_METAL_ENERGY_TRANSFER: triplet/RTP/TADF/heavy-atom/lanthanide/metal energy-transfer photophysics.
- RACI_CI_ACCESS: restricted access to conical intersection or CI-mediated nonradiative decay.
- SOKR_ANTI_KASHA: suppression of Kasha's rule, anti-Kasha or higher-state emission.

Role/gain:
- primary: central supported mechanism for the case, gain 2.
- co_primary: another central supported mechanism, gain 2.
- secondary: supported contributor or important supported mechanism layer, gain 1.

Output JSON only:
{
  "reference_mechanisms": [
    {
      "label": "ONE_LABEL_FROM_POOL",
      "role": "primary|co_primary|secondary",
      "gain": 1_or_2,
      "source_diagnosis_ids": ["D..."],
      "rationale": "Short explanation grounded in the cited diagnosis units."
    }
  ],
  "excluded_mechanisms": [
    {
      "label": "ONE_LABEL_FROM_POOL",
      "source_diagnosis_ids": ["D..."],
      "reason": "Why it was not included, e.g. rejected, weakened, or only validation-needed."
    }
  ],
  "notes": "Any ambiguity or human-review warning."
}

Constraints:
- Return at least one primary or co_primary mechanism.
- Use at most four reference mechanisms.
- Every included mechanism must cite one or more supplied diagnosis IDs.
- Each cited diagnosis ID must exist in the payload.
- Do not cite evidence IDs as source_diagnosis_ids.
- If a mechanism appears only as rejected/weakened/underdetermined, put it in excluded_mechanisms instead of reference_mechanisms.
