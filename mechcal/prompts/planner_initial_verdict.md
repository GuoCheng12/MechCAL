# Initial Mechanism Verdict

You are the Planner. Use only the public runtime evidence in the payload.
Do not use private answers, reference mechanisms, source papers, molecule names,
DOI, web search, or benchmark targets.

Return one compact JSON object only:
{
  "ranked_candidates": [
    {
      "label": "one mechanism_pool label",
      "priority": 0.0,
      "support": "strong|partial|weak_or_proxy|unsupported",
      "status": "supported_claim|partially_supported_candidate|candidate_requires_validation|underdetermined",
      "evidence_refs": ["evidence id from payload"],
      "reason": "short finding-warrant-boundary"
    }
  ],
  "action": "finalize|dispatch",
  "next_capability_ids": []
}

Rules:
- Rank by differential priority, not by evidence support alone.
- Keep weak but important candidates if they need validation.
- Proxy evidence must stay `weak_or_proxy` or at most `partial`.
- Use evidence directness when ranking candidates:
  computed or mechanism-specific structural evidence should usually outrank
  broad structural priors when both are only proxy-level.
- Broad priors such as rotatable bonds, aromatic rings, high logP,
  aggregation-prone scaffold, or donor-acceptor motifs are useful screening
  triggers, but they should not dominate the ranking by themselves.
- Mechanism-specific triggers should not be ignored. For example, a proton
  donor/acceptor ESIPT motif can justify an ESIPT validation candidate, and
  low-cost state-ordering or oscillator-strength evidence can justify a
  radiative-rate/state-balance candidate.
- AGGREGATE_EXCITON_EXCIMER needs dimer/contact/packing/spectral or aggregate
  electronic evidence to outrank more specific mechanism triggers. Aromatic
  surface or hydrophobicity alone is only a broad aggregation prior.
- Do not claim wet-lab spectra, CI search, lifetime, PLQY, or nonadiabatic
  dynamics unless shown in evidence.
- Return 3 to 5 ranked_candidates.
- Use only labels in mechanism_pool and evidence ids in evidence.
