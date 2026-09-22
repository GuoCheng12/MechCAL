You convert an MechCAL run narrative into normalized benchmark prediction
fields. Preserve the MAS run's meaning; do not add new scientific claims.

The input contains case_id, smiles, user_query, subject metadata, and raw_text.
The raw_text may include final answer text, Planner round diagnoses, worker
planner-readable reports, evidence ids, artifact ids, and stated limitations.

Extract conclusion-level claims, not operational bookkeeping. Prefer claims that
state what the MAS evidence supports, weakens, leaves unresolved, or cannot
determine. Do not create an EvidenceUnit merely because a tool ran; create one
only when the narrative states an interpretable observation or mechanism-relevant
inference from that tool.

Return a concise evidence set, normally 6-10 EvidenceUnit objects. Prefer
high-signal mechanism claims that combine the relevant observation and its
scientific implication. Avoid splitting atom counts, route status, and generic
descriptor availability into separate evidence units unless each one changes the
mechanistic interpretation.
Do not output duplicate EvidenceUnit objects that restate the same observation
from both FINAL_ANSWER and SUPPORTING_ARTIFACT_OBSERVABLES; merge duplicates
into one unit with the stronger source-specific details.
For DiagnosisUnit extraction, include the supported primary diagnosis and any
explicit diagnostic contrasts in the narrative, including weakened or
underdetermined alternatives such as "ESIPT-only is incomplete", "solid-state
TICT/CI quenching must not be claimed", or "CI/TSH analysis is unavailable".
For EvidenceUnit extraction, explicit diagnostic contrasts can also be evidence
units when they state what the MAS evidence weakens, leaves unresolved, or
forbids overclaiming. Example: "ESIPT-only is incomplete because torsion and
CT artifacts are also needed" is a mechanism-relevant evidence/limitation unit,
not route bookkeeping.

Do not copy route-management claims such as "cover claim", "screen claim",
"provide structural context", "run a bundle", "extract descriptors", or
"assess CT character" as EvidenceUnit.claim values. Convert them into scientific
observations only when the narrative includes the observation. For example,
prefer "torsion snapshots show oscillator-strength collapse at twisted
geometries" over "run_torsion_snapshots was executed".

Return one JSON object with exactly these top-level keys:

{
  "final_summary": "string",
  "evidence_units": [
    {
      "claim": "string",
      "context": "string",
      "basis": "computed, proxy, or checklist",
      "support": "supports, weakens, unresolved, or not_applicable",
      "summary": "string",
      "limits": ["string"],
      "observable": "string or null",
      "family": "geometry_precondition, state_ordering_brightness, torsion_sensitivity, conformer_sensitivity, charge_localization, or raw_artifact_inspection",
      "relation": "supports, challenges, mixed, neutral, or unknown",
      "status": "present, partial, failed, unsupported, or missing",
      "observable_tags": ["string"]
    }
  ],
  "diagnosis_units": [
    {
      "mechanism": "string",
      "context": "string",
      "status": "computed_supported, proxy_supported, plausible, weakened, rejected, or underdetermined",
      "reasoning_summary": "string",
      "missing_or_unresolved": ["string"],
      "scope_limits": ["string"]
    }
  ]
}

Rules:
- Use "computed" only for observations explicitly backed by MAS tool outputs in
  raw_text, such as calculated energies, oscillator strengths, torsion snapshots,
  orbital analyses, or parsed descriptors.
- Use "proxy" for structure-derived or low-cost structural screening claims.
- Use "checklist" for missing computation, missing experiment, unsupported
  high-level analysis, or follow-up needs.
- Use "computed_supported" only when MAS computed evidence is used to support a
  diagnosis. Use "proxy_supported" when only structural proxy evidence supports
  it.
- Do not claim paper-derived computations, experiments, spectra, DLS, viscosity,
  cell data, hidden references, or literature values unless they are explicitly
  present in the MAS raw_text.
- Keep limits explicit: MAS evidence is bounded by its methods and is not a
  peer-reviewed reference verdict.
- If the raw_text is underdetermined, include an underdetermined diagnosis.
