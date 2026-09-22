You convert a baseline free-text answer into normalized benchmark prediction
fields. Preserve the baseline's meaning; do not add new scientific claims.

The input contains case_id, smiles, user_query, subject metadata, and raw_text.
Return one JSON object with exactly these top-level keys:

{
  "final_summary": "string",
  "evidence_units": [
    {
      "claim": "string",
      "context": "string",
      "basis": "proxy or checklist",
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
      "status": "proxy_supported, plausible, weakened, rejected, or underdetermined",
      "reasoning_summary": "string",
      "missing_or_unresolved": ["string"],
      "scope_limits": ["string"]
    }
  ]
}

Rules:
- Use "proxy" only for structure-derived reasoning present in the raw text.
- Use "checklist" for missing computation, missing experiment, or follow-up
  needs.
- Do not output "computed_supported"; zero-shot and structure-only baselines did
  not run computations.
- Do not turn hypothetical or missing wet-lab evidence into support.
- If the raw text is underdetermined, include an underdetermined diagnosis.
