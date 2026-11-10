You are a researcher assessing molecular photophysics.

Infer and rank three mechanisms from the supplied fixed mechanism pool. The
verified SMILES is the only case-specific input. Use the public mechanism
definitions and your existing scientific knowledge. Mechanism families may
overlap. Consider competing explanations before selecting the three most
plausible candidates; do not rank merely by how often a label occurs.

This is a single-response, closed-book task. Do not call tools, browse, identify
the source paper, or retrieve molecular names, database records or reference
answers. No computed results or experimental observations are provided. Do not
invent numerical results, measurements, citations or completed calculations.

For each prediction, identify the specific features visible in the SMILES and
explain why those features support the proposed mechanism in this molecule.
Distinguish structural observations from mechanistic inference. State the main
uncertainty or competing explanation. A proposed validation is not evidence
already obtained. Uncertainty does not prevent you from ranking a candidate.

Return only one JSON object following output_schema, with a single
mechanism_predictions array containing exactly three distinct labels in rank
order 1, 2, 3. Each prediction contains:
- label: an exact label from mechanism_pool.
- rank: the prediction's position in the ordered list.
- confidence: your confidence in the candidate, between 0 and 1. These values
  need not sum to one because mechanisms may coexist.
- claim_status: the scope of the claim actually justified by your evidence.
- support_strength: your assessment of the submitted support.
- evidence: one or more concise strings explaining the molecule-specific
  observations and their connection to the proposed mechanism.
- limitations: one or more concrete limitations of that interpretation.

Do not substitute generic textbook descriptions for molecule-specific evidence.
Do not output scoring criteria, placeholders, or a narrative outside the JSON.
