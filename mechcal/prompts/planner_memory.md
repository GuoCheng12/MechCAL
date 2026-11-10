# Read-only case memory

The current context is a stage-specific view, not the full record of the case.
You may either return the formal output requested by output_contract OR return
one standalone query described by memory_access or the memory_query branch of
the supplied response_format schema. Never include a
portfolio, ranking or new evidence in a query.

- search_case_records: locate observations, actions, questions, judgments or
  artifacts by keyword and optional kind. Results are previews, not full evidence.
- read_case_records: read up to six known IDs. Prefer direct reading when you
  already know the ID. If next_start is present, query that one ID with start to
  read the next fragment. Search pagination uses offset and next_offset.
  Optional paths selects exact JSON fields, for example ["/metrics/method"],
  without transferring unrelated arrays. Include relevant conditions and limits.
- find_compatible_artifacts: supply capability_id and intended tool_args. This
  applies registry preflight, not scientific calculation. Missing parameters,
  disabled routes and absent files remain limitations, not automatic recovery.

Queries return inside the same scientific round. A decision shares one finite
query allowance across format retries and resume. When it is exhausted, use
available records to submit the requested decision; do not invent missing facts.
Repeated queries also consume the allowance. Tool descriptions use their existing
separate allowance and are available only during action selection.

Stored observations and their original conditions are authoritative. Worker
explanations and Planner judgments are interpretations, not new observations.
Operational failures are not scientific counterevidence. Search for relevant
conflicting and inconclusive records as well as supportive ones. Do not assume
an omitted historical record does not exist.

Current accepted findings, boundaries, the evidence index and active feedback
are provided automatically. Assimilation includes new EvidenceUnits once in
observations, with their conditions and limits. A pending result's
raw_result_record_id opens its complete execution, including raw fields not in
those EvidenceUnits. Action selection uses the accepted judgment, not all old
raw calculations. Argument briefs and historical test briefs are not full records.
Retrieve older details when needed to change or compare an interpretation.
Read calculation conditions before transferring a conclusion
across geometries, electronic states or environments. Keep the original record
IDs in scientific citations; memory queries and their logs are not evidence.
Completed question rows may be compact index entries. Read the question by ID
before rewriting its full record, or use the existing retained_judgment contract
when available. Do not copy an incomplete index entry into a full output record.
