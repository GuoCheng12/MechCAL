# Case memory

Return a decision OR one read-only memory query via the supplied schema.
search_case_records finds IDs/previews by query/kind, with offset pagination.
read_case_records reads up to six IDs. Prefer paths for exact needed fields,
including relevant conditions/limits. An empty paths list reads the full record.
next_start marks a fragment: read that single ID at start=next_start to continue.
find_compatible_artifacts checks capability_id/tool_args against existing files;
it is preflight, not computation or a guarantee of success.

Queries share the displayed allowance across retries/resume and do not advance
scientific rounds. Exhaustion requires a decision using available facts, never
invented ones. Only this case is accessible. Memory/query logs are not evidence.
Observations are facts within their conditions. Judgments, Workers and reviews
are interpretations. Failures are not scientific counterevidence. Search also
for conflicting/inconclusive evidence, not only support.

All candidates, accepted bases and evidence IDs remain visible. Read brief questions
by question_id, hypotheses and historical tests by memory_record_id. New
EvidenceUnits retain conditions/limits. raw_result_record_id opens exact execution
data. Retrieve omitted details before reinterpreting them or transferring a
finding across conditions. Never copy a brief as a complete replacement record.
Use canonical IDs in citations, not memory handles as new evidence.
