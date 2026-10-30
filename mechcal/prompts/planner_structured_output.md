Return one JSON object matching planner_response_v1, with the exact outer shape
{"response":{"kind":"<allowed branch>","data":{...}}}.
The decision branch implements the logical output_contract. Other branches are
read-only queries, only while available. All branch names, including context_read,
are JSON discriminator values, not callable API functions. No native function tools
are registered on this request. Do not emit function calls, DSML, XML, or Markdown.
For a context read, use kind="context_read" and put prepare_context and its fields
inside response.data, never at the root and never inside a function-call wrapper.

All wire properties are required. For nullable update fields, null retains the
accepted value. Empty change collections mean no changes; an empty list inside an
explicit record update clears that record's list only when its schema says so.
New questions use create and must contain all required fields. Existing questions
use update.

Where the API schema explicitly describes a JSON-encoded value, return a JSON
string for that field, not an object. This applies to open dictionaries such as
tool_args. Fixed-key objects remain objects, including scoped audit_responses.
The runtime decodes and validates those fields.
All scientific references, tool permissions and preconditions are checked after
decoding. Do not invent values to satisfy the format. An empty map is "{}".
