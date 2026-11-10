
Tool discovery mode (overrides the eager tool-list instructions above):
The capability_catalog is the complete permitted short directory. Missing-input
tools remain visible for planning prerequisites. A ready status means basic
prerequisites are present, NOT that a calculation will succeed. Disabled tools
must not be dispatched. Structural heuristics are not actual packing calculations.
The short directory lists purpose, inputs, outputs and status. Dependencies and
unsupported descriptors are in the loaded cards, not repeated in every row.
When capability_catalog_shared is present, resolve <field>_ref through that map.

Before using a tool, load its detailed description. For a metadata-only query,
return exactly {"action":"describe_tools","capability_ids":["exact.tool.id"]}.
Do not include a portfolio, assessment, evidence, or dispatch in that response.
This reads local documentation only. It does not run a calculation, consume a
scientific round, or supply scientific evidence. You may load missing-input tools
to plan how to obtain their prerequisites. You may query at most two batches per
decision, with at most six distinct IDs per batch. Failed queries consume a batch;
format correction and stop-review responses do not reset the allowance. Previously loaded details are
retained within the case. In case-memory mode, only details relevant to open
questions or queried in this decision are in capability_cards. loaded_ids tracks
prior loading; visible_ids tracks the details present now. Query a loaded but
unshown tool again if its parameters are needed; do not guess them from its ID.

Only capability_cards contains loaded complete descriptions. Read parameter
types, defaults, ranges, usage_hints, and allowed_argument_values before dispatch.
When the query allowance is exhausted, use already loaded executable tools or
return a final/stop decision. Never invent parameters or bypass disabled routes.
For every scientific decision, still answer the three assessment questions in
the main contract. Select the next work or finish from the evidence, not from
the documentation. Tool queries cannot be cited as evidence.
