# ChemGraph-MAS (adapted)

This baseline fixes the upstream implementation to ChemGraph `v0.6.0`, commit
`203e18c8529869531fa4addbcdb3f1395a831eaa`.

The upstream Planner-Executor graph is retained. Executors receive only two
LangChain tool entry points, which route to the same local Macro and
Microscopic capability implementations used by MechCAL. The final Aggregator
uses the same backbone and emits the common mechanism-ranking JSON schema.

The adapter does not expose PubChem, molecule-name resolution, literature,
web, database, or upstream default chemistry tools. It does not add MechCAL's
evidence ledger, Coverage Reviewer, Support Auditor, or claim gate. A required
empty `EvidenceLedger` object exists only inside the local tool compatibility
context and is never populated or shared with any agent.

## Local environment

Install the pinned upstream ChemGraph revision in an isolated local
environment. The upstream clone and its virtual environment are deployment
assets and are excluded from this repository.

## Evaluation

The runner requires explicit per-case limits for model calls, tool calls, and
provider-reported total tokens:

```bash
mechcal-chemgraph --case-dir /path/to/cases \
  --case-id CASE_ID \
  --max-model-calls MODEL_CALL_LIMIT \
  --max-tool-calls TOOL_CALL_LIMIT \
  --max-total-tokens TOKEN_LIMIT
```

HTTP retries are disabled. Planner and schema-repair calls are still counted
by the shared model-call and token limits. Support-judge calls are evaluation
cost and are not included in the subject-system budget.

## Responses API

Use `--responses-api` for compatible models that are exposed only through the
OpenAI Responses API. This mode preserves the same ChemGraph graph, local
tools, and output schema. Set unsupported sampling controls to `omit` through
the environment configuration described in the root README.

Subject and judge credentials must be provided through local environment
variables. Neither credential is written to the output directory.
