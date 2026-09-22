# Model and Runtime Configuration

All credentials are supplied by the caller. The source release contains no
populated environment files or private API endpoints. Model aliases describe
the requested endpoint model; they do not independently verify a provider's
underlying model revision.

| Variable | Meaning | Default |
| --- | --- | --- |
| `MECHCAL_OPENAI_BASE_URL` | Subject API base URL | Required |
| `MECHCAL_OPENAI_API_KEY` | Subject API credential | Required |
| `MECHCAL_OPENAI_MODEL` | Exact API-facing subject identifier | `deepseek-v4-flash` |
| `MECHCAL_OPENAI_TEMPERATURE` | Requested temperature | `0` |
| `MECHCAL_OPENAI_TOP_P` | Requested top-p | `1` |
| `MECHCAL_OPENAI_REASONING_EFFORT` | Provider-supported reasoning effort | Omitted |
| `MECHCAL_OPENAI_MAX_TOKENS` | Per-request output limit | Omitted at base client |
| `MECHCAL_OPENAI_TIMEOUT` | Request timeout, seconds | `60` |
| `MECHCAL_OPENAI_MAX_RETRIES` | Additional client attempts | `2` |
| `MECHCAL_OPENAI_SEED` | Optional provider-supported seed | Omitted |
| `MECHCAL_LLM_TELEMETRY_PATH` | Optional local usage log | Disabled |

Set an unsupported sampling field to `omit`. An omitted field uses the
provider's behavior; it does not mean temperature or top-p equals one.
Planner and reviewer clients reserve an output limit of at least 2,400 tokens.
Other specialized calls can apply their own limits. The example environment
uses a 4,800-token base limit; this is a runnable configuration example, not
a claim that every historical experiment used that limit.

Configure the judge separately with `MECHCAL_SUPPORT_JUDGE_BASE_URL`,
`MECHCAL_SUPPORT_JUDGE_API_KEY`, `MECHCAL_SUPPORT_JUDGE_MODEL`, and
`MECHCAL_SUPPORT_JUDGE_TIMEOUT`. Unset endpoint and credential fields fall
back to subject settings. Judge usage is evaluation usage.

## Workflow Controls

The public single-case and benchmark commands default to a cap of 30 rounds.
They disable incremental portfolio input: each Planner update reconstructs
the ranking from available evidence and feedback. The benchmark runner
enables the Planner and internal reviewers, uses deterministic worker tools,
and enables Amesp unless `--no-amesp` is supplied. LLM worker planning and
reporting are optional in the single-case command.

The exported runtime retains the source stopping implementation. Its
configuration defaults are `convergence_min_rounds=6` and
`convergence_window=3`; additional evidence and failure conditions affect
when a run stops. A 30-round cap is not a 30-tool-call cap. Do not infer
measured call counts from these limits.

Use distinct output directories when changing models, prompts, rubric, or
workflow configuration. Evaluation runners can reuse cached outputs, so
reusing an old directory is not a fresh comparison.
