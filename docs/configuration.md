# Configuration

The current entry point reads a TOML file supplied by `--config`. It loads the
packaged `command_v1` workflow profile and applies explicit `[runtime]` overrides.
It does not import a historical run, audit script, dataset, or credential store.
Unknown options are rejected. The legacy environment-based entry points remain
available but are not equivalent to this profile.

## Model

`[model]` requires `base_url` and the exact API-facing `model` identifier.
`api_key_env` names an environment variable; it does not contain its value.
Inline API keys and URLs containing embedded credentials are rejected.

The supplied example uses `deepseek-flash`, `reasoning_effort = "high"`,
`api_protocol = "responses"`, timeout 180 seconds and two retries.
The existing provider adapter handles DeepSeek's native transport.
Temperature, seed and maximum output length are omitted by default; omission
does not mean temperature zero or unlimited provider output. Supported optional
fields are declared in `ModelConfig` in `mechcal/cli/deployment.py`.

`probe` deliberately uses an output cap of 2,048 tokens and no transport retry
for its single connectivity request. This does not alter normal run settings.
No secret value is written into the run manifest.

## Runtime

The packaged profile enables current native commands, case memory, hypothesis
updates, evidence follow-up, computational review, compact task context and
autonomous worker tasks. It preserves the current scientific workflow.

The deployment example explicitly limits exploration to 30 new scientific
computations. The internal profile itself permits unlimited exploration;
select the desired limit explicitly:

```toml
[runtime]
unlimited_exploration = false
max_exploration_rounds = 30
enable_amesp = true
```

For an unbounded run, remove `max_exploration_rounds` and set
`unlimited_exploration = true`. Do not set both a finite limit and unlimited mode.
`max_rounds` is retained for compatibility and is not the active bound in
unlimited command mode.

Worker task defaults are 12 calls, 4 computations, 900 seconds. Delivery allows
30 model turns. These are distinct from exploration rounds and provider limits.
Changes to these settings must be recorded when comparing experiments.

## Tool Locations

`[tools]` configures Amesp, Multiwfn and m2m. `[qc]` configures xTB, CREST,
DFTB+, PySCF and PySOC. Use executable names on PATH, absolute paths, or paths
relative to the TOML file, not to the shell's working directory. `~` and defined
environment variables are expanded. Unresolved variables are errors.

The launcher writes a resolved QC configuration inside the run directory and
passes its location to the local MCP process. Missing optional backends stay
unavailable; they are not replaced with invented results. Both Multiwfn and m2m
must be configured together for hole-electron analysis.

`allow_excited_scan` and `allow_amesp_soc` default to false. Do not enable an
unvalidated route merely to make a dependency check green.

## Result Isolation

Use a new output directory for every run. Case IDs must be filename-safe, not
paths. Historical results are never overwritten. API usage and raw outputs
remain local. The current single-molecule command does not call the external
judge, and cannot infer benchmark scores without separate reference data.

Judge and baseline settings are documented in [Evaluation](evaluation.md).
For private deployments, keep TOML files named `*.local.toml` and credentials
in the configured environment or a user-owned secret store outside Git.

## DeepSeek key (development machine)

Run once, in your own terminal (the key is read without echo and never enters the repository):

```bash
bash env/set_deepseek_key.sh
```

It stores `DEEPSEEK_API_KEY` in `~/.config/mechcal/secrets.env` (mode 600) and adds a block to
`~/.bashrc` that loads it together with `env/deepseek.env.sh`. That file holds the non-secret
settings: base URL `https://api.deepseek.com`, model `deepseek-flash`, Responses API, backbone
effort `high`, and the evidence judge at effort `max` (override `MECHCAL_JUDGE_*` to use another
judge). `mechcal ... doctor` then reports `credential_present: true`; `mechcal ... probe` makes
one short paid request to confirm access. To rotate the key, run the script again.
