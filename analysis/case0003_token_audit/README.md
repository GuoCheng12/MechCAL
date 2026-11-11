# Case 0003: latest three recorded runs, token audit

Published on 2026-11-12 at the project owner's request. Directory timestamps are
the first recorded model request in Asia/Shanghai (UTC+08:00), not file modification times.
These are the three latest Case 0003 runs available at export time.

## Three Runs

| Directory | Model calls | Scientific rounds | Calculations | Input tokens | Output tokens | Total tokens | Cached input | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [2026-11-11_231135_input-handoff](2026-11-11_231135_input-handoff/) | 15 | 1 | 1 | 224,339 | 28,846 | 253,185 | 81,152 | 377.146 s |
| [2026-11-12_003625_background-initial](2026-11-12_003625_background-initial/) | 13 | 1 | 1 | 190,160 | 21,888 | 212,048 | 71,552 | 319.857 s |
| [2026-11-12_004215_background-verified](2026-11-12_004215_background-verified/) | 14 | 2 | 2 | 188,374 | 21,253 | 209,627 | 78,592 | 383.958 s |

All three used `deepseek-flash` with `reasoning_effort=high`. These are bounded
interface/background-context canaries paused at their acceptance token thresholds.
None finalized a scientific answer or called the external judge. A round can
contain several model calls. Different stopping points and agent workloads mean
these totals alone are **not a controlled comparison of end-to-end savings**.

Original experiment names, in table order:

- `mechcal_input_handoff_0003_20261111_230658`
- `mechcal_background_0003_20261112`
- `mechcal_background_verified_0003_20261112`

## Where To Look

Each run directory contains:

| File | Purpose |
| --- | --- |
| `calls.csv` | One row per recorded API response, with agent, input/output/cache/reasoning tokens and latency. |
| `by_agent.csv` | Totals by Planner, Worker and reviewer role. |
| `token_summary.json` | Run status, totals, stage, model-call and scientific-action counts. |
| `api_telemetry.jsonl` | **Derived** per-call telemetry, with source line numbers and original usage counters. |
| `case/history/command_requests.jsonl` | Complete recorded model inputs, tool definitions, responses and usage, with local paths anonymized. |
| `case/command_checkpoint.json` | Latest recorded checkpoint. |
| `case/case_run.json` | Recorded run state, not a fabricated final answer. |
| `case/_payloads/` | Exact content-addressed arrays referenced by compact snapshots, where present. |
| `recorded/` | Original manifest/configuration, summary and budget-stop record, with paths anonymized. |
| `records.tar.gz` | All text records from the original experiment, including history, memory, operations, tool inputs/outputs and scientific result files. |
| `source_inventory.json` | Every source file's hash/size, inclusion decision and published hash/size. |

There was **no original standalone `api_telemetry.jsonl`** in these three runs.
The export derives that file from real `request`/`response` pairs in
`command_requests.jsonl`, matching both request ID and attempt. All 42 responses
have recorded usage. No token counts are inferred from character counts; missing
usage would remain null rather than becoming zero. The exporter rejects unmatched
calls and inconsistent counters instead of silently omitting them.

`cached_input_tokens` is a subset of input, and `reasoning_output_tokens` is a
subset of output. **Do not add either to total tokens again.** Request component
byte counts and fingerprints describe the original, pre-redaction messages.
They identify repeated content but are not a per-component token measurement.
`previous_identical_occurrences` counts exact serialized repeats in earlier input
items, not semantic duplicates or additional billed tokens.

To inspect one call, find its row in `calls.csv`, then use the corresponding
`provenance.request_line` and `provenance.response_line` from `api_telemetry.jsonl`
to locate its complete conversation in `case/history/command_requests.jsonl`.
Reviewer and Worker histories are retained, not only Planner messages.

## Archive Scope

The archives retain 818, 819 and 823 text files, respectively. Binary `.mo`
wavefunctions and process lock files are excluded; the inventory lists them
explicitly. No model weights, executable packages, credentials, private relay
addresses, original paper PDFs or SI PDFs are published here.

Workspace/home paths are replaced with `/SOURCE_WORKSPACE` and `/SOURCE_HOME`.
Original model text, scientific numbers, IDs and usage counters otherwise remain
unchanged. Historical provenance hashes are retained, not rewritten to suggest
the anonymized export is the original run. This is an **analysis archive**, not
a plug-and-play resume package: source paths no longer resolve, and wavefunctions
are omitted. The original local experiments have not been modified or deleted.

Keep `_payloads` beside each compact checkpoint. The normal storage reader can
restore all arrays, for example from this directory after installing MechCAL:

```python
from mechcal.runtime.snapshot_storage import read_case_run

case = read_case_run(
    "2026-11-12_004215_background-verified/case/command_checkpoint.json"
)
```

Archive paths preserve the original experiment layout. The earliest experiment
has an extra `subjects/0003/` level; the convenience copies under `case/` above
provide a consistent layout for all three runs.

## Export And Verification

`export_audit.py` performs only local reads/exports and never calls a model.
It requires a new output directory and leaves sources untouched:

```bash
python export_audit.py --source /path/to/original/experiment \
  --output /path/to/new-audit-directory --workspace /path/to/workspace \
  --home /path/to/user-home
pytest -q test_export_audit.py
```

Export checks cover exact attempt pairing, incomplete histories, missing usage,
counter consistency, cache/reasoning accounting and credential/path redaction.
Before publication, all archive members were checked against their inventories,
and all six state/checkpoint copies were loaded through MechCAL's storage reader.
The audit data are deliberately excluded from Python source distributions.
