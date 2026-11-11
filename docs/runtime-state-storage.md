# Runtime state and context efficiency

This change removes repeated local serialization and repeated review background.
It does not change the mechanism pool, calculation methods, ranking rules,
review validity checks, rubric, or judge. It makes no claim about end-to-end
scientific performance or online token savings without a new measured run.

## Snapshot-scoped source fingerprints

`RecordIndex` owns one immutable case snapshot. It computes each referenced
record's exact content hash once and shares it between query, review, editing,
and delivery views. Composite review and investigation hashes are also reused
within that index. The digest algorithm and values are unchanged.

An accepted mutation needs a new index. Passing an index for another case or
another snapshot is rejected. There is no process-global source cache and no
cross-case reuse. Callers must not mutate records while using their index.

## Persisted large arrays

Frequently updated files (`command_checkpoint.json`, `case_run.json`,
`artifacts/manifest.json`, and `evidence_units.json`) can contain an
`_mechcal_storage` manifest. Arrays of at least 128 items and 64 KiB move to
content-addressed `_payloads/<sha256>.json` files inside the case run directory.
Their original positions contain null placeholders identified by JSON pointers
in the manifest. Other values, including status and progress, remain inline.

This is lossless storage, not evidence summarization. Runtime records and read
paths contain the original arrays after loading. Identical arrays share a file.
Complete array files are published before their referencing snapshot. Repeated
saves verify existing payloads and do not rewrite unchanged array files. Packing
is shared only within one immutable save operation, not across later edits.

Use the reader for both old and new runs:

```python
from pathlib import Path
from mechcal.runtime.snapshot_storage import read_case_run, read_json

run_dir = Path("runs/CASE_ID")
case = read_case_run(run_dir / "command_checkpoint.json")
manifest = read_json(run_dir / "artifacts/manifest.json", blob_root=run_dir)
```

`RunStore.read_case_run()` defaults to `case_run.json`. Evaluation, rescoring,
legacy reporting, and command-loop resume use the same reader. Older
self-contained JSON remains readable without conversion. Scripts that directly
call `CaseRun.model_validate_json` must migrate to this reader for new compact
snapshots; they cannot safely consume the placeholders directly.

Copy the **whole case run directory, including `_payloads`**, when migrating or
archiving. A standalone compact checkpoint is not a complete archive. Missing,
corrupt, conflicting, or foreign-case payload references raise explicit errors;
they never become empty evidence or zero scores. Do not prune payload files
without checking references from every retained snapshot. Existing historical
experiments are not rewritten or deleted automatically.

## Review background reuse

Review receipts retain literal complete sections. The wire view can reference
unchanged question evidence, candidate comparisons, investigation state, and
execution summaries, or send exact deltas against an available receipt.
Changed claims, source bindings, and current stopping information remain subject
to the same review. Changing the review contract invalidates reuse. Reuse is
not scientific approval and does not close unresolved findings.

For factual issues targeting an argument, correction context also lists its
linked prediction labels and the current final-prose edit location. These are
existing dependency links, not guessed semantic equivalence. Generated
prediction evidence follows argument edits; the Planner still decides whether
separately written final prose needs correction. The runtime neither replaces
matching numbers automatically nor marks related passages wrong by association.

## Offline acceptance

Tests cover old/new reads, exact reconstruction, source hashes, missing/corrupt
arrays, case isolation, interrupted publication, unchanged blob writes, source
version invalidation, review-delta reconstruction, and judge source retention.

`examples/audit_snapshot_efficiency.py` replays the same completed checkpoint
against two code checkouts without making model requests or running scientific
jobs. Each invocation requires a new output directory and leaves the original
experiment unchanged:

```bash
python examples/audit_snapshot_efficiency.py \
  --source /path/to/checkout \
  --checkpoint /path/to/run/command_checkpoint.json \
  --output /path/to/new-audit-directory
```

Compare source and Planner-view hashes, local preparation/save timings,
checkpoint and unique-array bytes, and historical review-packet bytes. Packet
bytes are not billed tokens. Shared-machine timings are indicative rather than
a controlled hardware benchmark. This replay does not establish that a model
will stop repeating factual mistakes across different prose fields; that needs
a bounded online acceptance run, not suppression of real review findings.
