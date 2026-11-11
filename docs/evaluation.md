# Evaluation

The portable `mechcal ... run` command performs inference only. It does not
automatically launch a judge or read hidden references. The entry points below
are retained historical evaluation workflows, not aliases for the current
`command_v1` inference profile. Record the exact workflow and rubric when
comparing results.

`mechcal-evaluate` runs MechCAL and scores mechanism predictions.
`mechcal-direct`, `mechcal-tool-llm`, and `mechcal-common-prior` use the
same mechanism-ranking evaluator. `mechcal-rescore` evaluates saved runs
without repeating the inference workflow. Run each command with `--help`
for its options.

## Case Interface

Each released case JSON contains `case_id`,
`public_input.molecule.structure.value` (SMILES), and
`hidden_reference.reference_mechanisms`. Each reference mechanism has a
canonical `label`, a `role`, and a relevance `gain`. The mechanism pool is
defined in `mechcal/mechanism_pool.py`.

The inference payload uses the molecule's SMILES, the fixed pool, and the
task/output instructions. Hidden references stay in the evaluator. The
prediction contract contains ranked mechanism labels, evidence, limitations,
and claim-related fields. Pydantic contracts live in `mechcal/schemas/`;
the direct baseline JSON contract is in `mechcal/eval/mechanism_baselines.py`.

The reconstructed 119-case references are now public at
`benchmarks/photomechbench/raw_v2_119/cases/`, with source-linked evidence and
annotation provenance. Read the [dataset card](../benchmarks/photomechbench/README.md)
for reconstruction status, CC BY 4.0 licensing and version-specific limitations.
"Hidden" describes separation from inference, not a restriction on downloading
the dataset. `examples/offline_score.py` still uses explicitly synthetic labels
and scores to demonstrate the metric interface without an API call.

## Metrics and Rubric

The main metric implementation is `mechcal/eval/mechanism_ranking.py`.
It reports Top-1 Primary, Recall@3, ES-nDCG@3, and Support@Top1, alongside
diagnostic metrics retained by the research implementation.

The historical evaluator uses
`mechcal/eval/rubrics/mechanism_evidence_rubric_v2.json` and
`mechcal/eval/prompts/mechanism_evidence_support_judge.md`. The current source
also includes `mechanism_evidence_rubric_v3.json`, its corresponding prompt and
the submitted-source verification implementation in `mechcal/eval/zero_shot_v3.py`
and `mechcal/eval/tool_sources.py`. These are different evaluation versions;
installing this package does not silently convert an old result to v3.

Rubric suffixes are revision identifiers, not software release names. The
judge scores submitted evidence independently of the hidden reference.
Reference relevance and evidence support are combined by the metric code.
Full v3 benchmark orchestration and reference loading are not part of the
new single-molecule launcher. Keep judge access separate from inference access.

## Runs and Ablations

```bash
mechcal-evaluate --case-dir /path/to/cases --output-dir outputs/full
mechcal-evaluate --case-dir /path/to/cases --output-dir outputs/no-macro --ablation-mode mechcal_wo_macro
mechcal-shards run --case-dir /path/to/cases --output-dir outputs/sharded --shards 2 --start
mechcal-shards merge --case-dir /path/to/cases --output-dir outputs/sharded
```

The registered ablations also include `parallel_worker_summary`,
`one_pass_mechcal`, and `mechcal_wo_microscopic`. These are workflow variants;
their names do not imply matched numbers of model or tool calls.

For the released snapshot, replace `/path/to/cases` with
`benchmarks/photomechbench/raw_v2_119/cases`. Those case files are readable by
the existing evaluators without conversion. They must not be mounted as agent
memory or included in subject prompts. References score agreement with the
source interpretation, while the judge scores the system's own submitted evidence.

Supply the same explicit case set to each method. No benchmark selection
policy or private case exclusion is applied by the public commands. Check
case counts and evaluation failure fields before interpreting a summary.
Model and judge outputs are generated locally and excluded from distribution.

## Evaluation v2 (in progress)

The revision plan for artifact-grounded evidence scoring, metric-validity perturbations,
clustered statistics and dataset analyses is in [evaluation-v2.md](evaluation-v2.md).
`examples/dataset_report.py` produces the offline dataset report without API calls.
