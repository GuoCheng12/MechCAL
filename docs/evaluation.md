# Evaluation

`mechcal-evaluate` runs MechCAL and scores mechanism predictions.
`mechcal-direct`, `mechcal-tool-llm`, and `mechcal-common-prior` use the
same mechanism-ranking evaluator. `mechcal-rescore` evaluates saved runs
without repeating the inference workflow. Run each command with `--help`
for its options.

## Case Interface

Each case JSON contains `case_id`, `smiles`, and
`hidden_reference.reference_mechanisms`. Each reference mechanism has a
canonical `label`, a `role`, and a relevance `gain`. The mechanism pool is
defined in `mechcal/mechanism_pool.py`.

The inference payload uses the molecule's SMILES, the fixed pool, and the
task/output instructions. Hidden references stay in the evaluator. The
prediction contract contains ranked mechanism labels, evidence, limitations,
and claim-related fields. Pydantic contracts live in `mechcal/schemas/`;
the direct baseline JSON contract is in `mechcal/eval/mechanism_baselines.py`.

The release contains no benchmark examples with real hidden references.
`examples/offline_score.py` uses explicitly synthetic labels and scores to
demonstrate the metric interface.

## Metrics and Rubric

The main metric implementation is `mechcal/eval/mechanism_ranking.py`.
It reports Top-1 Primary, Recall@3, ES-nDCG@3, and Support@Top1, alongside
diagnostic metrics retained by the research implementation.

The frozen rubric is `mechcal/eval/rubrics/mechanism_evidence_rubric_v2.json`.
The `v2` suffix is the rubric revision, not the software release name.
The judge prompt is `mechcal/eval/prompts/mechanism_evidence_support_judge.md`.
The judge scores submitted evidence independently of the hidden reference.
Reference relevance and evidence support are combined by the metric code.

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

Supply the same explicit case set to each method. No benchmark selection
policy or private case exclusion is applied by the public commands. Check
case counts and evaluation failure fields before interpreting a summary.
Model and judge outputs are generated locally and excluded from distribution.
