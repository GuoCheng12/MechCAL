# MechCAL

MechCAL investigates molecular photophysics from a verified SMILES string.
A Planner directs scientific workers, revises competing mechanism hypotheses,
and submits a three-mechanism ranking with molecule-specific evidence.

The current workflow includes native structured commands, bounded autonomous
workers, case-local evidence memory, incremental review, and local MCP tools.
Scientific decisions remain with the agents. Execution status, parameters,
artifacts, references, and partial results are recorded by the runtime.

## Install, Configure, Run

Use Linux x86_64 and Python 3.11. No parent research workspace or saved
experiment is required. Install the Python package first:

```bash
git clone https://github.com/GuoCheng12/MechCAL.git
cd MechCAL
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c env/constraints-linux-py311.txt -e ".[science,mcp]"
cp configs/deepseek.example.toml mechcal.local.toml
```

Install your scientific tools following [Tools](docs/tools.md). Edit
`mechcal.local.toml` with their executable paths. Keep keys out of the file:

```bash
read -s -p "DeepSeek API key: " DEEPSEEK_API_KEY
export DEEPSEEK_API_KEY
mechcal --config mechcal.local.toml doctor --mcp
mechcal --config mechcal.local.toml probe
mechcal --config mechcal.local.toml run \
  --smiles 'C(c1ccccc1)(c1ccccc1)=C(c1ccccc1)c1ccccc1' \
  --case-id demo-001 --output outputs/demo-001
```

`doctor` checks packages, configured tool locations, and optionally the local
MCP registry. It makes **no model API calls** and does not establish numerical
validity for every molecule. `probe` makes one short, paid model request.
`run` starts the current command workflow and requires a **new** output directory.
It never automatically calls an external judge or restarts a failed run.

The example uses `deepseek-flash`, high reasoning, and a 30-computation
exploration limit. Worker and delivery-stage safeguards are separate. To remove
the exploration limit, remove `max_exploration_rounds` and set
`unlimited_exploration = true`. This can incur substantial API and compute cost.
Other providers can be selected in `[model]`; provider-specific native command
support must be verified. See [Configuration](docs/configuration.md).

The older `mechcal-run-case` and benchmark entry points remain available for
historical workflows. They do not select all current command-profile settings
by default. Use `mechcal ... run` for the current workflow.

## Worker Tools

| Tool | Main responsibility |
| --- | --- |
| RDKit, ASE | Structure preparation, descriptors, atom maps and geometry handling |
| Amesp | Ground-state and vertical excited-state analyses, selected coordinate comparisons |
| Multiwfn + Amesp m2m | Wavefunction conversion and hole-electron/spatial analysis |
| xTB | Low-cost ground-state, ionization/affinity and modeled intermolecular-contact calculations |
| CREST | Conformer exploration, followed by source-map and geometry checks |
| DFTB+ | Restricted-parameter-domain electronic and spin-state calculations |
| PySCF | Spin-free excited states and selected response calculations |
| PySOC | Optional, pinned-source spin-orbit analysis in its supported domain |

These are complementary capabilities, not interchangeable success guarantees.
Optional backends are enabled explicitly in `[qc]`. Unconfigured tools remain
unavailable. Excited-state scans and Amesp SOC remain disabled in the supplied
configuration; installing an executable does not override those gates.
Third-party binaries, parameter distributions and their licenses are not bundled.

## Results and Evaluation

Each run writes `manifest.json`, `qc_config.json`, `status.json`, API usage,
and a case directory containing checkpoints, actions, evidence and artifacts.
`summary.json` is produced after the orchestrator returns; an interrupted run
keeps its records and is not assigned a zero score. Outputs are ignored by Git.

The public [PhotoMechBench inputs](benchmarks/photomechbench/README.md) contain
119 case IDs and SMILES. Hidden references, source papers, inference histories,
model weights and credentials are not included. Inference from an arbitrary
SMILES needs no benchmark files. Scoring additionally needs authorized reference
data and separately configured judge access. See [Evaluation](docs/evaluation.md).

## Layout

```text
mechcal/agents/          Planner and workers
mechcal/runtime/         Commands, memory, review and persistence
mechcal/capabilities/    Tool registry, discovery and preconditions
mechcal/tools/           Scientific adapters and subprocess boundaries
mechcal/schemas/         Typed contracts
mechcal/prompts/         Agent instructions
mechcal/profiles/        Versioned current-workflow configuration
mechcal/eval/            Metrics, rubrics and judge implementations
mechcal/cli/             Installed portable and historical entry points
configs/                Credential-free deployment examples
tests/                  Portable synthetic regression tests
benchmarks/             Previously released public molecular inputs
```

## Development

```bash
python -m pip install -c env/constraints-linux-py311.txt -e ".[science,mcp,dev]"
pytest -q -m "not integration"
ruff check mechcal tests examples
python -m build
```

Research-only replay tests requiring private experiments and raw calculation
fixtures stay in the research archive, not this code release. Their omission
is not a claim that those scientific checks passed on a new machine.
See [Validation](docs/validation.md) for the deployment acceptance scope.

## License

MechCAL is [MIT licensed](LICENSE). External software and datasets retain
their own licenses. No credentials or proprietary tool distributions are
included in the source or Python package.
