# MechCAL

MechCAL is a multi-agent system for evidence-calibrated mechanism ranking
in molecular photophysics. Given a verified SMILES string, a Planner selects
scientific analyses and ranks mechanisms from a fixed pool of 11 families.
The ranking is rebuilt as evidence and review feedback become available.

The repository contains the method, scientific tool adapters, prompts,
MechRubric, evaluation code, and baseline adapters. Benchmark cases, hidden
references, papers, model weights, inference records, and credentials are
not included.

## Installation

Python 3.11 is the tested environment. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[science,mcp,dev]"
```

For conda, run `conda env create -f environment.yml` and then
`conda activate mechcal`. A core-only installation (`pip install -e .`)
supports schemas, metrics, and model clients. Molecular calculations require
the science dependencies. Amesp is installed separately.

## Run a Molecule

Set the subject model credentials in your environment. See
[`env/llm.example.sh`](env/llm.example.sh) for all configuration fields.

```bash
export MECHCAL_OPENAI_BASE_URL="https://your-provider.example/v1"
export MECHCAL_OPENAI_MODEL="your-model"
export MECHCAL_OPENAI_API_KEY="your-api-key"
export MECHCAL_OPENAI_MAX_TOKENS="4800"

mechcal-llm-smoke
mechcal-run-case \
  --smiles 'C(c1ccccc1)(c1ccccc1)=C(c1ccccc1)c1ccccc1' \
  --case-id example-001 \
  --llm-planner \
  --max-rounds 30 \
  --output-dir outputs/example
```

This uses local RDKit tools. Add `--amesp` after setting
`MECHCAL_AMESP_BIN` to an authorized Amesp executable. For MCP transport, add
`--tool-backend mcp-stdio`. Missing microscopic capabilities are reported in
the run; a run without Amesp does not reproduce the full evidence routes.

The command calls your configured model API. All generated records stay in
the ignored `outputs/` directory. No API call is made during installation or
the default offline test suite.

## Evaluate

Provide a directory of case JSON files and configure the evidence judge
with `MECHCAL_SUPPORT_JUDGE_BASE_URL`, `MECHCAL_SUPPORT_JUDGE_MODEL`, and
`MECHCAL_SUPPORT_JUDGE_API_KEY`.

```bash
mechcal-evaluate --case-dir /path/to/cases --output-dir outputs/mechcal --limit 3
mechcal-direct --case-dir /path/to/cases --output-dir outputs/direct --evidence-format --limit 3
mechcal-tool-llm --case-dir /path/to/cases --output-dir outputs/tool-llm --limit 3
mechcal-common-prior --case-dir /path/to/cases --output-dir outputs/prior
```

The first command enables Amesp by default; use `--no-amesp` only for a
deliberately reduced-tool run. Remove `--limit 3` for the supplied full case
set. Evaluation requires authorized data that is distributed separately.
Details are in [Evaluation](docs/evaluation.md) and
[Baselines](docs/baselines.md).

## Code Layout

```text
mechcal/agents/          Planner, workers, coverage review, support audit
mechcal/runtime/         Agent loop, gates, persistence, model client
mechcal/capabilities/    Capability catalog and evidence guide
mechcal/tools/           RDKit, Amesp, local and MCP adapters
mechcal/schemas/         Typed input, evidence, prediction, and run contracts
mechcal/prompts/         Agent prompts
mechcal/eval/            Metrics, MechRubric, judge, baseline implementations
mechcal/baselines/       Optional ChemGraph adapter
mechcal/cli/             Installed inference and evaluation commands
env/                    Credential-free configuration template
examples/               Offline usage examples
tests/                  Regression and packaging tests
docs/                   Configuration, tools, evaluation, and release notes
```

The Python import name is `mechcal`. This repository is self-contained and
does not require a parent research workspace.

## Validation

```bash
pytest -q -m "not integration"
ruff check mechcal tests examples
python examples/offline_score.py
python -m build
```

Subprocess MCP checks are marked `integration`. ChemGraph tests require its
optional environment and otherwise skip. See [Validation](docs/validation.md)
for results and limitations, [Tools](docs/tools.md) for tool setup, and
[Release](docs/release.md) for distribution scope.

## License

MechCAL is distributed under the [MIT License](LICENSE). External software,
models, and datasets retain their own licenses.
