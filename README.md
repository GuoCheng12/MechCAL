# MechCAL

MechCAL investigates molecular photophysical mechanisms from a verified SMILES.
A Planner chooses scientific analyses, compares candidate explanations and updates
a ranking as evidence arrives. The fixed mechanism pool contains **11 families**.

**Development snapshot, 2026-10-31.** This branch contains the current native
command-loop implementation, not a new stable release. It is not merged into
`main`. End-to-end scientific validation remains ongoing; interface tests do not
establish prediction quality or reproduce the paper's results.

## Current Workflow

```text
SMILES -> structure preparation -> coverage review
                                      |
                                      v
                       Planner hypotheses + confidence
                                      |
                   describe tools / read stored evidence
                                      |
                         structured compute request
                                      |
                    Macro or Microscopic tool execution
                                      |
                   durable results + Worker explanation
                                      |
                   updated arguments and mechanism ranking
                                      |
                    Reviewer reads the cited evidence
                                      |
                       feedback -> Planner -> final answer
```

- Planner chooses actions and maintains provisional confidence across candidates.
  Provisional confidence is separate from the reviewed final ranking.
- Worker and Reviewer explanations are natural language. Tool requests, evidence
  references and linked judgment edits use small structured commands.
- The runtime checks permissions, parameters, record ownership and consistency.
  It does not invent evidence or replace Planner's scientific ranking.
- Results stay on disk. Agents can request selected fields or inspect directories
  instead of carrying every raw output in every prompt.
- Calculations and explanations are separate records. A failed explanation does
  not erase a completed calculation. Failed calculations are not positive evidence.
- Discovery and execution use one `CapabilityRegistry`, through local adapters
  or stdio MCP. Missing backends and unsupported computations remain unavailable.

## Scientific Tools

These backends are implemented in this snapshot. **They are not all installed
by pip, enabled by default, applicable to every molecule or invoked in every run.**

| Tool | What MechCAL uses it for | Important boundary |
| --- | --- | --- |
| **RDKit** | SMILES parsing, descriptors, motifs, rotor/topology screens, conformers and force-field preparation | Packing screens are heuristics, not actual packing calculations. Force fields do not cover every metal-containing molecule. |
| **Amesp** | S0 geometry recovery, aTB1/TDA vertical states, oscillator strengths, orbital/transition analysis, selected-geometry comparisons and S0 scans | Low-cost electronic-structure evidence. Excited-state relaxation remains suspended. |
| **Multiwfn + Amesp m2m** | Read matched orbital/transition files to obtain hole/electron centroids, separation, overlap and atomic/fragment contributions | Fixed-geometry analysis, not a new quantum calculation, TICT trajectory or condensed-phase measurement. Both executables are required. |
| **xTB** | Vertical IP/EA estimates, ground-state vibrations, modeled dimer searches and intermolecular-distance scans | Ground-state/contact trends; a modeled dimer is not an observed crystal structure. |
| **DFTB+** | Selected excited-state energy/force probes and fixed-geometry singlet/triplet comparisons | Parameter and composition restrictions apply. Excited-state coordinate scans remain disabled pending derivative validation. |
| **PySCF** | Matched S0/S1 local forces and excitation-induced force differences using RHF/6-31G/CIS | Optional qualitative route, pinned to PySCF 2.14.0; connected neutral closed-shell CHNO, 2-40 atoms. Not relaxation or decay rates. |
| **PySOC** | State-specific spin-orbit couplings from validated DFTB+ outputs using ZEFF | Optional neutral closed-shell CHNO route. Couplings are not ISC rates or triplet yields. This adapter does not cover metals. |

**ASE, NumPy and SciPy** support geometry handling and numerical analysis.
**FastMCP/MCP** provide transport, not scientific calculations. External QC
subprocesses use bounded wall times and CPU threads; the wrapper requests no GPU.
See [Tool Setup](docs/tool-setup.md). Third-party executables, parameter sets and
licenses are not bundled.

## Installation

Linux and Python 3.11 are the development environment.

```bash
git clone --branch codex/command-protocol-refactor https://github.com/GuoCheng12/MechCAL.git
cd MechCAL
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[science,mcp,dev]"
python -m pip check
```

The science extra supplies RDKit/ASE and numerical libraries. Amesp, Multiwfn,
xTB, DFTB+, the optional PySCF environment and PySOC are installed separately.
Installation does not start calculations or call a model API.

## Model Configuration

Keep credentials in the environment, never in committed files. The model must
support native tool calls through the selected protocol.

```bash
export MECHCAL_OPENAI_BASE_URL="https://your-provider.example/v1"
export MECHCAL_OPENAI_MODEL="your-model-identifier"
export MECHCAL_OPENAI_API_PROTOCOL="responses"
export MECHCAL_OPENAI_TEMPERATURE="omit"
export MECHCAL_OPENAI_TOP_P="1.0"
export MECHCAL_OPENAI_REASONING_EFFORT="high"
export MECHCAL_OPENAI_TIMEOUT="180"
export MECHCAL_OPENAI_MAX_RETRIES="0"
export MECHCAL_OPENAI_MAX_TOKENS="16384"
read -r -s -p "Model API key: " MECHCAL_OPENAI_API_KEY
export MECHCAL_OPENAI_API_KEY
```

Use an exact identifier and thinking settings supported by your provider. The
client does not append `/v1` or automatically change protocols on failure.
Development used the official `https://api.deepseek.com` endpoint with
`deepseek-flash` and high reasoning. This is a configuration record, not an
availability or pricing guarantee for other accounts.

`MECHCAL_OPENAI_MAX_TOKENS` limits one response, not total cost. Provider context
and output limits still apply if that client-side setting is omitted.

## Run the Current Command Loop

The portable example explicitly enables native commands, on-demand discovery and
provisional confidence tracking. The older CLI remains for compatibility; its
defaults do not enable all these development features.

```bash
# Configuration check only; no API call or computation.
python examples/run_command_case.py --smiles 'CCO' --dry-run

# Reduced-tool demonstration without Amesp.
python examples/run_command_case.py --smiles 'CCO' --case-id example-001 \
  --max-computations 5 --output-dir outputs/example

# Configured scientific backends via local MCP; separate installs are required.
python examples/run_command_case.py \
  --smiles 'C(c1ccccc1)(c1ccccc1)=C(c1ccccc1)c1ccccc1' \
  --case-id example-002 --amesp --tool-backend mcp-stdio \
  --max-computations 20 --output-dir outputs/full
```

The computation limit counts scientific executions, not queries, reviews or LLM
calls. Planner can finish early. This example has **no cumulative token/call cap**;
start small and monitor provider usage. No external judge is called automatically.
A paused or failed run is not a zero-scored result.

Use a fresh directory for each new run. To continue a paused command checkpoint,
repeat the same configuration with `--resume`. Do not edit code or settings while
running. Monitor saved state without model calls:

```bash
watch -n 5 python scripts/monitor_command_run.py outputs/full/example-002
```

Records retain results, source references, parameters, execution status and reported
model usage. Provisional confidence is not a calibrated probability or benchmark score.

## Data and Evaluation

The existing [PhotoMechBench inputs](benchmarks/photomechbench/README.md) contain
**119 case IDs and SMILES**, unchanged by this snapshot. Hidden references,
papers, SI, generated answers and run logs are not included. Evaluation code and
MechRubric are under `mechcal/eval/`. Scoring requires separately authorized
reference data and judge credentials.

The external judge is separate from the internal Reviewer. The example does not
supply inference agents with the rubric or hidden references.

## Layout and Checks

```text
mechcal/agents/          Agent clients and native contracts
mechcal/runtime/         Command loop, memory, review and durable state
mechcal/capabilities/    Tool registry, schemas and readiness
mechcal/tools/           Scientific adapters and output parsers
mechcal/mcp/             Local MCP server
mechcal/prompts/         Agent prompts
mechcal/eval/            Rubrics, judge, metrics and baseline evaluators
examples/               Portable examples
scripts/                Read-only command-loop monitor
tests/                  Portable regression subset
benchmarks/             Previously published molecular inputs
docs/                   Setup and snapshot scope
```

Production modules and prompts are copied unchanged from the current development
checkout. Private experiment launchers and archive-dependent tests are excluded.
See [Snapshot Scope](docs/development-snapshot.md).

```bash
PYTHONPATH=.:tests pytest -q -m "not integration"
ruff check mechcal tests examples
python examples/offline_score.py
python -m build
```

Offline checks need no paid API calls. They do not certify every optional backend's
scientific accuracy or availability.

## License

MechCAL code is [MIT-licensed](LICENSE). External programs, model services,
parameter sets and datasets retain their own terms and required citations.
